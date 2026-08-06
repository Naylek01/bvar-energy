"""Build the six model-specific datasets used in the ECB energy STIP suite.

The paper does not estimate one common monthly BVAR. It estimates separate
models for six HICP energy components:

    1. car fuels          weekly, 24 lags (petrol and diesel sub-models)
    2. liquid fuels       weekly, 24 lags
    3. gas                monthly, 12 lags
    4. electricity        monthly, 12 lags + seasonal dummies in the model
    5. heat energy        monthly, 12 lags
    6. solid fuels        monthly, 12 lags

Run the full source pipeline and compile the six datasets:

    python src/data_pipeline/build_dataset_paper_six_models_v5.py

Compile already existing interim outputs only:

    python src/data_pipeline/build_dataset_paper_six_models_v5.py --compile-only

Outputs are written to data/processed/<build_vintage>/. The CSV files contain
model levels. The absolute-difference transformations used by the paper remain
a separate modelling/transformation step, so forecasts can later be integrated
back to price levels and taxes can be re-attributed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = Path(__file__).resolve().parent
RAW_BASE = PROJECT_ROOT / "data" / "raw"
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
SCRIPT_VERSION = "2026-08-05-v5-chain-linked-gas-backcast"

# Explicit mapping from the project Bloomberg workbook to model concepts.
# E5192EA is excluded because its supplied description is industrial production.
# The supplied descriptions identify both PDS1 and QS1 as gasoil contracts; PDS1
# therefore remains an explicit proxy for the paper's Eurobob gasoline assessment.
BLOOMBERG_COLUMNS = {
    "crude_oil_usd": "bbg_bcsl0018_index",
    "refined_petroleum_usd": "bbg_pds1_comdty",
    "refined_diesel_usd": "bbg_qs1_comdty",
    "ttf_eur_mwh": "bbg_ttfgdahd_bcfv_index",
    "eurusd": "bbg_eurusd_curncy",  # USD per EUR
}

# All three oil-market regressors leave this pipeline in EUR per barrel.
# PDS1 and QS1 are gasoil futures quoted per metric tonne and are converted
# with the 7.44 barrels/tonne factor reported for gasoil in Table A1.
COMMODITY_UNIT_CONVERSIONS = {
    "crude_oil_usd": {
        "input_unit": "USD/barrel",
        "output_unit": "EUR/barrel",
        "barrels_per_metric_tonne": None,
    },
    "refined_petroleum_usd": {
        "input_unit": "USD/metric tonne",
        "output_unit": "EUR/barrel",
        "barrels_per_metric_tonne": 7.44,
        "note": "PDS1 is a gasoil proxy, not the paper's Eurobob gasoline assessment.",
    },
    "refined_diesel_usd": {
        "input_unit": "USD/metric tonne",
        "output_unit": "EUR/barrel",
        "barrels_per_metric_tonne": 7.44,
    },
}

DEFAULT_WEEKLY_COMMODITY_SHIFT_WEEKS = 1
PRE_TAX_EUR_KWH_TO_EUR_MWH = 1_000.0

SOURCE_FILES = {
    "european_commission": ("wob_weekly.csv",),
    "world_bank": ("world_bank_monthly.csv", "world_bank.csv"),
    "bloomberg": ("bloomberg_panel.csv", "bloomberg_monthly.csv"),
    "haver": ("haver_monthly.csv",),
    "eurostat": ("eurostat_all.csv", "eurostat_semiannual.csv"),
}

PIPELINE_STEPS = (
    ("european_commission", "wob.py"),
    ("world_bank", "world_bank.py"),
    ("bloomberg", "bloomberg.py"),
    ("haver", "haver_offline_v6.py"),
    ("eurostat", "eurostat_energy_offline_v2.py"),
)

MODEL_SPECS = {
    "car_fuels": {
        "file": "car_fuels_weekly.csv",
        "frequency": "weekly",
        "lags": 24,
        "columns": [
            "wob_petrol_pre_tax",
            "wob_diesel_pre_tax",
            "crude_oil_eur",
            "refined_petroleum_eur",
            "refined_diesel_eur",
        ],
        "equations": {
            "petrol": [
                "wob_petrol_pre_tax",
                "crude_oil_eur",
                "refined_petroleum_eur",
            ],
            "diesel": [
                "wob_diesel_pre_tax",
                "crude_oil_eur",
                "refined_diesel_eur",
            ],
        },
        "target_columns": ["wob_petrol_pre_tax", "wob_diesel_pre_tax"],
    },
    "liquid_fuels": {
        "file": "liquid_fuels_weekly.csv",
        "frequency": "weekly",
        "lags": 24,
        "columns": [
            "wob_heating_oil_pre_tax",
            "crude_oil_eur",
            "refined_diesel_eur",
        ],
        "equations": {
            "liquid_fuels": [
                "wob_heating_oil_pre_tax",
                "crude_oil_eur",
                "refined_diesel_eur",
            ]
        },
        "target_columns": ["wob_heating_oil_pre_tax"],
    },
    "gas": {
        "file": "gas_monthly.csv",
        "frequency": "monthly",
        "lags": 12,
        "columns": ["gas_pre_tax", "natural_gas_eur_mwh"],
        "equations": {
            "gas": ["gas_pre_tax", "natural_gas_eur_mwh"]
        },
        "target_columns": ["gas_pre_tax"],
        "units": {
            "gas_pre_tax": "EUR/MWh",
            "natural_gas_eur_mwh": "EUR/MWh",
        },
    },
    "electricity": {
        "file": "electricity_monthly.csv",
        "frequency": "monthly",
        "lags": 12,
        "columns": ["electricity_pre_tax", "natural_gas_eur_mwh"],
        "equations": {
            "electricity": [
                "electricity_pre_tax",
                "natural_gas_eur_mwh",
            ]
        },
        "seasonal_dummies": True,
        "target_columns": ["electricity_pre_tax"],
        "units": {
            "electricity_pre_tax": "EUR/kWh",
            "natural_gas_eur_mwh": "EUR/MWh",
        },
    },
    "heat_energy": {
        "file": "heat_energy_monthly.csv",
        "frequency": "monthly",
        "lags": 12,
        "columns": ["hicp_heat_energy", "natural_gas_eur_mwh", "ppi_energy"],
        "equations": {
            "heat_energy": [
                "hicp_heat_energy",
                "natural_gas_eur_mwh",
                "ppi_energy",
            ]
        },
        "target_columns": ["hicp_heat_energy"],
    },
    "solid_fuels": {
        "file": "solid_fuels_monthly.csv",
        "frequency": "monthly",
        "lags": 12,
        "columns": ["hicp_solid_fuels", "natural_gas_eur_mwh", "ppi_energy"],
        "equations": {
            "solid_fuels": [
                "hicp_solid_fuels",
                "natural_gas_eur_mwh",
                "ppi_energy",
            ]
        },
        "target_columns": ["hicp_solid_fuels"],
    },
}

VINTAGE_RE = re.compile(r"^\d{4}-?\d{2}-?\d{2}$")

# ---------------------------------------------------------------------------
# Source-pipeline orchestration
# ---------------------------------------------------------------------------


def _first_existing_folder(parent: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return parent / names[0]


def _source_command(
    source: str,
    script_path: Path,
    *,
    refresh: bool,
    offline: bool,
) -> list[str]:
    command = [sys.executable, str(script_path)]

    if source == "european_commission":
        command += [
            "--raw-root",
            str(
                _first_existing_folder(
                    RAW_BASE,
                    ("european_commission", "european-commission"),
                )
            ),
            "--output-root",
            str(DEFAULT_INTERIM_ROOT / "european_commission"),
        ]
    elif source == "world_bank":
        command += [
            "--raw-root",
            str(_first_existing_folder(RAW_BASE, ("world_bank", "world-bank"))),
            "--output-root",
            str(DEFAULT_INTERIM_ROOT / "world_bank"),
        ]
    elif source == "bloomberg":
        command += [
            "--raw-root",
            str(
                _first_existing_folder(
                    RAW_BASE,
                    ("bloomberg", "bloomberg-data", "bloomberg_data"),
                )
            ),
            "--output-root",
            str(DEFAULT_INTERIM_ROOT / "bloomberg"),
        ]
    elif source == "haver":
        command += [
            "--raw-root",
            str(RAW_BASE / "haver"),
            "--output-root",
            str(DEFAULT_INTERIM_ROOT / "haver"),
        ]
        if refresh:
            command.append("--refresh")
        if offline:
            command.append("--offline")
    elif source == "eurostat":
        command += [
            "--raw-root",
            str(RAW_BASE / "eurostat"),
            "--output-root",
            str(DEFAULT_INTERIM_ROOT / "eurostat"),
        ]
        if refresh:
            command.append("--refresh")
        if offline:
            command.append("--offline")
    else:
        raise ValueError(f"Unknown source pipeline: {source}")

    return command


def run_source_pipelines(
    *,
    refresh: bool = False,
    offline: bool = False,
) -> list[dict[str, object]]:
    """Run all five source scripts in dependency order."""
    if refresh and offline:
        raise ValueError("--refresh and --offline cannot be used together.")

    records: list[dict[str, object]] = []
    for number, (source, filename) in enumerate(PIPELINE_STEPS, start=1):
        script_path = PIPELINE_DIR / filename
        if not script_path.is_file():
            raise FileNotFoundError(
                f"Missing pipeline script for {source}: {script_path}"
            )

        command = _source_command(
            source,
            script_path,
            refresh=refresh,
            offline=offline,
        )
        printable = " ".join(f'"{part}"' if " " in part else part for part in command)

        print("\n" + "=" * 76)
        print(f"[{number}/{len(PIPELINE_STEPS)}] Running {source}")
        print(printable)
        print("=" * 76, flush=True)

        started = time.perf_counter()
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        elapsed = time.perf_counter() - started
        record = {
            "source": source,
            "script": str(script_path.resolve()),
            "command": command,
            "return_code": int(completed.returncode),
            "elapsed_seconds": round(elapsed, 3),
        }
        records.append(record)

        if completed.returncode != 0:
            raise RuntimeError(
                f"The {source} pipeline failed with return code "
                f"{completed.returncode}. Compilation stopped."
            )

    return records


# ---------------------------------------------------------------------------
# File discovery and generic cleaning
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vintage_key(name: str) -> tuple[int, str]:
    digits = re.sub(r"\D", "", name)
    return (int(digits), name) if len(digits) == 8 else (-1, name)


def _latest_vintage_folder(source_dir: Path) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing interim source directory: {source_dir}")

    candidates = [
        path
        for path in source_dir.iterdir()
        if path.is_dir() and VINTAGE_RE.fullmatch(path.name)
    ]
    if not candidates:
        raise FileNotFoundError(f"No dated vintage folder below {source_dir}")
    return max(candidates, key=lambda path: _vintage_key(path.name))


def _resolve_vintage_folder(
    interim_root: Path,
    source: str,
    requested_vintage: str | None,
) -> Path:
    source_dir = interim_root / source
    if requested_vintage is None:
        return _latest_vintage_folder(source_dir)

    digits = re.sub(r"\D", "", requested_vintage)
    variants = {requested_vintage, digits}
    if len(digits) == 8:
        variants.add(f"{digits[:4]}-{digits[4:6]}-{digits[6:]}")

    for variant in variants:
        candidate = source_dir / variant
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Vintage {requested_vintage!r} not found below {source_dir}"
    )


def _find_source_file(vintage_dir: Path, filenames: tuple[str, ...]) -> Path:
    for filename in filenames:
        direct = vintage_dir / filename
        if direct.is_file():
            return direct
    for filename in filenames:
        matches = [path for path in vintage_dir.rglob(filename) if path.is_file()]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime_ns)
    raise FileNotFoundError(f"None of {filenames} found below {vintage_dir}")


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path}: file contains no rows.")

    date_column = next(
        (
            column
            for column in frame.columns
            if str(column).strip().lower()
            in {"date", "dates", "time", "period", "observation_date"}
        ),
        frame.columns[0],
    )
    dates = pd.to_datetime(frame.pop(date_column), errors="coerce")
    if dates.isna().all():
        raise ValueError(f"{path}: could not parse date column {date_column!r}.")

    valid = dates.notna()
    frame = frame.loc[valid].copy()
    frame.index = pd.DatetimeIndex(dates.loc[valid]).tz_localize(None)
    frame.index.name = "date"
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = (
        frame.replace([np.inf, -np.inf], np.nan)
        .groupby(level=0)
        .last()
        .sort_index()
        .dropna(how="all")
    )
    if frame.empty:
        raise ValueError(f"{path}: no usable numeric observations.")
    return frame


def _required_series(
    frame: pd.DataFrame,
    candidates: str | tuple[str, ...],
    *,
    source: str,
    logical_name: str,
) -> pd.Series:
    options = (candidates,) if isinstance(candidates, str) else candidates
    for column in options:
        key = column.lower()
        if key in frame.columns:
            series = frame[key].astype(float).rename(logical_name)
            if series.notna().sum() == 0:
                raise ValueError(f"{source}.{column} is entirely missing.")
            return series
    raise KeyError(
        f"Missing {logical_name!r} in {source}. Expected one of {options}. "
        f"Available columns: {list(frame.columns)}"
    )


def _monthly_mean(series: pd.Series) -> pd.Series:
    out = series.groupby(series.index.to_period("M")).mean()
    out.index = out.index.to_timestamp(how="start")
    out.index.name = "date"
    return out.sort_index()


def _monthly_last(series: pd.Series) -> pd.Series:
    out = series.groupby(series.index.to_period("M")).last()
    out.index = out.index.to_timestamp(how="start")
    out.index.name = "date"
    return out.sort_index()


def _weekly_mean(series: pd.Series) -> pd.Series:
    """Monday-Sunday mean, labelled by the Monday starting the week."""
    out = series.groupby(series.index.to_period("W-SUN")).mean()
    out.index = out.index.start_time
    out.index.name = "date"
    return out.sort_index()


def _weekly_last(series: pd.Series) -> pd.Series:
    out = series.groupby(series.index.to_period("W-SUN")).last()
    out.index = out.index.start_time
    out.index.name = "date"
    return out.sort_index()


def _regular_panel(
    columns: dict[str, pd.Series],
    *,
    frequency: str,
    target_columns: list[str],
) -> pd.DataFrame:
    panel = pd.concat(columns, axis=1).sort_index()

    target_first_dates = [
        panel[column].first_valid_index()
        for column in target_columns
        if panel[column].first_valid_index() is not None
    ]
    if len(target_first_dates) != len(target_columns):
        missing = [column for column in target_columns if panel[column].notna().sum() == 0]
        raise ValueError(f"Target series are entirely missing: {missing}")

    start = min(target_first_dates)
    end = panel.last_valid_index()
    if end is None:
        raise ValueError("Cannot build a panel with no observed values.")

    if frequency == "weekly":
        start = pd.Timestamp(start).to_period("W-SUN").start_time
        end = pd.Timestamp(end).to_period("W-SUN").start_time
        index = pd.date_range(start, end, freq="W-MON", name="date")
    elif frequency == "monthly":
        start = pd.Timestamp(start).to_period("M").to_timestamp(how="start")
        end = pd.Timestamp(end).to_period("M").to_timestamp(how="start")
        index = pd.date_range(start, end, freq="MS", name="date")
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    regular = panel.reindex(index)
    observed_any = regular.notna().any(axis=1)
    if not observed_any.any():
        raise ValueError("Cannot build a panel with no observed values.")
    first = observed_any[observed_any].index[0]
    last = observed_any[observed_any].index[-1]
    return regular.loc[first:last]


# ---------------------------------------------------------------------------
# Paper-specific construction
# ---------------------------------------------------------------------------


def _commodity_prices_in_euro(bloomberg: pd.DataFrame) -> dict[str, pd.Series]:
    fx = _required_series(
        bloomberg,
        BLOOMBERG_COLUMNS["eurusd"],
        source="bloomberg",
        logical_name="eurusd",
    )
    if (fx.dropna() <= 0).any():
        raise ValueError("EURUSD must be strictly positive.")

    output: dict[str, pd.Series] = {}
    for logical in (
        "crude_oil_usd",
        "refined_petroleum_usd",
        "refined_diesel_usd",
    ):
        price = _required_series(
            bloomberg,
            BLOOMBERG_COLUMNS[logical],
            source="bloomberg",
            logical_name=logical,
        )
        aligned = pd.concat([price, fx], axis=1)
        eur_name = logical.replace("_usd", "_eur")
        converted = aligned[logical] / aligned["eurusd"]
        barrels_per_tonne = COMMODITY_UNIT_CONVERSIONS[logical][
            "barrels_per_metric_tonne"
        ]
        if barrels_per_tonne is not None:
            converted = converted / float(barrels_per_tonne)
        output[eur_name] = converted.rename(eur_name)

    output["eurusd"] = fx
    output["ttf_eur_mwh"] = _required_series(
        bloomberg,
        BLOOMBERG_COLUMNS["ttf_eur_mwh"],
        source="bloomberg",
        logical_name="ttf_eur_mwh",
    )
    return output


def _ols_scale_no_intercept(x: pd.Series, y: pd.Series, *, label: str) -> float:
    common = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(common) < 6:
        raise ValueError(
            f"{label}: at least 6 overlapping observations are required; "
            f"found {len(common)}."
        )
    denominator = float(np.dot(common["x"], common["x"]))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"{label}: invalid OLS denominator.")
    scale = float(np.dot(common["x"], common["y"]) / denominator)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{label}: estimated scale must be positive; got {scale}.")
    return scale


def _build_natural_gas_series(
    bloomberg_prices: dict[str, pd.Series],
    world_bank: pd.DataFrame,
) -> tuple[pd.Series, dict[str, object]]:
    """Monthly TTF, backcast with border-gas growth rates (chain-link).

    The observed TTF series is kept unchanged over its own sample. Earlier
    months follow the month-on-month dynamics of the World Bank ``Natural gas,
    Europe`` proxy, with the proxy level pinned to TTF at the junction. This is
    the ratio form of backward chain-linking and estimates neither an OLS scale
    nor an arbitrary overlap window.
    """
    ttf = _monthly_mean(bloomberg_prices["ttf_eur_mwh"]).rename("ttf")
    fx_monthly = _monthly_mean(bloomberg_prices["eurusd"]).rename("eurusd")
    wb_usd = _monthly_last(
        _required_series(
            world_bank,
            "wb_natural_gas_europe_usd_mmbtu",
            source="world_bank",
            logical_name="wb_usd_mmbtu",
        )
    )

    # Convert the proxy into EUR. The fixed MMBtu-to-MWh multiplier is not
    # required here: it cancels from proxy_t / proxy_anchor, while the anchor
    # ratio maps the chain-linked history exactly into the TTF EUR/MWh scale.
    aligned = pd.concat([wb_usd, fx_monthly], axis=1)
    invalid_fx = aligned["eurusd"].notna() & (aligned["eurusd"] <= 0)
    if invalid_fx.any():
        bad_date = invalid_fx[invalid_fx].index[0]
        raise ValueError(
            "EURUSD must be strictly positive when observed; "
            f"invalid value at {bad_date.date().isoformat()}."
        )
    proxy_eur = (
        aligned["wb_usd_mmbtu"] / aligned["eurusd"]
    ).rename("proxy_eur")

    first_ttf = ttf.first_valid_index()
    first_proxy = proxy_eur.first_valid_index()
    if first_ttf is None:
        raise ValueError("The Bloomberg TTF series is entirely missing.")
    if first_proxy is None:
        raise ValueError("The World Bank natural-gas proxy is entirely missing.")

    # Prefer the first observed TTF month as the exact junction. If the proxy is
    # unavailable in that month, use the earliest month where both series are
    # observed. This fallback is explicit in the diagnostics.
    proxy_at_first_ttf = proxy_eur.get(first_ttf, np.nan)
    if np.isfinite(proxy_at_first_ttf):
        anchor_date = first_ttf
        anchor_fallback_used = False
    else:
        overlap = pd.concat([proxy_eur, ttf], axis=1).dropna().sort_index()
        if overlap.empty:
            raise ValueError("No overlap between TTF and the border-gas proxy.")
        anchor_date = overlap.index.min()
        anchor_fallback_used = True

    ttf_anchor = float(ttf.loc[anchor_date])
    proxy_anchor = float(proxy_eur.loc[anchor_date])
    if not np.isfinite(ttf_anchor) or ttf_anchor <= 0:
        raise ValueError(
            "The TTF anchor must be finite and strictly positive; "
            f"got {ttf_anchor} at {anchor_date.date().isoformat()}."
        )
    if not np.isfinite(proxy_anchor) or proxy_anchor <= 0:
        raise ValueError(
            "The border-gas proxy anchor must be finite and strictly positive; "
            f"got {proxy_anchor} at {anchor_date.date().isoformat()}."
        )

    anchor_ratio = ttf_anchor / proxy_anchor
    backcast = (anchor_ratio * proxy_eur).rename("natural_gas_eur_mwh")

    full_index = pd.date_range(
        min(first_proxy, first_ttf),
        max(proxy_eur.last_valid_index(), ttf.last_valid_index()),
        freq="MS",
        name="date",
    )
    combined = ttf.reindex(full_index).rename("natural_gas_eur_mwh")
    backcast = backcast.reindex(full_index)

    # TTF always dominates from its first valid observation onward. The proxy is
    # used only for the pre-TTF history; no blending occurs in the overlap.
    backcast_mask = combined.isna() & (combined.index < first_ttf)
    combined.loc[backcast_mask] = backcast.loc[backcast_mask]

    unresolved_pre_ttf = (
        (combined.index < first_ttf)
        & combined.isna().to_numpy()
    )
    junction_backcast_value = float(backcast.loc[anchor_date])
    junction_gap = float(junction_backcast_value - ttf_anchor)

    info = {
        "method": (
            "Bloomberg TTF monthly mean; earlier months backcast with the "
            "World Bank Natural gas, Europe proxy via backward chain-linking. "
            "The proxy level is pinned to TTF at the anchor month; no overlap "
            "window and no level regression are estimated."
        ),
        "formula": (
            "backcast_t = TTF_anchor * proxy_EUR_t / proxy_EUR_anchor"
        ),
        "proxy_note": (
            "The paper uses Energy Intelligence European border gas. World "
            "Bank Natural gas, Europe is the public substitute and belongs to "
            "the same European border/import-price family."
        ),
        "proxy_input_unit": "USD/MMBtu",
        "proxy_currency_conversion": "USD/MMBtu divided by USD-per-EUR",
        "output_unit": "EUR/MWh (set by the TTF anchor ratio)",
        "anchor_date": anchor_date.date().isoformat(),
        "anchor_ratio": float(anchor_ratio),
        "anchor_fallback_used": bool(anchor_fallback_used),
        "first_ttf_date": first_ttf.date().isoformat(),
        "first_proxy_date": first_proxy.date().isoformat(),
        "junction_gap_eur_mwh": junction_gap,
        "backcast_observations": int(backcast_mask.sum()),
        "unresolved_pre_ttf_missing": int(unresolved_pre_ttf.sum()),
    }
    return combined, info

def _expand_semesters(
    series: pd.Series,
    monthly_index: pd.DatetimeIndex,
    *,
    carry_forward_edge: bool = True,
) -> tuple[pd.Series, int]:
    """Hold each semester for six months and optionally extend only the edge."""
    sparse = _monthly_last(series.dropna())
    if sparse.empty:
        raise ValueError(f"{series.name}: no semiannual observations available.")

    expanded = sparse.reindex(monthly_index).ffill(limit=5)
    carried = 0
    last_observation = sparse.last_valid_index()
    valid_until = last_observation + pd.DateOffset(months=5)

    if carry_forward_edge:
        tail = monthly_index > valid_until
        carried = int((tail & expanded.isna()).sum())
        expanded.loc[tail] = expanded.loc[tail].fillna(sparse.loc[last_observation])

    expanded.name = series.name
    return expanded, carried


def _build_pre_tax_price(
    *,
    hicp: pd.Series,
    wtax: pd.Series,
    vat: pd.Series,
    exc: pd.Series,
    name: str,
    price_unit_multiplier: float = 1.0,
    source_price_unit: str = "source price unit",
    output_price_unit: str = "source price unit",
) -> tuple[pd.Series, dict[str, object]]:
    """Construct a monthly pre-tax consumer price in an explicit output unit.

    The unit conversion is applied to both the Eurostat after-tax price and
    excise before estimating the HICP rescaling factor and removing taxes. VAT
    is a percentage and therefore requires no conversion. For the gas model we
    use a multiplier of 1,000 to harmonise gas_pre_tax with wholesale gas in
    EUR/MWh; other components can retain their native unit.
    """
    if not np.isfinite(price_unit_multiplier) or price_unit_multiplier <= 0:
        raise ValueError("price_unit_multiplier must be strictly positive.")

    hicp = _monthly_last(hicp).rename("hicp")
    wtax_sparse = (
        price_unit_multiplier * _monthly_last(wtax.dropna())
    ).rename("wtax")
    vat_sparse = _monthly_last(vat.dropna()).rename("vat")
    exc_sparse = (
        price_unit_multiplier * _monthly_last(exc.dropna())
    ).rename("exc")

    scale = _ols_scale_no_intercept(
        hicp.reindex(wtax_sparse.index),
        wtax_sparse,
        label=f"{name}: HICP -> Eurostat after-tax price ({output_price_unit})",
    )

    start = min(
        date
        for date in (
            hicp.first_valid_index(),
            vat_sparse.first_valid_index(),
            exc_sparse.first_valid_index(),
        )
        if date is not None
    )
    end = hicp.last_valid_index()
    if end is None:
        raise ValueError(f"{name}: HICP target is entirely missing.")
    index = pd.date_range(start, end, freq="MS", name="date")

    hicp_m = hicp.reindex(index)
    vat_m, vat_carried = _expand_semesters(vat_sparse, index)
    exc_m, exc_carried = _expand_semesters(exc_sparse, index)
    after_tax_implied = scale * hicp_m
    pre_tax = (after_tax_implied / (1.0 + vat_m / 100.0) - exc_m).rename(name)

    if (pre_tax.dropna() <= 0).any():
        bad = pre_tax[pre_tax <= 0].head().to_dict()
        raise ValueError(f"{name}: non-positive constructed pre-tax prices: {bad}")

    fitted_at_observations = (scale * hicp).reindex(wtax_sparse.index)
    fit = pd.concat(
        [wtax_sparse, fitted_at_observations.rename("fitted")],
        axis=1,
    ).dropna()
    rmse = float(np.sqrt(np.mean((fit["wtax"] - fit["fitted"]) ** 2)))

    # At semiannual dates, the constructed series should be close to the
    # directly observed Eurostat no-tax price. This is a diagnostic only:
    # the paper's target is the HICP-scaled monthly series.
    info = {
        "formula": (
            "pre_tax_EUR_MWh = gamma_EUR_MWh * HICP / "
            "(1 + VAT/100) - EXC_EUR_MWh"
        ),
        "source_price_unit": source_price_unit,
        "output_price_unit": output_price_unit,
        "price_unit_multiplier": float(price_unit_multiplier),
        "gamma": scale,
        "gamma_unit": f"{output_price_unit} per HICP index point",
        "ols_observations": int(len(fit)),
        "after_tax_fit_rmse": rmse,
        "after_tax_fit_rmse_unit": output_price_unit,
        "tax_months_carried_forward": {
            "vat": vat_carried,
            "excise": exc_carried,
        },
        "tax_edge_assumption": (
            "After the final published six-month semester, VAT and excise "
            "are held constant through the last available monthly HICP "
            "observation, matching the paper's unchanged-tax assumption."
        ),
        "first_valid": (
            pre_tax.first_valid_index().date().isoformat()
            if pre_tax.first_valid_index() is not None else None
        ),
        "last_valid": (
            pre_tax.last_valid_index().date().isoformat()
            if pre_tax.last_valid_index() is not None else None
        ),
    }
    return pre_tax, info


def _validate_model_panel(name: str, panel: pd.DataFrame, spec: dict) -> None:
    expected_columns = spec["columns"]
    if list(panel.columns) != expected_columns:
        raise ValueError(
            f"{name}: columns differ from the model specification. "
            f"Expected {expected_columns}, found {list(panel.columns)}"
        )
    if panel.empty:
        raise ValueError(f"{name}: output panel is empty.")
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError(f"{name}: dates must be unique and sorted.")
    if np.isinf(panel.to_numpy(dtype=float)).any():
        raise ValueError(f"{name}: infinite values found.")
    if panel.isna().all().any():
        bad = panel.columns[panel.isna().all()].tolist()
        raise ValueError(f"{name}: entirely missing columns: {bad}")

    if spec["frequency"] == "weekly":
        if not (panel.index.weekday == 0).all():
            raise ValueError(f"{name}: weekly observations must be Monday-labelled.")
        expected = pd.date_range(panel.index.min(), panel.index.max(), freq="W-MON")
    else:
        if not (panel.index.day == 1).all():
            raise ValueError(f"{name}: monthly dates must be month-starts.")
        expected = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")

    if not panel.index.equals(expected):
        raise ValueError(f"{name}: the calendar index is not regular.")


def _diagnostic_rows(
    model: str,
    panel: pd.DataFrame,
    spec: dict,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for column in panel.columns:
        observed = panel[column].dropna()
        if observed.empty:
            internal_gaps = 0
            leading_missing = len(panel)
            trailing_missing = len(panel)
        else:
            inside = panel[column].loc[observed.index.min(): observed.index.max()]
            internal_gaps = int(inside.isna().sum())
            leading_missing = int(panel.index.get_loc(observed.index.min()))
            trailing_missing = int(
                len(panel) - 1 - panel.index.get_loc(observed.index.max())
            )

        rows.append(
            {
                "diagnostic_type": "series",
                "model": model,
                "equation": None,
                "output_file": spec["file"],
                "frequency": spec["frequency"],
                "lags_in_paper": spec["lags"],
                "series": column,
                "observations": int(observed.size),
                "first_date": (
                    observed.index.min().date().isoformat()
                    if not observed.empty else None
                ),
                "last_date": (
                    observed.index.max().date().isoformat()
                    if not observed.empty else None
                ),
                "missing_values": int(panel[column].isna().sum()),
                "internal_gaps": internal_gaps,
                "leading_missing": leading_missing,
                "trailing_missing": trailing_missing,
                "coverage_percent": round(100.0 * observed.size / len(panel), 2),
                "balanced_level_observations": None,
                "balanced_difference_observations": None,
                "usable_after_difference_and_lags": None,
            }
        )

    for equation, columns in spec["equations"].items():
        equation_panel = panel[columns]
        balanced_levels = equation_panel.dropna()
        balanced_differences = equation_panel.diff().dropna()
        usable = max(0, len(balanced_differences) - int(spec["lags"]))
        rows.append(
            {
                "diagnostic_type": "equation_sample",
                "model": model,
                "equation": equation,
                "output_file": spec["file"],
                "frequency": spec["frequency"],
                "lags_in_paper": spec["lags"],
                "series": " | ".join(columns),
                "observations": int(len(balanced_levels)),
                "first_date": (
                    balanced_levels.index.min().date().isoformat()
                    if not balanced_levels.empty else None
                ),
                "last_date": (
                    balanced_levels.index.max().date().isoformat()
                    if not balanced_levels.empty else None
                ),
                "missing_values": int(len(panel) - len(balanced_levels)),
                "internal_gaps": int(
                    equation_panel.isna().any(axis=1).loc[
                        equation_panel.notna().any(axis=1).idxmax():
                        equation_panel.notna().any(axis=1)[::-1].idxmax()
                    ].sum()
                ) if equation_panel.notna().any(axis=1).any() else 0,
                "leading_missing": None,
                "trailing_missing": None,
                "coverage_percent": round(
                    100.0 * len(balanced_levels) / len(panel), 2
                ),
                "balanced_level_observations": int(len(balanced_levels)),
                "balanced_difference_observations": int(len(balanced_differences)),
                "usable_after_difference_and_lags": int(usable),
            }
        )

    return rows


def build_model_datasets(
    *,
    interim_root: Path = DEFAULT_INTERIM_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_vintage: str | None = None,
    build_vintage: str | None = None,
    pipeline_runs: list[dict[str, object]] | None = None,
    weekly_commodity_shift_weeks: int = DEFAULT_WEEKLY_COMMODITY_SHIFT_WEEKS,
) -> dict[str, Path]:
    """Compile the five interim sources into the paper's six datasets."""
    interim_root = Path(interim_root)
    build_vintage = build_vintage or datetime.now().strftime("%Y%m%d")
    if weekly_commodity_shift_weeks not in (0, 1):
        raise ValueError("weekly_commodity_shift_weeks must be 0 or 1.")

    source_frames: dict[str, pd.DataFrame] = {}
    source_files: dict[str, Path] = {}
    source_vintages: dict[str, str] = {}

    for source, filenames in SOURCE_FILES.items():
        vintage_dir = _resolve_vintage_folder(interim_root, source, source_vintage)
        source_file = _find_source_file(vintage_dir, filenames)
        source_frames[source] = _read_csv(source_file)
        source_files[source] = source_file
        source_vintages[source] = vintage_dir.name

    wob = source_frames["european_commission"]
    world_bank = source_frames["world_bank"]
    bloomberg = source_frames["bloomberg"]
    haver = source_frames["haver"]
    eurostat = source_frames["eurostat"]

    # Daily Bloomberg commodities are converted into EUR/barrel first.
    # With shift=1 (default), the WOB observation dated Monday t is aligned
    # with the commodity average from Monday t-7 through Sunday t-1. shift=0
    # reproduces the former same-week convention but uses information after t.
    bbg = _commodity_prices_in_euro(bloomberg)

    def _weekly_commodity(series: pd.Series) -> pd.Series:
        weekly = _weekly_mean(series)
        if weekly_commodity_shift_weeks:
            weekly.index = weekly.index + pd.DateOffset(
                weeks=weekly_commodity_shift_weeks
            )
            weekly.index.name = "date"
        return weekly

    crude_weekly = _weekly_commodity(bbg["crude_oil_eur"])
    refined_petroleum_weekly = _weekly_commodity(bbg["refined_petroleum_eur"])
    refined_diesel_weekly = _weekly_commodity(bbg["refined_diesel_eur"])

    wob_petrol = _weekly_last(
        _required_series(
            wob,
            "wob_petroleum_cpr_ntax",
            source="european_commission",
            logical_name="wob_petrol_pre_tax",
        )
    )
    wob_diesel = _weekly_last(
        _required_series(
            wob,
            "wob_diesel_cpr_ntax",
            source="european_commission",
            logical_name="wob_diesel_pre_tax",
        )
    )
    wob_heating = _weekly_last(
        _required_series(
            wob,
            ("wob_heating_oil_cpr_ntax", "wob_gas_cpr_ntax"),
            source="european_commission",
            logical_name="wob_heating_oil_pre_tax",
        )
    )

    car_fuels = _regular_panel(
        {
            "wob_petrol_pre_tax": wob_petrol,
            "wob_diesel_pre_tax": wob_diesel,
            "crude_oil_eur": crude_weekly,
            "refined_petroleum_eur": refined_petroleum_weekly,
            "refined_diesel_eur": refined_diesel_weekly,
        },
        frequency="weekly",
        target_columns=MODEL_SPECS["car_fuels"]["target_columns"],
    )
    liquid_fuels = _regular_panel(
        {
            "wob_heating_oil_pre_tax": wob_heating,
            "crude_oil_eur": crude_weekly,
            "refined_diesel_eur": refined_diesel_weekly,
        },
        frequency="weekly",
        target_columns=MODEL_SPECS["liquid_fuels"]["target_columns"],
    )

    natural_gas, gas_backcast_info = _build_natural_gas_series(
        bbg,
        world_bank,
    )

    hicp_gas = _required_series(
        haver,
        "hicp_gas",
        source="haver",
        logical_name="hicp_gas",
    )
    hicp_electricity = _required_series(
        haver,
        "hicp_electricity",
        source="haver",
        logical_name="hicp_electricity",
    )

    gas_pre_tax, gas_pre_tax_info = _build_pre_tax_price(
        hicp=hicp_gas,
        wtax=_required_series(
            eurostat,
            "estat_gas_household_cpr_wtax",
            source="eurostat",
            logical_name="gas_wtax",
        ),
        vat=_required_series(
            eurostat,
            "estat_gas_household_vat",
            source="eurostat",
            logical_name="gas_vat",
        ),
        exc=_required_series(
            eurostat,
            "estat_gas_household_exc",
            source="eurostat",
            logical_name="gas_exc",
        ),
        name="gas_pre_tax",
        price_unit_multiplier=PRE_TAX_EUR_KWH_TO_EUR_MWH,
        source_price_unit="EUR/kWh",
        output_price_unit="EUR/MWh",
    )
    electricity_pre_tax, electricity_pre_tax_info = _build_pre_tax_price(
        hicp=hicp_electricity,
        wtax=_required_series(
            eurostat,
            "estat_electricity_household_cpr_wtax",
            source="eurostat",
            logical_name="electricity_wtax",
        ),
        vat=_required_series(
            eurostat,
            "estat_electricity_household_vat",
            source="eurostat",
            logical_name="electricity_vat",
        ),
        exc=_required_series(
            eurostat,
            "estat_electricity_household_exc",
            source="eurostat",
            logical_name="electricity_exc",
        ),
        name="electricity_pre_tax",
        price_unit_multiplier=1.0,
        source_price_unit="EUR/kWh",
        output_price_unit="EUR/kWh",
    )

    hicp_heat = _monthly_last(
        _required_series(
            haver,
            ("hicp_heat_energy", "hicp_heat_cooling_energy"),
            source="haver",
            logical_name="hicp_heat_energy",
        )
    )
    hicp_solid = _monthly_last(
        _required_series(
            haver,
            "hicp_solid_fuels",
            source="haver",
            logical_name="hicp_solid_fuels",
        )
    )
    ppi_energy = _monthly_last(
        _required_series(
            haver,
            "ppi_energy",
            source="haver",
            logical_name="ppi_energy",
        )
    )

    gas = _regular_panel(
        {"gas_pre_tax": gas_pre_tax, "natural_gas_eur_mwh": natural_gas},
        frequency="monthly",
        target_columns=MODEL_SPECS["gas"]["target_columns"],
    )
    electricity = _regular_panel(
        {
            "electricity_pre_tax": electricity_pre_tax,
            "natural_gas_eur_mwh": natural_gas,
        },
        frequency="monthly",
        target_columns=MODEL_SPECS["electricity"]["target_columns"],
    )
    heat_energy = _regular_panel(
        {
            "hicp_heat_energy": hicp_heat,
            "natural_gas_eur_mwh": natural_gas,
            "ppi_energy": ppi_energy,
        },
        frequency="monthly",
        target_columns=MODEL_SPECS["heat_energy"]["target_columns"],
    )
    solid_fuels = _regular_panel(
        {
            "hicp_solid_fuels": hicp_solid,
            "natural_gas_eur_mwh": natural_gas,
            "ppi_energy": ppi_energy,
        },
        frequency="monthly",
        target_columns=MODEL_SPECS["solid_fuels"]["target_columns"],
    )

    panels = {
        "car_fuels": car_fuels,
        "liquid_fuels": liquid_fuels,
        "gas": gas,
        "electricity": electricity,
        "heat_energy": heat_energy,
        "solid_fuels": solid_fuels,
    }
    for name, panel in panels.items():
        _validate_model_panel(name, panel, MODEL_SPECS[name])

    output_dir = Path(output_root) / build_vintage
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove outputs from the superseded one-big-monthly-dataset design.
    for obsolete in (
        "energy_dataset_monthly.csv",
        "energy_dataset_diagnostics.csv",
    ):
        path = output_dir / obsolete
        if path.exists():
            path.unlink()

    outputs: dict[str, Path] = {}
    diagnostic_rows: list[dict[str, object]] = []
    for name, panel in panels.items():
        spec = MODEL_SPECS[name]
        path = output_dir / spec["file"]
        panel.to_csv(path, date_format="%Y-%m-%d")
        outputs[name] = path
        diagnostic_rows.extend(_diagnostic_rows(name, panel, spec))

    diagnostics_path = output_dir / "model_datasets_diagnostics.csv"
    sources_path = output_dir / "source_vintages.csv"
    manifest_path = output_dir / "manifest.json"
    pd.DataFrame(diagnostic_rows).to_csv(diagnostics_path, index=False)

    source_table = pd.DataFrame(
        [
            {
                "source": source,
                "vintage": source_vintages[source],
                "file": str(source_files[source].resolve()),
                "sha256": _sha256(source_files[source]),
                "input_rows": int(len(source_frames[source])),
                "input_columns": int(source_frames[source].shape[1]),
            }
            for source in SOURCE_FILES
        ]
    )
    source_table.to_csv(sources_path, index=False)

    manifest = {
        "paper": "A new model to forecast energy inflation in the euro area, ECB WPS 3062",
        "script_version": SCRIPT_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "build_vintage": build_vintage,
        "requested_source_vintage": source_vintage,
        "source_pipeline_runs": pipeline_runs or [],
        "source_vintages": source_vintages,
        "source_files": {
            source: str(path.resolve()) for source, path in source_files.items()
        },
        "source_sha256": {
            source: _sha256(path) for source, path in source_files.items()
        },
        "bloomberg_mapping": BLOOMBERG_COLUMNS,
        "commodity_unit_conversions": COMMODITY_UNIT_CONVERSIONS,
        "known_approximations_vs_paper": [
            "World Bank Natural gas, Europe substitutes for the paid Energy Intelligence European border-gas series; the pre-TTF history is chain-linked to TTF without estimated splice parameters.",
            "PDS1 is described as a European gasoil averaging future and is used as a proxy for the paper's Eurobob gasoline assessment.",
            "QS1 is a Low Sulphur Gas Oil future rather than the paper's Refinitiv spot assessment.",
            "The current pipeline builds one current vintage; the paper evaluates 38 real-time vintages.",
            "The available WOB workbook has a shorter history than the paper's 1994 starting sample.",
            "Eurostat household bands D2 for gas and DC for electricity are project choices.",
        ],
        "construction": {
            "weekly_calendar": {
                "week_definition": "Monday-Sunday, labelled by Monday",
                "commodity_shift_weeks": int(weekly_commodity_shift_weeks),
                "alignment": (
                    "shift=1 aligns WOB Monday t with the preceding completed "
                    "Monday-Sunday commodity week; shift=0 aligns it with the "
                    "same week and therefore includes post-Monday quotations."
                ),
                "paper_status": (
                    "The paper keeps weekly models but does not state this "
                    "daily-to-weekly alignment convention explicitly."
                ),
            },
            "pre_tax": {
                "gas_unit_harmonisation": (
                    "Gas household price and excise inputs are converted from "
                    "EUR/kWh to EUR/MWh by multiplying by 1,000 before "
                    "estimating gamma and removing taxes. Electricity retains "
                    "its native EUR/kWh unit in this pipeline version."
                ),
                "gas": gas_pre_tax_info,
                "electricity": electricity_pre_tax_info,
            },
            "natural_gas": gas_backcast_info,
            "transformations": (
                "Outputs contain levels. Apply ordinary absolute differences "
                "before BVAR estimation, as specified in the paper."
            ),
            "ragged_edge": (
                "Regular calendars are retained with NaN where a series has "
                "not yet been released. Do not drop the ragged edge here."
            ),
        },
        "models": {
            name: {
                **spec,
                "path": str(outputs[name].resolve()),
                "rows": int(len(panels[name])),
                "first_date": panels[name].index.min().date().isoformat(),
                "last_date": panels[name].index.max().date().isoformat(),
            }
            for name, spec in MODEL_SPECS.items()
        },
        "auxiliary_outputs": {
            "diagnostics": str(diagnostics_path.resolve()),
            "source_vintages": str(sources_path.resolve()),
        },
        "output_sha256": {
            **{name: _sha256(path) for name, path in outputs.items()},
            "diagnostics": _sha256(diagnostics_path),
            "source_vintages": _sha256(sources_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    outputs.update(
        {
            "diagnostics": diagnostics_path,
            "source_vintages": sources_path,
            "manifest": manifest_path,
        }
    )

    print(f"Build vintage: {build_vintage}")
    print("Source vintages:")
    for source, vintage in source_vintages.items():
        print(f"  {source:22s} {vintage}")
    print("\nPaper model datasets:")
    for name, panel in panels.items():
        spec = MODEL_SPECS[name]
        print(
            f"  {spec['file']:30s} {panel.index.min().date()} -> "
            f"{panel.index.max().date()} | {len(panel):,} rows | "
            f"{panel.shape[1]} series"
        )
    print(f"\nDiagnostics: {diagnostics_path}")
    print(f"Manifest:    {manifest_path}")
    return outputs


# Backward-compatible function name for code that imported build_dataset().
def build_dataset(**kwargs) -> dict[str, Path]:
    return build_model_datasets(**kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source-vintage",
        default=None,
        help=(
            "Optional common input vintage. By default the latest available "
            "vintage is selected separately for each source."
        ),
    )
    parser.add_argument(
        "--build-vintage",
        default=None,
        help="Output folder vintage. Default: today as YYYYMMDD.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Skip source scripts and compile existing interim outputs only.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh downloadable Haver and Eurostat snapshots before compiling.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use existing Haver and Eurostat raw snapshots only.",
    )
    parser.add_argument(
        "--weekly-commodity-shift-weeks",
        type=int,
        choices=(0, 1),
        default=DEFAULT_WEEKLY_COMMODITY_SHIFT_WEEKS,
        help=(
            "1 (default): WOB Monday uses the preceding completed commodity "
            "week. 0: use the same Monday-Sunday week."
        ),
    )
    return parser


def main() -> None:
    print(f"Running {Path(__file__).name} [{SCRIPT_VERSION}]")
    args = _parser().parse_args()

    pipeline_runs: list[dict[str, object]] = []
    if not args.compile_only:
        pipeline_runs = run_source_pipelines(
            refresh=args.refresh,
            offline=args.offline,
        )

    print("\n" + "=" * 76)
    print("[FINAL] Building the six paper-specific model datasets")
    print("=" * 76)
    build_model_datasets(
        interim_root=args.interim_root,
        output_root=args.output_root,
        source_vintage=args.source_vintage,
        build_vintage=args.build_vintage,
        pipeline_runs=pipeline_runs,
        weekly_commodity_shift_weeks=args.weekly_commodity_shift_weeks,
    )


if __name__ == "__main__":
    main()
