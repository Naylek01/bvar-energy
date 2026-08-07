"""Build the six ECB energy-STIP model datasets from ONE Excel workbook.

This version replaces the former raw -> five source scripts -> interim -> builder
architecture with one auditable workbook input.

Expected workbook sheets
------------------------
The normal names are:

    Haver
    Bloomberg
    European_Commission
    World_Bank
    Eurostat
    Metadata                 (optional; audit only)

``Bloomberg`` must be the VALUES-ONLY snapshot used by Python. A separate
``Bloomberg_Live`` sheet may contain Bloomberg formulas; this builder ignores it.
The source sheets may be displayed newest-first or oldest-first: every reader
parses the dates and sorts chronologically in memory before any time-series
calculation.

Recommended raw layout
----------------------

    data/raw/20260807/raw_energy_bvar.xlsx

or simply

    data/raw/raw_energy_bvar.xlsx

Run from the project root:

    python src/data_pipeline/build_dataset_paper_six_models_v9.py

or point explicitly to the workbook:

    python src/data_pipeline/build_dataset_paper_six_models_v9.py \
        --raw data/raw/20260807/raw_energy_bvar.xlsx

Outputs are written directly to ``data/processed/<build_vintage>/``. No source
pipeline is launched and no ``data/interim`` directory is read.

Important Bloomberg convention
------------------------------
* BCSL0018 Index      : USD/barrel -> EUR/barrel
* GNEBM1 PVMO Index   : USD/metric tonne -> EUR/metric tonne
                         (NO invented tonne-to-barrel conversion)
* QS1 Comdty          : USD/metric tonne -> EUR/barrel using 7.44 bbl/t,
                         the gasoil conversion used in the paper
* TTFGDAHD BCFV Index : EUR/MWh, no FX conversion
* EURUSD Curncy       : USD per EUR

The processed model CSVs contain LEVELS, exactly as before. Absolute
differences remain a model-layer transformation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ===========================================================================
# Paths and version
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
SCRIPT_VERSION = "2026-08-07-v9-single-workbook"


# ===========================================================================
# Workbook contract
# ===========================================================================

SHEET_ALIASES = {
    "haver": ("Haver",),
    "bloomberg": (
        "Bloomberg_copy",
        "Bloomberg_snapshot",
        "Bloomberg",
    ),
    "european_commission": (
        "European_Commission",
        "EC_WOB_VIEW",
    ),
    "world_bank": (
        "World_Bank",
        "WORLD_BANK_GAS_RAW",
    ),
    "eurostat": (
        "Eurostat",
        "EUROSTAT_ENERGY_VIEW",
    ),
    "metadata": ("Metadata",),
}

LIVE_SHEET_MARKERS = ("live", "formula", "bdh")

HAVER_TICKERS = {
    "H023HICP@EUDATA": "hicp_total",
    "H023HN22@EUDATA": "hicp_car_fuels",
    "H023HW51@EUDATA": "hicp_electricity",
    "H023HW52@EUDATA": "hicp_gas",
    "H023HW53@EUDATA": "hicp_liquid_fuels",
    "H023HW54@EUDATA": "hicp_solid_fuels",
    "H023HW55@EUDATA": "hicp_heat_energy",
    "H025PP@G10": "ppi_energy",
}

# Exact Bloomberg snapshot tickers. There is deliberately NO PDS1 fallback:
# GNEBM1 PVMO is now the project's refined-petroleum / Eurobob series.
BLOOMBERG_TICKERS = {
    "crude_oil_usd": "BCSL0018 Index",
    "ttf_eur_mwh": "TTFGDAHD BCFV Index",
    "eurusd": "EURUSD Curncy",
    "refined_diesel_usd": "QS1 Comdty",
    "refined_petroleum_usd": "GNEBM1 PVMO Index",
}

BLOOMBERG_UNITS = {
    "crude_oil_usd": {
        "ticker": BLOOMBERG_TICKERS["crude_oil_usd"],
        "input_unit": "USD/barrel",
        "output_unit": "EUR/barrel",
        "barrels_per_metric_tonne": None,
    },
    "refined_petroleum_usd": {
        "ticker": BLOOMBERG_TICKERS["refined_petroleum_usd"],
        "input_unit": "USD/metric tonne",
        "output_unit": "EUR/metric tonne",
        "barrels_per_metric_tonne": None,
        "note": (
            "Eurobob oxygenated gasoline FOB Rotterdam barges. The project "
            "retains the source tonne unit after FX conversion; no density-"
            "based tonne-to-barrel conversion is invented."
        ),
    },
    "refined_diesel_usd": {
        "ticker": BLOOMBERG_TICKERS["refined_diesel_usd"],
        "input_unit": "USD/metric tonne",
        "output_unit": "EUR/barrel",
        "barrels_per_metric_tonne": 7.44,
        "note": "Gasoil/diesel conversion used in the paper: 7.44 barrels/tonne.",
    },
    "ttf_eur_mwh": {
        "ticker": BLOOMBERG_TICKERS["ttf_eur_mwh"],
        "input_unit": "EUR/MWh",
        "output_unit": "EUR/MWh",
        "barrels_per_metric_tonne": None,
    },
    "eurusd": {
        "ticker": BLOOMBERG_TICKERS["eurusd"],
        "input_unit": "USD per EUR",
        "output_unit": "USD per EUR",
        "barrels_per_metric_tonne": None,
    },
}

EC_COLUMN_MAP = {
    "EUR_price_with_tax_euro95": "wob_petroleum_cpr_wtax",
    "EUR_price_wo_tax_euro95": "wob_petroleum_cpr_ntax",
    "EUR_price_with_tax_diesel": "wob_diesel_cpr_wtax",
    "EUR_price_wo_tax_diesel": "wob_diesel_cpr_ntax",
    "EUR_price_with_tax_heating_oil": "wob_gas_cpr_wtax",
    "EUR_price_wo_tax_heating_oil": "wob_gas_cpr_ntax",
}

WORLD_BANK_REQUIRED = ("wb_natural_gas_europe_usd_mmbtu",)

EUROSTAT_REQUIRED = (
    "estat_gas_household_cpr_wtax",
    "estat_gas_household_cpr_ntax",
    "estat_gas_household_vat",
    "estat_gas_household_exc",
    "estat_electricity_household_cpr_wtax",
    "estat_electricity_household_cpr_ntax",
    "estat_electricity_household_vat",
    "estat_electricity_household_exc",
)


# ===========================================================================
# Model specification
# ===========================================================================

DEFAULT_WEEKLY_COMMODITY_SHIFT_WEEKS = 1
PRE_TAX_EUR_KWH_TO_EUR_MWH = 1_000.0
MMBTU_PER_MWH = 3.412
COMPLETE_PERIOD_COVERAGE_RATIO = 0.80
COVERAGE_REFERENCE_PERIODS = 24
DEFAULT_TAX_CARRY_WARN_MONTHS = 6
DEFAULT_TAX_CARRY_ERROR_MONTHS = 12
SEMIANNUAL_MONTHS = (1, 7)

COVERAGE_TO_PANEL_COLUMN = {
    "crude_oil_eur": "crude_oil_eur",
    "refined_petroleum_eur": "refined_petroleum_eur",
    "refined_diesel_eur": "refined_diesel_eur",
    "ttf_eur_mwh": "natural_gas_eur_mwh",
    "eurusd": None,
}

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
        "units": {
            "wob_petrol_pre_tax": "EUR/1000L",
            "wob_diesel_pre_tax": "EUR/1000L",
            "crude_oil_eur": "EUR/barrel",
            "refined_petroleum_eur": "EUR/metric tonne",
            "refined_diesel_eur": "EUR/barrel",
        },
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
        "units": {
            "wob_heating_oil_pre_tax": "EUR/1000L",
            "crude_oil_eur": "EUR/barrel",
            "refined_diesel_eur": "EUR/barrel",
        },
    },
    "gas": {
        "file": "gas_monthly.csv",
        "frequency": "monthly",
        "lags": 12,
        "columns": ["gas_pre_tax", "natural_gas_eur_mwh"],
        "equations": {"gas": ["gas_pre_tax", "natural_gas_eur_mwh"]},
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
            "electricity": ["electricity_pre_tax", "natural_gas_eur_mwh"]
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
            "heat_energy": ["hicp_heat_energy", "natural_gas_eur_mwh", "ppi_energy"]
        },
        "target_columns": ["hicp_heat_energy"],
    },
    "solid_fuels": {
        "file": "solid_fuels_monthly.csv",
        "frequency": "monthly",
        "lags": 12,
        "columns": ["hicp_solid_fuels", "natural_gas_eur_mwh", "ppi_energy"],
        "equations": {
            "solid_fuels": ["hicp_solid_fuels", "natural_gas_eur_mwh", "ppi_energy"]
        },
        "target_columns": ["hicp_solid_fuels"],
    },
}


# ===========================================================================
# Generic utilities
# ===========================================================================


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_vintage_name(name: str) -> date | None:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(name, fmt).date()
        except ValueError:
            pass
    return None


def find_raw_workbook(raw_root: str | Path = DEFAULT_RAW_ROOT) -> Path:
    """Find the latest single-workbook raw snapshot without grabbing old source XLSXs."""
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Raw-data folder does not exist: {root}")

    candidates = [
        p for p in root.rglob("raw_energy_bvar*.xlsx")
        if p.is_file() and not p.name.startswith("~$")
    ]

    if not candidates:
        top_level = [
            p for p in root.glob("*.xlsx")
            if p.is_file() and not p.name.startswith("~$")
        ]
        if len(top_level) == 1:
            candidates = top_level
        elif len(top_level) > 1:
            listing = "\n  ".join(str(p) for p in top_level)
            raise FileNotFoundError(
                "No file matching raw_energy_bvar*.xlsx was found and several "
                f"top-level XLSX files exist below {root}:\n  {listing}\n"
                "Pass --raw explicitly."
            )

    if not candidates:
        raise FileNotFoundError(
            f"No raw_energy_bvar*.xlsx workbook found below {root}. "
            "Save the refreshed raw workbook there or pass --raw."
        )

    def rank(path: Path) -> tuple[int, int, int]:
        vintage = _parse_vintage_name(path.parent.name)
        vintage_ord = vintage.toordinal() if vintage else -1
        preferred = int(path.name.lower() == "raw_energy_bvar.xlsx")
        return vintage_ord, preferred, path.stat().st_mtime_ns

    return max(candidates, key=rank)


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _norm_text(value: object) -> str:
    return _clean_text(value).casefold()


def _parse_excel_date(value: object) -> pd.Timestamp:
    """Parse real Excel dates, date strings, or Excel serial numbers."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).tz_localize(None).normalize()
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and 1 <= numeric <= 100000:
            return (pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric, unit="D")).normalize()
        return pd.NaT

    text = _clean_text(value)
    if not text or text.startswith("#"):
        return pd.NaT
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_localize(None).normalize()


def _parse_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([_parse_excel_date(v) for v in values])


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _input_order(index: pd.DatetimeIndex) -> str:
    if len(index) < 2:
        return "single_or_empty"
    if index.is_monotonic_increasing:
        return "ascending"
    if index.is_monotonic_decreasing:
        return "descending"
    return "unsorted"


def _deduplicate_frame(frame: pd.DataFrame, *, source: str) -> tuple[pd.DataFrame, int, int]:
    duplicate_mask = frame.index.duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_dates = int(frame.index[duplicate_mask].nunique())

    if duplicate_rows:
        for stamp, block in frame.loc[duplicate_mask].groupby(level=0):
            if (block.nunique(dropna=True) > 1).any():
                raise ValueError(
                    f"{source}: duplicate date {pd.Timestamp(stamp).date()} carries "
                    "conflicting numeric values. Fix the workbook snapshot."
                )
        frame = frame.groupby(level=0).last()

    return frame, duplicate_rows, duplicate_dates


def _sheet_name(
    xls: pd.ExcelFile,
    logical: str,
    *,
    required: bool = True,
) -> str | None:

    lookup = {
        name.casefold(): name
        for name in xls.sheet_names
    }

    for alias in SHEET_ALIASES[logical]:
        resolved = lookup.get(alias.casefold())

        if resolved is None:
            continue

        if (
            logical == "bloomberg"
            and any(
                marker in resolved.casefold()
                for marker in LIVE_SHEET_MARKERS
            )
        ):
            continue

        return resolved

    if not required:
        return None

    raise ValueError(
        f"Workbook is missing the {logical!r} sheet. "
        f"Expected one of {SHEET_ALIASES[logical]} "
        f"(case-insensitive). "
        f"Available sheets: {xls.sheet_names}"
    )

def _finalize_source_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    raw_rows: int,
    invalid_date_rows: int,
    error_markers: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = frame.copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    input_idx = pd.DatetimeIndex(frame.index)
    order = _input_order(input_idx)
    frame, duplicate_rows, duplicate_dates = _deduplicate_frame(frame, source=source)
    all_missing_rows = int(frame.isna().all(axis=1).sum())
    frame = frame.dropna(how="all").sort_index()
    frame.index = pd.DatetimeIndex(frame.index, name="date")

    if frame.empty:
        raise ValueError(f"{source}: no usable numeric observations were read from the workbook.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{source}: internal date normalization failed.")

    stats = {
        "raw_rows": int(raw_rows),
        "retained_rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "input_order": order,
        "invalid_date_rows": int(invalid_date_rows),
        "duplicate_date_rows": duplicate_rows,
        "duplicate_dates": duplicate_dates,
        "all_missing_rows_dropped": all_missing_rows,
        "error_markers_coerced_to_nan": int(error_markers),
        "first_date": frame.index.min().date().isoformat(),
        "last_date": frame.index.max().date().isoformat(),
    }
    return frame, stats


# ===========================================================================
# Excel readers
# ===========================================================================


def _read_haver_sheet(xls: pd.ExcelFile, sheet: str) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object)
    raw_rows = len(raw)

    ticker_locations: dict[str, tuple[int, int]] = {}
    wanted = {ticker.casefold(): ticker for ticker in HAVER_TICKERS}
    scan_rows = min(len(raw), 60)
    for r in range(scan_rows):
        for c in range(raw.shape[1]):
            text = _norm_text(raw.iat[r, c])
            if text in wanted:
                ticker_locations[wanted[text]] = (r, c)

    missing = [ticker for ticker in HAVER_TICKERS if ticker not in ticker_locations]
    if missing:
        raise ValueError(
            f"Haver sheet is missing required ticker(s): {missing}. "
            "Refresh/save the Haver block before running the builder."
        )

    first_ticker_row = min(r for r, _ in ticker_locations.values())

    # Haver's horizontal block has one row of metadata/date headers immediately
    # above the series, but locate it by content rather than by hard-coded row.
    best_header_row = None
    best_date_columns: list[tuple[int, pd.Timestamp]] = []
    for r in range(max(0, first_ticker_row - 8), first_ticker_row):
        found: list[tuple[int, pd.Timestamp]] = []
        for c, value in enumerate(raw.iloc[r].tolist()):
            text = _clean_text(value)
            if re.fullmatch(r"\d{6}", text):
                year = int(text[:4])
                month = int(text[4:])
                if 1900 <= year <= 2200 and 1 <= month <= 12:
                    found.append((c, pd.Timestamp(year, month, 1)))
        if len(found) > len(best_date_columns):
            best_header_row = r
            best_date_columns = found

    if best_header_row is None or len(best_date_columns) < 12:
        raise ValueError(
            "Could not locate the YYYYMM date columns in the Haver sheet. "
            "Expected the standard Haver horizontal export layout."
        )

    dates = pd.DatetimeIndex([stamp for _, stamp in best_date_columns], name="date")
    panel = pd.DataFrame(index=dates)
    error_markers = 0

    for ticker, output_name in HAVER_TICKERS.items():
        row, _ = ticker_locations[ticker]
        values = []
        for col, _stamp in best_date_columns:
            value = raw.iat[row, col]
            if isinstance(value, str) and value.strip().startswith("#"):
                error_markers += 1
            values.append(value)
        panel[output_name] = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()

    # The horizontal Haver block has dates in columns, not rows; there are no
    # invalid date rows once YYYYMM headers have been identified.
    return _finalize_source_frame(
        panel,
        source="haver",
        raw_rows=raw_rows,
        invalid_date_rows=0,
        error_markers=error_markers,
    )


def _read_bloomberg_sheet(xls: pd.ExcelFile, sheet: str) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object)
    raw_rows = len(raw)
    if raw.empty:
        raise ValueError("Bloomberg sheet is empty.")

    wanted_norm = {logical: _norm_text(ticker) for logical, ticker in BLOOMBERG_TICKERS.items()}
    ticker_row = None
    ticker_columns: dict[str, int] = {}

    for r in range(min(80, len(raw))):
        row_lookup = {_norm_text(value): c for c, value in enumerate(raw.iloc[r].tolist()) if _clean_text(value)}
        found = {
            logical: row_lookup[norm]
            for logical, norm in wanted_norm.items()
            if norm in row_lookup
        }
        if len(found) == len(wanted_norm):
            ticker_row = r
            ticker_columns = found
            break

    if ticker_row is None:
        available = []
        for r in range(min(20, len(raw))):
            available.extend(_clean_text(v) for v in raw.iloc[r].tolist() if _clean_text(v))
        raise ValueError(
            "Bloomberg snapshot does not contain all required tickers. Expected: "
            f"{list(BLOOMBERG_TICKERS.values())}. Ensure Python reads the "
            "VALUES-ONLY 'Bloomberg' sheet, not Bloomberg_Live."
        )

    field_row = None
    for r in range(ticker_row + 1, min(ticker_row + 15, len(raw))):
        first = _norm_text(raw.iat[r, 0])
        fields = [_norm_text(raw.iat[r, c]) for c in ticker_columns.values()]
        if first in {"date", "dates"} and all(field in {"px_last", "last_price"} for field in fields):
            field_row = r
            break

    if field_row is None:
        raise ValueError(
            "Could not locate the Bloomberg 'Dates | PX_LAST ...' field row below the ticker row."
        )

    date_values = raw.iloc[field_row + 1 :, 0]
    dates = _parse_dates(date_values)
    valid_dates = ~dates.isna()
    invalid_date_rows = int((~valid_dates).sum())

    panel = pd.DataFrame(index=pd.DatetimeIndex(dates[valid_dates], name="date"))
    error_markers = 0
    for logical, col in ticker_columns.items():
        values = raw.iloc[field_row + 1 :, col].reset_index(drop=True)
        error_markers += int(
            values.map(lambda v: isinstance(v, str) and v.strip().startswith("#")).sum()
        )
        numeric = pd.to_numeric(values, errors="coerce")
        panel[logical] = numeric.loc[np.asarray(valid_dates)].to_numpy()

    frame, stats = _finalize_source_frame(
        panel,
        source="bloomberg",
        raw_rows=raw_rows,
        invalid_date_rows=invalid_date_rows,
        error_markers=error_markers,
    )
    stats["ticker_row_excel"] = int(ticker_row + 1)
    stats["field_row_excel"] = int(field_row + 1)
    stats["tickers"] = BLOOMBERG_TICKERS
    return frame, stats


def _read_simple_sheet(
    xls: pd.ExcelFile,
    sheet: str,
    *,
    source: str,
    rename: dict[str, str] | None = None,
    required_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_excel(xls, sheet_name=sheet, dtype=object)
    raw_rows = len(raw)
    if raw.empty:
        raise ValueError(f"{source}: sheet {sheet!r} is empty.")

    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = next(
        (c for c in raw.columns if c.strip().lower() in {"date", "dates", "period", "time"}),
        raw.columns[0],
    )

    dates = _parse_dates(raw[date_col])
    valid_dates = ~dates.isna()
    invalid_date_rows = int((~valid_dates).sum())

    data = raw.loc[np.asarray(valid_dates)].drop(columns=[date_col]).copy()
    if rename:
        data = data.rename(columns=rename)
    data.columns = [str(c).strip() for c in data.columns]

    error_markers = 0
    for column in data.columns:
        error_markers += int(
            data[column].map(lambda v: isinstance(v, str) and v.strip().startswith("#")).sum()
        )
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data.index = pd.DatetimeIndex(dates[valid_dates], name="date")

    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required column(s) {missing} in sheet {sheet!r}. "
            f"Available columns: {list(data.columns)}"
        )

    frame, stats = _finalize_source_frame(
        data,
        source=source,
        raw_rows=raw_rows,
        invalid_date_rows=invalid_date_rows,
        error_markers=error_markers,
    )
    return frame, stats


def _read_workbook_sources(raw_path: Path) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, object]],
    dict[str, str],
    list[dict[str, object]],
]:
    """Read all five source blocks directly from one saved workbook."""
    try:
        xls = pd.ExcelFile(raw_path, engine="openpyxl")
    except ImportError as exc:
        raise ImportError(
            "Reading the raw .xlsx workbook requires openpyxl. Install it with "
            "'python -m pip install openpyxl'."
        ) from exc

    resolved = {
        logical: _sheet_name(xls, logical)
        for logical in ("haver", "bloomberg", "european_commission", "world_bank", "eurostat")
    }

    frames: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict[str, object]] = {}

    frames["haver"], stats["haver"] = _read_haver_sheet(xls, resolved["haver"])
    frames["bloomberg"], stats["bloomberg"] = _read_bloomberg_sheet(xls, resolved["bloomberg"])
    frames["european_commission"], stats["european_commission"] = _read_simple_sheet(
        xls,
        resolved["european_commission"],
        source="european_commission",
        rename=EC_COLUMN_MAP,
        required_columns=(
            "wob_petroleum_cpr_ntax",
            "wob_diesel_cpr_ntax",
            "wob_gas_cpr_ntax",
        ),
    )
    frames["world_bank"], stats["world_bank"] = _read_simple_sheet(
        xls,
        resolved["world_bank"],
        source="world_bank",
        required_columns=WORLD_BANK_REQUIRED,
    )
    frames["eurostat"], stats["eurostat"] = _read_simple_sheet(
        xls,
        resolved["eurostat"],
        source="eurostat",
        required_columns=EUROSTAT_REQUIRED,
    )

    # Metadata is not an input to any transformation. Preserve it only as an
    # audit snapshot in the manifest when present.
    metadata_records: list[dict[str, object]] = []
    metadata_sheet = _sheet_name(xls, "metadata", required=False)
    if metadata_sheet is not None:
        metadata_raw = pd.read_excel(xls, sheet_name=metadata_sheet, header=None, dtype=object)
        for row in metadata_raw.itertuples(index=False, name=None):
            values = [_clean_text(v) for v in row]
            if any(values):
                metadata_records.append({f"column_{i+1}": value for i, value in enumerate(values)})
        resolved["metadata"] = metadata_sheet

    return frames, stats, resolved, metadata_records


# ===========================================================================
# Frequency helpers and partial-period diagnostics
# ===========================================================================


def _required_series(
    frame: pd.DataFrame,
    candidates: str | tuple[str, ...],
    *,
    source: str,
    logical_name: str,
) -> pd.Series:
    options = (candidates,) if isinstance(candidates, str) else candidates
    for column in options:
        if column in frame.columns:
            series = frame[column].astype(float).rename(logical_name)
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


def _aggregation_coverage(
    raw: pd.Series,
    *,
    frequency: str,
    label: str,
    method: str,
    label_shift_weeks: int = 0,
) -> dict[str, object]:
    observed = raw.dropna().sort_index()
    if observed.empty:
        raise ValueError(f"{label}: no observations to aggregate.")

    if frequency == "monthly":
        periods = observed.index.to_period("M")
    elif frequency == "weekly":
        periods = observed.index.to_period("W-SUN")
    else:
        raise ValueError(f"Unsupported aggregation frequency: {frequency}")

    counts = observed.groupby(periods).size().sort_index()
    last_period = counts.index[-1]
    last_count = int(counts.iloc[-1])
    reference = counts.iloc[:-1].tail(COVERAGE_REFERENCE_PERIODS)
    expected = max(int(round(float(reference.median()))), 1) if len(reference) else last_count
    ratio = float(last_count) / float(expected) if expected else float("nan")
    complete = bool(ratio >= COMPLETE_PERIOD_COVERAGE_RATIO)
    panel_label = last_period.start_time + pd.DateOffset(weeks=label_shift_weeks)

    period_code = "M" if frequency == "monthly" else "W-SUN"
    final_days = observed[observed.index.to_period(period_code) == last_period]
    partial_mean = float(final_days.mean())
    last_input_value = float(final_days.iloc[-1])

    recent = observed.tail(COVERAGE_REFERENCE_PERIODS * max(expected, 1))
    level_diff = recent.diff().dropna()
    daily_innovation_var = float(level_diff.var(ddof=1)) if len(level_diff) > 1 else None
    positive = recent[recent > 0]
    logret = np.log(positive).diff().dropna() if len(positive) > 1 else positive.iloc[:0]
    daily_logreturn_var = float(logret.var(ddof=1)) if len(logret) > 1 else None

    return {
        "series": label,
        "frequency": frequency,
        "aggregation": method,
        "final_period_start": last_period.start_time.date().isoformat(),
        "final_period_end": last_period.end_time.date().isoformat(),
        "panel_label_date": panel_label.date().isoformat(),
        "label_shift_weeks": int(label_shift_weeks),
        "final_period_input_observations": last_count,
        "expected_input_observations": expected,
        "coverage_ratio": round(ratio, 4),
        "final_period_complete": complete,
        "first_input_date": final_days.index.min().date().isoformat(),
        "last_input_date": observed.index.max().date().isoformat(),
        "partial_mean": partial_mean,
        "last_input_value": last_input_value,
        "daily_innovation_var": daily_innovation_var,
        "daily_logreturn_var": daily_logreturn_var,
        "reference_periods_used": int(len(reference)),
    }


def partial_period_conditional_estimate(
    partial_mean: float,
    last_input_value: float,
    k: int,
    n: int,
    daily_var: float | None = None,
) -> dict[str, float | None]:
    if n <= 0 or k <= 0 or k > n:
        raise ValueError(f"Require 0 < k <= n; got k={k}, n={n}.")
    cond_mean = (k * partial_mean + (n - k) * last_input_value) / n
    if daily_var is None or n == k:
        cond_var = 0.0 if n == k else None
    else:
        cond_var = daily_var / n**2 * sum(r * r for r in range(1, n - k + 1))
    return {
        "conditional_mean": float(cond_mean),
        "conditional_variance": None if cond_var is None else float(cond_var),
        "partial_mean_bias": float(cond_mean - partial_mean),
    }


def _final_period_input_records(
    raw: pd.Series,
    *,
    frequency: str,
    panel_label_date: str,
    panel_column: str | None,
    series_label: str,
) -> list[dict[str, object]]:
    observed = raw.dropna().sort_index()
    period = "M" if frequency == "monthly" else "W-SUN"
    last_period = observed.index.to_period(period)[-1]
    final = observed[observed.index.to_period(period) == last_period]
    return [
        {
            "series": series_label,
            "panel_column": panel_column if panel_column is not None else "",
            "panel_label_date": panel_label_date,
            "frequency": frequency,
            "input_date": stamp.date().isoformat(),
            "input_value": float(value),
        }
        for stamp, value in final.items()
    ]


def _bind_coverage_to_panels(
    coverage: list[dict[str, object]],
    panels: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    bound: list[dict[str, object]] = []
    for original in coverage:
        row = dict(original)
        series = str(row["series"])
        if series not in COVERAGE_TO_PANEL_COLUMN:
            raise KeyError(f"{series} has no COVERAGE_TO_PANEL_COLUMN entry.")

        panel_column = COVERAGE_TO_PANEL_COLUMN[series]
        row["panel_column"] = panel_column
        label = pd.Timestamp(row["panel_label_date"])
        models: list[str] = []
        files: list[str] = []
        present: list[bool] = []

        if panel_column is not None:
            for name, panel in panels.items():
                if panel_column not in panel.columns:
                    continue
                models.append(name)
                files.append(MODEL_SPECS[name]["file"])
                present.append(bool(label in panel.index and pd.notna(panel.at[label, panel_column])))

        row["affected_models"] = "|".join(models)
        row["affected_files"] = "|".join(files)
        row["present_in_panels"] = int(sum(present))
        row["panel_targets"] = len(models)
        bound.append(row)
    return bound


def _warn_on_partial_periods(coverage: list[dict[str, object]]) -> None:
    for row in coverage:
        if row["final_period_complete"]:
            continue
        warnings.warn(
            f"{row['series']}: {row['panel_label_date']} is based on "
            f"{row['final_period_input_observations']}/"
            f"{row['expected_input_observations']} expected input observations "
            f"({row['coverage_ratio']:.0%}); it is a partial aggregate.",
            stacklevel=2,
        )


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


# ===========================================================================
# Paper-specific transformations
# ===========================================================================


def _commodity_prices_in_euro(bloomberg: pd.DataFrame) -> dict[str, pd.Series]:
    fx = _required_series(
        bloomberg, "eurusd", source="bloomberg", logical_name="eurusd"
    )
    if (fx.dropna() <= 0).any():
        raise ValueError("EURUSD must be strictly positive.")

    output: dict[str, pd.Series] = {}
    for logical in ("crude_oil_usd", "refined_petroleum_usd", "refined_diesel_usd"):
        price = _required_series(
            bloomberg, logical, source="bloomberg", logical_name=logical
        )
        aligned = pd.concat([price, fx], axis=1)
        eur_name = logical.replace("_usd", "_eur")
        converted = aligned[logical] / aligned["eurusd"]
        barrels_per_tonne = BLOOMBERG_UNITS[logical]["barrels_per_metric_tonne"]
        if barrels_per_tonne is not None:
            converted = converted / float(barrels_per_tonne)
        output[eur_name] = converted.rename(eur_name)

    output["eurusd"] = fx
    output["ttf_eur_mwh"] = _required_series(
        bloomberg, "ttf_eur_mwh", source="bloomberg", logical_name="ttf_eur_mwh"
    )
    return output


def _ols_scale_no_intercept(x: pd.Series, y: pd.Series, *, label: str) -> float:
    common = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(common) < 6:
        raise ValueError(f"{label}: at least 6 overlapping observations required; found {len(common)}.")
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

    aligned = pd.concat([wb_usd, fx_monthly], axis=1)
    invalid_fx = aligned["eurusd"].notna() & (aligned["eurusd"] <= 0)
    if invalid_fx.any():
        bad_date = invalid_fx[invalid_fx].index[0]
        raise ValueError(f"EURUSD invalid at {bad_date.date().isoformat()}.")
    proxy_eur = (aligned["wb_usd_mmbtu"] / aligned["eurusd"]).rename("proxy_eur")

    first_ttf = ttf.first_valid_index()
    first_proxy = proxy_eur.first_valid_index()
    if first_ttf is None:
        raise ValueError("Bloomberg TTF is entirely missing.")
    if first_proxy is None:
        raise ValueError("World Bank natural-gas proxy is entirely missing.")

    proxy_at_first_ttf = proxy_eur.get(first_ttf, np.nan)
    if np.isfinite(proxy_at_first_ttf):
        anchor_date = first_ttf
        anchor_fallback_used = False
    else:
        overlap0 = pd.concat([proxy_eur, ttf], axis=1).dropna().sort_index()
        if overlap0.empty:
            raise ValueError("No overlap between TTF and World Bank gas proxy.")
        anchor_date = overlap0.index.min()
        anchor_fallback_used = True

    ttf_anchor = float(ttf.loc[anchor_date])
    proxy_anchor = float(proxy_eur.loc[anchor_date])
    if not np.isfinite(ttf_anchor) or ttf_anchor <= 0:
        raise ValueError("TTF chain-link anchor must be positive.")
    if not np.isfinite(proxy_anchor) or proxy_anchor <= 0:
        raise ValueError("World Bank gas proxy chain-link anchor must be positive.")

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
    backcast_mask = combined.isna() & (combined.index < first_ttf)
    combined.loc[backcast_mask] = backcast.loc[backcast_mask]

    unresolved_pre_ttf = (combined.index < first_ttf) & combined.isna().to_numpy()
    implied_level_ratio = float(anchor_ratio / MMBTU_PER_MWH)
    if not 0.6 <= implied_level_ratio <= 1.6:
        warnings.warn(
            "natural_gas_eur_mwh: chain-link anchor implies TTF/proxy level "
            f"ratio {implied_level_ratio:.2f} after MMBtu-to-MWh adjustment.",
            stacklevel=2,
        )

    common = ttf.index.intersection(proxy_eur.index).sort_values()
    overlap = pd.DataFrame({"ttf": ttf.reindex(common), "proxy": proxy_eur.reindex(common)}).dropna()
    growth = np.log(overlap).diff().dropna() if len(overlap) > 1 else overlap.iloc[:0]
    if len(growth) >= 12:
        growth_correlation = float(growth["ttf"].corr(growth["proxy"]))
        growth_rmse = float(np.sqrt(np.mean((growth["ttf"] - growth["proxy"]) ** 2)))
    else:
        growth_correlation = None
        growth_rmse = None

    info = {
        "method": (
            "Bloomberg TTF monthly mean; pre-TTF months are backcast with "
            "World Bank Natural gas, Europe via backward chain-linking."
        ),
        "formula": "backcast_t = TTF_anchor * proxy_EUR_t / proxy_EUR_anchor",
        "proxy_input_unit": "USD/MMBtu",
        "proxy_currency_conversion": "USD/MMBtu divided by USD-per-EUR",
        "output_unit": "EUR/MWh (set by TTF anchor)",
        "anchor_date": anchor_date.date().isoformat(),
        "anchor_ratio": float(anchor_ratio),
        "anchor_fallback_used": bool(anchor_fallback_used),
        "first_ttf_date": first_ttf.date().isoformat(),
        "first_proxy_date": first_proxy.date().isoformat(),
        "mmbtu_per_mwh_reference": MMBTU_PER_MWH,
        "implied_ttf_over_proxy_level_ratio": implied_level_ratio,
        "overlap_months": int(len(overlap)),
        "overlap_log_growth_correlation": growth_correlation,
        "overlap_log_growth_rmse": growth_rmse,
        "backcast_observations": int(backcast_mask.sum()),
        "unresolved_pre_ttf_missing": int(unresolved_pre_ttf.sum()),
    }
    return combined, info


def _expand_semesters(
    series: pd.Series,
    monthly_index: pd.DatetimeIndex,
    *,
    carry_forward_edge: bool = True,
    warn_after_months: int = DEFAULT_TAX_CARRY_WARN_MONTHS,
    error_after_months: int = DEFAULT_TAX_CARRY_ERROR_MONTHS,
) -> tuple[pd.Series, int]:
    sparse = _monthly_last(series.dropna())
    if sparse.empty:
        raise ValueError(f"{series.name}: no semiannual observations available.")

    unexpected = sorted(set(sparse.index.month) - set(SEMIANNUAL_MONTHS))
    if unexpected:
        raise ValueError(
            f"{series.name}: semiannual observations must be January/July; found months {unexpected}."
        )

    expanded = sparse.reindex(monthly_index).ffill(limit=5)
    carried = 0
    last_observation = sparse.last_valid_index()
    valid_until = last_observation + pd.DateOffset(months=5)

    if carry_forward_edge:
        tail = monthly_index > valid_until
        carried = int((tail & expanded.isna()).sum())
        expanded.loc[tail] = expanded.loc[tail].fillna(sparse.loc[last_observation])
        if error_after_months is not None and carried > error_after_months:
            raise ValueError(
                f"{series.name}: last semester {last_observation.date()} would be "
                f"carried {carried} months, above the {error_after_months}-month limit."
            )
        if carried > warn_after_months:
            warnings.warn(
                f"{series.name}: holding last published semester constant for {carried} months "
                "beyond its six-month validity window.",
                stacklevel=2,
            )

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
    tax_carry_warn_months: int = DEFAULT_TAX_CARRY_WARN_MONTHS,
    tax_carry_error_months: int = DEFAULT_TAX_CARRY_ERROR_MONTHS,
) -> tuple[pd.Series, dict[str, object]]:
    if not np.isfinite(price_unit_multiplier) or price_unit_multiplier <= 0:
        raise ValueError("price_unit_multiplier must be strictly positive.")

    hicp = _monthly_last(hicp).rename("hicp")
    wtax_sparse = (price_unit_multiplier * _monthly_last(wtax.dropna())).rename("wtax")
    vat_sparse = _monthly_last(vat.dropna()).rename("vat")
    exc_sparse = (price_unit_multiplier * _monthly_last(exc.dropna())).rename("exc")

    scale = _ols_scale_no_intercept(
        hicp.reindex(wtax_sparse.index),
        wtax_sparse,
        label=f"{name}: HICP -> Eurostat after-tax price ({output_price_unit})",
    )

    start = min(
        stamp
        for stamp in (
            hicp.first_valid_index(),
            vat_sparse.first_valid_index(),
            exc_sparse.first_valid_index(),
        )
        if stamp is not None
    )
    end = hicp.last_valid_index()
    if end is None:
        raise ValueError(f"{name}: HICP target is entirely missing.")
    index = pd.date_range(start, end, freq="MS", name="date")

    hicp_m = hicp.reindex(index)
    vat_m, vat_carried = _expand_semesters(
        vat_sparse,
        index,
        warn_after_months=tax_carry_warn_months,
        error_after_months=tax_carry_error_months,
    )
    exc_m, exc_carried = _expand_semesters(
        exc_sparse,
        index,
        warn_after_months=tax_carry_warn_months,
        error_after_months=tax_carry_error_months,
    )
    after_tax_implied = scale * hicp_m
    pre_tax = (after_tax_implied / (1.0 + vat_m / 100.0) - exc_m).rename(name)

    if (pre_tax.dropna() <= 0).any():
        bad = pre_tax[pre_tax <= 0].head().to_dict()
        raise ValueError(f"{name}: non-positive constructed pre-tax prices: {bad}")

    fit = pd.concat(
        [wtax_sparse, (scale * hicp).reindex(wtax_sparse.index).rename("fitted")], axis=1
    ).dropna()
    rmse = float(np.sqrt(np.mean((fit["wtax"] - fit["fitted"]) ** 2)))
    unit_tag = output_price_unit.replace("/", "_").replace(" ", "_")

    info = {
        "formula": f"pre_tax_{unit_tag} = gamma_{unit_tag} * HICP / (1 + VAT/100) - EXC_{unit_tag}",
        "source_price_unit": source_price_unit,
        "output_price_unit": output_price_unit,
        "price_unit_multiplier": float(price_unit_multiplier),
        "gamma": scale,
        "gamma_unit": f"{output_price_unit} per HICP index point",
        "ols_observations": int(len(fit)),
        "after_tax_fit_rmse": rmse,
        "after_tax_fit_rmse_unit": output_price_unit,
        "tax_months_carried_forward": {"vat": vat_carried, "excise": exc_carried},
        "first_valid": pre_tax.first_valid_index().date().isoformat() if pre_tax.first_valid_index() is not None else None,
        "last_valid": pre_tax.last_valid_index().date().isoformat() if pre_tax.last_valid_index() is not None else None,
    }
    return pre_tax, info


# ===========================================================================
# Panel validation and diagnostics
# ===========================================================================


def _validate_model_panel(name: str, panel: pd.DataFrame, spec: dict) -> None:
    if list(panel.columns) != spec["columns"]:
        raise ValueError(
            f"{name}: expected columns {spec['columns']}, found {list(panel.columns)}"
        )
    if panel.empty:
        raise ValueError(f"{name}: output panel is empty.")
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError(f"{name}: dates must be unique and sorted ascending internally.")
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
        raise ValueError(f"{name}: calendar index is not regular.")


def _usable_design_rows(equation_panel: pd.DataFrame, lags: int) -> int:
    transformed = equation_panel.diff()
    blocks = [transformed]
    blocks.extend(transformed.shift(lag) for lag in range(1, lags + 1))
    return int(len(pd.concat(blocks, axis=1).dropna()))


def _diagnostic_rows(model: str, panel: pd.DataFrame, spec: dict) -> list[dict[str, object]]:
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
            trailing_missing = int(len(panel) - 1 - panel.index.get_loc(observed.index.max()))

        rows.append({
            "diagnostic_type": "series",
            "model": model,
            "equation": None,
            "output_file": spec["file"],
            "frequency": spec["frequency"],
            "lags_in_paper": spec["lags"],
            "series": column,
            "observations": int(observed.size),
            "first_date": observed.index.min().date().isoformat() if not observed.empty else None,
            "last_date": observed.index.max().date().isoformat() if not observed.empty else None,
            "missing_values": int(panel[column].isna().sum()),
            "internal_gaps": internal_gaps,
            "leading_missing": leading_missing,
            "trailing_missing": trailing_missing,
            "coverage_percent": round(100.0 * observed.size / len(panel), 2),
            "balanced_level_observations": None,
            "balanced_difference_observations": None,
            "usable_after_difference_and_lags": None,
            "usable_edge_only_upper_bound": None,
            "rows_lost_to_interior_gaps": None,
        })

    for equation, columns in spec["equations"].items():
        equation_panel = panel[columns]
        balanced_levels = equation_panel.dropna()
        balanced_differences = equation_panel.diff().dropna()
        usable = _usable_design_rows(equation_panel, int(spec["lags"]))
        naive_usable = max(0, len(balanced_differences) - int(spec["lags"]))
        any_obs = equation_panel.notna().any(axis=1)
        if any_obs.any():
            first = any_obs[any_obs].index[0]
            last = any_obs[any_obs].index[-1]
            internal = int(equation_panel.isna().any(axis=1).loc[first:last].sum())
        else:
            internal = 0

        rows.append({
            "diagnostic_type": "equation_sample",
            "model": model,
            "equation": equation,
            "output_file": spec["file"],
            "frequency": spec["frequency"],
            "lags_in_paper": spec["lags"],
            "series": " | ".join(columns),
            "observations": int(len(balanced_levels)),
            "first_date": balanced_levels.index.min().date().isoformat() if not balanced_levels.empty else None,
            "last_date": balanced_levels.index.max().date().isoformat() if not balanced_levels.empty else None,
            "missing_values": int(len(panel) - len(balanced_levels)),
            "internal_gaps": internal,
            "leading_missing": None,
            "trailing_missing": None,
            "coverage_percent": round(100.0 * len(balanced_levels) / len(panel), 2),
            "balanced_level_observations": int(len(balanced_levels)),
            "balanced_difference_observations": int(len(balanced_differences)),
            "usable_after_difference_and_lags": int(usable),
            "usable_edge_only_upper_bound": int(naive_usable),
            "rows_lost_to_interior_gaps": int(naive_usable - usable),
        })
    return rows


# ===========================================================================
# Main build
# ===========================================================================


def build_model_datasets(
    *,
    raw_path: str | Path | None = None,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    build_vintage: str | None = None,
    weekly_commodity_shift_weeks: int = DEFAULT_WEEKLY_COMMODITY_SHIFT_WEEKS,
    overwrite: bool = False,
    max_tax_carry_months: int = DEFAULT_TAX_CARRY_ERROR_MONTHS,
) -> dict[str, Path]:
    """Read one Excel snapshot and write the six model-specific datasets."""
    if weekly_commodity_shift_weeks not in (0, 1):
        raise ValueError("weekly_commodity_shift_weeks must be 0 or 1.")
    if max_tax_carry_months < 0:
        raise ValueError("max_tax_carry_months must be non-negative.")

    raw = Path(raw_path) if raw_path is not None else find_raw_workbook(raw_root)
    if not raw.is_file():
        raise FileNotFoundError(raw)
    if raw.suffix.lower() != ".xlsx":
        raise ValueError(f"The single raw input must be an .xlsx workbook: {raw}")

    snapshot_vintage = _parse_vintage_name(raw.parent.name)
    build_vintage = build_vintage or (
        snapshot_vintage.strftime("%Y%m%d") if snapshot_vintage else datetime.now().strftime("%Y%m%d")
    )
    output_dir = Path(output_root) / build_vintage
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains a completed build. Pass --overwrite "
            "or choose another --build-vintage."
        )

    frames, read_stats, sheets, metadata_records = _read_workbook_sources(raw)
    wob = frames["european_commission"]
    world_bank = frames["world_bank"]
    bloomberg = frames["bloomberg"]
    haver = frames["haver"]
    eurostat = frames["eurostat"]

    # ------------------------------------------------------------------
    # Bloomberg: daily inputs -> EUR units -> weekly/monthly aggregation
    # ------------------------------------------------------------------
    bbg = _commodity_prices_in_euro(bloomberg)

    def _weekly_commodity(series: pd.Series) -> pd.Series:
        weekly = _weekly_mean(series)
        if weekly_commodity_shift_weeks:
            weekly.index = weekly.index + pd.DateOffset(weeks=weekly_commodity_shift_weeks)
            weekly.index.name = "date"
        return weekly

    crude_weekly = _weekly_commodity(bbg["crude_oil_eur"])
    refined_petroleum_weekly = _weekly_commodity(bbg["refined_petroleum_eur"])
    refined_diesel_weekly = _weekly_commodity(bbg["refined_diesel_eur"])

    coverage_raw_series: dict[str, tuple[pd.Series, str]] = {
        "crude_oil_eur": (bbg["crude_oil_eur"], "weekly"),
        "refined_petroleum_eur": (bbg["refined_petroleum_eur"], "weekly"),
        "refined_diesel_eur": (bbg["refined_diesel_eur"], "weekly"),
        "ttf_eur_mwh": (bbg["ttf_eur_mwh"], "monthly"),
        "eurusd": (bbg["eurusd"], "monthly"),
    }

    aggregation_coverage = [
        _aggregation_coverage(
            bbg[logical],
            frequency="weekly",
            label=logical,
            method="Monday-Sunday mean of daily Bloomberg snapshot values",
            label_shift_weeks=int(weekly_commodity_shift_weeks),
        )
        for logical in ("crude_oil_eur", "refined_petroleum_eur", "refined_diesel_eur")
    ]
    aggregation_coverage.extend(
        _aggregation_coverage(
            bbg[logical],
            frequency="monthly",
            label=logical,
            method="calendar-month mean of daily Bloomberg snapshot values",
        )
        for logical in ("ttf_eur_mwh", "eurusd")
    )

    # ------------------------------------------------------------------
    # WOB weekly consumer prices, already supplied as euro-area aggregates
    # ------------------------------------------------------------------
    wob_petrol = _weekly_last(
        _required_series(
            wob, "wob_petroleum_cpr_ntax",
            source="european_commission", logical_name="wob_petrol_pre_tax",
        )
    )
    wob_diesel = _weekly_last(
        _required_series(
            wob, "wob_diesel_cpr_ntax",
            source="european_commission", logical_name="wob_diesel_pre_tax",
        )
    )
    wob_heating = _weekly_last(
        _required_series(
            wob, "wob_gas_cpr_ntax",
            source="european_commission", logical_name="wob_heating_oil_pre_tax",
        )
    )

    _warn_on_partial_periods(aggregation_coverage)

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

    # ------------------------------------------------------------------
    # Monthly gas / electricity / heat / solid fuels
    # ------------------------------------------------------------------
    natural_gas, gas_backcast_info = _build_natural_gas_series(bbg, world_bank)

    hicp_gas = _required_series(haver, "hicp_gas", source="haver", logical_name="hicp_gas")
    hicp_electricity = _required_series(
        haver, "hicp_electricity", source="haver", logical_name="hicp_electricity"
    )

    gas_pre_tax, gas_pre_tax_info = _build_pre_tax_price(
        hicp=hicp_gas,
        wtax=_required_series(eurostat, "estat_gas_household_cpr_wtax", source="eurostat", logical_name="gas_wtax"),
        vat=_required_series(eurostat, "estat_gas_household_vat", source="eurostat", logical_name="gas_vat"),
        exc=_required_series(eurostat, "estat_gas_household_exc", source="eurostat", logical_name="gas_exc"),
        name="gas_pre_tax",
        price_unit_multiplier=PRE_TAX_EUR_KWH_TO_EUR_MWH,
        source_price_unit="EUR/kWh",
        output_price_unit="EUR/MWh",
        tax_carry_error_months=max_tax_carry_months,
    )
    electricity_pre_tax, electricity_pre_tax_info = _build_pre_tax_price(
        hicp=hicp_electricity,
        wtax=_required_series(eurostat, "estat_electricity_household_cpr_wtax", source="eurostat", logical_name="electricity_wtax"),
        vat=_required_series(eurostat, "estat_electricity_household_vat", source="eurostat", logical_name="electricity_vat"),
        exc=_required_series(eurostat, "estat_electricity_household_exc", source="eurostat", logical_name="electricity_exc"),
        name="electricity_pre_tax",
        price_unit_multiplier=1.0,
        source_price_unit="EUR/kWh",
        output_price_unit="EUR/kWh",
        tax_carry_error_months=max_tax_carry_months,
    )

    hicp_heat = _monthly_last(
        _required_series(haver, "hicp_heat_energy", source="haver", logical_name="hicp_heat_energy")
    )
    hicp_solid = _monthly_last(
        _required_series(haver, "hicp_solid_fuels", source="haver", logical_name="hicp_solid_fuels")
    )
    ppi_energy = _monthly_last(
        _required_series(haver, "ppi_energy", source="haver", logical_name="ppi_energy")
    )

    gas = _regular_panel(
        {"gas_pre_tax": gas_pre_tax, "natural_gas_eur_mwh": natural_gas},
        frequency="monthly",
        target_columns=MODEL_SPECS["gas"]["target_columns"],
    )
    electricity = _regular_panel(
        {"electricity_pre_tax": electricity_pre_tax, "natural_gas_eur_mwh": natural_gas},
        frequency="monthly",
        target_columns=MODEL_SPECS["electricity"]["target_columns"],
    )
    heat_energy = _regular_panel(
        {"hicp_heat_energy": hicp_heat, "natural_gas_eur_mwh": natural_gas, "ppi_energy": ppi_energy},
        frequency="monthly",
        target_columns=MODEL_SPECS["heat_energy"]["target_columns"],
    )
    solid_fuels = _regular_panel(
        {"hicp_solid_fuels": hicp_solid, "natural_gas_eur_mwh": natural_gas, "ppi_energy": ppi_energy},
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

    aggregation_coverage = _bind_coverage_to_panels(aggregation_coverage, panels)

    partial_period_inputs: list[dict[str, object]] = []
    for row in aggregation_coverage:
        if row["final_period_complete"]:
            continue
        raw_series, freq = coverage_raw_series[str(row["series"])]
        panel_column = row["panel_column"]
        partial_period_inputs.extend(
            _final_period_input_records(
                raw_series,
                frequency=freq,
                panel_label_date=str(row["panel_label_date"]),
                panel_column=None if pd.isna(panel_column) else str(panel_column),
                series_label=str(row["series"]),
            )
        )

    # ------------------------------------------------------------------
    # Write processed outputs. Keep model files chronologically ascending;
    # the model layer expects and validates this order.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    diagnostic_rows: list[dict[str, object]] = []

    for name, panel in panels.items():
        path = output_dir / MODEL_SPECS[name]["file"]
        panel.to_csv(path, date_format="%Y-%m-%d")
        outputs[name] = path
        diagnostic_rows.extend(_diagnostic_rows(name, panel, MODEL_SPECS[name]))

    diagnostics_path = output_dir / "model_datasets_diagnostics.csv"
    sources_path = output_dir / "source_vintages.csv"  # kept for compatibility
    coverage_path = output_dir / "aggregation_coverage.csv"
    partial_inputs_path = output_dir / "partial_period_inputs.csv"

    pd.DataFrame(diagnostic_rows).to_csv(diagnostics_path, index=False)
    pd.DataFrame(aggregation_coverage).to_csv(coverage_path, index=False)
    partial_inputs_frame = pd.DataFrame(
        partial_period_inputs,
        columns=["series", "panel_column", "panel_label_date", "frequency", "input_date", "input_value"],
    )
    partial_inputs_frame.to_csv(partial_inputs_path, index=False)

    workbook_hash = _sha256(raw)
    snapshot_label = raw.parent.name if snapshot_vintage else "unversioned_workbook"
    source_table = pd.DataFrame([
        {
            "source": source,
            "sheet": sheets[source],
            "snapshot_vintage": snapshot_label,
            "raw_workbook": str(raw.resolve()),
            "raw_workbook_sha256": workbook_hash,
            "input_rows": read_stats[source]["retained_rows"],
            "input_columns": read_stats[source]["columns"],
            "first_observation": read_stats[source]["first_date"],
            "last_observation": read_stats[source]["last_date"],
            "input_order_in_excel": read_stats[source]["input_order"],
            "invalid_date_rows": read_stats[source]["invalid_date_rows"],
            "duplicate_date_rows": read_stats[source]["duplicate_date_rows"],
            "all_missing_rows_dropped": read_stats[source]["all_missing_rows_dropped"],
            "error_markers_coerced_to_nan": read_stats[source]["error_markers_coerced_to_nan"],
        }
        for source in ("haver", "bloomberg", "european_commission", "world_bank", "eurostat")
    ])
    source_table.to_csv(sources_path, index=False)

    manifest = {
        "project": "ECB energy STIP six-model dataset build",
        "script_version": SCRIPT_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "build_vintage": build_vintage,
        "input_architecture": "single_excel_workbook",
        "raw_workbook": str(raw.resolve()),
        "raw_workbook_sha256": workbook_hash,
        "raw_workbook_bytes": raw.stat().st_size,
        "snapshot_vintage": snapshot_label,
        "sheet_mapping": sheets,
        "ignored_live_sheet": "Bloomberg_Live",
        "workbook_requirement": (
            "Bloomberg is the values-only snapshot consumed by Python. "
            "Bloomberg_Live may contain formulas and is intentionally ignored."
        ),
        "source_read_statistics": read_stats,
        "metadata_sheet_records": metadata_records,
        "bloomberg_mapping": BLOOMBERG_TICKERS,
        "bloomberg_units": BLOOMBERG_UNITS,
        "refined_petroleum_policy": (
            "GNEBM1 PVMO is converted only from USD/metric tonne to EUR/metric "
            "tonne using EURUSD. No tonne-to-barrel factor is applied."
        ),
        "known_approximations_vs_paper": [
            "World Bank Natural gas, Europe substitutes for the paid Energy Intelligence European border-gas series; the pre-TTF history is chain-linked to TTF.",
            "GNEBM1 PVMO (Eurobob oxygenated gasoline FOB Rotterdam barges) is used for refined petroleum; its EUR/metric-tonne unit is retained rather than inventing a gasoline tonne-to-barrel conversion.",
            "QS1 is a Low Sulphur Gas Oil future rather than the paper's Refinitiv spot assessment; 7.44 barrels/tonne is applied only to this gasoil/diesel series.",
            "The available WOB workbook starts in 2005, later than the paper's full historical sample.",
            "Eurostat household bands are the gas GJ20-199/D2 and electricity KWH2500-4999/DC contracts selected in the workbook Power Query.",
        ],
        "construction": {
            "excel_date_order": (
                "Source sheets may be newest-first. Every source is sorted "
                "ascending in memory before differencing, lags or aggregation."
            ),
            "weekly_calendar": {
                "week_definition": "Monday-Sunday, labelled by Monday",
                "commodity_shift_weeks": int(weekly_commodity_shift_weeks),
                "alignment": (
                    "shift=1 aligns WOB Monday t with the preceding completed "
                    "Monday-Sunday commodity week; shift=0 uses the same week."
                ),
            },
            "commodity_units": {
                "crude_oil_eur": "EUR/barrel",
                "refined_petroleum_eur": "EUR/metric tonne",
                "refined_diesel_eur": "EUR/barrel",
            },
            "pre_tax": {
                "gas": gas_pre_tax_info,
                "electricity": electricity_pre_tax_info,
            },
            "natural_gas": gas_backcast_info,
            "transformations": (
                "Outputs contain levels. Apply ordinary absolute differences "
                "in the BVAR model layer."
            ),
            "ragged_edge": (
                "Regular calendars are retained with NaN for unreleased cells. "
                "The builder never drops the ragged edge merely because one series is missing."
            ),
            "partial_final_period": (
                "aggregation_coverage.csv records the final weekly/monthly "
                "Bloomberg aggregation coverage; partial_period_inputs.csv "
                "stores the raw daily observations backing incomplete periods."
            ),
            "aggregation_coverage": aggregation_coverage,
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
            "aggregation_coverage": str(coverage_path.resolve()),
            "partial_period_inputs": str(partial_inputs_path.resolve()),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Hash after every file exists.
    manifest["output_sha256"] = {
        **{name: _sha256(path) for name, path in outputs.items()},
        "diagnostics": _sha256(diagnostics_path),
        "source_vintages": _sha256(sources_path),
        "aggregation_coverage": _sha256(coverage_path),
        "partial_period_inputs": _sha256(partial_inputs_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    outputs.update({
        "diagnostics": diagnostics_path,
        "source_vintages": sources_path,
        "aggregation_coverage": coverage_path,
        "partial_period_inputs": partial_inputs_path,
        "manifest": manifest_path,
    })

    print(f"Builder:       {SCRIPT_VERSION}")
    print(f"Raw workbook:  {raw}")
    print(f"Build vintage: {build_vintage}")
    print("Source sheets:")
    for source in ("haver", "bloomberg", "european_commission", "world_bank", "eurostat"):
        stat = read_stats[source]
        print(
            f"  {source:22s} {sheets[source]:22s} "
            f"{stat['first_date']} -> {stat['last_date']} "
            f"({stat['retained_rows']:,} rows; Excel order {stat['input_order']})"
        )

    partial = [row for row in aggregation_coverage if not row["final_period_complete"]]
    if partial:
        print("\nPartial final Bloomberg aggregation periods:")
        for row in partial:
            print(
                f"  {row['series']:24s} {row['panel_label_date']}  "
                f"{row['final_period_input_observations']}/"
                f"{row['expected_input_observations']} inputs "
                f"({row['coverage_ratio']:.0%})"
            )

    print("\nProcessed model datasets:")
    for name, panel in panels.items():
        spec = MODEL_SPECS[name]
        print(
            f"  {spec['file']:30s} {panel.index.min().date()} -> "
            f"{panel.index.max().date()} | {len(panel):,} rows | {panel.shape[1]} series"
        )
    print(f"\nOutput directory: {output_dir}")
    return outputs


# Backward-compatible call used elsewhere in the project.
def build_dataset(**kwargs) -> dict[str, Path]:
    return build_model_datasets(**kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Path to the single raw_energy_bvar.xlsx workbook. Default: auto-detect below --raw-root.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Folder searched for raw_energy_bvar*.xlsx (default: {DEFAULT_RAW_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Processed output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--build-vintage",
        default=None,
        help="Output folder name. Default: raw parent vintage if dated, otherwise today YYYYMMDD.",
    )
    parser.add_argument(
        "--weekly-commodity-shift-weeks",
        type=int,
        choices=(0, 1),
        default=DEFAULT_WEEKLY_COMMODITY_SHIFT_WEEKS,
        help="1 (default) aligns WOB Monday t with the preceding completed commodity week.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-tax-carry-months",
        type=int,
        default=DEFAULT_TAX_CARRY_ERROR_MONTHS,
        help="Maximum months taxes may be carried beyond the final six-month Eurostat semester.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    build_model_datasets(
        raw_path=args.raw,
        raw_root=args.raw_root,
        output_root=args.output_root,
        build_vintage=args.build_vintage,
        weekly_commodity_shift_weeks=args.weekly_commodity_shift_weeks,
        overwrite=args.overwrite,
        max_tax_carry_months=args.max_tax_carry_months,
    )


if __name__ == "__main__":
    main()
