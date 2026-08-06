"""Gas-specific adapter for the generic energy BVAR engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from energy_bvar_model import _read_dated_csv, load_energy_panel

GAS_COLUMN_ALIASES = {
    "natural_gas_eur_mwh": "natural_gas_wholesale",
    "natural_gas": "natural_gas_wholesale",
    "gas_pre_tax_price": "gas_pre_tax",
}


def load_gas_panel(
    path: str | Path,
    variables: Sequence[str] = ("natural_gas_wholesale", "gas_pre_tax"),
) -> pd.DataFrame:
    """Load the processed monthly gas panel without filling missing values."""
    return load_energy_panel(
        path,
        variables=variables,
        aliases=GAS_COLUMN_ALIASES,
        frequency="monthly",
    )

# -----------------------------------------------------------------------------
# Tax re-attribution and HICP reconstruction
# -----------------------------------------------------------------------------


def _find_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"Could not find {label}; tried {list(candidates)}.")


def load_gas_tax_context(
    gas_dataset_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict:
    """Recover gamma, VAT, unit-consistent excise and HICP gas.

    The processed-vintage manifest records the multiplier used to convert
    Eurostat EUR/kWh prices into the model's EUR/MWh unit. The same multiplier
    is applied to the raw excise series here before tax re-attribution.
    """
    gas_dataset_path = Path(gas_dataset_path)
    manifest_path = gas_dataset_path.parent / "manifest.json" if manifest_path is None else Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pre_tax_info = manifest["construction"]["pre_tax"]["gas"]
    gamma = float(pre_tax_info["gamma"])
    price_unit_multiplier = float(pre_tax_info.get("price_unit_multiplier", 1.0))
    output_price_unit = str(pre_tax_info.get("output_price_unit", "source unit"))
    if not np.isfinite(price_unit_multiplier) or price_unit_multiplier <= 0:
        raise ValueError("Invalid pre-tax price unit multiplier in manifest.")
    source_files = manifest.get("source_files", {})
    haver_path = Path(source_files.get("haver", ""))
    eurostat_path = Path(source_files.get("eurostat", ""))
    if not haver_path.exists() or not eurostat_path.exists():
        raise FileNotFoundError(
            "The manifest source paths are unavailable. Supply the original vintage on the same machine "
            "or load the tax series manually."
        )
    haver = _read_dated_csv(haver_path)
    eurostat = _read_dated_csv(eurostat_path)
    hicp_col = _find_column(haver, ["hicp_gas"], "HICP gas")
    vat_col = _find_column(eurostat, ["estat_gas_household_vat"], "gas VAT")
    exc_col = _find_column(eurostat, ["estat_gas_household_exc"], "gas excise")
    return {
        "gamma": gamma,
        "hicp_gas": haver[hicp_col].astype(float).rename("hicp_gas"),
        "vat_percent": eurostat[vat_col].astype(float).rename("vat_percent"),
        # Eurostat excise is stored in the source unit (normally EUR/kWh).
        # Apply the same multiplier used when the processed pre-tax series and
        # gamma were built so that the exact tax round trip remains valid.
        "excise": (
            price_unit_multiplier * eurostat[exc_col].astype(float)
        ).rename("excise"),
        "price_unit_multiplier": price_unit_multiplier,
        "price_unit": output_price_unit,
        "manifest_path": manifest_path,
        "haver_path": haver_path,
        "eurostat_path": eurostat_path,
    }


def expand_semester_series(
    sparse: pd.Series,
    monthly_index: pd.DatetimeIndex,
    carry_forward_edge: bool = True,
) -> pd.Series:
    """Assign each semiannual observation to exactly six months, then extend only the edge."""
    sparse = sparse.dropna().copy().sort_index()
    sparse.index = sparse.index.to_period("M").to_timestamp(how="start")
    out = pd.Series(np.nan, index=monthly_index, dtype=float, name=sparse.name)
    for date, value in sparse.items():
        valid = pd.date_range(date, periods=6, freq="MS")
        out.loc[out.index.intersection(valid)] = float(value)
    if carry_forward_edge and len(sparse):
        last_date = sparse.index[-1] + pd.DateOffset(months=5)
        out.loc[out.index > last_date] = float(sparse.iloc[-1])
    return out


def construct_pre_tax_price(
    hicp_index: pd.Series,
    vat_percent: pd.Series,
    excise: pd.Series,
    gamma: float,
) -> pd.Series:
    """Construct the pre-tax price with gamma and excise in the same unit."""
    index = hicp_index.index
    vat = expand_semester_series(vat_percent, index)
    exc = expand_semester_series(excise, index)
    return (gamma * hicp_index / (1.0 + vat / 100.0) - exc).rename("gas_pre_tax")


def reattribute_gas_taxes(
    pre_tax: pd.Series | np.ndarray,
    gamma: float,
    vat_percent: pd.Series | np.ndarray | float,
    excise: pd.Series | np.ndarray | float,
):
    """Invert the pre-tax construction exactly: HICP=(pre-tax+EXC)(1+VAT)/gamma."""
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    return (np.asarray(pre_tax) + np.asarray(excise)) * (1.0 + np.asarray(vat_percent) / 100.0) / gamma


def validate_tax_round_trip(context: Mapping, tolerance: float = 1e-10) -> pd.Series:
    hicp = context["hicp_gas"].dropna()
    common_index = pd.date_range(hicp.index.min(), hicp.index.max(), freq="MS")
    hicp = hicp.reindex(common_index)
    vat = expand_semester_series(context["vat_percent"], common_index)
    exc = expand_semester_series(context["excise"], common_index)
    pre_tax = construct_pre_tax_price(hicp, context["vat_percent"], context["excise"], context["gamma"])
    rebuilt = pd.Series(
        reattribute_gas_taxes(pre_tax, context["gamma"], vat, exc),
        index=common_index,
    )
    error = (rebuilt - hicp).abs().dropna()
    maximum = float(error.max()) if len(error) else np.nan
    if np.isfinite(maximum) and maximum > tolerance:
        raise AssertionError(f"Tax round-trip error {maximum:.3e} exceeds {tolerance:.3e}.")
    return pd.Series({"observations": len(error), "maximum_absolute_error": maximum, "tolerance": tolerance})


def forecast_to_hicp_gas(
    forecast: Mapping,
    result: Mapping,
    tax_context: Mapping,
    target_variable: str = "gas_pre_tax",
) -> dict:
    variables = list(forecast["variables"])
    if target_variable not in variables:
        raise ValueError(f"{target_variable!r} is not in the forecast variables.")
    target_index = variables.index(target_variable)
    dates = pd.DatetimeIndex(forecast["path_dates"])
    vat = expand_semester_series(tax_context["vat_percent"], dates)
    exc = expand_semester_series(tax_context["excise"], dates)
    hicp_paths = reattribute_gas_taxes(
        np.asarray(forecast["level_paths"])[:, :, target_index],
        tax_context["gamma"],
        vat.to_numpy()[None, :],
        exc.to_numpy()[None, :],
    )

    actual_hicp = tax_context["hicp_gas"].copy().sort_index()
    history = actual_hicp.loc[actual_hicp.index <= result["prep"]["balanced_end"]]
    full_index = history.index.append(dates[~dates.isin(history.index)])
    full_index = pd.DatetimeIndex(full_index).sort_values().unique()
    yoy = np.full((len(hicp_paths), len(dates)), np.nan)
    for draw in range(len(hicp_paths)):
        series = history.reindex(full_index)
        series.loc[dates] = hicp_paths[draw]
        inflation = 100.0 * (series / series.shift(12) - 1.0)
        yoy[draw] = inflation.reindex(dates).to_numpy()

    return {
        "hicp_level_paths": hicp_paths,
        "hicp_yoy_paths": yoy,
        "path_dates": dates,
        "future_dates": forecast["future_dates"],
        "tail_length": forecast["tail_length"],
        "future_hicp_level_paths": hicp_paths[:, forecast["tail_length"] :],
        "future_hicp_yoy_paths": yoy[:, forecast["tail_length"] :],
        "actual_hicp": actual_hicp,
        "gamma": tax_context["gamma"],
        "price_unit": tax_context.get("price_unit", "source unit"),
        "vat_percent_path": vat,
        "excise_path": exc,
    }

def gas_series_construction_table(gas_dataset_path: str | Path) -> pd.DataFrame:
    """Describe how the two processed gas-model level series were built.

    The table uses the processed-vintage manifest when it is available and
    falls back to the documented pipeline conventions otherwise. It separates
    source construction from the absolute-difference transformation applied by
    the estimation notebook.
    """
    gas_dataset_path = Path(gas_dataset_path)
    manifest_path = gas_dataset_path.parent / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}

    construction = manifest.get("construction", {})
    natural = construction.get("natural_gas", {})
    pre_tax = construction.get("pre_tax", {}).get("gas", {})

    natural_method = natural.get(
        "method",
        "Bloomberg TTF monthly mean in EUR/MWh; earlier months follow the "
        "World Bank Natural gas, Europe proxy converted to EUR and chain-linked "
        "to the TTF level at the junction, without an estimated splice window.",
    )
    natural_note = natural.get(
        "proxy_note",
        "The World Bank series is a project proxy for the historical border-gas "
        "series used in the paper.",
    )
    if natural.get("anchor_date"):
        natural_note += f" TTF anchor month: {natural['anchor_date']}."

    pre_tax_formula = pre_tax.get(
        "formula",
        "pre_tax_EUR_MWh = gamma_EUR_MWh * HICP_gas / "
        "(1 + VAT/100) - EXC_EUR_MWh",
    )
    pre_tax_note = pre_tax.get(
        "tax_edge_assumption",
        "Each semiannual VAT and excise observation is used for six months; "
        "after the final published semester the last tax values are held constant.",
    )
    if pre_tax.get("gamma") is not None:
        pre_tax_note += f" Estimated gamma: {float(pre_tax['gamma']):.6g}."
    pre_tax_unit = str(pre_tax.get("output_price_unit", "EUR/MWh"))

    table = pd.DataFrame(
        {
            "series_type": ["chain-linked backcast", "constructed pre-tax price"],
            "model_unit": ["EUR/MWh", pre_tax_unit],
            "source_inputs": [
                "Bloomberg TTF; World Bank Natural gas, Europe; EUR/USD",
                "Haver HICP gas; Eurostat after-tax price, VAT and excise",
            ],
            "level_construction": [natural_method, pre_tax_formula],
            "important_note": [natural_note, pre_tax_note],
            "notebook_transformation": [
                "ordinary absolute first difference: P_t - P_{t-1}",
                "ordinary absolute first difference: P_t - P_{t-1}",
            ],
        },
        index=["natural_gas_wholesale", "gas_pre_tax"],
    )
    table.index.name = "model series"
    return table


__all__ = [
    "load_gas_panel",
    "gas_series_construction_table",
    "load_gas_tax_context",
    "expand_semester_series",
    "construct_pre_tax_price",
    "reattribute_gas_taxes",
    "validate_tax_round_trip",
    "forecast_to_hicp_gas",
]
