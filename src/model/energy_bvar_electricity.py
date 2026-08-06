"""Electricity-specific adapter for the generic energy BVAR engine.

The monthly electricity model contains absolute changes in wholesale natural
 gas and pre-tax consumer electricity prices, plus eleven calendar-month
 seasonal dummies. January is the omitted reference month by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from energy_bvar_model import (
    _read_dated_csv,
    load_energy_panel,
    monthly_seasonal_dummies,
)


ELECTRICITY_COLUMN_ALIASES = {
    "natural_gas_eur_mwh": "natural_gas_wholesale",
    "natural_gas": "natural_gas_wholesale",
    "electricity_pre_tax_price": "electricity_pre_tax",
}


def load_electricity_panel(
    path: str | Path,
    variables: Sequence[str] = (
        "natural_gas_wholesale",
        "electricity_pre_tax",
    ),
) -> pd.DataFrame:
    """Load the processed monthly electricity panel without filling missing data."""
    return load_energy_panel(
        path,
        variables=variables,
        aliases=ELECTRICITY_COLUMN_ALIASES,
        frequency="monthly",
    )


def make_electricity_seasonal_dummies(
    index: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    reference_month: int = 1,
) -> pd.DataFrame:
    """Create the eleven seasonal dummies used by the electricity BVAR."""
    return monthly_seasonal_dummies(
        index,
        reference_month=reference_month,
        prefix="month",
    )


# -----------------------------------------------------------------------------
# Tax re-attribution and HICP reconstruction
# -----------------------------------------------------------------------------


def _find_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"Could not find {label}; tried {list(candidates)}.")


def load_electricity_tax_context(
    electricity_dataset_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict:
    """Recover gamma, VAT, excise and HICP electricity for the same vintage."""
    electricity_dataset_path = Path(electricity_dataset_path)
    manifest_path = (
        electricity_dataset_path.parent / "manifest.json"
        if manifest_path is None
        else Path(manifest_path)
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pre_tax_info = manifest["construction"]["pre_tax"]["electricity"]
    gamma = float(pre_tax_info["gamma"])
    price_unit_multiplier = float(pre_tax_info.get("price_unit_multiplier", 1.0))
    output_price_unit = str(pre_tax_info.get("output_price_unit", "EUR/kWh"))
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Invalid electricity gamma in the processed manifest.")
    if not np.isfinite(price_unit_multiplier) or price_unit_multiplier <= 0:
        raise ValueError("Invalid electricity price-unit multiplier in the manifest.")

    source_files = manifest.get("source_files", {})
    haver_path = Path(source_files.get("haver", ""))
    eurostat_path = Path(source_files.get("eurostat", ""))
    if not haver_path.exists() or not eurostat_path.exists():
        raise FileNotFoundError(
            "The manifest source paths are unavailable. Keep the source vintage "
            "on the same machine or load the tax series manually."
        )

    haver = _read_dated_csv(haver_path)
    eurostat = _read_dated_csv(eurostat_path)
    hicp_col = _find_column(haver, ["hicp_electricity"], "HICP electricity")
    vat_col = _find_column(
        eurostat,
        ["estat_electricity_household_vat"],
        "electricity VAT",
    )
    exc_col = _find_column(
        eurostat,
        ["estat_electricity_household_exc"],
        "electricity excise",
    )

    return {
        "gamma": gamma,
        "hicp_electricity": haver[hicp_col].astype(float).rename("hicp_electricity"),
        "vat_percent": eurostat[vat_col].astype(float).rename("vat_percent"),
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
    """Assign each semiannual value to six months and extend only the final edge."""
    sparse = sparse.dropna().copy().sort_index()
    sparse.index = sparse.index.to_period("M").to_timestamp(how="start")
    out = pd.Series(np.nan, index=monthly_index, dtype=float, name=sparse.name)
    for date, value in sparse.items():
        valid = pd.date_range(date, periods=6, freq="MS")
        out.loc[out.index.intersection(valid)] = float(value)
    if carry_forward_edge and len(sparse):
        last_covered_month = sparse.index[-1] + pd.DateOffset(months=5)
        out.loc[out.index > last_covered_month] = float(sparse.iloc[-1])
    return out


def construct_pre_tax_electricity(
    hicp_index: pd.Series,
    vat_percent: pd.Series,
    excise: pd.Series,
    gamma: float,
) -> pd.Series:
    """Construct pre-tax electricity in the unit used to estimate gamma."""
    index = pd.DatetimeIndex(hicp_index.index)
    vat = expand_semester_series(vat_percent, index)
    exc = expand_semester_series(excise, index)
    return (
        gamma * hicp_index / (1.0 + vat / 100.0) - exc
    ).rename("electricity_pre_tax")


def reattribute_electricity_taxes(
    pre_tax: pd.Series | np.ndarray,
    gamma: float,
    vat_percent: pd.Series | np.ndarray | float,
    excise: pd.Series | np.ndarray | float,
):
    """Invert the pre-tax construction exactly."""
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    return (
        (np.asarray(pre_tax) + np.asarray(excise))
        * (1.0 + np.asarray(vat_percent) / 100.0)
        / gamma
    )


def validate_electricity_tax_round_trip(
    context: Mapping,
    tolerance: float = 1e-10,
) -> pd.Series:
    hicp = context["hicp_electricity"].dropna()
    common_index = pd.date_range(hicp.index.min(), hicp.index.max(), freq="MS")
    hicp = hicp.reindex(common_index)
    vat = expand_semester_series(context["vat_percent"], common_index)
    exc = expand_semester_series(context["excise"], common_index)
    pre_tax = construct_pre_tax_electricity(
        hicp,
        context["vat_percent"],
        context["excise"],
        context["gamma"],
    )
    rebuilt = pd.Series(
        reattribute_electricity_taxes(
            pre_tax,
            context["gamma"],
            vat,
            exc,
        ),
        index=common_index,
    )
    error = (rebuilt - hicp).abs().dropna()
    maximum = float(error.max()) if len(error) else np.nan
    if np.isfinite(maximum) and maximum > tolerance:
        raise AssertionError(
            f"Electricity tax round-trip error {maximum:.3e} exceeds "
            f"{tolerance:.3e}."
        )
    return pd.Series(
        {
            "observations": len(error),
            "maximum_absolute_error": maximum,
            "tolerance": tolerance,
        }
    )


def forecast_to_hicp_electricity(
    forecast: Mapping,
    result: Mapping,
    tax_context: Mapping,
    target_variable: str = "electricity_pre_tax",
) -> dict:
    """Map pre-tax electricity forecast paths back to HICP levels and YoY rates."""
    variables = list(forecast["variables"])
    if target_variable not in variables:
        raise ValueError(f"{target_variable!r} is not in the forecast variables.")
    target_index = variables.index(target_variable)
    dates = pd.DatetimeIndex(forecast["path_dates"])
    vat = expand_semester_series(tax_context["vat_percent"], dates)
    exc = expand_semester_series(tax_context["excise"], dates)
    hicp_paths = reattribute_electricity_taxes(
        np.asarray(forecast["level_paths"])[:, :, target_index],
        tax_context["gamma"],
        vat.to_numpy()[None, :],
        exc.to_numpy()[None, :],
    )

    actual_hicp = tax_context["hicp_electricity"].copy().sort_index()
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
        "price_unit": tax_context.get("price_unit", "EUR/kWh"),
        "vat_percent_path": vat,
        "excise_path": exc,
    }


def electricity_series_construction_table(
    electricity_dataset_path: str | Path,
) -> pd.DataFrame:
    """Summarise the processed electricity-model inputs and their units."""
    electricity_dataset_path = Path(electricity_dataset_path)
    manifest_path = electricity_dataset_path.parent / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}

    construction = manifest.get("construction", {})
    natural = construction.get("natural_gas", {})
    pre_tax = construction.get("pre_tax", {}).get("electricity", {})

    natural_method = natural.get(
        "method",
        "Bloomberg TTF monthly mean in EUR/MWh; earlier observations are "
        "chain-linked to the World Bank European natural-gas proxy.",
    )
    pre_tax_formula = pre_tax.get(
        "formula",
        "pre_tax = gamma * HICP_electricity / (1 + VAT/100) - EXC",
    )
    pre_tax_unit = str(pre_tax.get("output_price_unit", "EUR/kWh"))
    pre_tax_note = pre_tax.get(
        "tax_edge_assumption",
        "Semiannual VAT and excise observations are held for six months and "
        "the final published values are carried through the edge.",
    )
    if pre_tax.get("gamma") is not None:
        pre_tax_note += f" Estimated gamma: {float(pre_tax['gamma']):.6g}."

    return pd.DataFrame(
        {
            "series_type": ["chain-linked wholesale price", "constructed pre-tax price"],
            "model_unit": ["EUR/MWh", pre_tax_unit],
            "source_inputs": [
                "Bloomberg TTF; World Bank Natural gas, Europe; EUR/USD",
                "Haver HICP electricity; Eurostat after-tax price, VAT and excise",
            ],
            "construction": [natural_method, pre_tax_formula],
            "note": [
                "Wholesale natural gas is the upstream explanatory variable.",
                pre_tax_note,
            ],
        },
        index=["natural_gas_wholesale", "electricity_pre_tax"],
    )


__all__ = [
    "load_electricity_panel",
    "make_electricity_seasonal_dummies",
    "load_electricity_tax_context",
    "expand_semester_series",
    "construct_pre_tax_electricity",
    "reattribute_electricity_taxes",
    "validate_electricity_tax_round_trip",
    "forecast_to_hicp_electricity",
    "electricity_series_construction_table",
]
