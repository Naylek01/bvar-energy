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
    load_energy_panel,
    monthly_seasonal_dummies,
)
from energy_bvar_source_context import load_manifest_source_frame, read_processed_manifest


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
    """Recover gamma, VAT, excise and HICP electricity for the same vintage.

    Supports both the legacy per-source CSV manifests and the v9
    single-workbook manifest without changing notebook calls.
    """
    electricity_dataset_path = Path(electricity_dataset_path)
    manifest, resolved_manifest_path = read_processed_manifest(
        electricity_dataset_path, manifest_path
    )
    pre_tax_info = manifest["construction"]["pre_tax"]["electricity"]
    gamma = float(pre_tax_info["gamma"])
    price_unit_multiplier = float(pre_tax_info.get("price_unit_multiplier", 1.0))
    output_price_unit = str(pre_tax_info.get("output_price_unit", "EUR/kWh"))
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Invalid electricity gamma in the processed manifest.")
    if not np.isfinite(price_unit_multiplier) or price_unit_multiplier <= 0:
        raise ValueError("Invalid electricity price-unit multiplier in the manifest.")

    haver, haver_source = load_manifest_source_frame(
        electricity_dataset_path,
        "haver",
        manifest_path=resolved_manifest_path,
        haver_ticker_map={"H023HW51@EUDATA": "hicp_electricity"},
    )
    eurostat, eurostat_source = load_manifest_source_frame(
        electricity_dataset_path,
        "eurostat",
        manifest_path=resolved_manifest_path,
    )
    hicp_col = _find_column(haver, ["hicp_electricity"], "HICP electricity")
    vat_col = _find_column(
        eurostat, ["estat_electricity_household_vat"], "electricity VAT"
    )
    exc_col = _find_column(
        eurostat, ["estat_electricity_household_exc"], "electricity excise"
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
        "excise_unit": output_price_unit,
        "vat_unit": "percentage points",
        "manifest_path": resolved_manifest_path,
        "haver_path": Path(haver_source["path"]),
        "eurostat_path": Path(eurostat_source["path"]),
        "haver_source_mode": haver_source["mode"],
        "eurostat_source_mode": eurostat_source["mode"],
        "haver_sheet": haver_source.get("sheet"),
        "eurostat_sheet": eurostat_source.get("sheet"),
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


def _normalise_tax_period_start(value, frequency: str) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("tax_scenario['start_date'] is invalid.")
    if frequency == "monthly":
        return date.to_period("M").to_timestamp(how="start")
    raise ValueError(f"Unsupported tax-scenario frequency {frequency!r}.")


def _tax_scenario_values(
    value,
    dates: pd.DatetimeIndex,
    *,
    label: str,
    frequency: str,
) -> np.ndarray:
    if isinstance(value, pd.Series):
        series = value.astype(float).copy()
        series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp(how="start")
        if series.index.has_duplicates:
            raise ValueError(f"{label} scenario has duplicate model periods.")
        values = series.reindex(dates).to_numpy(dtype=float)
        if np.isnan(values).any():
            missing = dates[np.isnan(values)].strftime("%Y-%m-%d").tolist()
            raise ValueError(f"{label} scenario is missing dates {missing}.")
        return values
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        return np.repeat(float(values), len(dates))
    if values.ndim != 1 or len(values) != len(dates):
        raise ValueError(
            f"{label} must be scalar, a dated Series, or have length {len(dates)}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values.")
    return values


# ENERGY_SCENARIO_TAX_CUTS_V1
def _validate_vat_percent(values: np.ndarray) -> None:
    """Validate a VAT *level* expressed in percentage points.

    Any finite value in [0, 100] is legitimate.  In particular, low positive
    rates below 1% are valid scenario outcomes after a tax cut; unit mistakes
    are prevented in the dashboard by showing baseline + delta -> final VAT.
    """
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("VAT must contain only finite percentage-point levels.")
    if np.any((values < 0.0) | (values > 100.0)):
        raise ValueError("VAT level must lie between 0 and 100 percentage points.")

def _apply_monthly_tax_scenario(
    baseline_vat: pd.Series,
    baseline_excise: pd.Series,
    tax_context: Mapping,
    tax_scenario: Mapping | None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict]:
    """Apply a dated tax scenario without altering the pre-scenario path."""
    baseline_vat = baseline_vat.astype(float).copy()
    baseline_excise = baseline_excise.astype(float).copy()
    dates = pd.DatetimeIndex(baseline_vat.index, name="date")
    if not dates.equals(pd.DatetimeIndex(baseline_excise.index, name="date")):
        raise ValueError("VAT and excise baseline paths have different calendars.")

    realised_vat = baseline_vat.copy()
    realised_excise = baseline_excise.copy()
    active = pd.Series(False, index=dates, dtype=bool)
    metadata = {
        "active": False,
        "requested_start_date": None,
        "effective_start_date": None,
        "vat_conditioned": False,
        "excise_conditioned": False,
        "excise_unit": tax_context.get("excise_unit", tax_context.get("price_unit")),
    }

    if tax_scenario is not None:
        scenario = dict(tax_scenario)
        if "start_date" not in scenario:
            raise ValueError("tax_scenario requires an explicit 'start_date'.")
        if "vat_percent" not in scenario and "excise" not in scenario:
            raise ValueError("tax_scenario must condition VAT and/or excise.")

        start = _normalise_tax_period_start(scenario["start_date"], "monthly")
        active_dates = dates[dates >= start]
        if len(active_dates) == 0:
            raise ValueError(
                f"Tax scenario starts at {start.date()}, after the available path."
            )
        active.loc[active_dates] = True

        if "vat_percent" in scenario:
            vat_values = _tax_scenario_values(
                scenario["vat_percent"], active_dates,
                label="vat_percent", frequency="monthly",
            )
            _validate_vat_percent(vat_values)
            realised_vat.loc[active_dates] = vat_values
            metadata["vat_conditioned"] = True

        if "excise" in scenario:
            expected_unit = str(
                tax_context.get("excise_unit", tax_context.get("price_unit", ""))
            )
            supplied_unit = scenario.get("excise_unit")
            if not supplied_unit:
                raise ValueError(
                    "An excise scenario requires 'excise_unit'. Expected unit: "
                    f"{expected_unit!r}."
                )
            normalise = lambda x: "".join(str(x).lower().split())
            if normalise(supplied_unit) != normalise(expected_unit):
                raise ValueError(
                    f"Excise unit mismatch: expected {expected_unit!r}, "
                    f"got {supplied_unit!r}."
                )
            excise_values = _tax_scenario_values(
                scenario["excise"], active_dates,
                label="excise", frequency="monthly",
            )
            realised_excise.loc[active_dates] = excise_values
            metadata["excise_conditioned"] = True

        metadata.update(
            {
                "active": True,
                "requested_start_date": str(pd.Timestamp(scenario["start_date"]).date()),
                "effective_start_date": str(pd.Timestamp(active_dates[0]).date()),
            }
        )

    tax_path = pd.DataFrame(
        {
            "baseline_vat_percent": baseline_vat,
            "scenario_vat_percent": realised_vat,
            "baseline_excise": baseline_excise,
            "scenario_excise": realised_excise,
            "scenario_active": active,
        },
        index=dates,
    )
    tax_path.index.name = "date"
    return realised_vat, realised_excise, tax_path, metadata


def _monthly_yoy_paths(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    history: pd.Series,
) -> np.ndarray:
    values = np.asarray(paths, dtype=float)
    history = history.astype(float).sort_index()
    full_index = pd.date_range(
        min(history.index.min(), dates.min()),
        max(history.index.max(), dates.max()),
        freq="MS",
    )
    out = np.full(values.shape, np.nan, dtype=float)
    for draw in range(len(values)):
        series = history.reindex(full_index)
        series.loc[dates] = values[draw]
        yoy = 100.0 * (series / series.shift(12) - 1.0)
        out[draw] = yoy.reindex(dates).to_numpy(dtype=float)
    return out


def _historical_monthly_tax_path(
    tax_context: Mapping,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return the monthly VAT/excise path actually applied before the forecast.

    Raw Eurostat observations are semiannual.  This helper deliberately calls
    :func:`expand_semester_series`, so plotting raw publication markers against
    this path is also a visual check of the six-month dating invariant.
    """
    raw_vat = tax_context["vat_percent"].dropna().sort_index()
    raw_excise = tax_context["excise"].dropna().sort_index()
    if raw_vat.empty or raw_excise.empty:
        raise ValueError("Historical VAT and excise series must be non-empty.")
    start = min(raw_vat.index.min(), raw_excise.index.min())
    monthly_index = pd.date_range(
        pd.Timestamp(start).to_period("M").to_timestamp(how="start"),
        pd.Timestamp(end_date).to_period("M").to_timestamp(how="start"),
        freq="MS",
        name="date",
    )
    return pd.DataFrame(
        {
            "applied_vat_percent": expand_semester_series(raw_vat, monthly_index),
            "applied_excise": expand_semester_series(raw_excise, monthly_index),
        },
        index=monthly_index,
    )


def forecast_to_hicp_electricity(
    forecast: Mapping,
    result: Mapping,
    tax_context: Mapping,
    target_variable: str = "electricity_pre_tax",
    *,
    tax_scenario: Mapping | None = None,
) -> dict:
    """Map pre-tax electricity paths to HICP under baseline and optional tax scenarios.

    The baseline follows the paper's constant-edge tax assumption. A scenario
    must carry an explicit ``start_date``; before that date its realised tax path
    is *identical* to baseline. Excise scenarios must state their unit.
    """
    variables = list(forecast["variables"])
    if target_variable not in variables:
        raise ValueError(f"{target_variable!r} is not in the forecast variables.")
    target_index = variables.index(target_variable)
    dates = pd.DatetimeIndex(forecast["path_dates"], name="date")

    baseline_vat = expand_semester_series(tax_context["vat_percent"], dates)
    baseline_excise = expand_semester_series(tax_context["excise"], dates)
    vat, excise, tax_path, scenario_metadata = _apply_monthly_tax_scenario(
        baseline_vat, baseline_excise, tax_context, tax_scenario
    )

    pre_tax_paths = np.asarray(forecast["level_paths"], dtype=float)[:, :, target_index]
    baseline_hicp_paths = reattribute_electricity_taxes(
        pre_tax_paths,
        tax_context["gamma"],
        baseline_vat.to_numpy()[None, :],
        baseline_excise.to_numpy()[None, :],
    )
    hicp_paths = reattribute_electricity_taxes(
        pre_tax_paths,
        tax_context["gamma"],
        vat.to_numpy()[None, :],
        excise.to_numpy()[None, :],
    )

    balanced_end = pd.Timestamp(result["prep"]["balanced_end"])
    actual_hicp = tax_context["hicp_electricity"].copy().sort_index()
    hicp_history = actual_hicp.loc[actual_hicp.index <= balanced_end]
    baseline_hicp_yoy = _monthly_yoy_paths(baseline_hicp_paths, dates, hicp_history)
    hicp_yoy = _monthly_yoy_paths(hicp_paths, dates, hicp_history)

    model_levels = result["prep"]["levels"][target_variable].astype(float).sort_index()
    pre_tax_history = model_levels.loc[model_levels.index <= balanced_end]
    pre_tax_yoy = _monthly_yoy_paths(pre_tax_paths, dates, pre_tax_history)
    original_levels = result["prep"].get("levels_original", result["prep"]["levels"])
    actual_pre_tax = original_levels[target_variable].astype(float).sort_index()

    tail_length = int(forecast["tail_length"])
    forecast_origin = pd.Timestamp(forecast.get("last_calendar_date", dates[tail_length - 1] if tail_length else dates[0]))
    tax_history = _historical_monthly_tax_path(tax_context, forecast_origin)
    return {
        # Backward-compatible HICP keys refer to the realised scenario path.
        "hicp_level_paths": hicp_paths,
        "hicp_yoy_paths": hicp_yoy,
        "future_hicp_level_paths": hicp_paths[:, tail_length:],
        "future_hicp_yoy_paths": hicp_yoy[:, tail_length:],
        "actual_hicp": actual_hicp,
        # Matched draw-by-draw comparison objects.
        "pre_tax_level_paths": pre_tax_paths,
        "pre_tax_inflation_paths": pre_tax_yoy,
        "baseline_post_tax_level_paths": baseline_hicp_paths,
        "baseline_post_tax_inflation_paths": baseline_hicp_yoy,
        "post_tax_level_paths": hicp_paths,
        "post_tax_inflation_paths": hicp_yoy,
        "actual_pre_tax": actual_pre_tax,
        "actual_pre_tax_inflation": 100.0 * (actual_pre_tax / actual_pre_tax.shift(12) - 1.0),
        "actual_post_tax_inflation": 100.0 * (actual_hicp / actual_hicp.shift(12) - 1.0),
        "path_dates": dates,
        "future_dates": pd.DatetimeIndex(forecast["future_dates"], name="date"),
        "tail_length": tail_length,
        "frequency": "monthly",
        "inflation_unit": "percent year-on-year",
        "gamma": tax_context["gamma"],
        "price_unit": tax_context.get("price_unit", "source unit"),
        "excise_unit": tax_context.get("excise_unit", tax_context.get("price_unit", "source unit")),
        "vat_percent_path": vat,
        "excise_path": excise,
        "baseline_vat_percent_path": baseline_vat,
        "baseline_excise_path": baseline_excise,
        "tax_path": tax_path,
        "tax_history": tax_history,
        "tax_history_source": "Eurostat semiannual observations expanded with expand_semester_series",
        "forecast_origin": forecast_origin,
        "tax_scenario": scenario_metadata,
        "missing_data_method": forecast.get("missing_data_method", result.get("missing_data_method", "dk")),
        "missing_treatment_exact": bool(
            forecast.get("missing_treatment_exact", result.get("missing_treatment_exact", True))
        ),
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
