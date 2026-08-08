"""Adapters for the weekly petrol, diesel and liquid-fuels BVARs.

The generic engine in :mod:`energy_bvar_model` handles both monthly and weekly
calendars. This module contains only component-specific information:

* the three-variable equation for each weekly model;
* loading of Monday-labelled processed datasets;
* masking of builder-flagged partial final Bloomberg aggregates (temporary policy);
* recovery of WOB VAT and excise series from the processed-vintage manifest;
* ex-post tax re-attribution under the paper's random-walk assumption;
* weekly-to-monthly aggregation helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from energy_bvar_model import load_energy_panel
from energy_bvar_source_context import load_manifest_source_frame, read_processed_manifest


WEEKLY_FUEL_COLUMN_ALIASES = {
    # The single-workbook v9 builder may use the economically clearer
    # ``refined_gasoline_eur`` name. The model keeps its established canonical
    # contract ``refined_petroleum_eur`` so old and new processed vintages load
    # without notebook changes.
    "refined_gasoline_eur": "refined_petroleum_eur",
}


_WEEKLY_FUEL_SPECS: dict[str, dict[str, object]] = {
    "petrol": {
        "component": "car_fuels",
        "label": "Car fuels — petrol",
        "dataset": "car_fuels_weekly.csv",
        "target": "wob_petrol_pre_tax",
        "variables": [
            "wob_petrol_pre_tax",
            "crude_oil_eur",
            "refined_petroleum_eur",
        ],
        "wob_prefixes": ["wob_petroleum"],
        "units": {
            "wob_petrol_pre_tax": "EUR per 1,000 litres",
            "crude_oil_eur": "EUR per barrel",
            "refined_petroleum_eur": "EUR per metric tonne",
        },
        "notes": (
            "The refined-petroleum regressor is the Bloomberg NWE Eurobob Oxy "
            "Barge Balance of the Month series, converted from USD to EUR."
        ),
    },
    "diesel": {
        "component": "car_fuels",
        "label": "Car fuels — diesel",
        "dataset": "car_fuels_weekly.csv",
        "target": "wob_diesel_pre_tax",
        "variables": [
            "wob_diesel_pre_tax",
            "crude_oil_eur",
            "refined_diesel_eur",
        ],
        "wob_prefixes": ["wob_diesel"],
        "units": {
            "wob_diesel_pre_tax": "EUR per 1,000 litres",
            "crude_oil_eur": "EUR per barrel",
            "refined_diesel_eur": "EUR per barrel",
        },
        "notes": (
            "The refined-diesel regressor is the project's QS1 low-sulphur "
            "gasoil-future proxy for the paper's spot assessment."
        ),
    },
    "liquid_fuels": {
        "component": "liquid_fuels",
        "label": "Liquid fuels — heating oil",
        "dataset": "liquid_fuels_weekly.csv",
        "target": "wob_heating_oil_pre_tax",
        "variables": [
            "wob_heating_oil_pre_tax",
            "crude_oil_eur",
            "refined_diesel_eur",
        ],
        "wob_prefixes": ["wob_heating_oil", "wob_gas"],
        "units": {
            "wob_heating_oil_pre_tax": "EUR per 1,000 litres",
            "crude_oil_eur": "EUR per barrel",
            "refined_diesel_eur": "EUR per barrel",
        },
        "notes": (
            "The WOB 'gas' product is heating gas oil, i.e. the liquid-fuels "
            "consumer-price series used by the paper."
        ),
    },
}


def weekly_fuel_spec(model: str) -> dict[str, object]:
    """Return a defensive copy of one weekly model specification."""
    key = str(model).strip().lower()
    if key not in _WEEKLY_FUEL_SPECS:
        raise ValueError(
            f"model must be one of {sorted(_WEEKLY_FUEL_SPECS)}, got {model!r}."
        )
    spec = _WEEKLY_FUEL_SPECS[key]
    return {
        **spec,
        "variables": list(spec["variables"]),
        "wob_prefixes": list(spec["wob_prefixes"]),
        "units": dict(spec["units"]),
        "frequency": "weekly",
        "lags": 24,
        "calendar_rule": "W-MON",
    }


def _read_manifest(dataset_path: str | Path) -> dict:
    dataset_path = Path(dataset_path)
    manifest_path = dataset_path.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not read processed-vintage manifest: {manifest_path}") from exc


def _read_aggregation_coverage(dataset_path: str | Path) -> pd.DataFrame:
    """Read the builder sidecar that links partial aggregates to panel cells."""
    path = Path(dataset_path).parent / "aggregation_coverage.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {
        "panel_column",
        "panel_label_date",
        "affected_files",
        "final_period_complete",
        "coverage_ratio",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required coverage columns {sorted(missing)}."
        )
    frame["panel_label_date"] = pd.to_datetime(
        frame["panel_label_date"], errors="raise"
    )
    return frame


def _affected_file_contains(value: object, filename: str) -> bool:
    if pd.isna(value):
        return False
    return filename in {piece.strip() for piece in str(value).split("|") if piece.strip()}


def weekly_partial_period_table(
    dataset_path: str | Path,
    model: str = "petrol",
) -> pd.DataFrame:
    """Return coverage rows that map to the selected weekly equation."""
    spec = weekly_fuel_spec(model)
    dataset_path = Path(dataset_path)
    coverage = _read_aggregation_coverage(dataset_path)
    if coverage.empty:
        return coverage
    variables = set(spec["variables"])
    mask = coverage["affected_files"].map(
        lambda value: _affected_file_contains(value, dataset_path.name)
    ) & coverage["panel_column"].isin(variables)
    columns = [
        column
        for column in [
            "series",
            "panel_column",
            "panel_label_date",
            "final_period_input_observations",
            "expected_input_observations",
            "coverage_ratio",
            "final_period_complete",
            "first_input_date",
            "last_input_date",
        ]
        if column in coverage.columns
    ]
    return coverage.loc[mask, columns].sort_values(
        ["panel_label_date", "panel_column"]
    ).reset_index(drop=True)


def load_weekly_fuel_panel(
    dataset_path: str | Path,
    model: str = "petrol",
    *,
    mask_partial_final_period: bool = True,
) -> pd.DataFrame:
    """Load one paper equation from a processed weekly model dataset.

    The builder deliberately preserves partial final-period Bloomberg averages
    in the processed dataset.  Until a positive ``obs_cov`` measurement equation
    is implemented, the model adapter masks builder-flagged incomplete final
    aggregates to NaN.  The raw values and their daily inputs remain available
    in ``aggregation_coverage.csv`` and ``partial_period_inputs.csv``.

    Interior WOB gaps are *never* interpolated here; they remain NaN and are
    drawn inside the Gibbs sampler by Durbin--Koopman data augmentation.
    """
    spec = weekly_fuel_spec(model)
    dataset_path = Path(dataset_path)
    panel = load_energy_panel(
        dataset_path,
        variables=spec["variables"],
        aliases=WEEKLY_FUEL_COLUMN_ALIASES,
        frequency="weekly",
    )
    if not (panel.index.dayofweek == 0).all():
        raise ValueError("Weekly model dates must all be Mondays.")
    expected = pd.date_range(panel.index.min(), panel.index.max(), freq="W-MON")
    if not panel.index.equals(expected.rename("date")):
        raise ValueError("The weekly panel does not follow a complete W-MON calendar.")

    masked_rows = []
    if mask_partial_final_period:
        coverage = weekly_partial_period_table(dataset_path, model)
        for _, row in coverage.iterrows():
            complete = row.get("final_period_complete")
            if isinstance(complete, str):
                complete = complete.strip().lower() in {"true", "1", "yes"}
            if bool(complete):
                continue
            variable = str(row["panel_column"])
            date = pd.Timestamp(row["panel_label_date"])
            if variable not in panel.columns:
                raise ValueError(
                    f"Coverage sidecar maps to unknown panel column {variable!r}."
                )
            if date not in panel.index:
                raise ValueError(
                    f"Coverage sidecar maps {variable!r} to {date.date()}, "
                    "which is not in the processed weekly calendar."
                )
            original = panel.at[date, variable]
            if pd.isna(original):
                continue
            panel.at[date, variable] = np.nan
            masked_rows.append(
                {
                    "panel_column": variable,
                    "panel_label_date": date,
                    "coverage_ratio": float(row["coverage_ratio"]),
                    "original_value": float(original),
                    "policy": "temporary partial-period mask pending obs_cov measurement model",
                }
            )

    panel.attrs["partial_period_mask"] = pd.DataFrame(masked_rows)
    panel.attrs["partial_period_policy"] = (
        "mask builder-flagged incomplete final aggregates to NaN"
        if mask_partial_final_period
        else "disabled"
    )
    return panel


def weekly_fuel_series_construction_table(
    dataset_path: str | Path,
    model: str = "petrol",
) -> pd.DataFrame:
    """Describe source, unit and timing conventions for a weekly equation."""
    spec = weekly_fuel_spec(model)
    manifest = _read_manifest(dataset_path)
    calendar = manifest.get("construction", {}).get("weekly_calendar", {})
    shift = calendar.get("commodity_shift_weeks", 1)
    alignment = calendar.get(
        "alignment",
        "WOB Monday t is aligned to the preceding completed Monday-Sunday "
        "commodity week when shift=1.",
    )

    rows = []
    for name in spec["variables"]:
        if name.startswith("wob_"):
            source = "European Commission Weekly Oil Bulletin"
            construction = "Official pre-tax consumer price, Monday observation"
        elif name == "crude_oil_eur":
            source = "Bloomberg crude oil and EUR/USD"
            construction = "Daily USD/barrel converted to EUR/barrel, then weekly mean"
        elif name == "refined_petroleum_eur":
            source = "Bloomberg NWE Eurobob Oxy Barge and EUR/USD"
            construction = (
                "Daily USD/metric-tonne quote converted to EUR/metric tonne "
                "with EUR/USD, then weekly mean"
            )
        else:
            source = "Bloomberg refined-diesel future and EUR/USD"
            construction = (
                "Daily USD/metric-tonne quote converted with 7.44 barrels per "
                "tonne and EUR/USD, then weekly mean"
            )
        rows.append(
            {
                "series": name,
                "model_unit": spec["units"][name],
                "source": source,
                "construction": construction,
                "weekly_alignment": alignment if not name.startswith("wob_") else "Monday WOB date",
                "commodity_shift_weeks": shift if not name.startswith("wob_") else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("series")


def _read_weekly_source(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    date_candidates = [
        column for column in frame.columns
        if str(column).lower() in {"date", "time", "period"}
    ]
    if not date_candidates:
        raise ValueError(f"No date column found in {path}.")
    date_col = date_candidates[0]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    frame = frame.set_index(date_col).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).to_period("W-SUN").start_time
    frame.index.name = "date"
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).last()
    return frame.apply(pd.to_numeric, errors="coerce")


def load_weekly_tax_context(
    dataset_path: str | Path,
    model: str = "petrol",
    wob_path: str | Path | None = None,
) -> dict:
    """Recover WOB pre-tax, excise, VAT and after-tax prices.

    Legacy vintages use the recorded European-Commission CSV. v9 vintages read
    the ``European_Commission`` sheet from the single raw workbook. The latter
    must contain the VAT and IDT series as well as with-tax / no-tax prices;
    those two prices alone do not identify VAT and excise separately.
    """
    spec = weekly_fuel_spec(model)
    dataset_path = Path(dataset_path)

    if wob_path is not None:
        wob = _read_weekly_source(wob_path)
        source_info = {"mode": "explicit_csv", "path": Path(wob_path)}
        manifest_path = dataset_path.parent / "manifest.json"
    else:
        ec_rename = {
            "EUR_price_with_tax_euro95": "wob_petroleum_cpr_wtax",
            "EUR_price_wo_tax_euro95": "wob_petroleum_cpr_ntax",
            "EUR_price_with_tax_diesel": "wob_diesel_cpr_wtax",
            "EUR_price_wo_tax_diesel": "wob_diesel_cpr_ntax",
            "EUR_price_with_tax_heating_oil": "wob_gas_cpr_wtax",
            "EUR_price_wo_tax_heating_oil": "wob_gas_cpr_ntax",
        }
        wob, source_info = load_manifest_source_frame(
            dataset_path,
            "european_commission",
            simple_rename=ec_rename,
        )
        wob.index = pd.DatetimeIndex(wob.index).to_period("W-SUN").start_time
        wob.index.name = "date"
        if wob.index.has_duplicates:
            wob = wob.groupby(level=0).last()
        manifest_path = Path(source_info["manifest_path"])

    selected_prefix = None
    required_suffixes = ("cpr_ntax", "idt", "vat", "cpr_wtax")
    for prefix in spec["wob_prefixes"]:
        if all(f"{prefix}_{suffix}" in wob.columns for suffix in required_suffixes):
            selected_prefix = prefix
            break

    if selected_prefix is None:
        price_only = []
        for prefix in spec["wob_prefixes"]:
            if all(f"{prefix}_{suffix}" in wob.columns for suffix in ("cpr_ntax", "cpr_wtax")):
                price_only.append(prefix)
        if price_only and source_info.get("mode") == "single_excel_workbook":
            raise KeyError(
                "The v9 European_Commission workbook sheet contains with-tax and "
                "no-tax prices but not the VAT/IDT columns required for separate "
                "tax scenarios. Add the corresponding wob_*_vat and wob_*_idt "
                "series to that sheet (or pass an explicit legacy WOB tax CSV). "
                "WTAX and NTAX alone cannot uniquely identify VAT and excise."
            )
        tried = [
            [f"{prefix}_{suffix}" for suffix in required_suffixes]
            for prefix in spec["wob_prefixes"]
        ]
        raise KeyError(
            f"Could not find a complete WOB tax block for {model!r}; tried {tried}."
        )

    context = pd.DataFrame(
        {
            "pre_tax": wob[f"{selected_prefix}_cpr_ntax"],
            "excise": wob[f"{selected_prefix}_idt"],
            "vat_percent": wob[f"{selected_prefix}_vat"],
            "after_tax": wob[f"{selected_prefix}_cpr_wtax"],
        }
    ).sort_index().dropna(how="all")
    if context.empty:
        raise ValueError("The WOB tax context is empty.")
    identity = (context["pre_tax"] + context["excise"]) * (
        1.0 + context["vat_percent"] / 100.0
    )
    gap = (identity - context["after_tax"]).abs()
    return {
        "model": model,
        "target": spec["target"],
        "data": context,
        "wob_path": Path(source_info["path"]),
        "wob_source_mode": source_info.get("mode"),
        "wob_sheet": source_info.get("sheet"),
        "manifest_path": manifest_path,
        "wob_prefix": selected_prefix,
        "excise_unit": "EUR per 1,000 litres",
        "vat_unit": "percentage points",
        "max_tax_identity_error": float(gap.max(skipna=True)),
    }


def _carry_series_to_dates(series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    observed = series.dropna().sort_index()
    if observed.empty:
        raise ValueError(f"No observations are available for {series.name!r}.")
    union = observed.index.union(dates).sort_values()
    carried = observed.reindex(union).ffill().reindex(dates)
    if carried.isna().any():
        first_bad = carried.index[carried.isna()][0]
        raise ValueError(
            f"No {series.name} observation exists on or before {first_bad.date()}."
        )
    return carried.to_numpy(dtype=float)


def _normalise_weekly_tax_start(value) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("tax_scenario['start_date'] is invalid.")
    return date.to_period("W-SUN").start_time


def _weekly_tax_scenario_values(
    value,
    dates: pd.DatetimeIndex,
    *,
    label: str,
) -> np.ndarray:
    if isinstance(value, pd.Series):
        series = value.astype(float).copy()
        series.index = pd.DatetimeIndex(series.index).to_period("W-SUN").start_time
        if series.index.has_duplicates:
            raise ValueError(f"{label} scenario has duplicate model weeks.")
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


def _validate_weekly_vat_percent(values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    if np.any((values < 0.0) | (values > 100.0)):
        raise ValueError("VAT must be supplied in percentage points between 0 and 100.")
    suspicious = (values > 0.0) & (values < 1.0)
    if suspicious.any():
        raise ValueError(
            "VAT is expressed in percentage points: use 22.0 for 22%, not 0.22. "
            "Values strictly between 0 and 1 are rejected to catch unit mistakes."
        )


def _apply_weekly_tax_scenario(
    baseline_vat: np.ndarray,
    baseline_excise: np.ndarray,
    dates: pd.DatetimeIndex,
    tax_context: Mapping,
    tax_scenario: Mapping | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    realised_vat = np.asarray(baseline_vat, dtype=float).copy()
    realised_excise = np.asarray(baseline_excise, dtype=float).copy()
    baseline_vat = np.asarray(baseline_vat, dtype=float)
    baseline_excise = np.asarray(baseline_excise, dtype=float)
    active = np.zeros(len(dates), dtype=bool)
    metadata = {
        "active": False,
        "requested_start_date": None,
        "effective_start_date": None,
        "vat_conditioned": False,
        "excise_conditioned": False,
        "excise_unit": tax_context.get("excise_unit"),
    }

    if tax_scenario is not None:
        scenario = dict(tax_scenario)
        if "start_date" not in scenario:
            raise ValueError("tax_scenario requires an explicit 'start_date'.")
        if "vat_percent" not in scenario and "excise" not in scenario:
            raise ValueError("tax_scenario must condition VAT and/or excise.")

        start = _normalise_weekly_tax_start(scenario["start_date"])
        active = np.asarray(dates >= start, dtype=bool)
        active_dates = dates[active]
        if len(active_dates) == 0:
            raise ValueError(
                f"Tax scenario starts at {start.date()}, after the available path."
            )

        if "vat_percent" in scenario:
            values = _weekly_tax_scenario_values(
                scenario["vat_percent"], active_dates, label="vat_percent"
            )
            _validate_weekly_vat_percent(values)
            realised_vat[active] = values
            metadata["vat_conditioned"] = True

        if "excise" in scenario:
            expected_unit = str(tax_context.get("excise_unit", ""))
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
            values = _weekly_tax_scenario_values(
                scenario["excise"], active_dates, label="excise"
            )
            realised_excise[active] = values
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


def _weekly_yoy_paths(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    history: pd.Series,
) -> np.ndarray:
    values = np.asarray(paths, dtype=float)
    history = history.astype(float).sort_index()
    full_index = pd.date_range(
        min(history.index.min(), dates.min()),
        max(history.index.max(), dates.max()),
        freq="W-MON",
    )
    out = np.full(values.shape, np.nan, dtype=float)
    for draw in range(len(values)):
        series = history.reindex(full_index)
        series.loc[dates] = values[draw]
        yoy = 100.0 * (series / series.shift(52) - 1.0)
        out[draw] = yoy.reindex(dates).to_numpy(dtype=float)
    return out


def reattribute_weekly_taxes(
    forecast: Mapping,
    tax_context: Mapping,
    target_variable: str | None = None,
    *,
    tax_scenario: Mapping | None = None,
) -> dict:
    """Re-attribute weekly taxes under baseline and optional dated scenarios.

    Baseline VAT/excise are carried forward from the latest published WOB value.
    A scenario requires an explicit start date; before that date the realised
    tax path is exactly baseline. Excise conditions must state their unit.
    """
    variables = list(forecast["variables"])
    target = target_variable or str(tax_context["target"])
    if target not in variables:
        raise ValueError(f"Target {target!r} is not in forecast variables {variables}.")
    if str(forecast.get("frequency", "monthly")) != "weekly":
        raise ValueError("Tax re-attribution here requires a weekly forecast object.")

    dates = pd.DatetimeIndex(forecast["path_dates"], name="date")
    context = tax_context["data"]
    baseline_excise = _carry_series_to_dates(context["excise"], dates)
    baseline_vat = _carry_series_to_dates(context["vat_percent"], dates)
    vat, excise, tax_path, scenario_metadata = _apply_weekly_tax_scenario(
        baseline_vat, baseline_excise, dates, tax_context, tax_scenario
    )

    j = variables.index(target)
    pre_tax_paths = np.asarray(forecast["level_paths"], dtype=float)[:, :, j]
    baseline_after_tax_paths = (pre_tax_paths + baseline_excise[None, :]) * (
        1.0 + baseline_vat[None, :] / 100.0
    )
    after_tax_paths = (pre_tax_paths + excise[None, :]) * (
        1.0 + vat[None, :] / 100.0
    )

    pre_tax_yoy = _weekly_yoy_paths(pre_tax_paths, dates, context["pre_tax"])
    baseline_after_tax_yoy = _weekly_yoy_paths(
        baseline_after_tax_paths, dates, context["after_tax"]
    )
    after_tax_yoy = _weekly_yoy_paths(after_tax_paths, dates, context["after_tax"])

    actual_pre = context["pre_tax"].astype(float).sort_index()
    actual_post = context["after_tax"].astype(float).sort_index()
    forecast_origin = pd.Timestamp(forecast.get("last_calendar_date", dates[int(forecast["tail_length"]) - 1] if int(forecast["tail_length"]) else dates[0]))
    tax_history = (
        context[["vat_percent", "excise"]]
        .astype(float)
        .sort_index()
        .loc[lambda frame: frame.index <= forecast_origin]
        .rename(columns={
            "vat_percent": "applied_vat_percent",
            "excise": "applied_excise",
        })
    )
    return {
        "target": target,
        "path_dates": dates,
        "tail_length": int(forecast["tail_length"]),
        "future_dates": pd.DatetimeIndex(forecast["future_dates"], name="date"),
        "pre_tax_paths": pre_tax_paths,
        "after_tax_paths": after_tax_paths,
        "post_tax_paths": after_tax_paths,
        "pre_tax_level_paths": pre_tax_paths,
        "pre_tax_inflation_paths": pre_tax_yoy,
        "baseline_post_tax_level_paths": baseline_after_tax_paths,
        "baseline_post_tax_inflation_paths": baseline_after_tax_yoy,
        "post_tax_level_paths": after_tax_paths,
        "post_tax_inflation_paths": after_tax_yoy,
        "actual_pre_tax": actual_pre,
        "actual_post_tax": actual_post,
        "actual_pre_tax_inflation": 100.0 * (actual_pre / actual_pre.shift(52) - 1.0),
        "actual_post_tax_inflation": 100.0 * (actual_post / actual_post.shift(52) - 1.0),
        "excise": excise,
        "vat_percent": vat,
        "baseline_excise": baseline_excise,
        "baseline_vat_percent": baseline_vat,
        "baseline_excise_path": pd.Series(baseline_excise, index=dates, name="baseline_excise"),
        "baseline_vat_percent_path": pd.Series(baseline_vat, index=dates, name="baseline_vat_percent"),
        "tax_path": tax_path,
        "tax_history": tax_history,
        "tax_history_source": "European Commission Weekly Oil Bulletin observations at weekly model frequency",
        "forecast_origin": forecast_origin,
        "tax_scenario": scenario_metadata,
        "excise_unit": tax_context.get("excise_unit"),
        "frequency": "weekly",
        "inflation_unit": "percent, 52-week change",
        "tax_assumption": (
            "conditioned future tax path"
            if scenario_metadata["active"]
            else "random walk / latest value carried forward"
        ),
        "missing_data_method": forecast.get("missing_data_method", "dk"),
        "missing_treatment_exact": bool(forecast.get("missing_treatment_exact", True)),
    }

def weekly_paths_to_monthly_mean(
    paths: np.ndarray,
    dates: Sequence[pd.Timestamp],
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Aggregate draw-by-week paths to calendar-month means."""
    values = np.asarray(paths, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if values.ndim != 2 or values.shape[1] != len(dates):
        raise ValueError("paths must have shape (draws, len(dates)).")
    periods = dates.to_period("M")
    unique_periods = periods.unique().sort_values()
    monthly = np.column_stack(
        [values[:, periods == period].mean(axis=1) for period in unique_periods]
    )
    monthly_dates = unique_periods.to_timestamp(how="start")
    return monthly, pd.DatetimeIndex(monthly_dates, name="date")


def weekly_forecast_horizon_table(
    forecast: Mapping,
    variable: str,
    horizons: Sequence[int] = (1, 4, 13, 26),
    quantiles: Sequence[float] = (5, 16, 50, 84, 95),
) -> pd.DataFrame:
    """Summarise future level draws at selected weekly horizons."""
    variables = list(forecast["variables"])
    if variable not in variables:
        raise ValueError(f"Unknown variable {variable!r}.")
    dates = pd.DatetimeIndex(forecast["future_dates"])
    paths = np.asarray(forecast["future_level_paths"], dtype=float)
    j = variables.index(variable)
    selected = []
    for horizon in horizons:
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("Weekly horizons must be positive.")
        if horizon > len(dates):
            continue
        values = paths[:, horizon - 1, j]
        q = np.percentile(values, quantiles)
        selected.append(
            {
                "horizon": f"{horizon} week" + ("" if horizon == 1 else "s"),
                "weeks_ahead": horizon,
                "target_date": dates[horizon - 1],
                "q05": q[0],
                "q16": q[1],
                "median": q[2],
                "q84": q[3],
                "q95": q[4],
            }
        )
    if not selected:
        raise ValueError("No requested horizon lies within the forecast path.")
    return pd.DataFrame(selected).set_index("horizon")


__all__ = [
    "WEEKLY_FUEL_COLUMN_ALIASES",
    "weekly_fuel_spec",
    "load_weekly_fuel_panel",
    "weekly_fuel_series_construction_table",
    "weekly_partial_period_table",
    "load_weekly_tax_context",
    "reattribute_weekly_taxes",
    "weekly_paths_to_monthly_mean",
    "weekly_forecast_horizon_table",
]
