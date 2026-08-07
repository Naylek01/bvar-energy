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
            "refined_petroleum_eur": "EUR per barrel",
        },
        "notes": (
            "The refined-petroleum regressor is the project's PDS1 gasoil-future "
            "proxy for the paper's Eurobob gasoline assessment."
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
        else:
            source = "Bloomberg refined-product future and EUR/USD"
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

    By default the source WOB file is read from
    ``manifest['source_files']['european_commission']``. ``wob_path`` can be
    supplied explicitly when the project has moved since the vintage was built.
    """
    spec = weekly_fuel_spec(model)
    manifest = _read_manifest(dataset_path)
    if wob_path is None:
        raw_path = manifest.get("source_files", {}).get("european_commission")
        if not raw_path:
            raise FileNotFoundError(
                "The processed manifest does not record the European Commission "
                "source path. Pass wob_path explicitly."
            )
        wob_path = raw_path
    wob = _read_weekly_source(wob_path)

    selected_prefix = None
    required_suffixes = ("cpr_ntax", "idt", "vat", "cpr_wtax")
    for prefix in spec["wob_prefixes"]:
        if all(f"{prefix}_{suffix}" in wob.columns for suffix in required_suffixes):
            selected_prefix = prefix
            break
    if selected_prefix is None:
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
    ).sort_index()
    context = context.dropna(how="all")
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
        "wob_path": Path(wob_path),
        "manifest_path": Path(dataset_path).parent / "manifest.json",
        "wob_prefix": selected_prefix,
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


def reattribute_weekly_taxes(
    forecast: Mapping,
    tax_context: Mapping,
    target_variable: str | None = None,
) -> dict:
    """Re-attribute excise and VAT to posterior pre-tax level paths.

    Excise and VAT follow random walks: the latest published value is carried
    forward over the forecast horizon. The calculation is performed draw by
    draw as ``(pre_tax + excise) * (1 + VAT/100)``.
    """
    variables = list(forecast["variables"])
    target = target_variable or str(tax_context["target"])
    if target not in variables:
        raise ValueError(f"Target {target!r} is not in forecast variables {variables}.")
    if str(forecast.get("frequency", "monthly")) != "weekly":
        raise ValueError("Tax re-attribution here requires a weekly forecast object.")

    dates = pd.DatetimeIndex(forecast["path_dates"], name="date")
    context = tax_context["data"]
    excise = _carry_series_to_dates(context["excise"], dates)
    vat = _carry_series_to_dates(context["vat_percent"], dates)
    j = variables.index(target)
    pre_tax_paths = np.asarray(forecast["level_paths"], dtype=float)[:, :, j]
    after_tax_paths = (pre_tax_paths + excise[None, :]) * (
        1.0 + vat[None, :] / 100.0
    )
    return {
        "target": target,
        "path_dates": dates,
        "tail_length": int(forecast["tail_length"]),
        "future_dates": pd.DatetimeIndex(forecast["future_dates"]),
        "pre_tax_paths": pre_tax_paths,
        "after_tax_paths": after_tax_paths,
        "excise": excise,
        "vat_percent": vat,
        "frequency": "weekly",
        "tax_assumption": "random walk / latest value carried forward",
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
    "weekly_fuel_spec",
    "load_weekly_fuel_panel",
    "weekly_fuel_series_construction_table",
    "weekly_partial_period_table",
    "load_weekly_tax_context",
    "reattribute_weekly_taxes",
    "weekly_paths_to_monthly_mean",
    "weekly_forecast_horizon_table",
]
