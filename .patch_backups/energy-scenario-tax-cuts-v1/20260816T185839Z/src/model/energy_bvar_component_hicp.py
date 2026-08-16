"""Component-level HICP post-processing for the seven energy BVAR models.

The BVAR forecast store contains the variables that are actually estimated by
one component model.  For gas, electricity, petrol, diesel and liquid fuels the
target is a pre-tax consumer price, not an HICP index.  The component HICP store
created here is therefore a second, derived predictive product:

    raw BVAR forecast -> component-specific HICP transformation -> HICP store

No BVAR is re-estimated.  The transformations are the same ones used by the
component notebooks / Notebook 10:

* gas and electricity: re-attribute the latest VAT/excise path and invert the
  pre-tax construction through the component gamma;
* petrol, diesel and liquid fuels: re-attribute WOB taxes, convert the weekly
  consumer-price path to complete monthly means, then use those price relatives
  to rebase onto the published HICP sub-index;
* heat energy and solid fuels: the BVAR target is already the HICP index.

The resulting object always exposes a monthly HICP level path and a monthly
HICP year-on-year path, so the dashboard can display the same economic object
for all seven component models without depending on the six-model aggregate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


HICP_STORE_SCHEMA_VERSION = "1.0"

_HICP_SERIES_BY_MODEL: dict[str, str] = {
    "gas": "hicp_gas",
    "electricity": "hicp_electricity",
    "heat_energy": "hicp_heat_energy",
    "solid_fuels": "hicp_solid_fuels",
    "car_fuels_petrol": "hicp_petrol",
    "car_fuels_diesel": "hicp_diesel",
    "liquid_fuels": "hicp_liquid_fuels",
}

_HICP_LABEL_BY_MODEL: dict[str, str] = {
    "gas": "HICP Gas",
    "electricity": "HICP Electricity",
    "heat_energy": "HICP Heat Energy",
    "solid_fuels": "HICP Solid Fuels",
    "car_fuels_petrol": "HICP Petrol",
    "car_fuels_diesel": "HICP Diesel",
    "liquid_fuels": "HICP Liquid Fuels",
}

_WEEKLY_HICP_COLUMNS: dict[str, str] = {
    "car_fuels_petrol": "hicp_petrol",
    "car_fuels_diesel": "hicp_diesel",
    "liquid_fuels": "hicp_liquid_fuels",
}


class ComponentHICPError(RuntimeError):
    """Raised when a component HICP predictive product cannot be formed."""


def _monthly_mean_series(series: pd.Series) -> pd.Series:
    series = series.astype(float).dropna().sort_index()
    grouped = series.groupby(series.index.to_period("M")).mean()
    grouped.index = pd.DatetimeIndex(
        grouped.index.to_timestamp(how="start"), name="date"
    )
    return grouped


def _monthly_yoy_paths(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    history: pd.Series,
) -> np.ndarray:
    values = np.asarray(paths, dtype=float)
    dates = pd.DatetimeIndex(dates, name="date")
    observed = history.astype(float).dropna().sort_index()
    if values.ndim != 2 or values.shape[1] != len(dates):
        raise ComponentHICPError(
            f"HICP paths must have shape (draws, {len(dates)}); got {values.shape}."
        )
    if observed.empty:
        return np.full_like(values, np.nan, dtype=float)

    full_index = pd.date_range(
        min(observed.index.min(), dates.min()),
        max(observed.index.max(), dates.max()),
        freq="MS",
    )
    out = np.full_like(values, np.nan, dtype=float)
    for draw in range(values.shape[0]):
        series = observed.reindex(full_index)
        series.loc[dates] = values[draw]
        yoy = 100.0 * (series / series.shift(12) - 1.0)
        out[draw] = yoy.reindex(dates).to_numpy(dtype=float)
    return out


def _read_hicp_history(processed_dir: Path, column: str) -> pd.Series:
    path = Path(processed_dir) / "hicp_indices_monthly.csv"
    if not path.is_file():
        raise ComponentHICPError(
            f"Weekly HICP post-processing requires {path}."
        )
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if column not in frame.columns:
        raise ComponentHICPError(f"{path} does not contain {column!r}.")
    series = pd.to_numeric(frame[column], errors="coerce").rename(column)
    series.index = pd.DatetimeIndex(series.index, name="date")
    return series


def _result_proxy(forecast: Mapping, history: pd.DataFrame) -> dict:
    """Minimal result object accepted by the notebook HICP adapters.

    Legacy forecast stores may not have ``draws.npz``.  Their saved observed
    history plus forecast metadata are sufficient for HICP post-processing:
    the HICP level paths themselves depend on the forecast and the vintage tax
    context, while the result is needed only for the balanced-end cut and for
    pre-tax comparison histories returned by the adapters.
    """
    if not isinstance(history.index, pd.DatetimeIndex):
        history = history.copy()
        history.index = pd.DatetimeIndex(pd.to_datetime(history.index), name="date")
    balanced_end = forecast.get("balanced_end")
    if balanced_end is None:
        path_dates = pd.DatetimeIndex(forecast["path_dates"])
        tail_length = int(forecast.get("tail_length", 0))
        if tail_length:
            balanced_end = path_dates[tail_length - 1]
        else:
            balanced_end = history.dropna(how="all").index.max()
    return {
        "prep": {
            "balanced_end": pd.Timestamp(balanced_end),
            "levels": history.copy(),
            "levels_original": history.copy(),
        },
        "missing_data_method": forecast.get("missing_data_method", "dk"),
        "missing_treatment_exact": bool(
            forecast.get("missing_treatment_exact", True)
        ),
    }


def _normalise(
    *,
    model_id: str,
    forecast_name: str,
    run_id: str | None,
    vintage: str | None,
    hicp_series: str,
    label: str,
    level_paths: np.ndarray,
    yoy_paths: np.ndarray,
    path_dates,
    future_dates,
    actual_hicp: pd.Series,
    source_frequency: str,
    transformation_method: str,
    draw_indices: np.ndarray | None = None,
    extra: Mapping | None = None,
) -> dict:
    dates = pd.DatetimeIndex(path_dates, name="date")
    future = pd.DatetimeIndex(future_dates, name="date")
    levels = np.asarray(level_paths, dtype=float)
    yoy = np.asarray(yoy_paths, dtype=float)
    if levels.ndim != 2 or yoy.shape != levels.shape:
        raise ComponentHICPError(
            f"{model_id}: HICP level/YoY shapes are inconsistent: "
            f"{levels.shape} vs {yoy.shape}."
        )
    if levels.shape[1] != len(dates):
        raise ComponentHICPError(
            f"{model_id}: HICP paths have {levels.shape[1]} dates but "
            f"calendar has {len(dates)}."
        )
    if len(future):
        positions = dates.get_indexer(future)
        if (positions < 0).any():
            raise ComponentHICPError(
                f"{model_id}: HICP future dates are not a subset of path dates."
            )
        first = int(positions.min())
        expected = dates[first:]
        if not expected.equals(future):
            raise ComponentHICPError(
                f"{model_id}: HICP future dates must be a trailing block of path_dates."
            )
        tail_length = first
    else:
        tail_length = len(dates)

    history = actual_hicp.astype(float).sort_index().rename(hicp_series)
    history.index = pd.DatetimeIndex(pd.to_datetime(history.index), name="date")

    out = {
        "hicp_schema_version": HICP_STORE_SCHEMA_VERSION,
        "model_id": str(model_id),
        "vintage": None if vintage is None else str(vintage),
        "run_id": None if run_id is None else str(run_id),
        "forecast_name": str(forecast_name),
        "hicp_series": hicp_series,
        "hicp_label": label,
        "hicp_level_paths": levels,
        "hicp_yoy_paths": yoy,
        "path_dates": dates,
        "future_dates": future,
        "tail_length": int(tail_length),
        "frequency": "monthly",
        "source_frequency": str(source_frequency),
        "inflation_unit": "% y/y",
        "level_unit": "HICP index",
        "transformation_method": str(transformation_method),
        "actual_hicp": history,
        "draw_indices": (
            np.arange(levels.shape[0], dtype=int)
            if draw_indices is None
            else np.asarray(draw_indices, dtype=int)
        ),
    }
    if extra:
        out.update(dict(extra))
    return out


def build_component_hicp_forecast(
    model_id: str,
    forecast: Mapping,
    *,
    result: Mapping,
    dataset_path: str | Path,
    project_root: str | Path | None = None,
    forecast_name: str = "unconditional",
) -> dict:
    """Build the baseline monthly HICP predictive product for one component."""
    from energy_bvar_pipeline import find_project_root, model_spec, resolve_model_id

    canonical = resolve_model_id(model_id)
    spec = model_spec(canonical)
    dataset = Path(dataset_path)
    root = find_project_root(project_root)
    processed_dir = dataset.parent
    run_id = result.get("metadata", {}).get("run_id")
    vintage = result.get("metadata", {}).get("vintage", processed_dir.name)
    hicp_series = _HICP_SERIES_BY_MODEL[canonical]
    label = _HICP_LABEL_BY_MODEL[canonical]

    if spec.family == "gas":
        from energy_bvar_gas import load_gas_tax_context, forecast_to_hicp_gas

        context = load_gas_tax_context(dataset)
        obj = forecast_to_hicp_gas(forecast, result, context)
        return _normalise(
            model_id=canonical,
            forecast_name=forecast_name,
            run_id=run_id,
            vintage=vintage,
            hicp_series=hicp_series,
            label=label,
            level_paths=obj["hicp_level_paths"],
            yoy_paths=obj["hicp_yoy_paths"],
            path_dates=obj["path_dates"],
            future_dates=obj["future_dates"],
            actual_hicp=obj["actual_hicp"],
            source_frequency=spec.frequency,
            transformation_method="forecast_to_hicp_gas",
            extra={
                "gamma": obj.get("gamma"),
                "price_unit": obj.get("price_unit"),
                "excise_unit": obj.get("excise_unit"),
                "tax_assumption": "latest published VAT/excise carried forward",
                "tax_path": obj.get("tax_path"),
                "tax_history": obj.get("tax_history"),
                "missing_data_method": obj.get("missing_data_method"),
                "missing_treatment_exact": obj.get("missing_treatment_exact"),
            },
        )

    if spec.family == "electricity":
        from energy_bvar_electricity import (
            load_electricity_tax_context,
            forecast_to_hicp_electricity,
        )

        context = load_electricity_tax_context(dataset)
        obj = forecast_to_hicp_electricity(forecast, result, context)
        return _normalise(
            model_id=canonical,
            forecast_name=forecast_name,
            run_id=run_id,
            vintage=vintage,
            hicp_series=hicp_series,
            label=label,
            level_paths=obj["hicp_level_paths"],
            yoy_paths=obj["hicp_yoy_paths"],
            path_dates=obj["path_dates"],
            future_dates=obj["future_dates"],
            actual_hicp=obj["actual_hicp"],
            source_frequency=spec.frequency,
            transformation_method="forecast_to_hicp_electricity",
            extra={
                "gamma": obj.get("gamma"),
                "price_unit": obj.get("price_unit"),
                "excise_unit": obj.get("excise_unit"),
                "tax_assumption": "latest published VAT/excise carried forward",
                "tax_path": obj.get("tax_path"),
                "tax_history": obj.get("tax_history"),
                "missing_data_method": obj.get("missing_data_method"),
                "missing_treatment_exact": obj.get("missing_treatment_exact"),
            },
        )

    if spec.family == "monthly_hicp":
        from energy_bvar_monthly_hicp import forecast_hicp_component

        obj = forecast_hicp_component(
            forecast, result, component=str(spec.spec_key)
        )
        return _normalise(
            model_id=canonical,
            forecast_name=forecast_name,
            run_id=run_id,
            vintage=vintage,
            hicp_series=hicp_series,
            label=label,
            level_paths=obj["hicp_level_paths"],
            yoy_paths=obj["hicp_yoy_paths"],
            path_dates=obj["path_dates"],
            future_dates=obj["future_dates"],
            actual_hicp=obj["actual_hicp"],
            source_frequency=spec.frequency,
            transformation_method="forecast_hicp_component",
            extra={
                "tax_assumption": "not applicable; BVAR target is the HICP index",
                "missing_data_method": obj.get("missing_data_method"),
                "missing_treatment_exact": obj.get("missing_treatment_exact"),
            },
        )

    if spec.family == "weekly_fuel":
        from energy_bvar_weekly_fuels import (
            load_weekly_tax_context,
            reattribute_weekly_taxes,
        )
        from energy_bvar_aggregate import (
            filter_positive_price_draws,
            rebase_price_paths_to_hicp_index,
            weekly_paths_with_history_to_monthly_mean,
        )

        context = load_weekly_tax_context(dataset, model=str(spec.spec_key))
        taxed = reattribute_weekly_taxes(forecast, context)
        monthly, monthly_dates, coverage = weekly_paths_with_history_to_monthly_mean(
            taxed["post_tax_level_paths"],
            taxed["path_dates"],
            context["data"]["after_tax"],
            return_coverage=True,
        )
        filtered = filter_positive_price_draws(
            monthly,
            admissibility_paths={
                "pre_tax_weekly": np.asarray(
                    taxed["pre_tax_level_paths"], dtype=float
                ),
                "post_tax_weekly": np.asarray(
                    taxed["post_tax_level_paths"], dtype=float
                ),
            },
            label=str(spec.spec_key),
        )
        monthly = np.asarray(filtered["baseline_paths"], dtype=float)
        historical_price = _monthly_mean_series(context["data"]["after_tax"])
        hicp_history = _read_hicp_history(
            processed_dir, _WEEKLY_HICP_COLUMNS[canonical]
        )
        rebased = rebase_price_paths_to_hicp_index(
            monthly,
            monthly_dates,
            historical_price,
            hicp_history,
        )
        dates = pd.DatetimeIndex(rebased["dates"], name="date")
        yoy = _monthly_yoy_paths(
            np.asarray(rebased["index_paths"], dtype=float), dates, hicp_history
        )

        raw_future = pd.DatetimeIndex(forecast.get("future_dates", []))
        if len(raw_future):
            first_future_month = raw_future[0].to_period("M").to_timestamp(how="start")
            future_dates = dates[dates >= first_future_month]
        else:
            future_dates = pd.DatetimeIndex([], name="date")

        diagnostics = dict(filtered["diagnostics"])
        return _normalise(
            model_id=canonical,
            forecast_name=forecast_name,
            run_id=run_id,
            vintage=vintage,
            hicp_series=hicp_series,
            label=label,
            level_paths=rebased["index_paths"],
            yoy_paths=yoy,
            path_dates=dates,
            future_dates=future_dates,
            actual_hicp=hicp_history,
            source_frequency=spec.frequency,
            transformation_method=(
                "reattribute_weekly_taxes -> complete monthly WOB means -> "
                "proportional HICP rebase"
            ),
            draw_indices=filtered["draw_indices"],
            extra={
                "tax_assumption": taxed.get("tax_assumption"),
                "excise_unit": taxed.get("excise_unit"),
                "anchor_date": rebased.get("anchor_date"),
                "anchor_price": rebased.get("anchor_price"),
                "anchor_hicp": rebased.get("anchor_hicp"),
                "rebase_method": rebased.get("method"),
                "total_draws": diagnostics.get("total_draws"),
                "retained_draws": diagnostics.get("retained_draws"),
                "rejected_draws": diagnostics.get("rejected_draws"),
                "rejection_rate": diagnostics.get("rejection_rate"),
                "distribution_interpretation": diagnostics.get(
                    "distribution_interpretation"
                ),
                "weekly_monthly_coverage": coverage.reset_index(),
                "tax_path": taxed.get("tax_path"),
                "tax_history": taxed.get("tax_history"),
                "missing_data_method": taxed.get("missing_data_method"),
                "missing_treatment_exact": taxed.get("missing_treatment_exact"),
            },
        )

    raise ComponentHICPError(
        f"{canonical}: no HICP post-processing rule for family {spec.family!r}."
    )


def materialize_component_hicp_store(
    run_directory: str | Path,
    *,
    project_root: str | Path | None = None,
    forecast_name: str = "unconditional",
    overwrite: bool = False,
) -> Path:
    """Create a missing HICP store for an already-saved component forecast.

    This is a migration path for legacy runs.  It reads the saved forecast and
    observed run history, reconstructs only the HICP post-processing layer, and
    writes ``hicp_metadata.json`` / ``hicp_draws.npz`` / ``hicp_history.csv``.
    It never runs the Gibbs sampler again.
    """
    from energy_bvar_io import (
        load_energy_bvar_forecast,
        load_energy_bvar_history,
        save_energy_bvar_hicp_forecast,
    )
    from energy_bvar_pipeline import build_panel, find_project_root, model_spec

    run_directory = Path(run_directory)
    metadata_path = run_directory / "metadata.json"
    if not metadata_path.is_file():
        raise ComponentHICPError(f"Run metadata not found: {metadata_path}")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_id = str(metadata["model_id"])
    vintage = str(metadata["vintage"])
    forecast_dir = run_directory / "forecasts" / str(forecast_name)
    hicp_metadata = forecast_dir / "hicp_metadata.json"
    hicp_draws = forecast_dir / "hicp_draws.npz"
    if hicp_metadata.is_file() and hicp_draws.is_file() and not overwrite:
        return forecast_dir

    forecast = load_energy_bvar_forecast(forecast_dir)
    try:
        history = load_energy_bvar_history(run_directory)
    except FileNotFoundError:
        panel = build_panel(model_id, vintage, project_root=project_root)
        history = panel.levels.copy()

    result = _result_proxy(forecast, history)
    result["metadata"] = dict(metadata)
    root = find_project_root(project_root)
    dataset = root / "data" / "processed" / vintage / model_spec(model_id).dataset_file
    if not dataset.is_file():
        raise ComponentHICPError(
            f"Processed dataset required for HICP post-processing is missing: {dataset}"
        )

    hicp = build_component_hicp_forecast(
        model_id,
        forecast,
        result=result,
        dataset_path=dataset,
        project_root=root,
        forecast_name=forecast_name,
    )
    save_energy_bvar_hicp_forecast(hicp, forecast_dir, overwrite=overwrite)
    return forecast_dir


# ---------------------------------------------------------------------------
# Interactive component tax scenarios
# ---------------------------------------------------------------------------

TAX_SCENARIO_MODEL_IDS: tuple[str, ...] = (
    "gas",
    "electricity",
    "car_fuels_petrol",
    "car_fuels_diesel",
    "liquid_fuels",
)


def _subset_forecast_draws(
    forecast: Mapping,
    max_draws: int | None,
    *,
    seed: int = 2026,
) -> tuple[dict, np.ndarray]:
    """Return a deterministic posterior subset while preserving paired paths."""
    out = dict(forecast)
    levels = np.asarray(forecast["level_paths"])
    n_total = int(levels.shape[0])
    if max_draws is None or int(max_draws) <= 0 or n_total <= int(max_draws):
        indices = np.arange(n_total, dtype=int)
    else:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(n_total, size=int(max_draws), replace=False))

    for key, value in list(forecast.items()):
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_total:
            out[key] = value[indices]
    out["_scenario_draw_indices"] = indices
    return out, indices


def component_tax_scenario_contract(
    model_id: str,
    forecast: Mapping,
    *,
    dataset_path: str | Path,
) -> dict:
    """Return tax-control defaults/units for one saved component forecast.

    The contract is deliberately component-local.  No six-model aggregate is
    required merely to vary VAT or excise for gas, electricity or a WOB fuel.
    """
    from energy_bvar_pipeline import model_spec, resolve_model_id

    canonical = resolve_model_id(model_id)
    spec = model_spec(canonical)
    supported = canonical in TAX_SCENARIO_MODEL_IDS
    if not supported:
        return {
            "supported": False,
            "model_id": canonical,
            "reason": (
                "No explicit VAT/excise bridge is defined for this component; "
                "its BVAR target is already an HICP index."
            ),
        }

    dataset = Path(dataset_path)
    dates = pd.DatetimeIndex(forecast["path_dates"], name="date")
    future_dates = pd.DatetimeIndex(forecast.get("future_dates", []), name="date")
    if len(future_dates) == 0:
        raise ComponentHICPError(f"{canonical}: the selected forecast has no future dates.")

    if spec.family == "gas":
        from energy_bvar_gas import expand_semester_series, load_gas_tax_context
        context = load_gas_tax_context(dataset)
        vat = expand_semester_series(context["vat_percent"], dates)
        excise = expand_semester_series(context["excise"], dates)
        default_start = pd.Timestamp(future_dates[0]).to_period("M").to_timestamp(how="start")
        frequency = "monthly"
    elif spec.family == "electricity":
        from energy_bvar_electricity import (
            expand_semester_series,
            load_electricity_tax_context,
        )
        context = load_electricity_tax_context(dataset)
        vat = expand_semester_series(context["vat_percent"], dates)
        excise = expand_semester_series(context["excise"], dates)
        default_start = pd.Timestamp(future_dates[0]).to_period("M").to_timestamp(how="start")
        frequency = "monthly"
    else:
        from energy_bvar_weekly_fuels import load_weekly_tax_context, reattribute_weekly_taxes
        context = load_weekly_tax_context(dataset, model=str(spec.spec_key))
        taxed = reattribute_weekly_taxes(forecast, context)
        vat = pd.Series(
            np.asarray(taxed["baseline_vat_percent"], dtype=float),
            index=dates,
            name="vat_percent",
        )
        excise = pd.Series(
            np.asarray(taxed["baseline_excise"], dtype=float),
            index=dates,
            name="excise",
        )
        default_start = pd.Timestamp(future_dates[0]).to_period("W-SUN").start_time
        frequency = "weekly"

    start_pos = int(np.searchsorted(dates.to_numpy(), np.datetime64(default_start), side="left"))
    start_pos = min(max(start_pos, 0), len(dates) - 1)
    return {
        "supported": True,
        "model_id": canonical,
        "frequency": frequency,
        "default_start": default_start,
        "min_start": pd.Timestamp(future_dates[0]),
        "max_start": pd.Timestamp(dates[-1]),
        "vat_unit": "percentage points",
        "excise_unit": str(context.get("excise_unit", "source unit")),
        "baseline_vat_at_start": float(vat.iloc[start_pos]),
        "baseline_excise_at_start": float(excise.iloc[start_pos]),
    }


def build_component_tax_scenario(
    model_id: str,
    forecast: Mapping,
    *,
    result: Mapping,
    dataset_path: str | Path,
    project_root: str | Path | None = None,
    start_date,
    vat_delta_pp: float = 0.0,
    excise_delta: float = 0.0,
    max_draws: int | None = 500,
    subset_seed: int = 2026,
) -> dict:
    """Apply a paired ex-post VAT/excise scenario to saved component draws.

    The raw BVAR path is held fixed.  Only the component's existing HICP bridge
    is rerun, so baseline and scenario are matched draw by draw.  The function
    returns monthly HICP level/YoY paths and paired scenario impacts.
    """
    from energy_bvar_pipeline import model_spec, resolve_model_id

    canonical = resolve_model_id(model_id)
    if canonical not in TAX_SCENARIO_MODEL_IDS:
        raise ComponentHICPError(
            f"{canonical}: no explicit VAT/excise scenario bridge is defined."
        )
    spec = model_spec(canonical)
    dataset = Path(dataset_path)
    reduced, selected_indices = _subset_forecast_draws(
        forecast, max_draws, seed=subset_seed
    )
    raw_dates = pd.DatetimeIndex(reduced["path_dates"], name="date")
    start = pd.Timestamp(start_date)
    vat_delta_pp = float(vat_delta_pp or 0.0)
    excise_delta = float(excise_delta or 0.0)

    if spec.family in {"gas", "electricity"}:
        if spec.family == "gas":
            from energy_bvar_gas import (
                expand_semester_series,
                forecast_to_hicp_gas as adapter,
                load_gas_tax_context as load_context,
            )
        else:
            from energy_bvar_electricity import (
                expand_semester_series,
                forecast_to_hicp_electricity as adapter,
                load_electricity_tax_context as load_context,
            )

        context = load_context(dataset)
        baseline_vat = expand_semester_series(context["vat_percent"], raw_dates)
        baseline_excise = expand_semester_series(context["excise"], raw_dates)
        start = start.to_period("M").to_timestamp(how="start")
        active_dates = raw_dates[raw_dates >= start]
        if len(active_dates) == 0:
            raise ComponentHICPError(
                f"Tax scenario starts after the selected forecast path: {start.date()}."
            )
        scenario = None
        if vat_delta_pp != 0.0 or excise_delta != 0.0:
            scenario = {"start_date": start}
            if vat_delta_pp != 0.0:
                scenario["vat_percent"] = pd.Series(
                    baseline_vat.loc[active_dates].to_numpy(dtype=float) + vat_delta_pp,
                    index=active_dates,
                )
            if excise_delta != 0.0:
                scenario["excise"] = pd.Series(
                    baseline_excise.loc[active_dates].to_numpy(dtype=float) + excise_delta,
                    index=active_dates,
                )
                scenario["excise_unit"] = str(context["excise_unit"])

        obj = adapter(reduced, result, context, tax_scenario=scenario)
        baseline_level = np.asarray(obj["baseline_post_tax_level_paths"], dtype=float)
        scenario_level = np.asarray(obj["post_tax_level_paths"], dtype=float)
        baseline_yoy = np.asarray(obj["baseline_post_tax_inflation_paths"], dtype=float)
        scenario_yoy = np.asarray(obj["post_tax_inflation_paths"], dtype=float)
        dates = pd.DatetimeIndex(obj["path_dates"], name="date")
        future_dates = pd.DatetimeIndex(obj["future_dates"], name="date")
        actual_hicp = pd.Series(obj["actual_hicp"], dtype=float).sort_index()
        tax_path = pd.DataFrame(obj["tax_path"]).copy()
        tax_history = pd.DataFrame(obj["tax_history"]).copy()
        raw_tax = pd.concat(
            [
                context["vat_percent"].rename("vat_percent"),
                context["excise"].rename("excise"),
            ],
            axis=1,
        ).sort_index()
        retained_indices = selected_indices
        rejection_rate = 0.0
        distribution_interpretation = "posterior predictive distribution"
        scenario_meta = dict(obj.get("tax_scenario") or {})
        excise_unit = str(obj.get("excise_unit", context.get("excise_unit", "source unit")))
        source_frequency = "monthly"

    else:
        from energy_bvar_weekly_fuels import load_weekly_tax_context, reattribute_weekly_taxes
        from energy_bvar_aggregate import (
            filter_positive_price_draws,
            rebase_price_paths_to_hicp_index,
            weekly_paths_with_history_to_monthly_mean,
        )

        context = load_weekly_tax_context(dataset, model=str(spec.spec_key))
        baseline_obj = reattribute_weekly_taxes(reduced, context)
        baseline_vat = pd.Series(
            np.asarray(baseline_obj["baseline_vat_percent"], dtype=float),
            index=raw_dates,
        )
        baseline_excise = pd.Series(
            np.asarray(baseline_obj["baseline_excise"], dtype=float),
            index=raw_dates,
        )
        start = start.to_period("W-SUN").start_time
        active_dates = raw_dates[raw_dates >= start]
        if len(active_dates) == 0:
            raise ComponentHICPError(
                f"Tax scenario starts after the selected forecast path: {start.date()}."
            )
        scenario = None
        if vat_delta_pp != 0.0 or excise_delta != 0.0:
            scenario = {"start_date": start}
            if vat_delta_pp != 0.0:
                scenario["vat_percent"] = pd.Series(
                    baseline_vat.loc[active_dates].to_numpy(dtype=float) + vat_delta_pp,
                    index=active_dates,
                )
            if excise_delta != 0.0:
                scenario["excise"] = pd.Series(
                    baseline_excise.loc[active_dates].to_numpy(dtype=float) + excise_delta,
                    index=active_dates,
                )
                scenario["excise_unit"] = str(context["excise_unit"])

        taxed = reattribute_weekly_taxes(reduced, context, tax_scenario=scenario)
        baseline_monthly, monthly_dates, _ = weekly_paths_with_history_to_monthly_mean(
            taxed["baseline_post_tax_level_paths"],
            taxed["path_dates"],
            context["data"]["after_tax"],
            return_coverage=True,
        )
        scenario_monthly, scenario_dates, _ = weekly_paths_with_history_to_monthly_mean(
            taxed["post_tax_level_paths"],
            taxed["path_dates"],
            context["data"]["after_tax"],
            return_coverage=True,
        )
        if not monthly_dates.equals(scenario_dates):
            raise ComponentHICPError("Baseline and tax-scenario monthly calendars differ.")

        price_filter = filter_positive_price_draws(
            baseline_monthly,
            scenario_monthly,
            admissibility_paths={
                "pre_tax_weekly": np.asarray(taxed["pre_tax_level_paths"], dtype=float),
                "baseline_after_tax_weekly": np.asarray(
                    taxed["baseline_post_tax_level_paths"], dtype=float
                ),
                "scenario_after_tax_weekly": np.asarray(
                    taxed["post_tax_level_paths"], dtype=float
                ),
            },
            label=str(spec.spec_key),
        )
        baseline_monthly = np.asarray(price_filter["baseline_paths"], dtype=float)
        scenario_monthly = np.asarray(price_filter["scenario_paths"], dtype=float)
        historical_price = _monthly_mean_series(context["data"]["after_tax"])
        hicp_history = _read_hicp_history(dataset.parent, _WEEKLY_HICP_COLUMNS[canonical])
        baseline_rebased = rebase_price_paths_to_hicp_index(
            baseline_monthly, monthly_dates, historical_price, hicp_history
        )
        scenario_rebased = rebase_price_paths_to_hicp_index(
            scenario_monthly, monthly_dates, historical_price, hicp_history
        )
        dates = pd.DatetimeIndex(baseline_rebased["dates"], name="date")
        if not dates.equals(pd.DatetimeIndex(scenario_rebased["dates"], name="date")):
            raise ComponentHICPError("Baseline and tax-scenario HICP calendars differ.")
        baseline_level = np.asarray(baseline_rebased["index_paths"], dtype=float)
        scenario_level = np.asarray(scenario_rebased["index_paths"], dtype=float)
        baseline_yoy = _monthly_yoy_paths(baseline_level, dates, hicp_history)
        scenario_yoy = _monthly_yoy_paths(scenario_level, dates, hicp_history)
        raw_future = pd.DatetimeIndex(reduced.get("future_dates", []))
        if len(raw_future):
            first_future_month = raw_future[0].to_period("M").to_timestamp(how="start")
            future_dates = dates[dates >= first_future_month]
        else:
            future_dates = pd.DatetimeIndex([], name="date")
        actual_hicp = hicp_history
        tax_path = pd.DataFrame(taxed["tax_path"]).copy()
        tax_history = pd.DataFrame(taxed["tax_history"]).copy()
        raw_tax = context["data"][["vat_percent", "excise"]].astype(float).sort_index()
        relative_indices = np.asarray(price_filter["draw_indices"], dtype=int)
        retained_indices = selected_indices[relative_indices]
        diagnostics = dict(price_filter["diagnostics"])
        rejection_rate = float(diagnostics.get("rejection_rate", 0.0))
        distribution_interpretation = str(
            diagnostics.get(
                "distribution_interpretation",
                "posterior predictive distribution conditional on admissibility",
            )
        )
        scenario_meta = dict(taxed.get("tax_scenario") or {})
        excise_unit = str(taxed.get("excise_unit", context.get("excise_unit", "source unit")))
        source_frequency = "weekly"

    with np.errstate(divide="ignore", invalid="ignore"):
        level_impact_pct = 100.0 * (scenario_level / baseline_level - 1.0)
    yoy_impact_pp = scenario_yoy - baseline_yoy

    actual_hicp = actual_hicp.astype(float).sort_index()
    actual_hicp.index = pd.DatetimeIndex(pd.to_datetime(actual_hicp.index), name="date")
    return {
        "model_id": canonical,
        "hicp_series": _HICP_SERIES_BY_MODEL[canonical],
        "hicp_label": _HICP_LABEL_BY_MODEL[canonical],
        "path_dates": dates,
        "future_dates": future_dates,
        "actual_hicp": actual_hicp,
        "baseline_level_paths": baseline_level,
        "scenario_level_paths": scenario_level,
        "baseline_yoy_paths": baseline_yoy,
        "scenario_yoy_paths": scenario_yoy,
        "level_impact_pct_paths": level_impact_pct,
        "yoy_impact_pp_paths": yoy_impact_pp,
        "draw_indices": retained_indices,
        "n_draws_effective": int(baseline_level.shape[0]),
        "source_frequency": source_frequency,
        "tax_path": tax_path,
        "tax_history": tax_history,
        "raw_tax_publications": raw_tax,
        "scenario_start": pd.Timestamp(start),
        "tax_scenario": scenario_meta,
        "vat_delta_pp": vat_delta_pp,
        "excise_delta": excise_delta,
        "excise_unit": excise_unit,
        "rejection_rate": rejection_rate,
        "distribution_interpretation": distribution_interpretation,
    }


def build_saved_component_tax_scenario(
    run_directory: str | Path,
    *,
    project_root: str | Path | None = None,
    forecast_name: str = "unconditional",
    start_date,
    vat_delta_pp: float = 0.0,
    excise_delta: float = 0.0,
    max_draws: int | None = 500,
) -> dict:
    """Scenario wrapper for an immutable saved component forecast store."""
    import json
    from energy_bvar_io import load_energy_bvar_forecast, load_energy_bvar_history
    from energy_bvar_pipeline import build_panel, find_project_root, model_spec

    run_directory = Path(run_directory)
    metadata_path = run_directory / "metadata.json"
    if not metadata_path.is_file():
        raise ComponentHICPError(f"Run metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_id = str(metadata["model_id"])
    vintage = str(metadata["vintage"])
    forecast_dir = run_directory / "forecasts" / str(forecast_name)
    forecast = load_energy_bvar_forecast(forecast_dir)
    try:
        history = load_energy_bvar_history(run_directory)
    except FileNotFoundError:
        history = build_panel(model_id, vintage, project_root=project_root).levels.copy()
    result = _result_proxy(forecast, history)
    result["metadata"] = dict(metadata)
    root = find_project_root(project_root)
    dataset = root / "data" / "processed" / vintage / model_spec(model_id).dataset_file
    if not dataset.is_file():
        raise ComponentHICPError(
            f"Processed dataset required for tax scenario is missing: {dataset}"
        )
    return build_component_tax_scenario(
        model_id,
        forecast,
        result=result,
        dataset_path=dataset,
        project_root=root,
        start_date=start_date,
        vat_delta_pp=vat_delta_pp,
        excise_delta=excise_delta,
        max_draws=max_draws,
    )


__all__ = [
    "HICP_STORE_SCHEMA_VERSION",
    "ComponentHICPError",
    "build_component_hicp_forecast",
    "materialize_component_hicp_store",
    "TAX_SCENARIO_MODEL_IDS",
    "component_tax_scenario_contract",
    "build_component_tax_scenario",
    "build_saved_component_tax_scenario",
]
