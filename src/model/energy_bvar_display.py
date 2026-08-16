"""Compact, one-read display artefacts for the energy BVAR dashboard.

The econometric caches deliberately remain rich (``draws.npz`` and
``forecast_draws.npz``).  Dash should not reopen those large arrays every time a
user changes a fan width, variable or chart basis.  This module materialises the
small quantities needed for interactive display into exactly one Parquet file:
``display_v1.parquet``.

The file is a long-form table with a stable schema.  It can contain:

* observed history;
* posterior forecast/nowcast fans (5/16/50/84/95 percentiles);
* aggregate baseline/scenario/impact fans;
* draw-wise aggregate contribution fans;
* compact model summaries and diagnostics;
* metadata rows used by the UI.

No econometric object is re-estimated here.  Quantiles are always computed from
posterior paths, never by aggregating component medians.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from energy_bvar_pipeline import build_panel, find_project_root, model_contract, model_spec

DISPLAY_SCHEMA_VERSION = "1.0"
DISPLAY_FILENAME = "display_v1.parquet"
DISPLAY_QUANTILES = (0.05, 0.16, 0.50, 0.84, 0.95)
DISPLAY_QUANTILE_COLUMNS = ("q05", "q16", "q50", "q84", "q95")

_DISPLAY_COLUMNS = (
    "display_schema_version",
    "record_type",
    "scope",
    "model_id",
    "vintage",
    "run_id",
    "forecast_name",
    "basis",
    "metric",
    "series",
    "label",
    "unit",
    "date",
    "segment",
    "is_future",
    "value",
    "q05",
    "q16",
    "q50",
    "q84",
    "q95",
    "key",
    "text_value",
    "extra_value",
)


class DisplayError(RuntimeError):
    """Raised when the compact display contract cannot be produced or loaded."""


def _empty_rows(n: int) -> pd.DataFrame:
    frame = pd.DataFrame(index=range(n), columns=_DISPLAY_COLUMNS)
    frame["display_schema_version"] = DISPLAY_SCHEMA_VERSION
    return frame


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in _DISPLAY_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    out = out.loc[:, _DISPLAY_COLUMNS]
    out["display_schema_version"] = DISPLAY_SCHEMA_VERSION
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("value", *DISPLAY_QUANTILE_COLUMNS, "extra_value"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "is_future" in out:
        # Nullable Boolean keeps metadata rows null instead of coercing to True.
        out["is_future"] = out["is_future"].astype("boolean")
    return out


def _meta_rows(
    *,
    scope: str,
    model_id: str,
    vintage: str,
    run_id: str,
    forecast_name: str,
    values: Mapping,
) -> pd.DataFrame:
    rows = []
    for key, raw in values.items():
        numeric = np.nan
        text = None
        if raw is None:
            text = None
        elif isinstance(raw, (bool, np.bool_)):
            text = "true" if bool(raw) else "false"
        elif isinstance(raw, (int, float, np.integer, np.floating)) and np.isfinite(raw):
            numeric = float(raw)
            text = str(raw)
        elif isinstance(raw, (str, pd.Timestamp)):
            text = str(raw)
        else:
            text = json.dumps(raw, sort_keys=True, default=str)
        rows.append(
            {
                "record_type": "meta",
                "scope": scope,
                "model_id": model_id,
                "vintage": vintage,
                "run_id": run_id,
                "forecast_name": forecast_name,
                "key": str(key),
                "text_value": text,
                "value": numeric,
            }
        )
    return _normalise_frame(pd.DataFrame(rows))


def _history_rows(
    series: pd.Series,
    *,
    scope: str,
    model_id: str,
    vintage: str,
    run_id: str,
    forecast_name: str,
    metric: str,
    label: str,
    unit: str,
) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    dates = pd.DatetimeIndex(s.index[valid], name="date")
    values = s.loc[valid].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "record_type": "history",
            "scope": scope,
            "model_id": model_id,
            "vintage": vintage,
            "run_id": run_id,
            "forecast_name": forecast_name,
            "basis": "observed",
            "metric": metric,
            "series": str(series.name),
            "label": label,
            "unit": unit,
            "date": dates,
            "segment": "observed",
            "is_future": False,
            "value": values,
        }
    )
    return _normalise_frame(frame)


def _fan_rows(
    paths: np.ndarray,
    dates: Sequence[pd.Timestamp],
    series_names: Sequence[str],
    *,
    scope: str,
    model_id: str,
    vintage: str,
    run_id: str,
    forecast_name: str,
    basis: str,
    metric: str,
    labels: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    future_dates: Sequence[pd.Timestamp] | None = None,
    segment_for_nonfuture: str = "nowcast",
) -> pd.DataFrame:
    values = np.asarray(paths, dtype=float)
    dates = pd.DatetimeIndex(dates, name="date")
    names = [str(name) for name in series_names]
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3:
        raise DisplayError(
            f"{metric}/{basis}: expected paths with 2 or 3 dimensions, got {values.shape}."
        )
    if values.shape[1] != len(dates) or values.shape[2] != len(names):
        raise DisplayError(
            f"{metric}/{basis}: path shape {values.shape} is incompatible with "
            f"{len(dates)} dates and {len(names)} series."
        )
    if values.shape[0] < 1:
        raise DisplayError(f"{metric}/{basis}: no posterior draws are available.")

    quantiles = np.nanquantile(values, DISPLAY_QUANTILES, axis=0)
    finite_count = np.sum(np.isfinite(values), axis=0)
    finite_sum = np.nansum(values, axis=0)
    posterior_mean = np.divide(
        finite_sum,
        finite_count,
        out=np.full(finite_sum.shape, np.nan, dtype=float),
        where=finite_count > 0,
    )
    future_index = pd.DatetimeIndex([] if future_dates is None else future_dates)
    future_set = set(future_index)
    rows = []
    for j, name in enumerate(names):
        for t, date in enumerate(dates):
            is_future = date in future_set if len(future_index) else False
            rows.append(
                {
                    "record_type": "fan",
                    "scope": scope,
                    "model_id": model_id,
                    "vintage": vintage,
                    "run_id": run_id,
                    "forecast_name": forecast_name,
                    "basis": basis,
                    "metric": metric,
                    "series": name,
                    "label": (labels or {}).get(name, name),
                    "unit": (units or {}).get(name),
                    "date": date,
                    "segment": "forecast" if is_future else segment_for_nonfuture,
                    "is_future": bool(is_future),
                    "value": posterior_mean[t, j],
                    "q05": quantiles[0, t, j],
                    "q16": quantiles[1, t, j],
                    "q50": quantiles[2, t, j],
                    "q84": quantiles[3, t, j],
                    "q95": quantiles[4, t, j],
                }
            )
    return _normalise_frame(pd.DataFrame(rows))


def _parameter_rows(
    run_directory: Path,
    *,
    model_id: str,
    vintage: str,
    run_id: str,
    forecast_name: str,
) -> list[pd.DataFrame]:
    blocks: list[pd.DataFrame] = []
    summary_path = next(
        (path for path in (run_directory / "summary.parquet", run_directory / "summary.csv") if path.is_file()),
        None,
    )
    if summary_path is not None:
        try:
            summary = (
                pd.read_parquet(summary_path)
                if summary_path.suffix == ".parquet"
                else pd.read_csv(summary_path)
            )
            rows = []
            for _, row in summary.iterrows():
                rows.append(
                    {
                        "record_type": "parameter",
                        "scope": "component",
                        "model_id": model_id,
                        "vintage": vintage,
                        "run_id": run_id,
                        "forecast_name": forecast_name,
                        "basis": "posterior",
                        "metric": str(row.get("block", "parameter")),
                        "series": str(row.get("parameter", "")),
                        "value": row.get("posterior_mean", np.nan),
                        "q05": row.get("q05", np.nan),
                        "q16": row.get("q16", np.nan),
                        "q50": row.get("posterior_median", np.nan),
                        "q84": row.get("q84", np.nan),
                        "q95": row.get("q95", np.nan),
                        "extra_value": row.get("ESS", np.nan),
                    }
                )
            blocks.append(_normalise_frame(pd.DataFrame(rows)))
        except (ImportError, ModuleNotFoundError):
            # A Parquet summary can exist on a machine where the dashboard
            # environment lacks a Parquet engine. display_v1 itself will later
            # emit the clearer dependency error in _write_display.
            pass

    diagnostics_path = next(
        (path for path in (run_directory / "diagnostics.parquet", run_directory / "diagnostics.csv") if path.is_file()),
        None,
    )
    if diagnostics_path is not None:
        try:
            diagnostics = (
                pd.read_parquet(diagnostics_path)
                if diagnostics_path.suffix == ".parquet"
                else pd.read_csv(diagnostics_path)
            )
            numeric_candidates = [
                column
                for column in diagnostics.columns
                if column not in {"diagnostic_group", "parameter"}
                and pd.api.types.is_numeric_dtype(diagnostics[column])
            ]
            rows = []
            for _, row in diagnostics.iterrows():
                for column in numeric_candidates:
                    value = row.get(column)
                    if pd.isna(value):
                        continue
                    rows.append(
                        {
                            "record_type": "diagnostic",
                            "scope": "component",
                            "model_id": model_id,
                            "vintage": vintage,
                            "run_id": run_id,
                            "forecast_name": forecast_name,
                            "basis": str(row.get("diagnostic_group", "diagnostic")),
                            "metric": str(column),
                            "series": str(row.get("parameter", "")),
                            "value": value,
                        }
                    )
            if rows:
                blocks.append(_normalise_frame(pd.DataFrame(rows)))
        except (ImportError, ModuleNotFoundError):
            pass
    return blocks


def _source_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(path).resolve() for path in paths}, key=lambda p: str(p)):
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"|missing|")
            continue
        stat = path.stat()
        digest.update(f"|{stat.st_size}|{stat.st_mtime_ns}|".encode("utf-8"))
        # JSON metadata is small and identity-bearing: hash its content too.
        if path.suffix.lower() == ".json" and stat.st_size <= 5_000_000:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_display(frame: pd.DataFrame, destination: Path, *, overwrite: bool) -> Path:
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise DisplayError(
            f"Display artifact already exists at {destination}. Pass overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = _normalise_frame(frame)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
    except (ImportError, ModuleNotFoundError) as exc:
        if temporary.exists():
            temporary.unlink()
        raise DisplayError(
            "Writing display_v1.parquet requires a Parquet engine. Install "
            "'pyarrow' in the dashboard environment (and add it to requirements.txt)."
        ) from exc
    os.replace(temporary, destination)
    return destination


def _infer_project_root_from_run(run_directory: Path) -> Path:
    # Expected layout: <project>/results/<model>/<vintage>/<run_id>/
    candidate = run_directory.resolve()
    if len(candidate.parents) >= 4 and candidate.parents[2].name == "results":
        return candidate.parents[3]
    return find_project_root(candidate)


def _component_history(
    run_directory: Path,
    *,
    model_id: str,
    vintage: str,
    root: Path,
    variables: Sequence[str],
) -> tuple[pd.DataFrame, Path, str]:
    """Load observed history from the saved run, with a legacy fallback.

    New forecast/result stores persist ``history.parquet`` (or CSV) beside the
    run.  That is the preferred dashboard contract because it makes display
    materialisation independent of ``data/processed`` and of adapter call
    signatures.  Runs created before this contract are supported once through
    ``build_panel``; the resulting display artefact is then self-contained.
    """
    from energy_bvar_io import load_energy_bvar_history

    try:
        history = load_energy_bvar_history(run_directory)
        source = (
            run_directory / "history.parquet"
            if (run_directory / "history.parquet").is_file()
            else run_directory / "history.csv"
        )
        mode = "saved_run_history"
    except FileNotFoundError:
        panel = build_panel(model_id, vintage, project_root=root)
        history = panel.levels.copy()
        source = panel.dataset_path
        mode = "legacy_processed_fallback"

    missing = [name for name in variables if name not in history.columns]
    if missing:
        raise DisplayError(
            f"Observed history for {model_id!r} is missing {missing}. "
            "Re-save the forecast with the current energy_bvar_io.py or rebuild "
            "the processed vintage before materialising display_v1."
        )
    return history.loc[:, list(variables)].sort_index(), source, mode


def build_component_display(
    run_directory: str | Path,
    *,
    project_root: str | Path | None = None,
    forecast_name: str = "unconditional",
    destination: str | Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Build ``display_v1.parquet`` for one saved component run.

    The forecast store is the primary source.  Observed history comes from the
    lightweight run-level history artefact for new runs; only legacy runs fall
    back once to the processed model panel.
    """
    from energy_bvar_io import load_energy_bvar_forecast

    run_directory = Path(run_directory)
    metadata_path = run_directory / "metadata.json"
    if not metadata_path.is_file():
        raise DisplayError(f"Run metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {"model_id", "vintage", "run_id"}
    missing = required.difference(metadata)
    if missing:
        raise DisplayError(f"Run metadata is missing {sorted(missing)}.")

    model_id = str(metadata["model_id"])
    vintage = str(metadata["vintage"])
    run_id = str(metadata["run_id"])
    spec = model_spec(model_id)
    root = (
        _infer_project_root_from_run(run_directory)
        if project_root is None
        else Path(project_root).resolve()
    )

    forecast_dir = run_directory / "forecasts" / str(forecast_name)
    forecast = load_energy_bvar_forecast(forecast_dir)
    dates = pd.DatetimeIndex(forecast["path_dates"], name="date")
    future_dates = pd.DatetimeIndex(forecast["future_dates"], name="date")
    variables = [str(name) for name in forecast["variables"]]

    contract = model_contract(model_id, vintage, project_root=root)
    units = {name: str(contract.get("units", {}).get(name, "")) for name in variables}
    target = str(contract.get("target") or metadata.get("target") or variables[-1])
    labels = {name: name.replace("_", " ").title() for name in variables}

    history, history_source, history_mode = _component_history(
        run_directory,
        model_id=model_id,
        vintage=vintage,
        root=root,
        variables=variables,
    )

    blocks: list[pd.DataFrame] = []
    for variable in variables:
        blocks.append(
            _history_rows(
                history[variable].rename(variable),
                scope="component",
                model_id=model_id,
                vintage=vintage,
                run_id=run_id,
                forecast_name=str(forecast_name),
                metric="level",
                label=labels[variable],
                unit=units.get(variable, ""),
            )
        )

    blocks.append(
        _fan_rows(
            np.asarray(forecast["level_paths"], dtype=float),
            dates,
            variables,
            scope="component",
            model_id=model_id,
            vintage=vintage,
            run_id=run_id,
            forecast_name=str(forecast_name),
            basis="unconditional",
            metric="level",
            labels=labels,
            units=units,
            future_dates=future_dates,
        )
    )
    if "diff_paths" in forecast:
        change_units = {
            name: (f"Δ {units.get(name, '')}".strip()) for name in variables
        }
        blocks.append(
            _fan_rows(
                np.asarray(forecast["diff_paths"], dtype=float),
                dates,
                variables,
                scope="component",
                model_id=model_id,
                vintage=vintage,
                run_id=run_id,
                forecast_name=str(forecast_name),
                basis="unconditional",
                metric="absolute_change",
                labels=labels,
                units=change_units,
                future_dates=future_dates,
            )
        )

    # Every component now carries a dedicated monthly HICP predictive store.
    # New runs write it in run_component(); legacy runs are migrated once here
    # from the already-saved raw forecast (no Gibbs re-estimation).
    from energy_bvar_component_hicp import materialize_component_hicp_store
    from energy_bvar_io import load_energy_bvar_hicp_forecast

    hicp_metadata_path = forecast_dir / "hicp_metadata.json"
    hicp_draws_path = forecast_dir / "hicp_draws.npz"
    if not (hicp_metadata_path.is_file() and hicp_draws_path.is_file()):
        materialize_component_hicp_store(
            run_directory,
            project_root=root,
            forecast_name=str(forecast_name),
            overwrite=False,
        )
    hicp = load_energy_bvar_hicp_forecast(forecast_dir)
    hicp_series = str(hicp["hicp_series"])
    hicp_label = str(hicp.get("hicp_label") or hicp_series.replace("_", " ").title())
    hicp_dates = pd.DatetimeIndex(hicp["path_dates"], name="date")
    hicp_future_dates = pd.DatetimeIndex(hicp["future_dates"], name="date")
    actual_hicp = pd.Series(hicp["actual_hicp"], dtype=float).rename(hicp_series)
    actual_hicp.index = pd.DatetimeIndex(actual_hicp.index, name="date")

    blocks.append(
        _history_rows(
            actual_hicp,
            scope="component_hicp",
            model_id=model_id,
            vintage=vintage,
            run_id=run_id,
            forecast_name=str(forecast_name),
            metric="hicp_level",
            label=hicp_label,
            unit="HICP index",
        )
    )
    blocks.append(
        _fan_rows(
            np.asarray(hicp["hicp_level_paths"], dtype=float),
            hicp_dates,
            [hicp_series],
            scope="component_hicp",
            model_id=model_id,
            vintage=vintage,
            run_id=run_id,
            forecast_name=str(forecast_name),
            basis="hicp_baseline",
            metric="hicp_level",
            labels={hicp_series: hicp_label},
            units={hicp_series: "HICP index"},
            future_dates=hicp_future_dates,
        )
    )
    actual_hicp_yoy = (100.0 * (actual_hicp / actual_hicp.shift(12) - 1.0)).rename(hicp_series)
    blocks.append(
        _history_rows(
            actual_hicp_yoy,
            scope="component_hicp",
            model_id=model_id,
            vintage=vintage,
            run_id=run_id,
            forecast_name=str(forecast_name),
            metric="yoy",
            label=hicp_label,
            unit="% y/y",
        )
    )
    blocks.append(
        _fan_rows(
            np.asarray(hicp["hicp_yoy_paths"], dtype=float),
            hicp_dates,
            [hicp_series],
            scope="component_hicp",
            model_id=model_id,
            vintage=vintage,
            run_id=run_id,
            forecast_name=str(forecast_name),
            basis="hicp_baseline",
            metric="yoy",
            labels={hicp_series: hicp_label},
            units={hicp_series: "% y/y"},
            future_dates=hicp_future_dates,
        )
    )

    blocks.extend(
        _parameter_rows(
            run_directory,
            model_id=model_id,
            vintage=vintage,
            run_id=run_id,
            forecast_name=str(forecast_name),
        )
    )

    source_paths = [
        metadata_path,
        forecast_dir / "forecast_metadata.json",
        forecast_dir / "forecast_draws.npz",
        forecast_dir / "hicp_metadata.json",
        forecast_dir / "hicp_draws.npz",
        forecast_dir / "hicp_history.csv",
        history_source,
    ]
    fingerprint = _source_fingerprint(source_paths)
    meta_values = {
        "display_kind": "component",
        "source_fingerprint": fingerprint,
        "history_source": history_mode,
        "model_label": spec.label,
        "aggregate_key": spec.aggregate_key,
        "frequency": spec.frequency,
        "p": spec.p,
        "target": target,
        "variables": variables,
        "forecast_draws": int(np.asarray(forecast["level_paths"]).shape[0]),
        "forecast_horizon": int(forecast.get("H", len(future_dates))),
        "tail_length": int(forecast.get("tail_length", 0)),
        "future_start": None if len(future_dates) == 0 else future_dates.min(),
        "future_end": None if len(future_dates) == 0 else future_dates.max(),
        "has_gibbs_draws": (run_directory / "draws.npz").is_file(),
        "has_hicp_store": True,
        "hicp_series": hicp_series,
        "hicp_label": hicp_label,
        "hicp_frequency": str(hicp.get("frequency", "monthly")),
        "hicp_transformation_method": hicp.get("transformation_method"),
        "hicp_draws": int(np.asarray(hicp["hicp_level_paths"]).shape[0]),
        "hicp_rejection_rate": hicp.get("rejection_rate"),
        "hicp_distribution_interpretation": hicp.get("distribution_interpretation"),
        "missing_data_method": forecast.get(
            "missing_data_method", metadata.get("missing_data_method")
        ),
        "missing_data_approximation_used": forecast.get(
            "missing_data_approximation_used", metadata.get("missing_data_approximation_used")
        ),
        "code_version": metadata.get("code_version"),
        "config_hash": metadata.get("config_hash"),
        "data_hash": metadata.get("data_hash"),
    }
    blocks.append(
        _meta_rows(
            scope="component",
            model_id=model_id,
            vintage=vintage,
            run_id=run_id,
            forecast_name=str(forecast_name),
            values=meta_values,
        )
    )

    frame = pd.concat(blocks, ignore_index=True, sort=False)
    destination = forecast_dir / DISPLAY_FILENAME if destination is None else Path(destination)
    return _write_display(frame, destination, overwrite=overwrite)


def _aggregate_history(
    project_root: Path,
    vintage: str,
) -> tuple[pd.Series, pd.Series]:
    path = project_root / "data" / "processed" / vintage / "hicp_indices_monthly.csv"
    if not path.is_file():
        raise DisplayError(f"Aggregate HICP history not found: {path}")
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "hicp_energy" not in frame.columns:
        raise DisplayError(f"{path} does not contain 'hicp_energy'.")
    level = pd.to_numeric(frame["hicp_energy"], errors="coerce").rename("hicp_energy")
    yoy = (100.0 * (level / level.shift(12) - 1.0)).rename("hicp_energy")
    return level, yoy



def _aggregate_bvar_fitted_rows(
    aggregate_directory: Path,
    project_root: Path,
    vintage: str,
    *,
    run_id: str,
    forecast_name: str,
) -> tuple[list[pd.DataFrame], dict]:
    """True posterior BVAR in-sample fitted component and aggregate rows.

    Each component path is the posterior one-step conditional mean from the
    BVAR coefficient draws, transformed through the same HICP bridge as the
    corresponding forecast.  Petrol and diesel are first combined with the
    observed residual transport-fuel index, then the six model blocks are
    chain-linked with the production Laspeyres weights.

    If legacy component runs were saved without posterior coefficient draws,
    the aggregate display remains usable and surfaces an explicit availability
    flag instead of silently substituting the observed-component reconstruction.
    """
    try:
        from energy_bvar_fitted import (
            FittedUnavailableError,
            materialize_aggregate_bvar_fitted,
        )
        fitted = materialize_aggregate_bvar_fitted(
            aggregate_directory,
            project_root=project_root,
            max_draws=500,
            seed=2026,
        )
    except Exception as exc:
        # Import lazily and classify availability errors without making the
        # core aggregate display fail. The message is carried in meta rows.
        return [], {
            "bvar_fitted_available": False,
            "bvar_fitted_reason": str(exc),
        }

    dates = pd.DatetimeIndex(fitted["dates"], name="date")
    components = [str(name) for name in fitted["components"]]
    labels = {
        "car_fuels": "Car fuels",
        "liquid_fuels": "Liquid fuels",
        "gas": "Gas",
        "electricity": "Electricity",
        "heat_energy": "Heat energy",
        "solid_fuels": "Solid fuels",
    }
    blocks: list[pd.DataFrame] = []
    for metric, key, unit in (
        ("level", "component_level_paths", "HICP index"),
        ("yoy", "component_yoy_paths", "% y/y"),
    ):
        block = _fan_rows(
            np.asarray(fitted[key], dtype=float),
            dates,
            components,
            scope="aggregate_bvar_fitted",
            model_id="hicp_energy_aggregate",
            vintage=str(vintage),
            run_id=str(run_id),
            forecast_name=str(forecast_name),
            basis="posterior_in_sample_fit",
            metric=metric,
            labels=labels,
            units={name: unit for name in components},
            future_dates=[],
            segment_for_nonfuture="fit",
        )
        block["record_type"] = "bvar_fitted_component"
        blocks.append(block)

    for metric, key, unit in (
        ("level", "energy_level_paths", "HICP index"),
        ("yoy", "energy_yoy_paths", "% y/y"),
    ):
        block = _fan_rows(
            np.asarray(fitted[key], dtype=float),
            dates,
            ["hicp_energy"],
            scope="aggregate_bvar_fitted",
            model_id="hicp_energy_aggregate",
            vintage=str(vintage),
            run_id=str(run_id),
            forecast_name=str(forecast_name),
            basis="posterior_in_sample_fit_laspeyres",
            metric=metric,
            labels={"hicp_energy": "BVAR-fitted HICP Energy"},
            units={"hicp_energy": unit},
            future_dates=[],
            segment_for_nonfuture="fit",
        )
        block["record_type"] = "bvar_fitted_aggregate"
        blocks.append(block)

    return blocks, {
        "bvar_fitted_available": True,
        "bvar_fitted_reason": None,
        "bvar_fitted_draws": int(fitted.get("n_draws", 0)),
        "bvar_fitted_definition": fitted.get("fitted_definition"),
        "bvar_fitted_path_start": None if len(dates) == 0 else dates.min(),
        "bvar_fitted_path_end": None if len(dates) == 0 else dates.max(),
    }


def _aggregate_diagnostic_rows(
    metadata: Mapping,
    *,
    vintage: str,
    run_id: str,
    forecast_name: str,
) -> pd.DataFrame:
    rows = []

    def add(metric: str, series: str, value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        rows.append(
            {
                "record_type": "diagnostic",
                "scope": "aggregate",
                "model_id": "hicp_energy_aggregate",
                "vintage": vintage,
                "run_id": run_id,
                "forecast_name": forecast_name,
                "basis": "audit",
                "metric": metric,
                "series": series,
                "value": numeric,
            }
        )

    for name, value in dict(metadata.get("weekly_rejection_rates", {})).items():
        add("weekly_rejection_rate", str(name), value)
    for name, value in dict(metadata.get("tax_round_trip_max_errors", {})).items():
        add("tax_round_trip_max_error", str(name), value)
    for name, value in dict(metadata.get("historical_validation_max_errors", {})).items():
        add("historical_reconstruction_max_error", str(name), value)
    add("six_model_history_max_error", "hicp_energy", metadata.get("six_model_history_max_error"))
    add(
        "six_model_history_annual_reanchored_max_error",
        "hicp_energy",
        metadata.get("six_model_history_annual_reanchored_max_error"),
    )
    add(
        "drawwise_contribution_additivity_error",
        "hicp_energy",
        metadata.get("maximum_drawwise_contribution_additivity_error"),
    )
    return _normalise_frame(pd.DataFrame(rows)) if rows else _normalise_frame(pd.DataFrame())


def build_aggregate_display(
    aggregate_directory: str | Path,
    *,
    project_root: str | Path | None = None,
    destination: str | Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Build ``display_v1.parquet`` for one saved HICP Energy aggregate run."""
    aggregate_directory = Path(aggregate_directory)
    metadata_path = aggregate_directory / "metadata.json"
    draws_path = aggregate_directory / "aggregate_draws.npz"
    if not metadata_path.is_file() or not draws_path.is_file():
        raise DisplayError(
            f"Aggregate store requires metadata.json and aggregate_draws.npz: {aggregate_directory}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vintage = str(metadata.get("vintage", aggregate_directory.parent.name))
    run_id = str(metadata.get("aggregate_run_id", aggregate_directory.name))
    forecast_name = str(metadata.get("forecast_name", "unconditional"))

    if project_root is None:
        # <project>/results/hicp_energy_aggregate/<vintage>/<aggregate_run_id>/
        resolved = aggregate_directory.resolve()
        root = resolved.parents[3] if len(resolved.parents) >= 4 and resolved.parents[2].name == "results" else find_project_root(resolved)
    else:
        root = Path(project_root).resolve()

    with np.load(draws_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "baseline_level_paths",
        "baseline_yoy_paths",
        "baseline_contribution_paths",
        "scenario_level_paths",
        "scenario_yoy_paths",
        "scenario_contribution_paths",
        "scenario_impact_yoy_paths",
    }
    missing = required.difference(arrays)
    if missing:
        raise DisplayError(f"Aggregate draws cache is missing {sorted(missing)}.")

    # Notebook 10 stores the calendar in metadata.json under "path_dates" and
    # does not write it into the npz; run_aggregate writes both. Accept either
    # so stores from both producers load.
    if "aggregate_dates" in arrays:
        dates = pd.DatetimeIndex(pd.to_datetime(arrays["aggregate_dates"]), name="date")
    elif metadata.get("path_dates"):
        dates = pd.DatetimeIndex(pd.to_datetime(metadata["path_dates"]), name="date")
    else:
        raise DisplayError(
            "Aggregate store carries no path dates: neither aggregate_dates in "
            f"aggregate_draws.npz nor path_dates in metadata.json ({aggregate_directory})."
        )

    n_periods = int(np.asarray(arrays["baseline_level_paths"]).shape[1])
    if len(dates) != n_periods:
        # Silently misaligning dates and paths would produce a plausible but
        # wrong chart, which is worse than refusing to draw one.
        raise DisplayError(
            f"Aggregate path dates ({len(dates)}) do not match the level paths "
            f"({n_periods} periods) in {aggregate_directory}."
        )
    components = [str(name) for name in metadata.get("components", [])]
    if not components:
        n_components = np.asarray(arrays["baseline_contribution_paths"]).shape[-1]
        components = [f"component_{i+1}" for i in range(n_components)]
    if len(components) != np.asarray(arrays["baseline_contribution_paths"]).shape[-1]:
        raise DisplayError("Aggregate metadata component names do not match contribution paths.")

    level_history, yoy_history = _aggregate_history(root, vintage)
    last_observed = level_history.dropna().index.max()
    future_dates = dates[dates > last_observed]

    labels = {"hicp_energy": "HICP Energy"}
    units_level = {"hicp_energy": "HICP index"}
    units_yoy = {"hicp_energy": "% y/y"}
    component_labels = {name: name.replace("_", " ").title() for name in components}
    component_units = {name: "percentage points" for name in components}

    blocks: list[pd.DataFrame] = [
        _history_rows(
            level_history,
            scope="aggregate",
            model_id="hicp_energy_aggregate",
            vintage=vintage,
            run_id=run_id,
            forecast_name=forecast_name,
            metric="level",
            label="HICP Energy",
            unit="HICP index",
        ),
        _history_rows(
            yoy_history,
            scope="aggregate",
            model_id="hicp_energy_aggregate",
            vintage=vintage,
            run_id=run_id,
            forecast_name=forecast_name,
            metric="yoy",
            label="HICP Energy",
            unit="% y/y",
        ),
    ]

    # Optional true posterior in-sample BVAR fitted overlays. They are cached
    # beside the aggregate run and summarised into display_v1. Legacy runs that
    # lack coefficient draws remain fully readable; the UI receives an explicit
    # availability reason instead of a misleading observed reconstruction.
    fitted_blocks, fitted_meta = _aggregate_bvar_fitted_rows(
        aggregate_directory, root, vintage, run_id=run_id, forecast_name=forecast_name
    )
    blocks.extend(fitted_blocks)

    for basis, level_key, yoy_key in (
        ("baseline", "baseline_level_paths", "baseline_yoy_paths"),
        ("scenario", "scenario_level_paths", "scenario_yoy_paths"),
    ):
        blocks.append(
            _fan_rows(
                arrays[level_key],
                dates,
                ["hicp_energy"],
                scope="aggregate",
                model_id="hicp_energy_aggregate",
                vintage=vintage,
                run_id=run_id,
                forecast_name=forecast_name,
                basis=basis,
                metric="level",
                labels=labels,
                units=units_level,
                future_dates=future_dates,
            )
        )
        blocks.append(
            _fan_rows(
                arrays[yoy_key],
                dates,
                ["hicp_energy"],
                scope="aggregate",
                model_id="hicp_energy_aggregate",
                vintage=vintage,
                run_id=run_id,
                forecast_name=forecast_name,
                basis=basis,
                metric="yoy",
                labels=labels,
                units=units_yoy,
                future_dates=future_dates,
            )
        )

    blocks.append(
        _fan_rows(
            arrays["scenario_impact_yoy_paths"],
            dates,
            ["hicp_energy"],
            scope="aggregate",
            model_id="hicp_energy_aggregate",
            vintage=vintage,
            run_id=run_id,
            forecast_name=forecast_name,
            basis="scenario_minus_baseline",
            metric="yoy_impact",
            labels=labels,
            units={"hicp_energy": "percentage points"},
            future_dates=future_dates,
        )
    )

    for basis, key in (
        ("baseline", "baseline_contribution_paths"),
        ("scenario", "scenario_contribution_paths"),
    ):
        block = _fan_rows(
            arrays[key],
            dates,
            components,
            scope="aggregate",
            model_id="hicp_energy_aggregate",
            vintage=vintage,
            run_id=run_id,
            forecast_name=forecast_name,
            basis=basis,
            metric="contribution_yoy",
            labels=component_labels,
            units=component_units,
            future_dates=future_dates,
        )
        block["record_type"] = "contribution"
        blocks.append(block)

    diagnostics = _aggregate_diagnostic_rows(
        metadata,
        vintage=vintage,
        run_id=run_id,
        forecast_name=forecast_name,
    )
    if not diagnostics.empty:
        blocks.append(diagnostics)

    source_paths = [
        metadata_path,
        draws_path,
        root / "data" / "processed" / vintage / "hicp_indices_monthly.csv",
        root / "data" / "processed" / vintage / "hicp_weights_annual.csv",
    ]
    fingerprint = _source_fingerprint(source_paths)
    meta_values = {
        "display_kind": "aggregate",
        "source_fingerprint": fingerprint,
        "aggregate_run_id": run_id,
        "forecast_name": forecast_name,
        "n_aggregate_draws_requested": metadata.get("n_aggregate_draws_requested"),
        "n_aggregate_draws_effective": metadata.get("n_aggregate_draws_effective"),
        "weekly_tax_mode_effective": metadata.get("weekly_tax_mode_effective"),
        "scenario_active": metadata.get("scenario_active", False),
        "components": components,
        "path_start": None if len(dates) == 0 else dates.min(),
        "path_end": None if len(dates) == 0 else dates.max(),
        "last_observed_hicp_energy": last_observed,
        "aggregate_module_version": metadata.get("aggregate_module_version"),
        "predictive_distribution_interpretation": metadata.get("predictive_distribution_interpretation"),
        "cross_model_dependence": metadata.get("cross_model_dependence"),
        **fitted_meta,
    }
    blocks.append(
        _meta_rows(
            scope="aggregate",
            model_id="hicp_energy_aggregate",
            vintage=vintage,
            run_id=run_id,
            forecast_name=forecast_name,
            values=meta_values,
        )
    )

    frame = pd.concat(blocks, ignore_index=True, sort=False)
    destination = aggregate_directory / DISPLAY_FILENAME if destination is None else Path(destination)
    return _write_display(frame, destination, overwrite=overwrite)


def build_display_artifact(
    directory: str | Path,
    *,
    project_root: str | Path | None = None,
    forecast_name: str = "unconditional",
    destination: str | Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Auto-detect a component or aggregate store and build its display file."""
    directory = Path(directory)
    if (directory / "aggregate_draws.npz").is_file():
        return build_aggregate_display(
            directory,
            project_root=project_root,
            destination=destination,
            overwrite=overwrite,
        )
    return build_component_display(
        directory,
        project_root=project_root,
        forecast_name=forecast_name,
        destination=destination,
        overwrite=overwrite,
    )


def load_display_artifact(path_or_directory: str | Path) -> pd.DataFrame:
    """Read and validate one compact dashboard artifact with one filesystem I/O."""
    path = Path(path_or_directory)
    if path.is_dir():
        path = path / DISPLAY_FILENAME
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_parquet(path)
    except (ImportError, ModuleNotFoundError) as exc:
        raise DisplayError(
            "Reading display_v1.parquet requires 'pyarrow' in the dashboard environment."
        ) from exc
    missing = set(_DISPLAY_COLUMNS).difference(frame.columns)
    if missing:
        raise DisplayError(f"Display artifact is missing columns {sorted(missing)}.")
    versions = set(frame["display_schema_version"].dropna().astype(str).unique())
    if versions != {DISPLAY_SCHEMA_VERSION}:
        raise DisplayError(
            f"Unsupported display schema version(s) {sorted(versions)}; expected {DISPLAY_SCHEMA_VERSION}."
        )
    return _normalise_frame(frame)


def display_metadata(frame: pd.DataFrame) -> dict[str, str | float]:
    """Extract ``record_type='meta'`` rows from an already-loaded display frame."""
    meta = frame.loc[frame["record_type"] == "meta"]
    out: dict[str, str | float] = {}
    for _, row in meta.iterrows():
        key = str(row["key"])
        text = row.get("text_value")
        value = row.get("value")
        out[key] = text if pd.notna(text) else (float(value) if pd.notna(value) else None)
    return out


__all__ = [
    "DISPLAY_SCHEMA_VERSION",
    "DISPLAY_FILENAME",
    "DISPLAY_QUANTILES",
    "DisplayError",
    "build_component_display",
    "build_aggregate_display",
    "build_display_artifact",
    "load_display_artifact",
    "display_metadata",
]
