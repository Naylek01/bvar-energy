"""Compact display artefact for the locked Headline-HICP dashboard.

The dashboard reads one ``display_v1.parquet`` file per selected Headline run.
No plotting callback reopens ``draws.npz`` or recomputes the chain link.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from headline_bvar_diagnostics import saved_headline_diagnostic_rows
from headline_bvar_structural import STRUCTURAL_FILENAME
from headline_bvar_pipeline import (
    MAX_PUBLISHED_HORIZON_MONTHS,
    NATIVE_VARIABLES,
    model_contract,
    reconstruct_headline_component_history,
)
from headline_core import (
    CORE_COMPONENTS,
    CORE_LABEL,
    CORE_LONG_LABEL,
    CORE_SERIES,
    EUROSTAT_CORE_CODE,
    aggregate_core_draws,
    core_reconstruction_diagnostics,
    historical_core_yoy_contributions,
    load_or_fetch_official_core,
    persist_core_reconstruction_diagnostics,
    reconstruct_core_history,
)

DISPLAY_FILENAME = "display_v1.parquet"
DISPLAY_SCHEMA_VERSION = "headline-1.0"
QUANTILES = (0.05, 0.16, 0.50, 0.84, 0.95)
QCOLS = ("q05", "q16", "q50", "q84", "q95")

LABELS = {
    "hicp_total": "Headline HICP",
    "hicp_energy": "Energy",
    "hicp_food": "Food",
    "hicp_neig": "NEIG",
    "hicp_services": "Services",
    "hicp_core": "Core HICP",
}

_COLUMNS = (
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
    "response",
    "shock",
    "horizon",
    "posterior_mean",
    "posterior_sd",
    "ess",
    "mcse_mean",
    "mcse_over_sd",
    "prob_90",
    "prob_95",
    "prob_97",
)


class HeadlineDisplayError(RuntimeError):
    pass


def _rows(n: int) -> pd.DataFrame:
    out = pd.DataFrame(index=range(n), columns=_COLUMNS)
    out["display_schema_version"] = DISPLAY_SCHEMA_VERSION
    return out


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in _COLUMNS:
        if col not in out:
            out[col] = np.nan
    out = out.loc[:, _COLUMNS]
    out["display_schema_version"] = DISPLAY_SCHEMA_VERSION
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in (
        "value", *QCOLS, "extra_value", "horizon", "posterior_mean",
        "posterior_sd", "ess", "mcse_mean", "mcse_over_sd",
        "prob_90", "prob_95", "prob_97",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["is_future"] = out["is_future"].astype("boolean")
    return out


def _quantile_frame(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    series: str,
    metric: str,
    unit: str,
    vintage: str,
    run_id: str,
    forecast_name: str,
    tail_length: int,
    scope: str = "headline",
) -> pd.DataFrame:
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(dates):
        raise HeadlineDisplayError(
            f"{series}/{metric}: expected paths (draws, dates), got {arr.shape} "
            f"for {len(dates)} dates."
        )
    q = np.nanquantile(arr, QUANTILES, axis=0).T
    out = _rows(len(dates))
    out["record_type"] = "fan"
    out["scope"] = scope
    out["model_id"] = "headline_joint"
    out["vintage"] = vintage
    out["run_id"] = run_id
    out["forecast_name"] = forecast_name
    out["basis"] = "baseline"
    out["metric"] = metric
    out["series"] = series
    out["label"] = LABELS.get(series, series)
    out["unit"] = unit
    out["date"] = dates
    out["segment"] = [
        "nowcast" if i < tail_length else "forecast"
        for i in range(len(dates))
    ]
    out["is_future"] = np.arange(len(dates)) >= int(tail_length)
    out["value"] = np.nanmean(arr, axis=0)
    for j, col in enumerate(QCOLS):
        out[col] = q[:, j]
    return out


def _history_rows(
    series: pd.Series,
    *,
    variable: str,
    metric: str,
    unit: str,
    vintage: str,
    run_id: str,
    forecast_name: str,
) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    out = _rows(len(s))
    out["record_type"] = "history"
    out["scope"] = "headline"
    out["model_id"] = "headline_joint"
    out["vintage"] = vintage
    out["run_id"] = run_id
    out["forecast_name"] = forecast_name
    out["basis"] = "baseline"
    out["metric"] = metric
    out["series"] = variable
    out["label"] = LABELS.get(variable, variable)
    out["unit"] = unit
    out["date"] = s.index
    out["segment"] = "observed"
    out["is_future"] = False
    out["value"] = s.to_numpy(dtype=float)
    return out



def historical_headline_contribution_rows(
    native_history: pd.DataFrame,
    weights: pd.DataFrame,
    official_total: pd.Series,
    *,
    vintage: str,
    run_id: str,
    forecast_name: str,
) -> pd.DataFrame:
    """Exact observed pp contributions to official Headline HICP YoY.

    The chain-linked historical component index contributions are converted to
    component shares and rescaled to the official Headline Total at each date.
    Current and 12-month-lag official-rescaled component levels therefore sum
    exactly to the official current and lag totals, so the four pp
    contributions sum exactly to official Headline year-on-year inflation.
    """
    native = native_history[NATIVE_VARIABLES].copy().sort_index()
    native.index = pd.DatetimeIndex(native.index).to_period("M").to_timestamp(how="start")

    w = weights[NATIVE_VARIABLES].copy()
    w.index = pd.Index(pd.to_numeric(w.index, errors="raise").astype(int), name=w.index.name)

    official = pd.to_numeric(official_total, errors="coerce").dropna().sort_index()
    official.index = pd.DatetimeIndex(official.index).to_period("M").to_timestamp(how="start")

    components = reconstruct_headline_component_history(native, w, official)
    reconstructed = components.sum(axis=1)
    if components.empty or (reconstructed <= 0).any():
        raise HeadlineDisplayError("Historical Headline component reconstruction is invalid.")

    shares = components.div(reconstructed, axis=0)
    scaled = shares.mul(official.reindex(components.index), axis=0)

    lagged_scaled = scaled.copy()
    lagged_scaled.index = lagged_scaled.index + pd.DateOffset(months=12)
    lagged_scaled = lagged_scaled.reindex(scaled.index)

    lagged_official = official.copy()
    lagged_official.index = lagged_official.index + pd.DateOffset(months=12)
    lagged_official = lagged_official.reindex(scaled.index)

    contribution = 100.0 * (scaled - lagged_scaled).div(lagged_official, axis=0)
    contribution = contribution.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if contribution.empty:
        raise HeadlineDisplayError(
            "No exact historical Headline YoY contribution dates are available."
        )

    target = 100.0 * (
        official.reindex(contribution.index)
        / lagged_official.reindex(contribution.index)
        - 1.0
    )
    additivity_error = float(
        np.max(
            np.abs(
                contribution.sum(axis=1).to_numpy(dtype=float)
                - target.to_numpy(dtype=float)
            )
        )
    )
    if not np.isfinite(additivity_error) or additivity_error > 1e-10:
        raise HeadlineDisplayError(
            "Historical Headline contribution additivity failed: "
            f"{additivity_error:.3e} pp."
        )

    blocks = []
    for variable in NATIVE_VARIABLES:
        block = _history_rows(
            contribution[variable],
            variable=variable,
            metric="yoy_contribution",
            unit="pp",
            vintage=str(vintage),
            run_id=str(run_id),
            forecast_name=forecast_name,
        )
        block["record_type"] = "contribution_history"
        blocks.append(block)

    return _normalise(pd.concat(blocks, ignore_index=True, sort=False))


def _align_to_existing_display_schema(
    new_rows: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    """Cast append-only rows to the exact schema of an accepted display frame."""
    aligned = new_rows.reindex(columns=existing.columns).copy()
    for col in existing.columns:
        try:
            aligned[col] = pd.array(aligned[col], dtype=existing[col].dtype)
        except (TypeError, ValueError) as exc:
            raise HeadlineDisplayError(
                f"Could not align historical contribution column {col!r} "
                f"to existing dtype {existing[col].dtype}: {exc}"
            ) from exc
    return aligned


def _assert_preexisting_rows_exact(
    expected: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    context: str,
) -> None:
    """Hard append-only guard for every row that predates Delivery 7."""
    if len(expected) != len(candidate):
        raise HeadlineDisplayError(
            f"{context}: pre-existing row count changed "
            f"({len(expected)} -> {len(candidate)})."
        )
    try:
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True),
            candidate.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as exc:
        raise HeadlineDisplayError(
            f"{context}: pre-existing display rows changed."
        ) from exc


def materialize_headline_historical_contributions(
    run_directory: str | Path,
    *,
    forecast_name: str = "unconditional",
) -> Path:
    """Append/refresh exact historical contribution rows transactionally.

    The accepted ``display_v1.parquet`` is read raw. Existing rows are never
    normalised before the append. A temporary parquet is written and re-read
    first; the live display is atomically replaced only if every pre-existing
    row is still exactly equal, including dtypes.
    """
    run_dir = Path(run_directory)
    fc_dir = run_dir / "forecasts" / forecast_name
    display_path = fc_dir / DISPLAY_FILENAME
    meta_path = fc_dir / "headline_metadata.json"
    native_history_path = run_dir / "headline_native_history.csv"
    weights_path = run_dir / "headline_weights_annual.csv"
    official_path = run_dir / "headline_official_total.csv"

    for path in (
        display_path,
        meta_path,
        native_history_path,
        weights_path,
        official_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Raw read is essential. load_headline_display() deliberately normalises
    # metadata for dashboard consumption and therefore cannot satisfy an
    # append-only persistence contract.
    frame = pd.read_parquet(display_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    native_history = pd.read_csv(
        native_history_path,
        index_col=0,
        parse_dates=True,
    ).sort_index()
    weights = pd.read_csv(weights_path, index_col=0)
    official_total = pd.read_csv(
        official_path,
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0].sort_index()

    historical = historical_headline_contribution_rows(
        native_history,
        weights,
        official_total,
        vintage=str(meta["vintage"]),
        run_id=str(meta["run_id"]),
        forecast_name=forecast_name,
    )

    # Idempotent refresh: only rows owned by this materializer may be replaced.
    old = frame.loc[
        ~frame["record_type"].astype(str).eq("contribution_history")
    ].copy()
    historical_aligned = _align_to_existing_display_schema(historical, old)

    combined = pd.concat(
        [old, historical_aligned],
        ignore_index=True,
        sort=False,
    )
    _assert_preexisting_rows_exact(
        old,
        combined.iloc[: len(old)],
        context="pre-write append guard",
    )

    temporary = display_path.with_name(display_path.name + ".delivery7.tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        combined.to_parquet(temporary, index=False, compression="zstd")

        # Validate the actual pandas/pyarrow round trip before the accepted
        # artifact can be replaced.
        roundtrip = pd.read_parquet(temporary)
        roundtrip_old = roundtrip.loc[
            ~roundtrip["record_type"].astype(str).eq("contribution_history")
        ].copy()
        _assert_preexisting_rows_exact(
            old,
            roundtrip_old,
            context="parquet round-trip guard",
        )

        roundtrip_hist = roundtrip.loc[
            roundtrip["record_type"].astype(str).eq("contribution_history")
        ]
        if len(roundtrip_hist) != len(historical_aligned):
            raise HeadlineDisplayError(
                "Parquet round-trip changed the historical contribution row count."
            )

        os.replace(temporary, display_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return display_path


def _core_error_rows(
    message: str,
    *,
    meta: dict,
    forecast_name: str,
) -> pd.DataFrame:
    rows = _rows(1)
    rows["record_type"] = "core_status"
    rows["scope"] = "headline"
    rows["model_id"] = "headline_joint"
    rows["vintage"] = str(meta["vintage"])
    rows["run_id"] = str(meta["run_id"])
    rows["forecast_name"] = forecast_name
    rows["basis"] = "baseline"
    rows["series"] = CORE_SERIES
    rows["label"] = CORE_LONG_LABEL
    rows["key"] = "core_materialization_error"
    rows["text_value"] = str(message)
    return rows


def _core_diagnostic_rows(
    diagnostics: pd.Series,
    aggregate_core: dict,
    *,
    meta: dict,
    forecast_name: str,
) -> pd.DataFrame:
    """Compact Core reconstruction/aggregation diagnostics for the dashboard."""
    values = {
        "core_official_code": EUROSTAT_CORE_CODE,
        "core_reconstruction_level_mae": diagnostics.get("level_mae"),
        "core_reconstruction_level_max_abs_error": diagnostics.get(
            "level_max_abs_error"
        ),
        "core_reconstruction_yoy_mae_pp": diagnostics.get("yoy_mae_pp"),
        "core_reconstruction_yoy_max_abs_error_pp": diagnostics.get(
            "yoy_max_abs_error_pp"
        ),
        "core_drawwise_level_additivity_max_abs_error": aggregate_core.get(
            "core_drawwise_level_additivity_max_abs_error"
        ),
        "core_yoy_contribution_additivity_max_abs_error": aggregate_core.get(
            "core_yoy_contribution_additivity_max_abs_error"
        ),
    }
    rows = _rows(len(values))
    rows["record_type"] = "core_diagnostic"
    rows["scope"] = "headline"
    rows["model_id"] = "headline_joint"
    rows["vintage"] = str(meta["vintage"])
    rows["run_id"] = str(meta["run_id"])
    rows["forecast_name"] = forecast_name
    rows["basis"] = "baseline"
    rows["series"] = CORE_SERIES
    rows["label"] = CORE_LONG_LABEL
    rows["key"] = list(values)

    text_values = []
    numeric_values = []
    for value in values.values():
        if isinstance(value, str):
            text_values.append(value)
            numeric_values.append(np.nan)
        else:
            text_values.append(np.nan)
            numeric_values.append(value)
    rows["text_value"] = text_values
    rows["extra_value"] = numeric_values
    return rows


def _core_display_rows(
    *,
    run_dir: Path,
    native_history: pd.DataFrame,
    weights: pd.DataFrame,
    native_paths: np.ndarray,
    dates: pd.DatetimeIndex,
    tail: int,
    meta: dict,
    forecast_name: str,
) -> list[pd.DataFrame]:
    """Build official history, draw-wise fans and exact Core contributions."""
    observed_end = pd.Timestamp(native_history.index.max())
    official_core, _source_meta = load_or_fetch_official_core(
        run_dir,
        native_history=native_history,
        observed_end=observed_end,
    )

    reconstructed = reconstruct_core_history(
        native_history.loc[:, list(CORE_COMPONENTS)],
        weights,
        anchor_series=official_core,
    )
    diagnostics = core_reconstruction_diagnostics(
        reconstructed,
        official_core,
    )
    persist_core_reconstruction_diagnostics(
        run_dir,
        reconstructed=reconstructed,
        diagnostics=diagnostics,
    )

    aggregate_core = aggregate_core_draws(
        native_paths,
        dates,
        native_variables=NATIVE_VARIABLES,
        native_history=native_history,
        annual_weights=weights,
        official_core=official_core,
    )

    core_level = np.asarray(
        aggregate_core["core_level_paths"], dtype=float
    )
    core_yoy = np.asarray(
        aggregate_core["core_yoy_paths"], dtype=float
    )
    core_contrib = np.asarray(
        aggregate_core["core_yoy_contribution_paths"], dtype=float
    )

    if core_level.shape != native_paths.shape[:2]:
        raise HeadlineDisplayError("Core level path dimensions are invalid.")
    if core_yoy.shape != core_level.shape:
        raise HeadlineDisplayError("Core YoY path dimensions are invalid.")
    if core_contrib.shape != (
        native_paths.shape[0],
        native_paths.shape[1],
        len(CORE_COMPONENTS),
    ):
        raise HeadlineDisplayError(
            "Core YoY contribution path dimensions are invalid."
        )

    frames: list[pd.DataFrame] = []
    frames.append(
        _history_rows(
            official_core,
            variable=CORE_SERIES,
            metric="level",
            unit="HICP index",
            vintage=str(meta["vintage"]),
            run_id=str(meta["run_id"]),
            forecast_name=forecast_name,
        )
    )
    frames.append(
        _history_rows(
            100.0 * (official_core / official_core.shift(12) - 1.0),
            variable=CORE_SERIES,
            metric="yoy",
            unit="% y/y",
            vintage=str(meta["vintage"]),
            run_id=str(meta["run_id"]),
            forecast_name=forecast_name,
        )
    )
    frames.append(
        _quantile_frame(
            core_level,
            dates,
            series=CORE_SERIES,
            metric="level",
            unit="HICP index",
            vintage=str(meta["vintage"]),
            run_id=str(meta["run_id"]),
            forecast_name=forecast_name,
            tail_length=tail,
        )
    )
    frames.append(
        _quantile_frame(
            core_yoy,
            dates,
            series=CORE_SERIES,
            metric="yoy",
            unit="% y/y",
            vintage=str(meta["vintage"]),
            run_id=str(meta["run_id"]),
            forecast_name=forecast_name,
            tail_length=tail,
        )
    )

    historical_contrib = historical_core_yoy_contributions(
        native_history.loc[:, list(CORE_COMPONENTS)],
        weights,
        official_core,
    )
    for variable in CORE_COMPONENTS:
        block = _history_rows(
            historical_contrib[variable],
            variable=variable,
            metric="core_yoy_contribution",
            unit="pp",
            vintage=str(meta["vintage"]),
            run_id=str(meta["run_id"]),
            forecast_name=forecast_name,
        )
        block["record_type"] = "core_contribution_history"
        frames.append(block)

    for j, variable in enumerate(CORE_COMPONENTS):
        block = _quantile_frame(
            core_contrib[:, :, j],
            dates,
            series=variable,
            metric="core_yoy_contribution",
            unit="pp",
            vintage=str(meta["vintage"]),
            run_id=str(meta["run_id"]),
            forecast_name=forecast_name,
            tail_length=tail,
        )
        block["record_type"] = "core_contribution"
        frames.append(block)

    frames.append(
        _core_diagnostic_rows(
            diagnostics,
            aggregate_core,
            meta=meta,
            forecast_name=forecast_name,
        )
    )
    return frames



def _core_owned_mask(frame: pd.DataFrame) -> pd.Series:
    record = frame["record_type"].astype(str)
    series = frame["series"].astype(str)
    return (
        record.isin(
            [
                "core_contribution_history",
                "core_contribution",
                "core_diagnostic",
                "core_status",
            ]
        )
        | (
            series.eq(CORE_SERIES)
            & record.isin(["history", "fan"])
        )
    )


def materialize_headline_core(
    run_directory: str | Path,
    *,
    forecast_name: str = "unconditional",
) -> Path:
    """Append/refresh Core rows without changing any pre-existing non-Core row.

    This is the migration path for already-saved Headline runs.  It preserves
    fitted rows, structural rows, diagnostics and every legacy display dtype.
    """
    run_dir = Path(run_directory)
    fc_dir = run_dir / "forecasts" / forecast_name
    display_path = fc_dir / DISPLAY_FILENAME
    meta_path = fc_dir / "headline_metadata.json"
    draws_path = fc_dir / "headline_draws.npz"
    native_history_path = run_dir / "headline_native_history.csv"
    weights_path = run_dir / "headline_weights_annual.csv"

    for path in (
        display_path,
        meta_path,
        draws_path,
        native_history_path,
        weights_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    existing = pd.read_parquet(display_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dates = pd.DatetimeIndex(pd.to_datetime(meta["path_dates"]), name="date")
    tail = int(meta["tail_length"])

    with np.load(draws_path, allow_pickle=False) as archive:
        if "native_level_paths" not in archive.files:
            raise HeadlineDisplayError(
                "Saved Headline forecast has no native_level_paths."
            )
        native_paths = np.asarray(
            archive["native_level_paths"], dtype=float
        )

    if native_paths.shape[1:] != (len(dates), len(NATIVE_VARIABLES)):
        raise HeadlineDisplayError(
            "Saved native Headline paths are incompatible with Core materialisation."
        )

    native_history = pd.read_csv(
        native_history_path,
        index_col=0,
        parse_dates=True,
    ).sort_index()
    native_history.index = pd.DatetimeIndex(native_history.index, name="date")
    weights = pd.read_csv(weights_path, index_col=0)

    new_core = _normalise(
        pd.concat(
            _core_display_rows(
                run_dir=run_dir,
                native_history=native_history,
                weights=weights,
                native_paths=native_paths,
                dates=dates,
                tail=tail,
                meta=meta,
                forecast_name=forecast_name,
            ),
            ignore_index=True,
            sort=False,
        )
    )

    old = existing.loc[~_core_owned_mask(existing)].copy()
    core_aligned = _align_to_existing_display_schema(new_core, old)
    combined = pd.concat(
        [old, core_aligned],
        ignore_index=True,
        sort=False,
    )

    _assert_preexisting_rows_exact(
        old,
        combined.iloc[: len(old)],
        context="Core pre-write append guard",
    )

    temporary = display_path.with_name(
        display_path.name + ".core_materialize.tmp"
    )
    if temporary.exists():
        temporary.unlink()
    try:
        combined.to_parquet(temporary, index=False, compression="zstd")
        roundtrip = pd.read_parquet(temporary)
        roundtrip_old = roundtrip.iloc[: len(old)].copy()
        _assert_preexisting_rows_exact(
            old,
            roundtrip_old,
            context="Core parquet round-trip guard",
        )
        roundtrip_core = roundtrip.iloc[len(old):]
        if len(roundtrip_core) != len(core_aligned):
            raise HeadlineDisplayError(
                "Core parquet round-trip changed the appended row count."
            )
        if _core_owned_mask(roundtrip_core).sum() != len(roundtrip_core):
            raise HeadlineDisplayError(
                "Core materializer produced rows outside its ownership contract."
            )
        os.replace(temporary, display_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return display_path


def _meta_rows(meta: dict) -> pd.DataFrame:
    contract = model_contract()
    values = {
        "model_status": contract["status"],
        "lag_order": str(contract["lag_order"]),
        "seasonal_dummies": str(contract["seasonal_dummies"]),
        "seasonal_reference_month": str(contract["seasonal_reference_month"]),
        "food_source": "Eurostat FOOD",
        "food_weight_source": contract["food_weight_source"],
        "max_published_horizon_months": str(
            contract["publication_horizon_max_months"]
        ),
        "aggregation_method": str(meta.get("aggregation_method", "")),
        "future_weight_policy": str(meta.get("future_weight_policy", "")),
        "level_additivity_max_abs_error": str(
            meta.get("draw_wise_level_additivity_max_abs_error", "")
        ),
        "yoy_contribution_additivity_max_abs_error": str(
            meta.get("yoy_contribution_additivity_max_abs_error", "")
        ),
    }
    out = _rows(len(values))
    out["record_type"] = "metadata"
    out["scope"] = "headline"
    out["model_id"] = "headline_joint"
    out["vintage"] = str(meta["vintage"])
    out["run_id"] = str(meta["run_id"])
    out["forecast_name"] = str(meta["forecast_name"])
    out["key"] = list(values)
    out["text_value"] = list(values.values())
    return out


def build_headline_display(
    run_directory: str | Path,
    *,
    forecast_name: str = "unconditional",
    overwrite: bool = True,
) -> Path:
    run_dir = Path(run_directory)
    fc_dir = run_dir / "forecasts" / forecast_name
    meta_path = fc_dir / "headline_metadata.json"
    draws_path = fc_dir / "headline_draws.npz"
    native_history_path = run_dir / "headline_native_history.csv"
    weights_path = run_dir / "headline_weights_annual.csv"
    official_path = run_dir / "headline_official_total.csv"

    for path in (
        meta_path,
        draws_path,
        native_history_path,
        weights_path,
        official_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lock = dict(meta.get("model_lock", {}))
    if lock.get("status") != "locked":
        raise HeadlineDisplayError("Headline run is not marked locked.")
    if int(lock.get("publication_horizon_max_months", -1)) != MAX_PUBLISHED_HORIZON_MONTHS:
        raise HeadlineDisplayError("Headline horizon lock changed unexpectedly.")

    dates = pd.DatetimeIndex(pd.to_datetime(meta["path_dates"]), name="date")
    future_dates = pd.DatetimeIndex(pd.to_datetime(meta["future_dates"]), name="date")
    tail = int(meta["tail_length"])
    if not dates[tail:].equals(future_dates):
        raise HeadlineDisplayError("Headline future-date contract is inconsistent.")
    if len(future_dates) > MAX_PUBLISHED_HORIZON_MONTHS:
        raise HeadlineDisplayError(
            f"Stored Headline forecast has {len(future_dates)} future months; "
            f"maximum is {MAX_PUBLISHED_HORIZON_MONTHS}."
        )

    with np.load(draws_path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}

    native_paths = np.asarray(arrays["native_level_paths"], dtype=float)
    component_yoy = np.asarray(arrays["component_yoy_paths"], dtype=float)
    headline_level = np.asarray(arrays["headline_level_paths"], dtype=float)
    headline_yoy = np.asarray(arrays["headline_yoy_paths"], dtype=float)
    yoy_contrib = np.asarray(arrays["headline_yoy_contribution_paths"], dtype=float)

    if native_paths.shape[1:] != (len(dates), len(NATIVE_VARIABLES)):
        raise HeadlineDisplayError("Native Headline component path dimensions changed.")
    if component_yoy.shape != native_paths.shape:
        raise HeadlineDisplayError("Component YoY path dimensions changed.")
    if headline_level.shape != native_paths.shape[:2]:
        raise HeadlineDisplayError("Headline level path dimensions changed.")
    if headline_yoy.shape != headline_level.shape:
        raise HeadlineDisplayError("Headline YoY path dimensions changed.")
    if yoy_contrib.shape != native_paths.shape:
        raise HeadlineDisplayError("Headline YoY contribution path dimensions changed.")

    native_history = pd.read_csv(
        native_history_path, index_col=0, parse_dates=True
    ).sort_index()
    native_history.index = pd.DatetimeIndex(native_history.index, name="date")
    weights = pd.read_csv(weights_path, index_col=0)
    official_total = pd.read_csv(
        official_path, index_col=0, parse_dates=True
    ).iloc[:, 0].sort_index()
    official_total.index = pd.DatetimeIndex(official_total.index, name="date")

    frames = []

    # Official Headline history: this is what the user sees before the forecast.
    frames.append(
        _history_rows(
            official_total,
            variable="hicp_total",
            metric="level",
            unit="HICP index",
            vintage=str(meta["vintage"]),
            run_id=str(meta["run_id"]),
            forecast_name=forecast_name,
        )
    )
    frames.append(
        _history_rows(
            100.0 * (official_total / official_total.shift(12) - 1.0),
            variable="hicp_total",
            metric="yoy",
            unit="% y/y",
            vintage=str(meta["vintage"]),
            run_id=str(meta["run_id"]),
            forecast_name=forecast_name,
        )
    )

    # Component history.
    for variable in NATIVE_VARIABLES:
        s = native_history[variable]
        frames.append(
            _history_rows(
                s,
                variable=variable,
                metric="level",
                unit="HICP index",
                vintage=str(meta["vintage"]),
                run_id=str(meta["run_id"]),
                forecast_name=forecast_name,
            )
        )
        frames.append(
            _history_rows(
                100.0 * (s / s.shift(12) - 1.0),
                variable=variable,
                metric="yoy",
                unit="% y/y",
                vintage=str(meta["vintage"]),
                run_id=str(meta["run_id"]),
                forecast_name=forecast_name,
            )
        )

    frames.append(
        historical_headline_contribution_rows(
            native_history,
            weights,
            official_total,
            vintage=str(meta["vintage"]),
            run_id=str(meta["run_id"]),
            forecast_name=forecast_name,
        )
    )

    # Draw-wise posterior fans.
    frames.extend(
        [
            _quantile_frame(
                headline_level,
                dates,
                series="hicp_total",
                metric="level",
                unit="HICP index",
                vintage=str(meta["vintage"]),
                run_id=str(meta["run_id"]),
                forecast_name=forecast_name,
                tail_length=tail,
            ),
            _quantile_frame(
                headline_yoy,
                dates,
                series="hicp_total",
                metric="yoy",
                unit="% y/y",
                vintage=str(meta["vintage"]),
                run_id=str(meta["run_id"]),
                forecast_name=forecast_name,
                tail_length=tail,
            ),
        ]
    )
    for j, variable in enumerate(NATIVE_VARIABLES):
        frames.append(
            _quantile_frame(
                native_paths[:, :, j],
                dates,
                series=variable,
                metric="level",
                unit="HICP index",
                vintage=str(meta["vintage"]),
                run_id=str(meta["run_id"]),
                forecast_name=forecast_name,
                tail_length=tail,
            )
        )
        frames.append(
            _quantile_frame(
                component_yoy[:, :, j],
                dates,
                series=variable,
                metric="yoy",
                unit="% y/y",
                vintage=str(meta["vintage"]),
                run_id=str(meta["run_id"]),
                forecast_name=forecast_name,
                tail_length=tail,
            )
        )
        contribution = _quantile_frame(
            yoy_contrib[:, :, j],
            dates,
            series=variable,
            metric="yoy_contribution",
            unit="pp",
            vintage=str(meta["vintage"]),
            run_id=str(meta["run_id"]),
            forecast_name=forecast_name,
            tail_length=tail,
        )
        contribution["record_type"] = "contribution"
        frames.append(contribution)

    frames.append(_meta_rows(meta))

    # Saved-run diagnostics. Root-frequency statistics are computed once here,
    # never from a Dash callback.
    diagnostic_rows = saved_headline_diagnostic_rows(run_dir)
    if not diagnostic_rows.empty:
        frames.append(diagnostic_rows)

    # Structural summaries are also materialised once after estimation.
    structural_path = run_dir / STRUCTURAL_FILENAME
    if structural_path.is_file():
        structural_rows = pd.read_parquet(structural_path)
        if not structural_rows.empty:
            frames.append(structural_rows)

    # Core is derived only after every pre-existing Headline row has been built,
    # so the legacy Headline display remains byte-for-byte/order stable in
    # content while Core rows are appended additively.
    try:
        frames.extend(
            _core_display_rows(
                run_dir=run_dir,
                native_history=native_history,
                weights=weights,
                native_paths=native_paths,
                dates=dates,
                tail=tail,
                meta=meta,
                forecast_name=forecast_name,
            )
        )
    except Exception as exc:
        # Core is additive dashboard functionality. A transient Eurostat/API
        # issue must not invalidate an otherwise accepted Headline run/display.
        frames.append(
            _core_error_rows(
                str(exc),
                meta=meta,
                forecast_name=forecast_name,
            )
        )

    frame = pd.concat(frames, ignore_index=True, sort=False)
    # Identity columns on diagnostic/structural sidecars are filled here so the
    # entire compact display remains self-describing.
    for column, value in (
        ("scope", "headline"),
        ("model_id", "headline_joint"),
        ("vintage", str(meta["vintage"])),
        ("run_id", str(meta["run_id"])),
        ("forecast_name", forecast_name),
        ("basis", "baseline"),
    ):
        if column not in frame:
            frame[column] = value
        else:
            frame[column] = frame[column].where(frame[column].notna(), value)
    frame = _normalise(frame)

    destination = fc_dir / DISPLAY_FILENAME
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, destination)
    return destination


def load_headline_display(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise HeadlineDisplayError("Headline display artefact is empty.")
    versions = set(frame["display_schema_version"].dropna().astype(str))
    if versions != {DISPLAY_SCHEMA_VERSION}:
        raise HeadlineDisplayError(
            f"Unexpected Headline display schema versions: {sorted(versions)}."
        )
    return _normalise(frame)


__all__ = [
    "DISPLAY_FILENAME",
    "DISPLAY_SCHEMA_VERSION",
    "LABELS",
    "historical_headline_contribution_rows",
    "materialize_headline_historical_contributions",
    "materialize_headline_core",
    "build_headline_display",
    "load_headline_display",
]
