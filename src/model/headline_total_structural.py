from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from energy_bvar_model import impulse_responses
from headline_bvar_conditional import load_saved_headline_posterior
from headline_joint_bvar import (
    NATIVE_VARIABLES,
    STATE_VARIABLE_MAP,
    aggregate_headline_draws,
)


HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION = "headline-total-structural-irf-v1.1"
# LOT3_HEADLINE_TOTAL_HORIZON_1_12_V1_1
HEADLINE_TOTAL_STRUCTURAL_MIN_HORIZON = 1
HEADLINE_TOTAL_STRUCTURAL_MAX_HORIZON = 12
DEFAULT_HEADLINE_TOTAL_STRUCTURAL_DRAWS = 300
MAX_HEADLINE_TOTAL_STRUCTURAL_DRAWS = 1000
HEADLINE_TOTAL_STRUCTURAL_HORIZON = 12
HEADLINE_TOTAL_STRUCTURAL_IDENTIFICATION = "recursive"
HEADLINE_TOTAL_STRUCTURAL_SHOCK_UNIT = "structural_std"
HEADLINE_TOTAL_STRUCTURAL_SHOCK_SIZE = 1.0


class HeadlineTotalStructuralError(RuntimeError):
    """Raised when the exact Headline-total structural IRF contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeadlineTotalStructuralError(message)


def _as_list(value: Any) -> list[Any]:
    """Convert list-like saved metadata without NumPy truth-value evaluation."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return [value]


def headline_total_irf_draw_indices(
    n_draws: int,
    requested: int = DEFAULT_HEADLINE_TOTAL_STRUCTURAL_DRAWS,
) -> np.ndarray:
    """Return deterministic evenly spaced posterior indices.

    This is intentionally independent of the dashboard layer.  The Lot-3
    regression verifier proves it equals the current interactive structural
    selector for the validated repository state.
    """
    n = int(n_draws)
    requested = int(requested)
    _require(n > 0, f"n_draws must be positive, found {n}.")
    _require(requested > 0, f"requested draws must be positive, found {requested}.")
    _require(
        requested <= MAX_HEADLINE_TOTAL_STRUCTURAL_DRAWS,
        f"requested draws {requested} exceed the validated maximum "
        f"{MAX_HEADLINE_TOTAL_STRUCTURAL_DRAWS}.",
    )
    count = min(n, requested)
    return np.linspace(0, n - 1, count, dtype=int)


def _subset_result_draws_exact(
    result: Mapping[str, Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    n_total = int(result["n_draws"])
    idx = np.asarray(indices, dtype=int)
    _require(idx.ndim == 1, f"draw_indices must be one-dimensional, found {idx.shape}.")
    _require(len(idx) > 0, "draw_indices cannot be empty.")
    _require(len(np.unique(idx)) == len(idx), "draw_indices contain duplicates.")
    _require(
        np.all((idx >= 0) & (idx < n_total)),
        f"draw_indices are outside [0, {n_total - 1}].",
    )

    out = dict(result)
    subset_keys: list[str] = []
    for key, value in result.items():
        if key in {"prep", "metadata", "prior", "variables", "p", "n_draws"}:
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_total:
            out[key] = value[idx].copy()
            subset_keys.append(str(key))
        elif isinstance(value, list) and len(value) == n_total:
            out[key] = [value[int(i)] for i in idx]
            subset_keys.append(str(key))

    out["n_draws"] = int(len(idx))
    out["_headline_total_original_draw_indices"] = idx.copy()
    out["_headline_total_subset_keys"] = tuple(subset_keys)

    for required in ("B", "A", "log_variance", "outlier_scales"):
        _require(
            required in subset_keys,
            f"Saved posterior draw array {required!r} was not subset draw-by-draw.",
        )
    return out


def _historical_deterministic_exog(
    prep: Mapping[str, Any],
    future_dates: pd.DatetimeIndex,
) -> pd.DataFrame | None:
    names = [str(x) for x in _as_list(prep.get("exog_names"))]
    if not names:
        return None

    historical = prep.get("exog")
    _require(
        historical is not None,
        "Fitted Headline model has exog_names but no historical prep['exog'].",
    )
    hist = pd.DataFrame(historical).copy()
    missing = [name for name in names if name not in hist.columns]
    _require(not missing, f"Historical deterministic exog lacks fitted columns: {missing}.")
    hist = hist.loc[:, names].astype(float)
    hist.index = pd.DatetimeIndex(hist.index).to_period("M").to_timestamp(how="start")

    rows: list[np.ndarray] = []
    for date in future_dates:
        month_rows = hist.loc[hist.index.month == int(pd.Timestamp(date).month)]
        _require(
            not month_rows.empty,
            f"No historical deterministic exog row exists for month {date.month}.",
        )
        unique = np.unique(month_rows.to_numpy(dtype=float), axis=0)
        _require(
            len(unique) == 1,
            f"Historical deterministic exog has {len(unique)} different patterns "
            f"for calendar month {date.month}; refusing to guess a future value.",
        )
        rows.append(unique[0])

    out = pd.DataFrame(rows, index=future_dates, columns=names, dtype=float)
    _require(
        np.isfinite(out.to_numpy(dtype=float)).all(),
        "Future deterministic exog contains non-finite values.",
    )
    return out


def _reference_lag_states(
    prep: Mapping[str, Any],
    *,
    reference_date: pd.Timestamp,
    p: int,
    n: int,
) -> np.ndarray:
    balanced = prep.get("balanced")
    _require(balanced is not None, "Saved prep has no balanced state-difference panel.")
    frame = pd.DataFrame(balanced).copy()
    frame.index = pd.DatetimeIndex(frame.index).to_period("M").to_timestamp(how="start")
    ref = pd.Timestamp(reference_date).to_period("M").to_timestamp(how="start")
    available = frame.loc[frame.index <= ref]
    _require(
        len(available) >= p,
        f"Only {len(available)} balanced state rows are available through {ref.date()} "
        f"for p={p}.",
    )
    values = available.iloc[-p:].to_numpy(dtype=float)
    _require(
        values.shape == (p, n),
        f"Last p balanced states have shape {values.shape}, expected {(p, n)}.",
    )
    _require(np.isfinite(values).all(), "Initial lag-state block contains non-finite values.")
    # Same convention as prepare_bvar_panel: most recent lag first.
    return values[::-1].copy()


def _deterministic_zero_innovation_changes(
    subset: Mapping[str, Any],
    *,
    horizon: int,
    reference_date: pd.Timestamp,
) -> tuple[np.ndarray, pd.DatetimeIndex, pd.DataFrame | None]:
    prep = subset["prep"]
    variables = list(subset["variables"])
    B_draws = np.asarray(subset["B"], dtype=float)
    p = int(subset["p"])
    n = len(variables)
    D = len(B_draws)

    future_dates = pd.date_range(
        pd.Timestamp(reference_date).to_period("M").to_timestamp(how="start")
        + pd.offsets.MonthBegin(1),
        periods=int(horizon),
        freq="MS",
        name="date",
    )
    exog = _historical_deterministic_exog(prep, future_dates)
    m = 0 if exog is None else int(exog.shape[1])

    expected_k = 1 + n * p + m
    _require(
        B_draws.ndim == 3 and B_draws.shape[1:] == (expected_k, n),
        f"B draw shape {B_draws.shape} is inconsistent with constant + {p}x{n} "
        f"lags + {m} deterministic exog columns (k={expected_k}).",
    )

    lag0 = _reference_lag_states(
        prep,
        reference_date=reference_date,
        p=p,
        n=n,
    )
    exog_values = None if exog is None else exog.to_numpy(dtype=float)
    changes = np.empty((D, int(horizon), n), dtype=float)

    for d in range(D):
        lags = lag0.copy()
        B = B_draws[d]
        for h in range(int(horizon)):
            parts = [np.array([1.0]), lags.reshape(-1)]
            if exog_values is not None:
                parts.append(exog_values[h])
            x = np.concatenate(parts)
            y = x @ B
            _require(
                np.isfinite(y).all(),
                f"Non-finite deterministic baseline state at draw={d}, horizon={h + 1}.",
            )
            changes[d, h] = y
            if p > 1:
                lags[1:] = lags[:-1]
            lags[0] = y

    return changes, future_dates, exog


def _native_baseline_levels(
    *,
    native_levels: pd.DataFrame,
    reference_date: pd.Timestamp,
    deterministic_changes: np.ndarray,
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    native = native_levels.loc[:, list(NATIVE_VARIABLES)].copy()
    native.index = pd.DatetimeIndex(native.index).to_period("M").to_timestamp(how="start")
    ref = pd.Timestamp(reference_date).to_period("M").to_timestamp(how="start")
    _require(ref in native.index, f"Native HICP reference level is missing at {ref.date()}.")
    anchor = native.loc[ref].to_numpy(dtype=float)
    _require(
        np.isfinite(anchor).all() and np.all(anchor > 0),
        f"Native HICP reference anchor is invalid: {anchor}.",
    )

    D, H, n = deterministic_changes.shape
    _require(
        n == len(NATIVE_VARIABLES),
        f"Deterministic state dimension {n} != native dimension {len(NATIVE_VARIABLES)}.",
    )
    levels = np.empty((D, H + 1, n), dtype=float)
    levels[:, 0, :] = anchor[None, :]
    levels[:, 1:, :] = anchor[None, None, :] * np.exp(
        np.cumsum(deterministic_changes, axis=1)
    )
    dates = pd.date_range(ref, periods=H + 1, freq="MS", name="date")
    _require(np.isfinite(levels).all() and np.all(levels > 0), "Baseline native levels are invalid.")
    return levels, dates, anchor


def _aggregate_native_paths(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    native_levels: pd.DataFrame,
    weights: pd.DataFrame,
    official_total: pd.Series,
) -> dict[str, object]:
    return aggregate_headline_draws(
        {
            "native_level_paths": np.asarray(paths, dtype=float),
            "native_variables": list(NATIVE_VARIABLES),
            "path_dates": pd.DatetimeIndex(dates, name="date"),
            "tail_length": 0,
        },
        native_levels=native_levels,
        weights=weights,
        official_total=official_total,
    )


def _reference_date_from_result(
    result: Mapping[str, Any],
    reference_date: Any,
) -> pd.Timestamp:
    """Resolve the only reference date validated by the v1 contract.

    LOT 3 Block 2 proved the exact Headline re-aggregation only at the current
    structural endpoint, where the final fitted date, balanced_end and
    last_calendar_date coincide. Historical reference dates are deliberately
    rejected until they receive a separate numerical proof.
    """
    prep = result["prep"]
    dates = pd.DatetimeIndex(prep["dates"]).to_period("M").to_timestamp(how="start")
    _require(len(dates) > 0, "Saved posterior prep has no structural dates.")

    fitted_end = pd.Timestamp(dates[-1])
    balanced_end = pd.Timestamp(prep["balanced_end"]).to_period("M").to_timestamp(how="start")
    last_calendar = pd.Timestamp(prep["last_calendar_date"]).to_period("M").to_timestamp(how="start")
    _require(
        fitted_end == balanced_end == last_calendar,
        "Contract "
        f"{HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION} requires the fitted "
        "structural endpoint, balanced_end and last_calendar_date to coincide; "
        f"found {fitted_end.date()} / {balanced_end.date()} / {last_calendar.date()}.",
    )

    ref = fitted_end if reference_date is None else pd.Timestamp(reference_date)
    ref = pd.Timestamp(ref).to_period("M").to_timestamp(how="start")
    _require(
        ref == fitted_end,
        f"Contract {HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION} is validated "
        f"only at the current structural endpoint {fitted_end.date()}; "
        f"found reference_date={ref.date()}.",
    )
    return ref


def _structural_contract_gate(
    *,
    horizon: int,
    identification: str,
    shock_unit: str,
    shock_size: float,
    include_outlier_scale: bool,
) -> None:
    resolved_horizon = int(horizon)
    _require(
        HEADLINE_TOTAL_STRUCTURAL_MIN_HORIZON
        <= resolved_horizon
        <= HEADLINE_TOTAL_STRUCTURAL_MAX_HORIZON,
        f"Contract {HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION} supports "
        f"horizon in [{HEADLINE_TOTAL_STRUCTURAL_MIN_HORIZON}, "
        f"{HEADLINE_TOTAL_STRUCTURAL_MAX_HORIZON}], found {horizon}.",
    )
    _require(
        str(identification) == HEADLINE_TOTAL_STRUCTURAL_IDENTIFICATION,
        f"Contract {HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION} currently supports only "
        f"identification={HEADLINE_TOTAL_STRUCTURAL_IDENTIFICATION!r}.",
    )
    _require(
        str(shock_unit) == HEADLINE_TOTAL_STRUCTURAL_SHOCK_UNIT,
        f"Contract {HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION} currently supports only "
        f"shock_unit={HEADLINE_TOTAL_STRUCTURAL_SHOCK_UNIT!r}.",
    )
    _require(
        float(shock_size) == HEADLINE_TOTAL_STRUCTURAL_SHOCK_SIZE,
        f"Contract {HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION} currently supports only "
        f"shock_size={HEADLINE_TOTAL_STRUCTURAL_SHOCK_SIZE:g}.",
    )
    _require(
        not bool(include_outlier_scale),
        f"Contract {HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION} requires "
        "include_outlier_scale=False.",
    )


def _qsummary(arr: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(arr, dtype=float)
    q = np.nanquantile(x, [0.16, 0.50, 0.84], axis=0)
    return {
        "mean": np.nanmean(x, axis=0).tolist(),
        "q16": q[0].tolist(),
        "q50": q[1].tolist(),
        "q84": q[2].tolist(),
    }


def run_headline_total_structural_irf_draws(
    result: Mapping[str, Any],
    *,
    native_levels: pd.DataFrame,
    weights: pd.DataFrame,
    official_total: pd.Series,
    requested_draws: int = DEFAULT_HEADLINE_TOTAL_STRUCTURAL_DRAWS,
    horizon: int = HEADLINE_TOTAL_STRUCTURAL_HORIZON,
    identification: str = HEADLINE_TOTAL_STRUCTURAL_IDENTIFICATION,
    reference_date: Any = None,
    shock_unit: str = HEADLINE_TOTAL_STRUCTURAL_SHOCK_UNIT,
    shock_size: float = HEADLINE_TOTAL_STRUCTURAL_SHOCK_SIZE,
    include_outlier_scale: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute exact Headline HICP Total structural IRFs draw-by-draw.

    This v1.1 contract is deliberately narrower than the generic component IRF
    engine. It implements the recursively identified, regular one-structural-
    standard-deviation response validated in LOT 3 Block 2.  FEVD is not part
    of this contract.
    """
    _structural_contract_gate(
        horizon=horizon,
        identification=identification,
        shock_unit=shock_unit,
        shock_size=shock_size,
        include_outlier_scale=include_outlier_scale,
    )

    native_variables = list(NATIVE_VARIABLES)
    state_variables = list(result["variables"])
    state_map = dict(STATE_VARIABLE_MAP)
    expected_states = [state_map[name] for name in native_variables]
    _require(
        state_variables == expected_states,
        f"Headline state/native ordering changed: {state_variables} != {expected_states}.",
    )

    n_total = int(result["n_draws"])
    draw_indices = headline_total_irf_draw_indices(n_total, requested_draws)
    subset = _subset_result_draws_exact(result, draw_indices)
    ref = _reference_date_from_result(result, reference_date)

    prep = result["prep"]
    balanced_end = pd.Timestamp(prep["balanced_end"]).to_period("M").to_timestamp(how="start")
    last_calendar = pd.Timestamp(prep["last_calendar_date"]).to_period("M").to_timestamp(how="start")
    _require(
        ref == balanced_end == last_calendar,
        "Headline Total structural v1.1 reference-date endpoint contract changed "
        f"unexpectedly: {ref.date()} / {balanced_end.date()} / {last_calendar.date()}.",
    )

    irf = impulse_responses(
        subset,
        identification=identification,
        reference_date=ref,
        horizon=int(horizon),
        shock_unit=shock_unit,
        shock_size=float(shock_size),
        include_outlier_scale=False,
        seed=int(seed),
    )
    used_local = np.asarray(irf["used_draw_indices"], dtype=int)
    _require(
        np.array_equal(used_local, np.arange(len(draw_indices), dtype=int)),
        "Recursive component IRF changed/reordered the selected posterior draws.",
    )
    used_original = draw_indices[used_local]
    _require(
        np.array_equal(used_original, draw_indices),
        "Original posterior draw pairing failed after recursive identification.",
    )

    change_irfs = np.asarray(irf["change_irfs"], dtype=float)
    cumulative_irfs = np.asarray(irf["cumulative_level_irfs"], dtype=float)
    expected_shape = (len(draw_indices), int(horizon) + 1, 4, 4)
    _require(
        change_irfs.shape == expected_shape and cumulative_irfs.shape == expected_shape,
        f"Unexpected component IRF shapes: {change_irfs.shape} / {cumulative_irfs.shape}; "
        f"expected {expected_shape}.",
    )
    cumsum_error = float(
        np.max(np.abs(cumulative_irfs - np.cumsum(change_irfs, axis=1)))
    )
    _require(cumsum_error <= 1e-14, f"Component cumulative IRF identity failed: {cumsum_error}.")

    deterministic_changes, _, deterministic_exog = _deterministic_zero_innovation_changes(
        subset,
        horizon=int(horizon),
        reference_date=ref,
    )
    baseline_native, path_dates, native_anchor = _native_baseline_levels(
        native_levels=native_levels,
        reference_date=ref,
        deterministic_changes=deterministic_changes,
    )
    baseline_agg = _aggregate_native_paths(
        baseline_native,
        path_dates,
        native_levels=native_levels,
        weights=weights,
        official_total=official_total,
    )

    baseline_headline = np.asarray(baseline_agg["headline_level_paths"], dtype=float)
    baseline_yoy = np.asarray(baseline_agg["headline_yoy_paths"], dtype=float)
    baseline_contrib = np.asarray(
        baseline_agg["headline_component_index_contributions"], dtype=float
    )
    _require(
        baseline_headline.shape == (len(draw_indices), int(horizon) + 1),
        f"Baseline Headline level shape changed: {baseline_headline.shape}.",
    )
    _require(baseline_yoy.shape == baseline_headline.shape, "Baseline Headline YoY shape changed.")
    baseline_additivity_error = float(
        np.max(np.abs(baseline_contrib.sum(axis=2) - baseline_headline))
    )
    _require(
        baseline_additivity_error <= 1e-10,
        f"Baseline Headline component-index additivity failed: {baseline_additivity_error}.",
    )

    shock_names_state = list(irf["shock_names"])
    state_to_native = {state_map[name]: name for name in native_variables}
    shock_names = [state_to_native.get(name, str(name).replace("state__", "")) for name in shock_names_state]
    _require(shock_names == native_variables, f"Unexpected recursive shock order: {shock_names}.")

    level_response = np.empty((len(draw_indices), int(horizon) + 1, 4), dtype=float)
    yoy_response = np.empty_like(level_response)
    shocked_additivity_errors: list[float] = []

    for j, shock in enumerate(shock_names):
        multiplier = np.exp(cumulative_irfs[:, :, :, j])
        _require(
            multiplier.shape == baseline_native.shape,
            f"Component multiplier shape changed for {shock}: {multiplier.shape}.",
        )
        shocked_native = baseline_native * multiplier
        shocked_agg = _aggregate_native_paths(
            shocked_native,
            path_dates,
            native_levels=native_levels,
            weights=weights,
            official_total=official_total,
        )
        shocked_headline = np.asarray(shocked_agg["headline_level_paths"], dtype=float)
        shocked_yoy = np.asarray(shocked_agg["headline_yoy_paths"], dtype=float)
        shocked_contrib = np.asarray(
            shocked_agg["headline_component_index_contributions"], dtype=float
        )
        error = float(np.max(np.abs(shocked_contrib.sum(axis=2) - shocked_headline)))
        _require(
            error <= 1e-10,
            f"Shocked Headline component-index additivity failed for {shock}: {error}.",
        )
        shocked_additivity_errors.append(error)
        level_response[:, :, j] = 100.0 * (shocked_headline / baseline_headline - 1.0)
        yoy_response[:, :, j] = shocked_yoy - baseline_yoy

    # Reconstruction diagnostic at the structural reference date.
    official = official_total.copy().sort_index()
    official.index = pd.DatetimeIndex(official.index).to_period("M").to_timestamp(how="start")
    if ref in official.index and np.isfinite(float(official.loc[ref])):
        h0_vs_official = float(
            np.max(np.abs(baseline_headline[:, 0] - float(official.loc[ref])))
        )
    else:
        h0_vs_official = float("nan")

    # Literal zero-shock identity is important enough to carry as a numeric diagnostic.
    zero_agg = _aggregate_native_paths(
        baseline_native,
        path_dates,
        native_levels=native_levels,
        weights=weights,
        official_total=official_total,
    )
    zero_level = 100.0 * (
        np.asarray(zero_agg["headline_level_paths"], dtype=float) / baseline_headline - 1.0
    )
    zero_yoy = np.asarray(zero_agg["headline_yoy_paths"], dtype=float) - baseline_yoy
    zero_response_error = float(max(np.max(np.abs(zero_level)), np.max(np.abs(zero_yoy))))
    _require(zero_response_error == 0.0, f"Literal zero-shock response is not bit-zero: {zero_response_error}.")

    deterministic_exog_names = [] if deterministic_exog is None else list(deterministic_exog.columns)
    weight_year = np.asarray(
        baseline_agg.get("headline_weight_year_used", baseline_agg.get("weight_year_used", [])),
        dtype=int,
    )
    carried = np.asarray(
        baseline_agg.get(
            "headline_weight_carried_forward",
            baseline_agg.get("carried_weights", []),
        ),
        dtype=bool,
    )

    return {
        "contract": HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION,
        "identification": str(identification),
        "reference_date": ref,
        "dates": pd.DatetimeIndex(path_dates, name="date"),
        "horizon": int(horizon),
        "shock_unit": str(shock_unit),
        "shock_size": float(shock_size),
        "include_outlier_scale": False,
        "seed": int(seed),
        "draw_indices": np.asarray(draw_indices, dtype=int).copy(),
        "n_draws": int(len(draw_indices)),
        "posterior_draws_available": int(n_total),
        "native_variables": list(native_variables),
        "state_variables": list(state_variables),
        "shock_names": list(shock_names),
        "component_change_irfs": change_irfs.copy(),
        "component_cumulative_log_irfs": cumulative_irfs.copy(),
        "baseline_native_level_paths": baseline_native.copy(),
        "baseline_headline_level_paths": baseline_headline.copy(),
        "baseline_headline_yoy_paths": baseline_yoy.copy(),
        "headline_level_response_pct": level_response,
        "headline_yoy_response_pp": yoy_response,
        "headline_weight_year_used": weight_year.copy(),
        "headline_weight_carried_forward": carried.copy(),
        "diagnostics": {
            "component_cumulative_cumsum_max_error": cumsum_error,
            "baseline_component_index_additivity_max_error": baseline_additivity_error,
            "shocked_component_index_additivity_max_error": float(max(shocked_additivity_errors)),
            "zero_shock_response_max_error": zero_response_error,
            "baseline_h0_vs_official_total_index_points": h0_vs_official,
            "future_stochastic_innovations": 0,
            "future_outlier_innovations": 0,
            "future_deterministic_exog_columns": deterministic_exog_names,
            "future_deterministic_exog_method": (
                "none" if not deterministic_exog_names else "exact_historical_calendar_month_pattern"
            ),
            "native_reference_levels": {
                name: float(value) for name, value in zip(native_variables, native_anchor)
            },
            "reference_date_policy": "current_structural_endpoint_only",
            "fevd_used": False,
            "bvar_reestimated": False,
        },
    }


def compact_headline_total_structural_irf(
    draw_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact exact draw-level Headline Total IRFs into dashboard-ready bands."""
    contract = str(draw_result.get("contract") or "")
    _require(
        contract == HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION,
        f"Unexpected Headline Total structural contract {contract!r}.",
    )
    dates = pd.DatetimeIndex(draw_result["dates"], name="date")
    shock_names = list(draw_result["shock_names"])
    level = np.asarray(draw_result["headline_level_response_pct"], dtype=float)
    yoy = np.asarray(draw_result["headline_yoy_response_pp"], dtype=float)
    _require(
        level.shape == yoy.shape == (int(draw_result["n_draws"]), len(dates), len(shock_names)),
        f"Headline Total response array shapes are inconsistent: {level.shape} / {yoy.shape}.",
    )

    shocks: dict[str, Any] = {}
    for j, shock in enumerate(shock_names):
        shocks[str(shock)] = {
            "level_response_pct": _qsummary(level[:, :, j]),
            "yoy_response_pp": _qsummary(yoy[:, :, j]),
        }

    diagnostics = dict(draw_result.get("diagnostics") or {})
    return {
        "contract": HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION,
        "dates": [pd.Timestamp(date).date().isoformat() for date in dates],
        "shock_names": list(shock_names),
        "shocks": shocks,
        "meta": {
            "identification": str(draw_result["identification"]),
            "reference_date": pd.Timestamp(draw_result["reference_date"]).date().isoformat(),
            "horizon": int(draw_result["horizon"]),
            "shock_unit": str(draw_result["shock_unit"]),
            "shock_size": float(draw_result["shock_size"]),
            "include_outlier_scale": bool(draw_result["include_outlier_scale"]),
            "seed": int(draw_result["seed"]),
            "n_draws": int(draw_result["n_draws"]),
            "posterior_draws_available": int(draw_result["posterior_draws_available"]),
            "draw_indices": np.asarray(draw_result["draw_indices"], dtype=int).tolist(),
            "native_variables": list(draw_result["native_variables"]),
            "state_variables": list(draw_result["state_variables"]),
            "headline_weight_year_used": np.asarray(
                draw_result.get("headline_weight_year_used", []), dtype=int
            ).tolist(),
            "headline_weight_carried_forward": np.asarray(
                draw_result.get("headline_weight_carried_forward", []), dtype=bool
            ).tolist(),
            "diagnostics": diagnostics,
            "level_metric": "100 * (Headline_shocked / Headline_baseline - 1)",
            "level_unit": "%",
            "yoy_metric": "Headline_shocked_yoy - Headline_baseline_yoy",
            "yoy_unit": "pp",
            "future_stochastic_innovations": 0,
            "future_outlier_innovations": 0,
            "reference_date_policy": "current_structural_endpoint_only",
            "fevd_used": False,
            "bvar_reestimated": False,
        },
    }


def run_saved_headline_total_structural_irf_draws(
    run_directory: str | Path,
    *,
    requested_draws: int = DEFAULT_HEADLINE_TOTAL_STRUCTURAL_DRAWS,
    horizon: int = HEADLINE_TOTAL_STRUCTURAL_HORIZON,
    identification: str = HEADLINE_TOTAL_STRUCTURAL_IDENTIFICATION,
    reference_date: Any = None,
    shock_unit: str = HEADLINE_TOTAL_STRUCTURAL_SHOCK_UNIT,
    shock_size: float = HEADLINE_TOTAL_STRUCTURAL_SHOCK_SIZE,
    include_outlier_scale: bool = False,
    seed: int = 42,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a saved Headline posterior and return exact draw-level Total IRFs."""
    posterior = load_saved_headline_posterior(
        run_directory,
        project_root=project_root,
    )
    out = run_headline_total_structural_irf_draws(
        posterior.result,
        native_levels=posterior.inputs.native_levels,
        weights=posterior.inputs.weights,
        official_total=posterior.inputs.official_total,
        requested_draws=requested_draws,
        horizon=horizon,
        identification=identification,
        reference_date=reference_date,
        shock_unit=shock_unit,
        shock_size=shock_size,
        include_outlier_scale=include_outlier_scale,
        seed=seed,
    )
    out["vintage"] = str(posterior.vintage)
    out["run_directory"] = str(Path(run_directory).expanduser().resolve())
    return out


def run_saved_headline_total_structural_irf(
    run_directory: str | Path,
    *,
    requested_draws: int = DEFAULT_HEADLINE_TOTAL_STRUCTURAL_DRAWS,
    horizon: int = HEADLINE_TOTAL_STRUCTURAL_HORIZON,
    identification: str = HEADLINE_TOTAL_STRUCTURAL_IDENTIFICATION,
    reference_date: Any = None,
    shock_unit: str = HEADLINE_TOTAL_STRUCTURAL_SHOCK_UNIT,
    shock_size: float = HEADLINE_TOTAL_STRUCTURAL_SHOCK_SIZE,
    include_outlier_scale: bool = False,
    seed: int = 42,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return compact exact Headline Total structural IRF summaries."""
    draws = run_saved_headline_total_structural_irf_draws(
        run_directory,
        requested_draws=requested_draws,
        horizon=horizon,
        identification=identification,
        reference_date=reference_date,
        shock_unit=shock_unit,
        shock_size=shock_size,
        include_outlier_scale=include_outlier_scale,
        seed=seed,
        project_root=project_root,
    )
    payload = compact_headline_total_structural_irf(draws)
    payload["vintage"] = str(draws.get("vintage") or "")
    payload["run_directory"] = str(draws.get("run_directory") or "")
    return payload


__all__ = [
    "HEADLINE_TOTAL_STRUCTURAL_IRF_CONTRACT_VERSION",
    "DEFAULT_HEADLINE_TOTAL_STRUCTURAL_DRAWS",
    "MAX_HEADLINE_TOTAL_STRUCTURAL_DRAWS",
    "HEADLINE_TOTAL_STRUCTURAL_HORIZON",
    "HEADLINE_TOTAL_STRUCTURAL_MIN_HORIZON",
    "HEADLINE_TOTAL_STRUCTURAL_MAX_HORIZON",
    "HEADLINE_TOTAL_STRUCTURAL_IDENTIFICATION",
    "HEADLINE_TOTAL_STRUCTURAL_SHOCK_UNIT",
    "HEADLINE_TOTAL_STRUCTURAL_SHOCK_SIZE",
    "HeadlineTotalStructuralError",
    "headline_total_irf_draw_indices",
    "run_headline_total_structural_irf_draws",
    "compact_headline_total_structural_irf",
    "run_saved_headline_total_structural_irf_draws",
    "run_saved_headline_total_structural_irf",
]
