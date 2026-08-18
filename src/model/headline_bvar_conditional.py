"""Conditional forecasts for the locked joint Headline-HICP BVAR.

Delivery 3 correctness contract
-------------------------------
* the saved Headline posterior is never re-estimated;
* the computational horizon is always the locked 12-month horizon;
* a shorter 3/6-month user condition is represented as a partial condition
  inside that 12-month DK problem; later months remain latent;
* baseline reconstruction uses the forecast draw-count/seed/outlier contract
  persisted with the run (or explicitly backfilled for legacy runs);
* the reconstructed baseline must be bit-identical to the persisted baseline
  before any conditional result is returned;
* Energy -> Headline transfers monthly growth rates and re-anchors them on the
  published Headline HICP-Energy level, never raw index levels;
* Energy/Headline vintages must match exactly.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from energy_bvar_aggregate import historical_model_component_reconstruction
from energy_bvar_model import make_bvar_svo_prior, prepare_bvar_panel
from headline_joint_bvar import (
    NATIVE_VARIABLES,
    STATE_VARIABLES,
    aggregate_headline_draws,
    native_to_state_levels,
)
from headline_bvar_integrity import (
    current_headline_source_contract,
    load_integrity_contract,
)
from headline_bvar_pipeline import (
    BASELINE_P,
    MAX_PUBLISHED_HORIZON_MONTHS,
    SEASONAL_EXOG_PRIOR_SCALE,
    build_inputs,
    exact_headline_yoy_contribution_paths,
    forecast_locked_headline,
    make_joint_seasonals,
    production_prior_config,
)

CONDITIONAL_CONTRACT_VERSION = "headline-conditional-v2"
ENERGY_BRIDGE_CONTRACT_VERSION = "energy-headline-bridge-v3"
ENERGY_BASELINE_CACHE_CONTRACT_VERSION = "headline_energy_baseline_v1"
SELECTED_ENERGY_BASELINE_CONTRACT_VERSION = "energy-headline-selected-baseline-v1"
COMPUTATIONAL_HORIZON = MAX_PUBLISHED_HORIZON_MONTHS
DEFAULT_CONDITIONAL_SEED = 2026  # legacy import compatibility only; run contract wins
CONDITION_STATISTICS = ("mean", "q16", "q50", "q84") + tuple(f"p{x:02d}" for x in range(1, 100))
MANUAL_METRICS = ("level", "yoy")
ENERGY_RATIO_PATHOLOGICAL_THRESHOLD = 0.02


class HeadlineConditionalError(RuntimeError):
    pass


@dataclass(frozen=True)
class SavedHeadlinePosterior:
    run_directory: Path
    metadata: dict[str, Any]
    integrity: dict[str, Any]
    inputs: Any
    result: dict[str, Any]

    @property
    def vintage(self) -> str:
        return str(self.metadata["vintage"])

    @property
    def run_id(self) -> str:
        return str(self.metadata["run_id"])

    @property
    def forecast_draws(self) -> int:
        value = self.integrity.get("n_forecast_draws")
        if value is None:
            raise HeadlineConditionalError(
                "Saved Headline run has no recoverable n_forecast_draws contract."
            )
        return int(value)

    @property
    def forecast_seed(self) -> int:
        value = self.integrity.get("forecast_seed")
        if value is None:
            raise HeadlineConditionalError(
                "Saved Headline run has no recoverable forecast_seed contract."
            )
        return int(value)

    @property
    def simulate_future_outliers(self) -> bool:
        value = self.integrity.get("simulate_future_outliers")
        if value is None:
            raise HeadlineConditionalError(
                "Saved Headline run has no recoverable future-outlier contract."
            )
        return bool(value)


@dataclass(frozen=True)
class HeadlineConditionalDrawBundle:
    """Draw-level Headline conditional object retained before posterior summaries."""

    draw_indices: np.ndarray
    future_dates: pd.DatetimeIndex
    native_level_paths: np.ndarray
    component_yoy_paths: np.ndarray
    headline_level_paths: np.ndarray
    headline_yoy_paths: np.ndarray
    contribution_yoy_paths: np.ndarray


_ENERGY_BASELINE_DRAW_CACHE: dict[
    tuple[Any, ...], tuple[np.ndarray, HeadlineConditionalDrawBundle]
] = {}


def _normalise_months(values: Sequence[Any]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values))).to_period("M").to_timestamp(how="start")


def _project_root_from_run(run_dir: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    for candidate in (run_dir, *run_dir.parents):
        if (candidate / "src" / "model").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise HeadlineConditionalError(
        "Could not resolve project root for Headline conditional lineage checks."
    )


def same_vintage(headline_vintage: str, source_vintage: str) -> None:
    """Hard guard: scenarios can never cross processed vintages."""
    if str(headline_vintage) != str(source_vintage):
        raise HeadlineConditionalError(
            "Headline/Energy vintage mismatch: "
            f"Headline={headline_vintage}, Energy={source_vintage}. "
            "Select/re-run both domains on the same processed vintage before conditioning Headline."
        )


def _outlier_grid() -> np.ndarray:
    p = production_prior_config()
    return np.arange(
        float(p.outlier_grid_min),
        float(p.outlier_grid_max) + 0.5 * float(p.outlier_grid_step),
        float(p.outlier_grid_step),
        dtype=float,
    )


def _timestamp_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return True
    return pd.Timestamp(left) == pd.Timestamp(right)


def load_saved_headline_posterior(
    run_directory: str | Path,
    *,
    project_root: str | Path | None = None,
) -> SavedHeadlinePosterior:
    """Rebuild the saved-run runtime with explicit lineage verification.

    Headline never had a supported production ``linear`` missing-data branch,
    so there is intentionally no signature introspection/legacy emulation here.
    """
    run_dir = Path(run_directory).resolve()
    meta_path = run_dir / "metadata.json"
    draws_path = run_dir / "draws.npz"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if not draws_path.is_file():
        raise FileNotFoundError(draws_path)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(metadata.get("model_id")) != "headline_joint":
        raise HeadlineConditionalError(
            f"Expected model_id='headline_joint', found {metadata.get('model_id')!r}."
        )
    if int(metadata.get("p", BASELINE_P)) != BASELINE_P:
        raise HeadlineConditionalError(
            f"Conditional loader only supports the locked BVAR({BASELINE_P})."
        )
    unsupported_missing = metadata.get("missing_data_method")
    if unsupported_missing not in (None, "", "dk"):
        raise HeadlineConditionalError(
            "Headline conditional loading supports only the production DK contract; "
            f"found missing_data_method={unsupported_missing!r}."
        )

    run_id = str(metadata.get("run_id") or "")
    vintage = str(metadata.get("vintage") or "")
    if not run_id or not vintage:
        raise HeadlineConditionalError("Saved Headline metadata has no run_id/vintage.")
    if run_dir.name != run_id:
        raise HeadlineConditionalError(
            f"Run-directory lineage mismatch: directory={run_dir.name}, metadata={run_id}."
        )
    if run_dir.parent.name != vintage:
        raise HeadlineConditionalError(
            f"Run-directory vintage mismatch: directory={run_dir.parent.name}, metadata={vintage}."
        )

    root = _project_root_from_run(run_dir, project_root)
    integrity = load_integrity_contract(run_dir)
    inputs = build_inputs(vintage, project_root=root)
    state = native_to_state_levels(inputs.native_levels)
    seasonals = make_joint_seasonals(state.index)
    prep = prepare_bvar_panel(
        state,
        p=BASELINE_P,
        variables=STATE_VARIABLES,
        exog=seasonals,
        frequency="monthly",
    )

    current_contract = current_headline_source_contract(vintage, project_root=root)
    expected_hash = integrity.get("source_data_hash")
    if expected_hash is not None and str(expected_hash) != str(current_contract["source_data_hash"]):
        raise HeadlineConditionalError(
            "Headline source-data lineage mismatch: the processed vintage no longer matches "
            "the saved/backfilled run contract. Refusing to condition a stale posterior."
        )
    expected_exog = integrity.get("exog_names")
    if expected_exog is not None and list(expected_exog) != list(prep.get("exog_names", [])):
        raise HeadlineConditionalError(
            "Headline deterministic-regressor contract changed since estimation."
        )
    if not _timestamp_equal(integrity.get("balanced_end"), prep.get("balanced_end")):
        raise HeadlineConditionalError("Headline balanced_end changed since estimation.")
    if not _timestamp_equal(integrity.get("last_calendar_date"), prep.get("last_calendar_date")):
        raise HeadlineConditionalError("Headline last_calendar_date changed since estimation.")

    with np.load(draws_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    required = ("B", "A", "phi", "outlier_probabilities", "log_variance")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise HeadlineConditionalError(
            "Saved Headline posterior is incomplete for conditional forecasting: "
            + ", ".join(missing)
        )
    n_draws = int(np.asarray(arrays["B"]).shape[0])
    if n_draws < 1:
        raise HeadlineConditionalError("Saved Headline posterior contains no draws.")
    for name in required[1:]:
        if int(np.asarray(arrays[name]).shape[0]) != n_draws:
            raise HeadlineConditionalError(f"Posterior draw count differs for {name}.")

    prior = make_bvar_svo_prior(
        prep,
        production_prior_config(),
        exog_prior_scale=SEASONAL_EXOG_PRIOR_SCALE,
    )
    # The forecaster only needs outlier_grid from the prior, but using the full
    # reconstructed prior makes the runtime contract identical to the saved-run
    # reproducibility harness.
    result: dict[str, Any] = {
        "metadata": metadata,
        "prep": prep,
        "variables": list(STATE_VARIABLES),
        "p": BASELINE_P,
        "n_draws": n_draws,
        "prior": prior,
        "missing_data_method": "dk",
    }
    result.update(arrays)
    return SavedHeadlinePosterior(run_dir, metadata, integrity, inputs, result)


def future_dates_for_saved(posterior: SavedHeadlinePosterior, H: int) -> pd.DatetimeIndex:
    H = int(H)
    if H < 1 or H > MAX_PUBLISHED_HORIZON_MONTHS:
        raise ValueError(f"H must lie in 1..{MAX_PUBLISHED_HORIZON_MONTHS}.")
    last_date = pd.Timestamp(posterior.result["prep"]["last_calendar_date"])
    return pd.date_range(
        last_date + pd.offsets.MonthBegin(1), periods=H, freq="MS", name="date"
    )


def parse_manual_values(text: str | Sequence[float], H: int) -> np.ndarray:
    """Parse exactly H finite numbers; no silent scalar broadcasting."""
    H = int(H)
    if isinstance(text, str):
        tokens = [x for x in re.split(r"[\s,;]+", text.strip()) if x]
        try:
            values = np.asarray([float(x) for x in tokens], dtype=float)
        except ValueError as exc:
            raise HeadlineConditionalError("Manual path contains a non-numeric value.") from exc
    else:
        values = np.asarray(list(text), dtype=float)
    if values.ndim != 1 or len(values) != H:
        raise HeadlineConditionalError(
            f"Manual path must contain exactly {H} values; received {len(values)}."
        )
    if not np.isfinite(values).all():
        raise HeadlineConditionalError("Manual path contains NaN/inf.")
    return values


def manual_condition_levels(
    posterior: SavedHeadlinePosterior,
    *,
    variable: str,
    metric: str,
    values: str | Sequence[float],
    H: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return the finite conditioned months as HICP levels."""
    if variable not in NATIVE_VARIABLES:
        raise KeyError(f"Unknown Headline variable {variable!r}.")
    metric = str(metric).lower()
    if metric not in MANUAL_METRICS:
        raise ValueError(f"metric must be one of {MANUAL_METRICS}.")
    future_dates = future_dates_for_saved(posterior, H)
    raw = parse_manual_values(values, H)
    if metric == "level":
        levels = raw
    else:
        history = posterior.inputs.native_levels[variable].astype(float)
        history.index = _normalise_months(history.index)
        den = []
        for date in future_dates:
            lag = pd.Timestamp(date) - pd.DateOffset(months=12)
            if lag not in history.index or pd.isna(history.loc[lag]):
                raise HeadlineConditionalError(
                    f"Cannot convert YoY condition: {variable} level missing at {lag.date()}."
                )
            den.append(float(history.loc[lag]))
        levels = np.asarray(den, dtype=float) * (1.0 + raw / 100.0)
    if not np.isfinite(levels).all() or np.any(levels <= 0):
        raise HeadlineConditionalError("Conditioned HICP levels must be finite and positive.")
    return future_dates, levels


def _energy_path_summaries(paths: np.ndarray) -> dict[str, list[float]]:
    """Compact pointwise Energy summaries computed directly from saved draws.

    P01..P99 are computed independently at each month.  They are therefore
    pointwise percentile paths, not a claim that one posterior draw occupies
    the same percentile rank at every horizon.
    """
    values = np.asarray(paths, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise HeadlineConditionalError(
            "Energy bridge paths must have shape (draws, >=2 monthly dates)."
        )
    if not np.isfinite(values).any(axis=0).all():
        raise HeadlineConditionalError(
            "Energy bridge has a month with no finite posterior draw."
        )
    probs = np.arange(1, 100, dtype=float) / 100.0
    pct = np.nanquantile(values, probs, axis=0)
    out = {
        "mean": np.nanmean(values, axis=0).tolist(),
        **{f"p{i:02d}": pct[i - 1].tolist() for i in range(1, 100)},
    }
    # Backward-compatible named fan statistics. They are exact aliases of
    # the corresponding directly-computed pointwise percentiles.
    out["q16"] = list(out["p16"])
    out["q50"] = list(out["p50"])
    out["q84"] = list(out["p84"])
    return out


def energy_level_paths_headline_bridge(
    *,
    dates: Sequence[pd.Timestamp],
    baseline_level_paths: np.ndarray,
    scenario_level_paths: np.ndarray,
    vintage: str,
    aggregate_run_id: str,
    forecast_name: str,
    scenario_active: bool,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the compact Energy -> Headline bridge from raw aggregate draws."""
    idx = _normalise_months(dates)
    baseline = np.asarray(baseline_level_paths, dtype=float)
    scenario = np.asarray(scenario_level_paths, dtype=float)
    if (
        scenario.ndim != 2
        or baseline.shape != scenario.shape
        or scenario.shape[1] != len(idx)
    ):
        raise HeadlineConditionalError(
            "Energy aggregate level-path dimensions are inconsistent."
        )
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        raise HeadlineConditionalError(
            "Energy aggregate bridge dates must be unique and increasing."
        )
    extra = dict(lineage or {})
    source_kind = str(extra.get("source_kind") or "energy_scenario")
    source_label = str(extra.get("source_label") or "Energy scenario")
    return {
        "contract": ENERGY_BRIDGE_CONTRACT_VERSION,
        "vintage": str(vintage),
        "aggregate_run_id": str(aggregate_run_id),
        "forecast_name": str(forecast_name),
        "scenario_active": bool(scenario_active),
        "n_draws": int(scenario.shape[0]),
        "dates": [pd.Timestamp(x).isoformat() for x in idx],
        "baseline_level": _energy_path_summaries(baseline),
        "scenario_level": _energy_path_summaries(scenario),
        "available_percentiles": list(range(1, 100)),
        "scenario_signature": extra.get("scenario_signature"),
        "scenario_components": list(extra.get("scenario_components") or []),
        "scenario_starts": dict(extra.get("scenario_starts") or {}),
        "source_kind": source_kind,
        "source_label": source_label,
        "source_metadata": {
            key: value
            for key, value in extra.items()
            if key not in {
                "scenario_signature", "scenario_components", "scenario_starts",
                "source_kind", "source_label",
            }
        },
        "percentile_contract": "pointwise_direct_from_energy_aggregate_draws",
        "transfer_contract": "month_to_month_growth_reanchored_on_headline_energy",
        "interpretation": (
            "one deterministic Energy HICP summary path is converted to monthly growth; "
            "P01..P99 are pointwise percentiles computed directly from Energy aggregate "
            "draws; full Energy-path uncertainty is not integrated out inside the Headline BVAR"
        ),
    }


def energy_outcome_headline_bridge(
    outcome: Any,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact bridge for an Energy aggregate outcome."""
    return energy_level_paths_headline_bridge(
        dates=outcome.dates,
        baseline_level_paths=np.asarray(outcome.baseline["level_paths"], dtype=float),
        scenario_level_paths=np.asarray(outcome.scenario["level_paths"], dtype=float),
        vintage=str(outcome.vintage),
        aggregate_run_id=str(outcome.aggregate_run_id),
        forecast_name=str(outcome.forecast_name),
        scenario_active=bool(outcome.scenario_active),
        lineage=lineage,
    )


def energy_headline_ratio_diagnostic(
    *,
    project_root: str | Path,
    vintage: str,
) -> dict[str, Any]:
    """Historical comparability diagnostic; never used as a rescaling factor."""
    processed = Path(project_root) / "data" / "processed" / str(vintage)
    reconstruction = historical_model_component_reconstruction(processed)
    reconstructed = pd.to_numeric(
        reconstruction["energy"]["index"], errors="coerce"
    ).rename("reconstructed")
    published = pd.to_numeric(
        reconstruction["inputs"]["indices"]["hicp_energy"], errors="coerce"
    ).rename("published")
    comparison = pd.concat([reconstructed, published], axis=1, sort=False).dropna()
    comparison = comparison.loc[
        (comparison["reconstructed"] > 0) & (comparison["published"] > 0)
    ]
    ratio = comparison["reconstructed"] / comparison["published"]
    windows: dict[str, Any] = {}
    maxima = []
    for months in (12, 24, 36):
        block = ratio.tail(months)
        if len(block) < min(months, 12):
            windows[str(months)] = {"available": False, "observations": int(len(block))}
            continue
        scale = float(block.median())
        drift = block / scale - 1.0
        maximum = float(drift.abs().max())
        maxima.append(maximum)
        windows[str(months)] = {
            "available": True,
            "observations": int(len(block)),
            "start": pd.Timestamp(block.index.min()).isoformat(),
            "end": pd.Timestamp(block.index.max()).isoformat(),
            "median_ratio": scale,
            "max_relative_ratio_drift": maximum,
            "max_relative_ratio_drift_percent": 100.0 * maximum,
        }
    max_drift = max(maxima) if maxima else None
    return {
        "windows": windows,
        "max_drift_across_available_windows": max_drift,
        "pathological_threshold": ENERGY_RATIO_PATHOLOGICAL_THRESHOLD,
        "pathological_flag": (
            None if max_drift is None else bool(max_drift > ENERGY_RATIO_PATHOLOGICAL_THRESHOLD)
        ),
    }


def energy_bridge_condition(
    energy_store: Mapping[str, Any],
    *,
    posterior: SavedHeadlinePosterior,
    target_dates: Sequence[pd.Timestamp],
    statistic: str = "mean",
    basis: str = "scenario",
    project_root: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transfer Energy-scenario monthly growth onto Headline HICP-Energy.

    For target month t, the bridge requires Energy aggregate levels at t-1 and
    t, forms their gross growth ratio, and applies it cumulatively to the
    observed Headline HICP-Energy level at the month immediately preceding the
    first target. Raw Energy aggregate levels are never imposed on Headline.
    """
    if not energy_store:
        raise HeadlineConditionalError(
            "No live HICP-Energy scenario is available. Configure the Energy scenario first."
        )
    bridge = energy_store.get("headline_bridge", energy_store)
    if str(bridge.get("contract")) != ENERGY_BRIDGE_CONTRACT_VERSION:
        raise HeadlineConditionalError(
            "The Energy scenario store does not carry the Delivery-3 Headline bridge. "
            "Restart the dashboard and recompute the Energy scenario once."
        )
    if not bool(bridge.get("scenario_active", False)):
        raise HeadlineConditionalError("The selected Energy aggregate has no active scenario.")
    same_vintage(posterior.vintage, str(bridge.get("vintage")))
    statistic = str(statistic or "mean").strip().lower()
    if statistic.startswith("q") and statistic[1:].isdigit():
        statistic = f"p{int(statistic[1:]):02d}"
    if statistic != "mean":
        match = re.fullmatch(r"p(\d{1,2})", statistic)
        if match is None or not (1 <= int(match.group(1)) <= 99):
            raise ValueError("Energy path statistic must be 'mean' or a percentile P01..P99.")
        statistic = f"p{int(match.group(1)):02d}"

    basis = str(basis or "scenario").strip().lower()
    if basis not in {"baseline", "scenario"}:
        raise ValueError("Energy condition basis must be 'baseline' or 'scenario'.")

    dates = _normalise_months(bridge.get("dates") or [])
    path_map = dict(bridge.get(f"{basis}_level") or {})
    values = np.asarray(path_map.get(statistic, []), dtype=float)
    if len(dates) != len(values):
        raise HeadlineConditionalError("Energy bridge dates/path lengths differ.")
    if dates.has_duplicates:
        raise HeadlineConditionalError("Energy bridge contains duplicate months.")
    series = pd.Series(values, index=dates).sort_index()
    targets = _normalise_months(target_dates)
    if len(targets) == 0:
        raise HeadlineConditionalError("Headline conditional target calendar is empty.")

    gross_growth = []
    for date in targets:
        previous = pd.Timestamp(date) - pd.DateOffset(months=1)
        if date not in series.index or previous not in series.index:
            raise HeadlineConditionalError(
                "Energy scenario cannot construct the required monthly growth for "
                f"{pd.Timestamp(date).date()}: both {previous.date()} and "
                f"{pd.Timestamp(date).date()} must be present in the Energy path."
            )
        prev_value = float(series.loc[previous])
        this_value = float(series.loc[date])
        if not np.isfinite(prev_value) or not np.isfinite(this_value) or prev_value <= 0 or this_value <= 0:
            raise HeadlineConditionalError("Energy scenario contains invalid HICP levels.")
        gross_growth.append(this_value / prev_value)
    gross_growth_arr = np.asarray(gross_growth, dtype=float)

    history = posterior.inputs.native_levels["hicp_energy"].astype(float).copy()
    history.index = _normalise_months(history.index)
    anchor_date = pd.Timestamp(targets[0]) - pd.DateOffset(months=1)
    if anchor_date not in history.index or pd.isna(history.loc[anchor_date]):
        raise HeadlineConditionalError(
            "Headline HICP-Energy anchor is unavailable at the month immediately "
            f"preceding the scenario: {anchor_date.date()}."
        )
    anchor_level = float(history.loc[anchor_date])
    if not np.isfinite(anchor_level) or anchor_level <= 0:
        raise HeadlineConditionalError("Headline HICP-Energy anchor level is invalid.")
    conditioned = anchor_level * np.cumprod(gross_growth_arr)

    root = _project_root_from_run(posterior.run_directory, project_root)
    ratio_diag = energy_headline_ratio_diagnostic(project_root=root, vintage=posterior.vintage)
    if ratio_diag.get("pathological_flag") is True:
        maximum = float(ratio_diag["max_drift_across_available_windows"])
        raise HeadlineConditionalError(
            "Energy -> Headline bridge rejected: historical Energy reconstruction and "
            "published HICP Energy are not comparable enough for scenario transfer "
            f"(max ratio drift {100.0 * maximum:.3f}% > 2.000%)."
        )

    lineage = {
        "source_type": "energy_scenario",
        "energy_vintage": str(bridge.get("vintage")),
        "energy_aggregate_run_id": str(bridge.get("aggregate_run_id")),
        "energy_forecast_name": str(bridge.get("forecast_name")),
        "energy_scenario_signature": bridge.get("scenario_signature"),
        "energy_scenario_components": list(bridge.get("scenario_components") or []),
        "condition_statistic": statistic,
        "condition_statistic_label": ("Posterior mean" if statistic == "mean" else f"Pointwise percentile {statistic.upper()}"),
        "energy_condition_basis": basis,
        "energy_condition_horizon_months": int(len(targets)),
        "energy_computational_horizon_months": int(COMPUTATIONAL_HORIZON),
        "energy_free_propagation_horizon_months": int(COMPUTATIONAL_HORIZON - len(targets)),
        "energy_condition_horizon_contract": (
            "Energy levels are imposed only for the first condition_horizon months; "
            "remaining months through the locked computational horizon stay latent "
            "inside the joint DK conditional forecast."
        ),
        "energy_percentile_contract": bridge.get("percentile_contract"),
        "energy_path_uncertainty_propagated": False,
        "energy_transfer_method": "mom_growth_reanchored_on_headline_energy",
        "headline_energy_anchor_date": anchor_date.isoformat(),
        "headline_energy_anchor_level": anchor_level,
        "energy_gross_growth": gross_growth_arr.tolist(),
        "energy_headline_ratio_diagnostic": ratio_diag,
    }
    return conditioned, lineage


def _aggregate_pair(
    component_forecast: Mapping[str, Any],
    posterior: SavedHeadlinePosterior,
) -> tuple[dict[str, Any], np.ndarray]:
    aggregate = aggregate_headline_draws(
        component_forecast,
        native_levels=posterior.inputs.native_levels,
        weights=posterior.inputs.weights,
        official_total=posterior.inputs.official_total,
    )
    contrib = exact_headline_yoy_contribution_paths(aggregate, inputs=posterior.inputs)
    return dict(aggregate), np.asarray(contrib, dtype=float)


def _qsummary(paths: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(paths, dtype=float)
    q = np.nanquantile(arr, [0.05, 0.16, 0.50, 0.84, 0.95], axis=0)
    return {
        "mean": np.nanmean(arr, axis=0).tolist(),
        **{
            name: q[i].tolist()
            for i, name in enumerate(("q05", "q16", "q50", "q84", "q95"))
        },
    }


def _scenario_identity(meta: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        meta, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _compact_payload(
    *,
    metadata: Mapping[str, Any],
    future_dates: pd.DatetimeIndex,
    baseline_component: Mapping[str, Any],
    conditional_component: Mapping[str, Any],
    baseline_aggregate: Mapping[str, Any],
    conditional_aggregate: Mapping[str, Any],
    baseline_contrib: np.ndarray,
    conditional_contrib: np.ndarray,
) -> dict[str, Any]:
    tail = int(baseline_component["tail_length"])
    sl = slice(tail, None)
    fans: dict[str, Any] = {}
    base_native = np.asarray(baseline_component["native_level_paths"], dtype=float)[:, sl, :]
    cond_native = np.asarray(conditional_component["native_level_paths"], dtype=float)[:, sl, :]
    base_cyoy = np.asarray(baseline_component["component_yoy_paths"], dtype=float)[:, sl, :]
    cond_cyoy = np.asarray(conditional_component["component_yoy_paths"], dtype=float)[:, sl, :]
    for j, name in enumerate(NATIVE_VARIABLES):
        fans[name] = {
            "level": {
                "baseline": _qsummary(base_native[:, :, j]),
                "conditional": _qsummary(cond_native[:, :, j]),
            },
            "yoy": {
                "baseline": _qsummary(base_cyoy[:, :, j]),
                "conditional": _qsummary(cond_cyoy[:, :, j]),
            },
        }
    base_h_level = np.asarray(baseline_aggregate["headline_level_paths"], dtype=float)[:, tail:]
    cond_h_level = np.asarray(conditional_aggregate["headline_level_paths"], dtype=float)[:, tail:]
    base_h_yoy = np.asarray(baseline_aggregate["headline_yoy_paths"], dtype=float)[:, tail:]
    cond_h_yoy = np.asarray(conditional_aggregate["headline_yoy_paths"], dtype=float)[:, tail:]
    fans["hicp_total"] = {
        "level": {"baseline": _qsummary(base_h_level), "conditional": _qsummary(cond_h_level)},
        "yoy": {"baseline": _qsummary(base_h_yoy), "conditional": _qsummary(cond_h_yoy)},
    }
    component_impacts = cond_cyoy - base_cyoy
    contrib_impact = (
        np.asarray(conditional_contrib, dtype=float)[:, tail:, :]
        - np.asarray(baseline_contrib, dtype=float)[:, tail:, :]
    )
    return {
        "contract": CONDITIONAL_CONTRACT_VERSION,
        "meta": dict(metadata),
        "dates": [pd.Timestamp(x).isoformat() for x in future_dates],
        "fans": fans,
        "headline_yoy_impact": _qsummary(cond_h_yoy - base_h_yoy),
        "headline_level_impact": _qsummary(cond_h_level - base_h_level),
        "component_yoy_impact": {
            name: _qsummary(component_impacts[:, :, j])
            for j, name in enumerate(NATIVE_VARIABLES)
        },
        "contribution_yoy_impact": {
            name: {
                **_qsummary(contrib_impact[:, :, j]),
                "mean": np.nanmean(contrib_impact[:, :, j], axis=0).tolist(),
            }
            for j, name in enumerate(NATIVE_VARIABLES)
        },
    }


def _assert_persisted_baseline_identity(
    posterior: SavedHeadlinePosterior,
    baseline: Mapping[str, Any],
) -> None:
    """Hard runtime guard: scenario baseline cannot differ from display baseline."""
    saved_path = posterior.run_directory / "forecasts" / "unconditional" / "headline_draws.npz"
    if not saved_path.is_file():
        raise HeadlineConditionalError(
            "Persisted Headline baseline draws are missing; cannot verify scenario pairing."
        )
    with np.load(saved_path, allow_pickle=False) as archive:
        if "native_level_paths" not in archive.files:
            raise HeadlineConditionalError(
                "Persisted Headline baseline has no native_level_paths."
            )
        saved_native = np.asarray(archive["native_level_paths"], dtype=float)
    actual_native = np.asarray(baseline["native_level_paths"], dtype=float)
    if not np.array_equal(actual_native, saved_native, equal_nan=True):
        finite = np.isfinite(actual_native) & np.isfinite(saved_native)
        maximum = (
            float(np.max(np.abs(actual_native[finite] - saved_native[finite])))
            if finite.any() and actual_native.shape == saved_native.shape
            else float("inf")
        )
        raise HeadlineConditionalError(
            "Scenario baseline is not bit-identical to the persisted H=12 baseline; "
            f"max_abs_error={maximum:.3e}. Refusing to display a moving baseline."
        )


def headline_energy_baseline_cache_key(
    *,
    vintage: str,
    headline_run_id: str,
    source_aggregate_run_id: str,
    forecast_name: str,
    condition_statistic: str,
    condition_horizon: int,
) -> tuple[Any, ...]:
    """Stable draw-cache key; every dimension that can change A is explicit."""
    return (
        ENERGY_BASELINE_CACHE_CONTRACT_VERSION,
        str(vintage),
        str(headline_run_id),
        str(source_aggregate_run_id),
        str(forecast_name),
        str(condition_statistic),
        int(condition_horizon),
        ENERGY_BRIDGE_CONTRACT_VERSION,
        CONDITIONAL_CONTRACT_VERSION,
    )


def clear_headline_energy_baseline_cache() -> None:
    """Clear only the process-local Energy-baseline draw cache."""
    _ENERGY_BASELINE_DRAW_CACHE.clear()


def _expand_finite_conditions(
    native_level_conditions: Mapping[str, Sequence[float]],
    condition_horizon: int,
) -> dict[str, np.ndarray]:
    if not native_level_conditions:
        raise HeadlineConditionalError("At least one native HICP condition is required.")
    expanded_conditions: dict[str, np.ndarray] = {}
    for name, raw in native_level_conditions.items():
        if name not in NATIVE_VARIABLES:
            raise KeyError(f"Unknown conditioned variable {name!r}.")
        values = np.asarray(raw, dtype=float)
        if values.ndim != 1 or len(values) != int(condition_horizon):
            raise HeadlineConditionalError(
                f"{name}: expected {int(condition_horizon)} finite condition values."
            )
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise HeadlineConditionalError(
                f"{name}: condition levels must be finite and positive."
            )
        expanded = np.full(COMPUTATIONAL_HORIZON, np.nan, dtype=float)
        expanded[: int(condition_horizon)] = values
        expanded_conditions[str(name)] = expanded
    return expanded_conditions


def _condition_draw_bundle(
    posterior: SavedHeadlinePosterior,
    *,
    native_level_conditions: Mapping[str, Sequence[float]],
    H: int,
) -> HeadlineConditionalDrawBundle:
    """Run one conditional forecast and retain draw arrays before `_qsummary`."""
    condition_horizon = int(H)
    if condition_horizon < 1 or condition_horizon > MAX_PUBLISHED_HORIZON_MONTHS:
        raise ValueError(f"H must lie in 1..{MAX_PUBLISHED_HORIZON_MONTHS}.")
    expanded_conditions = _expand_finite_conditions(
        native_level_conditions, condition_horizon
    )
    conditional = forecast_locked_headline(
        posterior.result,
        inputs=posterior.inputs,
        H=COMPUTATIONAL_HORIZON,
        n_draws=posterior.forecast_draws,
        native_level_conditions=expanded_conditions,
        allow_partial_level_conditions=True,
        simulate_future_outliers=posterior.simulate_future_outliers,
        seed=posterior.forecast_seed,
    )
    aggregate, contrib = _aggregate_pair(conditional, posterior)
    tail = int(conditional["tail_length"])
    future_dates = pd.DatetimeIndex(conditional["future_dates"], name="date")
    expected = future_dates_for_saved(posterior, COMPUTATIONAL_HORIZON)
    if not future_dates.equals(expected):
        raise HeadlineConditionalError(
            "Conditional H=12 future calendar changed unexpectedly."
        )
    return HeadlineConditionalDrawBundle(
        draw_indices=np.asarray(conditional["draw_indices"], dtype=int).copy(),
        future_dates=future_dates,
        native_level_paths=np.asarray(
            conditional["native_level_paths"], dtype=float
        )[:, tail:, :].copy(),
        component_yoy_paths=np.asarray(
            conditional["component_yoy_paths"], dtype=float
        )[:, tail:, :].copy(),
        headline_level_paths=np.asarray(
            aggregate["headline_level_paths"], dtype=float
        )[:, tail:].copy(),
        headline_yoy_paths=np.asarray(
            aggregate["headline_yoy_paths"], dtype=float
        )[:, tail:].copy(),
        contribution_yoy_paths=np.asarray(
            contrib, dtype=float
        )[:, tail:, :].copy(),
    )


def _assert_energy_marginal_pairing(
    baseline: HeadlineConditionalDrawBundle,
    scenario: HeadlineConditionalDrawBundle,
) -> None:
    if not np.array_equal(baseline.draw_indices, scenario.draw_indices):
        raise HeadlineConditionalError(
            "Energy baseline/scenario posterior draw pairing failed."
        )
    if not baseline.future_dates.equals(scenario.future_dates):
        raise HeadlineConditionalError(
            "Energy baseline/scenario Headline calendars differ."
        )
    checks = (
        ("native_level_paths", baseline.native_level_paths, scenario.native_level_paths),
        ("component_yoy_paths", baseline.component_yoy_paths, scenario.component_yoy_paths),
        ("headline_level_paths", baseline.headline_level_paths, scenario.headline_level_paths),
        ("headline_yoy_paths", baseline.headline_yoy_paths, scenario.headline_yoy_paths),
        (
            "contribution_yoy_paths",
            baseline.contribution_yoy_paths,
            scenario.contribution_yoy_paths,
        ),
    )
    for name, left, right in checks:
        if left.shape != right.shape:
            raise HeadlineConditionalError(
                f"Energy baseline/scenario {name} shapes differ: "
                f"{left.shape} vs {right.shape}."
            )


def run_saved_headline_energy_marginal_draws(
    run_directory: str | Path,
    *,
    energy_store: Mapping[str, Any],
    source_aggregate_run_id: str,
    forecast_name: str,
    H: int,
    statistic: str = "mean",
    lineage: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return paired draw-level A/B for the marginal Energy-scenario effect.

    A = Headline | Energy baseline path.
    B = Headline | Energy scenario path.
    The first H Energy levels are conditioned; months H+1..12 remain latent.
    """
    condition_horizon = int(H)
    posterior = load_saved_headline_posterior(
        run_directory, project_root=project_root
    )
    target_dates = future_dates_for_saved(posterior, condition_horizon)
    bridge = dict((energy_store or {}).get("headline_bridge") or {})
    store_meta = dict((energy_store or {}).get("meta") or {})
    stored_source = str(store_meta.get("aggregate_run_id") or "")
    if stored_source and stored_source != str(source_aggregate_run_id):
        raise HeadlineConditionalError(
            "Energy source aggregate mismatch: "
            f"store={stored_source}, requested={source_aggregate_run_id}."
        )
    bridge_forecast = str(bridge.get("forecast_name") or "")
    if bridge_forecast and bridge_forecast != str(forecast_name):
        raise HeadlineConditionalError(
            "Energy forecast contract mismatch: "
            f"bridge={bridge_forecast}, requested={forecast_name}."
        )

    baseline_values, baseline_lineage = energy_bridge_condition(
        energy_store,
        posterior=posterior,
        target_dates=target_dates,
        statistic=statistic,
        basis="baseline",
        project_root=project_root,
    )
    scenario_values, scenario_lineage = energy_bridge_condition(
        energy_store,
        posterior=posterior,
        target_dates=target_dates,
        statistic=statistic,
        basis="scenario",
        project_root=project_root,
    )
    statistic_effective = str(scenario_lineage["condition_statistic"])
    cache_key = headline_energy_baseline_cache_key(
        vintage=posterior.vintage,
        headline_run_id=posterior.run_id,
        source_aggregate_run_id=str(source_aggregate_run_id),
        forecast_name=str(forecast_name),
        condition_statistic=statistic_effective,
        condition_horizon=condition_horizon,
    )
    cached = _ENERGY_BASELINE_DRAW_CACHE.get(cache_key)
    cache_hit = cached is not None
    if cached is None:
        baseline = _condition_draw_bundle(
            posterior,
            native_level_conditions={"hicp_energy": baseline_values},
            H=condition_horizon,
        )
        _ENERGY_BASELINE_DRAW_CACHE[cache_key] = (
            np.asarray(baseline_values, dtype=float).copy(),
            baseline,
        )
    else:
        cached_values, baseline = cached
        if not np.array_equal(
            np.asarray(cached_values, dtype=float),
            np.asarray(baseline_values, dtype=float),
            equal_nan=True,
        ):
            raise HeadlineConditionalError(
                "Energy baseline cache key resolved to different condition levels; "
                "source aggregate immutability contract was violated."
            )

    scenario = _condition_draw_bundle(
        posterior,
        native_level_conditions={"hicp_energy": scenario_values},
        H=condition_horizon,
    )
    _assert_energy_marginal_pairing(baseline, scenario)

    meta: dict[str, Any] = {
        "scenario_contract_version": CONDITIONAL_CONTRACT_VERSION,
        "headline_vintage": posterior.vintage,
        "headline_run_id": posterior.run_id,
        "source_aggregate_run_id": str(source_aggregate_run_id),
        "scenario_aggregate_run_id": str(bridge.get("aggregate_run_id") or ""),
        "forecast_name": str(forecast_name),
        "condition_variables": ["hicp_energy"],
        "condition_dates": [x.isoformat() for x in target_dates],
        "baseline_condition_values": np.asarray(baseline_values, dtype=float).tolist(),
        "scenario_condition_values": np.asarray(scenario_values, dtype=float).tolist(),
        "condition_statistic": statistic_effective,
        "condition_statistic_label": scenario_lineage.get("condition_statistic_label"),
        "condition_horizon": condition_horizon,
        "computational_horizon": int(COMPUTATIONAL_HORIZON),
        "conditioned_horizon_months": condition_horizon,
        "free_propagation_horizon_months": int(
            COMPUTATIONAL_HORIZON - condition_horizon
        ),
        "horizon_interpretation": (
            "Energy is conditioned for the first condition_horizon months; "
            "the remaining months through computational_horizon are latent "
            "joint-DK propagation."
        ),
        "n_draws": posterior.forecast_draws,
        "forecast_seed": posterior.forecast_seed,
        "simulate_future_outliers": posterior.simulate_future_outliers,
        "paired_baseline_conditional": True,
        "paired_energy_baseline_scenario": True,
        "impact_reference": "energy_baseline_conditioned",
        "impact_definition": (
            "Headline | Energy scenario path minus "
            "Headline | Energy baseline path"
        ),
        "bvar_reestimated": False,
        "energy_path_uncertainty_propagated": False,
        "baseline_cache_contract": ENERGY_BASELINE_CACHE_CONTRACT_VERSION,
        "baseline_cache_key": list(cache_key),
        "lineage_verification_status": posterior.integrity.get(
            "verification_status", "unverified"
        ),
        "publication_horizon_max_months": MAX_PUBLISHED_HORIZON_MONTHS,
        "baseline_energy_condition_lineage": dict(baseline_lineage),
        "scenario_energy_condition_lineage": dict(scenario_lineage),
    }
    meta.update(dict(lineage or {}))
    meta["scenario_id"] = _scenario_identity(meta)
    # Runtime cache state is diagnostic only and must never change scenario identity.
    meta["baseline_cache_hit"] = bool(cache_hit)
    return {
        "meta": meta,
        "future_dates": scenario.future_dates,
        "baseline": baseline,
        "scenario": scenario,
    }


def compact_headline_energy_marginal(
    draw_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact A/B draw bundles only after draw-wise differences remain available."""
    baseline = draw_result["baseline"]
    scenario = draw_result["scenario"]
    if not isinstance(baseline, HeadlineConditionalDrawBundle) or not isinstance(
        scenario, HeadlineConditionalDrawBundle
    ):
        raise TypeError(
            "draw_result does not contain HeadlineConditionalDrawBundle objects."
        )
    _assert_energy_marginal_pairing(baseline, scenario)
    baseline_component = {
        "tail_length": 0,
        "native_level_paths": baseline.native_level_paths,
        "component_yoy_paths": baseline.component_yoy_paths,
    }
    scenario_component = {
        "tail_length": 0,
        "native_level_paths": scenario.native_level_paths,
        "component_yoy_paths": scenario.component_yoy_paths,
    }
    baseline_aggregate = {
        "headline_level_paths": baseline.headline_level_paths,
        "headline_yoy_paths": baseline.headline_yoy_paths,
    }
    scenario_aggregate = {
        "headline_level_paths": scenario.headline_level_paths,
        "headline_yoy_paths": scenario.headline_yoy_paths,
    }
    return _compact_payload(
        metadata=dict(draw_result.get("meta") or {}),
        future_dates=scenario.future_dates,
        baseline_component=baseline_component,
        conditional_component=scenario_component,
        baseline_aggregate=baseline_aggregate,
        conditional_aggregate=scenario_aggregate,
        baseline_contrib=baseline.contribution_yoy_paths,
        conditional_contrib=scenario.contribution_yoy_paths,
    )


def run_saved_headline_energy_marginal(
    run_directory: str | Path,
    *,
    energy_store: Mapping[str, Any],
    source_aggregate_run_id: str,
    forecast_name: str,
    H: int,
    statistic: str = "mean",
    lineage: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    """Browser-safe B-A payload using draw-level paired Energy conditioning.

    When ``persist=True`` the saved scenario keeps the same artifact filenames
    as the existing Headline conditional path, but ``baseline_*`` now means the
    Energy-baseline-conditioned A object. Metadata makes that denominator
    explicit through ``impact_reference`` and ``impact_definition``.
    """
    draw_result = run_saved_headline_energy_marginal_draws(
        run_directory,
        energy_store=energy_store,
        source_aggregate_run_id=source_aggregate_run_id,
        forecast_name=forecast_name,
        H=H,
        statistic=statistic,
        lineage=lineage,
        project_root=project_root,
    )
    payload = compact_headline_energy_marginal(draw_result)
    if not persist:
        return payload

    baseline = draw_result["baseline"]
    scenario = draw_result["scenario"]
    if not isinstance(baseline, HeadlineConditionalDrawBundle) or not isinstance(
        scenario, HeadlineConditionalDrawBundle
    ):
        raise TypeError(
            "draw_result does not contain HeadlineConditionalDrawBundle objects."
        )
    _assert_energy_marginal_pairing(baseline, scenario)
    meta = dict(payload.get("meta") or {})
    scenario_id = str(meta.get("scenario_id") or "")
    if not scenario_id:
        raise HeadlineConditionalError(
            "Energy marginal payload has no scenario_id; refusing to persist."
        )
    directory = Path(run_directory) / "scenarios" / scenario_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    np.savez_compressed(
        directory / "scenario_draws.npz",
        future_dates=scenario.future_dates.astype("datetime64[ns]").to_numpy(),
        draw_indices=np.asarray(scenario.draw_indices, dtype=int),
        baseline_native_level_paths=np.asarray(
            baseline.native_level_paths, dtype=float
        ),
        conditional_native_level_paths=np.asarray(
            scenario.native_level_paths, dtype=float
        ),
        baseline_component_yoy_paths=np.asarray(
            baseline.component_yoy_paths, dtype=float
        ),
        conditional_component_yoy_paths=np.asarray(
            scenario.component_yoy_paths, dtype=float
        ),
        baseline_headline_level_paths=np.asarray(
            baseline.headline_level_paths, dtype=float
        ),
        conditional_headline_level_paths=np.asarray(
            scenario.headline_level_paths, dtype=float
        ),
        baseline_headline_yoy_paths=np.asarray(
            baseline.headline_yoy_paths, dtype=float
        ),
        conditional_headline_yoy_paths=np.asarray(
            scenario.headline_yoy_paths, dtype=float
        ),
        baseline_yoy_contribution_paths=np.asarray(
            baseline.contribution_yoy_paths, dtype=float
        ),
        conditional_yoy_contribution_paths=np.asarray(
            scenario.contribution_yoy_paths, dtype=float
        ),
    )
    payload["meta"]["directory"] = str(directory)
    (directory / "display.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload


def run_saved_headline_conditional(
    run_directory: str | Path,
    *,
    native_level_conditions: Mapping[str, Sequence[float]],
    H: int,
    lineage: Mapping[str, Any] | None = None,
    n_draws: int | None = None,
    seed: int | None = None,
    project_root: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Paired H=12 baseline/conditional forecast from one saved posterior.

    ``H`` is the *condition horizon* only. The stochastic forecast is always
    solved over the locked 12-month computational horizon so the baseline is
    invariant to the UI display/condition horizon.
    """
    condition_horizon = int(H)
    if condition_horizon < 1 or condition_horizon > MAX_PUBLISHED_HORIZON_MONTHS:
        raise ValueError(
            f"H must lie in 1..{MAX_PUBLISHED_HORIZON_MONTHS}."
        )
    posterior = load_saved_headline_posterior(run_directory, project_root=project_root)
    contract_draws = posterior.forecast_draws
    contract_seed = posterior.forecast_seed
    contract_outliers = posterior.simulate_future_outliers
    if n_draws is not None and int(n_draws) != contract_draws:
        raise HeadlineConditionalError(
            "Conditional n_draws must equal the persisted baseline contract: "
            f"requested={int(n_draws)}, persisted={contract_draws}."
        )
    if seed is not None and int(seed) != contract_seed:
        raise HeadlineConditionalError(
            "Conditional seed must equal the persisted baseline contract: "
            f"requested={int(seed)}, persisted={contract_seed}."
        )

    condition_dates = future_dates_for_saved(posterior, condition_horizon)
    full_future_dates = future_dates_for_saved(posterior, COMPUTATIONAL_HORIZON)
    finite_conditions: dict[str, np.ndarray] = {}
    expanded_conditions: dict[str, np.ndarray] = {}
    if not native_level_conditions:
        raise HeadlineConditionalError("At least one native HICP condition is required.")
    for name, raw in native_level_conditions.items():
        if name not in NATIVE_VARIABLES:
            raise KeyError(f"Unknown conditioned variable {name!r}.")
        values = np.asarray(raw, dtype=float)
        if values.ndim != 1 or len(values) != condition_horizon:
            raise HeadlineConditionalError(
                f"{name}: expected {condition_horizon} finite condition values."
            )
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise HeadlineConditionalError(
                f"{name}: condition levels must be finite and positive."
            )
        finite_conditions[str(name)] = values
        expanded = np.full(COMPUTATIONAL_HORIZON, np.nan, dtype=float)
        expanded[:condition_horizon] = values
        expanded_conditions[str(name)] = expanded

    baseline = forecast_locked_headline(
        posterior.result,
        inputs=posterior.inputs,
        H=COMPUTATIONAL_HORIZON,
        n_draws=contract_draws,
        native_level_conditions=None,
        simulate_future_outliers=contract_outliers,
        seed=contract_seed,
    )
    _assert_persisted_baseline_identity(posterior, baseline)
    conditional = forecast_locked_headline(
        posterior.result,
        inputs=posterior.inputs,
        H=COMPUTATIONAL_HORIZON,
        n_draws=contract_draws,
        native_level_conditions=expanded_conditions,
        allow_partial_level_conditions=True,
        simulate_future_outliers=contract_outliers,
        seed=contract_seed,
    )
    if not np.array_equal(
        np.asarray(baseline["draw_indices"]), np.asarray(conditional["draw_indices"])
    ):
        raise HeadlineConditionalError("Baseline/conditional posterior draw pairing failed.")
    if not pd.DatetimeIndex(baseline["path_dates"]).equals(
        pd.DatetimeIndex(conditional["path_dates"])
    ):
        raise HeadlineConditionalError("Baseline/conditional calendars differ.")

    base_agg, base_contrib = _aggregate_pair(baseline, posterior)
    cond_agg, cond_contrib = _aggregate_pair(conditional, posterior)
    future_dates = pd.DatetimeIndex(conditional["future_dates"], name="date")
    if not future_dates.equals(full_future_dates):
        raise HeadlineConditionalError("Conditional H=12 future calendar changed unexpectedly.")

    meta: dict[str, Any] = {
        "scenario_contract_version": CONDITIONAL_CONTRACT_VERSION,
        "headline_vintage": posterior.vintage,
        "headline_run_id": posterior.run_id,
        "condition_variables": sorted(finite_conditions),
        "condition_dates": [x.isoformat() for x in condition_dates],
        "condition_values": {k: v.tolist() for k, v in finite_conditions.items()},
        "condition_horizon": condition_horizon,
        "computational_horizon": COMPUTATIONAL_HORIZON,
        "n_draws": contract_draws,
        "forecast_seed": contract_seed,
        "simulate_future_outliers": contract_outliers,
        "paired_baseline_conditional": True,
        "persisted_baseline_bit_identical": True,
        "bvar_reestimated": False,
        "lineage_verification_status": posterior.integrity.get("verification_status", "unverified"),
        "publication_horizon_max_months": MAX_PUBLISHED_HORIZON_MONTHS,
    }
    meta.update(dict(lineage or {}))
    scenario_id = _scenario_identity(meta)
    meta["scenario_id"] = scenario_id

    payload = _compact_payload(
        metadata=meta,
        future_dates=future_dates,
        baseline_component=baseline,
        conditional_component=conditional,
        baseline_aggregate=base_agg,
        conditional_aggregate=cond_agg,
        baseline_contrib=base_contrib,
        conditional_contrib=cond_contrib,
    )

    if persist:
        directory = posterior.run_directory / "scenarios" / scenario_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        tail = int(conditional["tail_length"])
        np.savez_compressed(
            directory / "scenario_draws.npz",
            future_dates=future_dates.astype("datetime64[ns]").to_numpy(),
            draw_indices=np.asarray(conditional["draw_indices"], dtype=int),
            baseline_native_level_paths=np.asarray(
                baseline["native_level_paths"], dtype=float
            )[:, tail:, :],
            conditional_native_level_paths=np.asarray(
                conditional["native_level_paths"], dtype=float
            )[:, tail:, :],
            baseline_component_yoy_paths=np.asarray(
                baseline["component_yoy_paths"], dtype=float
            )[:, tail:, :],
            conditional_component_yoy_paths=np.asarray(
                conditional["component_yoy_paths"], dtype=float
            )[:, tail:, :],
            baseline_headline_level_paths=np.asarray(
                base_agg["headline_level_paths"], dtype=float
            )[:, tail:],
            conditional_headline_level_paths=np.asarray(
                cond_agg["headline_level_paths"], dtype=float
            )[:, tail:],
            baseline_headline_yoy_paths=np.asarray(
                base_agg["headline_yoy_paths"], dtype=float
            )[:, tail:],
            conditional_headline_yoy_paths=np.asarray(
                cond_agg["headline_yoy_paths"], dtype=float
            )[:, tail:],
            baseline_yoy_contribution_paths=np.asarray(base_contrib, dtype=float)[:, tail:, :],
            conditional_yoy_contribution_paths=np.asarray(cond_contrib, dtype=float)[:, tail:, :],
        )
        (directory / "display.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        payload["meta"]["directory"] = str(directory)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_energy_aggregate_condition(
    aggregate_directory: str | Path,
    *,
    posterior: SavedHeadlinePosterior,
    source_aggregate_run_id: str,
    forecast_name: str,
    statistic: str = "mean",
    project_root: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the baseline-only selected-aggregate Energy condition.

    This contract is intentionally separate from ``energy_bridge_condition``:
    an ordinary saved Energy aggregate is not a scenario and therefore never
    receives or bypasses the Lot-1 ``scenario_active`` gate.

    The selected aggregate contributes one deterministic summary path only.
    Monthly growth rates are transferred onto the published Headline
    HICP-Energy scale.  Raw Energy aggregate index levels are never imposed.
    """
    statistic = str(statistic or "mean").strip().lower()
    if statistic != "mean":
        raise ValueError(
            "Selected-aggregate Headline contribution currently supports only "
            "statistic='mean'. Full upstream Energy path uncertainty is not "
            "propagated through the Headline conditional."
        )

    aggregate_dir = Path(aggregate_directory).resolve()
    if not aggregate_dir.is_dir():
        raise HeadlineConditionalError(
            f"Selected Energy aggregate directory does not exist: {aggregate_dir}"
        )

    requested_run_id = str(source_aggregate_run_id or "").strip()
    if not requested_run_id:
        raise HeadlineConditionalError("source_aggregate_run_id is required.")
    if aggregate_dir.name != requested_run_id:
        raise HeadlineConditionalError(
            "Selected Energy aggregate directory/run mismatch: "
            f"directory={aggregate_dir.name}, requested={requested_run_id}."
        )

    metadata_path = aggregate_dir / "metadata.json"
    display_path = aggregate_dir / "display_v1.parquet"
    if not metadata_path.is_file():
        raise HeadlineConditionalError(
            f"Selected Energy aggregate metadata is missing: {metadata_path}"
        )
    if not display_path.is_file():
        raise HeadlineConditionalError(
            f"Selected Energy aggregate display is missing: {display_path}"
        )

    try:
        aggregate_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HeadlineConditionalError(
            f"Could not read selected Energy aggregate metadata: {metadata_path}"
        ) from exc

    stored_run_id = str(aggregate_meta.get("aggregate_run_id") or "").strip()
    if stored_run_id and stored_run_id != requested_run_id:
        raise HeadlineConditionalError(
            "Selected Energy aggregate metadata/run mismatch: "
            f"metadata={stored_run_id}, requested={requested_run_id}."
        )
    aggregate_vintage = str(aggregate_meta.get("vintage") or "").strip()
    if not aggregate_vintage:
        raise HeadlineConditionalError(
            "Selected Energy aggregate metadata has no vintage."
        )
    same_vintage(posterior.vintage, aggregate_vintage)

    stored_forecast = str(aggregate_meta.get("forecast_name") or "").strip()
    requested_forecast = str(forecast_name or "").strip()
    if not requested_forecast:
        raise HeadlineConditionalError("forecast_name is required.")
    if stored_forecast and stored_forecast != requested_forecast:
        raise HeadlineConditionalError(
            "Selected Energy aggregate forecast mismatch: "
            f"metadata={stored_forecast}, requested={requested_forecast}."
        )

    try:
        frame = pd.read_parquet(display_path)
    except Exception as exc:
        raise HeadlineConditionalError(
            f"Could not read selected Energy aggregate display: {display_path}"
        ) from exc

    required = {"record_type", "series", "metric", "basis", "date", "value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HeadlineConditionalError(
            "Selected Energy aggregate display is missing required columns: "
            + ", ".join(missing)
        )

    mask = (
        frame["record_type"].astype(str).eq("fan")
        & frame["series"].astype(str).eq("hicp_energy")
        & frame["metric"].astype(str).eq("level")
        & frame["basis"].astype(str).eq("baseline")
    )
    if "segment" in frame.columns:
        mask &= frame["segment"].astype(str).isin(["nowcast", "forecast"])
    selected = frame.loc[mask, ["date", "value"]].copy()
    if selected.empty:
        raise HeadlineConditionalError(
            "Selected Energy aggregate display has no baseline HICP-Energy "
            "model-path level rows."
        )

    selected["date"] = (
        pd.DatetimeIndex(pd.to_datetime(selected["date"], errors="coerce"))
        .to_period("M")
        .to_timestamp(how="start")
    )
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    if selected["date"].isna().any() or selected["value"].isna().any():
        raise HeadlineConditionalError(
            "Selected Energy aggregate baseline contains invalid dates/levels."
        )
    if selected["date"].duplicated(keep=False).any():
        duplicates = sorted(
            {
                pd.Timestamp(x).date().isoformat()
                for x in selected.loc[
                    selected["date"].duplicated(keep=False), "date"
                ]
            }
        )
        raise HeadlineConditionalError(
            "Selected Energy aggregate baseline must contain exactly one row per "
            "month; duplicate months: " + ", ".join(duplicates)
        )
    selected = selected.sort_values("date")
    if not np.isfinite(selected["value"].to_numpy(dtype=float)).all():
        raise HeadlineConditionalError(
            "Selected Energy aggregate baseline contains non-finite levels."
        )
    if np.any(selected["value"].to_numpy(dtype=float) <= 0):
        raise HeadlineConditionalError(
            "Selected Energy aggregate baseline levels must be strictly positive."
        )

    full_future = future_dates_for_saved(posterior, COMPUTATIONAL_HORIZON)
    first_future = pd.Timestamp(full_future[0])
    source_anchor_date = first_future - pd.DateOffset(months=1)
    source_anchor_date = source_anchor_date.to_period("M").to_timestamp(how="start")
    level_map = {
        pd.Timestamp(row.date): float(row.value)
        for row in selected.itertuples(index=False)
    }
    if source_anchor_date not in level_map:
        raise HeadlineConditionalError(
            "Selected Energy aggregate baseline lacks the month immediately "
            "preceding the first Headline forecast month: "
            f"{source_anchor_date.date()}."
        )

    available: list[pd.Timestamp] = []
    first_missing: pd.Timestamp | None = None
    for raw_date in full_future:
        date = pd.Timestamp(raw_date)
        if date in level_map:
            if first_missing is not None:
                raise HeadlineConditionalError(
                    "Selected Energy aggregate baseline future path is not "
                    "contiguous: "
                    f"{first_missing.date()} is missing but {date.date()} is present."
                )
            available.append(date)
        else:
            first_missing = date
    if not available:
        raise HeadlineConditionalError(
            "Selected Energy aggregate baseline has no consecutive future month "
            "from the first Headline forecast date."
        )

    condition_horizon = len(available)
    source_dates = [source_anchor_date, *available]
    source_levels = np.asarray([level_map[x] for x in source_dates], dtype=float)
    gross_growth = source_levels[1:] / source_levels[:-1]
    if not np.isfinite(gross_growth).all() or np.any(gross_growth <= 0):
        raise HeadlineConditionalError(
            "Selected Energy aggregate monthly growth path is invalid."
        )

    history = posterior.inputs.native_levels["hicp_energy"].astype(float).copy()
    history.index = _normalise_months(history.index)
    if source_anchor_date not in history.index or pd.isna(
        history.loc[source_anchor_date]
    ):
        raise HeadlineConditionalError(
            "Headline HICP-Energy anchor is missing at "
            f"{source_anchor_date.date()}."
        )
    headline_anchor = float(history.loc[source_anchor_date])
    if not np.isfinite(headline_anchor) or headline_anchor <= 0:
        raise HeadlineConditionalError(
            "Headline HICP-Energy anchor must be finite and strictly positive."
        )
    conditioned_levels = headline_anchor * np.cumprod(gross_growth)

    reconstructed_growth = np.r_[
        conditioned_levels[0] / headline_anchor,
        conditioned_levels[1:] / conditioned_levels[:-1],
    ]
    growth_error = float(
        np.max(np.abs(reconstructed_growth - gross_growth))
    )
    if growth_error > 1e-12:
        raise HeadlineConditionalError(
            "Selected Energy aggregate growth-transfer identity failed: "
            f"max_abs_error={growth_error:.3e}."
        )

    root = _project_root_from_run(
        posterior.run_directory,
        project_root,
    )
    ratio_diag = energy_headline_ratio_diagnostic(
        project_root=root,
        vintage=posterior.vintage,
    )
    if ratio_diag.get("pathological_flag") is True:
        maximum = float(ratio_diag["max_drift_across_available_windows"])
        raise HeadlineConditionalError(
            "Selected aggregate Energy/Headline historical ratio is pathological: "
            f"max drift={100.0 * maximum:.3f}% exceeds "
            f"{100.0 * ENERGY_RATIO_PATHOLOGICAL_THRESHOLD:.3f}%."
        )

    upstream_draws = aggregate_meta.get("n_aggregate_draws_effective")
    try:
        upstream_draws = (
            None if upstream_draws is None else int(upstream_draws)
        )
    except (TypeError, ValueError):
        upstream_draws = None

    lineage = {
        "contract": SELECTED_ENERGY_BASELINE_CONTRACT_VERSION,
        "headline_vintage": posterior.vintage,
        "headline_run_id": posterior.run_id,
        "source_aggregate_run_id": requested_run_id,
        "source_aggregate_directory": str(aggregate_dir),
        "source_aggregate_display_sha256": _sha256_file(display_path),
        "forecast_name": requested_forecast,
        "condition_statistic": "mean",
        "condition_statistic_label": "Posterior mean",
        "condition_variables": ["hicp_energy"],
        "condition_dates": [x.isoformat() for x in available],
        "condition_values": conditioned_levels.tolist(),
        "source_energy_anchor_date": source_anchor_date.isoformat(),
        "source_energy_anchor_level": float(source_levels[0]),
        "headline_energy_anchor_date": source_anchor_date.isoformat(),
        "headline_energy_anchor_level": headline_anchor,
        "energy_gross_growth": gross_growth.tolist(),
        "growth_transfer_max_abs_error": growth_error,
        "energy_transfer_method": "mom_growth_reanchored_on_headline_energy",
        "condition_horizon": condition_horizon,
        "computational_horizon": int(COMPUTATIONAL_HORIZON),
        "conditioned_horizon_months": condition_horizon,
        "free_propagation_horizon_months": int(
            COMPUTATIONAL_HORIZON - condition_horizon
        ),
        "horizon_interpretation": (
            "All consecutive selected-aggregate future months available from "
            "the first Headline forecast month are conditioned; remaining months "
            "through the locked H=12 DK problem stay latent."
        ),
        "upstream_energy_draws_available": upstream_draws,
        "upstream_energy_draws_propagated": 0,
        "energy_path_uncertainty_propagated": False,
        "energy_uncertainty_interpretation": (
            "The selected Energy aggregate contributes one deterministic "
            "posterior-mean path. Its upstream aggregate draws are not propagated "
            "through the Headline conditional."
        ),
        "scenario_bridge_used": False,
        "energy_headline_ratio_diagnostic": ratio_diag,
    }
    return conditioned_levels, lineage


def _selected_energy_contribution_diagnostics(
    bundle: HeadlineConditionalDrawBundle,
    *,
    conditioned_levels: np.ndarray,
    condition_horizon: int,
) -> dict[str, float]:
    energy_index = list(NATIVE_VARIABLES).index("hicp_energy")
    imposed = np.asarray(
        bundle.native_level_paths[:, : int(condition_horizon), energy_index],
        dtype=float,
    )
    expected = np.asarray(conditioned_levels, dtype=float)[None, :]
    condition_error = float(np.max(np.abs(imposed - expected)))

    contrib = np.asarray(bundle.contribution_yoy_paths, dtype=float)
    headline = np.asarray(bundle.headline_yoy_paths, dtype=float)
    complete = np.isfinite(headline) & np.all(np.isfinite(contrib), axis=2)
    if not complete.any():
        raise HeadlineConditionalError(
            "Selected-aggregate conditional forecast has no complete finite "
            "Headline-contribution cells."
        )
    summed = np.sum(contrib, axis=2)
    additivity_error = float(
        np.max(np.abs(summed[complete] - headline[complete]))
    )

    if condition_error > 1e-8:
        raise HeadlineConditionalError(
            "Selected-aggregate Energy condition is not respected: "
            f"max_abs_error={condition_error:.3e}."
        )
    if additivity_error > 1e-9:
        raise HeadlineConditionalError(
            "Selected-aggregate Headline contributions are not additive: "
            f"max_abs_error={additivity_error:.3e} pp."
        )
    return {
        "condition_level_max_abs_error": condition_error,
        "contribution_additivity_max_abs_error": additivity_error,
    }


def run_saved_headline_selected_energy_contribution_draws(
    run_directory: str | Path,
    *,
    aggregate_directory: str | Path,
    source_aggregate_run_id: str,
    forecast_name: str,
    statistic: str = "mean",
    lineage: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Condition Headline on the selected saved Energy aggregate baseline.

    The selected aggregate is an ordinary baseline object, not a scenario.
    All consecutive future Energy months available from the first Headline
    forecast month are conditioned.  The rest of the locked H=12 problem is
    latent joint-DK propagation.

    The returned contribution paths are exact draw-wise Headline YoY
    contributions.  Upstream Energy aggregate draws are not propagated;
    only the selected aggregate posterior-mean level path is transferred.
    """
    posterior = load_saved_headline_posterior(
        run_directory,
        project_root=project_root,
    )
    conditioned_levels, energy_lineage = _selected_energy_aggregate_condition(
        aggregate_directory,
        posterior=posterior,
        source_aggregate_run_id=source_aggregate_run_id,
        forecast_name=forecast_name,
        statistic=statistic,
        project_root=project_root,
    )
    condition_horizon = int(energy_lineage["condition_horizon"])
    bundle = _condition_draw_bundle(
        posterior,
        native_level_conditions={"hicp_energy": conditioned_levels},
        H=condition_horizon,
    )
    diagnostics = _selected_energy_contribution_diagnostics(
        bundle,
        conditioned_levels=conditioned_levels,
        condition_horizon=condition_horizon,
    )

    extra_lineage = dict(lineage or {})
    meta: dict[str, Any] = {
        **extra_lineage,
        **energy_lineage,
        **diagnostics,
        "headline_conditional_draws": int(posterior.forecast_draws),
        "forecast_seed": int(posterior.forecast_seed),
        "simulate_future_outliers": bool(posterior.simulate_future_outliers),
        "contribution_definition": (
            "Exact chain-linked Headline YoY contribution under conditioning "
            "on the selected bottom-up HICP Energy posterior-mean path."
        ),
        "contribution_band_interpretation": (
            "Bands, where non-degenerate, are dispersion across saved Headline "
            "posterior draws conditional on one deterministic selected-Energy "
            "posterior-mean path; they do not include upstream Energy aggregate "
            "path uncertainty."
        ),
        "bvar_reestimated": False,
        "lineage_verification_status": posterior.integrity.get(
            "verification_status", "unverified"
        ),
    }
    return {
        "meta": meta,
        "future_dates": bundle.future_dates,
        "condition_values": np.asarray(conditioned_levels, dtype=float).copy(),
        "conditional": bundle,
    }


def compact_headline_selected_energy_contribution(
    draw_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact the selected-aggregate conditional contribution draw bundle."""
    bundle = draw_result.get("conditional")
    if not isinstance(bundle, HeadlineConditionalDrawBundle):
        raise TypeError(
            "draw_result does not contain a HeadlineConditionalDrawBundle."
        )
    energy_index = list(NATIVE_VARIABLES).index("hicp_energy")
    contributions = np.asarray(bundle.contribution_yoy_paths, dtype=float)
    component_yoy = np.asarray(bundle.component_yoy_paths, dtype=float)
    native_levels = np.asarray(bundle.native_level_paths, dtype=float)
    meta = dict(draw_result.get("meta") or {})
    condition_horizon = int(meta.get("condition_horizon") or 0)
    if condition_horizon < 1 or condition_horizon > COMPUTATIONAL_HORIZON:
        raise HeadlineConditionalError(
            "Selected-aggregate contribution payload has an invalid condition horizon."
        )

    return {
        "contract": SELECTED_ENERGY_BASELINE_CONTRACT_VERSION,
        "meta": meta,
        "dates": [pd.Timestamp(x).isoformat() for x in bundle.future_dates],
        "conditioned": [
            bool(i < condition_horizon)
            for i in range(len(bundle.future_dates))
        ],
        "energy_contribution_yoy": _qsummary(
            contributions[:, :, energy_index]
        ),
        "component_yoy_contribution": {
            name: _qsummary(contributions[:, :, j])
            for j, name in enumerate(NATIVE_VARIABLES)
        },
        "headline_yoy": _qsummary(bundle.headline_yoy_paths),
        "energy_component_yoy": _qsummary(
            component_yoy[:, :, energy_index]
        ),
        "energy_level": _qsummary(
            native_levels[:, :, energy_index]
        ),
    }


def run_saved_headline_selected_energy_contribution(
    run_directory: str | Path,
    *,
    aggregate_directory: str | Path,
    source_aggregate_run_id: str,
    forecast_name: str,
    statistic: str = "mean",
    lineage: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Public compact baseline-only selected-Energy contribution runner."""
    draw_result = run_saved_headline_selected_energy_contribution_draws(
        run_directory,
        aggregate_directory=aggregate_directory,
        source_aggregate_run_id=source_aggregate_run_id,
        forecast_name=forecast_name,
        statistic=statistic,
        lineage=lineage,
        project_root=project_root,
    )
    return compact_headline_selected_energy_contribution(draw_result)


__all__ = [
    "CONDITIONAL_CONTRACT_VERSION",
    "ENERGY_BRIDGE_CONTRACT_VERSION",
    "ENERGY_BASELINE_CACHE_CONTRACT_VERSION",
    "SELECTED_ENERGY_BASELINE_CONTRACT_VERSION",
    "COMPUTATIONAL_HORIZON",
    "DEFAULT_CONDITIONAL_SEED",
    "CONDITION_STATISTICS",
    "MANUAL_METRICS",
    "ENERGY_RATIO_PATHOLOGICAL_THRESHOLD",
    "HeadlineConditionalError",
    "SavedHeadlinePosterior",
    "HeadlineConditionalDrawBundle",
    "same_vintage",
    "load_saved_headline_posterior",
    "future_dates_for_saved",
    "parse_manual_values",
    "manual_condition_levels",
    "energy_level_paths_headline_bridge",
    "energy_outcome_headline_bridge",
    "energy_headline_ratio_diagnostic",
    "energy_bridge_condition",
    "headline_energy_baseline_cache_key",
    "clear_headline_energy_baseline_cache",
    "run_saved_headline_energy_marginal_draws",
    "compact_headline_energy_marginal",
    "run_saved_headline_energy_marginal",
    "run_saved_headline_selected_energy_contribution_draws",
    "compact_headline_selected_energy_contribution",
    "run_saved_headline_selected_energy_contribution",
    "run_saved_headline_conditional",
]
