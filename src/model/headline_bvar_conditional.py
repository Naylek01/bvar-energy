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

    dates = _normalise_months(bridge.get("dates") or [])
    path_map = dict(bridge.get("scenario_level") or {})
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


__all__ = [
    "CONDITIONAL_CONTRACT_VERSION",
    "ENERGY_BRIDGE_CONTRACT_VERSION",
    "COMPUTATIONAL_HORIZON",
    "DEFAULT_CONDITIONAL_SEED",
    "CONDITION_STATISTICS",
    "MANUAL_METRICS",
    "ENERGY_RATIO_PATHOLOGICAL_THRESHOLD",
    "HeadlineConditionalError",
    "SavedHeadlinePosterior",
    "same_vintage",
    "load_saved_headline_posterior",
    "future_dates_for_saved",
    "parse_manual_values",
    "manual_condition_levels",
    "energy_level_paths_headline_bridge",
    "energy_outcome_headline_bridge",
    "energy_headline_ratio_diagnostic",
    "energy_bridge_condition",
    "run_saved_headline_conditional",
]
