"""Structural-analysis materialisation for the locked Headline joint BVAR.

The dashboard never runs IRFs/FEVD/HD interactively. This module computes the
recursive structural objects once from a saved or freshly estimated posterior,
then writes a compact parquet consumed by the Headline display artefact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from energy_bvar_io import load_energy_bvar_draws
from energy_bvar_model import (
    forecast_error_variance_decomposition,
    historical_decomposition,
    impulse_responses,
    prepare_bvar_panel,
)
from headline_bvar_pipeline import (
    BASELINE_P,
    MAX_PUBLISHED_HORIZON_MONTHS,
    NATIVE_VARIABLES,
    STATE_VARIABLES,
    make_joint_seasonals,
)
from headline_joint_bvar import native_to_state_levels

STRUCTURAL_FILENAME = "headline_structural_v1.parquet"
STRUCTURAL_METADATA_FILENAME = "headline_structural_metadata.json"
STRUCTURAL_SCHEMA_VERSION = "headline-structural-1.0"
DEFAULT_STRUCTURAL_DRAWS = 1000
DEFAULT_HD_MONTHS = 120


class HeadlineStructuralError(RuntimeError):
    pass


def _posterior_indices(total: int, maximum: int) -> np.ndarray:
    total = int(total)
    maximum = int(maximum)
    if total < 1 or maximum < 1:
        raise ValueError("Posterior draw counts must be positive.")
    if total <= maximum:
        return np.arange(total, dtype=int)
    # Deterministic, even posterior coverage without RNG-dependent dashboard output.
    return np.unique(np.linspace(0, total - 1, maximum, dtype=int))


def subset_result_draws(result: Mapping, maximum: int = DEFAULT_STRUCTURAL_DRAWS) -> dict:
    """Shallow-copy a result and subset every posterior array consistently."""
    out = dict(result)
    total = int(np.asarray(result["B"]).shape[0])
    indices = _posterior_indices(total, maximum)
    for key, value in result.items():
        if key in {"prep", "prior", "ols", "metadata", "prior_config", "sampler_config", "headline_joint_context"}:
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == total:
            out[key] = np.asarray(value)[indices].copy()
    out["n_draws"] = int(len(indices))
    out["structural_draw_indices"] = indices
    return out


def load_saved_headline_result(run_directory: str | Path) -> dict:
    """Reconstruct the minimal full result object needed by structural routines."""
    run_dir = Path(run_directory)
    metadata, draws = load_energy_bvar_draws(run_dir)
    native_path = run_dir / "headline_native_history.csv"
    if not native_path.is_file():
        raise FileNotFoundError(native_path)

    native = pd.read_csv(native_path, index_col=0, parse_dates=True).sort_index()
    native.index = pd.DatetimeIndex(native.index, name="date")
    missing = [name for name in NATIVE_VARIABLES if name not in native.columns]
    if missing:
        raise HeadlineStructuralError(f"headline_native_history.csv is missing {missing}.")
    native = native[NATIVE_VARIABLES].astype(float)

    state = native_to_state_levels(native)
    seasonals = make_joint_seasonals(state.index)
    prep = prepare_bvar_panel(
        state,
        p=BASELINE_P,
        variables=STATE_VARIABLES,
        exog=seasonals,
        frequency="monthly",
    )

    result = dict(draws)
    result.update(
        {
            "metadata": metadata,
            "prep": prep,
            "p": BASELINE_P,
            "variables": list(STATE_VARIABLES),
            "frequency": "monthly",
            "n_draws": int(np.asarray(draws["B"]).shape[0]),
        }
    )
    return result


def _summary_rows(
    draws: np.ndarray,
    *,
    record_type: str,
    metric: str,
    responses: list[str],
    shocks: list[str],
    horizons: np.ndarray,
    unit: str,
) -> pd.DataFrame:
    arr = np.asarray(draws, dtype=float)
    if arr.ndim != 4:
        raise ValueError("Structural draws must have shape draw×horizon×response×shock.")
    q = np.nanquantile(arr, [0.05, 0.16, 0.50, 0.84, 0.95], axis=0)
    mean = np.nanmean(arr, axis=0)
    rows = []
    for h_idx, horizon in enumerate(horizons):
        for r_idx, response in enumerate(responses):
            for s_idx, shock in enumerate(shocks):
                rows.append(
                    {
                        "record_type": record_type,
                        "scope": "headline",
                        "metric": metric,
                        "response": response,
                        "shock": shock,
                        "horizon": int(horizon),
                        "unit": unit,
                        "value": float(mean[h_idx, r_idx, s_idx]),
                        "q05": float(q[0, h_idx, r_idx, s_idx]),
                        "q16": float(q[1, h_idx, r_idx, s_idx]),
                        "q50": float(q[2, h_idx, r_idx, s_idx]),
                        "q84": float(q[3, h_idx, r_idx, s_idx]),
                        "q95": float(q[4, h_idx, r_idx, s_idx]),
                    }
                )
    return pd.DataFrame(rows)


def build_headline_structural_artifact(
    result: Mapping,
    run_directory: str | Path,
    *,
    maximum_draws: int = DEFAULT_STRUCTURAL_DRAWS,
    horizon: int = MAX_PUBLISHED_HORIZON_MONTHS,
    hd_months: int = DEFAULT_HD_MONTHS,
    overwrite: bool = True,
) -> dict[str, Path]:
    """Materialise recursive IRF, FEVD and exact HD summaries for the dashboard."""
    horizon = int(horizon)
    if horizon < 1 or horizon > MAX_PUBLISHED_HORIZON_MONTHS:
        raise ValueError(
            f"Structural horizon must be in [1, {MAX_PUBLISHED_HORIZON_MONTHS}]."
        )
    run_dir = Path(run_directory)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / STRUCTURAL_FILENAME
    metadata_target = run_dir / STRUCTURAL_METADATA_FILENAME
    if target.exists() and not overwrite:
        raise FileExistsError(target)

    structural = subset_result_draws(result, maximum=maximum_draws)
    responses = list(structural["variables"])
    reference_date = pd.Timestamp(structural["prep"]["dates"][-1])

    irf = impulse_responses(
        structural,
        identification="recursive",
        reference_date=reference_date,
        horizon=horizon,
        shock_unit="structural_std",
        shock_size=1.0,
        include_outlier_scale=False,
    )
    shocks = list(irf["shock_names"])
    change = 100.0 * np.asarray(irf["change_irfs"], dtype=float)
    cumulative_log = 100.0 * np.asarray(
        irf["cumulative_level_irfs"], dtype=float
    )
    frames = [
        _summary_rows(
            change,
            record_type="structural_irf",
            metric="monthly_log_change",
            responses=responses,
            shocks=shocks,
            horizons=np.asarray(irf["horizons"]),
            unit="100 × Δlog points",
        ),
        _summary_rows(
            cumulative_log,
            record_type="structural_irf",
            metric="cumulative_log_change",
            responses=responses,
            shocks=shocks,
            horizons=np.asarray(irf["horizons"]),
            unit="100 × cumulative Δlog",
        ),
    ]

    fevd = forecast_error_variance_decomposition(
        structural,
        identification="recursive",
        reference_date=reference_date,
        horizon=horizon,
        include_outlier_scale=False,
    )
    frames.append(
        _summary_rows(
            np.asarray(fevd["fevd_draws"], dtype=float),
            record_type="structural_fevd",
            metric="share",
            responses=responses,
            shocks=list(fevd["shock_names"]),
            horizons=np.asarray(fevd["horizons"]),
            unit="share",
        )
    )

    hd = historical_decomposition(
        structural,
        identification="recursive",
        reference_date=reference_date,
        split_outlier_amplification=True,
        reconstruction_tol=1e-8,
    )
    raw = np.asarray(hd["contributions"], dtype=float)
    n_shocks = len(hd["shock_names"])
    if raw.shape[-1] != 2 * n_shocks:
        raise HeadlineStructuralError(
            "Expected regular + outlier-amplification HD components for each shock."
        )
    shock_contrib = np.stack(
        [raw[..., 2 * j] + raw[..., 2 * j + 1] for j in range(n_shocks)],
        axis=-1,
    )
    dates = pd.DatetimeIndex(hd["dates"], name="date")
    base = np.asarray(hd["base"], dtype=float)
    observed = np.asarray(hd["observed"], dtype=float)

    def _rolling_12_draws(x: np.ndarray) -> np.ndarray:
        out = np.full_like(x, np.nan, dtype=float)
        csum = np.cumsum(x, axis=1)
        out[:, 11:] = csum[:, 11:]
        if x.shape[1] > 12:
            out[:, 12:] -= csum[:, :-12]
        return out

    def _rolling_12_observed(x: np.ndarray) -> np.ndarray:
        out = np.full_like(x, np.nan, dtype=float)
        csum = np.cumsum(x, axis=0)
        out[11:] = csum[11:]
        if x.shape[0] > 12:
            out[12:] -= csum[:-12]
        return out

    shock_12m = 100.0 * _rolling_12_draws(shock_contrib)
    base_12m = 100.0 * _rolling_12_draws(base)
    observed_12m = 100.0 * _rolling_12_observed(observed)

    reconstructed_12m = base_12m + np.nansum(shock_12m, axis=3)
    observed_broadcast = np.broadcast_to(observed_12m, reconstructed_12m.shape)
    adding_mask = np.isfinite(reconstructed_12m) & np.isfinite(observed_broadcast)
    hd_12m_add_error = float(
        np.max(np.abs(reconstructed_12m[adding_mask] - observed_broadcast[adding_mask]))
    ) if np.any(adding_mask) else float("nan")

    sl = slice(max(0, len(dates) - int(hd_months)), None)
    dates = dates[sl]
    hd_rows = []
    q_levels = [0.05, 0.16, 0.50, 0.84, 0.95]
    for r, response in enumerate(responses):
        for s_idx, shock in enumerate(hd["shock_names"]):
            arr = shock_12m[:, sl, r, s_idx]
            q = np.nanquantile(arr, q_levels, axis=0)
            mean = np.nanmean(arr, axis=0)
            for t, date in enumerate(dates):
                hd_rows.append(
                    {
                        "record_type": "structural_hd",
                        "scope": "headline",
                        "metric": "shock_contribution_12m_log",
                        "response": response,
                        "shock": shock,
                        "date": date,
                        "unit": "12m contribution (100 × log points)",
                        "value": float(mean[t]),
                        "q05": float(q[0, t]),
                        "q16": float(q[1, t]),
                        "q50": float(q[2, t]),
                        "q84": float(q[3, t]),
                        "q95": float(q[4, t]),
                    }
                )
        arr = base_12m[:, sl, r]
        q = np.nanquantile(arr, q_levels, axis=0)
        mean = np.nanmean(arr, axis=0)
        for t, date in enumerate(dates):
            hd_rows.append(
                {
                    "record_type": "structural_hd",
                    "scope": "headline",
                    "metric": "base_12m_log",
                    "response": response,
                    "date": date,
                    "unit": "12m contribution (100 × log points)",
                    "value": float(mean[t]),
                    "q05": float(q[0, t]),
                    "q16": float(q[1, t]),
                    "q50": float(q[2, t]),
                    "q84": float(q[3, t]),
                    "q95": float(q[4, t]),
                }
            )
            hd_rows.append(
                {
                    "record_type": "structural_hd",
                    "scope": "headline",
                    "metric": "observed_12m_log",
                    "response": response,
                    "date": date,
                    "unit": "12m contribution (100 × log points)",
                    "value": float(observed_12m[sl, r][t]),
                }
            )
    frames.append(pd.DataFrame(hd_rows))

    metadata = {
        "structural_schema_version": STRUCTURAL_SCHEMA_VERSION,
        "identification": "recursive",
        "recursive_ordering": responses,
        "reference_date": reference_date.isoformat(),
        "horizon_months": horizon,
        "posterior_draws_available": int(np.asarray(result["B"]).shape[0]),
        "posterior_draws_used": int(np.asarray(structural["B"]).shape[0]),
        "include_outlier_scale_irf_fevd": False,
        "hd_split_outlier_amplification": True,
        "hd_components_recombined_by_shock": True,
        "hd_months_displayed": int(len(dates)),
        "hd_max_reconstruction_error": float(hd["max_reconstruction_error"]),
        "hd_12m_log_adding_up_max_error": float(hd_12m_add_error),
        "fevd_max_share_sum_error": float(fevd["max_share_sum_error"]),
    }
    meta_rows = []
    for key, value in metadata.items():
        meta_rows.append(
            {
                "record_type": "structural_meta",
                "scope": "headline",
                "metric": "structural",
                "key": key,
                "text_value": json.dumps(value) if isinstance(value, (list, dict)) else str(value),
                "value": float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else np.nan,
            }
        )
    frames.append(pd.DataFrame(meta_rows))

    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["structural_schema_version"] = STRUCTURAL_SCHEMA_VERSION
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)
    metadata_target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"structural": target, "metadata": metadata_target}


__all__ = [
    "STRUCTURAL_FILENAME",
    "STRUCTURAL_METADATA_FILENAME",
    "STRUCTURAL_SCHEMA_VERSION",
    "DEFAULT_STRUCTURAL_DRAWS",
    "DEFAULT_HD_MONTHS",
    "HeadlineStructuralError",
    "subset_result_draws",
    "load_saved_headline_result",
    "build_headline_structural_artifact",
]
