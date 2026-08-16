"""Saved-run diagnostics for the locked Headline joint BVAR dashboard.

This module does not re-estimate the model. It turns the persisted posterior
and diagnostic sidecars into compact, dashboard-ready records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from energy_bvar_model import var_companion

ROOT_PERIODS = (np.inf, 12.0, 6.0, 4.0, 3.0, 2.4, 2.0)
ROOT_LABELS = ("zero", "12m", "6m", "4m", "3m", "2.4m", "2m")
ROOT_ANGLE_TOLERANCE = np.pi / 24.0
ROOT_DIAGNOSTIC_DRAWS = 1000


class HeadlineDiagnosticsError(RuntimeError):
    pass


def _read_table(run_directory: Path, stem: str) -> pd.DataFrame:
    parquet = run_directory / f"{stem}.parquet"
    csv = run_directory / f"{stem}.csv"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    if csv.is_file():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {parquet.name} nor {csv.name} exists in {run_directory}.")


def _target_angle(period: float) -> float:
    if np.isinf(period):
        return 0.0
    return 2.0 * np.pi / float(period)


def posterior_root_frequency_summary(
    B_draws: np.ndarray,
    *,
    n: int,
    p: int,
    periods: Iterable[float] = ROOT_PERIODS,
    angle_tolerance: float = ROOT_ANGLE_TOLERANCE,
    n_draws: int = ROOT_DIAGNOSTIC_DRAWS,
) -> pd.DataFrame:
    """Replicate the locked-notebook posterior root-frequency diagnostic.

    Posterior draws are selected deterministically with evenly spaced indices.
    Within each target-frequency band (±pi/24 by default), the most persistent
    root is retained. If a draw has no root in the band, that draw is NaN for
    that frequency, exactly as in the locked notebook diagnostic.
    """
    B = np.asarray(B_draws, dtype=float)
    if B.ndim != 3:
        raise ValueError("B_draws must have shape (draw, coefficient, equation).")
    if B.shape[2] != int(n):
        raise ValueError("B_draws equation dimension does not match n.")
    if angle_tolerance <= 0:
        raise ValueError("angle_tolerance must be positive.")

    periods = tuple(float(x) for x in periods)
    target_angles = np.asarray([_target_angle(x) for x in periods], dtype=float)
    keep = min(int(n_draws), B.shape[0])
    if keep < 1:
        raise ValueError("n_draws must be positive.")
    draw_idx = np.linspace(0, B.shape[0] - 1, keep, dtype=int)
    selected = np.full((keep, len(periods)), np.nan, dtype=float)

    for out_s, s in enumerate(draw_idx):
        eig = np.linalg.eigvals(var_companion(B[int(s)], int(n), int(p)))
        angles = np.abs(np.angle(eig))
        moduli = np.abs(eig)
        for j, target in enumerate(target_angles):
            mask = np.abs(angles - target) <= float(angle_tolerance)
            if np.any(mask):
                selected[out_s, j] = float(np.max(moduli[mask]))

    rows = []
    labels = ROOT_LABELS if len(periods) == len(ROOT_LABELS) else tuple(
        "zero" if np.isinf(x) else f"{x:g}m" for x in periods
    )
    for j, (label, period) in enumerate(zip(labels, periods)):
        x = selected[:, j]
        finite = x[np.isfinite(x)]
        if len(finite):
            q05, q50, q95 = np.quantile(finite, [0.05, 0.50, 0.95])
            p90 = float(np.mean(finite > 0.90))
            p95 = float(np.mean(finite > 0.95))
            p97 = float(np.mean(finite > 0.97))
        else:
            q05 = q50 = q95 = p90 = p95 = p97 = np.nan
        rows.append(
            {
                "frequency": label,
                "period_months": np.nan if np.isinf(period) else float(period),
                "q05": float(q05),
                "median": float(q50),
                "q95": float(q95),
                "prob_gt_090": float(p90),
                "prob_gt_095": float(p95),
                "prob_gt_097": float(p97),
                "draws": int(len(finite)),
                "requested_draws": int(keep),
            }
        )
    return pd.DataFrame(rows)


def saved_headline_diagnostic_rows(run_directory: str | Path) -> pd.DataFrame:
    """Return compact long-form diagnostic records for ``display_v1.parquet``."""
    run_dir = Path(run_directory)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    frames: list[pd.DataFrame] = []

    # Persisted sampler/stability diagnostics created by energy_bvar_io.
    diagnostics = _read_table(run_dir, "diagnostics")
    if not diagnostics.empty:
        rows = []
        for _, row in diagnostics.iterrows():
            group = str(row.get("diagnostic_group", ""))
            parameter = str(row.get("parameter", ""))
            rows.append(
                {
                    "record_type": "diagnostic",
                    "scope": "headline",
                    "metric": group,
                    "series": parameter,
                    "label": parameter,
                    "key": parameter,
                    "value": pd.to_numeric(row.get("value"), errors="coerce"),
                    "posterior_mean": pd.to_numeric(row.get("posterior_mean"), errors="coerce"),
                    "posterior_sd": pd.to_numeric(row.get("posterior_sd"), errors="coerce"),
                    "ess": pd.to_numeric(row.get("ESS"), errors="coerce"),
                    "mcse_mean": pd.to_numeric(row.get("MCSE_mean"), errors="coerce"),
                    "mcse_over_sd": pd.to_numeric(row.get("MCSE_over_sd"), errors="coerce"),
                }
            )
        frames.append(pd.DataFrame(rows))

    # Accounting reconstruction diagnostics persisted by the Headline pipeline.
    reconstruction_path = run_dir / "headline_reconstruction_diagnostics.csv"
    if reconstruction_path.is_file():
        recon = pd.read_csv(reconstruction_path)
        if recon.shape[1] >= 2:
            first, second = recon.columns[:2]
            values = []
            for _, row in recon.iterrows():
                values.append(
                    {
                        "record_type": "diagnostic",
                        "scope": "headline",
                        "metric": "reconstruction",
                        "series": str(row[first]),
                        "label": str(row[first]),
                        "key": str(row[first]),
                        "value": pd.to_numeric(row[second], errors="coerce"),
                    }
                )
            if values:
                frames.append(pd.DataFrame(values))

    # Frequency-specific companion-root diagnostics from the saved B draws.
    draws_path = run_dir / "draws.npz"
    if draws_path.is_file():
        metadata_path = run_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("model_id")) != "headline_joint":
                raise HeadlineDiagnosticsError("Run metadata is not headline_joint.")
            if int(metadata.get("p", 12)) != 12:
                raise HeadlineDiagnosticsError("Headline diagnostic contract requires p=12.")
        with np.load(draws_path, allow_pickle=False) as z:
            if "B" not in z.files:
                raise HeadlineDiagnosticsError("draws.npz does not contain B draws.")
            B = np.asarray(z["B"], dtype=float)
        if B.ndim != 3:
            raise HeadlineDiagnosticsError("Stored B draws have an unexpected shape.")
        n = int(B.shape[2])
        # B rows are 1 + n*p + deterministic regressors. Headline production p=12.
        p = 12
        roots = posterior_root_frequency_summary(B, n=n, p=p)
        root_rows = pd.DataFrame(
            {
                "record_type": "root",
                "scope": "headline",
                "metric": "posterior_root_band",
                "series": roots["frequency"],
                "label": roots["frequency"],
                "horizon": roots["period_months"],
                "q05": roots["q05"],
                "q50": roots["median"],
                "q95": roots["q95"],
                "prob_90": roots["prob_gt_090"],
                "prob_95": roots["prob_gt_095"],
                "prob_97": roots["prob_gt_097"],
                "extra_value": roots["draws"],
            }
        )
        frames.append(root_rows)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


__all__ = [
    "ROOT_PERIODS",
    "ROOT_LABELS",
    "ROOT_ANGLE_TOLERANCE",
    "ROOT_DIAGNOSTIC_DRAWS",
    "HeadlineDiagnosticsError",
    "posterior_root_frequency_summary",
    "saved_headline_diagnostic_rows",
]
