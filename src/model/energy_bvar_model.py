"""BVAR with stochastic volatility, model-based outliers, and DK forecasts.

This module implements the reusable constant-coefficient BVAR engine used for
energy-price forecasting in ECB Working Paper 3062. Component-specific data
construction, tax re-attribution and plotting live in separate modules:

    y_t = B' x_t + nu_t
    nu_t = A^{-1} O_t Lambda_t^{1/2} epsilon_t

where A is constant and unit lower triangular, log diag(Lambda_t) follows
independent random walks, and the diagonal outlier scales in O_t are one in
regular periods and larger than one in outlier periods.

Conventions
-----------
* B has shape (1 + n*p, n): constant first, then lag blocks.
* vec(B) is column-major (order="F"), equation by equation.
* The estimation sample is the maximal balanced block of absolute changes.
  The end-of-sample ragged edge is retained for Durbin--Koopman forecasts.
* The KSC seven-component approximation is used for the log-volatility paths.
* The continuous U(2, 20) outlier support is represented by a configurable
  finite grid. The default grid is the integer support 2, ..., 20.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.mlemodel import MLEModel


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BVARSVOPriorConfig:
    """Prior settings not fully reported in the main ECB paper.

    Minnesota parameters use the standard-deviation convention:

        sd(own lag l)  = lambda1 / l**lambda3
        sd(cross lag)  = sigma_i/sigma_j * lambda1*lambda2/l**lambda3
        sd(constant i) = sigma_i * lambda4

    The coefficient prior is centred on white noise (all lag means equal 0).
    """

    lambda1: float = 0.20
    lambda2: float = 0.50
    lambda3: float = 1.00
    lambda4: float = 10.0
    a_prior_var: float = 10.0
    phi_prior_mean: float = 0.02
    phi_prior_df: float = 10.0
    h0_var: float = 4.0
    outlier_mean_frequency: float = 1.0 / 48.0
    outlier_prior_observations: float = 120.0
    outlier_grid_min: float = 2.0
    outlier_grid_max: float = 20.0
    outlier_grid_step: float = 1.0
    ksc_offset_scale: float = 1e-6
    ksc_offset_floor: float = 1e-12

    def validate(self) -> None:
        positive = {
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "lambda3": self.lambda3,
            "lambda4": self.lambda4,
            "a_prior_var": self.a_prior_var,
            "phi_prior_mean": self.phi_prior_mean,
            "phi_prior_df": self.phi_prior_df,
            "h0_var": self.h0_var,
            "outlier_prior_observations": self.outlier_prior_observations,
            "outlier_grid_step": self.outlier_grid_step,
            "ksc_offset_scale": self.ksc_offset_scale,
            "ksc_offset_floor": self.ksc_offset_floor,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"Prior parameters must be positive: {bad}")
        if not 0 < self.outlier_mean_frequency < 1:
            raise ValueError("outlier_mean_frequency must lie in (0, 1).")
        if self.outlier_grid_min < 1 or self.outlier_grid_max <= self.outlier_grid_min:
            raise ValueError("The outlier grid must satisfy 1 <= min < max.")
        if self.phi_prior_df <= 2:
            raise ValueError("phi_prior_df must exceed 2 for a finite prior mean.")


@dataclass(frozen=True)
class SamplerConfig:
    reps: int = 6_000
    burn: int = 3_000
    thin: int = 1
    seed: int = 42
    max_stability_tries: int = 1_000
    progress_every: int = 500
    sv_sampler: str = "centered_ksc"

    def validate(self) -> None:
        if self.reps < 2:
            raise ValueError("reps must be at least 2.")
        if not 0 <= self.burn < self.reps:
            raise ValueError("burn must satisfy 0 <= burn < reps.")
        if self.thin < 1:
            raise ValueError("thin must be at least 1.")
        if self.max_stability_tries < 1:
            raise ValueError("max_stability_tries must be at least 1.")
        if self.progress_every < 0:
            raise ValueError("progress_every cannot be negative.")
        if self.sv_sampler != "centered_ksc":
            raise ValueError("Only sv_sampler='centered_ksc' is implemented. ASIS is reserved for a later sampler update.")


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------




def _read_dated_csv(path: str | Path, frequency: str = "monthly") -> pd.DataFrame:
    """Read a dated CSV and normalise its calendar index."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    date_candidates = [
        c for c in frame.columns if c.lower() in {"date", "time", "period"}
    ]
    if not date_candidates:
        raise ValueError(f"No date column found in {path}.")
    date_col = date_candidates[0]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    frame = frame.set_index(date_col).sort_index()
    dates = pd.DatetimeIndex(frame.index)
    if frequency == "monthly":
        frame.index = dates.to_period("M").to_timestamp(how="start")
    elif frequency == "weekly":
        frame.index = dates.to_period("W-SUN").start_time
    else:
        raise ValueError("frequency must be 'monthly' or 'weekly'.")
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).last()
    frame.index.name = "date"
    return frame


def load_energy_panel(
    path: str | Path,
    variables: Sequence[str],
    *,
    aliases: Mapping[str, str] | None = None,
    frequency: str = "monthly",
) -> pd.DataFrame:
    """Load a model panel without filling missing observations.

    The generic engine receives already constructed model levels. Source-specific
    transformations and aliases are supplied by component adapters.
    """
    frame = _read_dated_csv(path, frequency=frequency)
    if aliases:
        frame = frame.rename(columns=dict(aliases))
    variables = list(variables)
    missing = [name for name in variables if name not in frame.columns]
    if missing:
        raise KeyError(
            f"Missing model columns {missing}. Available columns: {list(frame.columns)}"
        )
    frame = frame[variables].astype(float)
    if frequency == "monthly":
        full_index = pd.date_range(
            frame.index.min(), frame.index.max(), freq="MS", name="date"
        )
    else:
        full_index = pd.date_range(
            frame.index.min(), frame.index.max(), freq="W-MON", name="date"
        )
    frame = frame.reindex(full_index)
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("The model panel contains infinite values.")
    return frame

def _prepare_var_regression(data: pd.DataFrame, p: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    if p < 1:
        raise ValueError("p must be at least 1.")
    if len(data) <= p:
        raise ValueError("Need more than p observations.")
    if data.isna().any().any():
        raise ValueError("The balanced estimation panel contains NaNs.")
    arr = data.to_numpy(dtype=float)
    Y = arr[p:]
    lag_blocks = [arr[p - lag : len(arr) - lag] for lag in range(1, p + 1)]
    X = np.column_stack([np.ones(len(Y)), *lag_blocks])
    return Y, X, data.index[p:]


def prepare_bvar_panel(
    levels: pd.DataFrame,
    p: int = 12,
    variables: Sequence[str] | None = None,
) -> dict:
    """Prepare levels, absolute changes, balanced estimation sample and ragged edge.

    Missing rows are allowed only at the beginning and at the end of the
    transformed panel. An interior gap is rejected because silently deleting it
    would change the monthly spacing of the VAR.
    """
    if not isinstance(levels, pd.DataFrame):
        raise TypeError("levels must be a pandas DataFrame.")
    if variables is None:
        variables = list(levels.columns)
    variables = list(variables)
    frame = levels[variables].copy().astype(float).sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("levels must have a DatetimeIndex.")
    frame.index = frame.index.to_period("M").to_timestamp(how="start")
    if frame.index.has_duplicates:
        raise ValueError("The monthly panel contains duplicate dates.")
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="MS", name="date")
    frame = frame.reindex(full_index)
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("The level panel contains infinite values.")

    differences = frame.diff()
    complete = differences.notna().all(axis=1).to_numpy()
    if not complete.any():
        raise ValueError("No fully observed monthly changes are available.")
    first = int(np.flatnonzero(complete)[0])
    last = int(np.flatnonzero(complete)[-1])
    interior_bad = np.flatnonzero(~complete[first : last + 1]) + first
    if len(interior_bad):
        dates = [differences.index[i].date().isoformat() for i in interior_bad[:10]]
        suffix = " ..." if len(interior_bad) > 10 else ""
        raise ValueError(
            "Interior missing observations in the transformed VAR panel at "
            + ", ".join(dates)
            + suffix
        )

    balanced = differences.iloc[first : last + 1].copy()
    if len(balanced) <= p + 5:
        raise ValueError(
            f"Only {len(balanced)} balanced changes remain; this is too short for VAR({p})."
        )
    Y, X, regression_dates = _prepare_var_regression(balanced, p)
    balanced_end = balanced.index[-1]
    ragged = differences.loc[differences.index > balanced_end].copy()

    state_rows = balanced.iloc[-p:].to_numpy(dtype=float)[::-1]
    last_companion_state = state_rows.reshape(-1)
    level_at_balanced_end = frame.loc[balanced_end].to_numpy(dtype=float)

    return {
        "levels": frame,
        "differences": differences,
        "balanced": balanced,
        "ragged": ragged,
        "Y": Y,
        "X": X,
        "dates": regression_dates,
        "variables": variables,
        "p": int(p),
        "n": len(variables),
        "balanced_start": balanced.index[0],
        "balanced_end": balanced_end,
        "last_calendar_date": frame.index[-1],
        "last_companion_state": last_companion_state,
        "level_at_balanced_end": level_at_balanced_end,
        "n_balanced_changes": len(balanced),
        "n_regression_observations": len(Y),
        "n_ragged_months": len(ragged),
    }


def fit_var_ols_from_prepared(prep: Mapping) -> dict:
    Y = np.asarray(prep["Y"], dtype=float)
    X = np.asarray(prep["X"], dtype=float)
    B = np.linalg.solve(X.T @ X, X.T @ Y)
    resid = Y - X @ B
    Sigma = resid.T @ resid / max(len(Y) - X.shape[1], 1)
    return {
        "B": B,
        "resid": resid,
        "Sigma": 0.5 * (Sigma + Sigma.T),
        "spectral_radius": spectral_radius(B, prep["n"], prep["p"]),
    }


def ar1_residual_scales(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    arr = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=float)
    if arr.ndim != 2 or len(arr) < 4:
        raise ValueError("AR(1) scales require a two-dimensional sample with at least four rows.")
    out = np.empty(arr.shape[1])
    for j in range(arr.shape[1]):
        y = arr[1:, j]
        x = np.column_stack([np.ones(len(y)), arr[:-1, j]])
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        e = y - x @ b
        out[j] = np.sqrt(e @ e / max(len(e) - 2, 1))
    if np.any(~np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("Invalid preliminary AR(1) residual scales.")
    return out


# -----------------------------------------------------------------------------
# Priors and labels
# -----------------------------------------------------------------------------


def coefficient_labels(variables: Sequence[str], p: int) -> list[str]:
    variables = list(variables)
    labels: list[str] = []
    for equation in variables:
        labels.append(f"{equation}: const")
        for lag in range(1, p + 1):
            labels.extend(f"{equation}: {regressor} L{lag}" for regressor in variables)
    return labels


def make_bvar_svo_prior(prep: Mapping, config: BVARSVOPriorConfig | None = None) -> dict:
    config = BVARSVOPriorConfig() if config is None else config
    config.validate()
    data = prep["balanced"]
    variables = list(prep["variables"])
    n = prep["n"]
    p = prep["p"]
    k = 1 + n * p
    scales = ar1_residual_scales(data)

    B0 = np.zeros((k, n))
    variances = np.empty((k, n))
    for eq in range(n):
        variances[0, eq] = (scales[eq] * config.lambda4) ** 2
        for lag in range(1, p + 1):
            for reg in range(n):
                row = 1 + (lag - 1) * n + reg
                if eq == reg:
                    sd = config.lambda1 / lag**config.lambda3
                else:
                    sd = (
                        scales[eq]
                        / scales[reg]
                        * config.lambda1
                        * config.lambda2
                        / lag**config.lambda3
                    )
                variances[row, eq] = sd**2

    b0 = B0.reshape(-1, order="F")
    v0_diag = variances.reshape(-1, order="F")
    if np.any(v0_diag <= 0):
        raise ValueError("The coefficient prior contains non-positive variances.")

    outlier_alpha = config.outlier_mean_frequency * config.outlier_prior_observations
    outlier_beta = (1.0 - config.outlier_mean_frequency) * config.outlier_prior_observations
    grid = np.arange(
        config.outlier_grid_min,
        config.outlier_grid_max + 0.5 * config.outlier_grid_step,
        config.outlier_grid_step,
        dtype=float,
    )
    if len(grid) == 0:
        raise ValueError("The outlier grid is empty.")

    phi_shape = config.phi_prior_df / 2.0
    phi_scale = config.phi_prior_mean * (phi_shape - 1.0)

    return {
        "B0": B0,
        "b0": b0,
        "V0_diag": v0_diag,
        "V0_inv_diag": 1.0 / v0_diag,
        "coefficient_labels": coefficient_labels(variables, p),
        "scales": scales,
        "a_mean": 0.0,
        "a_var": float(config.a_prior_var),
        "phi_shape": float(phi_shape),
        "phi_scale": float(phi_scale),
        "phi_prior_mean": float(config.phi_prior_mean),
        "h0_var": float(config.h0_var),
        "outlier_alpha": float(outlier_alpha),
        "outlier_beta": float(outlier_beta),
        "outlier_grid": grid,
        "config": asdict(config),
    }


# -----------------------------------------------------------------------------
# Companion form and linear algebra
# -----------------------------------------------------------------------------


def var_companion(B: np.ndarray, n: int, p: int) -> np.ndarray:
    B = np.asarray(B, dtype=float)
    if B.shape != (1 + n * p, n):
        raise ValueError(f"B must have shape {(1 + n * p, n)}, got {B.shape}.")
    F = np.zeros((n * p, n * p))
    F[:n] = B[1:].T
    if p > 1:
        F[n:, :-n] = np.eye(n * (p - 1))
    return F


def spectral_radius(B: np.ndarray, n: int, p: int) -> float:
    eig = np.linalg.eigvals(var_companion(B, n, p))
    return float(np.max(np.abs(eig)))


def var_is_stable(B: np.ndarray, n: int, p: int, tolerance: float = 1.0) -> bool:
    try:
        radius = spectral_radius(B, n, p)
    except np.linalg.LinAlgError:
        return False
    return bool(np.isfinite(radius) and radius < tolerance)


def _safe_cholesky(matrix: np.ndarray, jitter: float = 1e-12, max_tries: int = 8) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    eye = np.eye(matrix.shape[0])
    bump = 0.0
    for attempt in range(max_tries):
        try:
            return np.linalg.cholesky(matrix + bump * eye)
        except np.linalg.LinAlgError:
            scale = max(float(np.trace(matrix) / max(len(matrix), 1)), 1.0)
            bump = jitter * scale if attempt == 0 else bump * 10.0
    raise np.linalg.LinAlgError(
        f"Cholesky failed; smallest eigenvalue={np.linalg.eigvalsh(matrix).min():.3e}."
    )


def unit_lower_from_free(free: np.ndarray, n: int) -> np.ndarray:
    free = np.asarray(free, dtype=float).reshape(-1)
    expected = n * (n - 1) // 2
    if len(free) != expected:
        raise ValueError(f"Expected {expected} free A elements, got {len(free)}.")
    A = np.eye(n)
    idx = 0
    for row in range(1, n):
        A[row, :row] = free[idx : idx + row]
        idx += row
    return A


def free_from_unit_lower(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    return np.concatenate([A[row, :row] for row in range(1, len(A))])


# -----------------------------------------------------------------------------
# Gibbs blocks
# -----------------------------------------------------------------------------


def draw_bvar_coefficients_sv(
    Y: np.ndarray,
    X: np.ndarray,
    A: np.ndarray,
    structural_variances: np.ndarray,
    prior: Mapping,
    n: int,
    p: int,
    rng: np.random.Generator,
    max_stability_tries: int = 1_000,
) -> tuple[np.ndarray, int, float]:
    """Draw B from its heteroskedastic GLS conditional, truncated to stability."""
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=float)
    q = np.asarray(structural_variances, dtype=float)
    T, n_y = Y.shape
    k = X.shape[1]
    if n_y != n or A.shape != (n, n) or q.shape != (T, n):
        raise ValueError("Y, A or structural_variances have inconsistent dimensions.")
    if np.any(q <= 0) or not np.all(np.isfinite(q)):
        raise ValueError("All structural variances must be finite and positive.")

    dim = n * k
    precision = np.diag(np.asarray(prior["V0_inv_diag"], dtype=float))
    rhs = np.asarray(prior["V0_inv_diag"], dtype=float) * np.asarray(prior["b0"], dtype=float)
    identity_n = np.eye(n)

    for t in range(T):
        H_t = np.kron(identity_n, X[t : t + 1])
        Z_t = A @ H_t
        inv_q = 1.0 / q[t]
        precision += Z_t.T @ (inv_q[:, None] * Z_t)
        rhs += Z_t.T @ (inv_q * (A @ Y[t]))

    precision = 0.5 * (precision + precision.T)
    mean = np.linalg.solve(precision, rhs)
    L = _safe_cholesky(precision)

    last_radius = np.nan
    for attempt in range(1, max_stability_tries + 1):
        draw = mean + np.linalg.solve(L.T, rng.standard_normal(dim))
        B = draw.reshape(k, n, order="F")
        last_radius = spectral_radius(B, n, p)
        if np.isfinite(last_radius) and last_radius < 1.0:
            return B, attempt, float(last_radius)

    raise RuntimeError(
        f"No stable B draw after {max_stability_tries} attempts; last radius={last_radius:.6f}."
    )


def draw_constant_cholesky_A(
    residuals: np.ndarray,
    structural_variances: np.ndarray,
    prior: Mapping,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the constant unit-lower-triangular A row by row using weighted regressions."""
    residuals = np.asarray(residuals, dtype=float)
    q = np.asarray(structural_variances, dtype=float)
    T, n = residuals.shape
    if q.shape != (T, n):
        raise ValueError("structural_variances has the wrong shape.")
    A = np.eye(n)
    prior_var = float(prior["a_var"])
    prior_mean = float(prior["a_mean"])

    for row in range(1, n):
        Z = residuals[:, :row]
        y = -residuals[:, row]
        weights = 1.0 / q[:, row]
        precision = np.eye(row) / prior_var + Z.T @ (weights[:, None] * Z)
        rhs = np.full(row, prior_mean / prior_var) + Z.T @ (weights * y)
        covariance = np.linalg.inv(0.5 * (precision + precision.T))
        mean = covariance @ rhs
        A[row, :row] = mean + _safe_cholesky(covariance) @ rng.standard_normal(row)
    return A


def structural_residuals(Y: np.ndarray, X: np.ndarray, B: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reduced = np.asarray(Y, dtype=float) - np.asarray(X, dtype=float) @ np.asarray(B, dtype=float)
    structural = reduced @ np.asarray(A, dtype=float).T
    return reduced, structural


# Kim--Shephard--Chib seven-component approximation to log(chi-square_1).
KSC7_PROB = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
KSC7_MEAN = np.array([-11.40039, -5.24321, -9.83726, 1.50746, -0.65098, 0.52478, -2.35859])
KSC7_VAR = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])


def _draw_ksc_indicators(log_eps2: np.ndarray, log_variance: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    residual = log_eps2[:, None] - log_variance[:, None] - KSC7_MEAN[None, :]
    log_prob = (
        np.log(KSC7_PROB)[None, :]
        - 0.5 * np.log(2.0 * np.pi * KSC7_VAR)[None, :]
        - 0.5 * residual**2 / KSC7_VAR[None, :]
    )
    log_prob -= log_prob.max(axis=1, keepdims=True)
    prob = np.exp(log_prob)
    prob /= prob.sum(axis=1, keepdims=True)
    cdf = np.cumsum(prob, axis=1)
    indicators = np.sum(rng.random(len(log_eps2))[:, None] > cdf, axis=1)
    return np.minimum(indicators, 6).astype(int), prob


def _ffbs_random_walk_scalar(
    observation: np.ndarray,
    observation_variance: np.ndarray,
    phi: float,
    h0_mean: float,
    h0_var: float,
    rng: np.random.Generator,
) -> np.ndarray:
    observation = np.asarray(observation, dtype=float).reshape(-1)
    observation_variance = np.asarray(observation_variance, dtype=float).reshape(-1)
    if phi <= 0 or h0_var <= 0:
        raise ValueError("phi and h0_var must be positive.")
    T = len(observation)
    m = np.empty(T + 1)
    P = np.empty(T + 1)
    m[0] = h0_mean
    P[0] = h0_var

    for t in range(1, T + 1):
        pred_m = m[t - 1]
        pred_P = P[t - 1] + phi
        f = pred_P + observation_variance[t - 1]
        gain = pred_P / f
        m[t] = pred_m + gain * (observation[t - 1] - pred_m)
        P[t] = max((1.0 - gain) * pred_P, 1e-14)

    state = np.empty(T + 1)
    state[-1] = m[-1] + np.sqrt(P[-1]) * rng.standard_normal()
    for t in range(T - 1, -1, -1):
        denom = P[t] + phi
        gain = P[t] / denom
        mean = m[t] + gain * (state[t + 1] - m[t])
        variance = max(P[t] - gain * P[t], 1e-14)
        state[t] = mean + np.sqrt(variance) * rng.standard_normal()
    return state


def sample_log_stochastic_volatility_ksc(
    log_variance_path: np.ndarray,
    residuals_adjusted_for_outliers: np.ndarray,
    phi: float,
    h0_mean: float,
    h0_var: float,
    rng: np.random.Generator,
    offset: float,
    return_diagnostics: bool = False,
):
    """Block-draw h_0:T where h_t = log(lambda_t) using KSC + FFBS."""
    old_h = np.asarray(log_variance_path, dtype=float).reshape(-1)
    eps = np.asarray(residuals_adjusted_for_outliers, dtype=float).reshape(-1)
    if len(old_h) != len(eps) + 1:
        raise ValueError("log_variance_path must contain T+1 elements.")
    if offset <= 0:
        raise ValueError("offset must be positive.")
    log_eps2 = np.log(eps**2 + offset)
    indicators, probabilities = _draw_ksc_indicators(log_eps2, old_h[1:], rng)
    observation = log_eps2 - KSC7_MEAN[indicators]
    new_h = _ffbs_random_walk_scalar(
        observation,
        KSC7_VAR[indicators],
        phi,
        h0_mean,
        h0_var,
        rng,
    )
    if not return_diagnostics:
        return new_h
    move = np.abs(new_h[1:] - old_h[1:])
    return new_h, {
        "indicators": indicators,
        "indicator_probabilities": probabilities,
        "component_counts": np.bincount(indicators, minlength=7),
        "mean_abs_move": float(move.mean()),
    }


def draw_phi(log_variance_path: np.ndarray, prior: Mapping, rng: np.random.Generator) -> float:
    increments = np.diff(np.asarray(log_variance_path, dtype=float))
    shape = float(prior["phi_shape"]) + 0.5 * len(increments)
    scale = float(prior["phi_scale"]) + 0.5 * float(increments @ increments)
    return float(1.0 / rng.gamma(shape=shape, scale=1.0 / scale))



def draw_sv_block(
    log_variance_path: np.ndarray,
    phi: np.ndarray,
    structural_residuals_draw: np.ndarray,
    outlier_scales: np.ndarray,
    h0_mean: np.ndarray,
    h0_var: float,
    offsets: np.ndarray,
    prior: Mapping,
    rng: np.random.Generator,
    *,
    method: str = "centered_ksc",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Draw the complete stochastic-volatility block ``(h, phi)``.

    The current implementation is the original centred KSC/FFBS sampler.  The
    function deliberately preserves the old random-number order: all volatility
    paths are drawn first, followed by all ``phi`` draws.  A future ASIS update
    can therefore replace this block without touching the rest of the Gibbs
    sampler.
    """
    if method != "centered_ksc":
        raise NotImplementedError(
            "ASIS is not implemented yet; use method='centered_ksc'."
        )

    h = np.asarray(log_variance_path, dtype=float).copy()
    phi_draw = np.asarray(phi, dtype=float).copy()
    residuals = np.asarray(structural_residuals_draw, dtype=float)
    scales = np.asarray(outlier_scales, dtype=float)
    n = residuals.shape[1]
    diagnostics: list[dict] = []

    for j in range(n):
        adjusted = residuals[:, j] / scales[:, j]
        h[:, j], diag = sample_log_stochastic_volatility_ksc(
            h[:, j],
            adjusted,
            phi_draw[j],
            h0_mean[j],
            h0_var,
            rng,
            offset=offsets[j],
            return_diagnostics=True,
        )
        diagnostics.append(diag)

    for j in range(n):
        phi_draw[j] = draw_phi(h[:, j], prior, rng)

    return h, phi_draw, diagnostics


def draw_outlier_states(
    standardized_structural_residuals: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw outlier indicators and scales from their discrete-grid conditional.

    For candidate scale o, the conditional Gaussian likelihood kernel is

        o^{-1} exp[-r^2 / (2 o^2)],  r = u / sqrt(lambda).
    """
    r = np.asarray(standardized_structural_residuals, dtype=float)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    grid = np.asarray(grid, dtype=float).reshape(-1)
    T, n = r.shape
    if len(p) != n:
        raise ValueError("One outlier probability is required per variable.")
    candidate = np.r_[1.0, grid]
    scales = np.empty((T, n))
    indicators = np.empty((T, n), dtype=bool)
    posterior_outlier_probability = np.empty((T, n))

    for j in range(n):
        prior_weights = np.r_[1.0 - p[j], np.repeat(p[j] / len(grid), len(grid))]
        log_kernel = (
            np.log(prior_weights)[None, :]
            - np.log(candidate)[None, :]
            - 0.5 * (r[:, j, None] / candidate[None, :]) ** 2
        )
        log_kernel -= log_kernel.max(axis=1, keepdims=True)
        weights = np.exp(log_kernel)
        weights /= weights.sum(axis=1, keepdims=True)
        posterior_outlier_probability[:, j] = 1.0 - weights[:, 0]
        cdf = np.cumsum(weights, axis=1)
        index = np.sum(rng.random(T)[:, None] > cdf, axis=1)
        index = np.minimum(index, len(candidate) - 1)
        scales[:, j] = candidate[index]
        indicators[:, j] = index > 0

    return scales, indicators, posterior_outlier_probability


def draw_outlier_probabilities(
    indicators: np.ndarray,
    prior: Mapping,
    rng: np.random.Generator,
) -> np.ndarray:
    indicators = np.asarray(indicators, dtype=bool)
    T = indicators.shape[0]
    count = indicators.sum(axis=0)
    return rng.beta(
        float(prior["outlier_alpha"]) + count,
        float(prior["outlier_beta"]) + T - count,
    )


def conditional_log_likelihood(
    Y: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
    A: np.ndarray,
    log_variance: np.ndarray,
    outlier_scales: np.ndarray,
) -> float:
    _, u = structural_residuals(Y, X, B, A)
    q = outlier_scales**2 * np.exp(log_variance)
    if np.any(q <= 0) or not np.all(np.isfinite(q)):
        return -np.inf
    n = Y.shape[1]
    return float(
        -0.5
        * np.sum(
            n * np.log(2.0 * np.pi)
            + np.log(q).sum(axis=1)
            + (u**2 / q).sum(axis=1)
        )
    )


# -----------------------------------------------------------------------------
# Gibbs orchestrator
# -----------------------------------------------------------------------------


def gibbs_bvar_sv_outlier(
    levels: pd.DataFrame,
    p: int = 12,
    variables: Sequence[str] | None = None,
    prior_config: BVARSVOPriorConfig | None = None,
    sampler_config: SamplerConfig | None = None,
) -> dict:
    """Estimate the constant-coefficient BVAR-SV-outlier model."""
    prior_config = BVARSVOPriorConfig() if prior_config is None else prior_config
    sampler_config = SamplerConfig() if sampler_config is None else sampler_config
    prior_config.validate()
    sampler_config.validate()
    prep = prepare_bvar_panel(levels, p=p, variables=variables)
    prior = make_bvar_svo_prior(prep, prior_config)
    Y = prep["Y"]
    X = prep["X"]
    T, n = Y.shape
    rng = np.random.default_rng(sampler_config.seed)

    ols = fit_var_ols_from_prepared(prep)
    B = ols["B"].copy()
    if not var_is_stable(B, n, p):
        B = prior["B0"].copy()
    A = np.eye(n)
    _, u = structural_residuals(Y, X, B, A)
    base_variance = np.maximum(np.var(u, axis=0, ddof=1), 1e-8)
    h0_mean = np.log(base_variance)
    log_variance_path = np.repeat(h0_mean[None, :], T + 1, axis=0)
    phi = np.full(n, prior["phi_prior_mean"])
    outlier_scales = np.ones((T, n))
    outlier_indicators = np.zeros((T, n), dtype=bool)
    outlier_probabilities = np.full(
        n,
        prior["outlier_alpha"] / (prior["outlier_alpha"] + prior["outlier_beta"]),
    )
    offsets = np.maximum(
        prior_config.ksc_offset_floor,
        prior_config.ksc_offset_scale * np.median(np.maximum(u**2, 1e-16), axis=0),
    )

    out_B: list[np.ndarray] = []
    out_A: list[np.ndarray] = []
    out_h: list[np.ndarray] = []
    out_phi: list[np.ndarray] = []
    out_o: list[np.ndarray] = []
    out_z: list[np.ndarray] = []
    out_p: list[np.ndarray] = []
    out_radius: list[float] = []
    out_ll: list[float] = []
    out_post_outlier_prob: list[np.ndarray] = []
    ksc_counts = np.zeros((n, 7), dtype=np.int64)
    ksc_move_sum = np.zeros(n)
    ksc_kept_sweeps = 0
    total_B_proposals = 0
    unstable_B_proposals = 0

    for iteration in range(sampler_config.reps):
        structural_variance = outlier_scales**2 * np.exp(log_variance_path[1:])
        B, attempts, radius = draw_bvar_coefficients_sv(
            Y,
            X,
            A,
            structural_variance,
            prior,
            n,
            p,
            rng,
            max_stability_tries=sampler_config.max_stability_tries,
        )
        total_B_proposals += attempts
        unstable_B_proposals += attempts - 1

        reduced, _ = structural_residuals(Y, X, B, A)
        A = draw_constant_cholesky_A(reduced, structural_variance, prior, rng)
        _, u = structural_residuals(Y, X, B, A)

        log_variance_path, phi, sv_diagnostics = draw_sv_block(
            log_variance_path=log_variance_path,
            phi=phi,
            structural_residuals_draw=u,
            outlier_scales=outlier_scales,
            h0_mean=h0_mean,
            h0_var=prior["h0_var"],
            offsets=offsets,
            prior=prior,
            rng=rng,
            method=sampler_config.sv_sampler,
        )
        if iteration >= sampler_config.burn:
            for j, ksc_diag in enumerate(sv_diagnostics):
                ksc_counts[j] += ksc_diag["component_counts"]
                ksc_move_sum[j] += ksc_diag["mean_abs_move"]

        lambda_t = np.exp(np.clip(log_variance_path[1:], -745.0, 700.0))
        standardized = u / np.sqrt(lambda_t)
        outlier_scales, outlier_indicators, posterior_outlier_probability = draw_outlier_states(
            standardized,
            outlier_probabilities,
            prior["outlier_grid"],
            rng,
        )
        outlier_probabilities = draw_outlier_probabilities(outlier_indicators, prior, rng)

        keep = (
            iteration >= sampler_config.burn
            and (iteration - sampler_config.burn) % sampler_config.thin == 0
        )
        if keep:
            out_B.append(B.copy())
            out_A.append(A.copy())
            out_h.append(log_variance_path.copy())
            out_phi.append(phi.copy())
            out_o.append(outlier_scales.copy())
            out_z.append(outlier_indicators.copy())
            out_p.append(outlier_probabilities.copy())
            out_radius.append(radius)
            out_ll.append(
                conditional_log_likelihood(
                    Y, X, B, A, log_variance_path[1:], outlier_scales
                )
            )
            out_post_outlier_prob.append(posterior_outlier_probability.copy())
            ksc_kept_sweeps += 1

        if sampler_config.progress_every and (iteration + 1) % sampler_config.progress_every == 0:
            rejection = unstable_B_proposals / max(total_B_proposals, 1)
            print(
                f"iteration {iteration + 1:,}/{sampler_config.reps:,} | "
                f"B instability rejection {rejection:.2%} | radius {radius:.4f}"
            )

    if not out_B:
        raise RuntimeError("No posterior draws were retained.")

    result = {
        "B": np.asarray(out_B),
        "A": np.asarray(out_A),
        "log_variance": np.asarray(out_h),
        "phi": np.asarray(out_phi),
        "outlier_scales": np.asarray(out_o),
        "outlier_indicators": np.asarray(out_z),
        "outlier_probabilities": np.asarray(out_p),
        "posterior_outlier_probability_draws": np.asarray(out_post_outlier_prob),
        "spectral_radius": np.asarray(out_radius),
        "log_likelihood": np.asarray(out_ll),
        "prior": prior,
        "prep": prep,
        "ols": ols,
        "prior_config": asdict(prior_config),
        "sampler_config": asdict(sampler_config),
        "ksc_offsets": offsets,
        "ksc_component_counts": ksc_counts,
        "ksc_mean_abs_move": ksc_move_sum / max(ksc_kept_sweeps, 1),
        "B_total_proposals": int(total_B_proposals),
        "B_unstable_proposals": int(unstable_B_proposals),
        "B_instability_rejection_rate": float(
            unstable_B_proposals / max(total_B_proposals, 1)
        ),
        "n_draws": len(out_B),
        "variables": list(prep["variables"]),
        "p": p,
        "sv_sampler": sampler_config.sv_sampler,
        "outlier_support": prior["outlier_grid"].copy(),
    }
    return result


# -----------------------------------------------------------------------------
# Posterior summaries and diagnostics
# -----------------------------------------------------------------------------


def chain_acf(x: np.ndarray, nlags: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < 2:
        return np.ones(1)
    nlags = min(int(nlags), len(x) - 1)
    centred = x - x.mean()
    denom = centred @ centred
    if denom <= 0:
        return np.r_[1.0, np.zeros(nlags)]
    return np.array(
        [1.0] + [float(centred[:-lag] @ centred[lag:] / denom) for lag in range(1, nlags + 1)]
    )


def effective_sample_size(x: np.ndarray) -> float:
    """Geyer initial-positive-sequence ESS for one retained chain."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    if n < 3 or np.allclose(x, x[0]):
        return float(n)
    rho = chain_acf(x, nlags=n - 1)
    tau = 1.0
    for lag in range(1, len(rho) - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair < 0:
            break
        tau += 2.0 * pair
    return float(n / max(tau, 1.0))


def posterior_summary(draws: np.ndarray, labels: Sequence[str], prior_mean=None, prior_sd=None) -> pd.DataFrame:
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != len(labels):
        raise ValueError("draws must be (n_draws, n_parameters) and match labels.")
    q = np.percentile(draws, [5, 16, 50, 84, 95], axis=0)
    table = pd.DataFrame(
        {
            "posterior_mean": draws.mean(axis=0),
            "posterior_median": q[2],
            "posterior_sd": draws.std(axis=0, ddof=1),
            "q05": q[0],
            "q16": q[1],
            "q84": q[3],
            "q95": q[4],
            "ESS": [effective_sample_size(draws[:, j]) for j in range(draws.shape[1])],
        },
        index=list(labels),
    )
    if prior_mean is not None:
        table.insert(0, "prior_mean", np.asarray(prior_mean, dtype=float))
    if prior_sd is not None:
        position = 1 if prior_mean is not None else 0
        table.insert(position, "prior_sd", np.asarray(prior_sd, dtype=float))
    return table


def coefficient_posterior_table(result: Mapping) -> pd.DataFrame:
    B = np.asarray(result["B"], dtype=float)
    flat = B.reshape(len(B), -1, order="F")
    prior = result["prior"]
    return posterior_summary(
        flat,
        prior["coefficient_labels"],
        prior_mean=prior["b0"],
        prior_sd=np.sqrt(prior["V0_diag"]),
    )


def key_parameter_draws(result: Mapping) -> dict[str, np.ndarray]:
    B = np.asarray(result["B"])
    A = np.asarray(result["A"])
    phi = np.asarray(result["phi"])
    p = np.asarray(result["outlier_probabilities"])
    variables = list(result["variables"])
    out: dict[str, np.ndarray] = {
        "log likelihood": np.asarray(result["log_likelihood"]),
        "spectral radius": np.asarray(result["spectral_radius"]),
    }
    if len(variables) >= 2:
        out[f"A[{variables[1]},{variables[0]}]"] = A[:, 1, 0]
    for j, name in enumerate(variables):
        out[f"phi[{name}]"] = phi[:, j]
        out[f"p_outlier[{name}]"] = p[:, j]
        out[f"own L1[{name}]"] = B[:, 1 + j, j]
    return out


def mcmc_diagnostics(result: Mapping) -> pd.DataFrame:
    rows = {}
    for name, draw in key_parameter_draws(result).items():
        draw = np.asarray(draw, dtype=float)
        ess = effective_sample_size(draw)
        sd = draw.std(ddof=1)
        rows[name] = {
            "posterior_mean": draw.mean(),
            "posterior_sd": sd,
            "ESS": ess,
            "MCSE_mean": sd / np.sqrt(max(ess, 1.0)),
            "MCSE_over_sd": 1.0 / np.sqrt(max(ess, 1.0)),
        }
    return pd.DataFrame(rows).T


def stationarity_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        series = data[column].dropna().to_numpy(dtype=float)
        adf_stat, adf_p, adf_lags, adf_nobs, *_ = adfuller(series, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, *_ = kpss(series, regression="c", nlags="auto")
        rows.append(
            {
                "variable": column,
                "ADF statistic": adf_stat,
                "ADF p-value": adf_p,
                "ADF lags": adf_lags,
                "KPSS statistic": kpss_stat,
                "KPSS p-value": kpss_p,
                "KPSS lags": kpss_lags,
                "observations": adf_nobs,
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def stability_diagnostics(result: Mapping) -> pd.Series:
    radii = np.asarray(result["spectral_radius"], dtype=float)
    return pd.Series(
        {
            "retained_draws": len(radii),
            "median_spectral_radius": np.median(radii),
            "q95_spectral_radius": np.percentile(radii, 95),
            "maximum_spectral_radius": radii.max(),
            "retained_unstable_share": np.mean(radii >= 1.0),
            "B_total_proposals": result["B_total_proposals"],
            "B_unstable_proposals": result["B_unstable_proposals"],
            "B_instability_rejection_rate": result["B_instability_rejection_rate"],
        }
    )


def posterior_outlier_probability(result: Mapping) -> pd.DataFrame:
    probability = np.asarray(result["outlier_indicators"], dtype=float).mean(axis=0)
    return pd.DataFrame(
        probability,
        index=result["prep"]["dates"],
        columns=result["variables"],
    )


def posterior_volatility_summary(result: Mapping, include_outliers: bool = False) -> dict:
    log_h = np.asarray(result["log_variance"], dtype=float)[:, 1:, :]
    sd = np.exp(0.5 * np.clip(log_h, -745.0, 700.0))
    if include_outliers:
        sd = sd * np.asarray(result["outlier_scales"], dtype=float)
    q = np.percentile(sd, [16, 50, 84], axis=0)
    return {
        "dates": result["prep"]["dates"],
        "variables": result["variables"],
        "lower": q[0],
        "median": q[1],
        "upper": q[2],
        "include_outliers": include_outliers,
    }


def standardized_structural_residuals(result: Mapping) -> pd.DataFrame:
    B = np.median(np.asarray(result["B"]), axis=0)
    A = np.median(np.asarray(result["A"]), axis=0)
    log_h = np.median(np.asarray(result["log_variance"]), axis=0)[1:]
    o = np.median(np.asarray(result["outlier_scales"]), axis=0)
    _, u = structural_residuals(result["prep"]["Y"], result["prep"]["X"], B, A)
    std = u / (o * np.exp(0.5 * log_h))
    return pd.DataFrame(std, index=result["prep"]["dates"], columns=result["variables"])


def residual_diagnostic_table(result: Mapping) -> pd.DataFrame:
    eps = standardized_structural_residuals(result)
    rows = []
    for column in eps:
        x = eps[column].to_numpy()
        rows.append(
            {
                "variable": column,
                "mean": x.mean(),
                "sd": x.std(ddof=1),
                "skewness": pd.Series(x).skew(),
                "excess_kurtosis": pd.Series(x).kurt(),
                "acf_1": chain_acf(x, 1)[1],
                "maximum_absolute": np.max(np.abs(x)),
            }
        )
    return pd.DataFrame(rows).set_index("variable")


# -----------------------------------------------------------------------------
# Durbin--Koopman ragged-edge and conditional forecasting
# -----------------------------------------------------------------------------


def build_time_varying_covariances(
    A: np.ndarray,
    log_variance: np.ndarray,
    outlier_scales: np.ndarray,
) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    log_variance = np.asarray(log_variance, dtype=float)
    outlier_scales = np.asarray(outlier_scales, dtype=float)
    if log_variance.shape != outlier_scales.shape:
        raise ValueError("log_variance and outlier_scales must have the same shape.")
    Ainv = np.linalg.solve(A, np.eye(A.shape[0]))
    q = outlier_scales**2 * np.exp(np.clip(log_variance, -745.0, 700.0))
    return np.asarray([Ainv @ np.diag(q_t) @ Ainv.T for q_t in q])


def _companion_components(B: np.ndarray, n: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    F = var_companion(B, n, p)
    intercept = np.zeros(n * p)
    intercept[:n] = B[0]
    selection = np.vstack([np.eye(n), np.zeros((n * (p - 1), n))])
    design = np.hstack([np.eye(n), np.zeros((n, n * (p - 1)))])
    return F, intercept, selection, design


def _durbin_koopman_companion_draw(
    endog: np.ndarray,
    B: np.ndarray,
    covariance_sequence: np.ndarray,
    initial_companion_state: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    endog = np.asarray(endog, dtype=float)
    covariance_sequence = np.asarray(covariance_sequence, dtype=float)
    L, n = endog.shape
    p = (B.shape[0] - 1) // n
    if covariance_sequence.shape != (L, n, n):
        raise ValueError("covariance_sequence must have shape (L, n, n).")
    F, intercept, selection, design = _companion_components(B, n, p)
    k_states = n * p

    initial_mean = intercept + F @ np.asarray(initial_companion_state, dtype=float)
    initial_cov = selection @ covariance_sequence[0] @ selection.T

    model = MLEModel(endog=endog, k_states=k_states, k_posdef=n)
    model["design"] = design
    model["obs_cov"] = np.zeros((n, n))
    model["transition"] = F
    model["state_intercept"] = intercept
    model["selection"] = selection

    transition_covariance = np.empty_like(covariance_sequence)
    if L > 1:
        transition_covariance[:-1] = covariance_sequence[1:]
    transition_covariance[-1] = covariance_sequence[-1]
    model["state_cov"] = np.moveaxis(transition_covariance, 0, -1)
    model.initialize_known(initial_mean, 0.5 * (initial_cov + initial_cov.T))

    simulator = model.simulation_smoother(method="kfs")
    simulator.simulate(random_state=rng)
    return simulator.simulated_state.T.copy()


def _normalise_level_conditions(
    conditions: Mapping[str, float | Sequence[float]] | None,
    variables: Sequence[str],
    H: int,
    levels: pd.DataFrame,
) -> dict[str, np.ndarray]:
    if conditions is None:
        return {}
    out: dict[str, np.ndarray] = {}
    for variable, value in conditions.items():
        if variable not in variables:
            raise ValueError(f"Unknown conditioned variable {variable!r}.")
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.repeat(float(arr), H)
        if arr.ndim != 1 or len(arr) != H:
            raise ValueError(f"Condition for {variable!r} must be scalar or length H={H}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Level conditions must be finite; omit unconstrained variables instead.")
        if not np.isfinite(levels[variable].iloc[-1]):
            raise ValueError(
                f"The latest level of conditioned variable {variable!r} is missing. "
                "A future level path cannot be converted into exact differences."
            )
        out[variable] = arr
    return out


def forecast_bvar_sv_outlier(
    result: Mapping,
    H: int = 12,
    level_conditions: Mapping[str, float | Sequence[float]] | None = None,
    n_draws: int | None = None,
    simulate_future_outliers: bool = True,
    seed: int = 123,
) -> dict:
    """Draw ragged-edge nowcasts and future forecasts with DK smoothing.

    Conditions are supplied as future *level paths*. They are converted to
    exact future absolute changes and entered as observed values in the
    state-space system; all unconstrained variables remain missing.
    """
    if H < 1:
        raise ValueError("H must be at least 1.")
    prep = result["prep"]
    levels = prep["levels"]
    variables = list(result["variables"])
    n = len(variables)
    conditions = _normalise_level_conditions(level_conditions, variables, H, levels)
    rng = np.random.default_rng(seed)

    tail_dates = levels.index[levels.index > prep["balanced_end"]]
    future_dates = pd.date_range(levels.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq="MS")
    path_dates = tail_dates.append(future_dates)
    tail_length = len(tail_dates)
    L = len(path_dates)

    observed_changes = levels.diff().reindex(path_dates)
    # pandas may expose a read-only NumPy view (notably with Copy-on-Write).
    # The template is mutated below when future observations are set to NaN
    # and conditioning paths are inserted, so force an independent writable copy.
    endog_template = observed_changes.to_numpy(dtype=float, copy=True)
    if H:
        endog_template[tail_length:] = np.nan
    for variable, path in conditions.items():
        column = variables.index(variable)
        previous = float(levels[variable].iloc[-1])
        endog_template[tail_length:, column] = np.diff(np.r_[previous, path])

    available_draws = len(result["B"])
    if n_draws is None or n_draws >= available_draws:
        draw_indices = np.arange(available_draws)
    else:
        draw_indices = np.sort(rng.choice(available_draws, size=int(n_draws), replace=False))

    diff_paths = np.empty((len(draw_indices), L, n))
    level_paths = np.empty_like(diff_paths)
    future_log_variance = np.empty_like(diff_paths)
    future_outlier_scales = np.empty_like(diff_paths)
    base_level = np.asarray(prep["level_at_balanced_end"], dtype=float)
    outlier_grid = np.asarray(result["prior"]["outlier_grid"], dtype=float)

    for out_index, draw_index in enumerate(draw_indices):
        B = np.asarray(result["B"])[draw_index]
        A = np.asarray(result["A"])[draw_index]
        phi = np.asarray(result["phi"])[draw_index]
        p_outlier = np.asarray(result["outlier_probabilities"])[draw_index]
        last_h = np.asarray(result["log_variance"])[draw_index, -1]

        shocks_h = rng.standard_normal((L, n)) * np.sqrt(phi)[None, :]
        h_path = last_h[None, :] + np.cumsum(shocks_h, axis=0)
        if simulate_future_outliers:
            z = rng.random((L, n)) < p_outlier[None, :]
            o_path = np.ones((L, n))
            for j in range(n):
                count = int(z[:, j].sum())
                if count:
                    o_path[z[:, j], j] = rng.choice(outlier_grid, size=count, replace=True)
        else:
            o_path = np.ones((L, n))

        Sigma_path = build_time_varying_covariances(A, h_path, o_path)
        state_path = _durbin_koopman_companion_draw(
            endog_template,
            B,
            Sigma_path,
            prep["last_companion_state"],
            rng,
        )
        diff_path = state_path[:, :n]
        diff_paths[out_index] = diff_path
        level_paths[out_index] = base_level[None, :] + np.cumsum(diff_path, axis=0)
        future_log_variance[out_index] = h_path
        future_outlier_scales[out_index] = o_path

    return {
        "diff_paths": diff_paths,
        "level_paths": level_paths,
        "path_dates": path_dates,
        "tail_dates": tail_dates,
        "future_dates": future_dates,
        "tail_length": tail_length,
        "future_diff_paths": diff_paths[:, tail_length:],
        "future_level_paths": level_paths[:, tail_length:],
        "future_log_variance": future_log_variance,
        "future_outlier_scales": future_outlier_scales,
        "variables": variables,
        "draw_indices": draw_indices,
        "level_conditions": conditions,
        "simulate_future_outliers": bool(simulate_future_outliers),
        "balanced_end": prep["balanced_end"],
        "last_calendar_date": prep["last_calendar_date"],
        "H": H,
    }


# -----------------------------------------------------------------------------
# Lightweight posterior predictive check
# -----------------------------------------------------------------------------


def posterior_predictive_statistics(
    result: Mapping,
    n_draws: int = 200,
    seed: int = 456,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare observed transformed-data statistics with replicated paths."""
    rng = np.random.default_rng(seed)
    Y = result["prep"]["Y"]
    n = Y.shape[1]
    p = result["p"]
    available = len(result["B"])
    indices = np.arange(available) if n_draws >= available else rng.choice(available, n_draws, replace=False)
    replicated = []

    initial = result["prep"]["balanced"].iloc[:p].to_numpy(dtype=float)
    for draw_index in indices:
        B = result["B"][draw_index]
        A = result["A"][draw_index]
        Ainv = np.linalg.solve(A, np.eye(n))
        log_h = result["log_variance"][draw_index, 1:]
        o = result["outlier_scales"][draw_index]
        history = [row.copy() for row in initial]
        path = np.empty_like(Y)
        for t in range(len(Y)):
            x = np.r_[1.0, np.concatenate(history[-p:][::-1])]
            structural_shock = o[t] * np.exp(0.5 * log_h[t]) * rng.standard_normal(n)
            y_new = x @ B + Ainv @ structural_shock
            path[t] = y_new
            history.append(y_new)
        replicated.append(path)
    replicated = np.asarray(replicated)

    def stats(arr: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                arr.mean(axis=-2),
                arr.std(axis=-2, ddof=1),
                np.max(np.abs(arr), axis=-2),
            ],
            axis=-1,
        )

    observed_stats = stats(Y[None, :, :])[0]
    replicated_stats = stats(replicated)
    labels = []
    obs_values = []
    rep_columns = []
    for j, name in enumerate(result["variables"]):
        for stat_index, stat_name in enumerate(["mean", "sd", "maximum_absolute"]):
            labels.append(f"{name}: {stat_name}")
            obs_values.append(observed_stats[j, stat_index])
            rep_columns.append(replicated_stats[:, j, stat_index])
    summary = pd.DataFrame(
        {
            "observed": obs_values,
            "replicated_median": [np.median(x) for x in rep_columns],
            "replicated_q05": [np.percentile(x, 5) for x in rep_columns],
            "replicated_q95": [np.percentile(x, 95) for x in rep_columns],
        },
        index=labels,
    )
    draws = pd.DataFrame(np.column_stack(rep_columns), columns=labels)
    return summary, draws


# -----------------------------------------------------------------------------
# Presentation-oriented tables
# -----------------------------------------------------------------------------


def coefficient_posterior_tables(result: Mapping) -> dict[str, pd.DataFrame]:
    """Return one posterior coefficient table per VAR equation.

    Keeping equations separate prevents the larger coefficients
    from determining the visual scale used for another equation.
    """
    table = coefficient_posterior_table(result)
    out: dict[str, pd.DataFrame] = {}
    for equation in result["variables"]:
        prefix = f"{equation}:"
        selected = table.loc[[str(index).startswith(prefix) for index in table.index]].copy()
        selected.index = [str(index).split(":", 1)[1].strip() for index in selected.index]
        selected.index.name = "regressor"
        out[equation] = selected
    return out


def nowcast_summary_table(
    forecast: Mapping,
    observed_levels: pd.DataFrame,
    variable: str,
    quantiles: Sequence[float] = (5, 16, 50, 84, 95),
    missing_only: bool = True,
) -> pd.DataFrame:
    """Summarise the ragged-edge part of a DK forecast for one equation."""
    variables = list(forecast["variables"])
    if variable not in variables:
        raise ValueError(f"Unknown variable {variable!r}.")
    tail_length = int(forecast["tail_length"])
    columns = ["status", "observed", "q05", "q16", "median", "q84", "q95"]
    if tail_length == 0:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    j = variables.index(variable)
    dates = pd.DatetimeIndex(forecast["tail_dates"], name="date")
    paths = np.asarray(forecast["level_paths"], dtype=float)[:, :tail_length, j]
    q = np.percentile(paths, quantiles, axis=0)
    actual = observed_levels[variable].reindex(dates)
    status = np.where(actual.notna(), "observed / conditioned", "nowcast")
    table = pd.DataFrame(
        {
            "status": status,
            "observed": actual.to_numpy(dtype=float),
            "q05": q[0],
            "q16": q[1],
            "median": q[2],
            "q84": q[3],
            "q95": q[4],
        },
        index=dates,
    )
    if missing_only:
        table = table.loc[table["status"] == "nowcast"]
    return table


def _short_horizon_targets(
    future_dates: Sequence[pd.Timestamp],
    origin_date: pd.Timestamp,
) -> list[tuple[str, int, pd.Timestamp]]:
    dates = pd.DatetimeIndex(future_dates)
    if len(dates) < 3:
        raise ValueError("At least three future months are required for 1m/3m summaries.")
    origin = pd.Timestamp(origin_date).to_period("M").to_timestamp(how="start")
    year_end_year = origin.year if origin.month < 12 else origin.year + 1
    year_end = pd.Timestamp(year=year_end_year, month=12, day=1)
    if year_end not in dates:
        required = (year_end.year - origin.year) * 12 + year_end.month - origin.month
        raise ValueError(
            f"Forecast horizon is too short for end-{year_end_year}. "
            f"Need at least {required} future months."
        )
    targets = [
        ("1 month", 1, dates[0]),
        ("3 months", 3, dates[2]),
        (f"End of {year_end_year}", int(np.flatnonzero(dates == year_end)[0]) + 1, year_end),
    ]
    # December/January origins can make the end-of-year target coincide with 12m;
    # preserve all requested labels while avoiding duplicate rows by date/horizon.
    return targets


def path_horizon_table(
    paths: np.ndarray,
    dates: Sequence[pd.Timestamp],
    origin_date: pd.Timestamp,
    quantiles: Sequence[float] = (5, 16, 50, 84, 95),
) -> pd.DataFrame:
    """Create the 1-month, 3-month and end-of-year table for path draws."""
    paths = np.asarray(paths, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if paths.ndim != 2 or paths.shape[1] != len(dates):
        raise ValueError("paths must have shape (draws, len(dates)).")
    targets = _short_horizon_targets(dates, pd.Timestamp(origin_date))
    rows = []
    for label, horizon, date in targets:
        index = int(np.flatnonzero(dates == date)[0])
        values = paths[:, index]
        q = np.percentile(values, quantiles)
        rows.append(
            {
                "horizon": label,
                "months_ahead": horizon,
                "target_date": date,
                "q05": q[0],
                "q16": q[1],
                "median": q[2],
                "q84": q[3],
                "q95": q[4],
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


def forecast_horizon_table(
    forecast: Mapping,
    variable: str,
    quantiles: Sequence[float] = (5, 16, 50, 84, 95),
) -> pd.DataFrame:
    """Short-horizon level forecast table for one BVAR equation."""
    variables = list(forecast["variables"])
    if variable not in variables:
        raise ValueError(f"Unknown variable {variable!r}.")
    j = variables.index(variable)
    return path_horizon_table(
        np.asarray(forecast["future_level_paths"], dtype=float)[:, :, j],
        forecast["future_dates"],
        forecast["last_calendar_date"],
        quantiles=quantiles,
    )

# -----------------------------------------------------------------------------
# Structural analysis: recursive and sign-restricted IRF, FEVD and HD
# -----------------------------------------------------------------------------


def _var_lag_matrices(B: np.ndarray, n: int, p: int) -> list[np.ndarray]:
    """Return VAR lag matrices B_1,...,B_p from the module's (k,n) convention."""
    B = np.asarray(B, dtype=float)
    expected = 1 + n * p
    if B.shape != (expected, n):
        raise ValueError(f"B must have shape {(expected, n)}, got {B.shape}.")
    return [B[1 + lag * n : 1 + (lag + 1) * n].T for lag in range(p)]


def _reference_regression_index(result: Mapping, reference_date=None) -> tuple[int, pd.Timestamp]:
    dates = pd.DatetimeIndex(result["prep"]["dates"])
    if len(dates) == 0:
        raise ValueError("The estimation sample contains no regression dates.")
    if reference_date is None:
        return len(dates) - 1, pd.Timestamp(dates[-1])
    target = pd.Timestamp(reference_date).to_period("M").to_timestamp(how="start")
    matches = np.flatnonzero(dates == target)
    if len(matches) == 0:
        raise ValueError(
            f"reference_date {target.date()} is outside the balanced regression sample "
            f"[{dates[0].date()}, {dates[-1].date()}]."
        )
    return int(matches[0]), target


def _regular_impact_matrix(A: np.ndarray, log_variance: np.ndarray) -> np.ndarray:
    """A^{-1} Lambda^{1/2}; outlier scale deliberately excluded."""
    sd = np.exp(0.5 * np.clip(np.asarray(log_variance, dtype=float), -745.0, 700.0))
    return np.linalg.solve(np.asarray(A, dtype=float), np.diag(sd))


def _full_impact_matrix(
    A: np.ndarray,
    log_variance: np.ndarray,
    outlier_scale: np.ndarray,
) -> np.ndarray:
    """A^{-1} O Lambda^{1/2}, used for exact historical reconstruction."""
    sd = np.exp(0.5 * np.clip(np.asarray(log_variance, dtype=float), -745.0, 700.0))
    return np.linalg.solve(
        np.asarray(A, dtype=float),
        np.diag(np.asarray(outlier_scale, dtype=float) * sd),
    )


def recursive_impact_draws(
    result: Mapping,
    reference_date=None,
    include_outlier_scale: bool = False,
) -> dict:
    """Construct one recursively identified impact matrix per posterior draw.

    By default the transitory outlier multiplier is set to one, so the impacts
    describe regular structural shocks at the selected stochastic-volatility
    state. Set ``include_outlier_scale=True`` only for an event-specific exercise.
    """
    ref_index, ref_date = _reference_regression_index(result, reference_date)
    A_draws = np.asarray(result["A"], dtype=float)
    h_draws = np.asarray(result["log_variance"], dtype=float)
    o_draws = np.asarray(result["outlier_scales"], dtype=float)
    impacts = []
    for s in range(len(A_draws)):
        h = h_draws[s, ref_index + 1]
        if include_outlier_scale:
            P = _full_impact_matrix(A_draws[s], h, o_draws[s, ref_index])
        else:
            P = _regular_impact_matrix(A_draws[s], h)
        impacts.append(P)
    return {
        "impact_draws": np.asarray(impacts),
        "rotations": np.repeat(np.eye(len(result["variables"]))[None, :, :], len(impacts), axis=0),
        "used_draw_indices": np.arange(len(impacts), dtype=int),
        "shock_names": list(result["variables"]),
        "identification": "recursive",
        "reference_date": ref_date,
        "include_outlier_scale": bool(include_outlier_scale),
        "acceptance_rate": 1.0,
    }


def _orthogonal_draw(rng: np.random.Generator, n: int) -> np.ndarray:
    Q, R = np.linalg.qr(rng.normal(size=(n, n)))
    Q = Q @ np.diag(np.where(np.diag(R) >= 0.0, 1.0, -1.0))
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1.0
    return Q


def _irf_from_B_and_impact(B: np.ndarray, p: int, impact: np.ndarray, horizon: int) -> np.ndarray:
    n = impact.shape[0]
    lags = _var_lag_matrices(B, n=n, p=p)
    out = np.zeros((horizon + 1, n, n), dtype=float)
    out[0] = impact
    for h in range(1, horizon + 1):
        for lag in range(1, min(p, h) + 1):
            out[h] += lags[lag - 1] @ out[h - lag]
    return out


def _restriction_ok(
    irf: np.ndarray,
    variables: Sequence[str],
    shock_names: Sequence[str],
    sign_restrictions: Mapping,
    cumulative_restrictions: Mapping | None,
    tolerance: float,
) -> bool:
    var_index = {name: i for i, name in enumerate(variables)}
    shock_index = {name: i for i, name in enumerate(shock_names)}
    for shock, response_map in sign_restrictions.items():
        j = shock_index[shock]
        for response, horizon_map in response_map.items():
            i = var_index[response]
            for h, sign in horizon_map.items():
                value = irf[int(h), i, j]
                if int(sign) > 0 and not value > tolerance:
                    return False
                if int(sign) < 0 and not value < -tolerance:
                    return False
    if cumulative_restrictions:
        for shock, response_map in cumulative_restrictions.items():
            j = shock_index[shock]
            for response, rules in response_map.items():
                i = var_index[response]
                for horizon, sign in rules.items():
                    value = irf[: int(horizon) + 1, i, j].sum()
                    if int(sign) > 0 and not value > tolerance:
                        return False
                    if int(sign) < 0 and not value < -tolerance:
                        return False
    return True


def sign_restricted_impact_draws(
    result: Mapping,
    sign_restrictions: Mapping,
    reference_date=None,
    cumulative_restrictions: Mapping | None = None,
    restriction_horizon: int | None = None,
    include_outlier_scale: bool = False,
    max_tries: int = 10_000,
    seed: int = 42,
    tolerance: float = 1e-10,
) -> dict:
    """Identify structural shocks by orthogonal rotations and sign restrictions.

    ``sign_restrictions`` has the form
    ``{shock: {response: {horizon: +1/-1}}}``.  The shock insertion order fixes
    the columns of the accepted impact matrix.  The same accepted rotation can
    subsequently be used at every date in the historical decomposition.
    """
    variables = list(result["variables"])
    n = len(variables)
    shock_names = list(sign_restrictions)
    if len(shock_names) != n:
        raise ValueError(
            "Provide one named restriction block per structural shock. "
            f"Expected {n}, received {len(shock_names)}."
        )
    unknown_responses = {
        response
        for block in sign_restrictions.values()
        for response in block
        if response not in variables
    }
    if unknown_responses:
        raise ValueError(f"Unknown response variables: {sorted(unknown_responses)}")
    requested_horizons = [
        int(h)
        for block in sign_restrictions.values()
        for rules in block.values()
        for h in rules
    ]
    if cumulative_restrictions:
        requested_horizons += [
            int(h)
            for block in cumulative_restrictions.values()
            for rules in block.values()
            for h in rules
        ]
    H = max(requested_horizons + [0]) if restriction_horizon is None else int(restriction_horizon)
    if H < max(requested_horizons + [0]):
        raise ValueError("restriction_horizon is shorter than a requested restriction.")

    baseline = recursive_impact_draws(
        result,
        reference_date=reference_date,
        include_outlier_scale=include_outlier_scale,
    )
    rng = np.random.default_rng(seed)
    accepted_impacts = []
    accepted_rotations = []
    used = []
    attempts = []
    B_draws = np.asarray(result["B"], dtype=float)
    p = int(result["p"])

    for s, P0 in enumerate(baseline["impact_draws"]):
        found = False
        for attempt in range(1, int(max_tries) + 1):
            Q = _orthogonal_draw(rng, n)
            P = P0 @ Q
            irf = _irf_from_B_and_impact(B_draws[s], p, P, H)
            if _restriction_ok(
                irf,
                variables,
                shock_names,
                sign_restrictions,
                cumulative_restrictions,
                tolerance,
            ):
                accepted_impacts.append(P)
                accepted_rotations.append(Q)
                used.append(s)
                attempts.append(attempt)
                found = True
                break
        if not found:
            continue

    if not accepted_impacts:
        raise RuntimeError(
            "No posterior draw satisfied the sign restrictions. Relax the signs, "
            "increase max_tries, or inspect whether the restrictions conflict."
        )
    return {
        "impact_draws": np.asarray(accepted_impacts),
        "rotations": np.asarray(accepted_rotations),
        "used_draw_indices": np.asarray(used, dtype=int),
        "shock_names": shock_names,
        "identification": "sign_restrictions",
        "reference_date": baseline["reference_date"],
        "include_outlier_scale": bool(include_outlier_scale),
        "acceptance_rate": len(accepted_impacts) / len(B_draws),
        "mean_attempts_per_accepted_draw": float(np.mean(attempts)),
        "max_tries": int(max_tries),
        "sign_restrictions": sign_restrictions,
        "cumulative_restrictions": cumulative_restrictions,
    }


def impulse_responses(
    result: Mapping,
    identification: str = "recursive",
    reference_date=None,
    sign_restrictions: Mapping | None = None,
    cumulative_restrictions: Mapping | None = None,
    horizon: int = 24,
    shock_unit: str = "structural_std",
    shock_size: float = 1.0,
    normalization_variables: Mapping[str, str] | None = None,
    include_outlier_scale: bool = False,
    max_tries: int = 10_000,
    seed: int = 42,
) -> dict:
    """Posterior IRFs in monthly changes and cumulative price levels."""
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    identification = identification.lower()
    if identification in {"recursive", "cholesky"}:
        identified = recursive_impact_draws(
            result, reference_date, include_outlier_scale
        )
    elif identification in {"sign", "sign_restrictions", "sign-restrictions"}:
        if sign_restrictions is None:
            raise ValueError("sign_restrictions is required for sign identification.")
        identified = sign_restricted_impact_draws(
            result,
            sign_restrictions=sign_restrictions,
            reference_date=reference_date,
            cumulative_restrictions=cumulative_restrictions,
            include_outlier_scale=include_outlier_scale,
            max_tries=max_tries,
            seed=seed,
        )
    else:
        raise ValueError("identification must be 'recursive' or 'sign'.")

    B_draws = np.asarray(result["B"], dtype=float)
    variables = list(result["variables"])
    p = int(result["p"])
    change_irfs = []
    scale_factors = []
    for P, s in zip(identified["impact_draws"], identified["used_draw_indices"]):
        if shock_unit == "structural_std":
            scaled = P * float(shock_size)
            factors = np.full(len(variables), float(shock_size))
        elif shock_unit == "level":
            targets = []
            for j, shock_name in enumerate(identified["shock_names"]):
                if normalization_variables and shock_name in normalization_variables:
                    target_name = normalization_variables[shock_name]
                elif shock_name in variables:
                    target_name = shock_name
                elif sign_restrictions and shock_name in sign_restrictions:
                    impact_candidates = [
                        response
                        for response, rules in sign_restrictions[shock_name].items()
                        if 0 in {int(h) for h in rules}
                    ]
                    if not impact_candidates:
                        raise ValueError(
                            f"No impact-normalisation variable is available for {shock_name!r}. "
                            "Pass normalization_variables explicitly."
                        )
                    target_name = impact_candidates[0]
                else:
                    raise ValueError(
                        f"Pass normalization_variables for shock {shock_name!r}."
                    )
                if target_name not in variables:
                    raise ValueError(f"Unknown normalization variable {target_name!r}.")
                targets.append(variables.index(target_name))
            own_impacts = np.array([P[targets[j], j] for j in range(len(variables))])
            if np.any(np.abs(own_impacts) < 1e-12):
                raise ValueError("A shock has a near-zero normalisation impact.")
            factors = float(shock_size) / own_impacts
            scaled = P @ np.diag(factors)
        else:
            raise ValueError("shock_unit must be 'structural_std' or 'level'.")
        change_irfs.append(_irf_from_B_and_impact(B_draws[int(s)], p, scaled, horizon))
        scale_factors.append(factors)
    change_irfs = np.asarray(change_irfs)
    return {
        **identified,
        "change_irfs": change_irfs,
        "cumulative_level_irfs": np.cumsum(change_irfs, axis=1),
        "horizons": np.arange(horizon + 1),
        "variables": variables,
        "shock_unit": shock_unit,
        "shock_size": float(shock_size),
        "scale_factors": np.asarray(scale_factors),
        "axes": ("posterior_draw", "horizon", "response", "shock"),
    }


def forecast_error_variance_decomposition(
    result: Mapping,
    identification: str = "recursive",
    reference_date=None,
    sign_restrictions: Mapping | None = None,
    cumulative_restrictions: Mapping | None = None,
    horizon: int = 24,
    include_outlier_scale: bool = False,
    max_tries: int = 10_000,
    seed: int = 42,
) -> dict:
    """Date-specific structural FEVD based on regular one-standard-deviation shocks."""
    irf = impulse_responses(
        result,
        identification=identification,
        reference_date=reference_date,
        sign_restrictions=sign_restrictions,
        cumulative_restrictions=cumulative_restrictions,
        horizon=horizon - 1,
        shock_unit="structural_std",
        shock_size=1.0,
        include_outlier_scale=include_outlier_scale,
        max_tries=max_tries,
        seed=seed,
    )
    squared = np.cumsum(irf["change_irfs"] ** 2, axis=1)
    total = squared.sum(axis=3, keepdims=True)
    shares = np.divide(squared, total, out=np.zeros_like(squared), where=total > 0)
    return {
        **{k: v for k, v in irf.items() if k not in {"change_irfs", "cumulative_level_irfs"}},
        "fevd_draws": shares,
        "horizons": np.arange(1, horizon + 1),
        "axes": ("posterior_draw", "forecast_horizon", "response", "shock"),
        "max_share_sum_error": float(np.max(np.abs(shares.sum(axis=3) - 1.0))),
    }


def historical_decomposition(
    result: Mapping,
    identification: str = "recursive",
    reference_date=None,
    sign_restrictions: Mapping | None = None,
    cumulative_restrictions: Mapping | None = None,
    split_outlier_amplification: bool = True,
    max_tries: int = 10_000,
    seed: int = 42,
    reconstruction_tol: float = 1e-8,
) -> dict:
    """Exact draw-by-draw historical decomposition of monthly price changes.

    A sign-restricted draw uses one accepted rotation for all historical dates.
    When ``split_outlier_amplification`` is true, every structural shock is split
    into a regular component and the transitory amplification generated by O_t.
    """
    if identification.lower() in {"recursive", "cholesky"}:
        identified = recursive_impact_draws(result, reference_date, False)
    else:
        if sign_restrictions is None:
            raise ValueError("sign_restrictions is required for sign identification.")
        identified = sign_restricted_impact_draws(
            result,
            sign_restrictions,
            reference_date=reference_date,
            cumulative_restrictions=cumulative_restrictions,
            include_outlier_scale=False,
            max_tries=max_tries,
            seed=seed,
        )

    prep = result["prep"]
    balanced = np.asarray(prep["balanced"], dtype=float)
    Y = np.asarray(prep["Y"], dtype=float)
    X = np.asarray(prep["X"], dtype=float)
    dates = pd.DatetimeIndex(prep["dates"])
    B_draws = np.asarray(result["B"], dtype=float)
    A_draws = np.asarray(result["A"], dtype=float)
    h_draws = np.asarray(result["log_variance"], dtype=float)
    o_draws = np.asarray(result["outlier_scales"], dtype=float)
    p = int(result["p"])
    n = len(result["variables"])
    T = len(Y)

    all_contrib = []
    all_base = []
    all_recon = []
    all_eps = []
    max_errors = []
    component_names = []
    if split_outlier_amplification:
        for name in identified["shock_names"]:
            component_names.extend([f"{name}: regular", f"{name}: outlier amplification"])
    else:
        component_names = list(identified["shock_names"])

    for Q, draw_index in zip(identified["rotations"], identified["used_draw_indices"]):
        s = int(draw_index)
        B = B_draws[s]
        lag_mats = _var_lag_matrices(B, n, p)
        reduced = Y - X @ B
        base_full = np.zeros_like(balanced)
        base_full[:p] = balanced[:p]
        n_components = 2 * n if split_outlier_amplification else n
        contrib_full = np.zeros((len(balanced), n, n_components), dtype=float)
        eps_path = np.zeros((T, n), dtype=float)

        for r in range(T):
            t = p + r
            base_full[t] = B[0]
            for lag in range(1, p + 1):
                base_full[t] += lag_mats[lag - 1] @ base_full[t - lag]
                for c in range(n_components):
                    contrib_full[t, :, c] += lag_mats[lag - 1] @ contrib_full[t - lag, :, c]

            Preg = _regular_impact_matrix(A_draws[s], h_draws[s, r + 1]) @ Q
            Pfull = _full_impact_matrix(A_draws[s], h_draws[s, r + 1], o_draws[s, r]) @ Q
            eps = np.linalg.solve(Pfull, reduced[r])
            eps_path[r] = eps
            if split_outlier_amplification:
                for j in range(n):
                    contrib_full[t, :, 2 * j] += Preg[:, j] * eps[j]
                    contrib_full[t, :, 2 * j + 1] += (Pfull[:, j] - Preg[:, j]) * eps[j]
            else:
                for j in range(n):
                    contrib_full[t, :, j] += Pfull[:, j] * eps[j]

        base = base_full[p:]
        contributions = contrib_full[p:]
        reconstructed = base + contributions.sum(axis=2)
        error = float(np.max(np.abs(reconstructed - Y)))
        if error > reconstruction_tol:
            raise RuntimeError(
                f"Historical decomposition failed reconstruction for draw {s}: "
                f"max error={error:.3e}."
            )
        all_base.append(base)
        all_contrib.append(contributions)
        all_recon.append(reconstructed)
        all_eps.append(eps_path)
        max_errors.append(error)

    return {
        **identified,
        "contributions": np.asarray(all_contrib),
        "base": np.asarray(all_base),
        "reconstructed": np.asarray(all_recon),
        "structural_shocks": np.asarray(all_eps),
        "observed": Y,
        "dates": dates,
        "variables": list(result["variables"]),
        "component_names": component_names,
        "split_outlier_amplification": bool(split_outlier_amplification),
        "max_reconstruction_error": float(max(max_errors)),
        "axes": ("posterior_draw", "time", "response", "component"),
    }


# -----------------------------------------------------------------------------
# Reproducible run metadata
# -----------------------------------------------------------------------------


def _stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_model_data(levels: pd.DataFrame) -> str:
    """Hash the exact transformed-input panel passed to the model."""
    frame = levels.copy()
    digest = hashlib.sha256()
    digest.update(_stable_json(list(frame.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame.index, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def hash_run_config(
    prior_config: BVARSVOPriorConfig,
    sampler_config: SamplerConfig,
    *,
    p: int,
    variables: Sequence[str],
) -> str:
    payload = {
        "p": int(p),
        "variables": list(variables),
        "prior": asdict(prior_config),
        "sampler": asdict(sampler_config),
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def build_run_metadata(
    *,
    model_id: str,
    vintage: str,
    levels: pd.DataFrame,
    p: int,
    variables: Sequence[str],
    prior_config: BVARSVOPriorConfig,
    sampler_config: SamplerConfig,
    code_version: str = "unversioned",
    result_schema_version: str = "1.0",
) -> dict:
    data_hash = hash_model_data(levels)
    config_hash = hash_run_config(
        prior_config,
        sampler_config,
        p=p,
        variables=variables,
    )
    identity = {
        "model_id": str(model_id),
        "vintage": str(vintage),
        "data_hash": data_hash,
        "config_hash": config_hash,
        "code_version": str(code_version),
        "seed": int(sampler_config.seed),
    }
    run_id = hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()
    return {
        **identity,
        "run_id": run_id,
        "result_schema_version": result_schema_version,
        "frequency": "monthly",
        "variables": list(variables),
        "p": int(p),
        "n_observations": int(len(levels)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_energy_bvar(
    levels: pd.DataFrame,
    *,
    model_id: str,
    vintage: str,
    p: int = 12,
    variables: Sequence[str] | None = None,
    prior_config: BVARSVOPriorConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    code_version: str = "unversioned",
) -> dict:
    """Run the generic engine and attach a cache-safe run identity."""
    prior_config = BVARSVOPriorConfig() if prior_config is None else prior_config
    sampler_config = SamplerConfig() if sampler_config is None else sampler_config
    variables = list(levels.columns) if variables is None else list(variables)
    result = gibbs_bvar_sv_outlier(
        levels=levels,
        p=p,
        variables=variables,
        prior_config=prior_config,
        sampler_config=sampler_config,
    )
    result["metadata"] = build_run_metadata(
        model_id=model_id,
        vintage=vintage,
        levels=levels[variables],
        p=p,
        variables=variables,
        prior_config=prior_config,
        sampler_config=sampler_config,
        code_version=code_version,
    )
    return result



# Backward-compatible structural aliases. New notebooks should use the generic names.
recursive_impact_draws_gas = recursive_impact_draws
sign_restricted_impact_draws_gas = sign_restricted_impact_draws
gas_impulse_responses = impulse_responses
gas_fevd = forecast_error_variance_decomposition
gas_historical_decomposition = historical_decomposition


__all__ = [
    "BVARSVOPriorConfig",
    "SamplerConfig",
    "load_energy_panel",
    "prepare_bvar_panel",
    "fit_var_ols_from_prepared",
    "make_bvar_svo_prior",
    "draw_sv_block",
    "gibbs_bvar_sv_outlier",
    "run_energy_bvar",
    "build_run_metadata",
    "hash_model_data",
    "hash_run_config",
    "coefficient_posterior_table",
    "coefficient_posterior_tables",
    "mcmc_diagnostics",
    "stationarity_tests",
    "stability_diagnostics",
    "posterior_outlier_probability",
    "posterior_volatility_summary",
    "standardized_structural_residuals",
    "residual_diagnostic_table",
    "forecast_bvar_sv_outlier",
    "forecast_horizon_table",
    "nowcast_summary_table",
    "path_horizon_table",
    "posterior_predictive_statistics",
    "chain_acf",
    "effective_sample_size",
    "var_companion",
    "spectral_radius",
    "var_is_stable",
    "build_time_varying_covariances",
    "recursive_impact_draws",
    "sign_restricted_impact_draws",
    "impulse_responses",
    "forecast_error_variance_decomposition",
    "historical_decomposition",
]
