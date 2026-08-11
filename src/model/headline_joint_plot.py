"""Plot helpers for the joint Headline-HICP BVAR.

The statistical/model transformations remain in ``headline_joint_bvar.py``.
This file only formats reusable figures for notebooks and the future dashboard.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from headline_joint_bvar import NATIVE_VARIABLES, native_to_state_levels


def plot_joint_native_levels(
    native_levels: pd.DataFrame,
    *,
    last_obs: int | None = None,
) -> dict[str, plt.Figure]:
    """One figure per native HICP component."""
    frame = native_levels[NATIVE_VARIABLES]
    if last_obs is not None:
        frame = frame.iloc[-int(last_obs):]
    figures = {}
    for name in NATIVE_VARIABLES:
        fig, ax = plt.subplots(figsize=(10.5, 3.8))
        ax.plot(frame.index, frame[name].to_numpy())
        ax.set_title(f"{name} — native HICP index")
        ax.set_ylabel("HICP index")
        ax.set_xlabel("")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        figures[name] = fig
    return figures


def plot_joint_model_changes(
    native_levels: pd.DataFrame,
    *,
    last_obs: int | None = None,
) -> dict[str, plt.Figure]:
    """One figure per model variable: 100 × Δlog(HICP)."""
    state = native_to_state_levels(native_levels)
    changes = 100.0 * state.diff()
    if last_obs is not None:
        changes = changes.iloc[-int(last_obs):]
    figures = {}
    for native, state_name in zip(NATIVE_VARIABLES, state.columns):
        fig, ax = plt.subplots(figsize=(10.5, 3.8))
        ax.plot(changes.index, changes[state_name].to_numpy())
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(f"{native} — 100 × Δlog(HICP)")
        ax.set_ylabel("% m/m log approximation")
        ax.set_xlabel("")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        figures[native] = fig
    return figures


def plot_joint_component_forecast_fans(
    forecast: Mapping,
    *,
    kind: str = "yoy",
    history: pd.DataFrame | None = None,
    last_obs: int = 60,
) -> dict[str, plt.Figure]:
    """One fan chart per Energy/Food/NEIG/Services component."""
    dates = pd.DatetimeIndex(forecast["path_dates"], name="date")
    variables = list(forecast["native_variables"])
    if kind == "level":
        paths = np.asarray(forecast["native_level_paths"], dtype=float)
        ylabel = "HICP index"
    elif kind == "yoy":
        paths = np.asarray(forecast["component_yoy_paths"], dtype=float)
        ylabel = "% y/y"
    else:
        raise ValueError("kind must be 'level' or 'yoy'.")

    figures = {}
    for j, name in enumerate(variables):
        q05, q16, q50, q84, q95 = np.nanpercentile(
            paths[:, :, j],
            [5, 16, 50, 84, 95],
            axis=0,
        )
        fig, ax = plt.subplots(figsize=(10.5, 4.2))
        if history is not None and name in history.columns:
            s = history[name].copy()
            if kind == "yoy":
                s = 100.0 * (s / s.shift(12) - 1.0)
            s = s.dropna().iloc[-int(last_obs):]
            ax.plot(s.index, s.to_numpy(), label="Observed")
        ax.fill_between(dates, q05, q95, alpha=0.15, label="90%")
        ax.fill_between(dates, q16, q84, alpha=0.25, label="68%")
        ax.plot(dates, q50, label="Posterior median")
        if len(forecast["future_dates"]):
            ax.axvline(pd.Timestamp(forecast["future_dates"][0]), linestyle="--", linewidth=0.9)
        if kind == "yoy":
            ax.axhline(0.0, linewidth=0.8)
        ax.set_title(f"{name} — joint BVAR forecast")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        figures[name] = fig
    return figures


def plot_headline_aggregate_forecast(
    aggregate_forecast: Mapping,
    official_total: pd.Series,
    *,
    kind: str = "yoy",
    last_obs: int = 72,
) -> plt.Figure:
    """Headline fan chart from draw-by-draw chained component aggregation."""
    dates = pd.DatetimeIndex(aggregate_forecast["path_dates"], name="date")
    if kind == "level":
        paths = np.asarray(aggregate_forecast["headline_level_paths"], dtype=float)
        observed = official_total.copy()
        ylabel = "HICP index"
    elif kind == "yoy":
        paths = np.asarray(aggregate_forecast["headline_yoy_paths"], dtype=float)
        observed = 100.0 * (official_total / official_total.shift(12) - 1.0)
        ylabel = "% y/y"
    else:
        raise ValueError("kind must be 'level' or 'yoy'.")

    q05, q16, q50, q84, q95 = np.nanpercentile(
        paths,
        [5, 16, 50, 84, 95],
        axis=0,
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    observed = observed.dropna().iloc[-int(last_obs):]
    ax.plot(observed.index, observed.to_numpy(), label="Official HICP Total")
    ax.fill_between(dates, q05, q95, alpha=0.15, label="90%")
    ax.fill_between(dates, q16, q84, alpha=0.25, label="68%")
    ax.plot(dates, q50, label="Bottom-up posterior median")
    if len(aggregate_forecast["future_dates"]):
        ax.axvline(
            pd.Timestamp(aggregate_forecast["future_dates"][0]),
            linestyle="--",
            linewidth=0.9,
        )
    if kind == "yoy":
        ax.axhline(0.0, linewidth=0.8)
    ax.set_title("Headline HICP — joint BVAR bottom-up forecast")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_headline_historical_reconstruction(
    reconstructed: pd.Series,
    official_total: pd.Series,
) -> plt.Figure:
    """Compare bottom-up historical index reconstruction with official total."""
    frame = pd.concat(
        [
            reconstructed.rename("Bottom-up reconstruction"),
            official_total.rename("Official HICP Total"),
        ],
        axis=1,
    ).dropna()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(frame.index, frame["Official HICP Total"].to_numpy(), label="Official")
    ax.plot(
        frame.index,
        frame["Bottom-up reconstruction"].to_numpy(),
        label="Reconstructed",
        linestyle="--",
    )
    ax.set_title("Historical Headline HICP reconstruction")
    ax.set_ylabel("HICP index")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


__all__ = [
    "plot_joint_native_levels",
    "plot_joint_model_changes",
    "plot_joint_component_forecast_fans",
    "plot_headline_aggregate_forecast",
    "plot_headline_historical_reconstruction",
]
