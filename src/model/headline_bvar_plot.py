"""Plot adapters for the Headline inflation BVAR suite.

The existing ``energy_bvar_plot`` module remains the shared source for generic
posterior, diagnostic and HICP forecast figures.  This file only adds plots
whose titles/units depend on the headline transformation contract, plus a thin
wrapper around the existing dashboard-compatible HICP fan chart.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from energy_bvar_plot import plot_hicp_forecast_fan


def _axes_array(axes):
    try:
        return list(axes.ravel())
    except AttributeError:
        return [axes]


def plot_headline_native_levels(
    native_levels: pd.DataFrame,
    *,
    component_label: str,
    units: Mapping[str, str] | None = None,
    last_obs: int | None = None,
):
    """Plot native input levels without implying an Energy-specific transform."""
    data = native_levels if last_obs is None else native_levels.iloc[-int(last_obs):]
    n = data.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), sharex=True)
    for ax, column in zip(_axes_array(axes), data.columns):
        ax.plot(data.index, data[column], linewidth=1.2)
        ax.set_title(column)
        ax.set_ylabel((units or {}).get(column, "native units"))
        ax.grid(alpha=0.25)
    _axes_array(axes)[-1].set_xlabel("Date")
    fig.suptitle(f"{component_label} — native input levels")
    fig.tight_layout()
    return fig


def plot_headline_model_differences(
    differences: pd.DataFrame,
    *,
    component_label: str,
    transform_labels: Mapping[str, str] | None = None,
    last_obs: int | None = None,
):
    """Plot the transformed first differences actually entering the sampler."""
    data = differences if last_obs is None else differences.iloc[-int(last_obs):]
    n = data.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), sharex=True)
    for ax, column in zip(_axes_array(axes), data.columns):
        ax.plot(data.index, data[column], linewidth=1.0)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title((transform_labels or {}).get(column, column))
        ax.set_ylabel("model change")
        ax.grid(alpha=0.25)
    _axes_array(axes)[-1].set_xlabel("Date")
    fig.suptitle(f"{component_label} — model-space transformation")
    fig.tight_layout()
    return fig


def plot_headline_hicp_forecast_fan(
    forecast: Mapping,
    *,
    kind: str = "yoy",
    last_obs: int = 72,
):
    """Reuse the Energy dashboard HICP fan contract with the component label."""
    label = str(forecast.get("component_label", forecast.get("component", "component")))
    return plot_hicp_forecast_fan(
        forecast,
        kind=kind,
        last_obs=int(last_obs),
        component_label=label,
    )


def plot_headline_driver_assumptions(
    native_history: pd.DataFrame,
    forecast: Mapping,
    *,
    drivers: list[str] | None = None,
    last_obs: int = 48,
):
    """Plot historical driver levels and the native future paths used in a forecast."""
    paths = forecast.get("native_driver_paths")
    if not paths:
        raise ValueError("Forecast does not contain native_driver_paths.")
    future_dates = pd.DatetimeIndex(forecast["future_dates"], name="date")
    selected = list(paths) if drivers is None else list(drivers)
    missing = [name for name in selected if name not in paths]
    if missing:
        raise KeyError(f"Unknown future driver path(s): {missing}.")

    fig, axes = plt.subplots(len(selected), 1, figsize=(11, 3.0 * len(selected)), sharex=True)
    for ax, name in zip(_axes_array(axes), selected):
        history = pd.to_numeric(native_history[name], errors="coerce").dropna()
        history = history.iloc[-int(last_obs):]
        ax.plot(history.index, history.to_numpy(), label="Observed / available", linewidth=1.2)
        ax.plot(future_dates, paths[name], label="Forecast assumption", linewidth=1.2, linestyle="--")
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.legend()
    _axes_array(axes)[-1].set_xlabel("Date")
    fig.suptitle("Headline component — exogenous driver assumptions")
    fig.tight_layout()
    return fig


def plot_headline_scenario_impact(
    baseline: Mapping,
    scenario: Mapping,
    *,
    kind: str = "yoy",
):
    """Plot median scenario minus baseline HICP paths, draw-by-draw compatible."""
    import numpy as np

    if kind == "yoy":
        base = np.asarray(baseline["future_hicp_yoy_paths"], dtype=float)
        alt = np.asarray(scenario["future_hicp_yoy_paths"], dtype=float)
        ylabel = "percentage points"
        title_kind = "YoY inflation"
    elif kind == "level":
        base = np.asarray(baseline["future_hicp_level_paths"], dtype=float)
        alt = np.asarray(scenario["future_hicp_level_paths"], dtype=float)
        ylabel = "HICP index points"
        title_kind = "HICP level"
    else:
        raise ValueError("kind must be 'yoy' or 'level'.")

    if base.shape != alt.shape:
        raise ValueError("Baseline and scenario draw arrays must have identical shapes.")
    dates = pd.DatetimeIndex(baseline["future_dates"], name="date")
    if not dates.equals(pd.DatetimeIndex(scenario["future_dates"])):
        raise ValueError("Baseline and scenario future dates differ.")

    delta = alt - base
    q16, q50, q84 = np.nanpercentile(delta, [16, 50, 84], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(dates, q16, q84, alpha=0.25, label="68% interval")
    ax.plot(dates, q50, label="Median scenario impact")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.set_title(f"Scenario impact on {title_kind}: scenario − baseline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


__all__ = [
    "plot_headline_native_levels",
    "plot_headline_model_differences",
    "plot_headline_hicp_forecast_fan",
    "plot_headline_driver_assumptions",
    "plot_headline_scenario_impact",
]
