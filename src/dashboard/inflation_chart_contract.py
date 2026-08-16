"""Shared display/statistic contracts for Energy and Headline charts.

This module deliberately contains no model logic.  It centralises two display
contracts that were previously reimplemented inconsistently across dashboards:

1. posterior statistic selection: ``mean`` is column ``value`` and ``median``
   is column ``q50``; no heuristic fallback is allowed;
2. short monthly horizons: bars use categorical ``+1m/+2m/...`` labels rather
   than continuous datetime axes that can render misleading day ticks.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

STATISTIC_COLUMNS = {"mean": "value", "median": "q50"}


class InflationChartContractError(ValueError):
    pass


def statistic_column(frame: pd.DataFrame, statistic: str) -> str:
    """Return the explicit display column for a posterior statistic.

    No fallback is permitted. If ``mean`` is requested, the display artefact
    must actually carry posterior means in ``value``.
    """
    key = str(statistic).lower().strip()
    if key not in STATISTIC_COLUMNS:
        raise InflationChartContractError(
            f"Unknown statistic {statistic!r}; choose 'mean' or 'median'."
        )
    column = STATISTIC_COLUMNS[key]
    if column not in frame.columns:
        raise InflationChartContractError(
            f"Statistic {key!r} requires display column {column!r}."
        )
    values = pd.to_numeric(frame[column], errors="coerce")
    if len(frame) and not values.notna().all():
        bad = int(values.isna().sum())
        raise InflationChartContractError(
            f"Statistic {key!r} requires finite {column!r} values; {bad} rows are missing."
        )
    return column


def short_horizon_labels(dates: Sequence) -> list[str]:
    """Return deterministic categorical labels +1m, +2m, ... for monthly paths."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if len(idx) == 0:
        return []
    if idx.hasnans:
        raise InflationChartContractError("Short-horizon dates contain NaT.")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise InflationChartContractError(
            "Short-horizon dates must be strictly increasing and unique."
        )
    expected = pd.date_range(idx[0], periods=len(idx), freq="MS")
    if not idx.equals(expected):
        raise InflationChartContractError(
            "Short-horizon dates must be contiguous monthly-start observations."
        )
    return [f"+{i}m" for i in range(1, len(idx) + 1)]


def displayed_mean_additivity_error(
    frame: pd.DataFrame,
    *,
    components: Sequence[str],
    headline_series: str,
    contribution_metric: str,
    headline_metric: str,
    horizon: int,
) -> float:
    """Max displayed mean gap between component contributions and aggregate.

    This tests the *displayed* objects, not draw-wise identities that are true
    by construction. It therefore catches mean/median mixing in chart builders.
    """
    if frame.empty:
        return float("nan")
    contrib = frame.loc[
        frame["record_type"].astype(str).eq("contribution")
        & frame["metric"].astype(str).eq(contribution_metric)
        & frame["series"].astype(str).isin(list(components))
        & frame["is_future"].fillna(False).astype(bool)
    ].copy()
    headline = frame.loc[
        frame["record_type"].astype(str).eq("fan")
        & frame["metric"].astype(str).eq(headline_metric)
        & frame["series"].astype(str).eq(headline_series)
        & frame["is_future"].fillna(False).astype(bool)
    ].copy()
    if contrib.empty or headline.empty:
        return float("nan")
    dates = sorted(pd.DatetimeIndex(contrib["date"].dropna().unique()))[: int(horizon)]
    contrib = contrib.loc[contrib["date"].isin(dates)]
    headline = headline.loc[headline["date"].isin(dates)].sort_values("date")
    ccol = statistic_column(contrib, "mean")
    hcol = statistic_column(headline, "mean")
    sums = contrib.groupby("date", sort=True)[ccol].sum()
    head = headline.set_index("date")[hcol]
    common = sums.index.intersection(head.index)
    if len(common) != len(dates):
        return float("inf")
    return float(np.max(np.abs(sums.loc[common] - head.loc[common])))


def apply_short_horizon_axis(fig: go.Figure, labels: Sequence[str]) -> go.Figure:
    """Force a categorical axis with deterministic short-horizon ordering."""
    labels = [str(x) for x in labels]
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=labels,
        tickmode="array",
        tickvals=labels,
        ticktext=labels,
        title_text=None,
    )
    return fig


__all__ = [
    "STATISTIC_COLUMNS",
    "InflationChartContractError",
    "statistic_column",
    "short_horizon_labels",
    "apply_short_horizon_axis",
    "displayed_mean_additivity_error",
]
