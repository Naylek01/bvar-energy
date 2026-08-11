"""Read-only diagnostic helpers for the Energy BVAR dashboard.

The estimation engine already persists ``diagnostics.parquet`` (or CSV fallback)
for every component run.  This module turns that persisted contract into compact
first-class dashboard objects without re-running the sampler or opening the full
posterior draw archive.

The functions deliberately avoid imposing project-specific pass/fail thresholds
on ESS or MCSE.  Stability has one mathematical boundary, spectral radius < 1;
all other diagnostics are displayed as measurements for comparison across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DIAGNOSTICS_DASHBOARD_CONTRACT_VERSION = "energy-estimation-diagnostics-v1"


def _read_table(directory: Path, stem: str) -> pd.DataFrame:
    parquet = directory / f"{stem}.parquet"
    csv = directory / f"{stem}.csv"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    if csv.is_file():
        frame = pd.read_csv(csv)
        if len(frame.columns) and str(frame.columns[0]).startswith("Unnamed"):
            frame = frame.drop(columns=[frame.columns[0]])
        return frame
    raise FileNotFoundError(
        f"Neither {parquet.name} nor {csv.name} exists in {directory}."
    )


def load_component_diagnostics(
    run_directory: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    directory = Path(run_directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = _read_table(directory, "diagnostics")
    required = {"diagnostic_group", "parameter"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Diagnostics artifact is missing columns {sorted(missing)} in {directory}."
        )
    frame = frame.copy()
    frame["diagnostic_group"] = frame["diagnostic_group"].astype(str)
    frame["parameter"] = frame["parameter"].astype(str)
    for column in (
        "posterior_mean",
        "posterior_sd",
        "ESS",
        "MCSE_mean",
        "MCSE_over_sd",
        "value",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return metadata, frame


def _stability_values(frame: pd.DataFrame) -> dict[str, float]:
    block = frame.loc[frame["diagnostic_group"].eq("stability")].copy()
    if block.empty or "value" not in block.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in block.iterrows():
        value = row.get("value")
        if pd.notna(value):
            out[str(row["parameter"])] = float(value)
    return out


def component_diagnostic_summary(
    metadata: Mapping[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    mcmc = frame.loc[frame["diagnostic_group"].eq("mcmc")].copy()
    phi = mcmc.loc[mcmc["parameter"].str.startswith("phi[")].copy()
    ess_phi = pd.to_numeric(phi.get("ESS"), errors="coerce").dropna()
    mcse_ratio = pd.to_numeric(mcmc.get("MCSE_over_sd"), errors="coerce").dropna()
    stability = _stability_values(frame)

    radius_median = stability.get("median_spectral_radius")
    radius_q95 = stability.get("q95_spectral_radius")
    radius_max = stability.get("maximum_spectral_radius")
    retained_unstable = stability.get("retained_unstable_share")

    if radius_max is None:
        stability_state = "unavailable"
    elif radius_max < 1.0 and (retained_unstable is None or retained_unstable == 0.0):
        stability_state = "stable"
    else:
        stability_state = "unstable"

    exog_names = metadata.get("exog_names") or []
    if isinstance(exog_names, str):
        exog_names = [exog_names]

    return {
        "contract_version": DIAGNOSTICS_DASHBOARD_CONTRACT_VERSION,
        "model_id": metadata.get("model_id"),
        "vintage": metadata.get("vintage"),
        "run_id": metadata.get("run_id"),
        "frequency": metadata.get("frequency"),
        "missing_data_method": metadata.get("missing_data_method"),
        "exog_names": [str(name) for name in exog_names],
        "n_exog": int(len(exog_names)),
        "min_phi_ess": None if ess_phi.empty else float(ess_phi.min()),
        "median_phi_ess": None if ess_phi.empty else float(ess_phi.median()),
        "max_mcse_over_sd": None if mcse_ratio.empty else float(mcse_ratio.max()),
        "median_spectral_radius": radius_median,
        "q95_spectral_radius": radius_q95,
        "maximum_spectral_radius": radius_max,
        "retained_unstable_share": retained_unstable,
        "B_total_proposals": stability.get("B_total_proposals"),
        "B_unstable_proposals": stability.get("B_unstable_proposals"),
        "B_instability_rejection_rate": stability.get("B_instability_rejection_rate"),
        "retained_draws": stability.get("retained_draws"),
        "stability_state": stability_state,
    }


def phi_ess_table(frame: pd.DataFrame) -> pd.DataFrame:
    block = frame.loc[
        frame["diagnostic_group"].eq("mcmc")
        & frame["parameter"].str.startswith("phi[")
    ].copy()
    if block.empty:
        return pd.DataFrame(columns=["Parameter", "ESS", "MCSE / posterior SD"])
    out = pd.DataFrame(
        {
            "Parameter": block["parameter"].astype(str),
            "ESS": pd.to_numeric(block.get("ESS"), errors="coerce"),
            "MCSE / posterior SD": pd.to_numeric(
                block.get("MCSE_over_sd"), errors="coerce"
            ),
        }
    )
    return out.sort_values("ESS", na_position="last").reset_index(drop=True)


def key_mcmc_table(frame: pd.DataFrame) -> pd.DataFrame:
    block = frame.loc[frame["diagnostic_group"].eq("mcmc")].copy()
    if block.empty:
        return pd.DataFrame(
            columns=["Parameter", "Posterior mean", "Posterior SD", "ESS", "MCSE / SD"]
        )
    out = pd.DataFrame(
        {
            "Parameter": block["parameter"].astype(str),
            "Posterior mean": pd.to_numeric(block.get("posterior_mean"), errors="coerce"),
            "Posterior SD": pd.to_numeric(block.get("posterior_sd"), errors="coerce"),
            "ESS": pd.to_numeric(block.get("ESS"), errors="coerce"),
            "MCSE / SD": pd.to_numeric(block.get("MCSE_over_sd"), errors="coerce"),
        }
    )
    return out.reset_index(drop=True)


def stability_table(frame: pd.DataFrame) -> pd.DataFrame:
    values = _stability_values(frame)
    labels = {
        "retained_draws": "Retained posterior draws",
        "median_spectral_radius": "Median spectral radius",
        "q95_spectral_radius": "95% spectral radius",
        "maximum_spectral_radius": "Maximum spectral radius",
        "retained_unstable_share": "Retained unstable share",
        "B_total_proposals": "B proposals",
        "B_unstable_proposals": "Rejected unstable B proposals",
        "B_instability_rejection_rate": "Instability rejection rate",
    }
    rows = []
    for key in labels:
        if key in values:
            rows.append({"Diagnostic": labels[key], "Value": float(values[key])})
    return pd.DataFrame(rows)


def empty_diagnostic_figure(message: str = "No diagnostics available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font={"size": 13, "color": "#64748b"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=330,
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def phi_ess_figure(frame: pd.DataFrame, *, uirevision: str = "phi-ess") -> go.Figure:
    table = phi_ess_table(frame)
    if table.empty:
        return empty_diagnostic_figure("No φ ESS diagnostics in this run")
    table = table.sort_values("ESS", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=table["ESS"],
            y=table["Parameter"],
            orientation="h",
            name="ESS",
            hovertemplate="%{y}<br>ESS %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "Stochastic-volatility mixing — ESS(φ)", "x": 0.01, "xanchor": "left"},
        xaxis_title="Effective sample size",
        yaxis_title="",
        height=max(330, 70 + 42 * len(table)),
        margin={"l": 120, "r": 24, "t": 72, "b": 58},
        paper_bgcolor="white",
        plot_bgcolor="white",
        uirevision=uirevision,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def _aggregate_diagnostics_from_display(directory: Path) -> dict[str, Any]:
    path = directory / "display_v1.parquet"
    if not path.is_file():
        return {}
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return {}
    required = {"record_type", "metric", "series", "value"}
    if not required.issubset(frame.columns):
        return {}
    block = frame.loc[frame["record_type"].astype(str).eq("diagnostic")].copy()
    out: dict[str, Any] = {}
    if block.empty:
        return out
    for _, row in block.iterrows():
        value = pd.to_numeric(pd.Series([row.get("value")]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        key = str(row.get("metric"))
        series = str(row.get("series"))
        if key == "weekly_rejection_rate":
            out.setdefault("weekly_rejection_rates", {})[series] = float(value)
        elif key == "historical_reconstruction_max_error":
            out.setdefault("historical_validation_max_errors", {})[series] = float(value)
        else:
            out[key] = float(value)
    return out


def load_aggregate_validation(aggregate_directory: str | Path) -> dict[str, Any]:
    directory = Path(aggregate_directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    display = _aggregate_diagnostics_from_display(directory)

    weekly = dict(metadata.get("weekly_rejection_rates", {}) or {})
    weekly.update(dict(display.get("weekly_rejection_rates", {}) or {}))
    hist = dict(metadata.get("historical_validation_max_errors", {}) or {})
    hist.update(dict(display.get("historical_validation_max_errors", {}) or {}))

    additivity = display.get(
        "drawwise_contribution_additivity_error",
        metadata.get("maximum_drawwise_contribution_additivity_error"),
    )
    history = display.get(
        "six_model_history_max_error",
        metadata.get("six_model_history_max_error"),
    )
    annual = display.get(
        "six_model_history_annual_reanchored_max_error",
        metadata.get("six_model_history_annual_reanchored_max_error"),
    )

    finite_weekly = [float(v) for v in weekly.values() if pd.notna(v)]
    finite_hist = [float(v) for v in hist.values() if pd.notna(v)]

    return {
        "aggregate_run_id": metadata.get("aggregate_run_id", directory.name),
        "vintage": metadata.get("vintage", directory.parent.name),
        "n_aggregate_draws_effective": metadata.get("n_aggregate_draws_effective"),
        "drawwise_contribution_additivity_error": None if additivity is None else float(additivity),
        "six_model_history_max_error": (
            None if history is None else float(history)
        ),
        "six_model_history_annual_reanchored_max_error": (
            None if annual is None else float(annual)
        ),
        "max_weekly_rejection_rate": (
            None if not finite_weekly else float(max(finite_weekly))
        ),
        "max_historical_component_reconstruction_error": (
            None if not finite_hist else float(max(finite_hist))
        ),
        "weekly_rejection_rates": weekly,
        "historical_validation_max_errors": hist,
    }


__all__ = [
    "DIAGNOSTICS_DASHBOARD_CONTRACT_VERSION",
    "load_component_diagnostics",
    "component_diagnostic_summary",
    "phi_ess_table",
    "key_mcmc_table",
    "stability_table",
    "phi_ess_figure",
    "empty_diagnostic_figure",
    "load_aggregate_validation",
]
