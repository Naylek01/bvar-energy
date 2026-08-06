"""Adapters for the monthly heat-energy and solid-fuels BVARs.

Both models use the generic monthly BVAR-SV-outlier engine with 12 lags and
three endogenous level series transformed internally into absolute changes:

* wholesale natural-gas prices;
* PPI Energy;
* the relevant HICP component.

No deterministic seasonal block and no tax re-attribution are required for
these two components because the model target is the HICP index itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from energy_bvar_model import load_energy_panel


MONTHLY_HICP_COMPONENTS = {
    "heat_energy": {
        "dataset_file": "heat_energy_monthly.csv",
        "target": "hicp_heat_energy",
        "label": "heat energy",
        "variables": (
            "natural_gas_wholesale",
            "ppi_energy",
            "hicp_heat_energy",
        ),
        "units": {
            "natural_gas_wholesale": "EUR/MWh",
            "ppi_energy": "Index",
            "hicp_heat_energy": "HICP index",
        },
        "hicp_ticker": "H023HW55@EUDATA",
    },
    "solid_fuels": {
        "dataset_file": "solid_fuels_monthly.csv",
        "target": "hicp_solid_fuels",
        "label": "solid fuels",
        "variables": (
            "natural_gas_wholesale",
            "ppi_energy",
            "hicp_solid_fuels",
        ),
        "units": {
            "natural_gas_wholesale": "EUR/MWh",
            "ppi_energy": "Index",
            "hicp_solid_fuels": "HICP index",
        },
        "hicp_ticker": "H023HW54@EUDATA",
    },
}


MONTHLY_HICP_COLUMN_ALIASES = {
    "natural_gas_eur_mwh": "natural_gas_wholesale",
    "natural_gas": "natural_gas_wholesale",
    "hicp_heat_cooling_energy": "hicp_heat_energy",
}


def monthly_hicp_component_spec(component: str) -> dict:
    """Return a defensive copy of one supported component specification."""
    try:
        spec = MONTHLY_HICP_COMPONENTS[component]
    except KeyError as exc:
        raise ValueError(
            f"Unknown component {component!r}. "
            f"Choose one of {tuple(MONTHLY_HICP_COMPONENTS)}."
        ) from exc
    return {
        **spec,
        "variables": list(spec["variables"]),
        "units": dict(spec["units"]),
    }


def load_monthly_hicp_component_panel(
    path: str | Path,
    component: str,
    variables: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load heat-energy or solid-fuels levels without filling missing data."""
    spec = monthly_hicp_component_spec(component)
    selected = spec["variables"] if variables is None else list(variables)
    return load_energy_panel(
        path,
        variables=selected,
        aliases=MONTHLY_HICP_COLUMN_ALIASES,
        frequency="monthly",
    )


def forecast_hicp_component(
    forecast: Mapping,
    result: Mapping,
    *,
    component: str,
) -> dict:
    """Extract HICP level paths and construct year-on-year inflation paths.

    The generic forecast already integrates absolute changes back to levels.
    This helper selects the component target and appends the observed pre-origin
    history required to compute 12-month inflation for every posterior path.
    """
    spec = monthly_hicp_component_spec(component)
    target = spec["target"]
    variables = list(forecast["variables"])
    if target not in variables:
        raise ValueError(
            f"Target {target!r} is absent from forecast variables {variables}."
        )

    target_index = variables.index(target)
    dates = pd.DatetimeIndex(forecast["path_dates"])
    level_paths = np.asarray(forecast["level_paths"], dtype=float)[:, :, target_index]
    if level_paths.ndim != 2 or level_paths.shape[1] != len(dates):
        raise ValueError("Forecast level paths and dates have incompatible dimensions.")

    actual_hicp = (
        result["prep"]["levels"][target]
        .astype(float)
        .sort_index()
        .rename(target)
    )
    balanced_end = pd.Timestamp(result["prep"]["balanced_end"])
    history = actual_hicp.loc[actual_hicp.index <= balanced_end].dropna()
    if history.empty:
        raise ValueError("No observed HICP history exists before the forecast path.")

    full_index = pd.date_range(
        min(history.index.min(), dates.min()),
        max(history.index.max(), dates.max()),
        freq="MS",
    )
    yoy_paths = np.full_like(level_paths, np.nan, dtype=float)
    for draw in range(len(level_paths)):
        series = history.reindex(full_index)
        series.loc[dates] = level_paths[draw]
        yoy = 100.0 * (series / series.shift(12) - 1.0)
        yoy_paths[draw] = yoy.reindex(dates).to_numpy(dtype=float)

    tail_length = int(forecast.get("tail_length", 0))
    return {
        "hicp_level_paths": level_paths,
        "hicp_yoy_paths": yoy_paths,
        "path_dates": dates,
        "future_dates": pd.DatetimeIndex(forecast["future_dates"]),
        "tail_length": tail_length,
        "future_hicp_level_paths": level_paths[:, tail_length:],
        "future_hicp_yoy_paths": yoy_paths[:, tail_length:],
        "actual_hicp": actual_hicp,
        "target_variable": target,
        "component": component,
        "component_label": spec["label"],
    }


def monthly_hicp_series_construction_table(
    dataset_path: str | Path,
    component: str,
) -> pd.DataFrame:
    """Describe the three processed level series used by one component model."""
    spec = monthly_hicp_component_spec(component)
    dataset_path = Path(dataset_path)
    manifest_path = dataset_path.parent / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}

    natural = manifest.get("construction", {}).get("natural_gas", {})
    natural_method = natural.get(
        "method",
        "Bloomberg TTF monthly mean in EUR/MWh; earlier observations are "
        "chain-linked to the World Bank Natural gas, Europe proxy.",
    )
    natural_note = natural.get(
        "proxy_note",
        "The World Bank series is the project proxy for the historical "
        "European border-gas series used in the paper.",
    )
    if natural.get("anchor_date"):
        natural_note += f" TTF anchor month: {natural['anchor_date']}."

    target = spec["target"]
    return pd.DataFrame(
        {
            "series_type": [
                "chain-linked wholesale price",
                "producer-price index",
                "consumer-price index",
            ],
            "model_unit": ["EUR/MWh", "Index", "HICP index"],
            "source_inputs": [
                "Bloomberg TTF; World Bank Natural gas, Europe; EUR/USD",
                "Haver PPI Energy (H025PP@G10)",
                f"Haver HICP {spec['label']} ({spec['hicp_ticker']})",
            ],
            "construction": [
                natural_method,
                "Monthly Haver level; no source-stage transformation.",
                "Monthly Haver level; no tax reconstruction is applied.",
            ],
            "note": [
                natural_note,
                "The notebook applies an absolute monthly difference before estimation.",
                "The notebook forecasts the HICP level directly and derives YoY inflation ex post.",
            ],
        },
        index=["natural_gas_wholesale", "ppi_energy", target],
    )


__all__ = [
    "MONTHLY_HICP_COMPONENTS",
    "monthly_hicp_component_spec",
    "load_monthly_hicp_component_panel",
    "forecast_hicp_component",
    "monthly_hicp_series_construction_table",
]
