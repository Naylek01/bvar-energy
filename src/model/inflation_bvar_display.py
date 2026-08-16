"""Display facade shared by Energy and Headline dashboard domains."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import energy_bvar_display as _energy
from headline_bvar_display import (
    DISPLAY_SCHEMA_VERSION as HEADLINE_DISPLAY_SCHEMA_VERSION,
    build_headline_display,
    load_headline_display,
)

DISPLAY_FILENAME = _energy.DISPLAY_FILENAME
DISPLAY_SCHEMA_VERSION = _energy.DISPLAY_SCHEMA_VERSION
DISPLAY_QUANTILES = _energy.DISPLAY_QUANTILES
DisplayError = _energy.DisplayError
build_aggregate_display = _energy.build_aggregate_display


def _is_headline_run_directory(run_directory: Path) -> bool:
    metadata = run_directory / "metadata.json"
    if metadata.is_file():
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            return str(payload.get("model_id")) == "headline_joint"
        except Exception:
            pass
    return (run_directory / "headline_native_history.csv").is_file()


def _is_headline_display_path(path: Path) -> bool:
    directory = path if path.is_dir() else path.parent
    return (
        (directory / "headline_metadata.json").is_file()
        or (directory / "headline_draws.npz").is_file()
    )


def build_component_display(
    run_directory: str | Path,
    *,
    project_root: str | Path | None = None,
    forecast_name: str = "unconditional",
    destination: str | Path | None = None,
    overwrite: bool = True,
):
    """Dispatch display materialisation without changing the Energy API."""
    run_directory = Path(run_directory)
    if _is_headline_run_directory(run_directory):
        expected = (
            run_directory / "forecasts" / str(forecast_name) / DISPLAY_FILENAME
        )
        if destination is not None and Path(destination).resolve() != expected.resolve():
            raise ValueError(
                "Headline display destination is fixed by the run contract; "
                f"expected {expected}."
            )
        return build_headline_display(
            run_directory,
            forecast_name=forecast_name,
            overwrite=overwrite,
        )
    return _energy.build_component_display(
        run_directory,
        project_root=project_root,
        forecast_name=forecast_name,
        destination=destination,
        overwrite=overwrite,
    )


def build_display_artifact(
    directory: str | Path,
    *,
    project_root: str | Path | None = None,
    forecast_name: str = "unconditional",
    destination: str | Path | None = None,
    overwrite: bool = True,
):
    directory = Path(directory)
    if (directory / "aggregate_draws.npz").is_file():
        return _energy.build_aggregate_display(
            directory,
            project_root=project_root,
            destination=destination,
            overwrite=overwrite,
        )
    return build_component_display(
        directory,
        project_root=project_root,
        forecast_name=forecast_name,
        destination=destination,
        overwrite=overwrite,
    )


def load_display_artifact(path_or_directory: str | Path) -> pd.DataFrame:
    path = Path(path_or_directory)
    if path.is_dir():
        path = path / DISPLAY_FILENAME
    if _is_headline_display_path(path):
        return load_headline_display(path)
    return _energy.load_display_artifact(path)


def display_metadata(frame: pd.DataFrame) -> dict[str, str | float]:
    versions = (
        set(frame["display_schema_version"].dropna().astype(str).unique())
        if "display_schema_version" in frame
        else set()
    )
    if versions != {HEADLINE_DISPLAY_SCHEMA_VERSION}:
        return _energy.display_metadata(frame)

    meta = frame.loc[frame["record_type"].astype(str).isin(["metadata", "meta"])]
    out: dict[str, str | float] = {}
    for _, row in meta.iterrows():
        key = str(row.get("key"))
        text = row.get("text_value")
        numeric = row.get("extra_value")
        if pd.notna(text):
            out[key] = str(text)
        elif pd.notna(numeric):
            out[key] = float(numeric)
        else:
            out[key] = None

    identity = frame.loc[
        frame["model_id"].notna()
        & frame["vintage"].notna()
        & frame["run_id"].notna()
    ]
    if not identity.empty:
        first = identity.iloc[0]
        out.setdefault("model_id", str(first["model_id"]))
        out.setdefault("vintage", str(first["vintage"]))
        out.setdefault("run_id", str(first["run_id"]))

    out.setdefault("model_label", "Headline HICP")
    out.setdefault("target", "hicp_total")
    out.setdefault("frequency", "monthly")
    out.setdefault("scope", "headline")
    out.setdefault("max_published_horizon_months", 12)
    return out


__all__ = [
    "DISPLAY_SCHEMA_VERSION",
    "DISPLAY_FILENAME",
    "DISPLAY_QUANTILES",
    "DisplayError",
    "build_component_display",
    "build_aggregate_display",
    "build_display_artifact",
    "load_display_artifact",
    "display_metadata",
]
