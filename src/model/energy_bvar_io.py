"""Serialization and cache helpers for energy BVAR results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from energy_bvar_model import (
    coefficient_posterior_table,
    mcmc_diagnostics,
    stability_diagnostics,
)

_HEAVY_ARRAYS = (
    "B",
    "A",
    "log_variance",
    "phi",
    "outlier_scales",
    "outlier_indicators",
    "outlier_probabilities",
    "posterior_outlier_probability_draws",
    "spectral_radius",
    "log_likelihood",
    "completed_differences_draws",
    "missing_level_draws",
    "last_companion_state_draws",
)


def _write_table(frame: pd.DataFrame, base: Path) -> Path:
    try:
        path = base.with_suffix(".parquet")
        frame.to_parquet(path)
        return path
    except (ImportError, ModuleNotFoundError):
        path = base.with_suffix(".csv")
        frame.to_csv(path)
        return path


def model_summary_frame(result: Mapping) -> pd.DataFrame:
    coefficients = coefficient_posterior_table(result).reset_index().rename(columns={"index": "parameter"})
    coefficients.insert(0, "block", "coefficient")

    phi = np.asarray(result["phi"], dtype=float)
    p_out = np.asarray(result["outlier_probabilities"], dtype=float)
    rows = []
    for j, name in enumerate(result["variables"]):
        for block, draws in (("phi", phi[:, j]), ("outlier_probability", p_out[:, j])):
            q = np.percentile(draws, [5, 16, 50, 84, 95])
            rows.append({
                "block": block,
                "parameter": name,
                "posterior_mean": float(np.mean(draws)),
                "posterior_median": float(q[2]),
                "posterior_sd": float(np.std(draws, ddof=1)),
                "q05": float(q[0]),
                "q16": float(q[1]),
                "q84": float(q[3]),
                "q95": float(q[4]),
                "ESS": np.nan,
            })
    extra = pd.DataFrame(rows)
    return pd.concat([coefficients, extra], ignore_index=True, sort=False)


def diagnostics_frame(result: Mapping) -> pd.DataFrame:
    mcmc = mcmc_diagnostics(result).reset_index().rename(columns={"index": "parameter"})
    mcmc.insert(0, "diagnostic_group", "mcmc")
    stability = stability_diagnostics(result).rename("value").reset_index().rename(columns={"index": "parameter"})
    stability.insert(0, "diagnostic_group", "stability")
    return pd.concat([mcmc, stability], ignore_index=True, sort=False)


def save_energy_bvar_result(result: Mapping, output_root: str | Path) -> dict[str, Path]:
    metadata = dict(result.get("metadata", {}))
    required = {"model_id", "vintage", "run_id"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"Result metadata is missing {sorted(missing)}. Use run_energy_bvar().")

    directory = (
        Path(output_root)
        / str(metadata["model_id"])
        / str(metadata["vintage"])
        / str(metadata["run_id"])
    )
    directory.mkdir(parents=True, exist_ok=True)

    metadata_path = directory / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    summary_path = _write_table(model_summary_frame(result), directory / "summary")
    diagnostics_path = _write_table(diagnostics_frame(result), directory / "diagnostics")
    draws_path = directory / "draws.npz"
    np.savez_compressed(
        draws_path,
        **{name: np.asarray(result[name]) for name in _HEAVY_ARRAYS if name in result},
    )
    return {
        "directory": directory,
        "metadata": metadata_path,
        "summary": summary_path,
        "diagnostics": diagnostics_path,
        "draws": draws_path,
    }


def load_energy_bvar_draws(run_directory: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    directory = Path(run_directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with np.load(directory / "draws.npz", allow_pickle=False) as archive:
        draws = {name: archive[name] for name in archive.files}
    return metadata, draws


__all__ = [
    "model_summary_frame",
    "diagnostics_frame",
    "save_energy_bvar_result",
    "load_energy_bvar_draws",
]
