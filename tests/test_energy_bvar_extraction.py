"""Regression test for the stage-5 energy BVAR extraction.

Run from the project root after keeping the previous implementation beside the
new module:

    python test_energy_bvar_extraction.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NEW_MODULE = ROOT / "energy_bvar_model.py"
OLD_CANDIDATES = (
    ROOT / "energy_bvar_sv_outlier_model_function_v7.py",
    ROOT / "energy_bvar_sv_outlier_model_function_v7(1).py",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_levels() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    observations = 60
    innovations = rng.normal(size=(observations, 2))
    changes = np.zeros((observations, 2))
    B1 = np.array([[0.35, 0.10], [-0.05, 0.25]])
    B2 = np.array([[0.05, 0.00], [0.02, 0.04]])
    for t in range(2, observations):
        changes[t] = B1 @ changes[t - 1] + B2 @ changes[t - 2] + innovations[t]
    levels = 100.0 + np.cumsum(changes, axis=0)
    return pd.DataFrame(
        levels,
        index=pd.date_range("2010-01-01", periods=observations, freq="MS"),
        columns=["x", "y"],
    )


def main() -> None:
    old_path = next((path for path in OLD_CANDIDATES if path.exists()), None)
    if old_path is None:
        raise FileNotFoundError(
            "Keep the previous v7 module beside this test for the one-off "
            "mechanical-extraction comparison."
        )
    if not NEW_MODULE.exists():
        raise FileNotFoundError(NEW_MODULE)

    old = load_module("energy_bvar_old", old_path)
    new = load_module("energy_bvar_new", NEW_MODULE)
    levels = synthetic_levels()

    prior_kwargs = {
        "lambda1": 0.10,
        "lambda2": 0.50,
        "lambda3": 1.0,
        "lambda4": 10.0,
        "a_prior_var": 10.0,
        "phi_prior_mean": 0.02,
        "phi_prior_df": 10.0,
        "h0_var": 4.0,
        "outlier_mean_frequency": 1.0 / 48.0,
        "outlier_prior_observations": 120.0,
        "outlier_grid_min": 2.0,
        "outlier_grid_max": 20.0,
        "outlier_grid_step": 1.0,
        "ksc_offset_scale": 1e-6,
    }
    sampler_kwargs = {
        "reps": 14,
        "burn": 7,
        "thin": 1,
        "seed": 42,
        "max_stability_tries": 1_000,
        "progress_every": 0,
    }

    old_result = old.gibbs_bvar_sv_outlier(
        levels,
        p=2,
        variables=["x", "y"],
        prior_config=old.BVARSVOPriorConfig(**prior_kwargs),
        sampler_config=old.SamplerConfig(**sampler_kwargs),
    )
    new_result = new.gibbs_bvar_sv_outlier(
        levels,
        p=2,
        variables=["x", "y"],
        prior_config=new.BVARSVOPriorConfig(**prior_kwargs),
        sampler_config=new.SamplerConfig(**sampler_kwargs),
    )

    keys = (
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
    )
    for key in keys:
        np.testing.assert_array_equal(old_result[key], new_result[key])
        print(f"PASS {key}: {np.asarray(new_result[key]).shape}")

    print("Exact small-chain equivalence passed.")


if __name__ == "__main__":
    main()
