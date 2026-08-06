"""Small integration test for heat-energy and solid-fuels adapters."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
MODEL_DIR = PROJECT_ROOT / "src" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from energy_bvar_model import (  # noqa: E402
    BVARSVOPriorConfig,
    SamplerConfig,
    forecast_bvar_sv_outlier,
    historical_decomposition,
    run_energy_bvar,
)
from energy_bvar_monthly_hicp import (  # noqa: E402
    forecast_hicp_component,
    monthly_hicp_component_spec,
)


def synthetic_levels(target: str) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2000-01-01", periods=110, freq="MS")
    innovations = rng.normal(size=(len(dates), 3))
    loading = np.array(
        [
            [1.30, 0.00, 0.00],
            [0.25, 0.60, 0.00],
            [0.10, 0.12, 0.35],
        ]
    )
    changes = innovations @ loading.T
    levels = np.cumsum(changes, axis=0) + np.array([45.0, 100.0, 95.0])
    frame = pd.DataFrame(
        levels,
        index=dates,
        columns=["natural_gas_wholesale", "ppi_energy", target],
    )
    frame.loc[dates[-2]:, target] = np.nan
    return frame


def check_component(component: str) -> None:
    spec = monthly_hicp_component_spec(component)
    levels = synthetic_levels(spec["target"])
    result = run_energy_bvar(
        levels=levels,
        model_id=component,
        vintage="synthetic",
        p=2,
        variables=spec["variables"],
        prior_config=BVARSVOPriorConfig(),
        sampler_config=SamplerConfig(
            reps=24,
            burn=12,
            thin=1,
            seed=42,
            max_stability_tries=1_000,
            progress_every=0,
        ),
        code_version="integration-test",
    )
    forecast = forecast_bvar_sv_outlier(
        result,
        H=3,
        n_draws=6,
        seed=2026,
    )
    hicp = forecast_hicp_component(
        forecast,
        result,
        component=component,
    )
    decomposition = historical_decomposition(
        result,
        identification="recursive",
        reference_date=result["prep"]["balanced_end"],
    )

    assert result["variables"] == spec["variables"]
    assert hicp["future_hicp_level_paths"].shape == (6, 3)
    assert hicp["future_hicp_yoy_paths"].shape == (6, 3)
    assert np.isfinite(hicp["future_hicp_level_paths"]).all()
    assert decomposition["max_reconstruction_error"] < 1e-10
    print(f"PASS {component}")


def main() -> None:
    check_component("heat_energy")
    check_component("solid_fuels")


if __name__ == "__main__":
    main()
