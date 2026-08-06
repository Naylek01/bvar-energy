"""Run and save the monthly electricity BVAR-SV-outlier model.

Place this script in ``src/model`` beside the model modules, or run it from the
project root after adding ``src/model`` to PYTHONPATH.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate the bvar-energy project root.")


PROJECT_ROOT = find_project_root(Path.cwd())
MODEL_DIR = PROJECT_ROOT / "src" / "model"
if MODEL_DIR.exists():
    sys.path.insert(0, str(MODEL_DIR))

from energy_bvar_electricity import (  # noqa: E402
    load_electricity_panel,
    make_electricity_seasonal_dummies,
)
from energy_bvar_io import save_energy_bvar_result  # noqa: E402
from energy_bvar_model import (  # noqa: E402
    BVARSVOPriorConfig,
    SamplerConfig,
    forecast_bvar_sv_outlier,
    run_energy_bvar,
)


P = 12
SEED = 42
REFERENCE_MONTH = 1
EXOG_PRIOR_SCALE = 10.0
VARIABLES = ["natural_gas_wholesale", "electricity_pre_tax"]


def latest_electricity_dataset() -> Path:
    candidates = sorted(
        path
        for path in (PROJECT_ROOT / "data" / "processed").glob(
            "*/electricity_monthly.csv"
        )
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            "No data/processed/<vintage>/electricity_monthly.csv was found."
        )
    return candidates[-1]


def main() -> None:
    dataset = latest_electricity_dataset()
    levels = load_electricity_panel(dataset, variables=VARIABLES)
    exog = make_electricity_seasonal_dummies(
        levels.index,
        reference_month=REFERENCE_MONTH,
    )

    prior_config = BVARSVOPriorConfig(
        lambda1=0.20,
        lambda2=0.50,
        lambda3=1.00,
        lambda4=10.0,
        a_prior_var=10.0,
        phi_prior_mean=0.02,
        phi_prior_df=10.0,
        h0_var=4.0,
        outlier_mean_frequency=1.0 / 48.0,
        outlier_prior_observations=120.0,
        outlier_grid_min=2.0,
        outlier_grid_max=20.0,
        outlier_grid_step=1.0,
        ksc_offset_scale=1e-6,
    )
    sampler_config = SamplerConfig(
        reps=6_000,
        burn=3_000,
        thin=1,
        seed=SEED,
        max_stability_tries=1_000,
        progress_every=500,
    )

    result = run_energy_bvar(
        levels=levels,
        model_id="electricity",
        vintage=dataset.parent.name,
        p=P,
        variables=VARIABLES,
        exog=exog,
        exog_prior_scale=EXOG_PRIOR_SCALE,
        prior_config=prior_config,
        sampler_config=sampler_config,
        code_version="energy_bvar_model-exog-v1",
    )

    forecast_origin = result["prep"]["last_calendar_date"]
    months_to_year_end = 12 - forecast_origin.month or 12
    horizon = max(3, months_to_year_end)
    future_dates = pd.date_range(
        forecast_origin + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    future_exog = make_electricity_seasonal_dummies(
        future_dates,
        reference_month=REFERENCE_MONTH,
    )
    forecast = forecast_bvar_sv_outlier(
        result,
        H=horizon,
        future_exog=future_exog,
        n_draws=min(1_000, result["n_draws"]),
        simulate_future_outliers=True,
        seed=2026,
    )

    paths = save_energy_bvar_result(result, PROJECT_ROOT / "results")
    target_index = VARIABLES.index("electricity_pre_tax")
    terminal = forecast["future_level_paths"][:, -1, target_index]

    print(f"Dataset: {dataset}")
    print(f"Run ID: {result['metadata']['run_id']}")
    print(f"Saved run: {paths['directory']}")
    print(
        "Terminal electricity_pre_tax forecast: "
        f"median={np.median(terminal):.6g}, "
        f"q05={np.quantile(terminal, 0.05):.6g}, "
        f"q95={np.quantile(terminal, 0.95):.6g}"
    )


if __name__ == "__main__":
    main()
