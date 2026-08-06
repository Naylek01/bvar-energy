"""Run and save the heat-energy or solid-fuels monthly BVAR.

Examples from the project root
------------------------------
python src/model/run_monthly_hicp_bvar.py heat_energy
python src/model/run_monthly_hicp_bvar.py solid_fuels
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


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

from energy_bvar_io import save_energy_bvar_result  # noqa: E402
from energy_bvar_model import (  # noqa: E402
    BVARSVOPriorConfig,
    SamplerConfig,
    forecast_bvar_sv_outlier,
    run_energy_bvar,
)
from energy_bvar_monthly_hicp import (  # noqa: E402
    MONTHLY_HICP_COMPONENTS,
    load_monthly_hicp_component_panel,
    monthly_hicp_component_spec,
)


P = 12
SEED = 42


def latest_dataset(component: str) -> Path:
    spec = monthly_hicp_component_spec(component)
    candidates = sorted(
        path
        for path in (PROJECT_ROOT / "data" / "processed").glob(
            f"*/{spec['dataset_file']}"
        )
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No data/processed/<vintage>/{spec['dataset_file']} was found."
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=tuple(MONTHLY_HICP_COMPONENTS))
    parser.add_argument("--reps", type=int, default=6_000)
    parser.add_argument("--burn", type=int, default=3_000)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    spec = monthly_hicp_component_spec(args.component)
    dataset = latest_dataset(args.component)
    levels = load_monthly_hicp_component_panel(
        dataset,
        component=args.component,
        variables=spec["variables"],
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
        reps=args.reps,
        burn=args.burn,
        thin=args.thin,
        seed=args.seed,
        max_stability_tries=1_000,
        progress_every=500,
    )

    result = run_energy_bvar(
        levels=levels,
        model_id=args.component,
        vintage=dataset.parent.name,
        p=P,
        variables=spec["variables"],
        prior_config=prior_config,
        sampler_config=sampler_config,
        code_version="energy_bvar_model-exog-v1",
    )

    forecast_origin = result["prep"]["last_calendar_date"]
    months_to_year_end = 12 - forecast_origin.month or 12
    horizon = max(3, months_to_year_end)
    forecast = forecast_bvar_sv_outlier(
        result,
        H=horizon,
        n_draws=min(1_000, result["n_draws"]),
        simulate_future_outliers=True,
        seed=2026,
    )

    paths = None
    if not args.no_save:
        paths = save_energy_bvar_result(result, PROJECT_ROOT / "results")

    target_index = spec["variables"].index(spec["target"])
    terminal = forecast["future_level_paths"][:, -1, target_index]

    print(f"Component: {args.component}")
    print(f"Dataset: {dataset}")
    print(f"Run ID: {result['metadata']['run_id']}")
    if paths is not None:
        print(f"Saved run: {paths['directory']}")
    print(
        f"Terminal {spec['target']} forecast: "
        f"median={np.median(terminal):.6g}, "
        f"q05={np.quantile(terminal, 0.05):.6g}, "
        f"q95={np.quantile(terminal, 0.95):.6g}"
    )


if __name__ == "__main__":
    main()
