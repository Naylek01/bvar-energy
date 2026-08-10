"""Build, materialise, register and promote one HICP Energy aggregate run.

Run from the bvar-energy project root:

    python src/model/run_hicp_energy_aggregate.py --vintage 20260808

If --vintage is omitted, the latest complete common vintage is used.
The script reuses saved component forecast stores; it does NOT re-estimate BVARs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "src" / "model").exists():
            return candidate
    raise FileNotFoundError("Could not locate the bvar-energy project root.")


PROJECT_ROOT = find_project_root(Path.cwd())
MODEL_DIR = PROJECT_ROOT / "src" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from energy_bvar_aggregate_pipeline import (  # noqa: E402
    assert_reference_audit,
    resolve_aggregate_vintage,
    run_aggregate,
)
from energy_bvar_display import build_aggregate_display, load_display_artifact  # noqa: E402
from energy_bvar_pipeline import CANONICAL_MODEL_IDS  # noqa: E402
from energy_bvar_registry import (  # noqa: E402
    default_registry_path,
    list_forecasts,
    promote_aggregate,
    promoted_run_ids,
    scan_results,
)


def _unique_run_ids(registry_path: Path, vintage: str, forecast_name: str) -> dict[str, str]:
    """Fallback when component runs have not yet been explicitly promoted.

    Safe only when exactly one valid saved forecast exists for every canonical
    model. Ambiguity is refused instead of selecting by timestamp.
    """
    frame = list_forecasts(
        registry_path,
        vintage=vintage,
        forecast_name=forecast_name,
        valid_only=True,
        present_only=True,
    )
    out: dict[str, str] = {}
    problems: list[str] = []
    for model_id in CANONICAL_MODEL_IDS:
        block = frame.loc[frame["model_id"].astype(str) == str(model_id)]
        ids = sorted(block["run_id"].astype(str).unique()) if not block.empty else []
        if len(ids) != 1:
            problems.append(f"{model_id}: found {len(ids)} usable runs {ids}")
        else:
            out[model_id] = ids[0]
    if problems:
        raise RuntimeError(
            "Cannot choose component runs safely. Promote one run per model or "
            "remove the ambiguity:\n  - " + "\n  - ".join(problems)
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vintage", default=None)
    parser.add_argument("--forecast-name", default="unconditional")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--pairing-seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    results_root = PROJECT_ROOT / "results"
    registry_path = default_registry_path(results_root=results_root)

    # First scan: make sure the registry reflects the actual stores on disk.
    scan_results(results_root=results_root, registry_path=registry_path)

    # Resolve the actual complete vintage before asking the registry for runs.
    vintage, _ = resolve_aggregate_vintage(
        args.vintage,
        project_root=PROJECT_ROOT,
        results_root=results_root,
        forecast_name=args.forecast_name,
        run_ids=None,
    )

    try:
        run_ids = promoted_run_ids(
            registry_path,
            vintage,
            forecast_name=args.forecast_name,
            require_all=True,
            require_draws=False,
        )
        selection_mode = "promoted component runs"
    except Exception:
        run_ids = _unique_run_ids(registry_path, vintage, args.forecast_name)
        selection_mode = "unique usable component runs"

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Vintage: {vintage}")
    print(f"Forecast contract: {args.forecast_name}")
    print(f"Component selection: {selection_mode}")
    for model_id in CANONICAL_MODEL_IDS:
        print(f"  {model_id:20s} {run_ids[model_id]}")

    outcome = run_aggregate(
        vintage=vintage,
        project_root=PROJECT_ROOT,
        results_root=results_root,
        forecast_name=args.forecast_name,
        run_ids=run_ids,
        n_aggregate_draws=args.draws,
        pairing_seed=args.pairing_seed,
        weekly_tax_mode="strict",
        persist=True,
        overwrite=args.overwrite,
    )
    if outcome.directory is None:
        raise RuntimeError("run_aggregate returned no output directory despite persist=True.")

    display_path = build_aggregate_display(
        outcome.directory,
        project_root=PROJECT_ROOT,
        overwrite=True,
    )

    # Re-scan after both the aggregate store and display_v1 exist, then promote.
    scan_results(results_root=results_root, registry_path=registry_path)
    if not args.no_promote:
        promote_aggregate(registry_path, vintage, outcome.aggregate_run_id)
        scan_results(results_root=results_root, registry_path=registry_path)

    display = load_display_artifact(display_path)
    model_hicp = display.loc[
        (display["scope"].astype(str) == "model_hicp")
        & (display["record_type"].astype(str) == "fan")
    ]
    yoy_models = sorted(
        model_hicp.loc[
            model_hicp["metric"].astype(str) == "component_hicp_yoy", "series"
        ].astype(str).unique()
    )

    print("\nAggregate run created")
    print(f"  aggregate_run_id: {outcome.aggregate_run_id}")
    print(f"  directory:        {outcome.directory}")
    print(f"  display:          {display_path}")
    print(f"  effective draws:  {outcome.n_aggregate_draws_effective}")
    print(f"  promoted:         {not args.no_promote}")
    print(f"  HICP YoY models:  {', '.join(yoy_models) if yoy_models else 'NONE'}")

    # For the known reference vintage this is a useful regression check. It is
    # printed rather than used as a hard failure because a legitimate data/config
    # change can move the reference figures.
    if str(vintage) == "20260808":
        try:
            print("\nReference audit")
            print(assert_reference_audit(outcome).to_string(index=False))
        except Exception as exc:
            print(f"Reference audit warning: {exc}")

    expected = set(CANONICAL_MODEL_IDS)
    if set(yoy_models) != expected:
        missing = sorted(expected.difference(yoy_models))
        raise RuntimeError(
            "Aggregate run was saved but its display does not expose HICP YoY "
            f"for all seven models. Missing: {missing}"
        )

    print("\nOK: the new aggregate store exposes HICP YoY for all seven model IDs.")
    print("Restart/refresh the Dash app. The promoted aggregate will be used by Forecast.")


if __name__ == "__main__":
    main()
