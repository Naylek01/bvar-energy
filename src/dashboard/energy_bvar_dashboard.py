"""Dash front-end for the ECB-style Energy BVAR suite.

Dashboard foundation
---------------------
* persistent URL routing;
* central context store: vintage / model_id / run_id / forecast_name / draw_mode;
* SQLite-backed selectors with promoted-run awareness;
* exactly one ``display_v1.parquet`` read when the selected run changes;
* Forecast page driven exclusively by the in-memory display store;
* 68%, 90%, or combined posterior fans with ``uirevision``;
* observed / nowcast / forecast segmentation and a compact values table;
* Diskcache background estimation with progress/cancel and a global lock;
* full posterior persistence is mandatory for future Structural analysis.

The econometric code remains in ``src/model``.  This module only reads registry
and display artefacts and never re-estimates a model from a plotting callback.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from io import StringIO
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import diskcache
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (
    ALL,
    Dash,
    DiskcacheManager,
    Input,
    Output,
    State,
    callback,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate


# ---------------------------------------------------------------------------
# Project / model-library discovery
# ---------------------------------------------------------------------------


def _find_project_root(start: Path) -> Path:
    explicit = os.getenv("ENERGY_BVAR_PROJECT_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "model").is_dir() and (candidate / "data").exists():
            return candidate
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "model").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).parent)
MODEL_DIR = PROJECT_ROOT / "src" / "model"
if not MODEL_DIR.is_dir():
    # Also supports placing this file directly beside the model modules while
    # the dashboard structure is being introduced.
    MODEL_DIR = PROJECT_ROOT
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from energy_bvar_dashboard_aggregate import (  # noqa: E402
    aggregate_contribution_figure,
    aggregate_diagnostics_table,
    aggregate_fan_figure,
    aggregate_kpis,
    aggregate_metric_options,
    aggregate_live_scenario_payload,
    aggregate_live_scenario_figure,
    aggregate_live_impact_figure,
    aggregate_live_contribution_impact_figure,
    aggregate_live_scenario_kpis,
    empty_aggregate_figure,
)
from energy_bvar_dashboard_aggregate_ext import (  # noqa: E402
    adapt_yaxis_to_visible_window,
    aggregate_contribution_timeline_figure,
    aggregate_provenance_table_v2,
    build_historical_contribution_frame,
    tidy_aggregate_figure,
)
from energy_bvar_dashboard_diagnostics import (  # noqa: E402
    component_diagnostic_summary,
    empty_diagnostic_figure,
    key_mcmc_table,
    load_aggregate_validation,
    load_component_diagnostics,
    phi_ess_figure,
    stability_table,
)
from energy_bvar_dashboard_dataset import (  # noqa: E402
    DatasetBuildError,
    dataset_build_lock_state,
    inspect_dataset_build_environment,
    run_dataset_build,
)
from energy_bvar_dashboard_structural import (  # noqa: E402
    DEFAULT_HORIZON as STRUCTURAL_DEFAULT_HORIZON,
    DEFAULT_SHOCK_SIZE,
    DEFAULT_SHOCK_UNIT,
    DEFAULT_STRUCTURAL_DRAWS,
    MAX_HORIZON as STRUCTURAL_MAX_HORIZON,
    MAX_STRUCTURAL_DRAWS,
    StructuralDashboardError,
    compute_structural_v1,
    fevd_figure,
    historical_decomposition_figure,
    irf_figure,
    resolve_run_directory as resolve_structural_run_directory,
    structural_run_contract,
    volatility_history_figure,
)
from energy_bvar_dashboard_conditional import (  # noqa: E402
    CONDITIONAL_CONTRACT_VERSION,
    ConditionalScenarioError,
    clear_conditional_set,
    compute_conditional_scenario,
    conditional_aggregate_contract,
    conditional_aggregate_figure,
    conditional_aggregate_impact_figure,
    conditional_component_contract,
    conditional_component_figure,
    conditional_kpis,
    conditional_path_figure,
    conditional_set_components,
    conditional_set_payload,
    conditional_set_signature,
    conditional_set_summary,
    conditional_target_figure,
    empty_conditional_figure,
    empty_conditional_set,
    remove_conditional_component,
    scenario_marginal_impact_figure,
    upsert_conditional_component,
)
from energy_bvar_dashboard_scenarios import (  # noqa: E402
    clear_scenario_set,
    empty_scenario_figure,
    remove_scenario_component,
    scenario_impact_figure,
    scenario_kpis,
    scenario_main_figure,
    scenario_payload,
    scenario_set_components,
    scenario_set_payload,
    scenario_set_signature,
    scenario_set_summary,
    scenario_set_to_tax_scenarios,
    scenario_tax_figure,
    upsert_scenario_component,
)
from energy_bvar_component_hicp import (  # noqa: E402
    TAX_SCENARIO_MODEL_IDS,
    build_saved_component_tax_scenario,
    component_tax_scenario_contract,
)
from energy_bvar_aggregate_pipeline import run_aggregate  # noqa: E402
from energy_bvar_fitted import (  # noqa: E402
    load_energy_aggregate_fitted,
)
from energy_bvar_display import (  # noqa: E402
    DISPLAY_FILENAME,
    build_aggregate_display,
    build_component_display,
    display_metadata,
    load_display_artifact,
)
from energy_bvar_pipeline import (  # noqa: E402
    CANONICAL_MODEL_IDS,
    available_vintages,
    build_panel,
    model_spec,
    planned_run_metadata,
    resolve_common_vintage,
    run_component,
    vintage_coverage,
)
from energy_bvar_model import BVARSVOPriorConfig, SamplerConfig  # noqa: E402
from energy_bvar_registry import (  # noqa: E402
    default_registry_path,
    init_registry,
    list_aggregates,
    list_forecasts,
    list_runs,
    promote_aggregate,
    promote_run,
    promoted_run_ids,
    scan_results,
    set_run_status,
)

RESULTS_ROOT = Path(
    os.getenv("ENERGY_BVAR_RESULTS_ROOT", str(PROJECT_ROOT / "results"))
).expanduser().resolve()
REGISTRY_PATH = Path(
    os.getenv(
        "ENERGY_BVAR_REGISTRY",
        str(default_registry_path(results_root=RESULTS_ROOT)),
    )
).expanduser().resolve()
AUTO_BUILD_DISPLAY = os.getenv("ENERGY_BVAR_AUTO_BUILD_DISPLAY", "1") not in {
    "0",
    "false",
    "False",
}
AUTO_BUILD_FITTED = os.getenv("ENERGY_BVAR_AUTO_BUILD_FITTED", "1") not in {
    "0",
    "false",
    "False",
}
FITTED_MAX_DRAWS = max(
    1, int(os.getenv("ENERGY_BVAR_FITTED_MAX_DRAWS", "300"))
)
FITTED_PAIRING_SEED = int(
    os.getenv("ENERGY_BVAR_FITTED_PAIRING_SEED", "2026")
)

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
init_registry(REGISTRY_PATH)

CACHE_DIR = Path(
    os.getenv("ENERGY_BVAR_DASH_CACHE", str(RESULTS_ROOT / ".dash_cache"))
).expanduser().resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_diskcache = diskcache.Cache(str(CACHE_DIR))
background_callback_manager = DiskcacheManager(_diskcache)
ESTIMATION_LOCK_PATH = RESULTS_ROOT / ".estimation.lock"

ESTIMATION_PROFILE_DIR = PROJECT_ROOT / "configs" / "estimation"
ESTIMATION_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

ESTIMATION_PROFILE_VERSION = 1

# ---------------------------------------------------------------------------
# Production suite boundaries
# ---------------------------------------------------------------------------
# Keep the Energy suite explicit.  The future joint Headline BVAR must NOT be
# appended to the seven-model Energy aggregation contract: it is a separate
# model family with a separate output/aggregation layer.
ENERGY_SUITE_MODEL_IDS = tuple(CANONICAL_MODEL_IDS)
RESERVED_HEADLINE_JOINT_MODEL_ID = "headline_joint"

# Notebook 10 is now a reference/audit notebook.  Production aggregation is
# owned by energy_bvar_aggregate_pipeline.run_aggregate and is executed by the
# dashboard automatically after a successful seven-model suite.
ENERGY_AGGREGATE_FORECAST_NAME = "unconditional"
ENERGY_AGGREGATE_DRAWS = 500
ENERGY_AGGREGATE_PAIRING_SEED = 2026
ENERGY_AGGREGATE_WEEKLY_TAX_MODE = "strict"


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform best-effort liveness check for stale lock recovery."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_estimation_lock() -> dict:
    if not ESTIMATION_LOCK_PATH.is_file():
        return {}
    try:
        return json.loads(ESTIMATION_LOCK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clear_stale_estimation_lock() -> bool:
    """Remove a lock whose worker PID no longer exists."""
    if not ESTIMATION_LOCK_PATH.exists():
        return False
    payload = _read_estimation_lock()
    pid = int(payload.get("pid", -1)) if payload else -1
    if pid > 0 and _pid_is_alive(pid):
        return False
    try:
        ESTIMATION_LOCK_PATH.unlink()
        return True
    except FileNotFoundError:
        return False


@contextmanager
def _estimation_lock(model_id: str, vintage: str, run_id: str):
    """Global single-estimation lock shared by all dashboard workers.

    The file is created atomically.  If a cancelled background worker leaves a
    stale lock behind, the next estimation removes it after confirming that the
    recorded PID is no longer alive.

    Dataset construction and model estimation are mutually exclusive: a BVAR
    must never start while ``data/processed`` is being rebuilt.
    """
    dataset_state = dataset_build_lock_state(PROJECT_ROOT)
    if dataset_state.get("active"):
        raise RuntimeError(
            "Processed datasets are currently being rebuilt"
            + (
                f" for vintage {dataset_state.get('build_vintage')}"
                if dataset_state.get("build_vintage")
                else ""
            )
            + ". Wait for the dataset build to finish before estimating."
        )

    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "model_id": str(model_id),
        "vintage": str(vintage),
        "run_id": str(run_id),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for attempt in range(2):
        try:
            fd = os.open(
                ESTIMATION_LOCK_PATH,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            if attempt == 0 and _clear_stale_estimation_lock():
                continue
            current = _read_estimation_lock()
            owner = current.get("model_id", "another model") if current else "another model"
            started = current.get("started_at_utc", "unknown time") if current else "unknown time"
            raise RuntimeError(
                f"Another estimation is already running ({owner}, started {started})."
            )
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            break
    try:
        yield payload
    finally:
        try:
            current = _read_estimation_lock()
            if current.get("token") == token:
                ESTIMATION_LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            pass


def _paper_baseline_configs(model_id: str):
    """Canonical prior/sampler objects used by the current notebooks."""
    spec = model_spec(model_id)
    return BVARSVOPriorConfig(), SamplerConfig(seed=spec.seed)


def _baseline_per_model_rows() -> list[dict]:
    rows = []
    for model_id in ENERGY_SUITE_MODEL_IDS:
        spec = model_spec(model_id)
        prior, sampler = _paper_baseline_configs(model_id)
        rows.append(
            {
                "model_id": model_id,
                "model": spec.label,
                "p": int(spec.p),
                "lambda1": float(prior.lambda1),
                "lambda2": float(prior.lambda2),
                "lambda3": float(prior.lambda3),
                "lambda4": float(prior.lambda4),
                "reps": int(sampler.reps),
                "burn": int(sampler.burn),
                "thin": int(sampler.thin),
                "seed": int(sampler.seed),
            }
        )
    return rows


def _baseline_config_payload() -> dict:
    prior = BVARSVOPriorConfig()
    sampler = SamplerConfig(seed=42)
    return {
        "version": ESTIMATION_PROFILE_VERSION,
        "profile_mode": "paper_baseline",
        "scope": "selected_only",
        "prior": {
            "lambda1": prior.lambda1,
            "lambda2": prior.lambda2,
            "lambda3": prior.lambda3,
            "lambda4": prior.lambda4,
            "a_prior_var": prior.a_prior_var,
            "phi_prior_mean": prior.phi_prior_mean,
            "phi_prior_df": prior.phi_prior_df,
            "h0_var": prior.h0_var,
            "outlier_every": 1.0 / prior.outlier_mean_frequency,
            "outlier_prior_observations": prior.outlier_prior_observations,
            "outlier_grid_min": prior.outlier_grid_min,
            "outlier_grid_max": prior.outlier_grid_max,
            "outlier_grid_step": prior.outlier_grid_step,
            "ksc_offset_scale": prior.ksc_offset_scale,
            "ksc_offset_floor": prior.ksc_offset_floor,
        },
        "sampler": {
            "reps": sampler.reps,
            "burn": sampler.burn,
            "thin": sampler.thin,
            "seed": sampler.seed,
            "max_stability_tries": sampler.max_stability_tries,
            "progress_every": sampler.progress_every,
            "dk_projection_mode": sampler.dk_projection_mode,
            "dk_level_relative_gate": sampler.dk_level_relative_gate,
            "dk_difference_relative_gate": sampler.dk_difference_relative_gate,
            "dk_catastrophic_level_relative_gate": sampler.dk_catastrophic_level_relative_gate,
            "dk_catastrophic_difference_relative_gate": sampler.dk_catastrophic_difference_relative_gate,
        },
        "per_model": _baseline_per_model_rows(),
    }


def _profile_slug(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    value = value.strip("._-")
    if not value:
        raise ValueError("Profile name must contain at least one letter or number.")
    return value[:80]


def _saved_profile_paths() -> list[Path]:
    return sorted(
        (p for p in ESTIMATION_PROFILE_DIR.glob("*.json") if p.is_file()),
        key=lambda p: p.stem.lower(),
    )


def _saved_profile_options() -> list[dict]:
    options = [
        {"label": "Canonical project baseline", "value": "paper_baseline"},
        {"label": "Custom", "value": "custom"},
    ]
    for path in _saved_profile_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            label = str(payload.get("name") or path.stem)
        except Exception:
            label = path.stem
        options.append({"label": label, "value": f"saved:{path.stem}"})
    return options


def _load_saved_profile(value: str) -> dict:
    if not str(value).startswith("saved:"):
        raise ValueError("Not a saved profile selector.")
    stem = str(value).split(":", 1)[1]
    path = ESTIMATION_PROFILE_DIR / f"{stem}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Saved profile not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != ESTIMATION_PROFILE_VERSION:
        raise ValueError(
            f"Unsupported estimation profile version {payload.get('version')!r}."
        )
    return payload


def _safe_float(value, name: str) -> float:
    if value is None or value == "":
        raise ValueError(f"{name} is required.")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _safe_int(value, name: str) -> int:
    number = _safe_float(value, name)
    integer = int(round(number))
    if abs(number - integer) > 1e-10:
        raise ValueError(f"{name} must be an integer.")
    return integer


def _prior_from_mapping(values: dict) -> BVARSVOPriorConfig:
    outlier_every = _safe_float(values.get("outlier_every"), "outlier frequency denominator")
    if outlier_every <= 1:
        raise ValueError("Outlier mean frequency must be less than one per period.")
    prior = BVARSVOPriorConfig(
        lambda1=_safe_float(values.get("lambda1"), "lambda1"),
        lambda2=_safe_float(values.get("lambda2"), "lambda2"),
        lambda3=_safe_float(values.get("lambda3"), "lambda3"),
        lambda4=_safe_float(values.get("lambda4"), "lambda4"),
        a_prior_var=_safe_float(values.get("a_prior_var"), "A prior variance"),
        phi_prior_mean=_safe_float(values.get("phi_prior_mean"), "phi prior mean"),
        phi_prior_df=_safe_float(values.get("phi_prior_df"), "phi prior df"),
        h0_var=_safe_float(values.get("h0_var"), "initial log-vol variance"),
        outlier_mean_frequency=1.0 / outlier_every,
        outlier_prior_observations=_safe_float(
            values.get("outlier_prior_observations"),
            "outlier prior observations",
        ),
        outlier_grid_min=_safe_float(values.get("outlier_grid_min"), "outlier grid minimum"),
        outlier_grid_max=_safe_float(values.get("outlier_grid_max"), "outlier grid maximum"),
        outlier_grid_step=_safe_float(values.get("outlier_grid_step"), "outlier grid step"),
        ksc_offset_scale=_safe_float(values.get("ksc_offset_scale"), "KSC offset scale"),
        ksc_offset_floor=_safe_float(values.get("ksc_offset_floor"), "KSC offset floor"),
    )
    prior.validate()
    return prior


def _sampler_from_mapping(values: dict, *, seed_fallback: int = 42) -> SamplerConfig:
    sampler = SamplerConfig(
        reps=_safe_int(values.get("reps"), "repetitions"),
        burn=_safe_int(values.get("burn"), "burn-in"),
        thin=_safe_int(values.get("thin"), "thinning"),
        seed=_safe_int(values.get("seed", seed_fallback), "seed"),
        max_stability_tries=_safe_int(
            values.get("max_stability_tries"),
            "max stability tries",
        ),
        progress_every=_safe_int(values.get("progress_every"), "progress interval"),
        dk_projection_mode=str(values.get("dk_projection_mode", "strict")),
        dk_level_relative_gate=_safe_float(
            values.get("dk_level_relative_gate"),
            "DK level relative gate",
        ),
        dk_difference_relative_gate=_safe_float(
            values.get("dk_difference_relative_gate"),
            "DK difference relative gate",
        ),
        dk_catastrophic_level_relative_gate=_safe_float(
            values.get("dk_catastrophic_level_relative_gate"),
            "DK catastrophic level gate",
        ),
        dk_catastrophic_difference_relative_gate=_safe_float(
            values.get("dk_catastrophic_difference_relative_gate"),
            "DK catastrophic difference gate",
        ),
    )
    sampler.validate()
    return sampler


def _row_for_model(config_store: dict, model_id: str) -> dict:
    for row in config_store.get("per_model", []):
        if str(row.get("model_id")) == str(model_id):
            return dict(row)
    for row in _baseline_per_model_rows():
        if row["model_id"] == model_id:
            return row
    raise ValueError(f"No per-model configuration row for {model_id!r}.")


def _model_estimation_configs(
    model_id: str,
    selected_model_id: str,
    config_store: dict | None,
) -> tuple[BVARSVOPriorConfig, SamplerConfig, dict]:
    """Resolve effective prior/sampler/spec overrides for one model.

    Scope semantics
    ---------------
    selected_only:
        Custom settings apply only to the model selected in the Estimation UI.
        The other six models retain the canonical project baseline in Estimate-all-7.
    all_7:
        Shared prior and sampler settings apply to every BVAR. Structural model
        properties (lag order, seasonal reference month, exog prior scale) stay
        at each model's baseline values.
    per_model:
        The editable seven-row table overrides lambda1-4, p, reps, burn, thin
        and seed for each model. The remaining SV/outlier/numerical prior
        settings come from the shared custom editor.
    """
    if not config_store or not config_store.get("valid"):
        raise ValueError(
            "Estimation configuration is invalid. Correct the highlighted settings first."
        )
    profile_mode = str(config_store.get("profile_mode", "paper_baseline"))
    scope = str(config_store.get("scope", "selected_only"))
    spec = model_spec(model_id)

    if profile_mode == "paper_baseline":
        prior, sampler = _paper_baseline_configs(model_id)
        return prior, sampler, {}

    if scope == "selected_only" and model_id != selected_model_id:
        prior, sampler = _paper_baseline_configs(model_id)
        return prior, sampler, {}

    shared_prior = dict(config_store["prior"])
    shared_sampler = dict(config_store["sampler"])

    if scope == "per_model":
        row = _row_for_model(config_store, model_id)
        shared_prior.update(
            {
                key: row[key]
                for key in ("lambda1", "lambda2", "lambda3", "lambda4")
            }
        )
        shared_sampler.update(
            {
                key: row[key]
                for key in ("reps", "burn", "thin", "seed")
            }
        )
        spec_overrides = {"p": _safe_int(row["p"], f"{model_id} lag order")}
    else:
        # selected_only uses the selected model's structure editor. all_7 keeps
        # each model's native structure to avoid accidentally imposing p=12 on
        # the weekly models or p=24 on monthly models.
        spec_overrides = {}
        if scope == "selected_only" and model_id == selected_model_id:
            structure = dict(config_store.get("selected_structure", {}))
            if structure.get("p") is not None:
                spec_overrides["p"] = _safe_int(
                    structure["p"], f"{model_id} lag order"
                )
            if model_id == "electricity":
                if structure.get("exog_prior_scale") is not None:
                    spec_overrides["exog_prior_scale"] = _safe_float(
                        structure["exog_prior_scale"],
                        "electricity exogenous prior scale",
                    )
                if structure.get("reference_month") is not None:
                    reference = _safe_int(
                        structure["reference_month"], "electricity reference month"
                    )
                    if reference not in range(1, 13):
                        raise ValueError("Electricity reference month must lie in 1..12.")
                    spec_overrides["reference_month"] = reference

    prior = _prior_from_mapping(shared_prior)
    sampler = _sampler_from_mapping(shared_sampler, seed_fallback=spec.seed)
    return prior, sampler, spec_overrides


def _planned_run_state(
    model_id: str,
    vintage: str,
    *,
    selected_model_id: str | None = None,
    config_store: dict | None = None,
) -> dict:
    """Inspect the deterministic run location without starting Gibbs."""
    selected_model_id = selected_model_id or model_id
    if config_store is None:
        config_store = {
            **_baseline_config_payload(),
            "valid": True,
        }
    prior, sampler, spec_overrides = _model_estimation_configs(
        model_id,
        selected_model_id,
        config_store,
    )
    metadata = planned_run_metadata(
        model_id,
        vintage,
        project_root=PROJECT_ROOT,
        prior_config=prior,
        sampler_config=sampler,
        spec_overrides=spec_overrides,
    )
    run_id = str(metadata["run_id"])
    directory = RESULTS_ROOT / str(metadata["model_id"]) / str(metadata["vintage"]) / run_id
    forecast_dir = directory / "forecasts" / "unconditional"
    return {
        "metadata": metadata,
        "run_id": run_id,
        "directory": directory,
        "prior": prior,
        "sampler": sampler,
        "spec_overrides": spec_overrides,
        "has_draws": (directory / "draws.npz").is_file(),
        "has_forecast": (forecast_dir / "forecast_metadata.json").is_file()
        and (forecast_dir / "forecast_draws.npz").is_file(),
        "has_hicp": (forecast_dir / "hicp_metadata.json").is_file()
        and (forecast_dir / "hicp_draws.npz").is_file(),
        "has_display": (forecast_dir / DISPLAY_FILENAME).is_file(),
    }


def _config_differs_from_baseline(
    model_id: str,
    selected_model_id: str,
    config_store: dict,
) -> bool:
    try:
        prior, sampler, spec_overrides = _model_estimation_configs(
            model_id, selected_model_id, config_store
        )
        baseline_prior, baseline_sampler = _paper_baseline_configs(model_id)
        return (
            prior != baseline_prior
            or sampler != baseline_sampler
            or bool(spec_overrides)
        )
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Small data helpers
# ---------------------------------------------------------------------------


def _scan_registry() -> dict[str, Any]:
    try:
        report = scan_results(results_root=RESULTS_ROOT, registry_path=REGISTRY_PATH)
        return {
            "ok": True,
            "revision": pd.Timestamp.utcnow().isoformat(),
            "message": (
                f"{report.component_runs_seen} runs · {report.forecasts_seen} forecasts · "
                f"{report.aggregates_seen} aggregates"
            ),
            "unexpected": list(report.unexpected_directories),
        }
    except Exception as exc:  # surfaced in the UI, not swallowed
        return {
            "ok": False,
            "revision": pd.Timestamp.utcnow().isoformat(),
            "message": f"Registry scan failed: {exc}",
            "unexpected": [],
        }


def _json_frame(frame: pd.DataFrame) -> str:
    return frame.to_json(orient="split", date_format="iso", double_precision=15)


def _frame_from_store(store: dict | None) -> pd.DataFrame:
    if not store or not store.get("frame_json"):
        return pd.DataFrame()
    frame = pd.read_json(StringIO(store["frame_json"]), orient="split")
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _short_run(run_id: str) -> str:
    run_id = str(run_id)
    return run_id if len(run_id) <= 12 else run_id[:12]


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if not np.isfinite(value):
        return "—"
    absolute = abs(value)
    if absolute != 0 and (absolute < 0.01 or absolute >= 10_000):
        return f"{value:.2e}"
    return f"{value:,.2f}"


def _forecast_rows(frame: pd.DataFrame, metric: str, series: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = (
        (frame["record_type"] == "fan")
        & (frame["metric"] == metric)
        & (frame["series"] == series)
    )
    return frame.loc[mask].sort_values("date").copy()


def _history_rows(frame: pd.DataFrame, metric: str, series: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = (
        (frame["record_type"] == "history")
        & (frame["metric"] == metric)
        & (frame["series"] == series)
    )
    return frame.loc[mask].sort_values("date").copy()


def _rgba(hex_color: str, opacity: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{opacity})"


def _add_band(
    fig: go.Figure,
    fan: pd.DataFrame,
    lower: str,
    upper: str,
    name: str,
    *,
    opacity: float,
    color: str = "#2563eb",
) -> None:
    if fan.empty:
        return
    x = pd.concat([fan["date"], fan["date"].iloc[::-1]], ignore_index=True)
    y = pd.concat([fan[upper], fan[lower].iloc[::-1]], ignore_index=True)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="toself",
            fillcolor=_rgba(color, opacity),
            line={"width": 0},
            hoverinfo="skip",
            name=name,
            legendgroup="fan",
        )
    )


def _anchor_fan_line(block: pd.DataFrame, date, value) -> pd.DataFrame:
    if block.empty or date is None or pd.isna(date) or value is None or pd.isna(value):
        return block
    anchor = {column: np.nan for column in block.columns}
    anchor["date"] = pd.Timestamp(date)
    for column in ("q05", "q16", "q50", "q84", "q95"):
        if column in block.columns:
            anchor[column] = float(value)
    return pd.concat([pd.DataFrame([anchor]), block], ignore_index=True).sort_values("date")

def forecast_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    series: str,
    fan_mode: str,
    context: dict,
) -> go.Figure:
    """Continuous observed -> nowcast -> forecast chart with distinct nowcast styling."""
    history = _history_rows(frame, metric, series)
    fan = _forecast_rows(frame, metric, series)
    nowcast = fan.loc[fan["segment"].astype(str) == "nowcast"].copy() if not fan.empty else fan
    forecast = fan.loc[fan["segment"].astype(str) == "forecast"].copy() if not fan.empty else fan
    fig = go.Figure()

    now_color = "#A66A00"
    fc_color = "#2563eb"
    if fan_mode in {"90", "both"}:
        _add_band(fig, nowcast, "q05", "q95", "Nowcast 90% interval", opacity=0.09, color=now_color)
        _add_band(fig, forecast, "q05", "q95", "Forecast 90% interval", opacity=0.11, color=fc_color)
    if fan_mode in {"68", "both"}:
        _add_band(fig, nowcast, "q16", "q84", "Nowcast 68% interval", opacity=0.18, color=now_color)
        _add_band(fig, forecast, "q16", "q84", "Forecast 68% interval", opacity=0.22, color=fc_color)

    if not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["date"], y=history["value"], mode="lines", name="Observed",
                line={"width": 2.1, "color": "#0f172a"},
                hovertemplate="%{x|%Y-%m-%d}<br>Observed: %{y:.3f}<extra></extra>",
            )
        )

    anchor_date = None
    anchor_value = None
    if not history.empty:
        anchor_date = pd.Timestamp(history["date"].iloc[-1])
        anchor_value = float(history["value"].iloc[-1])

    now_line = _anchor_fan_line(nowcast, anchor_date, anchor_value)
    if not now_line.empty:
        fig.add_trace(
            go.Scatter(
                x=now_line["date"], y=now_line["q50"], mode="lines", name="Nowcast median",
                line={"width": 2.4, "color": now_color},
                hovertemplate="%{x|%Y-%m-%d}<br>Nowcast: %{y:.3f}<extra></extra>",
            )
        )
        anchor_date = pd.Timestamp(now_line["date"].iloc[-1])
        anchor_value = float(now_line["q50"].iloc[-1])

    fc_line = _anchor_fan_line(forecast, anchor_date, anchor_value)
    if not fc_line.empty:
        fig.add_trace(
            go.Scatter(
                x=fc_line["date"], y=fc_line["q50"], mode="lines", name="Forecast median",
                line={"width": 2.4, "dash": "dash", "color": fc_color},
                hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.3f}<extra></extra>",
            )
        )

    if not history.empty and not nowcast.empty:
        last_obs = pd.Timestamp(history["date"].iloc[-1])
        fig.add_vline(x=last_obs, line_width=1, line_dash="dash", line_color="#94a3b8")
    if not forecast.empty:
        first_future = pd.Timestamp(forecast["date"].min())
        fig.add_vline(x=first_future, line_width=1, line_dash="dot", line_color="#94a3b8")
        fig.add_annotation(
            x=first_future, y=1, yref="paper", text="Forecast", showarrow=False,
            xanchor="left", yanchor="bottom", font={"size": 11, "color": "#64748b"},
        )

    label = series.replace("_", " ").title()
    unit = ""
    selected = frame.loc[
        (frame["series"] == series)
        & (frame["metric"].astype(str) == str(metric))
        & frame["unit"].notna(),
        "unit",
    ]
    if len(selected):
        unit = str(selected.iloc[0])

    fig.update_layout(
        template="plotly_white",
        margin={"l": 54, "r": 24, "t": 54, "b": 42},
        height=520,
        title={"text": label, "x": 0.01, "xanchor": "left", "font": {"size": 18, "color": "#111827"}},
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#374151", "size": 12},
        xaxis_title=None, yaxis_title=unit, hovermode="x unified", dragmode="pan",
        hoverlabel={"bgcolor": "white", "bordercolor": "#e5e7eb", "font": {"color": "#111827"}},
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right", "font": {"size": 11}},
        uirevision=f"{context.get('model_id')}::{series}::{metric}",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e5e7eb", tickfont={"color": "#6b7280"})
    fig.update_yaxes(gridcolor="#eef0f3", zerolinecolor="#d1d5db", tickfont={"color": "#6b7280"})
    return fig

def _empty_forecast_figure(message: str = "Select a run to display its forecast") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.54,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        align="center",
        font={"size": 14, "color": "#6b7280"},
    )
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin={"l": 40, "r": 20, "t": 30, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


# ---------------------------------------------------------------------------
# UI pieces
# ---------------------------------------------------------------------------


def _selector(label: str, component_id: str, placeholder: str) -> html.Div:
    return html.Div(
        [
            html.Label(label, className="selector-label"),
            dcc.Dropdown(
                id=component_id,
                placeholder=placeholder,
                clearable=False,
                className="selector-dropdown",
            ),
        ],
        className="selector-block",
    )


def _nav_link(label: str, href: str, icon: str) -> dcc.Link:
    return dcc.Link(
        [html.Span(icon, className="nav-icon"), html.Span(label)],
        href=href,
        className="nav-link",
    )


def _stat_card(title: str, value_id: str, subtitle_id: str | None = None) -> html.Div:
    children = [
        html.Div(title, className="stat-title"),
        html.Div("—", id=value_id, className="stat-value"),
    ]
    if subtitle_id:
        children.append(html.Div("", id=subtitle_id, className="stat-subtitle"))
    return html.Div(children, className="stat-card")


_GRAPH_CONFIG: dict = {
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

# Aggregate plots have long fitted histories with very different local scales.
# Keep Plotly's autoscale button available and let double-click restore both
# axes. The main aggregate callback also re-fits Y to the visible X window.
_AGG_GRAPH_CONFIG: dict = {
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset+autosize",
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def forecast_page() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Forecast", className="page-title"),
                            html.P(
                                "Observed data, ragged-edge nowcast and posterior predictive forecast from the selected saved run.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Metric", className="control-label"),
                                    dcc.Dropdown(
                                        id="forecast-metric",
                                        clearable=False,
                                        className="compact-dropdown",
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label("Series", className="control-label"),
                                    dcc.Dropdown(
                                        id="forecast-series",
                                        clearable=False,
                                        className="compact-dropdown wide-control",
                                    ),
                                ],
                                className="control-block wide-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Fan", className="control-label"),
                                    dcc.RadioItems(
                                        id="forecast-fan",
                                        options=[
                                            {"label": "68%", "value": "68"},
                                            {"label": "90%", "value": "90"},
                                            {"label": "Both", "value": "both"},
                                        ],
                                        value="68",
                                        inline=True,
                                        className="fan-radio",
                                    ),
                                ],
                                className="control-block",
                            ),
                        ],
                        className="chart-controls",
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(
                [
                    _stat_card("Latest observed", "stat-observed", "stat-observed-date"),
                    _stat_card("First future median", "stat-first", "stat-first-date"),
                    _stat_card("Terminal median", "stat-terminal", "stat-terminal-date"),
                    _stat_card("Forecast horizon", "stat-horizon", "stat-horizon-unit"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    dcc.Loading(
                        dcc.Graph(
                            id="forecast-graph",
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                                "doubleClick": "reset",
                                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                            },
                        ),
                        type="circle",
                    )
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Path values", className="panel-title"),
                            html.P(
                                "Quantiles are read from the compact display artefact; changing the fan does not reopen posterior draw files.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dash_table.DataTable(
                        id="forecast-values-table",
                        columns=[
                            {"name": "Date", "id": "date"},
                            {"name": "Segment", "id": "segment"},
                            {"name": "q05", "id": "q05", "type": "numeric"},
                            {"name": "q16", "id": "q16", "type": "numeric"},
                            {"name": "Median", "id": "q50", "type": "numeric"},
                            {"name": "q84", "id": "q84", "type": "numeric"},
                            {"name": "q95", "id": "q95", "type": "numeric"},
                        ],
                        data=[],
                        page_size=12,
                        sort_action="native",
                        style_as_list_view=True,
                        style_cell={"fontFamily": "Inter, Segoe UI, sans-serif"},
                    ),
                ],
                className="panel table-panel",
            ),
        ],
        className="page-body",
    )



def aggregate_page() -> html.Div:
    """HICP Energy aggregate diagnostics and scenario propagation."""
    return html.Div(
        [
            # Keep the title independent from controls so it never collapses
            # into a narrow responsive grid column.
            html.Div(
                [
                    html.H2("HICP Energy aggregate", className="page-title"),
                    html.P(
                        "Draw-wise chain-linked HICP Energy, posterior model fit, "
                        "historical/forecast contributions and aggregation diagnostics.",
                        className="page-subtitle",
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(
                [
                    _selector("Aggregate run", "agg-select", "Select an aggregate"),
                    _selector("Metric", "agg-metric", "Metric"),
                    html.Div(
                        [
                            html.Span("Fan", className="control-label"),
                            dcc.RadioItems(
                                id="agg-fan",
                                options=[
                                    {"label": "68%", "value": "68"},
                                    {"label": "90%", "value": "90"},
                                    {"label": "Both", "value": "both"},
                                ],
                                value="68",
                                className="fan-radio",
                                inline=True,
                            ),
                        ],
                        className="control-block",
                    ),
                    html.Div(
                        [
                            html.Span("Model fit overlays", className="control-label"),
                            dcc.Checklist(
                                id="agg-historical-overlays",
                                options=[],
                                value=[],
                                inline=True,
                                className="fan-radio",
                            ),
                            html.Div(id="agg-overlay-note", className="control-help"),
                        ],
                        className="control-block",
                    ),
                ],
                className="panel chart-controls",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(190px, 1fr))",
                    "gap": "14px",
                    "alignItems": "start",
                    "marginBottom": "14px",
                },
            ),
            html.Div(id="agg-banner"),
            html.Div(
                [
                    _stat_card("Latest observed", "agg-observed", "agg-observed-date"),
                    _stat_card("First future median", "agg-first", "agg-first-date"),
                    _stat_card("Terminal median", "agg-terminal", "agg-terminal-date"),
                    _stat_card("Posterior draws", "agg-draws", "agg-draws-unit"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [dcc.Loading(dcc.Graph(id="agg-graph", config=_AGG_GRAPH_CONFIG, style={"height": "680px"}), type="circle")],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Active scenarios on HICP Energy", className="panel-title"),
                            html.P(
                                "Every active conditional and tax scenario is listed here. "
                                "Conditional curves are marginal effects versus the saved baseline. "
                                "Tax scenarios are also propagated jointly in the panel below. "
                                "Marginal effects are not added together.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(id="agg-all-scenarios-summary", className="selection-banner"),
                    dcc.Loading(
                        dcc.Graph(
                            id="agg-conditional-scenarios-impact",
                            config=_AGG_GRAPH_CONFIG,
                        ),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Combined VAT / excise scenario set", className="panel-title"),
                                    html.P(
                                        "All active tax scenarios are propagated jointly, draw-by-draw, through "
                                        "the selected HICP Energy aggregate. Conditional scenarios are shown "
                                        "separately above as marginal effects against the same saved baseline.",
                                        className="panel-subtitle",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            html.Div(id="agg-scenario-banner", className="selection-banner"),
                            html.Div(
                                [
                                    _stat_card("Baseline terminal YoY", "agg-scen-baseline", "agg-scen-date"),
                                    _stat_card("Scenario terminal YoY", "agg-scen-scenario", "agg-scen-date-2"),
                                    _stat_card("Terminal impact", "agg-scen-impact", "agg-scen-interval"),
                                    _stat_card("Paired aggregate draws", "agg-scen-draws", "agg-scen-draws-note"),
                                ],
                                className="stats-grid",
                            ),
                            dcc.Loading(dcc.Graph(id="agg-scenario-graph", config=_GRAPH_CONFIG), type="circle"),
                            html.Div(
                                [
                                    html.Div(
                                        [dcc.Loading(dcc.Graph(id="agg-scenario-impact", config=_GRAPH_CONFIG), type="circle")],
                                        className="panel chart-panel",
                                    ),
                                    html.Div(
                                        [dcc.Loading(dcc.Graph(id="agg-scenario-contrib-impact", config=_GRAPH_CONFIG), type="circle")],
                                        className="panel chart-panel",
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
                                    "gap": "16px",
                                },
                            ),
                        ]
                    )
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("HICP Energy contribution decomposition", className="panel-title"),
                                    html.P(
                                        "Observed historical contributions use the exact Laspeyres-term identity; "
                                        "the model path uses saved draw-wise posterior contribution medians.",
                                        className="panel-subtitle",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("Display", className="control-label"),
                                            dcc.RadioItems(
                                                id="agg-contrib-mode",
                                                options=[
                                                    {"label": "Stacked bars", "value": "bars"},
                                                    {"label": "Lines", "value": "lines"},
                                                ],
                                                value="bars",
                                                inline=True,
                                                className="fan-radio",
                                            ),
                                        ],
                                        className="control-block",
                                    ),
                                    html.Div(
                                        [
                                            html.Span("Labels", className="control-label"),
                                            dcc.Checklist(
                                                id="agg-contrib-labels",
                                                options=[{"label": "Show latest values", "value": "latest"}],
                                                value=[],
                                                inline=True,
                                                className="fan-radio",
                                            ),
                                        ],
                                        className="control-block",
                                    ),
                                ],
                                className="chart-controls",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "gap": "20px",
                            "flexWrap": "wrap",
                            "alignItems": "flex-start",
                        },
                    ),
                    dcc.Loading(dcc.Graph(id="agg-contrib", config=_AGG_GRAPH_CONFIG), type="circle"),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("How this HICP Energy forecast was built", className="panel-title"),
                            html.P(
                                "Source model runs, posterior pairing, tax treatment, HICP weights and validation checks "
                                "used for the selected aggregate. Short run IDs are shown first; the exact IDs remain in Details.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dash_table.DataTable(
                        id="agg-table",
                        columns=[
                            {"name": "Section", "id": "Section"},
                            {"name": "Setting / source", "id": "Item"},
                            {"name": "Value", "id": "Value"},
                            {"name": "Details", "id": "Detail"},
                        ],
                        data=[],
                        page_size=24,
                        style_as_list_view=True,
                        style_cell={
                            "fontFamily": "Inter, Segoe UI, sans-serif",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "maxWidth": "520px",
                        },
                    ),
                ],
                className="panel table-panel",
            ),
        ],
        className="page-body",
    )

def scenario_page() -> html.Div:
    """Conditional observable paths plus ex-post VAT/excise scenarios."""

    conditional_controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Aggregate run", className="control-label"),
                    dcc.Dropdown(
                        id="conditional-agg-select",
                        options=[],
                        value=None,
                        clearable=False,
                        className="compact-dropdown",
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Label("Component BVAR", className="control-label"),
                    dcc.Dropdown(
                        id="conditional-component-select",
                        options=[],
                        value=None,
                        clearable=False,
                        className="compact-dropdown",
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Label("Conditioned variable", className="control-label"),
                    dcc.Dropdown(
                        id="conditional-variable-select",
                        options=[],
                        value=None,
                        clearable=False,
                        className="compact-dropdown",
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Label("Path definition", className="control-label"),
                    dcc.RadioItems(
                        id="conditional-path-mode",
                        options=[
                            {"label": "% vs last observed", "value": "percent"},
                            {"label": "Absolute level", "value": "level"},
                        ],
                        value="percent",
                        inline=True,
                        className="fan-radio",
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Label(
                        "Change vs last observed (%)",
                        id="conditional-path-value-label",
                        className="control-label",
                    ),
                    dcc.Input(
                        id="conditional-path-value",
                        type="number",
                        value=10.0,
                        step=0.1,
                        debounce=0.4,
                        className="compact-dropdown",
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Label("Fan", className="control-label"),
                    dcc.RadioItems(
                        id="conditional-fan",
                        options=[
                            {"label": "68%", "value": "68"},
                            {"label": "90%", "value": "90"},
                            {"label": "Both", "value": "both"},
                        ],
                        value="68",
                        inline=True,
                        className="fan-radio",
                    ),
                ],
                className="control-block",
            ),
        ],
        className="chart-controls",
    )

    tax_controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Component", className="control-label"),
                    dcc.Dropdown(
                        id="scenario-component-select",
                        options=[],
                        value=None,
                        clearable=False,
                        className="compact-dropdown",
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Label("Scenario start", className="control-label"),
                    dcc.DatePickerSingle(
                        id="scenario-start-date",
                        display_format="YYYY-MM-DD",
                        first_day_of_week=1,
                        clearable=False,
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Label("VAT change (pp)", className="control-label"),
                    dcc.Input(
                        id="scenario-vat-delta",
                        type="number",
                        value=0.0,
                        step=0.1,
                        debounce=0.4,
                        className="compact-dropdown",
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Label(
                        id="scenario-excise-label",
                        children="Excise change",
                        className="control-label",
                    ),
                    dcc.Input(
                        id="scenario-excise-delta",
                        type="number",
                        value=0.0,
                        step=0.1,
                        debounce=0.4,
                        className="compact-dropdown",
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Label("Fan", className="control-label"),
                    dcc.RadioItems(
                        id="scenario-fan",
                        options=[
                            {"label": "68%", "value": "68"},
                            {"label": "90%", "value": "90"},
                            {"label": "Both", "value": "both"},
                        ],
                        value="68",
                        inline=True,
                        className="fan-radio",
                    ),
                ],
                className="control-block",
            ),
            html.Button(
                "Add / update",
                id="scenario-apply",
                n_clicks=0,
                className="refresh-button",
            ),
            html.Button(
                "Remove",
                id="scenario-remove",
                n_clicks=0,
                className="refresh-button",
            ),
            html.Button(
                "Reset all",
                id="scenario-reset-all",
                n_clicks=0,
                className="refresh-button",
            ),
        ],
        className="chart-controls",
    )

    return html.Div(
        [
            html.Div(
                [
                    html.H2("Scenarios", className="page-title"),
                    html.P(
                        "Conditional observable-price paths answer “what if the future "
                        "commodity path were X?”. VAT/excise scenarios remain ex-post "
                        "accounting shocks. Neither exercise re-estimates the BVAR; "
                        "Structural remains the separate identified-innovation analysis.",
                        className="page-subtitle",
                    ),
                ],
                className="page-heading-row",
            ),

            # ----------------------------------------------------------
            # Conditional observable path
            # ----------------------------------------------------------
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Conditional commodity / observable path",
                                className="panel-title",
                            ),
                            html.P(
                                "V1 imposes one permanent future LEVEL path from the "
                                "first forecast period onward. Only non-target endogenous "
                                "variables are offered. Baseline and conditional forecasts "
                                "reuse the exact same posterior draws, future SV, outliers "
                                "and simulation-smoother randomness; the effect is then "
                                "propagated through the saved HICP Energy pairing.",
                                className="panel-subtitle",
                            ),
                        ]
                    ),
                    conditional_controls,
                    html.Div(
                        id="conditional-support-note",
                        className="selection-banner",
                    ),
                    html.Div(id="conditional-banner"),
                    html.Div(
                        [
                            html.Button(
                                "Add / update conditional",
                                id="conditional-run",
                                n_clicks=0,
                                className="estimation-run-button",
                            ),
                            html.Button(
                                "Remove selected",
                                id="conditional-remove",
                                n_clicks=0,
                                className="refresh-button",
                            ),
                            html.Button(
                                "Reset all conditionals",
                                id="conditional-reset-all",
                                n_clicks=0,
                                className="refresh-button",
                            ),
                            html.Button(
                                "Cancel",
                                id="conditional-cancel",
                                n_clicks=0,
                                disabled=True,
                                className="estimation-cancel-button",
                            ),
                        ],
                        className="estimation-actions",
                    ),
                    html.Div(
                        [
                            html.Progress(
                                id="conditional-progress",
                                value=0,
                                max=100,
                                className="estimation-progress",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Idle",
                                        id="conditional-phase",
                                        className="estimation-phase",
                                    ),
                                    html.Div(
                                        "Choose a saved aggregate, component and observable path, then compute.",
                                        id="conditional-progress-detail",
                                        className="estimation-progress-detail",
                                    ),
                                ],
                                className="estimation-progress-text",
                            ),
                        ],
                        className="estimation-progress-wrap",
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Active conditional scenario set", className="eyebrow"),
                            html.Div(id="conditional-set-summary"),
                        ],
                        className="panel",
                        style={"padding": "16px 18px", "marginBottom": "16px"},
                    )
                ]
            ),
            html.Div(
                [
                    _stat_card(
                        "Imposed future level",
                        "conditional-stat-level",
                        "conditional-stat-level-note",
                    ),
                    _stat_card(
                        "Component terminal effect",
                        "conditional-stat-component",
                        "conditional-stat-component-note",
                    ),
                    _stat_card(
                        "HICP Energy terminal effect",
                        "conditional-stat-aggregate",
                        "conditional-stat-aggregate-note",
                    ),
                    _stat_card(
                        "Paired aggregate draws",
                        "conditional-stat-draws",
                        "conditional-stat-draws-note",
                    ),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    dcc.Loading(
                        dcc.Graph(
                            id="conditional-path-graph",
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                    )
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Loading(
                                dcc.Graph(
                                    id="conditional-target-graph",
                                    config=_GRAPH_CONFIG,
                                ),
                                type="circle",
                            )
                        ],
                        className="panel chart-panel",
                    ),
                    html.Div(
                        [
                            dcc.Loading(
                                dcc.Graph(
                                    id="conditional-component-graph",
                                    config=_GRAPH_CONFIG,
                                ),
                                type="circle",
                            )
                        ],
                        className="panel chart-panel",
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                    "gap": "16px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Loading(
                                dcc.Graph(
                                    id="conditional-aggregate-graph",
                                    config=_GRAPH_CONFIG,
                                ),
                                type="circle",
                            )
                        ],
                        className="panel chart-panel",
                    ),
                    html.Div(
                        [
                            dcc.Loading(
                                dcc.Graph(
                                    id="conditional-aggregate-impact-graph",
                                    config=_GRAPH_CONFIG,
                                ),
                                type="circle",
                            )
                        ],
                        className="panel chart-panel",
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                    "gap": "16px",
                    "marginBottom": "24px",
                },
            ),

            # ----------------------------------------------------------
            # Existing tax scenario block
            # ----------------------------------------------------------
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("VAT / excise scenarios", className="panel-title"),
                            html.P(
                                "Build several ex-post VAT/excise shocks on different "
                                "energy components. Saved BVAR price draws remain fixed; "
                                "the active tax set is propagated draw-by-draw to HICP Energy.",
                                className="panel-subtitle",
                            ),
                        ]
                    ),
                    tax_controls,
                ],
                className="panel",
            ),
            html.Div(id="scenario-support-note", className="selection-banner"),
            html.Div(id="scenario-banner"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Active tax scenario set", className="eyebrow"),
                            html.Div(id="scenario-set-summary"),
                        ],
                        className="panel",
                        style={"padding": "16px 18px", "marginBottom": "16px"},
                    )
                ]
            ),
            html.Div(
                [
                    _stat_card(
                        "Baseline terminal YoY",
                        "scenario-baseline-terminal",
                        "scenario-terminal-date",
                    ),
                    _stat_card(
                        "Scenario terminal YoY",
                        "scenario-scenario-terminal",
                        "scenario-terminal-date-2",
                    ),
                    _stat_card(
                        "Terminal tax impact",
                        "scenario-impact-terminal",
                        "scenario-impact-interval",
                    ),
                    _stat_card(
                        "Paired posterior draws",
                        "scenario-draws",
                        "scenario-draws-note",
                    ),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    dcc.Loading(
                        dcc.Graph(
                            id="scenario-main-graph",
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                    )
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Loading(
                                dcc.Graph(
                                    id="scenario-level-impact",
                                    config=_GRAPH_CONFIG,
                                ),
                                type="circle",
                            )
                        ],
                        className="panel chart-panel",
                    ),
                    html.Div(
                        [
                            dcc.Loading(
                                dcc.Graph(
                                    id="scenario-yoy-impact",
                                    config=_GRAPH_CONFIG,
                                ),
                                type="circle",
                            )
                        ],
                        className="panel chart-panel",
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                    "gap": "16px",
                },
            ),
            html.Div(
                [
                    dcc.Loading(
                        dcc.Graph(
                            id="scenario-tax-graph",
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                    )
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Selected tax scenario — HICP Energy marginal effect",
                                className="panel-title",
                            ),
                            html.P(
                                "The selected VAT/excise scenario is propagated alone through "
                                "the same saved HICP Energy aggregate. The underlying pre-tax BVAR "
                                "target is unchanged by construction.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(
                        "BVAR target effect = 0 by construction for a pure tax scenario.",
                        className="selection-banner",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="scenario-tax-energy-graph",
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
        ],
        className="page-body",
    )



def structural_page() -> html.Div:
    """Recursive / Cholesky structural analysis from persisted Gibbs draws."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Structural analysis", className="page-title"),
                            html.P(
                                "Impulse responses, forecast-error variance decomposition and exact "
                                "historical decomposition from the selected saved posterior run. "
                                "V1 uses recursive (Cholesky) identification only.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(
                id="structural-run-banner",
                className="selection-banner",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Identification", className="control-label"),
                                    dcc.Dropdown(
                                        id="structural-identification",
                                        options=[
                                            {
                                                "label": "Recursive / Cholesky",
                                                "value": "recursive",
                                            }
                                        ],
                                        value="recursive",
                                        clearable=False,
                                        disabled=True,
                                        className="compact-dropdown",
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label("Reference volatility date", className="control-label"),
                                    dcc.Dropdown(
                                        id="structural-reference-date",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        className="compact-dropdown wide-control",
                                    ),
                                ],
                                className="control-block wide-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Horizon", className="control-label"),
                                    dcc.Input(
                                        id="structural-horizon",
                                        type="number",
                                        min=1,
                                        max=STRUCTURAL_MAX_HORIZON,
                                        step=1,
                                        value=STRUCTURAL_DEFAULT_HORIZON,
                                        className="est-profile-name-input",
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label("Posterior draws", className="control-label"),
                                    dcc.Input(
                                        id="structural-draws",
                                        type="number",
                                        min=1,
                                        max=MAX_STRUCTURAL_DRAWS,
                                        step=1,
                                        value=DEFAULT_STRUCTURAL_DRAWS,
                                        className="est-profile-name-input",
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label("Shock scaling", className="control-label"),
                                    dcc.Dropdown(
                                        id="structural-shock-unit",
                                        options=[
                                            {
                                                "label": "Structural standard deviations",
                                                "value": "structural_std",
                                            },
                                            {
                                                "label": "Unit-level impact",
                                                "value": "level",
                                            },
                                        ],
                                        value=DEFAULT_SHOCK_UNIT,
                                        clearable=False,
                                        className="compact-dropdown",
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label("Shock intensity", className="control-label"),
                                    dcc.RadioItems(
                                        id="structural-shock-size",
                                        options=[
                                            {"label": "1", "value": 1},
                                            {"label": "2", "value": 2},
                                            {"label": "3", "value": 3},
                                            {"label": "4", "value": 4},
                                        ],
                                        value=int(DEFAULT_SHOCK_SIZE),
                                        inline=True,
                                        className="fan-radio",
                                    ),
                                ],
                                className="control-block",
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "minmax(180px,0.8fr) minmax(240px,1.2fr) minmax(110px,0.45fr) minmax(130px,0.5fr) minmax(220px,0.9fr) minmax(180px,0.75fr)",
                            "gap": "14px",
                            "alignItems": "end",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="structural-hd-options",
                                options=[
                                    {
                                        "label": "Split realised outlier amplification in historical decomposition",
                                        "value": "split_outliers",
                                    }
                                ],
                                value=["split_outliers"],
                                className="estimation-checklist",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Recompute structural analysis",
                                        id="structural-run",
                                        n_clicks=0,
                                        className="refresh-button estimation-run-all-button",
                                    ),
                                    html.Button(
                                        "Cancel",
                                        id="structural-cancel",
                                        n_clicks=0,
                                        disabled=True,
                                        className="estimation-cancel-button",
                                    ),
                                ],
                                className="estimation-actions",
                            ),
                        ],
                        className="estimation-run-row",
                    ),
                    html.Div(
                        [
                            html.Progress(
                                id="structural-progress",
                                value=0,
                                max=100,
                                className="estimation-progress",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Idle",
                                        id="structural-phase",
                                        className="estimation-phase",
                                    ),
                                    html.Div(
                                        "The default analysis is computed automatically from the selected saved run.",
                                        id="structural-progress-detail",
                                        className="estimation-progress-detail",
                                    ),
                                ],
                                className="estimation-progress-text",
                            ),
                        ],
                        className="estimation-progress-wrap",
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Reference volatility history", className="panel-title"),
                                    html.P(
                                        "Use this history to choose the volatility state used to normalise IRFs and FEVD. "
                                        "Persistent SV is √λ; total scale is o√λ and therefore includes transient outlier "
                                        "amplification. Hover for exact values, zoom/pan horizontally, then choose the "
                                        "reference date above and recompute.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Volatility series", className="control-label"),
                                    dcc.Dropdown(
                                        id="structural-volatility-variable",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        className="compact-dropdown wide-control",
                                    ),
                                ],
                                className="control-block",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="structural-volatility-graph",
                            config=_AGG_GRAPH_CONFIG if "_AGG_GRAPH_CONFIG" in globals() else _GRAPH_CONFIG,
                            style={"height": "520px"},
                        ),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    _stat_card("Identification", "structural-stat-identification", "structural-stat-ordering"),
                    _stat_card("Reference date", "structural-stat-reference", "structural-stat-frequency"),
                    _stat_card("Posterior draws", "structural-stat-draws", "structural-stat-draws-total"),
                    _stat_card("HD reconstruction", "structural-stat-hd-error", "structural-stat-hd-error-note"),
                    _stat_card("FEVD sum error", "structural-stat-fevd-error", "structural-stat-fevd-error-note"),
                    _stat_card("IRF shock scale", "structural-stat-shock-scale", "structural-stat-shock-scale-note"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Impulse responses", className="panel-title"),
                                    html.P(
                                        "Regular structural shocks at the selected stochastic-volatility state. "
                                        "Structural-SD mode applies 1–4 standard deviations. Unit-level mode rescales "
                                        "each shock so the shocked variable moves by +1 to +4 units on impact. "
                                        "The transient outlier multiplier is excluded from IRFs.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Shock", className="control-label"),
                                            dcc.Dropdown(
                                                id="structural-shock",
                                                options=[],
                                                clearable=False,
                                                className="compact-dropdown wide-control",
                                            ),
                                        ],
                                        className="control-block wide-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Response", className="control-label"),
                                            dcc.Dropdown(
                                                id="structural-response",
                                                options=[],
                                                clearable=False,
                                                className="compact-dropdown wide-control",
                                            ),
                                        ],
                                        className="control-block wide-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("IRF object", className="control-label"),
                                            dcc.RadioItems(
                                                id="structural-irf-metric",
                                                options=[
                                                    {"label": "Cumulative level", "value": "cumulative"},
                                                    {"label": "Period change", "value": "change"},
                                                ],
                                                value="cumulative",
                                                inline=True,
                                                className="fan-radio",
                                            ),
                                        ],
                                        className="control-block",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Fan", className="control-label"),
                                            dcc.RadioItems(
                                                id="structural-irf-fan",
                                                options=[
                                                    {"label": "68%", "value": "68"},
                                                    {"label": "90%", "value": "90"},
                                                    {"label": "Both", "value": "both"},
                                                ],
                                                value="68",
                                                inline=True,
                                                className="fan-radio",
                                            ),
                                        ],
                                        className="control-block",
                                    ),
                                ],
                                className="chart-controls",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="structural-irf-graph",
                            config=_AGG_GRAPH_CONFIG if "_AGG_GRAPH_CONFIG" in globals() else _GRAPH_CONFIG,
                        ),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Forecast error variance decomposition", className="panel-title"),
                            html.P(
                                "Posterior median share of the selected response's forecast-error variance "
                                "at each horizon. Shares sum to 100% draw by draw.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="structural-fevd-graph",
                            config=_AGG_GRAPH_CONFIG if "_AGG_GRAPH_CONFIG" in globals() else _GRAPH_CONFIG,
                        ),
                        type="circle",
                    ),
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Historical decomposition", className="panel-title"),
                                    html.P(
                                        "Posterior-mean structural contributions. The posterior mean is used "
                                        "because it preserves the exact additive reconstruction identity; "
                                        "component-wise posterior medians generally do not.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("History window", className="control-label"),
                                    dcc.Dropdown(
                                        id="structural-hd-window",
                                        options=[
                                            {"label": "Last 5 years", "value": 60},
                                            {"label": "Last 10 years", "value": 120},
                                            {"label": "Last 13 years", "value": 156},
                                            {"label": "Full sample", "value": -1},
                                        ],
                                        value=156,
                                        clearable=False,
                                        className="compact-dropdown",
                                    ),
                                ],
                                className="control-block",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="structural-hd-graph",
                            config=_AGG_GRAPH_CONFIG if "_AGG_GRAPH_CONFIG" in globals() else _GRAPH_CONFIG,
                        ),
                        type="circle",
                    ),
                    html.Div(
                        "Structural ≠ scenario. This page asks what an identified innovation does. "
                        "Observable commodity-path assumptions belong in Scenarios and will be extended separately.",
                        className="estimation-required-note",
                    ),
                ],
                className="panel chart-panel",
            ),
        ],
        className="page-body",
    )



def estimation_page() -> html.Div:
    model_options = [
        {"label": model_spec(model_id).label, "value": model_id}
        for model_id in ENERGY_SUITE_MODEL_IDS
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Estimation", className="page-title"),
                            html.P(
                                "Production Energy-suite estimation from processed vintages. "
                                "A successful seven-model run automatically builds the HICP "
                                "Energy aggregate; full posterior draws remain persisted for "
                                "Structural analysis.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Model", className="control-label"),
                                    dcc.Dropdown(
                                        id="est-model-select",
                                        options=model_options,
                                        value="gas",
                                        clearable=False,
                                        className="compact-dropdown wide-control",
                                    ),
                                ],
                                className="control-block wide-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Data vintage", className="control-label"),
                                    dcc.Dropdown(
                                        id="est-vintage-select",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        className="compact-dropdown",
                                    ),
                                ],
                                className="control-block",
                            ),
                        ],
                        className="chart-controls",
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Data preparation", className="panel-title"),
                                    html.P(
                                        "After refreshing and saving the single Excel workbook, build the canonical "
                                        "processed Energy datasets here before running the BVARs. The builder writes "
                                        "only to data/processed/<vintage>; it never modifies saved results.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Workbook path", className="control-label"),
                                            dcc.Input(
                                                id="dataset-build-raw-path",
                                                type="text",
                                                value="",
                                                placeholder="Auto-detect below data/raw",
                                                className="est-profile-name-input",
                                                debounce=True,
                                            ),
                                        ],
                                        className="control-block wide-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Build vintage", className="control-label"),
                                            dcc.Input(
                                                id="dataset-build-vintage",
                                                type="text",
                                                value="",
                                                placeholder="Automatic",
                                                className="est-profile-name-input",
                                                debounce=True,
                                            ),
                                        ],
                                        className="control-block",
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "minmax(420px, 1.5fr) minmax(180px, 0.5fr)",
                                    "gap": "14px",
                                    "alignItems": "end",
                                    "width": "100%",
                                },
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(
                        id="dataset-build-environment",
                        className="selection-banner",
                    ),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="dataset-build-overwrite",
                                options=[
                                    {
                                        "label": "Replace processed files if this vintage already exists",
                                        "value": "overwrite",
                                    }
                                ],
                                value=["overwrite"],
                                className="estimation-checklist",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Build / refresh VAR datasets",
                                        id="dataset-build-run",
                                        n_clicks=0,
                                        className="refresh-button estimation-run-all-button",
                                    ),
                                    html.Button(
                                        "Cancel build",
                                        id="dataset-build-cancel",
                                        n_clicks=0,
                                        disabled=True,
                                        className="estimation-cancel-button",
                                    ),
                                ],
                                className="estimation-actions",
                            ),
                        ],
                        className="estimation-run-row",
                    ),
                    html.Div(
                        [
                            html.Progress(
                                id="dataset-build-progress",
                                value=0,
                                max=100,
                                className="estimation-progress",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Idle",
                                        id="dataset-build-phase",
                                        className="estimation-phase",
                                    ),
                                    html.Div(
                                        "Save the refreshed workbook, then build the processed vintage.",
                                        id="dataset-build-progress-detail",
                                        className="estimation-progress-detail",
                                    ),
                                ],
                                className="estimation-progress-text",
                            ),
                        ],
                        className="estimation-progress-wrap",
                    ),
                    html.Div(id="dataset-build-result-banner"),
                    html.Details(
                        [
                            html.Summary("Builder log"),
                            html.Pre(
                                id="dataset-build-log",
                                children="No dataset build has run in this dashboard session.",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "fontSize": "11px",
                                    "maxHeight": "320px",
                                    "overflowY": "auto",
                                    "marginTop": "10px",
                                },
                            ),
                        ],
                        style={"marginTop": "10px"},
                    ),
                    html.Div(
                        "Production build uses the canonical workbook builder with HICP aggregation inputs required. "
                        "The newly built vintage is automatically selected below when the build succeeds.",
                        className="estimation-required-note",
                    ),
                ],
                className="panel",
            ),
            html.Div(id="estimation-preview-banner", className="selection-banner"),
            html.Div(
                id="estimation-suite-banner",
                className="selection-banner estimation-suite-banner",
            ),
            html.Div(
                [
                    _stat_card("Frequency", "est-frequency", "est-calendar"),
                    _stat_card("Lag order", "est-lags", "est-target"),
                    _stat_card("Last data date", "est-last-date", "est-dataset"),
                    _stat_card("Planned run", "est-run-id", "est-existing-state"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Estimation configuration", className="panel-title"),
                                    html.P(
                                        "Canonical project baseline is immutable. It reproduces the project's validated production configuration; the ECB paper defines the model architecture but does not pin down every hyperparameter. Choose Custom or a saved profile to change priors and MCMC settings; every effective configuration enters the deterministic run identity.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Profile", className="control-label"),
                                            dcc.Dropdown(
                                                id="est-profile-select",
                                                options=_saved_profile_options(),
                                                value="paper_baseline",
                                                clearable=False,
                                                className="compact-dropdown est-profile-dropdown",
                                            ),
                                        ],
                                        className="control-block est-profile-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Scope", className="control-label"),
                                            dcc.RadioItems(
                                                id="est-config-scope",
                                                options=[
                                                    {"label": "Selected model only", "value": "selected_only"},
                                                    {"label": "All 7 models", "value": "all_7"},
                                                    {"label": "Per-model overrides", "value": "per_model"},
                                                ],
                                                value="selected_only",
                                                className="est-scope-radio",
                                                inline=True,
                                            ),
                                        ],
                                        className="control-block est-scope-control",
                                    ),
                                ],
                                className="est-config-top-controls",
                            ),
                        ],
                        className="panel-heading est-config-heading",
                    ),
                    html.Div(
                        id="est-profile-message",
                        className="est-profile-message",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Minnesota prior", className="eyebrow"),
                                    html.Div(
                                        [
                                            html.Label(["λ1", html.Span("Overall shrinkage")], className="est-field-label"),
                                            dcc.Input(id="est-lambda1", type="number", value=0.20, step=0.01, min=0.000001, className="est-number-input"),
                                            html.Label(["λ2", html.Span("Cross-variable shrinkage")], className="est-field-label"),
                                            dcc.Input(id="est-lambda2", type="number", value=0.50, step=0.01, min=0.000001, className="est-number-input"),
                                            html.Label(["λ3", html.Span("Lag decay")], className="est-field-label"),
                                            dcc.Input(id="est-lambda3", type="number", value=1.00, step=0.05, min=0.000001, className="est-number-input"),
                                            html.Label(["λ4", html.Span("Constant prior scale")], className="est-field-label"),
                                            dcc.Input(id="est-lambda4", type="number", value=10.00, step=0.5, min=0.000001, className="est-number-input"),
                                            html.Label(["A variance", html.Span("Contemporaneous matrix")], className="est-field-label"),
                                            dcc.Input(id="est-a-prior-var", type="number", value=10.0, step=0.5, min=0.000001, className="est-number-input"),
                                        ],
                                        className="est-field-grid",
                                    ),
                                ],
                                className="est-editor-card",
                            ),
                            html.Div(
                                [
                                    html.Div("Stochastic volatility / outliers", className="eyebrow"),
                                    html.Div(
                                        [
                                            html.Label(["φ mean", html.Span("SV innovation variance prior")], className="est-field-label"),
                                            dcc.Input(id="est-phi-mean", type="number", value=0.02, step=0.005, min=0.000001, className="est-number-input"),
                                            html.Label(["φ df", html.Span("Inverse-Gamma prior df")], className="est-field-label"),
                                            dcc.Input(id="est-phi-df", type="number", value=10.0, step=1, min=2.000001, className="est-number-input"),
                                            html.Label(["h₀ variance", html.Span("Initial log-volatility")], className="est-field-label"),
                                            dcc.Input(id="est-h0-var", type="number", value=4.0, step=0.5, min=0.000001, className="est-number-input"),
                                            html.Label(["Outlier every", html.Span("Mean periods, e.g. 48")], className="est-field-label"),
                                            dcc.Input(id="est-outlier-every", type="number", value=48.0, step=1, min=1.000001, className="est-number-input"),
                                            html.Label(["Prior obs.", html.Span("Outlier-frequency prior strength")], className="est-field-label"),
                                            dcc.Input(id="est-outlier-prior-obs", type="number", value=120.0, step=10, min=0.000001, className="est-number-input"),
                                        ],
                                        className="est-field-grid",
                                    ),
                                ],
                                className="est-editor-card",
                            ),
                            html.Div(
                                [
                                    html.Div("MCMC", className="eyebrow"),
                                    html.Div(
                                        [
                                            html.Label(["Repetitions", html.Span("Total Gibbs sweeps")], className="est-field-label"),
                                            dcc.Input(id="est-reps", type="number", value=6000, step=500, min=2, className="est-number-input"),
                                            html.Label(["Burn-in", html.Span("Discarded sweeps")], className="est-field-label"),
                                            dcc.Input(id="est-burn", type="number", value=3000, step=500, min=0, className="est-number-input"),
                                            html.Label(["Thinning", html.Span("Keep every n-th draw")], className="est-field-label"),
                                            dcc.Input(id="est-thin", type="number", value=1, step=1, min=1, className="est-number-input"),
                                            html.Label(["Seed", html.Span("Deterministic run identity")], className="est-field-label"),
                                            dcc.Input(id="est-seed", type="number", value=42, step=1, className="est-number-input"),
                                            html.Label(["Stability tries", html.Span("Max redraw attempts")], className="est-field-label"),
                                            dcc.Input(id="est-max-stability", type="number", value=1000, step=100, min=1, className="est-number-input"),
                                        ],
                                        className="est-field-grid",
                                    ),
                                    html.Div(id="est-retained-draws", className="est-retained-draws"),
                                ],
                                className="est-editor-card",
                            ),
                            html.Div(
                                [
                                    html.Div("Selected model structure", className="eyebrow"),
                                    html.Div(
                                        [
                                            html.Label(["VAR lags", html.Span("p")], className="est-field-label"),
                                            dcc.Input(id="est-custom-p", type="number", value=12, step=1, min=1, className="est-number-input"),
                                            html.Label(["Exog prior scale", html.Span("Electricity only")], className="est-field-label"),
                                            dcc.Input(id="est-exog-prior-scale", type="number", value=10.0, step=0.5, min=0.000001, className="est-number-input"),
                                            html.Label(["Reference month", html.Span("Electricity only, 1–12")], className="est-field-label"),
                                            dcc.Input(id="est-reference-month", type="number", value=1, step=1, min=1, max=12, className="est-number-input"),
                                        ],
                                        className="est-field-grid est-structure-grid",
                                    ),
                                    html.Div(
                                        "When scope is All 7 models, every model keeps its native structure (monthly p=12, weekly p=24, electricity seasonals).",
                                        className="est-editor-note",
                                    ),
                                ],
                                className="est-editor-card",
                            ),
                        ],
                        className="est-editor-grid",
                    ),
                    html.Details(
                        [
                            html.Summary("Advanced numerical settings"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Outlier grid minimum", className="est-field-label"),
                                            dcc.Input(id="est-outlier-grid-min", type="number", value=2.0, step=1, className="est-number-input"),
                                            html.Label("Outlier grid maximum", className="est-field-label"),
                                            dcc.Input(id="est-outlier-grid-max", type="number", value=20.0, step=1, className="est-number-input"),
                                            html.Label("Outlier grid step", className="est-field-label"),
                                            dcc.Input(id="est-outlier-grid-step", type="number", value=1.0, step=0.5, min=0.000001, className="est-number-input"),
                                            html.Label("KSC offset scale", className="est-field-label"),
                                            dcc.Input(id="est-ksc-offset-scale", type="number", value=1e-6, step=1e-6, min=1e-12, className="est-number-input"),
                                            html.Label("KSC offset floor", className="est-field-label"),
                                            dcc.Input(id="est-ksc-offset-floor", type="number", value=1e-12, step=1e-12, min=1e-16, className="est-number-input"),
                                            html.Label("Progress interval", className="est-field-label"),
                                            dcc.Input(id="est-progress-every", type="number", value=500, step=100, min=0, className="est-number-input"),
                                        ],
                                        className="est-field-grid est-advanced-grid",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("DK projection mode", className="est-field-label"),
                                            dcc.Dropdown(
                                                id="est-dk-mode",
                                                options=[
                                                    {"label": "Strict", "value": "strict"},
                                                    {"label": "Record only", "value": "record_only"},
                                                ],
                                                value="strict",
                                                clearable=False,
                                                className="compact-dropdown",
                                            ),
                                            html.Label("DK level relative gate", className="est-field-label"),
                                            dcc.Input(id="est-dk-level-gate", type="number", value=1e-6, step=1e-6, min=1e-12, className="est-number-input"),
                                            html.Label("DK difference relative gate", className="est-field-label"),
                                            dcc.Input(id="est-dk-difference-gate", type="number", value=1e-6, step=1e-6, min=1e-12, className="est-number-input"),
                                            html.Label("Catastrophic level gate", className="est-field-label"),
                                            dcc.Input(id="est-dk-cat-level-gate", type="number", value=1e-4, step=1e-5, min=1e-12, className="est-number-input"),
                                            html.Label("Catastrophic difference gate", className="est-field-label"),
                                            dcc.Input(id="est-dk-cat-difference-gate", type="number", value=1e-3, step=1e-4, min=1e-12, className="est-number-input"),
                                        ],
                                        className="est-field-grid est-advanced-grid",
                                    ),
                                ],
                                className="est-advanced-wrap",
                            ),
                        ],
                        className="est-advanced-details",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Per-model overrides", className="eyebrow"),
                                    html.Div(
                                        "Editable when scope is Per-model overrides. These rows control λ1–λ4, p and the core MCMC settings; SV/outlier/numerical settings remain the shared values above.",
                                        className="est-editor-note",
                                    ),
                                ],
                                className="est-per-model-heading",
                            ),
                            dash_table.DataTable(
                                id="est-per-model-table",
                                data=_baseline_per_model_rows(),
                                columns=[
                                    {"name": "Model", "id": "model", "editable": False},
                                    {"name": "p", "id": "p", "type": "numeric"},
                                    {"name": "λ1", "id": "lambda1", "type": "numeric"},
                                    {"name": "λ2", "id": "lambda2", "type": "numeric"},
                                    {"name": "λ3", "id": "lambda3", "type": "numeric"},
                                    {"name": "λ4", "id": "lambda4", "type": "numeric"},
                                    {"name": "Reps", "id": "reps", "type": "numeric"},
                                    {"name": "Burn", "id": "burn", "type": "numeric"},
                                    {"name": "Thin", "id": "thin", "type": "numeric"},
                                    {"name": "Seed", "id": "seed", "type": "numeric"},
                                ],
                                editable=True,
                                row_deletable=False,
                                sort_action="none",
                                page_action="none",
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "fontFamily": "inherit",
                                    "fontSize": "11px",
                                    "padding": "7px 8px",
                                    "textAlign": "right",
                                    "minWidth": "62px",
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "model"}, "textAlign": "left", "minWidth": "150px"},
                                ],
                                style_header={"fontWeight": 700},
                            ),
                        ],
                        id="est-per-model-wrap",
                        style={"display": "none"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Configuration identity", className="eyebrow"),
                                    html.Div(id="est-config-status", className="est-config-status"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("config hash"),
                                    html.Code(id="est-config-hash", children="—"),
                                    html.Span("planned run"),
                                    html.Code(id="est-config-run", children="—"),
                                ],
                                className="est-config-identity-values",
                            ),
                        ],
                        className="est-config-identity",
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="est-profile-name",
                                type="text",
                                placeholder="Profile name, e.g. tighter_minnesota",
                                className="est-profile-name-input",
                            ),
                            html.Button(
                                "Save as profile",
                                id="est-profile-save",
                                n_clicks=0,
                                className="est-profile-save-button",
                            ),
                            html.Div(
                                "Saved profiles live in configs/estimation/. The complete effective configuration is still persisted in each run metadata.",
                                className="est-editor-note",
                            ),
                        ],
                        className="est-profile-save-row",
                    ),
                    html.Div(
                        [
                            html.Div("Persistence contract", className="eyebrow"),
                            html.Div(
                                [
                                    html.Span("draws.npz", className="est-required-chip"),
                                    html.Span("forecast", className="est-required-chip"),
                                    html.Span("component HICP", className="est-required-chip"),
                                    html.Span("display_v1", className="est-required-chip"),
                                    html.Span("Structural-ready ✓", className="est-required-chip strong"),
                                ],
                                className="est-required-chips",
                            ),
                            html.Div(
                                "Full posterior persistence is mandatory and cannot be disabled from this page.",
                                className="estimation-required-note",
                            ),
                        ],
                        className="est-persistence-contract",
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Run", className="panel-title"),
                            html.P(
                                "A global lock prevents concurrent production work. "
                                "The Energy-suite action reuses identical complete component "
                                "runs and then builds/reuses their exact HICP Energy aggregate. "
                                "The aggregate-only action uses the seven promoted component runs.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="estimation-promote",
                                options=[
                                    {
                                        "label": "Promote completed run(s) when complete",
                                        "value": "promote",
                                    }
                                ],
                                value=["promote"],
                                className="estimation-checklist",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Estimate selected model",
                                        id="estimation-run",
                                        n_clicks=0,
                                        className="refresh-button estimation-run-button",
                                    ),
                                    html.Button(
                                        "Estimate Energy suite (7 + aggregate)",
                                        id="estimation-run-all",
                                        n_clicks=0,
                                        className="refresh-button estimation-run-all-button",
                                    ),
                                    html.Button(
                                        "Build Energy aggregate only",
                                        id="estimation-run-aggregate",
                                        n_clicks=0,
                                        className="refresh-button estimation-run-aggregate-button",
                                    ),
                                    html.Button(
                                        "Cancel",
                                        id="estimation-cancel",
                                        n_clicks=0,
                                        disabled=True,
                                        className="estimation-cancel-button",
                                    ),
                                ],
                                className="estimation-actions",
                            ),
                        ],
                        className="estimation-run-row",
                    ),
                    html.Div(
                        [
                            html.Progress(
                                id="estimation-progress",
                                value=0,
                                max=100,
                                className="estimation-progress",
                            ),
                            html.Div(
                                [
                                    html.Div("Idle", id="estimation-phase", className="estimation-phase"),
                                    html.Div("Select a model and vintage.", id="estimation-progress-detail", className="estimation-progress-detail"),
                                ],
                                className="estimation-progress-text",
                            ),
                        ],
                        className="estimation-progress-wrap",
                    ),
                    html.Div(
                        id="estimation-suite-status",
                        className="estimation-suite-status",
                    ),
                    html.Div(id="estimation-result-banner"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    _stat_card("Status", "est-result-status", "est-result-note"),
                    _stat_card("Run ID", "est-result-run", "est-result-promoted"),
                    _stat_card("Posterior draws", "est-result-draws", "est-result-draws-note"),
                    _stat_card("Elapsed", "est-result-elapsed", "est-result-directory"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Saved-run diagnostics", className="panel-title"),
                                    html.P(
                                        "Diagnostics are read from the persisted run artefacts produced by the same estimation. No Gibbs sampler is rerun when this panel changes.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Completed run", className="control-label"),
                                    dcc.Dropdown(
                                        id="est-diag-run-select",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        className="compact-dropdown wide-control",
                                    ),
                                ],
                                className="control-block wide-control",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(id="est-diag-run-note", className="selection-banner"),
                    html.Div(
                        [
                            _stat_card("Worst ESS(φ)", "est-diag-phi-ess", "est-diag-phi-ess-note"),
                            _stat_card("Median spectral radius", "est-diag-radius-median", "est-diag-radius-note"),
                            _stat_card("95% spectral radius", "est-diag-radius-q95", "est-diag-radius-q95-note"),
                            _stat_card("Instability rejection", "est-diag-rejection", "est-diag-rejection-note"),
                        ],
                        className="stats-grid",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="est-diag-phi-ess-graph",
                                        figure=empty_diagnostic_figure("Select a completed run"),
                                        config=_GRAPH_CONFIG,
                                    )
                                ],
                                style={"minWidth": "0"},
                            ),
                            html.Div(
                                [
                                    html.Div("Key MCMC diagnostics", className="eyebrow"),
                                    dash_table.DataTable(
                                        id="est-diag-mcmc-table",
                                        columns=[
                                            {"name": "Parameter", "id": "Parameter"},
                                            {"name": "Posterior mean", "id": "Posterior mean"},
                                            {"name": "Posterior SD", "id": "Posterior SD"},
                                            {"name": "ESS", "id": "ESS"},
                                            {"name": "MCSE / SD", "id": "MCSE / SD"},
                                        ],
                                        data=[],
                                        page_size=12,
                                        sort_action="native",
                                        style_table={"overflowX": "auto"},
                                        style_cell={"fontFamily": "inherit", "fontSize": "12px", "padding": "7px", "textAlign": "right"},
                                        style_cell_conditional=[{"if": {"column_id": "Parameter"}, "textAlign": "left", "minWidth": "180px"}],
                                        style_header={"fontWeight": 700},
                                    ),
                                ],
                                style={"minWidth": "0"},
                            ),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "minmax(320px, 0.8fr) minmax(520px, 1.2fr)", "gap": "18px", "alignItems": "start"},
                    ),
                    html.Div(
                        [
                            html.Div("Stability details", className="eyebrow"),
                            dash_table.DataTable(
                                id="est-diag-stability-table",
                                columns=[
                                    {"name": "Diagnostic", "id": "Diagnostic"},
                                    {"name": "Value", "id": "Value"},
                                ],
                                data=[],
                                style_table={"overflowX": "auto"},
                                style_cell={"fontFamily": "inherit", "fontSize": "12px", "padding": "7px", "textAlign": "right"},
                                style_cell_conditional=[{"if": {"column_id": "Diagnostic"}, "textAlign": "left", "minWidth": "220px"}],
                                style_header={"fontWeight": 700},
                            ),
                        ],
                        style={"marginTop": "16px"},
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("HICP Energy aggregate validation", className="panel-title"),
                            html.P(
                                "Validation diagnostics from the completed HICP Energy aggregate for the selected vintage. The additivity error is computed draw by draw before posterior summaries.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(id="est-agg-diag-note", className="selection-banner"),
                    html.Div(
                        [
                            _stat_card("Draw-wise additivity error", "est-agg-additivity", "est-agg-additivity-note"),
                            _stat_card("Six-model history error", "est-agg-history-error", "est-agg-history-note"),
                            _stat_card("Annual re-anchor error", "est-agg-anchor-error", "est-agg-anchor-note"),
                            _stat_card("Max weekly rejection", "est-agg-weekly-rejection", "est-agg-weekly-note"),
                        ],
                        className="stats-grid",
                    ),
                ],
                className="panel",
            ),
        ],
        className="page-body",
    )


def placeholder_page(title: str, description: str, next_slice: str) -> html.Div:
    return html.Div(
        [
            html.H2(title, className="page-title"),
            html.P(description, className="page-subtitle"),
            html.Div(
                [
                    html.Div("Wired shell", className="eyebrow"),
                    html.H3(next_slice, className="placeholder-title"),
                    html.P(
                        "The route is reserved and shares the same global context. Its econometric callback will be connected without changing the navigation or selection contract.",
                        className="placeholder-text",
                    ),
                ],
                className="panel placeholder-panel",
            ),
        ],
        className="page-body",
    )


def sidebar() -> html.Aside:
    return html.Aside(
        [
            html.Div(
                [
                    html.Div(className="brand-mark", title="BVAR"),
                    html.Div("Energy Inflation", className="brand-title"),
                ],
                className="brand-row",
            ),
            html.Nav(
                [
                    _nav_link("Forecast", "/forecast", "↗"),
                    _nav_link("Energy aggregate", "/aggregate", "Σ"),
                    _nav_link("Scenarios", "/scenarios", "△"),
                    _nav_link("Structural", "/structural", "ψ"),
                    _nav_link("Estimation", "/estimation", "⚙"),
                ],
                className="nav-stack",
            ),
            html.Div(
                [
                    html.Div("Storage", className="sidebar-meta-label"),
                    html.Div(str(RESULTS_ROOT), className="sidebar-path"),
                ],
                className="sidebar-footer",
            ),
        ],
        className="sidebar",
    )


def topbar() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    _selector("Vintage", "vintage-select", "Select vintage"),
                    _selector("Model", "model-select", "Select model"),
                    _selector("Forecast", "forecast-select", "Forecast contract"),
                    _selector("Run", "run-select", "Select run"),
                ],
                className="selectors-grid",
            ),
            html.Div(
                [
                    html.Button("Refresh", id="refresh-registry", n_clicks=0, className="refresh-button"),
                    html.Div(id="registry-status", className="registry-status"),
                ],
                className="refresh-area",
            ),
        ],
        className="topbar",
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    assets_folder=str(Path(__file__).parent / "assets"),
    suppress_callback_exceptions=True,
    background_callback_manager=background_callback_manager,
    title="Energy BVAR Dashboard",
    update_title="Updating…",
)
server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="registry-store", data=_scan_registry()),
        dcc.Store(id="ctx-store", storage_type="session"),
        dcc.Store(id="data-store", storage_type="memory"),
        dcc.Store(id="agg-store", storage_type="memory"),
        dcc.Store(id="scenario-store", storage_type="memory"),
        dcc.Store(id="conditional-store", storage_type="memory"),
        dcc.Store(id="scenario-tax-selected-agg-store", storage_type="memory"),
        dcc.Store(id="agg-scenario-store", storage_type="memory"),
        dcc.Store(id="estimation-result-store", storage_type="memory"),
        dcc.Store(id="dataset-build-result-store", storage_type="memory"),
        dcc.Store(id="structural-store", storage_type="memory"),
        dcc.Store(
            id="estimation-config-store",
            data={**_baseline_config_payload(), "valid": True, "errors": []},
            storage_type="memory",
        ),
        dcc.Store(id="estimation-profile-revision", data=0, storage_type="memory"),
        dcc.Store(id="estimation-suite-progress-store", data={}, storage_type="memory"),
        sidebar(),
        html.Main(
            [
                topbar(),
                html.Div(id="selection-banner", className="selection-banner"),
                html.Div(
                    [
                        html.Div(forecast_page(), id="page-forecast"),
                        html.Div(
                            aggregate_page(),
                            id="page-aggregate",
                            style={"display": "none"},
                        ),
                        html.Div(
                            scenario_page(),
                            id="page-scenarios",
                            style={"display": "none"},
                        ),
                        html.Div(
                            structural_page(),
                            id="page-structural",
                            style={"display": "none"},
                        ),
                        html.Div(
                            estimation_page(),
                            id="page-estimation",
                            style={"display": "none"},
                        ),
                    ],
                    className="page-container",
                ),
            ],
            className="main-shell",
        ),
    ],
    className="app-shell",
)


# ---------------------------------------------------------------------------
# Registry and global context callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("registry-store", "data"),
    Input("refresh-registry", "n_clicks"),
    Input("estimation-result-store", "data"),
    prevent_initial_call=True,
)
def refresh_registry(_: int, __: dict | None) -> dict:
    return _scan_registry()


@callback(Output("registry-status", "children"), Input("registry-store", "data"))
def registry_status(store: dict | None):
    if not store:
        return "Registry unavailable"
    cls = "status-dot ok" if store.get("ok") else "status-dot error"
    unexpected = store.get("unexpected") or []
    children = [html.Span(className=cls), html.Span(str(store.get("message", "")))]
    if unexpected:
        children.append(
            html.Span(
                f" · {len(unexpected)} ignored folder{'s' if len(unexpected) != 1 else ''}",
                title="\n".join(map(str, unexpected)),
            )
        )
    return html.Div(children)


@callback(
    Output("vintage-select", "options"),
    Output("vintage-select", "value"),
    Input("registry-store", "data"),
    State("vintage-select", "value"),
)
def vintage_options(_: dict | None, current: str | None):
    forecasts = list_forecasts(REGISTRY_PATH, valid_only=True, present_only=True)
    if forecasts.empty:
        return [], None
    vintages = sorted(forecasts["vintage"].astype(str).unique(), reverse=True)
    options = [{"label": value, "value": value} for value in vintages]
    return options, current if current in vintages else vintages[0]


@callback(
    Output("model-select", "options"),
    Output("model-select", "value"),
    Input("vintage-select", "value"),
    Input("registry-store", "data"),
    State("model-select", "value"),
)
def model_options(vintage: str | None, _: dict | None, current: str | None):
    if not vintage:
        return [], None
    forecasts = list_forecasts(
        REGISTRY_PATH, vintage=vintage, valid_only=True, present_only=True
    )
    if forecasts.empty:
        return [], None
    models = sorted(forecasts["model_id"].astype(str).unique())
    options = []
    for model_id in models:
        try:
            label = model_spec(model_id).label
        except Exception:
            label = model_id.replace("_", " ").title()
        options.append({"label": label, "value": model_id})
    values = [item["value"] for item in options]
    return options, current if current in values else values[0]


@callback(
    Output("forecast-select", "options"),
    Output("forecast-select", "value"),
    Input("vintage-select", "value"),
    Input("model-select", "value"),
    Input("registry-store", "data"),
    State("forecast-select", "value"),
)
def forecast_options(
    vintage: str | None,
    model_id: str | None,
    _: dict | None,
    current: str | None,
):
    if not vintage or not model_id:
        return [], None
    frame = list_forecasts(
        REGISTRY_PATH,
        model_id=model_id,
        vintage=vintage,
        valid_only=True,
        present_only=True,
    )
    names = sorted(frame["forecast_name"].astype(str).unique()) if not frame.empty else []
    options = [{"label": name.replace("_", " ").title(), "value": name} for name in names]
    preferred = "unconditional" if "unconditional" in names else (names[0] if names else None)
    return options, current if current in names else preferred


@callback(
    Output("run-select", "options"),
    Output("run-select", "value"),
    Input("vintage-select", "value"),
    Input("model-select", "value"),
    Input("forecast-select", "value"),
    Input("registry-store", "data"),
    State("run-select", "value"),
)
def run_options(
    vintage: str | None,
    model_id: str | None,
    forecast_name: str | None,
    _: dict | None,
    current: str | None,
):
    if not all([vintage, model_id, forecast_name]):
        return [], None
    forecasts = list_forecasts(
        REGISTRY_PATH,
        model_id=model_id,
        vintage=vintage,
        forecast_name=forecast_name,
        valid_only=True,
        present_only=True,
    )
    runs = list_runs(
        REGISTRY_PATH,
        model_id=model_id,
        vintage=vintage,
        status="complete",
        present_only=True,
    )
    if forecasts.empty or runs.empty:
        return [], None
    merged = forecasts.merge(
        runs[["run_id", "promoted", "created_at_utc", "missing_data_method", "code_version"]],
        on="run_id",
        how="inner",
    )
    if merged.empty:
        return [], None
    merged = merged.sort_values(["promoted", "created_at_utc", "run_id"], ascending=[False, False, True])
    options = []
    for _, row in merged.iterrows():
        tags = []
        if bool(row.get("promoted")):
            tags.append("PROMOTED")
        if not bool(row.get("has_display")):
            tags.append("display pending")
        detail = " · ".join(tags)
        label = f"{_short_run(row['run_id'])}{' · ' + detail if detail else ''}"
        options.append({"label": label, "value": str(row["run_id"])})
    values = [item["value"] for item in options]
    promoted = merged.loc[merged["promoted"] == 1, "run_id"].astype(str).tolist()
    default = promoted[0] if promoted else values[0]
    return options, current if current in values else default


@callback(
    Output("ctx-store", "data"),
    Input("vintage-select", "value"),
    Input("model-select", "value"),
    Input("run-select", "value"),
    Input("forecast-select", "value"),
    State("ctx-store", "data"),
)
def update_context(
    vintage: str | None,
    model_id: str | None,
    run_id: str | None,
    forecast_name: str | None,
    previous: dict | None,
):
    previous = dict(previous or {})
    previous.update(
        {
            "vintage": vintage,
            "model_id": model_id,
            "run_id": run_id,
            "forecast_name": forecast_name,
            "draw_mode": previous.get("draw_mode", "posterior"),
        }
    )
    return previous


@callback(
    Output("data-store", "data"),
    Output("selection-banner", "children"),
    Input("ctx-store", "data"),
    Input("registry-store", "data"),
)
def load_selected_display(context: dict | None, _: dict | None):
    if not context or not all(
        context.get(key) for key in ("vintage", "model_id", "run_id", "forecast_name")
    ):
        return None, html.Div("Select a saved run to load its display artefact.")

    forecasts = list_forecasts(
        REGISTRY_PATH,
        model_id=context["model_id"],
        vintage=context["vintage"],
        forecast_name=context["forecast_name"],
        valid_only=True,
        present_only=True,
    )
    selected = forecasts.loc[forecasts["run_id"].astype(str) == str(context["run_id"])]
    if selected.empty:
        return None, html.Div("Selected forecast store is no longer present in the registry.")

    row = selected.iloc[0]
    forecast_dir = Path(str(row["directory"]))
    display_path = forecast_dir / DISPLAY_FILENAME
    materialised = False
    try:
        if not display_path.is_file():
            if not AUTO_BUILD_DISPLAY:
                raise FileNotFoundError(
                    f"{display_path} is missing and ENERGY_BVAR_AUTO_BUILD_DISPLAY=0."
                )
            run_dir = forecast_dir.parent.parent
            build_component_display(
                run_dir,
                project_root=PROJECT_ROOT,
                forecast_name=context["forecast_name"],
            )
            materialised = True
        frame = load_display_artifact(display_path)
        # display_v1 files created before the component-HICP contract can be
        # perfectly valid for the raw BVAR forecast while still lacking HICP
        # Level / YoY.  Rebuild them once from the saved forecast; the builder
        # materialises hicp_draws.npz without re-estimating the BVAR.
        fan_mask = frame["record_type"].astype(str) == "fan"
        metrics_present = set(frame.loc[fan_mask, "metric"].dropna().astype(str))
        if AUTO_BUILD_DISPLAY and not {"hicp_level", "yoy"}.issubset(metrics_present):
            run_dir = forecast_dir.parent.parent
            build_component_display(
                run_dir,
                project_root=PROJECT_ROOT,
                forecast_name=context["forecast_name"],
                overwrite=True,
            )
            materialised = True
            frame = load_display_artifact(display_path)

        meta = display_metadata(frame)
        fan_rows = int((frame["record_type"].astype(str) == "fan").sum())
        if fan_rows == 0:
            raise ValueError(
                f"{DISPLAY_FILENAME} loaded successfully but contains no forecast fan rows."
            )
        final_metrics = set(
            frame.loc[frame["record_type"].astype(str) == "fan", "metric"]
            .dropna().astype(str)
        )
        if not {"hicp_level", "yoy"}.issubset(final_metrics):
            raise ValueError(
                "Component HICP post-processing is incomplete: expected both "
                "'hicp_level' and 'yoy' in display_v1."
            )
    except Exception as exc:
        return None, html.Div(
            [html.Strong("Display load failed: "), html.Span(str(exc))],
            className="banner-error",
        )

    label = meta.get("model_label") or context["model_id"]
    method = meta.get("missing_data_method") or "—"
    approximation_used = meta.get("missing_data_approximation_used")
    if str(method).lower() == "dk":
        missing_note = "exact augmentation"
    elif str(method).lower() == "linear":
        if str(approximation_used).lower() in {"true", "1"}:
            missing_note = "interpolation used"
        elif str(approximation_used).lower() in {"false", "0"}:
            missing_note = "interpolation not used"
        else:
            missing_note = "linear method"
    else:
        missing_note = ""
    built_note = " · display/HICP materialised now" if materialised else ""
    banner = html.Div(
        [
            html.Strong(str(label)),
            html.Span(f" · vintage {context['vintage']}"),
            html.Span(f" · run {_short_run(context['run_id'])}"),
            html.Span(f" · missing data: {method}{' (' + missing_note + ')' if missing_note else ''}"),
            html.Span(" · component HICP: ready"),
            html.Span(built_note),
        ]
    )
    return {
        "frame_json": _json_frame(frame),
        "meta": meta,
        "context": context,
        "display_path": str(display_path),
    }, banner


# ---------------------------------------------------------------------------
# Estimation-page callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("est-model-select", "value"),
    Input("url", "pathname"),
    State("model-select", "value"),
    State("est-model-select", "value"),
)
def sync_estimation_model(pathname, global_model, current):
    if (pathname or "/forecast") != "/estimation":
        return current or "gas"
    if global_model in ENERGY_SUITE_MODEL_IDS:
        return global_model
    return current if current in ENERGY_SUITE_MODEL_IDS else "gas"


@callback(
    Output("est-vintage-select", "options"),
    Output("est-vintage-select", "value"),
    Input("est-model-select", "value"),
    Input("dataset-build-result-store", "data"),
    State("vintage-select", "value"),
    State("est-vintage-select", "value"),
)
def estimation_vintage_options(model_id, build_result, global_vintage, current):
    if not model_id:
        return [], None
    try:
        vintages = available_vintages(model_id, project_root=PROJECT_ROOT)
    except Exception:
        return [], None
    vintages = sorted(map(str, vintages), reverse=True)
    options = [{"label": value, "value": value} for value in vintages]

    built_vintage = (
        str((build_result or {}).get("build_vintage"))
        if (build_result or {}).get("ok")
        else None
    )
    if built_vintage in vintages:
        value = built_vintage
    elif global_vintage in vintages:
        value = global_vintage
    elif current in vintages:
        value = current
    else:
        value = vintages[0] if vintages else None
    return options, value




@callback(
    Output("dataset-build-environment", "children"),
    Input("url", "pathname"),
    Input("dataset-build-raw-path", "value"),
    Input("dataset-build-result-store", "data"),
)
def dataset_build_environment(pathname, raw_path, build_result):
    if (pathname or "") != "/estimation":
        raise PreventUpdate
    try:
        info = inspect_dataset_build_environment(
            PROJECT_ROOT,
            raw_path=(raw_path or "").strip() or None,
        )
        lock = dict(info.get("lock", {}) or {})
        pieces = [
            html.Strong("Workbook ready"),
            html.Span(f" · {info['raw_relative']}"),
            html.Span(f" · saved {info['raw_modified_at']}"),
            html.Span(f" · builder {info['builder_relative']}"),
            html.Span(f" · automatic vintage {info['default_build_vintage']}"),
        ]
        if lock.get("active"):
            pieces.append(
                html.Span(
                    f" · BUILD RUNNING ({lock.get('build_vintage', 'unknown vintage')})"
                )
            )
        elif (build_result or {}).get("ok"):
            pieces.append(
                html.Span(
                    f" · last dashboard build {(build_result or {}).get('build_vintage')} complete"
                )
            )
        return html.Div(pieces)
    except Exception as exc:
        return html.Div(
            [
                html.Strong("Dataset preflight: "),
                html.Span(str(exc)),
            ],
            className="estimation-error-text",
        )


@callback(
    output=Output("dataset-build-result-store", "data"),
    inputs=[Input("dataset-build-run", "n_clicks")],
    state=[
        State("dataset-build-raw-path", "value"),
        State("dataset-build-vintage", "value"),
        State("dataset-build-overwrite", "value"),
    ],
    background=True,
    running=[
        (Output("dataset-build-run", "disabled"), True, False),
        (Output("dataset-build-cancel", "disabled"), False, True),
        (Output("dataset-build-raw-path", "disabled"), True, False),
        (Output("dataset-build-vintage", "disabled"), True, False),
    ],
    cancel=[Input("dataset-build-cancel", "n_clicks")],
    progress=[
        Output("dataset-build-progress", "value"),
        Output("dataset-build-phase", "children"),
        Output("dataset-build-progress-detail", "children"),
    ],
    progress_default=(
        0,
        "Idle",
        "Save the refreshed workbook, then build the processed vintage.",
    ),
    prevent_initial_call=True,
)
def build_processed_datasets(
    set_progress,
    n_clicks,
    raw_path,
    build_vintage,
    overwrite_values,
):
    if not n_clicks:
        raise PreventUpdate

    started = time.monotonic()

    # Do not rebuild data underneath an active sampler/aggregate process.
    if ESTIMATION_LOCK_PATH.exists():
        _clear_stale_estimation_lock()
    active_estimation = _read_estimation_lock()
    if active_estimation:
        pid = int(active_estimation.get("pid", -1) or -1)
        if pid > 0 and _pid_is_alive(pid):
            message = (
                "Cannot rebuild processed data while production estimation is running "
                f"({active_estimation.get('model_id', 'unknown model')})."
            )
            set_progress((0, "Build blocked", message))
            return {
                "ok": False,
                "status": "blocked",
                "message": message,
                "elapsed_seconds": float(time.monotonic() - started),
            }

    def push(percent, phase, detail):
        set_progress((int(percent), str(phase), str(detail)))

    try:
        result = run_dataset_build(
            PROJECT_ROOT,
            raw_path=(raw_path or "").strip() or None,
            build_vintage=(build_vintage or "").strip() or None,
            overwrite="overwrite" in set(overwrite_values or []),
            require_hicp=True,
            hicp_discovery=False,
            progress_callback=push,
        )
        return result
    except Exception as exc:
        message = str(exc)
        push(0, "Dataset build failed", message.splitlines()[0])
        return {
            "ok": False,
            "status": "failed",
            "message": message,
            "log": message,
            "elapsed_seconds": float(time.monotonic() - started),
        }


@callback(
    Output("dataset-build-result-banner", "children"),
    Output("dataset-build-log", "children"),
    Input("dataset-build-result-store", "data"),
)
def render_dataset_build_result(store):
    if not store:
        return "", "No dataset build has run in this dashboard session."

    ok = bool(store.get("ok"))
    if ok:
        vintage = str(store.get("build_vintage", ""))
        elapsed = float(store.get("elapsed_seconds", 0.0) or 0.0)
        model_count = len(dict(store.get("model_files", {}) or {}))
        hicp_count = len(dict(store.get("hicp_files", {}) or {}))
        banner = html.Div(
            [
                html.Strong("Processed vintage ready: "),
                html.Span(vintage),
                html.Span(
                    f" · {model_count} canonical dataset files"
                    f" · {hicp_count} HICP aggregation sidecars"
                    f" · {elapsed:.1f}s"
                ),
                html.Span(
                    f" · workbook SHA256 {str(store.get('raw_sha256', ''))[:12]}…"
                ),
            ],
            className="estimation-success-banner",
        )
        log = str(store.get("log", "")).strip()
        if not log:
            log = (
                f"Builder {store.get('builder_version', '')}\n"
                f"Workbook: {store.get('raw_path', '')}\n"
                f"Output: {store.get('output_dir', '')}"
            )
        return banner, log

    message = str(store.get("message", "Dataset build failed."))
    banner = html.Div(
        [html.Strong("Dataset build failed: "), html.Span(message.splitlines()[0])],
        className="estimation-error-banner",
    )
    return banner, str(store.get("log") or message)


@callback(
    Output("est-profile-select", "options"),
    Input("estimation-profile-revision", "data"),
)
def estimation_profile_options(_revision):
    return _saved_profile_options()


@callback(
    Output("est-lambda1", "value"),
    Output("est-lambda2", "value"),
    Output("est-lambda3", "value"),
    Output("est-lambda4", "value"),
    Output("est-a-prior-var", "value"),
    Output("est-phi-mean", "value"),
    Output("est-phi-df", "value"),
    Output("est-h0-var", "value"),
    Output("est-outlier-every", "value"),
    Output("est-outlier-prior-obs", "value"),
    Output("est-reps", "value"),
    Output("est-burn", "value"),
    Output("est-thin", "value"),
    Output("est-seed", "value"),
    Output("est-max-stability", "value"),
    Output("est-outlier-grid-min", "value"),
    Output("est-outlier-grid-max", "value"),
    Output("est-outlier-grid-step", "value"),
    Output("est-ksc-offset-scale", "value"),
    Output("est-ksc-offset-floor", "value"),
    Output("est-progress-every", "value"),
    Output("est-dk-mode", "value"),
    Output("est-dk-level-gate", "value"),
    Output("est-dk-difference-gate", "value"),
    Output("est-dk-cat-level-gate", "value"),
    Output("est-dk-cat-difference-gate", "value"),
    Output("est-config-scope", "value"),
    Output("est-per-model-table", "data"),
    Output("est-profile-message", "children"),
    Input("est-profile-select", "value"),
    prevent_initial_call=False,
)
def load_estimation_profile(profile_value):
    baseline = _baseline_config_payload()
    if profile_value in (None, "paper_baseline"):
        payload = baseline
        message = "Canonical project baseline loaded · controls are locked until you choose Custom or a saved profile."
    elif profile_value == "custom":
        # Starting a new custom experiment from the documented baseline is
        # explicit and reproducible.
        payload = {**baseline, "profile_mode": "custom"}
        message = "Custom profile · edit the parameters below."
    else:
        try:
            payload = _load_saved_profile(profile_value)
            payload = {
                **baseline,
                **payload,
                "prior": {**baseline["prior"], **payload.get("prior", {})},
                "sampler": {**baseline["sampler"], **payload.get("sampler", {})},
                "per_model": payload.get("per_model") or baseline["per_model"],
                "profile_mode": profile_value,
            }
            message = f"Loaded saved profile: {payload.get('name', profile_value)}"
        except Exception as exc:
            payload = baseline
            message = f"Could not load saved profile: {exc}"

    prior = payload["prior"]
    sampler = payload["sampler"]
    return (
        prior["lambda1"],
        prior["lambda2"],
        prior["lambda3"],
        prior["lambda4"],
        prior["a_prior_var"],
        prior["phi_prior_mean"],
        prior["phi_prior_df"],
        prior["h0_var"],
        prior["outlier_every"],
        prior["outlier_prior_observations"],
        sampler["reps"],
        sampler["burn"],
        sampler["thin"],
        sampler["seed"],
        sampler["max_stability_tries"],
        prior["outlier_grid_min"],
        prior["outlier_grid_max"],
        prior["outlier_grid_step"],
        prior["ksc_offset_scale"],
        prior["ksc_offset_floor"],
        sampler["progress_every"],
        sampler["dk_projection_mode"],
        sampler["dk_level_relative_gate"],
        sampler["dk_difference_relative_gate"],
        sampler["dk_catastrophic_level_relative_gate"],
        sampler["dk_catastrophic_difference_relative_gate"],
        payload.get("scope", "selected_only"),
        payload.get("per_model", baseline["per_model"]),
        message,
    )


@callback(
    Output("est-lambda1", "disabled"),
    Output("est-lambda2", "disabled"),
    Output("est-lambda3", "disabled"),
    Output("est-lambda4", "disabled"),
    Output("est-a-prior-var", "disabled"),
    Output("est-phi-mean", "disabled"),
    Output("est-phi-df", "disabled"),
    Output("est-h0-var", "disabled"),
    Output("est-outlier-every", "disabled"),
    Output("est-outlier-prior-obs", "disabled"),
    Output("est-reps", "disabled"),
    Output("est-burn", "disabled"),
    Output("est-thin", "disabled"),
    Output("est-seed", "disabled"),
    Output("est-max-stability", "disabled"),
    Output("est-custom-p", "disabled"),
    Output("est-exog-prior-scale", "disabled"),
    Output("est-reference-month", "disabled"),
    Output("est-outlier-grid-min", "disabled"),
    Output("est-outlier-grid-max", "disabled"),
    Output("est-outlier-grid-step", "disabled"),
    Output("est-ksc-offset-scale", "disabled"),
    Output("est-ksc-offset-floor", "disabled"),
    Output("est-progress-every", "disabled"),
    Output("est-dk-mode", "disabled"),
    Output("est-dk-level-gate", "disabled"),
    Output("est-dk-difference-gate", "disabled"),
    Output("est-dk-cat-level-gate", "disabled"),
    Output("est-dk-cat-difference-gate", "disabled"),
    Output("est-config-scope", "options"),
    Output("est-profile-save", "disabled"),
    Input("est-profile-select", "value"),
)
def estimation_profile_editability(profile_value):
    locked = profile_value in (None, "paper_baseline")
    options = [
        {"label": "Selected model only", "value": "selected_only", "disabled": locked},
        {"label": "All 7 models", "value": "all_7", "disabled": locked},
        {"label": "Per-model overrides", "value": "per_model", "disabled": locked},
    ]
    return (*([locked] * 29), options, locked)


@callback(
    Output("est-custom-p", "value"),
    Output("est-exog-prior-scale", "value"),
    Output("est-reference-month", "value"),
    Input("est-model-select", "value"),
    State("est-per-model-table", "data"),
)
def selected_model_structure_defaults(model_id, rows):
    if not model_id:
        return 12, 10.0, 1
    spec = model_spec(model_id)
    row_p = spec.p
    for row in rows or []:
        if str(row.get("model_id")) == str(model_id):
            row_p = row.get("p", spec.p)
            break
    return (
        int(row_p),
        10.0 if spec.exog_prior_scale is None else float(spec.exog_prior_scale),
        1 if spec.reference_month is None else int(spec.reference_month),
    )


@callback(
    Output("est-per-model-wrap", "style"),
    Input("est-config-scope", "value"),
)
def estimation_per_model_visibility(scope):
    return {"display": "block"} if scope == "per_model" else {"display": "none"}


@callback(
    Output("estimation-config-store", "data"),
    Output("est-retained-draws", "children"),
    Input("est-profile-select", "value"),
    Input("est-config-scope", "value"),
    Input("est-lambda1", "value"),
    Input("est-lambda2", "value"),
    Input("est-lambda3", "value"),
    Input("est-lambda4", "value"),
    Input("est-a-prior-var", "value"),
    Input("est-phi-mean", "value"),
    Input("est-phi-df", "value"),
    Input("est-h0-var", "value"),
    Input("est-outlier-every", "value"),
    Input("est-outlier-prior-obs", "value"),
    Input("est-reps", "value"),
    Input("est-burn", "value"),
    Input("est-thin", "value"),
    Input("est-seed", "value"),
    Input("est-max-stability", "value"),
    Input("est-custom-p", "value"),
    Input("est-exog-prior-scale", "value"),
    Input("est-reference-month", "value"),
    Input("est-outlier-grid-min", "value"),
    Input("est-outlier-grid-max", "value"),
    Input("est-outlier-grid-step", "value"),
    Input("est-ksc-offset-scale", "value"),
    Input("est-ksc-offset-floor", "value"),
    Input("est-progress-every", "value"),
    Input("est-dk-mode", "value"),
    Input("est-dk-level-gate", "value"),
    Input("est-dk-difference-gate", "value"),
    Input("est-dk-cat-level-gate", "value"),
    Input("est-dk-cat-difference-gate", "value"),
    Input("est-per-model-table", "data"),
)
def compile_estimation_configuration(
    profile_value,
    scope,
    lambda1,
    lambda2,
    lambda3,
    lambda4,
    a_prior_var,
    phi_mean,
    phi_df,
    h0_var,
    outlier_every,
    outlier_prior_obs,
    reps,
    burn,
    thin,
    seed,
    max_stability,
    custom_p,
    exog_prior_scale,
    reference_month,
    outlier_grid_min,
    outlier_grid_max,
    outlier_grid_step,
    ksc_offset_scale,
    ksc_offset_floor,
    progress_every,
    dk_mode,
    dk_level_gate,
    dk_difference_gate,
    dk_cat_level_gate,
    dk_cat_difference_gate,
    per_model_rows,
):
    profile_mode = "paper_baseline" if profile_value in (None, "paper_baseline") else str(profile_value)
    payload = {
        "version": ESTIMATION_PROFILE_VERSION,
        "profile_mode": profile_mode,
        "scope": scope or "selected_only",
        "prior": {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "lambda4": lambda4,
            "a_prior_var": a_prior_var,
            "phi_prior_mean": phi_mean,
            "phi_prior_df": phi_df,
            "h0_var": h0_var,
            "outlier_every": outlier_every,
            "outlier_prior_observations": outlier_prior_obs,
            "outlier_grid_min": outlier_grid_min,
            "outlier_grid_max": outlier_grid_max,
            "outlier_grid_step": outlier_grid_step,
            "ksc_offset_scale": ksc_offset_scale,
            "ksc_offset_floor": ksc_offset_floor,
        },
        "sampler": {
            "reps": reps,
            "burn": burn,
            "thin": thin,
            "seed": seed,
            "max_stability_tries": max_stability,
            "progress_every": progress_every,
            "dk_projection_mode": dk_mode,
            "dk_level_relative_gate": dk_level_gate,
            "dk_difference_relative_gate": dk_difference_gate,
            "dk_catastrophic_level_relative_gate": dk_cat_level_gate,
            "dk_catastrophic_difference_relative_gate": dk_cat_difference_gate,
        },
        "selected_structure": {
            "p": custom_p,
            "exog_prior_scale": exog_prior_scale,
            "reference_month": reference_month,
        },
        "per_model": per_model_rows or _baseline_per_model_rows(),
    }
    errors = []
    try:
        _prior_from_mapping(payload["prior"])
    except Exception as exc:
        errors.append(f"Prior: {exc}")
    try:
        sampler = _sampler_from_mapping(payload["sampler"])
        retained = len(range(sampler.burn, sampler.reps, sampler.thin))
    except Exception as exc:
        sampler = None
        retained = None
        errors.append(f"Sampler: {exc}")
    try:
        if payload["selected_structure"]["p"] is not None:
            if _safe_int(payload["selected_structure"]["p"], "lag order") < 1:
                raise ValueError("lag order must be positive.")
    except Exception as exc:
        errors.append(f"Structure: {exc}")

    if payload["scope"] == "per_model":
        for row in payload["per_model"]:
            model_id = str(row.get("model_id", "unknown"))
            try:
                if _safe_int(row.get("p"), f"{model_id} p") < 1:
                    raise ValueError("p must be positive")
                row_prior = dict(payload["prior"])
                for key in ("lambda1", "lambda2", "lambda3", "lambda4"):
                    row_prior[key] = row.get(key)
                _prior_from_mapping(row_prior)
                row_sampler = dict(payload["sampler"])
                for key in ("reps", "burn", "thin", "seed"):
                    row_sampler[key] = row.get(key)
                _sampler_from_mapping(row_sampler)
            except Exception as exc:
                errors.append(f"{model_id}: {exc}")

    payload["valid"] = not errors
    payload["errors"] = errors
    retained_text = (
        f"Retained posterior draws: {retained:,}"
        if retained is not None
        else "Retained posterior draws: invalid sampler settings"
    )
    return payload, retained_text


@callback(
    Output("estimation-profile-revision", "data"),
    Output("est-profile-message", "children", allow_duplicate=True),
    Input("est-profile-save", "n_clicks"),
    State("est-profile-name", "value"),
    State("estimation-config-store", "data"),
    State("estimation-profile-revision", "data"),
    prevent_initial_call=True,
)
def save_estimation_profile(n_clicks, name, config_store, revision):
    if not n_clicks:
        raise PreventUpdate
    if not config_store or not config_store.get("valid"):
        return revision or 0, "Cannot save an invalid configuration."
    try:
        slug = _profile_slug(name)
        payload = dict(config_store)
        payload.update(
            {
                "version": ESTIMATION_PROFILE_VERSION,
                "name": str(name).strip(),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        path = ESTIMATION_PROFILE_DIR / f"{slug}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return int(revision or 0) + 1, f"Saved profile: {path.relative_to(PROJECT_ROOT)}"
    except Exception as exc:
        return revision or 0, f"Could not save profile: {exc}"


@callback(
    Output("est-frequency", "children"),
    Output("est-calendar", "children"),
    Output("est-lags", "children"),
    Output("est-target", "children"),
    Output("est-last-date", "children"),
    Output("est-dataset", "children"),
    Output("est-run-id", "children"),
    Output("est-existing-state", "children"),
    Output("estimation-preview-banner", "children"),
    Output("estimation-suite-banner", "children"),
    Output("est-config-status", "children"),
    Output("est-config-hash", "children"),
    Output("est-config-run", "children"),
    Input("est-model-select", "value"),
    Input("est-vintage-select", "value"),
    Input("estimation-config-store", "data"),
)
def estimation_preview(model_id, vintage, config_store):
    if not model_id or not vintage:
        return (
            "—", "", "—", "", "—", "", "—", "",
            "Select a processed data vintage.",
            "Seven-model suite readiness will appear here.",
            "Waiting for a valid selection.",
            "—",
            "—",
        )
    if not config_store or not config_store.get("valid"):
        errors = [] if not config_store else config_store.get("errors", [])
        error_text = " · ".join(map(str, errors)) or "Invalid estimation settings."
        return (
            "—", "", "—", "", "—", "", "—", "",
            html.Div([html.Strong("Configuration invalid: "), html.Span(error_text)], className="estimation-error-text"),
            "Correct the estimation configuration before running the suite.",
            html.Div(error_text, className="estimation-error-text"),
            "invalid",
            "invalid",
        )
    try:
        prior, sampler, spec_overrides = _model_estimation_configs(
            model_id, model_id, config_store
        )
        spec = model_spec(model_id, **spec_overrides)
        panel = build_panel(
            model_id,
            vintage,
            project_root=PROJECT_ROOT,
            spec_overrides=spec_overrides,
        )
        state = _planned_run_state(
            model_id,
            vintage,
            selected_model_id=model_id,
            config_store=config_store,
        )
        last_date = pd.Timestamp(panel.levels.index.max()).date().isoformat()
        target = str(panel.target).replace("_", " ").title()
        run_id = state["run_id"]
        metadata = state["metadata"]
        if state["has_draws"] and state["has_forecast"]:
            existing = "complete · Gibbs will not rerun"
        elif state["directory"].exists():
            missing = []
            if not state["has_draws"]:
                missing.append("full posterior")
            if not state["has_forecast"]:
                missing.append("forecast")
            existing = "existing run · missing " + ", ".join(missing)
        else:
            existing = "new deterministic run"

        custom = _config_differs_from_baseline(model_id, model_id, config_store)
        profile_label = (
            "Canonical project baseline"
            if not custom
            else "Custom configuration"
            if config_store.get("profile_mode") == "custom"
            else str(config_store.get("profile_mode")).replace("saved:", "Saved profile · ")
        )
        config_status = html.Div(
            [
                html.Strong(profile_label),
                html.Span(
                    f" · retained draws {len(range(sampler.burn, sampler.reps, sampler.thin)):,}"
                ),
                html.Span(
                    f" · scope {str(config_store.get('scope', 'selected_only')).replace('_', ' ')}"
                ),
            ]
        )
        method = str(spec.missing_data_method)
        banner = html.Div(
            [
                html.Strong(spec.label),
                html.Span(f" · vintage {vintage}"),
                html.Span(f" · dataset {panel.dataset_path.name}"),
                html.Span(f" · p={panel.p}"),
                html.Span(f" · missing data: {method}"),
                html.Span(" · full posterior persistence required"),
            ]
        )

        coverage = vintage_coverage(
            models=ENERGY_SUITE_MODEL_IDS,
            project_root=PROJECT_ROOT,
        )
        if vintage in coverage.index:
            row = coverage.loc[vintage]
            missing_models = [
                model_spec(name).label
                for name in row.index[~row.to_numpy(dtype=bool)]
            ]
            ready = int(row.to_numpy(dtype=bool).sum())
        else:
            missing_models = [model_spec(name).label for name in ENERGY_SUITE_MODEL_IDS]
            ready = 0

        if ready == len(ENERGY_SUITE_MODEL_IDS):
            suite_banner = html.Div(
                [
                    html.Strong("Seven-model suite ready"),
                    html.Span(f" · vintage {vintage} has all 7 processed datasets"),
                    html.Span(" · execution is sequential"),
                    html.Span(
                        " · run identity is evaluated under the active estimation profile"
                    ),
                ]
            )
        else:
            suite_banner = html.Div(
                [
                    html.Strong(f"Seven-model suite incomplete ({ready}/7)"),
                    html.Span(
                        " · missing processed dataset(s): "
                        + ", ".join(missing_models)
                    ),
                ],
                className="estimation-error-text",
            )

        return (
            spec.frequency.title(),
            "monthly" if spec.frequency == "monthly" else "Monday weekly",
            str(panel.p),
            target,
            last_date,
            panel.dataset_path.name,
            _short_run(run_id),
            existing,
            banner,
            suite_banner,
            config_status,
            _short_run(str(metadata.get("config_hash", ""))),
            _short_run(run_id),
        )
    except Exception as exc:
        return (
            "—", "", "—", "", "—", "", "—", "",
            html.Div(
                [html.Strong("Preview failed: "), html.Span(str(exc))],
                className="estimation-error-text",
            ),
            html.Div(
                [html.Strong("Suite readiness unavailable: "), html.Span(str(exc))],
                className="estimation-error-text",
            ),
            html.Div(str(exc), className="estimation-error-text"),
            "error",
            "error",
        )


def _estimation_result_payload(
    *,
    ok: bool,
    status: str,
    model_id: str,
    vintage: str,
    run_id: str,
    message: str,
    elapsed: float,
    directory: Path,
    posterior_draws: int | None = None,
    forecast_draws: int | None = None,
    promoted: bool = False,
    reused: bool = False,
) -> dict:
    return {
        "ok": bool(ok),
        "status": str(status),
        "model_id": str(model_id),
        "vintage": str(vintage),
        "run_id": str(run_id),
        "message": str(message),
        "elapsed_seconds": float(elapsed),
        "directory": str(directory),
        "posterior_draws": None if posterior_draws is None else int(posterior_draws),
        "forecast_draws": None if forecast_draws is None else int(forecast_draws),
        "promoted": bool(promoted),
        "reused": bool(reused),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }



def _canonical_energy_aggregate_run_ids(run_ids: dict[str, str]) -> dict[str, str]:
    """Normalise component run IDs to the aggregation keys used by notebook 10.

    ``run_aggregate`` accepts canonical model IDs as well, but using aggregate
    keys here makes exact provenance comparison against saved aggregate metadata
    transparent (petrol/diesel rather than their on-disk model IDs).
    """
    source = {str(k): str(v) for k, v in dict(run_ids or {}).items()}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in ENERGY_SUITE_MODEL_IDS:
        spec = model_spec(name)
        run_id = (
            source.get(str(spec.model_id))
            or source.get(str(spec.aggregate_key))
            or source.get(str(name))
        )
        if not run_id:
            missing.append(str(spec.model_id))
            continue
        resolved[str(spec.aggregate_key)] = str(run_id)
    if missing:
        raise ValueError(
            "Cannot build HICP Energy aggregate: missing component run IDs "
            f"for {missing}."
        )
    return resolved


def _aggregate_run_ids_from_metadata(metadata: dict) -> dict[str, str]:
    """Extract aggregate-key -> component run_id from saved forecast paths."""
    out: dict[str, str] = {}
    for key, raw in dict(metadata.get("component_forecast_stores", {}) or {}).items():
        path = Path(str(raw))
        # <run>/forecasts/<forecast_name>
        try:
            run_id = path.parents[1].name
        except IndexError:
            continue
        if run_id:
            out[str(key)] = str(run_id)
    return out


def _matching_saved_energy_aggregate(
    vintage: str,
    run_ids: dict[str, str],
) -> tuple[str, Path] | None:
    """Find an already-complete baseline aggregate for exactly these seven runs.

    This keeps "Estimate Energy suite" idempotent without relying on file
    modification times or merely on a shared vintage.
    """
    expected = _canonical_energy_aggregate_run_ids(run_ids)
    root = RESULTS_ROOT / "hicp_energy_aggregate" / str(vintage)
    if not root.is_dir():
        return None

    for directory in sorted((p for p in root.iterdir() if p.is_dir())):
        metadata_path = directory / "metadata.json"
        draws_path = directory / "aggregate_draws.npz"
        if not metadata_path.is_file() or not draws_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if str(metadata.get("forecast_name", "unconditional")) != ENERGY_AGGREGATE_FORECAST_NAME:
            continue
        if bool(metadata.get("scenario_active", False)):
            continue
        if int(metadata.get("n_aggregate_draws_requested", -1)) != ENERGY_AGGREGATE_DRAWS:
            continue
        if int(metadata.get("pairing_seed", -1)) != ENERGY_AGGREGATE_PAIRING_SEED:
            continue
        if str(metadata.get("weekly_tax_mode_requested", "strict")) != ENERGY_AGGREGATE_WEEKLY_TAX_MODE:
            continue
        if _aggregate_run_ids_from_metadata(metadata) != expected:
            continue

        aggregate_run_id = str(metadata.get("aggregate_run_id") or directory.name)
        return aggregate_run_id, directory
    return None


def _build_or_reuse_energy_aggregate(
    vintage: str,
    run_ids: dict[str, str],
    *,
    promote_after: bool,
) -> dict:
    """Materialise the production HICP Energy aggregate for exact component runs."""
    exact_run_ids = _canonical_energy_aggregate_run_ids(run_ids)
    existing = _matching_saved_energy_aggregate(vintage, exact_run_ids)

    if existing is not None:
        aggregate_run_id, directory = existing
        build_aggregate_display(
            directory,
            project_root=PROJECT_ROOT,
            overwrite=True,
        )
        scan_results(results_root=RESULTS_ROOT, registry_path=REGISTRY_PATH)
        if promote_after:
            promote_aggregate(REGISTRY_PATH, str(vintage), aggregate_run_id)
        metadata = _aggregate_metadata(directory)
        return {
            "aggregate_run_id": aggregate_run_id,
            "directory": str(directory),
            "n_aggregate_draws": int(
                metadata.get("n_aggregate_draws_effective", 0) or 0
            ),
            "reused": True,
            "promoted": bool(promote_after),
        }

    outcome = run_aggregate(
        str(vintage),
        project_root=PROJECT_ROOT,
        results_root=RESULTS_ROOT,
        forecast_name=ENERGY_AGGREGATE_FORECAST_NAME,
        run_ids=exact_run_ids,
        n_aggregate_draws=ENERGY_AGGREGATE_DRAWS,
        pairing_seed=ENERGY_AGGREGATE_PAIRING_SEED,
        tax_scenarios=None,
        weekly_tax_mode=ENERGY_AGGREGATE_WEEKLY_TAX_MODE,
        persist=True,
        overwrite=False,
    )
    if outcome.directory is None:
        raise RuntimeError("run_aggregate returned no persisted output directory.")

    build_aggregate_display(
        outcome.directory,
        project_root=PROJECT_ROOT,
        overwrite=True,
    )
    scan_results(results_root=RESULTS_ROOT, registry_path=REGISTRY_PATH)
    if promote_after:
        promote_aggregate(
            REGISTRY_PATH,
            str(vintage),
            str(outcome.aggregate_run_id),
        )

    return {
        "aggregate_run_id": str(outcome.aggregate_run_id),
        "directory": str(outcome.directory),
        "n_aggregate_draws": int(outcome.n_aggregate_draws_effective),
        "reused": False,
        "promoted": bool(promote_after),
    }


def _energy_aggregate_result_payload(
    *,
    ok: bool,
    status: str,
    vintage: str,
    message: str,
    elapsed: float,
    aggregate: dict | None = None,
) -> dict:
    aggregate = dict(aggregate or {})
    return {
        "mode": "aggregate",
        "ok": bool(ok),
        "status": str(status),
        "vintage": str(vintage),
        "message": str(message),
        "elapsed_seconds": float(elapsed),
        "aggregate_run_id": str(aggregate.get("aggregate_run_id", "")),
        "directory": str(aggregate.get("directory", RESULTS_ROOT)),
        "n_aggregate_draws": aggregate.get("n_aggregate_draws"),
        "promoted": bool(aggregate.get("promoted", False)),
        "reused": bool(aggregate.get("reused", False)),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }



def _suite_progress_payload(
    vintage: str,
    statuses: dict[str, dict],
    *,
    current_model: str | None = None,
) -> dict:
    """Serializable live status for the seven sequential component runs."""
    models = []
    for model_id in ENERGY_SUITE_MODEL_IDS:
        item = dict(statuses.get(model_id, {}))
        item.setdefault("model_id", model_id)
        item.setdefault("label", model_spec(model_id).label)
        item.setdefault("status", "waiting")
        item.setdefault("run_id", "")
        item.setdefault("detail", "")
        item.setdefault("reused", False)
        item["current"] = model_id == current_model
        models.append(item)
    return {
        "mode": "suite",
        "vintage": str(vintage),
        "current_model": current_model,
        "models": models,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _estimation_suite_result_payload(
    *,
    ok: bool,
    status: str,
    vintage: str,
    message: str,
    elapsed: float,
    statuses: dict[str, dict],
    promoted: bool = False,
    aggregate: dict | None = None,
) -> dict:
    models = []
    for model_id in ENERGY_SUITE_MODEL_IDS:
        item = dict(statuses.get(model_id, {}))
        item.setdefault("model_id", model_id)
        item.setdefault("label", model_spec(model_id).label)
        item.setdefault("status", "waiting")
        item.setdefault("run_id", "")
        item.setdefault("reused", False)
        models.append(item)
    completed = sum(item.get("status") == "complete" for item in models)
    reused = sum(bool(item.get("reused")) for item in models if item.get("status") == "complete")
    return {
        "mode": "suite",
        "ok": bool(ok),
        "status": str(status),
        "vintage": str(vintage),
        "message": str(message),
        "elapsed_seconds": float(elapsed),
        "directory": str(RESULTS_ROOT),
        "promoted": bool(promoted),
        "models": models,
        "completed_count": int(completed),
        "reused_count": int(reused),
        "estimated_count": int(max(completed - reused, 0)),
        "aggregate": None if aggregate is None else dict(aggregate),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _push_estimation_progress(
    set_progress,
    percent: int,
    phase: str,
    detail: str,
    suite_payload: dict | None = None,
) -> None:
    set_progress(
        (
            int(max(0, min(100, percent))),
            str(phase),
            str(detail),
            {} if suite_payload is None else suite_payload,
        )
    )


def _initial_suite_statuses(
    vintage: str,
    selected_model_id: str,
    config_store: dict,
) -> dict[str, dict]:
    statuses: dict[str, dict] = {}
    for model_id in ENERGY_SUITE_MODEL_IDS:
        state = _planned_run_state(
            model_id,
            vintage,
            selected_model_id=selected_model_id,
            config_store=config_store,
        )
        complete = bool(state["has_draws"] and state["has_forecast"])
        statuses[model_id] = {
            "model_id": model_id,
            "label": model_spec(model_id).label,
            "status": "complete" if complete else "waiting",
            "run_id": state["run_id"],
            "detail": "complete run will be reused" if complete else "queued",
            "reused": complete,
        }
    return statuses


@callback(
    output=Output("estimation-result-store", "data"),
    inputs=[
        Input("estimation-run", "n_clicks"),
        Input("estimation-run-all", "n_clicks"),
        Input("estimation-run-aggregate", "n_clicks"),
    ],
    state=[
        State("est-model-select", "value"),
        State("est-vintage-select", "value"),
        State("estimation-promote", "value"),
        State("estimation-config-store", "data"),
    ],
    background=True,
    running=[
        (Output("estimation-run", "disabled"), True, False),
        (Output("estimation-run-all", "disabled"), True, False),
        (Output("estimation-run-aggregate", "disabled"), True, False),
        (Output("estimation-cancel", "disabled"), False, True),
        (Output("est-model-select", "disabled"), True, False),
        (Output("est-vintage-select", "disabled"), True, False),
    ],
    cancel=[Input("estimation-cancel", "n_clicks")],
    progress=[
        Output("estimation-progress", "value"),
        Output("estimation-phase", "children"),
        Output("estimation-progress-detail", "children"),
        Output("estimation-suite-progress-store", "data"),
    ],
    progress_default=(0, "Idle", "Select a model and vintage.", {}),
    prevent_initial_call=True,
)
def estimate_models(
    set_progress,
    selected_clicks,
    suite_clicks,
    aggregate_clicks,
    model_id,
    vintage,
    promote_values,
    config_store,
):
    trigger = ctx.triggered_id
    if trigger not in {
        "estimation-run",
        "estimation-run-all",
        "estimation-run-aggregate",
    }:
        raise PreventUpdate
    if not vintage:
        raise PreventUpdate

    started = time.monotonic()
    promote_after = "promote" in (promote_values or [])

    # ------------------------------------------------------------------
    # Aggregate-only recovery / refresh path.
    # Uses the seven promoted component runs so a past Energy-suite run can
    # be aggregated without rerunning Gibbs (e.g. vintage 20260901).
    # ------------------------------------------------------------------
    if trigger == "estimation-run-aggregate":
        if not aggregate_clicks:
            raise PreventUpdate
        try:
            scan_results(results_root=RESULTS_ROOT, registry_path=REGISTRY_PATH)
            run_ids = promoted_run_ids(
                REGISTRY_PATH,
                str(vintage),
                forecast_name=ENERGY_AGGREGATE_FORECAST_NAME,
                require_all=True,
                require_draws=True,
            )
            _push_estimation_progress(
                set_progress,
                10,
                "Energy aggregate",
                "Resolved the seven promoted component runs.",
            )
            with _estimation_lock(
                "hicp_energy_aggregate",
                str(vintage),
                f"aggregate-{vintage}",
            ):
                _push_estimation_progress(
                    set_progress,
                    30,
                    "Energy aggregate",
                    "Running tax bridges, weekly-to-monthly conversion and "
                    "draw-wise HICP Energy chain linking.",
                )
                aggregate = _build_or_reuse_energy_aggregate(
                    str(vintage),
                    run_ids,
                    promote_after=promote_after,
                )
            _push_estimation_progress(
                set_progress,
                100,
                "Energy aggregate complete",
                (
                    "Existing aggregate reused."
                    if aggregate.get("reused")
                    else "New HICP Energy aggregate saved."
                ),
            )
            return _energy_aggregate_result_payload(
                ok=True,
                status="complete",
                vintage=str(vintage),
                message=(
                    "HICP Energy aggregate ready from the seven promoted runs."
                ),
                elapsed=time.monotonic() - started,
                aggregate=aggregate,
            )
        except Exception as exc:
            _push_estimation_progress(
                set_progress,
                0,
                "Energy aggregate failed",
                str(exc),
            )
            return _energy_aggregate_result_payload(
                ok=False,
                status="failed",
                vintage=str(vintage),
                message=str(exc),
                elapsed=time.monotonic() - started,
            )

    if not config_store or not config_store.get("valid"):
        errors = [] if not config_store else config_store.get("errors", [])
        return _estimation_result_payload(
            ok=False,
            status="failed",
            model_id=model_id or "unknown",
            vintage=vintage,
            run_id="invalid-config",
            message="Invalid estimation configuration: " + " · ".join(map(str, errors)),
            elapsed=0.0,
            directory=RESULTS_ROOT,
        )

    # ------------------------------------------------------------------
    # Single selected model: preserve the validated Stage-1 behaviour.
    # ------------------------------------------------------------------
    if trigger == "estimation-run":
        if not selected_clicks or not model_id:
            raise PreventUpdate

        prior, sampler, spec_overrides = _model_estimation_configs(
            model_id, model_id, config_store
        )
        state = _planned_run_state(
            model_id,
            vintage,
            selected_model_id=model_id,
            config_store=config_store,
        )
        run_id = state["run_id"]
        directory = Path(state["directory"])

        if state["has_draws"] and state["has_forecast"]:
            try:
                _push_estimation_progress(
                    set_progress,
                    92,
                    "Existing run",
                    "Full posterior and forecast already exist; rebuilding display if needed.",
                )
                build_component_display(
                    directory,
                    project_root=PROJECT_ROOT,
                    forecast_name="unconditional",
                    overwrite=True,
                )
                scan_results(results_root=RESULTS_ROOT, registry_path=REGISTRY_PATH)
                if promote_after:
                    promote_run(
                        REGISTRY_PATH,
                        model_id,
                        vintage,
                        run_id,
                        required_forecast="unconditional",
                        require_draws=True,
                    )
                _push_estimation_progress(
                    set_progress,
                    100,
                    "Complete",
                    "Existing run reused; no Gibbs sampling was performed.",
                )
                return _estimation_result_payload(
                    ok=True,
                    status="complete",
                    model_id=model_id,
                    vintage=vintage,
                    run_id=run_id,
                    message="Existing complete run reused.",
                    elapsed=time.monotonic() - started,
                    directory=directory,
                    promoted=promote_after,
                    reused=True,
                )
            except Exception as exc:
                return _estimation_result_payload(
                    ok=False,
                    status="failed",
                    model_id=model_id,
                    vintage=vintage,
                    run_id=run_id,
                    message=str(exc),
                    elapsed=time.monotonic() - started,
                    directory=directory,
                )

        try:
            with _estimation_lock(model_id, vintage, run_id):
                set_run_status(
                    REGISTRY_PATH,
                    model_id=model_id,
                    vintage=vintage,
                    run_id=run_id,
                    status="running",
                    directory=directory,
                )
                _push_estimation_progress(
                    set_progress,
                    5,
                    "Preparing",
                    "Validated processed panel and deterministic run identity.",
                )

                def sampler_progress(iteration, total, diagnostics):
                    fraction = min(max(iteration / max(total, 1), 0.0), 1.0)
                    percent = int(round(8 + 72 * fraction))
                    radius = diagnostics.get("spectral_radius")
                    reject = diagnostics.get("instability_rejection")
                    pieces = [f"Gibbs iteration {iteration:,}/{total:,}"]
                    if radius is not None:
                        pieces.append(f"radius {float(radius):.4f}")
                    if reject is not None:
                        pieces.append(f"instability rejection {float(reject):.1%}")
                    if diagnostics.get("data_augmentation"):
                        pieces.append(
                            f"DK missing cells {int(diagnostics.get('missing_cells', 0))}"
                        )
                    _push_estimation_progress(
                        set_progress,
                        percent,
                        "Sampling BVAR",
                        " · ".join(pieces),
                    )

                outcome = run_component(
                    model_id,
                    vintage,
                    project_root=PROJECT_ROOT,
                    results_root=RESULTS_ROOT,
                    prior_config=prior,
                    sampler_config=sampler,
                    spec_overrides=spec_overrides,
                    forecast_name="unconditional",
                    persist=True,
                    persist_draws=True,
                    overwrite=True,
                    progress_callback=sampler_progress,
                )
                _push_estimation_progress(
                    set_progress,
                    88,
                    "Materialising display",
                    "BVAR, forecast, HICP bridge and full posterior are saved.",
                )
                build_component_display(
                    outcome.directory,
                    project_root=PROJECT_ROOT,
                    forecast_name="unconditional",
                    overwrite=True,
                )
                _push_estimation_progress(
                    set_progress,
                    95,
                    "Registering",
                    "Scanning results and updating SQLite provenance.",
                )
                scan_results(results_root=RESULTS_ROOT, registry_path=REGISTRY_PATH)
                if promote_after:
                    promote_run(
                        REGISTRY_PATH,
                        outcome.model_id,
                        outcome.vintage,
                        outcome.run_id,
                        required_forecast="unconditional",
                        require_draws=True,
                    )
                set_run_status(
                    REGISTRY_PATH,
                    model_id=outcome.model_id,
                    vintage=outcome.vintage,
                    run_id=outcome.run_id,
                    status="complete",
                    directory=outcome.directory,
                )
                _push_estimation_progress(
                    set_progress,
                    100,
                    "Complete",
                    "Run, full posterior, forecast, HICP and display are ready.",
                )
                return _estimation_result_payload(
                    ok=True,
                    status="complete",
                    model_id=outcome.model_id,
                    vintage=outcome.vintage,
                    run_id=outcome.run_id,
                    message="Estimation completed successfully.",
                    elapsed=time.monotonic() - started,
                    directory=Path(outcome.directory),
                    posterior_draws=outcome.n_posterior_draws,
                    forecast_draws=outcome.n_forecast_draws,
                    promoted=promote_after,
                    reused=False,
                )
        except Exception as exc:
            try:
                set_run_status(
                    REGISTRY_PATH,
                    model_id=model_id,
                    vintage=vintage,
                    run_id=run_id,
                    status="failed",
                    error=str(exc),
                    directory=directory,
                )
            except Exception:
                pass
            _push_estimation_progress(set_progress, 0, "Failed", str(exc))
            return _estimation_result_payload(
                ok=False,
                status="failed",
                model_id=model_id,
                vintage=vintage,
                run_id=run_id,
                message=str(exc),
                elapsed=time.monotonic() - started,
                directory=directory,
            )

    # ------------------------------------------------------------------
    # Seven-model suite.
    # ------------------------------------------------------------------
    if not suite_clicks:
        raise PreventUpdate

    try:
        common_vintage = resolve_common_vintage(
            vintage,
            models=ENERGY_SUITE_MODEL_IDS,
            project_root=PROJECT_ROOT,
        )
    except Exception as exc:
        _push_estimation_progress(set_progress, 0, "Suite blocked", str(exc), {})
        return _estimation_suite_result_payload(
            ok=False,
            status="failed",
            vintage=str(vintage),
            message=str(exc),
            elapsed=time.monotonic() - started,
            statuses={},
            promoted=False,
        )

    statuses = _initial_suite_statuses(
        common_vintage,
        model_id,
        config_store,
    )
    suite_payload = _suite_progress_payload(common_vintage, statuses)
    complete_at_start = sum(
        item["status"] == "complete" for item in statuses.values()
    )
    _push_estimation_progress(
        set_progress,
        3,
        "Suite preflight",
        f"Vintage {common_vintage} validated for all 7 models · "
        f"{complete_at_start} complete run(s) reusable.",
        suite_payload,
    )

    suite_lock_id = f"suite-{common_vintage}"
    current_model = None
    try:
        with _estimation_lock("all_7_models", common_vintage, suite_lock_id):
            total_models = len(ENERGY_SUITE_MODEL_IDS)

            for model_index, suite_model_id in enumerate(ENERGY_SUITE_MODEL_IDS):
                current_model = suite_model_id
                spec = model_spec(suite_model_id)
                prior, sampler, spec_overrides = _model_estimation_configs(
                    suite_model_id,
                    model_id,
                    config_store,
                )
                state = _planned_run_state(
                    suite_model_id,
                    common_vintage,
                    selected_model_id=model_id,
                    config_store=config_store,
                )
                run_id = state["run_id"]
                directory = Path(state["directory"])

                statuses[suite_model_id].update(
                    {
                        "status": "running",
                        "run_id": run_id,
                        "detail": "checking deterministic run",
                        "reused": False,
                    }
                )
                suite_payload = _suite_progress_payload(
                    common_vintage, statuses, current_model=suite_model_id
                )
                base = 4.0 + 84.0 * model_index / total_models
                width = 84.0 / total_models
                _push_estimation_progress(
                    set_progress,
                    int(round(base)),
                    f"{model_index + 1}/7 · {spec.label}",
                    "Checking existing full posterior and forecast contract.",
                    suite_payload,
                )

                # Reuse deterministic complete run. We still rematerialise its
                # display so the dashboard contract is current.
                if state["has_draws"] and state["has_forecast"]:
                    build_component_display(
                        directory,
                        project_root=PROJECT_ROOT,
                        forecast_name="unconditional",
                        overwrite=True,
                    )
                    statuses[suite_model_id].update(
                        {
                            "status": "complete",
                            "detail": "existing deterministic run reused",
                            "reused": True,
                        }
                    )
                    suite_payload = _suite_progress_payload(
                        common_vintage, statuses
                    )
                    _push_estimation_progress(
                        set_progress,
                        int(round(base + width)),
                        f"{model_index + 1}/7 · {spec.label}",
                        "Existing complete run reused; Gibbs skipped.",
                        suite_payload,
                    )
                    continue

                set_run_status(
                    REGISTRY_PATH,
                    model_id=suite_model_id,
                    vintage=common_vintage,
                    run_id=run_id,
                    status="running",
                    directory=directory,
                )

                statuses[suite_model_id]["detail"] = "Gibbs sampling"
                suite_payload = _suite_progress_payload(
                    common_vintage, statuses, current_model=suite_model_id
                )

                def suite_sampler_progress(iteration, total, diagnostics):
                    fraction = min(max(iteration / max(total, 1), 0.0), 1.0)
                    # Reserve the last 15% of each model slice for forecast/HICP/display.
                    percent = int(round(base + width * (0.05 + 0.80 * fraction)))
                    radius = diagnostics.get("spectral_radius")
                    reject = diagnostics.get("instability_rejection")
                    detail_parts = [f"Gibbs iteration {iteration:,}/{total:,}"]
                    if radius is not None:
                        detail_parts.append(f"radius {float(radius):.4f}")
                    if reject is not None:
                        detail_parts.append(
                            f"instability rejection {float(reject):.1%}"
                        )
                    if diagnostics.get("data_augmentation"):
                        detail_parts.append(
                            f"DK missing cells {int(diagnostics.get('missing_cells', 0))}"
                        )
                    statuses[suite_model_id]["detail"] = " · ".join(detail_parts)
                    _push_estimation_progress(
                        set_progress,
                        percent,
                        f"{model_index + 1}/7 · {spec.label}",
                        statuses[suite_model_id]["detail"],
                        _suite_progress_payload(
                            common_vintage,
                            statuses,
                            current_model=suite_model_id,
                        ),
                    )

                try:
                    outcome = run_component(
                        suite_model_id,
                        common_vintage,
                        project_root=PROJECT_ROOT,
                        results_root=RESULTS_ROOT,
                        prior_config=prior,
                        sampler_config=sampler,
                        spec_overrides=spec_overrides,
                        forecast_name="unconditional",
                        persist=True,
                        persist_draws=True,
                        overwrite=True,
                        progress_callback=suite_sampler_progress,
                    )
                    statuses[suite_model_id]["detail"] = (
                        "saving forecast · HICP bridge · display"
                    )
                    _push_estimation_progress(
                        set_progress,
                        int(round(base + width * 0.90)),
                        f"{model_index + 1}/7 · {spec.label}",
                        statuses[suite_model_id]["detail"],
                        _suite_progress_payload(
                            common_vintage,
                            statuses,
                            current_model=suite_model_id,
                        ),
                    )
                    build_component_display(
                        outcome.directory,
                        project_root=PROJECT_ROOT,
                        forecast_name="unconditional",
                        overwrite=True,
                    )
                    set_run_status(
                        REGISTRY_PATH,
                        model_id=outcome.model_id,
                        vintage=outcome.vintage,
                        run_id=outcome.run_id,
                        status="complete",
                        directory=outcome.directory,
                    )
                    statuses[suite_model_id].update(
                        {
                            "status": "complete",
                            "run_id": outcome.run_id,
                            "detail": (
                                f"estimated · {outcome.n_posterior_draws:,} posterior draws"
                            ),
                            "posterior_draws": outcome.n_posterior_draws,
                            "forecast_draws": outcome.n_forecast_draws,
                            "reused": False,
                        }
                    )
                except Exception as exc:
                    statuses[suite_model_id].update(
                        {
                            "status": "failed",
                            "detail": str(exc),
                            "reused": False,
                        }
                    )
                    try:
                        set_run_status(
                            REGISTRY_PATH,
                            model_id=suite_model_id,
                            vintage=common_vintage,
                            run_id=run_id,
                            status="failed",
                            error=str(exc),
                            directory=directory,
                        )
                    except Exception:
                        pass
                    _push_estimation_progress(
                        set_progress,
                        int(round(base)),
                        f"Failed · {spec.label}",
                        str(exc),
                        _suite_progress_payload(
                            common_vintage,
                            statuses,
                            current_model=suite_model_id,
                        ),
                    )
                    # Fail fast. Completed earlier models remain valid and will
                    # be reused on the next suite execution.
                    return _estimation_suite_result_payload(
                        ok=False,
                        status="failed",
                        vintage=common_vintage,
                        message=(
                            f"Suite stopped at {spec.label}. "
                            "Previously completed models are preserved and will be reused."
                        ),
                        elapsed=time.monotonic() - started,
                        statuses=statuses,
                        promoted=False,
                    )

                _push_estimation_progress(
                    set_progress,
                    int(round(base + width)),
                    f"{model_index + 1}/7 · {spec.label}",
                    "Component complete.",
                    _suite_progress_payload(common_vintage, statuses),
                )

            # All seven component runs are now complete.  Register/promote
            # those exact runs first, then build the HICP Energy aggregate from
            # their explicit run IDs.  This makes "Estimate Energy suite" an
            # end-to-end production action rather than stopping before notebook 10.
            _push_estimation_progress(
                set_progress,
                89,
                "Registering Energy components",
                "Scanning all seven component stores and validating provenance.",
                _suite_progress_payload(common_vintage, statuses),
            )
            scan_results(results_root=RESULTS_ROOT, registry_path=REGISTRY_PATH)

            if promote_after:
                _push_estimation_progress(
                    set_progress,
                    92,
                    "Promoting Energy components",
                    "Promoting all seven completed deterministic component runs.",
                    _suite_progress_payload(common_vintage, statuses),
                )
                for suite_model_id in ENERGY_SUITE_MODEL_IDS:
                    promote_run(
                        REGISTRY_PATH,
                        suite_model_id,
                        common_vintage,
                        statuses[suite_model_id]["run_id"],
                        required_forecast=ENERGY_AGGREGATE_FORECAST_NAME,
                        require_draws=True,
                    )

            exact_run_ids = {
                str(model_spec(suite_model_id).model_id):
                    str(statuses[suite_model_id]["run_id"])
                for suite_model_id in ENERGY_SUITE_MODEL_IDS
            }
            aggregate_progress = _suite_progress_payload(
                common_vintage, statuses
            )
            aggregate_progress["aggregate"] = {
                "status": "running",
                "label": "HICP Energy aggregate",
                "detail": (
                    "tax bridge · weekly-to-monthly · HICP rebasing · "
                    "draw-wise chain linking"
                ),
            }
            _push_estimation_progress(
                set_progress,
                95,
                "8/8 · HICP Energy aggregate",
                "Building the aggregate from the exact seven component runs.",
                aggregate_progress,
            )

            try:
                aggregate_info = _build_or_reuse_energy_aggregate(
                    common_vintage,
                    exact_run_ids,
                    promote_after=promote_after,
                )
            except Exception as exc:
                failed_progress = _suite_progress_payload(
                    common_vintage, statuses
                )
                failed_progress["aggregate"] = {
                    "status": "failed",
                    "label": "HICP Energy aggregate",
                    "detail": str(exc),
                }
                _push_estimation_progress(
                    set_progress,
                    95,
                    "Aggregate failed",
                    (
                        "All seven component BVARs are complete, but HICP Energy "
                        f"aggregation failed: {exc}"
                    ),
                    failed_progress,
                )
                return _estimation_suite_result_payload(
                    ok=False,
                    status="failed",
                    vintage=common_vintage,
                    message=(
                        "All 7 component BVARs are complete; HICP Energy "
                        f"aggregation failed: {exc}"
                    ),
                    elapsed=time.monotonic() - started,
                    statuses=statuses,
                    promoted=promote_after,
                    aggregate={
                        "status": "failed",
                        "error": str(exc),
                    },
                )

            estimated = sum(
                item.get("status") == "complete" and not item.get("reused")
                for item in statuses.values()
            )
            reused = sum(
                item.get("status") == "complete" and item.get("reused")
                for item in statuses.values()
            )

            final_progress = _suite_progress_payload(common_vintage, statuses)
            final_progress["aggregate"] = {
                "status": "complete",
                "label": "HICP Energy aggregate",
                "detail": (
                    "reused existing exact aggregate"
                    if aggregate_info.get("reused")
                    else (
                        f"saved · {int(aggregate_info.get('n_aggregate_draws') or 0):,} "
                        "effective draws"
                    )
                ),
                "run_id": aggregate_info.get("aggregate_run_id", ""),
                "reused": bool(aggregate_info.get("reused")),
            }
            _push_estimation_progress(
                set_progress,
                100,
                "Energy suite complete",
                (
                    f"{estimated} component model(s) estimated · {reused} reused · "
                    "HICP Energy aggregate "
                    + ("reused" if aggregate_info.get("reused") else "built")
                    + (" · suite promoted" if promote_after else "")
                ),
                final_progress,
            )
            return _estimation_suite_result_payload(
                ok=True,
                status="complete",
                vintage=common_vintage,
                message=(
                    f"Energy suite complete: {estimated} component model(s) "
                    f"estimated, {reused} reused, HICP Energy aggregate "
                    f"{'reused' if aggregate_info.get('reused') else 'built'}."
                ),
                elapsed=time.monotonic() - started,
                statuses=statuses,
                promoted=promote_after,
                aggregate=aggregate_info,
            )

    except Exception as exc:
        if current_model and current_model in statuses:
            if statuses[current_model].get("status") == "running":
                statuses[current_model]["status"] = "failed"
                statuses[current_model]["detail"] = str(exc)
        _push_estimation_progress(
            set_progress,
            0,
            "Suite failed",
            str(exc),
            _suite_progress_payload(common_vintage, statuses),
        )
        return _estimation_suite_result_payload(
            ok=False,
            status="failed",
            vintage=common_vintage,
            message=str(exc),
            elapsed=time.monotonic() - started,
            statuses=statuses,
            promoted=False,
        )


@callback(
    Output("estimation-suite-status", "children"),
    Input("estimation-suite-progress-store", "data"),
)
def render_estimation_suite_progress(store):
    if not store or store.get("mode") != "suite":
        return ""
    rows = []
    status_labels = {
        "waiting": "Waiting",
        "running": "Running",
        "complete": "Complete",
        "failed": "Failed",
    }
    display_items = list(store.get("models", []))
    aggregate_item = store.get("aggregate")
    if isinstance(aggregate_item, dict):
        display_items.append(
            {
                "model_id": "hicp_energy_aggregate",
                "label": aggregate_item.get("label", "HICP Energy aggregate"),
                "status": aggregate_item.get("status", "waiting"),
                "run_id": aggregate_item.get("run_id", ""),
                "detail": aggregate_item.get("detail", ""),
                "reused": aggregate_item.get("reused", False),
                "current": aggregate_item.get("status") == "running",
            }
        )

    for item in display_items:
        status = str(item.get("status", "waiting"))
        classes = "estimation-suite-row"
        if item.get("current"):
            classes += " current"
        classes += f" status-{status}"
        run_id = str(item.get("run_id", ""))
        detail = str(item.get("detail", ""))
        if item.get("reused") and status == "complete":
            detail = "reused · " + detail.replace("existing deterministic run reused", "complete deterministic run")
        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                status_labels.get(status, status.title()),
                                className=f"estimation-suite-badge status-{status}",
                            ),
                            html.Strong(str(item.get("label", item.get("model_id", "")))),
                        ],
                        className="estimation-suite-model",
                    ),
                    html.Div(
                        [
                            html.Span(detail, className="estimation-suite-detail"),
                            html.Code(
                                _short_run(run_id) if run_id else "—",
                                className="estimation-suite-runid",
                            ),
                        ],
                        className="estimation-suite-meta",
                    ),
                ],
                className=classes,
            )
        )
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Seven-model execution", className="eyebrow"),
                    html.Div(
                        f"Common vintage {store.get('vintage', '—')} · 7 components + HICP Energy aggregate",
                        className="estimation-suite-caption",
                    ),
                ],
                className="estimation-suite-heading",
            ),
            html.Div(rows, className="estimation-suite-list"),
        ],
        className="estimation-suite-panel",
    )


@callback(
    Output("estimation-result-banner", "children"),
    Output("est-result-status", "children"),
    Output("est-result-note", "children"),
    Output("est-result-run", "children"),
    Output("est-result-promoted", "children"),
    Output("est-result-draws", "children"),
    Output("est-result-draws-note", "children"),
    Output("est-result-elapsed", "children"),
    Output("est-result-directory", "children"),
    Input("estimation-result-store", "data"),
)
def render_estimation_result(store):
    if not store:
        return "", "—", "", "—", "", "—", "", "—", ""

    ok = bool(store.get("ok"))
    status = str(store.get("status", "unknown"))
    message = str(store.get("message", ""))
    is_suite = store.get("mode") == "suite"
    is_aggregate = store.get("mode") == "aggregate"

    banner = html.Div(
        [
            html.Strong(
                "Energy aggregate completed: " if ok and is_aggregate
                else "Energy suite completed: " if ok and is_suite
                else "Completed: " if ok
                else "Energy aggregate failed: " if is_aggregate
                else "Energy suite failed: " if is_suite
                else "Estimation failed: "
            ),
            html.Span(message),
        ],
        className="estimation-success-banner" if ok else "estimation-error-banner",
    )
    elapsed = float(store.get("elapsed_seconds", 0.0) or 0.0)

    if is_aggregate:
        draws = store.get("n_aggregate_draws")
        aggregate_run_id = str(store.get("aggregate_run_id", ""))
        return (
            banner,
            status.title(),
            "existing exact aggregate reused" if store.get("reused") else "aggregate built from promoted component runs" if ok else "see error above",
            _short_run(aggregate_run_id) if aggregate_run_id else "—",
            "PROMOTED" if store.get("promoted") else "not promoted",
            "—" if draws is None else f"{int(draws):,}",
            "effective aggregate posterior draws",
            f"{elapsed / 60.0:.1f} min" if elapsed >= 60 else f"{elapsed:.1f} s",
            str(store.get("directory", "")),
        )

    if is_suite:
        completed = int(store.get("completed_count", 0) or 0)
        reused = int(store.get("reused_count", 0) or 0)
        estimated = int(store.get("estimated_count", 0) or 0)
        model_rows = store.get("models", [])
        posterior_values = [
            item.get("posterior_draws")
            for item in model_rows
            if item.get("posterior_draws") is not None
        ]
        if posterior_values and len(set(map(int, posterior_values))) == 1:
            draws_text = f"{len(posterior_values)} × {int(posterior_values[0]):,}"
        elif posterior_values:
            draws_text = f"{sum(map(int, posterior_values)):,} total"
        else:
            draws_text = f"{completed} full stores"
        aggregate = dict(store.get("aggregate") or {})
        aggregate_id = str(aggregate.get("aggregate_run_id", ""))
        aggregate_ready = bool(aggregate_id)
        return (
            banner,
            status.title(),
            (
                f"{completed}/7 components complete · {estimated} estimated · "
                f"{reused} reused · aggregate "
                + ("ready" if aggregate_ready else "not ready")
            ),
            (
                f"7/7 + {_short_run(aggregate_id)}"
                if aggregate_ready
                else f"{completed} / 7"
            ),
            (
                "7 + AGGREGATE PROMOTED"
                if store.get("promoted") and completed == 7 and aggregate.get("promoted")
                else "promotion incomplete"
                if store.get("promoted")
                else "not promoted as suite"
            ),
            draws_text,
            (
                "component full posteriors · aggregate "
                + (
                    f"{int(aggregate.get('n_aggregate_draws') or 0):,} effective draws"
                    if aggregate_ready
                    else "missing"
                )
            ),
            f"{elapsed / 60.0:.1f} min" if elapsed >= 60 else f"{elapsed:.1f} s",
            (
                str(aggregate.get("directory"))
                if aggregate_ready
                else f"{store.get('vintage', '')} · {store.get('directory', '')}"
            ),
        )

    run_id = str(store.get("run_id", ""))
    posterior = store.get("posterior_draws")
    return (
        banner,
        status.title(),
        "existing run reused" if store.get("reused") else "new Gibbs execution" if ok else "see error above",
        _short_run(run_id),
        "PROMOTED" if store.get("promoted") else "not promoted",
        f"{int(posterior):,}" if posterior is not None else "saved",
        "full draws.npz required for Structural",
        f"{elapsed / 60.0:.1f} min" if elapsed >= 60 else f"{elapsed:.1f} s",
        str(store.get("directory", "")),
    )



# ---------------------------------------------------------------------------
# Estimation diagnostics: persisted artefacts only, no sampler execution
# ---------------------------------------------------------------------------


def _diagnostic_run_row(model_id: str | None, vintage: str | None, run_id: str | None):
    if not model_id or not vintage or not run_id:
        return None
    runs = list_runs(
        REGISTRY_PATH,
        model_id=str(model_id),
        vintage=str(vintage),
        status="complete",
        present_only=True,
    )
    if runs.empty:
        return None
    block = runs.loc[runs["run_id"].astype(str).eq(str(run_id))]
    return None if block.empty else block.iloc[0]


@callback(
    Output("est-diag-run-select", "options"),
    Output("est-diag-run-select", "value"),
    Input("est-model-select", "value"),
    Input("est-vintage-select", "value"),
    Input("registry-store", "data"),
    Input("estimation-result-store", "data"),
    State("est-diag-run-select", "value"),
)
def estimation_diagnostic_run_options(model_id, vintage, _registry, result_store, current):
    if not model_id or not vintage:
        return [], None
    runs = list_runs(
        REGISTRY_PATH,
        model_id=str(model_id),
        vintage=str(vintage),
        status="complete",
        present_only=True,
    )
    if runs.empty:
        return [], None
    runs = runs.sort_values(
        ["promoted", "created_at_utc", "run_id"],
        ascending=[False, False, True],
    )
    options = []
    for _, row in runs.iterrows():
        run_id = str(row["run_id"])
        tags = []
        if bool(row.get("promoted")):
            tags.append("PROMOTED")
        directory_value = row.get("directory")
        if directory_value is None or pd.isna(directory_value) or not str(directory_value).strip():
            directory = RESULTS_ROOT / str(model_id) / str(vintage) / run_id
        else:
            directory = Path(str(directory_value))
        if not ((directory / "diagnostics.parquet").is_file() or (directory / "diagnostics.csv").is_file()):
            tags.append("diagnostics missing")
        label = _short_run(run_id) + (" · " + " · ".join(tags) if tags else "")
        options.append({"label": label, "value": run_id})
    values = [item["value"] for item in options]

    preferred = None
    store = dict(result_store or {})
    if store.get("ok") and str(store.get("vintage")) == str(vintage):
        if store.get("mode") == "suite":
            for item in store.get("models", []):
                if str(item.get("model_id")) == str(model_id) and item.get("status") == "complete":
                    candidate = str(item.get("run_id", ""))
                    if candidate in values:
                        preferred = candidate
                        break
        elif str(store.get("model_id")) == str(model_id):
            candidate = str(store.get("run_id", ""))
            if candidate in values:
                preferred = candidate
    if preferred is None:
        promoted = runs.loc[runs["promoted"].fillna(0).astype(int).eq(1), "run_id"].astype(str).tolist()
        preferred = promoted[0] if promoted else values[0]
    return options, current if current in values else preferred


@callback(
    Output("est-diag-run-note", "children"),
    Output("est-diag-phi-ess", "children"),
    Output("est-diag-phi-ess-note", "children"),
    Output("est-diag-radius-median", "children"),
    Output("est-diag-radius-note", "children"),
    Output("est-diag-radius-q95", "children"),
    Output("est-diag-radius-q95-note", "children"),
    Output("est-diag-rejection", "children"),
    Output("est-diag-rejection-note", "children"),
    Output("est-diag-phi-ess-graph", "figure"),
    Output("est-diag-mcmc-table", "data"),
    Output("est-diag-stability-table", "data"),
    Input("est-model-select", "value"),
    Input("est-vintage-select", "value"),
    Input("est-diag-run-select", "value"),
)
def render_estimation_diagnostics(model_id, vintage, run_id):
    empty = (
        html.Div("Select a completed run."),
        "—", "", "—", "", "—", "", "—", "",
        empty_diagnostic_figure("Select a completed run"),
        [], [],
    )
    row = _diagnostic_run_row(model_id, vintage, run_id)
    if row is None:
        return empty
    directory_value = row.get("directory")
    if directory_value is None or pd.isna(directory_value) or not str(directory_value).strip():
        directory = RESULTS_ROOT / str(model_id) / str(vintage) / str(run_id)
    else:
        directory = Path(str(directory_value))
    try:
        metadata, diagnostics = load_component_diagnostics(directory)
        summary = component_diagnostic_summary(metadata, diagnostics)
        mcmc = key_mcmc_table(diagnostics)
        stability = stability_table(diagnostics)
        phi_fig = phi_ess_figure(diagnostics, uirevision=f"{run_id}::phi-ess")

        def fmt(value, spec=".3f"):
            if value is None or pd.isna(value):
                return "—"
            return format(float(value), spec)

        phi_ess = summary.get("min_phi_ess")
        median_radius = summary.get("median_spectral_radius")
        q95_radius = summary.get("q95_spectral_radius")
        rejection = summary.get("B_instability_rejection_rate")
        retained = summary.get("retained_draws")
        stability_state = summary.get("stability_state")
        exog = summary.get("exog_names") or []
        exog_text = "none" if not exog else ", ".join(map(str, exog))
        promoted_text = "PROMOTED · " if bool(row.get("promoted")) else ""
        note = html.Div(
            [
                html.Strong(f"{promoted_text}{model_spec(str(model_id)).label}"),
                html.Span(f" · run {_short_run(str(run_id))}"),
                html.Span(f" · {int(retained):,} retained draws" if retained is not None else ""),
                html.Span(f" · missing data: {summary.get('missing_data_method') or 'not recorded'}"),
                html.Span(f" · exogenous/deterministic block: {exog_text}"),
            ]
        )
        stability_note = (
            "all retained draws stable" if stability_state == "stable"
            else "one or more retained draws at/above unit radius" if stability_state == "unstable"
            else "stability unavailable"
        )
        phi_note = "minimum across SV innovation-variance chains; no arbitrary pass/fail threshold imposed"
        reject_note = (
            f"{int(summary.get('B_unstable_proposals') or 0):,} / {int(summary.get('B_total_proposals') or 0):,} B proposals rejected"
        )
        mcmc_display = mcmc.copy()
        for column, formatter in {
            "Posterior mean": lambda x: "—" if pd.isna(x) else f"{float(x):.4g}",
            "Posterior SD": lambda x: "—" if pd.isna(x) else f"{float(x):.4g}",
            "ESS": lambda x: "—" if pd.isna(x) else f"{float(x):,.0f}",
            "MCSE / SD": lambda x: "—" if pd.isna(x) else f"{float(x):.3f}",
        }.items():
            if column in mcmc_display:
                mcmc_display[column] = mcmc_display[column].map(formatter)
        stability_display = stability.copy()
        if "Value" in stability_display:
            stability_display["Value"] = stability_display["Value"].map(
                lambda x: "—" if pd.isna(x) else f"{float(x):.6g}"
            )
        return (
            note,
            "—" if phi_ess is None else f"{float(phi_ess):,.0f}",
            phi_note,
            fmt(median_radius, ".4f"),
            stability_note,
            fmt(q95_radius, ".4f"),
            "posterior 95th percentile of the companion-matrix radius",
            "—" if rejection is None else f"{100.0 * float(rejection):.1f}%",
            reject_note,
            phi_fig,
            mcmc_display.to_dict("records"),
            stability_display.to_dict("records"),
        )
    except Exception as exc:
        return (
            html.Div([html.Strong("Diagnostics unavailable: "), html.Span(str(exc))], className="estimation-error-text"),
            "—", "", "—", "", "—", "", "—", "",
            empty_diagnostic_figure(str(exc)), [], [],
        )


@callback(
    Output("est-agg-diag-note", "children"),
    Output("est-agg-additivity", "children"),
    Output("est-agg-additivity-note", "children"),
    Output("est-agg-history-error", "children"),
    Output("est-agg-history-note", "children"),
    Output("est-agg-anchor-error", "children"),
    Output("est-agg-anchor-note", "children"),
    Output("est-agg-weekly-rejection", "children"),
    Output("est-agg-weekly-note", "children"),
    Input("est-vintage-select", "value"),
    Input("registry-store", "data"),
    Input("estimation-result-store", "data"),
)
def render_estimation_aggregate_validation(vintage, _registry, _result_store):
    if not vintage:
        return "Select a vintage.", "—", "", "—", "", "—", "", "—", ""
    rows = list_aggregates(
        REGISTRY_PATH,
        vintage=str(vintage),
        forecast_name="unconditional",
        status="complete",
        present_only=True,
    )
    if rows.empty:
        return (
            html.Div("No completed HICP Energy aggregate is registered for this vintage."),
            "—", "", "—", "", "—", "", "—", "",
        )
    rows = rows.sort_values(["promoted", "created_at_utc", "aggregate_run_id"], ascending=[False, False, True])
    row = rows.iloc[0]
    directory_value = row.get("directory")
    if directory_value is None or pd.isna(directory_value) or not str(directory_value).strip():
        directory = RESULTS_ROOT / "hicp_energy_aggregate" / str(vintage) / str(row["aggregate_run_id"])
    else:
        directory = Path(str(directory_value))
    try:
        diag = load_aggregate_validation(directory)
        def sci(value):
            return "—" if value is None or pd.isna(value) else f"{float(value):.3e}"
        add = diag.get("drawwise_contribution_additivity_error")
        hist = diag.get("six_model_history_max_error")
        annual = diag.get("six_model_history_annual_reanchored_max_error")
        weekly = diag.get("max_weekly_rejection_rate")
        weekly_rates = diag.get("weekly_rejection_rates") or {}
        weekly_detail = " · ".join(f"{k}: {100*float(v):.1f}%" for k, v in weekly_rates.items()) if weekly_rates else "no weekly rejection diagnostics"
        note = html.Div(
            [
                html.Strong("PROMOTED · " if bool(row.get("promoted")) else ""),
                html.Span(f"aggregate {_short_run(str(row['aggregate_run_id']))}"),
                html.Span(f" · {int(diag.get('n_aggregate_draws_effective') or 0):,} effective draws"),
            ]
        )
        return (
            note,
            sci(add),
            "max |sum of component contributions − HICP Energy YoY| across posterior draws",
            sci(hist),
            "deterministic six-component historical reconstruction versus official HICP Energy",
            sci(annual),
            "maximum error after annual HICP re-anchoring",
            "—" if weekly is None else f"{100.0 * float(weekly):.1f}%",
            weekly_detail,
        )
    except Exception as exc:
        return (
            html.Div([html.Strong("Aggregate diagnostics unavailable: "), html.Span(str(exc))], className="estimation-error-text"),
            "—", "", "—", "", "—", "", "—", "",
        )



# ---------------------------------------------------------------------------
# Structural V1 callbacks
# ---------------------------------------------------------------------------


def _selected_structural_run_directory(context: dict | None) -> Path:
    if not context or not all(context.get(key) for key in ("model_id", "vintage", "run_id")):
        raise StructuralDashboardError("Select a model, vintage and complete saved run.")
    return resolve_structural_run_directory(
        RESULTS_ROOT,
        model_id=str(context["model_id"]),
        vintage=str(context["vintage"]),
        run_id=str(context["run_id"]),
    )


@callback(
    Output("structural-run-banner", "children"),
    Output("structural-reference-date", "options"),
    Output("structural-reference-date", "value"),
    Output("structural-shock", "options"),
    Output("structural-shock", "value"),
    Output("structural-response", "options"),
    Output("structural-response", "value"),
    Output("structural-volatility-variable", "options"),
    Output("structural-volatility-variable", "value"),
    Input("ctx-store", "data"),
    Input("url", "pathname"),
)
def structural_controls(context, pathname):
    if (pathname or "") != "/structural":
        raise PreventUpdate
    try:
        directory = _selected_structural_run_directory(context)
        contract = structural_run_contract(
            directory,
            project_root=PROJECT_ROOT,
        )
        variables = list(contract["variables"])
        target = str(contract["target"])
        dates = pd.to_datetime(contract["reference_dates"])
        date_options = [
            {
                "label": pd.Timestamp(date).strftime(
                    "%Y-%m-%d" if contract["frequency"] == "weekly" else "%Y-%m"
                ),
                "value": pd.Timestamp(date).isoformat(),
            }
            for date in dates
        ]
        variable_options = [
            {
                "label": name.replace("_", " ").title(),
                "value": name,
            }
            for name in variables
        ]
        banner_children = [
            html.Strong(contract["model_label"]),
            html.Span(f" · vintage {contract['vintage']}"),
            html.Span(f" · run {_short_run(contract['run_id'])}"),
            html.Span(
                f" · {contract['available_posterior_draws']:,} posterior draws available"
            ),
            html.Span(
                " · recursive ordering: "
                + " → ".join(name.replace("_", " ").title() for name in variables)
            ),
            html.Span(
                f" · missing data: {str(contract['missing_data_method']).upper()}"
            ),
        ]
        if variables and target == variables[0]:
            banner_children.append(
                html.Span(
                    " · Identification note: this saved BVAR is target-first. "
                    "Recursive contemporaneous interpretation follows that exact "
                    "estimated order; the dashboard does not silently reorder it.",
                    className="estimation-error-text",
                )
            )
        banner = html.Div(banner_children)
        return (
            banner,
            date_options,
            contract["default_reference_date"],
            variable_options,
            variables[0],
            variable_options,
            target if target in variables else variables[-1],
            variable_options,
            target if target in variables else variables[-1],
        )
    except Exception as exc:
        message = html.Div(
            [html.Strong("Structural analysis unavailable: "), html.Span(str(exc))],
            className="estimation-error-text",
        )
        return message, [], None, [], None, [], None, [], None


@callback(
    output=Output("structural-store", "data"),
    inputs=[
        Input("structural-run", "n_clicks"),
        Input("url", "pathname"),
        Input("ctx-store", "data"),
    ],
    state=[
        State("structural-reference-date", "value"),
        State("structural-horizon", "value"),
        State("structural-draws", "value"),
        State("structural-shock-unit", "value"),
        State("structural-shock-size", "value"),
        State("structural-hd-options", "value"),
    ],
    background=True,
    running=[
        (Output("structural-run", "disabled"), True, False),
        (Output("structural-cancel", "disabled"), False, True),
    ],
    cancel=[Input("structural-cancel", "n_clicks")],
    progress=[
        Output("structural-progress", "value"),
        Output("structural-phase", "children"),
        Output("structural-progress-detail", "children"),
    ],
    progress_default=(
        0,
        "Idle",
        "Opening Structural automatically computes the default analysis for the selected saved run.",
    ),
    prevent_initial_call=False,
)
def compute_structural_analysis(
    set_progress,
    n_clicks,
    pathname,
    context,
    reference_date,
    horizon,
    posterior_draws,
    shock_unit,
    shock_size,
    hd_options,
):
    # Structural V1.1 computes the default view automatically when the user
    # enters the page or selects another saved run. The button remains the
    # explicit recompute action after changing horizon/reference/draw settings.
    if (pathname or "") != "/structural":
        raise PreventUpdate

    trigger = ctx.triggered_id
    manual_recompute = trigger == "structural-run"
    effective_reference = reference_date if manual_recompute else None

    try:
        directory = _selected_structural_run_directory(context)
        set_progress(
            (
                10,
                "Loading posterior",
                f"Reading persisted draws for {_short_run(str(context.get('run_id', '')))}.",
            )
        )
        # compute_structural_v1 performs the same deterministic draw-subset
        # convention used by the notebooks (linspace across the retained chain).
        set_progress(
            (
                30,
                "Recursive identification",
                "Computing regular structural impact matrices and impulse responses.",
            )
        )
        payload = compute_structural_v1(
            directory,
            project_root=PROJECT_ROOT,
            reference_date=effective_reference,
            horizon=int(horizon or STRUCTURAL_DEFAULT_HORIZON),
            posterior_draws=int(posterior_draws or DEFAULT_STRUCTURAL_DRAWS),
            shock_unit=str(shock_unit or DEFAULT_SHOCK_UNIT),
            shock_size=float(shock_size or DEFAULT_SHOCK_SIZE),
            split_outlier_amplification="split_outliers" in set(hd_options or []),
        )
        set_progress(
            (
                100,
                "Structural analysis ready",
                "IRF, FEVD and historical decomposition were computed from saved Gibbs draws.",
            )
        )
        return payload
    except Exception as exc:
        set_progress((0, "Structural analysis failed", str(exc).splitlines()[0]))
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


@callback(
    Output("structural-stat-identification", "children"),
    Output("structural-stat-ordering", "children"),
    Output("structural-stat-reference", "children"),
    Output("structural-stat-frequency", "children"),
    Output("structural-stat-draws", "children"),
    Output("structural-stat-draws-total", "children"),
    Output("structural-stat-hd-error", "children"),
    Output("structural-stat-hd-error-note", "children"),
    Output("structural-stat-fevd-error", "children"),
    Output("structural-stat-fevd-error-note", "children"),
    Output("structural-stat-shock-scale", "children"),
    Output("structural-stat-shock-scale-note", "children"),
    Input("structural-store", "data"),
)
def structural_stats(store):
    if not store or not store.get("ok"):
        return "—", "", "—", "", "—", "", "—", "", "—", "", "—", ""
    diag = dict(store.get("diagnostics", {}) or {})
    ordering = " → ".join(
        str(name).replace("_", " ").title()
        for name in store.get("recursive_ordering", [])
    )
    ref = pd.Timestamp(store["reference_date"]).strftime(
        "%Y-%m-%d" if store.get("frequency") == "weekly" else "%Y-%m"
    )
    hd_error = diag.get("hd_max_reconstruction_error_drawwise")
    fevd_error = diag.get("fevd_max_share_sum_error")
    return (
        "Recursive",
        ordering,
        ref,
        str(store.get("frequency", "")).title(),
        f"{int(store.get('posterior_draws', 0)):,}",
        f"of {int(store.get('available_posterior_draws', 0)):,} saved draws",
        "—" if hd_error is None else f"{float(hd_error):.3e}",
        "max draw-wise |reconstructed − observed|",
        "—" if fevd_error is None else f"{float(fevd_error):.3e}",
        "max draw-wise |sum FEVD shares − 1|",
        (
            f"{float(store.get('shock_size', 1.0)):g} unit"
            if str(store.get("shock_unit")) == "level"
            else f"{float(store.get('shock_size', 1.0)):g}σ"
        ),
        (
            "impact-normalised shocked variable"
            if str(store.get("shock_unit")) == "level"
            else "regular structural standard deviations"
        ),
    )


@callback(
    Output("structural-volatility-graph", "figure"),
    Input("structural-store", "data"),
    Input("structural-volatility-variable", "value"),
    Input("structural-reference-date", "value"),
    Input("structural-volatility-graph", "relayoutData"),
)
def structural_volatility_graph(store, variable, reference_date, relayout_data):
    return volatility_history_figure(
        store,
        variable=variable,
        reference_date=reference_date,
        relayout_data=relayout_data,
    )


@callback(
    Output("structural-irf-graph", "figure"),
    Input("structural-store", "data"),
    Input("structural-response", "value"),
    Input("structural-shock", "value"),
    Input("structural-irf-metric", "value"),
    Input("structural-irf-fan", "value"),
)
def structural_irf_graph(store, response, shock, metric, fan_mode):
    return irf_figure(
        store,
        response=response,
        shock=shock,
        metric=metric or "cumulative",
        fan_mode=fan_mode or "68",
    )


@callback(
    Output("structural-fevd-graph", "figure"),
    Input("structural-store", "data"),
    Input("structural-response", "value"),
)
def structural_fevd_graph(store, response):
    return fevd_figure(store, response=response)


@callback(
    Output("structural-hd-graph", "figure"),
    Input("structural-store", "data"),
    Input("structural-response", "value"),
    Input("structural-hd-window", "value"),
    Input("structural-hd-graph", "relayoutData"),
)
def structural_hd_graph(store, response, window, relayout_data):
    last_obs = None if window is None or int(window) < 0 else int(window)
    return historical_decomposition_figure(
        store,
        response=response,
        last_obs=last_obs,
        relayout_data=relayout_data,
    )



# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@callback(
    Output("page-forecast", "style"),
    Output("page-aggregate", "style"),
    Output("page-scenarios", "style"),
    Output("page-structural", "style"),
    Output("page-estimation", "style"),
    Input("url", "pathname"),
)
def route(pathname: str | None):
    """Toggle pre-mounted pages instead of injecting callback targets dynamically.

    Keeping the Forecast controls in the initial DOM is important: ``data-store``
    can be populated during app bootstrap, and the metric/series callbacks must
    already have mounted Output components when that happens.
    """
    pathname = pathname or "/forecast"
    route_name = {
        "/": "forecast",
        "/forecast": "forecast",
        "/aggregate": "aggregate",
        "/scenarios": "scenarios",
        "/structural": "structural",
        "/estimation": "estimation",
    }.get(pathname, "forecast")
    visible = {"display": "block"}
    hidden = {"display": "none"}
    names = ("forecast", "aggregate", "scenarios", "structural", "estimation")
    return tuple(visible if name == route_name else hidden for name in names)


# ---------------------------------------------------------------------------
# Forecast-page callbacks: all downstream of data-store, no filesystem I/O
# ---------------------------------------------------------------------------


_METRIC_LABELS = {
    "level": "Level",
    "hicp_level": "HICP level",
    "absolute_change": "Absolute change",
    "yoy": "Year-on-year",
}


@callback(
    Output("forecast-metric", "options"),
    Output("forecast-metric", "value"),
    Input("data-store", "data"),
    State("forecast-metric", "value"),
)
def metric_options(store: dict | None, current: str | None):
    frame = _frame_from_store(store)
    if frame.empty:
        return [], None
    fan = frame.loc[frame["record_type"] == "fan"]
    metrics = [str(value) for value in fan["metric"].dropna().unique()]
    preferred_order = ["yoy", "hicp_level", "level", "absolute_change"]
    metrics = [m for m in preferred_order if m in metrics] + [m for m in metrics if m not in preferred_order]
    options = [{"label": _METRIC_LABELS.get(m, m.replace("_", " ").title()), "value": m} for m in metrics]
    default = "yoy" if "yoy" in metrics else ("hicp_level" if "hicp_level" in metrics else ("level" if "level" in metrics else (metrics[0] if metrics else None)))
    return options, current if current in metrics else default


@callback(
    Output("forecast-series", "options"),
    Output("forecast-series", "value"),
    Input("data-store", "data"),
    Input("forecast-metric", "value"),
    State("forecast-series", "value"),
)
def series_options(store: dict | None, metric: str | None, current: str | None):
    frame = _frame_from_store(store)
    if frame.empty or not metric:
        return [], None
    fan = frame.loc[(frame["record_type"] == "fan") & (frame["metric"] == metric)]
    if fan.empty:
        return [], None
    pairs = fan[["series", "label"]].dropna(subset=["series"]).drop_duplicates("series")
    options = [
        {
            "label": str(row["label"]) if pd.notna(row["label"]) else str(row["series"]),
            "value": str(row["series"]),
        }
        for _, row in pairs.iterrows()
    ]
    values = [item["value"] for item in options]
    target = None
    if store and store.get("meta"):
        target = store["meta"].get("target")
    default = target if target in values else (values[0] if values else None)
    return options, current if current in values else default


@callback(
    Output("forecast-graph", "figure"),
    Output("forecast-values-table", "data"),
    Output("stat-observed", "children"),
    Output("stat-observed-date", "children"),
    Output("stat-first", "children"),
    Output("stat-first-date", "children"),
    Output("stat-terminal", "children"),
    Output("stat-terminal-date", "children"),
    Output("stat-horizon", "children"),
    Output("stat-horizon-unit", "children"),
    Input("data-store", "data"),
    Input("forecast-metric", "value"),
    Input("forecast-series", "value"),
    Input("forecast-fan", "value"),
)
def update_forecast_view(
    store: dict | None,
    metric: str | None,
    series: str | None,
    fan_mode: str,
):
    frame = _frame_from_store(store)
    if frame.empty or not metric or not series:
        empty = _empty_forecast_figure(
            "Forecast display is not available yet. Check the run banner above."
        )
        return empty, [], "—", "", "—", "", "—", "", "—", ""

    context = dict((store or {}).get("context") or {})
    fig = forecast_figure(frame, metric=metric, series=series, fan_mode=fan_mode, context=context)
    history = _history_rows(frame, metric, series)
    fan = _forecast_rows(frame, metric, series)
    future = fan.loc[fan["segment"] == "forecast"].copy()

    table = fan[["date", "segment", "q05", "q16", "q50", "q84", "q95"]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    for column in ("q05", "q16", "q50", "q84", "q95"):
        table[column] = table[column].round(4)
    table_data = table.to_dict("records")

    observed_value = observed_date = first_value = first_date = terminal_value = terminal_date = "—"
    if not history.empty:
        last = history.iloc[-1]
        observed_value = _format_number(last["value"])
        observed_date = pd.Timestamp(last["date"]).strftime("%Y-%m-%d")
    if not future.empty:
        first = future.iloc[0]
        last = future.iloc[-1]
        first_value = _format_number(first["q50"])
        first_date = pd.Timestamp(first["date"]).strftime("%Y-%m-%d")
        terminal_value = _format_number(last["q50"])
        terminal_date = pd.Timestamp(last["date"]).strftime("%Y-%m-%d")

    horizon = int(len(future))
    meta = (store or {}).get("meta", {})
    if metric in {"hicp_level", "yoy"}:
        frequency = str(meta.get("hicp_frequency", "monthly")).lower()
    else:
        frequency = str(meta.get("frequency", "period")).lower()
    unit = "months" if frequency == "monthly" else "weeks" if frequency == "weekly" else "periods"
    return (
        fig,
        table_data,
        observed_value,
        observed_date,
        first_value,
        first_date,
        terminal_value,
        terminal_date,
        str(horizon),
        unit,
    )



# ---------------------------------------------------------------------------
# Tax-scenario page callbacks
# ---------------------------------------------------------------------------


def _normalise_scenario_period(value, frequency: str) -> pd.Timestamp:
    """Map a UI/contract timestamp to the exact model period start.

    The previous dashboard normalised the requested date but compared it with
    raw contract bounds.  A bound such as 2026-09-01 with a hidden time or
    timezone therefore rejected an apparently identical 2026-09-01 UI date.
    """
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Scenario date is invalid.")
    if timestamp.tzinfo is not None:
        # Scenario controls are calendar-period based, not instant based.
        timestamp = timestamp.tz_localize(None)
    frequency = str(frequency or "monthly").lower()
    if frequency == "weekly":
        return timestamp.to_period("W-SUN").start_time.normalize()
    if frequency == "monthly":
        return timestamp.to_period("M").to_timestamp(how="start").normalize()
    return timestamp.normalize()


def _scenario_component_contract_for_row(
    vintage: str,
    model_id: str,
    row,
):
    """Load and validate the actual saved-run tax-scenario contract."""
    from energy_bvar_io import load_energy_bvar_forecast

    forecast = load_energy_bvar_forecast(Path(str(row["directory"])))
    dataset = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / str(vintage)
        / model_spec(model_id).dataset_file
    )
    return component_tax_scenario_contract(
        model_id,
        forecast,
        dataset_path=dataset,
    )



# ---------------------------------------------------------------------------
# Conditional observable-path scenario callbacks — V2 active set
# ---------------------------------------------------------------------------


@callback(
    Output("conditional-agg-select", "options"),
    Output("conditional-agg-select", "value"),
    Input("vintage-select", "value"),
    Input("registry-store", "data"),
    State("conditional-agg-select", "value"),
)
def conditional_aggregate_options(vintage, _, current):
    if not vintage:
        return [], None
    frame = list_aggregates(REGISTRY_PATH, vintage=str(vintage), present_only=True)
    if frame.empty:
        return [], None
    frame = frame.loc[frame["status"].astype(str) == "complete"].copy()
    if frame.empty:
        return [], None
    frame = frame.sort_values(
        ["promoted", "created_at_utc", "aggregate_run_id"],
        ascending=[False, False, True],
    )

    options = []
    usable = []
    for _idx, row in frame.iterrows():
        directory = Path(str(row["directory"]))
        try:
            contract = conditional_aggregate_contract(directory)
            if not contract.get("model_ids"):
                continue
        except Exception:
            continue
        run_id = str(row["aggregate_run_id"])
        usable.append(run_id)
        tags = ["PROMOTED"] if bool(row.get("promoted")) else []
        suffix = " · " + " · ".join(tags) if tags else ""
        options.append(
            {
                "label": f"{_short_run(run_id)}{suffix}",
                "value": run_id,
            }
        )
    if not usable:
        return [], None
    return options, current if current in usable else usable[0]


@callback(
    Output("conditional-component-select", "options"),
    Output("conditional-component-select", "value"),
    Input("conditional-agg-select", "value"),
    Input("vintage-select", "value"),
    Input("conditional-store", "data"),
    State("conditional-component-select", "value"),
)
def conditional_component_options(
    aggregate_run_id, vintage, conditional_store, current
):
    row = _selected_aggregate_row(vintage, aggregate_run_id)
    if row is None:
        return [], None
    try:
        contract = conditional_aggregate_contract(Path(str(row["directory"])))
        model_ids = list(contract.get("model_ids", []) or [])
    except Exception:
        return [], None

    options = [
        {"label": model_spec(model_id).label, "value": model_id}
        for model_id in model_ids
    ]
    active = set(conditional_set_components(conditional_store))
    if current in model_ids:
        value = current
    elif active.intersection(model_ids):
        value = sorted(active.intersection(model_ids))[0]
    else:
        value = model_ids[0] if model_ids else None
    return options, value


@callback(
    Output("conditional-variable-select", "options"),
    Output("conditional-variable-select", "value"),
    Input("conditional-agg-select", "value"),
    Input("conditional-component-select", "value"),
    Input("vintage-select", "value"),
    Input("conditional-store", "data"),
    State("conditional-variable-select", "value"),
)
def conditional_variable_options(
    aggregate_run_id, model_id, vintage, conditional_store, current
):
    row = _selected_aggregate_row(vintage, aggregate_run_id)
    if row is None or not model_id:
        return [], None
    try:
        contract = conditional_component_contract(
            Path(str(row["directory"])),
            model_id=str(model_id),
            project_root=PROJECT_ROOT,
        )
    except Exception:
        return [], None

    units = dict(contract.get("units", {}) or {})
    variables = list(contract.get("condition_variables", []) or [])
    options = []
    for variable in variables:
        unit = str(units.get(variable, "") or "")
        label = variable.replace("_", " ").title()
        if unit:
            label += f" · {unit}"
        options.append({"label": label, "value": variable})

    existing = conditional_set_payload(conditional_store, model_id)
    existing_variable = (
        dict((existing or {}).get("meta", {}) or {}).get("condition_variable")
    )
    if existing_variable in variables:
        value = existing_variable
    elif current in variables:
        value = current
    else:
        value = variables[0] if variables else None
    return options, value


@callback(
    Output("conditional-path-value-label", "children"),
    Output("conditional-path-value", "value"),
    Output("conditional-support-note", "children"),
    Input("conditional-agg-select", "value"),
    Input("conditional-component-select", "value"),
    Input("conditional-variable-select", "value"),
    Input("conditional-path-mode", "value"),
    Input("vintage-select", "value"),
    Input("conditional-store", "data"),
)
def conditional_path_defaults(
    aggregate_run_id,
    model_id,
    condition_variable,
    path_mode,
    vintage,
    conditional_store,
):
    row = _selected_aggregate_row(vintage, aggregate_run_id)
    if row is None or not model_id or not condition_variable:
        return (
            "Scenario value",
            10.0 if path_mode != "level" else None,
            "Select an aggregate, component and conditioned variable.",
        )
    try:
        contract = conditional_component_contract(
            Path(str(row["directory"])),
            model_id=str(model_id),
            project_root=PROJECT_ROOT,
        )
        latest = dict(contract.get("latest_observed", {}) or {}).get(
            str(condition_variable), {}
        )
        last_value = float(latest["value"])
        last_date = pd.Timestamp(latest["date"]).date().isoformat()
        unit = str(
            latest.get("unit")
            or contract.get("units", {}).get(condition_variable, "")
        )
        future = pd.to_datetime(contract.get("future_dates", []))
        start = (
            pd.Timestamp(future[0]).date().isoformat() if len(future) else "—"
        )
        end = (
            pd.Timestamp(future[-1]).date().isoformat() if len(future) else "—"
        )

        existing = conditional_set_payload(conditional_store, model_id)
        em = dict((existing or {}).get("meta", {}) or {})
        same_variable = (
            str(em.get("condition_variable") or "") == str(condition_variable)
        )

        if str(path_mode) == "level":
            label = f"Permanent future level{f' ({unit})' if unit else ''}"
            if same_variable and str(em.get("path_mode")) == "level":
                value = float(em.get("path_value"))
            else:
                value = last_value
        else:
            label = "Change vs last observed (%)"
            if same_variable and str(em.get("path_mode")) == "percent":
                value = float(em.get("path_value"))
            else:
                value = 10.0

        note = (
            f"{contract['model_label']} · run {_short_run(contract['run_id'])} · "
            f"{condition_variable.replace('_', ' ')} last observed "
            f"{last_value:.6g}{(' ' + unit) if unit else ''} on {last_date} · "
            f"condition applied permanently from {start} through {end} · "
            f"{int(contract['n_forecast_draws']):,} saved forecast draws · "
            f"affects final HICP component: "
            f"{str(contract['affected_hicp_component']).replace('_', ' ')}. "
            "One active conditional is allowed per component BVAR; Add / update "
            "replaces that component's previous conditional."
        )
        return label, value, note
    except Exception as exc:
        return (
            "Scenario value",
            10.0 if path_mode != "level" else None,
            html.Div(
                [
                    html.Strong("Conditional controls unavailable: "),
                    html.Span(str(exc)),
                ],
                className="banner-error",
            ),
        )


@callback(
    output=Output("conditional-store", "data"),
    inputs=[
        Input("conditional-run", "n_clicks"),
        Input("conditional-remove", "n_clicks"),
        Input("conditional-reset-all", "n_clicks"),
    ],
    state=[
        State("conditional-agg-select", "value"),
        State("conditional-component-select", "value"),
        State("conditional-variable-select", "value"),
        State("conditional-path-mode", "value"),
        State("conditional-path-value", "value"),
        State("vintage-select", "value"),
        State("conditional-store", "data"),
    ],
    background=True,
    running=[
        (Output("conditional-run", "disabled"), True, False),
        (Output("conditional-cancel", "disabled"), False, True),
    ],
    cancel=[Input("conditional-cancel", "n_clicks")],
    progress=[
        Output("conditional-progress", "value"),
        Output("conditional-phase", "children"),
        Output("conditional-progress-detail", "children"),
    ],
    progress_default=(
        0,
        "Idle",
        "Choose a saved aggregate, component and observable path, then Add / update.",
    ),
    prevent_initial_call=True,
)
def mutate_conditional_set(
    set_progress,
    _run,
    _remove,
    _reset,
    aggregate_run_id,
    model_id,
    condition_variable,
    path_mode,
    path_value,
    vintage,
    current_store,
):
    trigger = ctx.triggered_id

    if trigger == "conditional-reset-all":
        set_progress((100, "Conditional set cleared", "No active conditional scenarios."))
        return clear_conditional_set(
            vintage=str(vintage) if vintage else None,
            aggregate_run_id=(
                str(aggregate_run_id) if aggregate_run_id else None
            ),
        )

    if trigger == "conditional-remove":
        if not model_id:
            return current_store
        set_progress(
            (
                100,
                "Conditional removed",
                f"{model_spec(model_id).label} removed from the active conditional set.",
            )
        )
        return remove_conditional_component(current_store, model_id)

    if trigger != "conditional-run":
        raise PreventUpdate

    if not aggregate_run_id or not model_id or not condition_variable:
        set_progress((0, "Conditional forecast failed", "Aggregate, component and variable are required."))
        return current_store
    if path_value is None:
        set_progress((0, "Conditional forecast failed", "Scenario path value is required."))
        return current_store

    row = _selected_aggregate_row(vintage, aggregate_run_id)
    if row is None:
        set_progress((0, "Conditional forecast failed", "Selected aggregate run is unavailable."))
        return current_store
    directory = Path(str(row["directory"]))

    cache_key = "conditional-v2::" + json.dumps(
        {
            "contract": CONDITIONAL_CONTRACT_VERSION,
            "aggregate_run_id": str(aggregate_run_id),
            "model_id": str(model_id),
            "condition_variable": str(condition_variable),
            "path_mode": str(path_mode),
            "path_value": float(path_value),
        },
        sort_keys=True,
    )
    cached = _diskcache.get(cache_key)
    if isinstance(cached, dict) and cached.get("ok"):
        payload = cached
    else:
        try:
            set_progress(
                (
                    10,
                    "Loading saved posterior",
                    "Reading the exact component forecast/run recorded by the selected aggregate.",
                )
            )
            set_progress(
                (
                    30,
                    "Paired conditional forecast",
                    "Replaying the saved unconditional path and imposing the observable future path with the same random stream.",
                )
            )
            payload = compute_conditional_scenario(
                directory,
                project_root=PROJECT_ROOT,
                model_id=str(model_id),
                condition_variable=str(condition_variable),
                path_mode=str(path_mode or "percent"),
                path_value=float(path_value),
            )
            _diskcache.set(cache_key, payload, expire=3600)
        except Exception as exc:
            set_progress((0, "Conditional forecast failed", str(exc).splitlines()[0]))
            return current_store

    updated = upsert_conditional_component(current_store, payload)
    set_progress(
        (
            100,
            "Conditional scenario active",
            f"{model_spec(model_id).label} added/updated. "
            f"{len(conditional_set_components(updated))} conditional component(s) active.",
        )
    )
    return updated


@callback(
    Output("conditional-set-summary", "children"),
    Input("conditional-store", "data"),
    Input("conditional-component-select", "value"),
)
def render_conditional_set_summary(store, selected_model_id):
    rows = conditional_set_summary(store)
    if not rows:
        return html.Div(
            "No active conditional scenario. Configure one above and click Add / update conditional.",
            className="placeholder-text",
        )
    cards = []
    for row in rows:
        selected = str(selected_model_id) == str(row["model_id"])
        impact = row.get("aggregate_terminal_impact")
        impact_label = (
            "—"
            if impact is None
            else f"{float(impact):+.3f} pp HICP Energy terminal"
        )
        cards.append(
            html.Button(
                [
                    html.Div(
                        [
                            html.Strong("Conditional · " + row["label"]),
                            html.Span(
                                "Selected",
                                style={
                                    "display": "inline-block" if selected else "none",
                                    "marginLeft": "8px",
                                    "fontSize": "10px",
                                    "fontWeight": "700",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.08em",
                                    "color": "#2563eb",
                                },
                            ),
                        ]
                    ),
                    html.Div(
                        f"{row['condition_variable'].replace('_',' ')} · "
                        f"{row['condition_description']} · {impact_label}",
                        style={
                            "marginTop": "3px",
                            "fontSize": "12px",
                            "color": "#64748b",
                        },
                    ),
                ],
                id={"type": "conditional-card", "model_id": row["model_id"]},
                n_clicks=0,
                style={
                    "display": "block",
                    "width": "100%",
                    "textAlign": "left",
                    "background": "#eff6ff" if selected else "#ffffff",
                    "border": (
                        "1px solid #2563eb"
                        if selected
                        else "1px solid #e2e8f0"
                    ),
                    "borderRadius": "10px",
                    "padding": "10px 12px",
                    "marginTop": "7px",
                    "cursor": "pointer",
                },
            )
        )
    return cards


@callback(
    Output("conditional-component-select", "value", allow_duplicate=True),
    Input({"type": "conditional-card", "model_id": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_conditional_card(clicks):
    if not clicks or not any(int(value or 0) > 0 for value in clicks):
        raise PreventUpdate
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict):
        raise PreventUpdate
    model_id = triggered.get("model_id")
    if not model_id:
        raise PreventUpdate
    return str(model_id)


@callback(
    Output("conditional-banner", "children"),
    Input("conditional-store", "data"),
    Input("conditional-component-select", "value"),
)
def conditional_result_banner(store, model_id):
    payload = conditional_set_payload(store, model_id)
    if not payload:
        return ""
    meta = dict(payload.get("meta", {}) or {})
    filter_diag = dict(meta.get("weekly_joint_filter", {}) or {})
    rejected = int(filter_diag.get("rejected_draws", 0) or 0)
    total = int(
        filter_diag.get(
            "total_draws", meta.get("n_forecast_draws", 0)
        )
        or 0
    )
    parts = [
        html.Strong(
            f"{meta.get('model_label', meta.get('model_id', 'Component'))} conditional path"
        ),
        html.Span(
            f" · {str(meta.get('condition_variable', '')).replace('_', ' ')}: "
            f"{meta.get('condition_description', '—')}"
        ),
        html.Span(
            f" · baseline replay error "
            f"{float(meta.get('baseline_replay_max_abs_error', 0.0)):.2e}"
        ),
        html.Span(
            f" · aggregate replay error "
            f"{float(meta.get('aggregate_yoy_replay_max_abs_error', 0.0)):.2e}"
        ),
        html.Span(
            f" · {int(meta.get('n_aggregate_draws_paired', 0)):,}/"
            f"{int(meta.get('n_aggregate_draws_original', 0)):,} aggregate draws paired"
        ),
    ]
    if rejected:
        parts.append(
            html.Span(
                f" · weekly joint admissibility rejected {rejected}/{total} forecast draws"
            )
        )
    return html.Div(parts, className="selection-banner")


@callback(
    Output("conditional-stat-level", "children"),
    Output("conditional-stat-level-note", "children"),
    Output("conditional-stat-component", "children"),
    Output("conditional-stat-component-note", "children"),
    Output("conditional-stat-aggregate", "children"),
    Output("conditional-stat-aggregate-note", "children"),
    Output("conditional-stat-draws", "children"),
    Output("conditional-stat-draws-note", "children"),
    Input("conditional-store", "data"),
    Input("conditional-component-select", "value"),
)
def conditional_stats(store, model_id):
    payload = conditional_set_payload(store, model_id)
    values = conditional_kpis(payload)
    if values.get("condition_level") is None:
        return "—", "", "—", "", "—", "", "—", ""

    unit = str(values.get("condition_unit") or "")
    comp = values.get("component_impact")
    comp_low = values.get("component_low")
    comp_high = values.get("component_high")
    agg = values.get("aggregate_impact")
    agg_low = values.get("aggregate_low")
    agg_high = values.get("aggregate_high")
    draws = values.get("n_draws")
    original = values.get("n_draws_original")
    date = (
        pd.Timestamp(values["terminal_date"]).date().isoformat()
        if values.get("terminal_date")
        else ""
    )

    return (
        f"{float(values['condition_level']):.4g}{(' ' + unit) if unit else ''}",
        str(values.get("condition_description") or ""),
        "—" if comp is None else f"{float(comp):+.3f} pp",
        (
            ""
            if comp_low is None or comp_high is None
            else f"68% [{float(comp_low):+.3f}, {float(comp_high):+.3f}] pp · {date}"
        ),
        "—" if agg is None else f"{float(agg):+.3f} pp",
        (
            ""
            if agg_low is None or agg_high is None
            else f"68% [{float(agg_low):+.3f}, {float(agg_high):+.3f}] pp · {date}"
        ),
        "—" if draws is None else f"{int(draws):,}",
        (
            ""
            if draws is None
            else f"matched from {int(original):,} saved aggregate draws"
        ),
    )


@callback(
    Output("conditional-path-graph", "figure"),
    Output("conditional-target-graph", "figure"),
    Output("conditional-component-graph", "figure"),
    Output("conditional-aggregate-graph", "figure"),
    Output("conditional-aggregate-impact-graph", "figure"),
    Input("conditional-store", "data"),
    Input("conditional-component-select", "value"),
    Input("conditional-fan", "value"),
)
def conditional_figures(store, model_id, fan_mode):
    payload = conditional_set_payload(store, model_id)
    if not payload:
        empty = empty_conditional_figure(
            "Add/update a conditional scenario for the selected component."
        )
        return empty, empty, empty, empty, empty

    meta = dict(payload.get("meta", {}) or {})
    revision = (
        f"{meta.get('aggregate_run_id','agg')}::{meta.get('model_id','model')}::"
        f"{meta.get('condition_variable','condition')}::{meta.get('path_mode','mode')}::"
        f"{meta.get('path_value','value')}"
    )
    mode = fan_mode or "68"
    return (
        conditional_path_figure(
            payload, fan_mode=mode, uirevision=revision + "::path"
        ),
        conditional_target_figure(
            payload, fan_mode=mode, uirevision=revision + "::target"
        ),
        conditional_component_figure(
            payload, fan_mode=mode, uirevision=revision + "::component"
        ),
        conditional_aggregate_figure(
            payload, fan_mode=mode, uirevision=revision + "::aggregate"
        ),
        conditional_aggregate_impact_figure(
            payload, fan_mode=mode, uirevision=revision + "::impact"
        ),
    )


def _scenario_component_forecast_row(vintage: str | None, model_id: str | None, forecast_name: str | None):
    if not vintage or not model_id or not forecast_name:
        return None
    forecasts = list_forecasts(REGISTRY_PATH, model_id=str(model_id), vintage=str(vintage), forecast_name=str(forecast_name), valid_only=True, present_only=True)
    if forecasts.empty:
        return None
    runs = list_runs(REGISTRY_PATH, model_id=str(model_id), vintage=str(vintage), status="complete", present_only=True)
    promoted = set(runs.loc[runs["promoted"].fillna(0).astype(int)==1,"run_id"].astype(str)) if not runs.empty else set()
    promoted_rows = forecasts.loc[forecasts["run_id"].astype(str).isin(promoted)]
    if len(promoted_rows)==1: return promoted_rows.iloc[0]
    if len(forecasts)==1: return forecasts.iloc[0]
    return None


@callback(Output("scenario-component-select","options"), Output("scenario-component-select","value"), Input("vintage-select","value"), Input("forecast-select","value"), Input("registry-store","data"), State("scenario-component-select","value"))
def scenario_component_options(vintage, forecast_name, _, current):
    if not vintage or not forecast_name:
        return [], None

    available = []
    options = []
    for model_id in TAX_SCENARIO_MODEL_IDS:
        row = _scenario_component_forecast_row(vintage, model_id, forecast_name)
        if row is None:
            continue
        try:
            _scenario_component_contract_for_row(str(vintage), model_id, row)
        except Exception:
            # Do not advertise a scenario that cannot actually be propagated.
            # The list is intentionally capability-driven rather than a static
            # list of model names.
            continue
        available.append(model_id)
        options.append({"label": model_spec(model_id).label, "value": model_id})

    value = current if current in available else (available[0] if available else None)
    return options, value


@callback(Output("scenario-start-date","date"), Output("scenario-start-date","min_date_allowed"), Output("scenario-start-date","max_date_allowed"), Output("scenario-vat-delta","value"), Output("scenario-excise-delta","value"), Output("scenario-excise-label","children"), Output("scenario-support-note","children"), Output("scenario-start-date","disabled"), Output("scenario-vat-delta","disabled"), Output("scenario-excise-delta","disabled"), Input("scenario-component-select","value"), Input("vintage-select","value"), Input("forecast-select","value"), Input("registry-store","data"), Input("scenario-store","data"))
def scenario_control_defaults(model_id, vintage, forecast_name, _, scenario_store):
    if not model_id or not vintage or not forecast_name:
        return None,None,None,0.0,0.0,"Excise change","No scenario-capable component forecast is available.",True,True,True
    row=_scenario_component_forecast_row(vintage,model_id,forecast_name)
    if row is None:
        return None,None,None,0.0,0.0,"Excise change",f"{model_spec(model_id).label}: no unique saved forecast. Promote one run if several coexist.",True,True,True
    try:
        contract = _scenario_component_contract_for_row(str(vintage), model_id, row)
    except Exception as exc:
        return None,None,None,0.0,0.0,"Excise change",f"Scenario controls unavailable: {exc}",True,True,True
    existing=scenario_set_payload(scenario_store,model_id); em=dict((existing or {}).get("meta",{}) or {})
    unit=str(contract.get("excise_unit","source unit"))
    frequency=str(contract.get("frequency","monthly"))
    min_start=_normalise_scenario_period(contract["min_start"], frequency)
    max_start=_normalise_scenario_period(contract["max_start"], frequency)
    default_start=_normalise_scenario_period(contract["default_start"], frequency)
    start=_normalise_scenario_period(em.get("scenario_start") or default_start, frequency)
    if start < min_start or start > max_start:
        start=default_start
    freq_label="monthly periods" if frequency=="monthly" else "weekly periods (Monday)"
    tax_source_note = (
        "WOB VAT + excise tax block · "
        if frequency == "weekly"
        else "VAT + excise tax bridge · "
    )
    note=(
        f"{model_spec(model_id).label} · run {str(row['run_id'])[:12]} · "
        f"available scenario window {min_start.date().isoformat()} — {max_start.date().isoformat()} "
        f"({freq_label}) · {tax_source_note}baseline at first future period: VAT "
        f"{contract['baseline_vat_at_start']:.2f}% · excise "
        f"{contract['baseline_excise_at_start']:.3f} {unit}. "
        "Enter a non-zero VAT and/or excise change; zero/zero removes the component from the active set."
    )
    return start.date(),min_start.date(),max_start.date(),float(em.get("vat_delta_pp",0.0) or 0.0),float(em.get("excise_delta",0.0) or 0.0),f"Excise change ({unit})",note,False,False,False


@callback(Output("scenario-store","data"), Output("scenario-banner","children"), Input("scenario-apply","n_clicks"), Input("scenario-remove","n_clicks"), Input("scenario-reset-all","n_clicks"), State("vintage-select","value"), State("forecast-select","value"), State("scenario-component-select","value"), State("scenario-start-date","date"), State("scenario-vat-delta","value"), State("scenario-excise-delta","value"), State("scenario-store","data"), prevent_initial_call=True)
def mutate_scenario_set(_apply,_remove,_reset,vintage,forecast_name,model_id,start_date,vat_delta,excise_delta,current_store):
    trigger=ctx.triggered_id
    if trigger=="scenario-reset-all": return clear_scenario_set(vintage=vintage,forecast_name=forecast_name), html.Div("All component tax scenarios cleared.",className="selection-banner")
    if not model_id: return no_update, html.Div("Select a component.",className="banner-error")
    if trigger=="scenario-remove": return remove_scenario_component(current_store,model_id), html.Div(f"{model_spec(model_id).label}: scenario removed.",className="selection-banner")
    if trigger!="scenario-apply": raise PreventUpdate
    if not vintage or not forecast_name or start_date is None: return no_update, html.Div("Vintage, forecast and start date are required.",className="banner-error")
    row=_scenario_component_forecast_row(vintage,model_id,forecast_name)
    if row is None: return no_update, html.Div(f"{model_spec(model_id).label}: no unique promoted/saved run is available.",className="banner-error")
    vat_delta=float(vat_delta or 0.0); excise_delta=float(excise_delta or 0.0)
    if abs(vat_delta)<1e-15 and abs(excise_delta)<1e-15:
        return remove_scenario_component(current_store,model_id), html.Div(f"{model_spec(model_id).label}: zero changes, component removed from the active set.",className="selection-banner")
    try:
        from energy_bvar_io import load_energy_bvar_forecast
        forecast_dir=Path(str(row["directory"])); run_dir=forecast_dir.parent.parent
        forecast=load_energy_bvar_forecast(forecast_dir)
        dataset=PROJECT_ROOT/"data"/"processed"/str(vintage)/model_spec(model_id).dataset_file
        contract=component_tax_scenario_contract(model_id,forecast,dataset_path=dataset)
        frequency=str(contract.get("frequency","monthly"))
        effective_start=_normalise_scenario_period(start_date, frequency)
        min_start=_normalise_scenario_period(contract["min_start"], frequency)
        max_start=_normalise_scenario_period(contract["max_start"], frequency)
        if effective_start < min_start or effective_start > max_start:
            return no_update, html.Div(
                f"Scenario start must lie inside the selected forecast horizon: "
                f"{min_start.date().isoformat()} — {max_start.date().isoformat()}.",
                className="banner-error",
            )
        result=build_saved_component_tax_scenario(run_dir,project_root=PROJECT_ROOT,forecast_name=str(forecast_name),start_date=effective_start,vat_delta_pp=vat_delta,excise_delta=excise_delta,max_draws=500)
        payload=scenario_payload(result); payload.setdefault("meta",{}).update({"vintage":str(vintage),"run_id":str(row["run_id"]),"forecast_name":str(forecast_name)})
        updated=upsert_scenario_component(current_store,payload,vintage=str(vintage),forecast_name=str(forecast_name))
    except Exception as exc:
        return no_update, html.Div([html.Strong("Scenario calculation failed: "),html.Span(str(exc))],className="banner-error")
    meta=payload["meta"]; count=len(scenario_set_components(updated))
    return updated, html.Div([html.Strong(str(result["hicp_label"])),html.Span(f" · VAT {vat_delta:+.2f} pp"),html.Span(f" · excise {excise_delta:+.3f} {meta['excise_unit']}"),html.Span(f" · {meta['n_draws_effective']} paired draws"),html.Span(f" · {count} active component scenario"+("s" if count!=1 else ""))],className="selection-banner")


@callback(
    Output("scenario-set-summary","children"),
    Input("scenario-store","data"),
    Input("scenario-component-select","value"),
)
def render_scenario_set_summary(store, selected_model_id):
    rows=scenario_set_summary(store)
    if not rows:
        return html.Div(
            "No active tax scenario. Choose a component, set VAT and/or excise, then Add / update.",
            className="placeholder-text",
        )
    cards=[]
    for row in rows:
        model_id=str(row["model_id"])
        selected=(str(selected_model_id)==model_id)
        start=pd.to_datetime(row.get("scenario_start"),errors="coerce")
        start_label="—" if pd.isna(start) else pd.Timestamp(start).date().isoformat()
        cards.append(
            html.Button(
                [
                    html.Div(
                        [
                            html.Strong(row["label"]),
                            html.Span("Selected", style={
                                "display":"inline-block" if selected else "none",
                                "marginLeft":"8px",
                                "fontSize":"10px",
                                "fontWeight":"700",
                                "textTransform":"uppercase",
                                "letterSpacing":"0.08em",
                                "color":"#2563eb",
                            }),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(f"start {start_label}"),
                            html.Span(f" · VAT {row['vat_delta_pp']:+.2f} pp"),
                            html.Span(f" · excise {row['excise_delta']:+.3f} {row['excise_unit']}"),
                        ],
                        style={"marginTop":"3px","fontSize":"12px","color":"#64748b"},
                    ),
                ],
                id={"type":"scenario-card","model_id":model_id},
                n_clicks=0,
                title=f"Select {row['label']} to edit it and inspect its component charts",
                style={
                    "display":"block",
                    "width":"100%",
                    "textAlign":"left",
                    "background":"#eff6ff" if selected else "#ffffff",
                    "border":"1px solid #2563eb" if selected else "1px solid #e2e8f0",
                    "borderRadius":"10px",
                    "padding":"10px 12px",
                    "marginTop":"7px",
                    "cursor":"pointer",
                    "color":"#0f172a",
                    "boxShadow":"0 1px 2px rgba(15,23,42,0.04)",
                },
            )
        )
    return cards


@callback(
    Output("scenario-component-select","value", allow_duplicate=True),
    Input({"type":"scenario-card","model_id":ALL},"n_clicks"),
    prevent_initial_call=True,
)
def select_scenario_card(clicks):
    # Dynamic-card creation can trigger the pattern callback with all zeros;
    # only an actual click should change the editor selection.
    if not clicks or not any(int(value or 0) > 0 for value in clicks):
        raise PreventUpdate
    triggered=ctx.triggered_id
    if not isinstance(triggered,dict) or triggered.get("type")!="scenario-card":
        raise PreventUpdate
    model_id=triggered.get("model_id")
    if not model_id:
        raise PreventUpdate
    return str(model_id)


@callback(Output("scenario-baseline-terminal","children"),Output("scenario-terminal-date","children"),Output("scenario-scenario-terminal","children"),Output("scenario-terminal-date-2","children"),Output("scenario-impact-terminal","children"),Output("scenario-impact-interval","children"),Output("scenario-draws","children"),Output("scenario-draws-note","children"),Input("scenario-store","data"),Input("scenario-component-select","value"))
def scenario_stat_cards(store,model_id):
    values=scenario_kpis(scenario_set_payload(store,model_id))
    if values.get("terminal_date") is None: return "—","","—","","—","","—",""
    date=pd.Timestamp(values["terminal_date"]).date().isoformat(); low=values.get("impact_low"); high=values.get("impact_high"); interval="" if low is None or high is None else f"68% [{low:+.2f}, {high:+.2f}] pp"; draws=values.get("n_draws")
    return f"{values['baseline_terminal']:.2f}%",date,f"{values['scenario_terminal']:.2f}%",date,f"{values['impact_terminal']:+.2f} pp",interval,("—" if draws is None else f"{int(draws):,}"),"matched baseline/scenario"


@callback(Output("scenario-main-graph","figure"),Output("scenario-level-impact","figure"),Output("scenario-yoy-impact","figure"),Output("scenario-tax-graph","figure"),Input("scenario-store","data"),Input("scenario-component-select","value"),Input("scenario-fan","value"))
def scenario_figures(store,model_id,fan_mode):
    payload=scenario_set_payload(store,model_id)
    if not payload:
        empty=empty_scenario_figure("Add a tax scenario for the selected component"); return empty,empty,empty,empty
    meta=dict(payload.get("meta",{}) or {}); revision=f"{meta.get('run_id','run')}::{meta.get('forecast_name','forecast')}::{model_id}::scenario"; mode=fan_mode or "68"
    return scenario_main_figure(payload,fan_mode=mode,uirevision=revision+"::main"),scenario_impact_figure(payload,metric="level",fan_mode=mode,uirevision=revision+"::level"),scenario_impact_figure(payload,metric="yoy",fan_mode=mode,uirevision=revision+"::yoy"),scenario_tax_figure(payload,uirevision=revision+"::tax")


@callback(
    Output("scenario-tax-selected-agg-store", "data"),
    Input("scenario-store", "data"),
    Input("scenario-component-select", "value"),
    Input("conditional-agg-select", "value"),
    Input("vintage-select", "value"),
)
def selected_tax_aggregate_marginal(
    tax_store, model_id, aggregate_run_id, vintage
):
    payload = scenario_set_payload(tax_store, model_id)
    if not payload or not model_id or not aggregate_run_id or not vintage:
        return None

    aggregate_row = _selected_aggregate_row(vintage, aggregate_run_id)
    if aggregate_row is None:
        return None
    directory = Path(str(aggregate_row["directory"]))
    metadata = _aggregate_metadata(directory)
    run_ids = _run_ids_from_aggregate_metadata(metadata)
    if not run_ids:
        return {
            "error": "Selected aggregate lacks exact component-run provenance."
        }

    single = clear_scenario_set(
        vintage=str(vintage),
        forecast_name=str(metadata.get("forecast_name") or "unconditional"),
    )
    single = upsert_scenario_component(
        single,
        payload,
        vintage=str(vintage),
        forecast_name=str(metadata.get("forecast_name") or "unconditional"),
    )
    tax_scenarios = scenario_set_to_tax_scenarios(single)
    signature = scenario_set_signature(single)
    cache_key = "selected-tax-agg-v2::" + json.dumps(
        {
            "aggregate": str(aggregate_run_id),
            "model_id": str(model_id),
            "signature": signature,
        },
        sort_keys=True,
        default=str,
    )
    cached = _diskcache.get(cache_key)
    if isinstance(cached, dict):
        return cached

    try:
        outcome = run_aggregate(
            str(vintage),
            project_root=PROJECT_ROOT,
            results_root=RESULTS_ROOT,
            forecast_name=str(metadata.get("forecast_name") or "unconditional"),
            run_ids=run_ids,
            n_aggregate_draws=min(
                500,
                max(
                    1,
                    int(metadata.get("n_aggregate_draws_requested") or 500),
                ),
            ),
            pairing_seed=int(metadata.get("pairing_seed") or 2026),
            tax_scenarios=tax_scenarios,
            weekly_tax_mode="strict",
            persist=False,
        )
        result = aggregate_live_scenario_payload(
            outcome,
            {
                "scenario_count": 1,
                "scenario_components": [str(model_id)],
                "scenario_signature": signature,
            },
        )
        result.setdefault("meta", {}).update(
            {
                "aggregate_run_id": str(aggregate_run_id),
                "selected_tax_model_id": str(model_id),
            }
        )
        _diskcache.set(cache_key, result, expire=3600)
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@callback(
    Output("scenario-tax-energy-graph", "figure"),
    Input("scenario-tax-selected-agg-store", "data"),
    Input("scenario-fan", "value"),
)
def selected_tax_energy_figure(store, fan_mode):
    if not store:
        return empty_aggregate_figure(
            "Add/select a tax scenario to display its marginal HICP Energy effect."
        )
    if store.get("error"):
        return empty_aggregate_figure(str(store["error"]))
    return tidy_aggregate_figure(
        aggregate_live_impact_figure(
            store,
            fan_mode=fan_mode or "68",
            uirevision=(
                f"{store.get('meta',{}).get('aggregate_run_id','agg')}::"
                f"{store.get('meta',{}).get('selected_tax_model_id','tax')}::marginal"
            ),
        )
    )


def _selected_aggregate_row(vintage: str | None, aggregate_run_id: str | None):
    if not vintage or not aggregate_run_id:
        return None
    frame = list_aggregates(REGISTRY_PATH, vintage=str(vintage), present_only=True)
    if frame.empty:
        return None
    selected = frame.loc[frame["aggregate_run_id"].astype(str) == str(aggregate_run_id)]
    return None if selected.empty else selected.iloc[0]


def _aggregate_metadata(directory: Path) -> dict:
    path = Path(directory) / "metadata.json"
    if not path.is_file():
        fallback = Path(directory) / "aggregate_config.json"
        path = fallback if fallback.is_file() else path
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _aggregate_forecast_origin(directory: Path) -> pd.Timestamp | None:
    """Latest first-future month across component stores = aggregate forecast origin."""
    metadata = _aggregate_metadata(directory)
    stores = dict(metadata.get("component_forecast_stores", {}) or {})
    starts: list[pd.Timestamp] = []
    for raw in stores.values():
        path = Path(str(raw)) / "forecast_metadata.json"
        if not path.is_file():
            continue
        try:
            fm = json.loads(path.read_text(encoding="utf-8"))
            future = list(fm.get("future_dates", []) or [])
            if not future:
                continue
            first = pd.Timestamp(future[0]).to_period("M").to_timestamp(how="start")
            starts.append(first)
        except Exception:
            continue
    return max(starts) if starts else None


def _run_ids_from_aggregate_metadata(metadata: dict) -> dict[str, str]:
    """Extract the exact component run IDs used by a saved aggregate."""
    out: dict[str, str] = {}
    for key, raw in dict(metadata.get("component_forecast_stores", {}) or {}).items():
        path = Path(str(raw))
        try:
            run_id = path.parents[1].name  # <run>/forecasts/<forecast_name>
        except IndexError:
            continue
        if run_id:
            out[str(key)] = str(run_id)
    return out


def _tax_scenarios_from_payload(payload: dict | None) -> dict[str, dict]:
    return scenario_set_to_tax_scenarios(payload)


# ---------------------------------------------------------------------------
# Aggregate page callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("agg-select", "options"),
    Output("agg-select", "value"),
    Input("vintage-select", "value"),
    Input("registry-store", "data"),
    State("agg-select", "value"),
)
def aggregate_options(vintage: str | None, _: dict | None, current: str | None):
    """Aggregate runs for the selected vintage.

    Failed stores are listed rather than filtered out. A dropdown that silently
    shows nothing gives no way to tell "none saved" from "saved but rejected by
    the scan", and the registry already holds the reason.
    """
    if not vintage:
        return [], None
    frame = list_aggregates(REGISTRY_PATH, present_only=True)
    if frame.empty:
        return [], None
    frame = frame.loc[frame["vintage"].astype(str) == str(vintage)]
    if frame.empty:
        return [], None
    frame = frame.sort_values(
        ["status", "promoted", "created_at_utc", "aggregate_run_id"],
        ascending=[True, False, False, True],
    )
    options = []
    for _index, row in frame.iterrows():
        tags = []
        if str(row.get("status")) != "complete":
            tags.append(str(row.get("status", "unusable")).upper())
        if bool(row.get("promoted")):
            tags.append("PROMOTED")
        if not bool(row.get("has_display")):
            tags.append("display pending")
        if row.get("scenario_active"):
            tags.append("tax scenario")
        suffix = " · " + " · ".join(tags) if tags else ""
        options.append(
            {
                "label": f"{_short_run(row['aggregate_run_id'])}{suffix}",
                "value": str(row["aggregate_run_id"]),
            }
        )
    values = [item["value"] for item in options]
    usable = frame.loc[frame["status"].astype(str) == "complete", "aggregate_run_id"]
    default = str(usable.iloc[0]) if not usable.empty else values[0]
    return options, current if current in values else default


def _no_aggregate_banner(vintage: str | None) -> html.Div:
    """Say what the registry actually holds instead of showing a blank page."""
    frame = list_aggregates(REGISTRY_PATH, present_only=True)
    if frame.empty:
        return html.Div(
            [
                html.Strong("No aggregate saved. "),
                html.Span(
                    "Nothing was found under results/hicp_energy_aggregate/. "
                    "Use Build Energy aggregate only on the Estimation page, or "
                    "run the full Energy suite. Notebook 10 is now a reference/audit notebook."
                ),
            ],
            className="selection-banner",
        )
    here = frame.loc[frame["vintage"].astype(str) == str(vintage)]
    if here.empty:
        others = ", ".join(sorted(frame["vintage"].astype(str).unique()))
        return html.Div(
            [
                html.Strong(f"No aggregate for vintage {vintage}. "),
                html.Span(f"The registry holds aggregates for: {others}."),
            ],
            className="selection-banner",
        )
    failed = here.loc[here["status"].astype(str) != "complete"]
    if not failed.empty:
        reason = str(failed["error"].dropna().iloc[0]) if failed["error"].notna().any() else "no reason recorded"
        return html.Div(
            [
                html.Strong(f"{len(failed)} aggregate store(s) rejected by the scan: "),
                html.Span(reason),
            ],
            className="banner-error",
        )
    return html.Div("Select an aggregate run.", className="selection-banner")


@callback(
    Output("agg-store", "data"),
    Output("agg-banner", "children"),
    Input("agg-select", "value"),
    Input("vintage-select", "value"),
    Input("registry-store", "data"),
)
def load_aggregate_display(aggregate_run_id: str | None, vintage: str | None, _: dict | None):
    """Read the aggregate forecast display and its independent BVAR-fit cache.

    The accounting display is the primary artefact.  Historical BVAR fitted
    paths are a diagnostic cache built from persisted posterior draws; failure
    to materialise that cache must never make the aggregate forecast unusable.
    """
    if not vintage:
        return None, html.Div("Select a vintage.", className="selection-banner")
    if not aggregate_run_id:
        return None, _no_aggregate_banner(vintage)

    frame_rows = list_aggregates(REGISTRY_PATH, vintage=vintage, present_only=True)
    selected = frame_rows.loc[
        frame_rows["aggregate_run_id"].astype(str) == str(aggregate_run_id)
    ]
    if selected.empty:
        return None, html.Div(
            "The selected aggregate is no longer present in the registry.",
            className="banner-error",
        )
    row = selected.iloc[0]
    if str(row.get("status")) != "complete":
        return None, html.Div(
            [
                html.Strong("This aggregate store was rejected by the scan: "),
                html.Span(str(row.get("error") or "no reason recorded")),
            ],
            className="banner-error",
        )

    directory = Path(str(row["directory"]))
    display_path = directory / DISPLAY_FILENAME
    display_materialised = False
    try:
        if not display_path.is_file():
            if not AUTO_BUILD_DISPLAY:
                raise FileNotFoundError(
                    f"{display_path} is missing and ENERGY_BVAR_AUTO_BUILD_DISPLAY=0."
                )
            build_aggregate_display(directory, project_root=PROJECT_ROOT)
            display_materialised = True
        frame = load_display_artifact(display_path)
        if not (frame["record_type"].astype(str) == "fan").any():
            raise ValueError(
                f"{DISPLAY_FILENAME} loaded but carries no aggregate fan rows."
            )
    except Exception as exc:
        return None, html.Div(
            [html.Strong("Aggregate load failed: "), html.Span(str(exc))],
            className="banner-error",
        )

    # Historical model fit is deliberately separate from display_v1.  It is
    # constructed from saved B coefficient draws and the processed vintage,
    # never from the deterministic HICP accounting reconstruction.
    fitted_frame = pd.DataFrame()
    fitted_meta: dict = {"available": False}
    fitted_materialised = False
    if AUTO_BUILD_FITTED:
        try:
            fitted_frame, fitted_meta, fitted_materialised = load_energy_aggregate_fitted(
                directory,
                project_root=PROJECT_ROOT,
                max_draws=FITTED_MAX_DRAWS,
                pairing_seed=FITTED_PAIRING_SEED,
            )
        except Exception as exc:
            fitted_meta = {
                "available": False,
                "reason": str(exc),
            }
    else:
        fitted_meta = {
            "available": False,
            "reason": "ENERGY_BVAR_AUTO_BUILD_FITTED=0",
        }

    meta = display_metadata(frame)

    # Read the aggregate metadata once for human-facing provenance.  This does
    # not mutate the persisted aggregate contract.
    aggregate_metadata: dict = {}
    try:
        aggregate_metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        meta["aggregate_metadata_read_error"] = str(exc)
    if aggregate_metadata:
        meta["component_run_ids"] = _run_ids_from_aggregate_metadata(
            aggregate_metadata
        )
        for key in (
            "pairing_seed",
            "weight_policy",
            "other_transport_fuels_forecast",
            "predictive_distribution_interpretation",
            "cross_model_dependence",
            "n_aggregate_draws_requested",
            "n_aggregate_draws_effective",
        ):
            if key in aggregate_metadata:
                meta[key] = aggregate_metadata.get(key)

    # Historical contribution diagnostics are accounting objects, not fitted
    # BVAR paths.  Build them once when the aggregate selection changes and
    # keep them in the in-memory store; no results/ artefact is modified.
    historical_contribution_frame = pd.DataFrame()
    try:
        historical_contribution_frame = build_historical_contribution_frame(
            PROJECT_ROOT, str(vintage)
        )
        meta["historical_contributions_available"] = True
        hist_error = pd.to_numeric(
            historical_contribution_frame.get("additivity_error"), errors="coerce"
        ).abs().dropna()
        meta["historical_contribution_additivity_max_error"] = (
            float(hist_error.max()) if len(hist_error) else 0.0
        )
    except Exception as exc:
        meta["historical_contributions_available"] = False
        meta["historical_contributions_reason"] = str(exc)

    # Never trust stale fit flags embedded by an older display builder.  The
    # fitted cache above is now the sole source of truth for model-fit overlays.
    meta["bvar_fitted_available"] = bool(fitted_meta.get("available"))
    meta["bvar_fitted_reason"] = fitted_meta.get("reason")
    meta["bvar_fitted_cache_version"] = fitted_meta.get("cache_version")
    meta["bvar_fitted_draws"] = fitted_meta.get("n_aggregate_fitted_draws")

    origin = _aggregate_forecast_origin(directory)
    if origin is not None:
        meta["forecast_origin"] = pd.Timestamp(origin).isoformat()
    if not meta.get("last_observed_hicp_energy"):
        history = frame.loc[
            (frame["record_type"].astype(str) == "history")
            & (frame["metric"].astype(str) == "yoy")
            & (frame["series"].astype(str) == "hicp_energy")
        ].dropna(subset=["value"])
        if not history.empty:
            meta["last_observed_hicp_energy"] = pd.Timestamp(
                history["date"].max()
            ).isoformat()

    status_bits = []
    if display_materialised:
        status_bits.append("display_v1 materialised now")
    if fitted_materialised:
        status_bits.append("BVAR fitted cache materialised now")
    banner = html.Div(
        [
            html.Strong("HICP Energy"),
            html.Span(f" · vintage {vintage}"),
            html.Span(f" · aggregate {_short_run(aggregate_run_id)}"),
            html.Span(f" · weekly tax: {meta.get('weekly_tax_mode_effective', '—')}"),
            html.Span(" · " + " · ".join(status_bits) if status_bits else ""),
        ],
        className="selection-banner",
    )
    return {
        "frame_json": _json_frame(frame),
        "meta": meta,
        "fitted_frame_json": (
            None if fitted_frame.empty else _json_frame(fitted_frame)
        ),
        "fitted_meta": fitted_meta,
        "historical_contribution_frame_json": (
            None
            if historical_contribution_frame.empty
            else _json_frame(historical_contribution_frame)
        ),
    }, banner


@callback(
    Output("agg-metric", "options"),
    Output("agg-metric", "value"),
    Input("agg-store", "data"),
    State("agg-metric", "value"),
)
def aggregate_metric_dropdown(store: dict | None, current: str | None):
    frame = _frame_from_store(store)
    options, default = aggregate_metric_options(frame)
    values = [item["value"] for item in options]
    return options, current if current in values else default


_FITTED_COMPONENT_LABELS = {
    "car_fuels": "Car fuels",
    "liquid_fuels": "Liquid fuels",
    "gas": "Gas",
    "electricity": "Electricity",
    "heat_energy": "Heat energy",
    "solid_fuels": "Solid fuels",
}
_FITTED_COMPONENT_COLORS = {
    "car_fuels": "#2563EB",
    "liquid_fuels": "#0EA5E9",
    "gas": "#F59E0B",
    "electricity": "#10B981",
    "heat_energy": "#EF4444",
    "solid_fuels": "#64748B",
}


def _fitted_frame_from_store(store: dict | None) -> pd.DataFrame:
    if not store or not store.get("fitted_frame_json"):
        return pd.DataFrame()
    frame = pd.read_json(StringIO(store["fitted_frame_json"]), orient="split")
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _historical_contribution_frame_from_store(store: dict | None) -> pd.DataFrame:
    if not store or not store.get("historical_contribution_frame_json"):
        return pd.DataFrame()
    frame = pd.read_json(
        StringIO(store["historical_contribution_frame_json"]), orient="split"
    )
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _aggregate_figure_base(
    frame: pd.DataFrame,
    *,
    metric: str,
    fan_mode: str,
    uirevision: str,
    meta: dict,
):
    """Render only the saved aggregate forecast contract.

    Legacy helper-level historical overlays are explicitly disabled.  The
    dashboard adds the new fitted cache itself below so old helper modules
    cannot accidentally plot the deterministic accounting reconstruction.
    """
    params = inspect.signature(aggregate_fan_figure).parameters
    kwargs = {
        "metric": metric,
        "fan_mode": fan_mode,
        "uirevision": uirevision,
    }
    if "forecast_origin" in params:
        kwargs["forecast_origin"] = meta.get("forecast_origin")
    if "last_observed" in params:
        kwargs["last_observed"] = meta.get("last_observed_hicp_energy")
    if "show_bvar_reconstruction" in params:
        kwargs["show_bvar_reconstruction"] = False
    if "show_six_model_components" in params:
        kwargs["show_six_model_components"] = False
    return aggregate_fan_figure(frame, **kwargs)


@callback(
    Output("agg-historical-overlays", "options"),
    Output("agg-historical-overlays", "value"),
    Output("agg-overlay-note", "children"),
    Input("agg-store", "data"),
    State("agg-historical-overlays", "value"),
)
def aggregate_overlay_options(store: dict | None, current: list[str] | None):
    frame = _frame_from_store(store)
    if frame.empty:
        return [], [], "Load an HICP Energy aggregate to enable model-fit overlays."

    fitted = _fitted_frame_from_store(store)
    fitted_meta = dict((store or {}).get("fitted_meta", {}) or {})
    fitted_available = bool(fitted_meta.get("available")) and not fitted.empty

    component_available = fitted_available and (
        fitted["basis"].astype(str).eq("component_bvar_fitted").any()
    )
    aggregate_available = fitted_available and (
        fitted["basis"].astype(str).eq("aggregate_bvar_fitted").any()
    )
    options = [
        {
            "label": (
                "Show six component BVAR fitted curves"
                if component_available
                else "Six component BVAR fitted curves — unavailable"
            ),
            "value": "component_fits",
            "disabled": not component_available,
        },
        {
            "label": (
                "Show aggregated BVAR fitted HICP Energy"
                if aggregate_available
                else "Aggregated BVAR fitted HICP Energy — unavailable"
            ),
            "value": "aggregate_fit",
            "disabled": not aggregate_available,
        },
    ]
    allowed = {
        item["value"] for item in options if not bool(item.get("disabled"))
    }
    selected = [value for value in (current or []) if value in allowed]

    if fitted_available:
        note = (
            "Component fitted = posterior one-step conditional BVAR fitted HICP "
            "paths after the same tax/frequency adapters. Aggregated fitted = "
            "draw-wise Laspeyres aggregation of those six paths; quantiles are "
            "computed only after aggregation. The deterministic accounting "
            "reconstruction is audit-only and is not plotted as model fit."
        )
    else:
        reason = str(fitted_meta.get("reason") or "fitted cache not materialised")
        note = "BVAR fitted overlays unavailable. Reason: " + reason
    return options, selected, note


@callback(
    Output("agg-observed", "children"),
    Output("agg-observed-date", "children"),
    Output("agg-first", "children"),
    Output("agg-first-date", "children"),
    Output("agg-terminal", "children"),
    Output("agg-terminal-date", "children"),
    Output("agg-draws", "children"),
    Output("agg-draws-unit", "children"),
    Input("agg-store", "data"),
    Input("agg-metric", "value"),
)
def aggregate_stat_cards(store: dict | None, metric: str | None):
    frame = _frame_from_store(store)
    if frame.empty or not metric:
        return ("—", "", "—", "", "—", "", "—", "")
    values = aggregate_kpis(frame, metric)

    def stamp(value):
        return "" if value is None else pd.Timestamp(value).date().isoformat()

    draws = values.get("n_draws")
    return (
        _format_number(values.get("latest_observed")),
        stamp(values.get("latest_observed_date")),
        _format_number(values.get("first_future")),
        stamp(values.get("first_future_date")),
        _format_number(values.get("terminal")),
        stamp(values.get("terminal_date")),
        "—" if draws is None else f"{draws:,}",
        "effective" if draws is not None else "",
    )


@callback(
    Output("agg-graph", "figure"),
    Input("agg-store", "data"),
    Input("agg-metric", "value"),
    Input("agg-fan", "value"),
    Input("agg-historical-overlays", "value"),
    Input("agg-graph", "relayoutData"),
)
def aggregate_graph(
    store: dict | None,
    metric: str | None,
    fan_mode: str | None,
    historical_overlays: list[str] | None,
    relayout_data: dict | None,
):
    frame = _frame_from_store(store)
    if frame.empty:
        return empty_aggregate_figure("Select a saved aggregate run")
    if not metric:
        return empty_aggregate_figure("Select a metric")

    meta = dict((store or {}).get("meta", {}) or {})
    revision = f"{meta.get('aggregate_run_id', 'agg')}::{metric}"
    overlays = set(historical_overlays or [])
    fig = _aggregate_figure_base(
        frame,
        metric=str(metric),
        fan_mode=fan_mode or "68",
        uirevision=revision,
        meta=meta,
    )
    fig = tidy_aggregate_figure(fig, uirevision=revision)

    if str(metric) not in {"level", "yoy"}:
        return fig

    fitted = _fitted_frame_from_store(store)
    if fitted.empty:
        return fig

    if "component_fits" in overlays:
        rows = fitted.loc[
            (fitted["basis"].astype(str) == "component_bvar_fitted")
            & (fitted["metric"].astype(str) == str(metric))
        ].copy()
        for series, label in _FITTED_COMPONENT_LABELS.items():
            part = rows.loc[rows["series"].astype(str) == series].sort_values("date")
            part = part.dropna(subset=["date", "q50"])
            if part.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=part["date"],
                    y=part["q50"].astype(float),
                    mode="lines",
                    name=f"{label} BVAR fitted",
                    line={
                        "width": 1.55,
                        "dash": "dot",
                        "color": _FITTED_COMPONENT_COLORS.get(series),
                    },
                    hovertemplate=f"%{{y:.2f}}<extra>{label} BVAR fitted</extra>",
                )
            )

    if "aggregate_fit" in overlays:
        part = fitted.loc[
            (fitted["basis"].astype(str) == "aggregate_bvar_fitted")
            & (fitted["series"].astype(str) == "hicp_energy")
            & (fitted["metric"].astype(str) == str(metric))
        ].sort_values("date")
        part = part.dropna(subset=["date", "q50"])
        if not part.empty:
            fig.add_trace(
                go.Scatter(
                    x=part["date"],
                    y=part["q50"].astype(float),
                    mode="lines",
                    name="Aggregated BVAR fitted HICP Energy",
                    line={"width": 2.7, "dash": "dash", "color": "#7C3AED"},
                    hovertemplate=(
                        "%{y:.2f}<extra>Aggregated BVAR fitted HICP Energy</extra>"
                    ),
                )
            )

    # Recompute the vertical range from the *visible* horizontal window.
    # This prevents large historical fitted spikes from being clipped after
    # zooming while keeping a much more readable local scale.
    fig = tidy_aggregate_figure(
        fig,
        height=680,
        uirevision=revision,
    )
    fig = adapt_yaxis_to_visible_window(
        fig,
        relayout_data,
        include_zero=(str(metric) == "yoy"),
        padding_fraction=0.08,
    )
    return fig


@callback(
    Output("agg-contrib", "figure"),
    Input("agg-store", "data"),
    Input("agg-contrib-mode", "value"),
    Input("agg-contrib-labels", "value"),
)
def aggregate_contributions(
    store: dict | None,
    display_mode: str | None,
    label_options: list[str] | None,
):
    frame = _frame_from_store(store)
    if frame.empty:
        return empty_aggregate_figure("Select a saved aggregate run")
    historical = _historical_contribution_frame_from_store(store)
    meta = dict((store or {}).get("meta", {}) or {})
    revision = f"{meta.get('aggregate_run_id', 'agg')}::contrib"
    labels = "latest" in set(label_options or [])
    return aggregate_contribution_timeline_figure(
        frame,
        historical,
        display_mode=display_mode or "bars",
        show_labels=labels,
        forecast_origin=meta.get("forecast_origin"),
        uirevision=revision + "::timeline",
    )


@callback(
    Output("agg-all-scenarios-summary", "children"),
    Input("conditional-store", "data"),
    Input("scenario-store", "data"),
    Input("agg-select", "value"),
)
def aggregate_all_scenario_summary(
    conditional_store, tax_store, aggregate_run_id
):
    conditional_rows = conditional_set_summary(conditional_store)
    tax_rows = scenario_set_summary(tax_store)
    if not conditional_rows and not tax_rows:
        return "No active conditional or tax scenario."

    pieces = []
    for row in conditional_rows:
        impact = row.get("aggregate_terminal_impact")
        impact_label = (
            ""
            if impact is None
            else f" · terminal {float(impact):+.3f} pp"
        )
        pieces.append(
            html.Div(
                [
                    html.Strong("Conditional · " + row["label"]),
                    html.Span(
                        f" · {row['condition_variable'].replace('_',' ')} "
                        f"{row['condition_description']}{impact_label}"
                    ),
                ],
                style={"marginBottom": "4px"},
            )
        )
    for row in tax_rows:
        pieces.append(
            html.Div(
                [
                    html.Strong("Tax · " + row["label"]),
                    html.Span(
                        f" · VAT {row['vat_delta_pp']:+.2f} pp"
                        f" · excise {row['excise_delta']:+.3f} {row['excise_unit']}"
                    ),
                ],
                style={"marginBottom": "4px"},
            )
        )
    pieces.append(
        html.Div(
            "Conditional lines below are standalone marginal effects against the saved "
            "baseline. Active tax scenarios are propagated jointly in the next panel; "
            "marginal effects are not summed.",
            className="control-help",
        )
    )
    return pieces


@callback(
    Output("agg-conditional-scenarios-impact", "figure"),
    Input("conditional-store", "data"),
    Input("agg-fan", "value"),
    Input("agg-select", "value"),
)
def aggregate_conditional_scenario_impacts(
    conditional_store, fan_mode, aggregate_run_id
):
    if not conditional_set_summary(conditional_store):
        return empty_aggregate_figure(
            "No active conditional scenario. Add one or more in Scenarios."
        )

    set_aggregate = str((conditional_store or {}).get("aggregate_run_id") or "")
    if set_aggregate and aggregate_run_id and set_aggregate != str(aggregate_run_id):
        return empty_aggregate_figure(
            "Active conditional scenarios belong to a different aggregate run."
        )
    return tidy_aggregate_figure(
        scenario_marginal_impact_figure(
            conditional_store,
            fan_mode=fan_mode or "68",
            uirevision=f"{aggregate_run_id or 'agg'}::conditional-marginals",
        ),
        uirevision=f"{aggregate_run_id or 'agg'}::conditional-marginals",
    )


@callback(
    Output("agg-scenario-store", "data"),
    Output("agg-scenario-banner", "children"),
    Input("url", "pathname"),
    Input("agg-select", "value"),
    Input("vintage-select", "value"),
    Input("scenario-store", "data"),
    Input("agg-store", "data"),
)
def compute_live_aggregate_tax_scenario(pathname,aggregate_run_id,vintage,scenario_store,aggregate_store):
    if (pathname or "/forecast")!="/aggregate": raise PreventUpdate
    if not aggregate_run_id or not vintage: return None,"Select an aggregate run."
    summaries=scenario_set_summary(scenario_store)
    if not summaries: return None,"No active component tax scenario. Configure one or more components in Scenarios."
    set_vintage=None if not scenario_store else scenario_store.get("vintage")
    if set_vintage is None:
        fp=scenario_set_payload(scenario_store,summaries[0]["model_id"]); set_vintage=dict((fp or {}).get("meta",{}) or {}).get("vintage")
    if set_vintage is not None and str(set_vintage)!=str(vintage): return None,f"The active scenario set belongs to vintage {set_vintage}; selected aggregate is {vintage}."
    tax_scenarios=_tax_scenarios_from_payload(scenario_store)
    if not tax_scenarios: return None,"The active scenario set contains no non-zero VAT or excise change."
    row=_selected_aggregate_row(vintage,aggregate_run_id)
    if row is None: return None,"The selected aggregate is unavailable."
    directory=Path(str(row["directory"])); metadata=_aggregate_metadata(directory); run_ids=_run_ids_from_aggregate_metadata(metadata)
    if not run_ids: return None,"This aggregate does not record its component forecast stores, so an exact live propagation cannot be reproduced."
    n_requested=int(metadata.get("n_aggregate_draws_requested") or 500); n_draws=min(500,max(1,n_requested)); pairing_seed=int(metadata.get("pairing_seed") or 2026); forecast_name=str(metadata.get("forecast_name") or "unconditional")
    set_forecast=None if not scenario_store else scenario_store.get("forecast_name")
    if set_forecast and str(set_forecast)!=forecast_name: return None,f"The active scenario set was built on {set_forecast!r}, while this aggregate uses {forecast_name!r}."
    signature=scenario_set_signature(scenario_store); starts={str(x["label"]):x.get("scenario_start") for x in summaries if x.get("scenario_start") is not None}; parsed=[pd.Timestamp(x) for x in starts.values()]
    combined_meta={"scenario_start":min(parsed).isoformat() if parsed else None,"scenario_starts":starts,"scenario_components":[x["model_id"] for x in summaries],"scenario_count":len(summaries),"scenario_signature":signature}
    cache_key="agg-tax-scenario-v2::"+json.dumps({"aggregate_run_id":str(aggregate_run_id),"vintage":str(vintage),"scenario_set":signature,"forecast_name":forecast_name,"n_draws":n_draws,"pairing_seed":pairing_seed},sort_keys=True,default=str)
    cached=_diskcache.get(cache_key)
    if isinstance(cached,dict): payload=cached
    else:
        try:
            outcome=run_aggregate(str(vintage),project_root=PROJECT_ROOT,results_root=RESULTS_ROOT,forecast_name=forecast_name,run_ids=run_ids,n_aggregate_draws=n_draws,pairing_seed=pairing_seed,tax_scenarios=tax_scenarios,weekly_tax_mode="strict",persist=False)
            payload=aggregate_live_scenario_payload(outcome,combined_meta); payload.setdefault("meta",{}).update({"aggregate_run_id":str(aggregate_run_id),"aggregate_forecast_name":forecast_name}); _diskcache.set(cache_key,payload,expire=3600)
        except Exception as exc:
            return None,html.Div([html.Strong("HICP Energy scenario propagation failed: "),html.Span(str(exc))],className="banner-error")
    parts=[html.Strong(f"Combined scenario · {len(summaries)} component"+("s" if len(summaries)!=1 else ""))]
    for item in summaries: parts.append(html.Span(f" · {item['label']}: VAT {item['vat_delta_pp']:+.2f} pp, excise {item['excise_delta']:+.3f} {item['excise_unit']}"))
    parts.append(html.Span(f" · propagated with {payload.get('meta',{}).get('n_draws','—')} paired aggregate draws"))
    return payload,html.Div(parts,className="selection-banner")


@callback(
    Output("agg-scen-baseline", "children"),
    Output("agg-scen-date", "children"),
    Output("agg-scen-scenario", "children"),
    Output("agg-scen-date-2", "children"),
    Output("agg-scen-impact", "children"),
    Output("agg-scen-interval", "children"),
    Output("agg-scen-draws", "children"),
    Output("agg-scen-draws-note", "children"),
    Input("agg-scenario-store", "data"),
)
def aggregate_scenario_cards(store: dict | None):
    values = aggregate_live_scenario_kpis(store)
    if values.get("date") is None:
        return "—", "", "—", "", "—", "", "—", ""
    date = pd.Timestamp(values["date"]).date().isoformat()
    return (
        f"{values['baseline']:.2f}%", date,
        f"{values['scenario']:.2f}%", date,
        f"{values['impact']:+.2f} pp",
        f"68% [{values['low']:+.2f}, {values['high']:+.2f}] pp",
        f"{int(values['n_draws']):,}" if values.get("n_draws") is not None else "—",
        "matched baseline/scenario",
    )


@callback(
    Output("agg-scenario-graph", "figure"),
    Output("agg-scenario-impact", "figure"),
    Output("agg-scenario-contrib-impact", "figure"),
    Input("agg-scenario-store", "data"),
    Input("agg-store", "data"),
    Input("agg-fan", "value"),
)
def aggregate_scenario_figures(
    scenario_store: dict | None, aggregate_store: dict | None, fan_mode: str | None
):
    if not scenario_store:
        empty = empty_aggregate_figure("Configure a component tax scenario in Scenarios")
        return empty, empty, empty
    frame = _frame_from_store(aggregate_store)
    meta = (aggregate_store or {}).get("meta", {})
    revision = f"{meta.get('aggregate_run_id', 'agg')}::live-tax"
    main = aggregate_live_scenario_figure(
        scenario_store, frame, fan_mode=fan_mode or "68",
        forecast_origin=meta.get("forecast_origin"),
        uirevision=revision + "::main",
    )
    impact = aggregate_live_impact_figure(
        scenario_store, fan_mode=fan_mode or "68", uirevision=revision + "::impact"
    )
    contribution = aggregate_live_contribution_impact_figure(
        scenario_store, uirevision=revision + "::contrib"
    )
    return (
        tidy_aggregate_figure(main, uirevision=revision + "::main"),
        tidy_aggregate_figure(impact, uirevision=revision + "::impact"),
        tidy_aggregate_figure(contribution, uirevision=revision + "::contrib"),
    )


@callback(
    Output("agg-table", "data"),
    Input("agg-store", "data"),
)
def aggregate_table(store: dict | None):
    frame = _frame_from_store(store)
    if frame.empty:
        return []
    meta = dict((store or {}).get("meta", {}) or {})
    return aggregate_provenance_table_v2(frame, meta).to_dict("records")


if __name__ == "__main__":
    host = os.getenv("ENERGY_BVAR_DASH_HOST", "127.0.0.1")
    port = int(os.getenv("ENERGY_BVAR_DASH_PORT", "8050"))
    debug = os.getenv("ENERGY_BVAR_DASH_DEBUG", "0") in {"1", "true", "True"}
    app.run(host=host, port=port, debug=debug)
