"""Dash front-end for the ECB-style Energy BVAR suite.

Dashboard foundation
---------------------
* persistent URL routing;
* central context store: vintage / model_id / run_id / forecast_name / draw_mode;
* one immutable registry/result snapshot, refreshed only by explicit user action;
* SQLite-backed selectors resolved from the frozen snapshot;
* display artefacts loaded once per snapshot/run and reused from server memory;
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
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

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

from dashboard_snapshot_cache import (
    clear_all as snapshot_clear_all,
    frame_from_store as snapshot_frame_from_store,
    get as snapshot_get,
    get_or_build as snapshot_get_or_build,
    put as snapshot_put,
    put_frame as snapshot_put_frame,
)

from inflation_table_contract import (  # noqa: E402
    DUAL_IMPACT_COLUMNS,
    FORECAST_COLUMNS,
    HD_COLUMNS,
    IRF_COLUMNS,
    PAIRED_EFFECT_COLUMNS,
    dual_impact_records,
    forecast_summary_records,
    paired_effect_records,
    readable_table,
    structural_fevd_table,
    structural_hd_records,
    structural_irf_records,
)


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

# ---------------------------------------------------------------------------
# Optional Economic Data feature boundary
# ---------------------------------------------------------------------------
#
# The feature is deliberately isolated from Energy / Headline / Core.  Setting
# DASH_ENABLE_ECONOMIC_DATA=0 disables it without importing its package.  If the
# package directory has been physically removed, the dashboard also degrades
# cleanly to the core application.
_ECONOMIC_DATA_REQUESTED = os.getenv(
    "DASH_ENABLE_ECONOMIC_DATA", "1"
).strip().lower() not in {"0", "false", "no", "off"}

ECONOMIC_DATA_ENABLED = False
_economic_data_page = None
_register_economic_data_callbacks = None

if _ECONOMIC_DATA_REQUESTED:
    try:
        from economic_data import (  # noqa: E402
            economic_data_page as _economic_data_page,
            register_callbacks as _register_economic_data_callbacks,
        )
        ECONOMIC_DATA_ENABLED = True
    except ModuleNotFoundError as exc:
        if not str(exc.name or "").startswith("economic_data"):
            raise
        print(
            "Economic Data feature package not found; "
            "continuing without /economic-data."
        )

from energy_bvar_dashboard_aggregate import (  # noqa: E402
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
    aggregate_contribution_timeline_figure,
    aggregate_provenance_table_v2,
    build_historical_contribution_frame,
    tidy_aggregate_figure,
)
from energy_bvar_tax_decomposition import (  # noqa: E402
    COMPONENTS as TAX_COMPONENTS,
    COMPONENT_LABELS as TAX_COMPONENT_LABELS,
    build_tax_contribution_decomposition,
    horizon_options as tax_horizon_options,
    tax_contribution_figure,
    tax_matrix_period_title,
    tax_matrix_records,
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
    compute_structural_volatility_v1,
    compute_structural_irf_fevd_v1,
    compute_structural_hd_v1,
    fevd_figure,
    historical_decomposition_figure,
    irf_figure,
    resolve_run_directory as resolve_structural_run_directory,
    structural_run_contract,
    reference_regime_date,
    reference_volatility_snapshot,
    volatility_sparkline_figure,
    relative_volatility_state_figure,
)
from energy_bvar_dashboard_conditional import (  # noqa: E402
    CONDITIONAL_CONTRACT_VERSION,
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
    conditional_set_summary,
    conditional_target_figure,
    empty_conditional_figure,
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
    scenario_frames,
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
from headline_bvar_conditional import (  # noqa: E402
    ENERGY_BRIDGE_CONTRACT_VERSION,
    energy_outcome_headline_bridge,
)
from energy_bvar_fitted import (  # noqa: E402
    load_energy_aggregate_fitted,
)
from inflation_bvar_display import (  # noqa: E402
    DISPLAY_FILENAME,
    build_aggregate_display,
    build_component_display,
    display_metadata,
    load_display_artifact,
)
from energy_bvar_pipeline import (  # noqa: E402
    CANONICAL_MODEL_IDS,
    build_panel,
    model_spec,
    planned_run_metadata,
    resolve_common_vintage,
    run_component,
    vintage_coverage,
)
from energy_bvar_model import BVARSVOPriorConfig, SamplerConfig  # noqa: E402
from headline_bvar_pipeline import (  # noqa: E402
    available_vintages as headline_available_vintages,
    build_inputs as build_headline_inputs,
)
from inflation_bvar_registry import (  # noqa: E402
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

from inflation_dashboard_shell import (  # noqa: E402
    HEADLINE_MODEL_ID,
    domain_from_path,
    model_label,
    models_for_domain,
)

# HEADLINE DASHBOARD SLICE 2
from headline_bvar_dashboard import (  # noqa: E402
    root_diagnostic_records,
    mcmc_diagnostic_records,
    validation_diagnostic_records,
    stability_diagnostic_kpis,
)
from headline_bvar_dashboard_structural import (  # noqa: E402
    headline_structural_page,
    register_headline_structural_callbacks,
)

# HEADLINE DASHBOARD SLICE 3
# HEADLINE DASHBOARD SLICE 4
# HEADLINE DASHBOARD SLICE 6
from headline_dashboard_slice6 import (  # noqa: E402
    headline_scenarios_page,
    register_headline_slice6_callbacks,
)

from headline_dashboard_slice5 import (  # noqa: E402
    headline_estimation_v2_page,
    headline_forecast_v2_page,
    register_headline_slice5_callbacks,
)
from headline_core_dashboard import (  # noqa: E402
    core_forecast_page,
    core_scenarios_page,
    register_core_dashboard_callbacks,
)
from inflation_overview_dashboard import (  # noqa: E402
    overview_page,
    register_overview_callbacks,
)

# HEADLINE DASHBOARD SLICE 5
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


def _component_run_reusable(state: Mapping[str, Any]) -> bool:
    """True only when the exact planned run has every production artefact.

    A saved posterior + raw forecast is not sufficient for Energy production:
    the component HICP bridge must also be present because the aggregate and
    scenarios consume that store. Display parquet is deliberately excluded
    because it is cheap and can be rematerialised without Gibbs.
    """
    return bool(
        state.get("has_draws")
        and state.get("has_forecast")
        and state.get("has_hicp")
    )


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


_ACTIVE_REGISTRY_SNAPSHOT: dict[str, Any] = {
    "snapshot_id": None,
    "forecasts": pd.DataFrame(),
    "runs": pd.DataFrame(),
    "aggregates": pd.DataFrame(),
    "production_inventory": [],
}


def _registry_table(name: str) -> pd.DataFrame:
    frame = _ACTIVE_REGISTRY_SNAPSHOT.get(str(name))
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _registry_snapshot_id() -> str:
    return str(_ACTIVE_REGISTRY_SNAPSHOT.get("snapshot_id") or "bootstrap")


def _frozen_production_inventory() -> list[dict]:
    return [
        dict(item)
        for item in (_ACTIVE_REGISTRY_SNAPSHOT.get("production_inventory") or [])
    ]


def _scan_registry() -> dict[str, Any]:
    """Build the immutable base snapshot at startup or explicit refresh."""
    global _ACTIVE_REGISTRY_SNAPSHOT
    try:
        report = scan_results(
            results_root=RESULTS_ROOT,
            registry_path=REGISTRY_PATH,
        )
        forecasts = list_forecasts(
            REGISTRY_PATH,
            valid_only=True,
            present_only=True,
        ).copy()
        runs = list_runs(
            REGISTRY_PATH,
            present_only=True,
        ).copy()
        aggregates = list_aggregates(
            REGISTRY_PATH,
            present_only=True,
        ).copy()
        production_inventory = _production_vintage_inventory()
        snapshot_id = pd.Timestamp.utcnow().isoformat()

        snapshot_clear_all()
        snapshot_put("registry_table", "forecasts", forecasts)
        snapshot_put("registry_table", "runs", runs)
        snapshot_put("registry_table", "aggregates", aggregates)
        snapshot_put(
            "production_inventory",
            "active",
            [dict(item) for item in production_inventory],
        )
        _ACTIVE_REGISTRY_SNAPSHOT = {
            "snapshot_id": snapshot_id,
            "forecasts": forecasts,
            "runs": runs,
            "aggregates": aggregates,
            "production_inventory": production_inventory,
        }
        return {
            "ok": True,
            "snapshot_id": snapshot_id,
            "revision": snapshot_id,
            "message": (
                f"{report.component_runs_seen} runs · {report.forecasts_seen} forecasts · "
                f"{report.aggregates_seen} aggregates · snapshot frozen"
            ),
            "unexpected": list(report.unexpected_directories),
        }
    except Exception as exc:
        previous = _registry_snapshot_id()
        return {
            "ok": False,
            "snapshot_id": previous,
            "revision": previous,
            "message": f"Registry refresh failed; previous snapshot kept: {exc}",
            "unexpected": [],
        }


def _json_frame(frame: pd.DataFrame) -> str:
    return frame.to_json(orient="split", date_format="iso", double_precision=15)


def _frame_from_store(store: dict | None) -> pd.DataFrame:
    return snapshot_frame_from_store(store)


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
    for column in ("value", "q05", "q16", "q50", "q84", "q95"):
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
                x=now_line["date"], y=now_line["value"], mode="lines", name="Nowcast posterior mean",
                line={"width": 2.4, "color": now_color},
                hovertemplate="%{x|%Y-%m-%d}<br>Nowcast: %{y:.3f}<extra></extra>",
            )
        )
        anchor_date = pd.Timestamp(now_line["date"].iloc[-1])
        anchor_value = float(now_line["value"].iloc[-1])

    fc_line = _anchor_fan_line(forecast, anchor_date, anchor_value)
    if not fc_line.empty:
        fig.add_trace(
            go.Scatter(
                x=fc_line["date"], y=fc_line["value"], mode="lines", name="Forecast posterior mean",
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
        margin={"l": 54, "r": 24, "t": 82, "b": 42},
        height=520,
        title={"text": label, "x": 0.01, "xanchor": "left", "font": {"size": 18, "color": "#111827"}},
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#374151", "size": 12},
        xaxis_title=None, yaxis_title=unit, hovermode="x unified", dragmode="pan",
        hoverlabel={"bgcolor": "white", "bordercolor": "#e5e7eb", "font": {"color": "#111827"}},
        legend={"orientation": "h", "y": 1.13, "x": 1, "xanchor": "right", "font": {"size": 11}},
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
                    _stat_card("First future mean", "stat-first", "stat-first-date"),
                    _stat_card("Terminal mean", "stat-terminal", "stat-terminal-date"),
                    _stat_card("Forecast horizon", "stat-horizon", "stat-horizon-unit"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    html.Div([html.Div("Key results", className="eyebrow"), html.H3("Readable outlook", className="panel-title"), html.P("M+1 / M+2 / M+3 are prioritised; M+6 / M+12 appear only when the saved path reaches them. Central forecasts are posterior means.", className="panel-subtitle")], className="panel-heading"),
                    readable_table("forecast-summary-table", FORECAST_COLUMNS, page_size=7),
                ],
                className="panel table-panel",
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
                            {"name": "Mean", "id": "value", "type": "numeric"},
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
                    _stat_card("First future mean", "agg-first", "agg-first-date"),
                    _stat_card("Terminal mean", "agg-terminal", "agg-terminal-date"),
                    _stat_card("Posterior draws", "agg-draws", "agg-draws-unit"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    html.Div([html.Div("Key results", className="eyebrow"), html.H3("HICP Energy outlook", className="panel-title"), html.P("Posterior mean and uncertainty at the priority month-ahead horizons.", className="panel-subtitle")], className="panel-heading"),
                    readable_table("agg-summary-table", FORECAST_COLUMNS, page_size=7),
                ],
                className="panel table-panel",
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
                                        "the model path uses saved draw-wise posterior contribution means.",
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
                            html.Div(
                                [
                                    html.H3("Tax-layer contribution decomposition", className="panel-title"),
                                    html.P(
                                        "Contributions in percentage points to year-on-year HICP Energy inflation. "
                                        "VAT is the final legal layer, so VAT-on-excise is included in VAT. "
                                        "A tax layer is split only when the bridge is identified in both the selected "
                                        "month and the same month one year earlier; otherwise the exact contribution "
                                        "remains Unsplit.",
                                        className="panel-subtitle",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("Component", className="control-label"),
                                            dcc.Dropdown(
                                                id="agg-tax-component",
                                                options=[
                                                    {"label": "All HICP Energy", "value": "all"},
                                                    *[
                                                        {
                                                            "label": TAX_COMPONENT_LABELS.get(name, name),
                                                            "value": name,
                                                        }
                                                        for name in TAX_COMPONENTS
                                                    ],
                                                ],
                                                value="all",
                                                clearable=False,
                                                searchable=False,
                                                style={"minWidth": "190px"},
                                            ),
                                        ],
                                        className="control-block",
                                    ),
                                    html.Div(
                                        [
                                            html.Span("Display", className="control-label"),
                                            dcc.RadioItems(
                                                id="agg-tax-mode",
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
                                            html.Span("Table period", className="control-label"),
                                            dcc.RadioItems(
                                                id="agg-tax-horizon",
                                                options=[],
                                                value=None,
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
                    html.Div(id="agg-tax-note", className="selection-banner"),
                    dcc.Loading(
                        dcc.Graph(id="agg-tax-contrib", config=_AGG_GRAPH_CONFIG),
                        type="circle",
                    ),
                    html.H4(
                        id="agg-tax-table-title",
                        children="Tax-layer contribution — select a table period",
                        style={
                            "margin": "12px 0 6px",
                            "fontSize": "14px",
                            "fontWeight": "700",
                            "color": "#111827",
                        },
                    ),
                    dash_table.DataTable(
                        id="agg-tax-table",
                        columns=[
                            {"name": "Component", "id": "Component"},
                            {"name": "Pre-tax / market", "id": "Pre-tax / market"},
                            {"name": "Excise", "id": "Excise"},
                            {"name": "VAT", "id": "VAT"},
                            {"name": "Unsplit", "id": "Unsplit"},
                            {"name": "Tax total", "id": "Tax total"},
                            {"name": "Total contribution", "id": "Total contribution"},
                        ],
                        data=[],
                        page_size=8,
                        style_as_list_view=True,
                        style_cell={
                            "fontFamily": "Inter, Segoe UI, sans-serif",
                            "textAlign": "right",
                            "padding": "8px 10px",
                            "whiteSpace": "nowrap",
                        },
                        style_cell_conditional=[
                            {
                                "if": {"column_id": "Component"},
                                "textAlign": "left",
                                "fontWeight": "600",
                            }
                        ],
                        style_header={
                            "fontWeight": "650",
                            "backgroundColor": "#F8FAFC",
                            "borderBottom": "1px solid #CBD5E1",
                        },
                    ),
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
                    html.Div([html.Div("Key scenario results", className="eyebrow"), html.H3("Conditional effect by horizon", className="panel-title"), html.P("Component and HICP Energy effects from the same paired conditional draws used by the charts.", className="panel-subtitle")], className="panel-heading"),
                    readable_table("conditional-effect-table", DUAL_IMPACT_COLUMNS, page_size=7),
                ],
                className="panel table-panel",
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
            html.Div(id="scenario-tax-preview", className="selection-banner"),
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
                    html.Div([html.Div("Key scenario results", className="eyebrow"), html.H3("Tax scenario effect by horizon", className="panel-title"), html.P("Legacy tax artefacts persist quantiles rather than a posterior mean; central values here are therefore the saved posterior medians.", className="panel-subtitle")], className="panel-heading"),
                    readable_table("scenario-effect-table", PAIRED_EFFECT_COLUMNS, page_size=7),
                ],
                className="panel table-panel",
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
    """Recursive structural analysis with an explicit joint-SV reference state."""
    badge_style = {
        "display": "inline-flex", "alignItems": "center", "gap": "6px",
        "padding": "6px 10px", "borderRadius": "999px",
        "border": "1px solid #DCE3EC", "background": "#F8FAFC",
        "fontSize": "12px", "fontWeight": 600, "color": "#334155",
    }
    effect_chip_style = {
        "display": "inline-flex", "alignItems": "center", "padding": "7px 10px",
        "borderRadius": "10px", "border": "1px solid #E2E8F0",
        "background": "#FFFFFF", "fontSize": "12px", "fontWeight": 600,
    }
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Structural analysis", className="page-title"),
                            html.P(
                                "Recursive structural analysis from the selected saved posterior. "
                                "The reference date selects the full vector of structural stochastic "
                                "variances Λₜ used by 1σ IRFs and FEVD; recursive historical decomposition "
                                "uses the realised volatility path and is reference-date invariant.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span("Recursive / Cholesky", style=badge_style),
                            html.Span("Regular SV state", style=badge_style),
                            html.Span("Outlier scale excluded from IRF / FEVD", style=badge_style),
                        ],
                        style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "justifyContent": "flex-end"},
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(id="structural-run-banner", className="selection-banner"),

            # Compact computation controls.  The reference date itself lives in
            # the dedicated joint-SV state panel below.
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Horizon", className="control-label"),
                                    dcc.Input(
                                        id="structural-horizon", type="number", min=1,
                                        max=STRUCTURAL_MAX_HORIZON, step=1,
                                        value=STRUCTURAL_DEFAULT_HORIZON,
                                        className="est-profile-name-input",
                                    ),
                                ], className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label("Posterior draws", className="control-label"),
                                    dcc.Input(
                                        id="structural-draws", type="number", min=1,
                                        max=MAX_STRUCTURAL_DRAWS, step=1,
                                        value=DEFAULT_STRUCTURAL_DRAWS,
                                        className="est-profile-name-input",
                                    ),
                                ], className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label("IRF shock definition", className="control-label"),
                                    dcc.RadioItems(
                                        id="structural-shock-unit",
                                        options=[
                                            {"label": "Standard deviation", "value": "structural_std"},
                                            {"label": "Level impact", "value": "level"},
                                        ],
                                        value=DEFAULT_SHOCK_UNIT,
                                        inline=True,
                                        className="fan-radio",
                                    ),
                                ], className="control-block wide-control",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Magnitude",
                                        id="structural-shock-size-label",
                                        className="control-label",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="structural-shock-size",
                                                type="number",
                                                min=1e-6,
                                                step=0.1,
                                                value=float(DEFAULT_SHOCK_SIZE),
                                                className="est-profile-name-input",
                                                style={"minWidth": "110px", "width": "100%"},
                                            ),
                                            html.Span(
                                                "σ",
                                                id="structural-shock-size-unit",
                                                style={
                                                    "fontSize": "12px",
                                                    "fontWeight": 600,
                                                    "whiteSpace": "nowrap",
                                                    "color": "#6B6E72",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "8px",
                                        },
                                    ),
                                ], className="control-block",
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "minmax(110px,.45fr) minmax(130px,.5fr) minmax(250px,1fr) minmax(180px,.75fr)",
                            "gap": "14px", "alignItems": "end",
                        },
                    ),
                    html.Div(
                        id="structural-shock-interpretation",
                        className="selection-banner",
                        style={"marginTop": "12px", "marginBottom": "2px"},
                    ),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="structural-hd-options",
                                options=[{
                                    "label": "Split realised outlier amplification in historical decomposition",
                                    "value": "split_outliers",
                                }],
                                value=["split_outliers"], className="estimation-checklist",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Refresh structural analysis", id="structural-run",
                                        n_clicks=0, className="refresh-button estimation-run-all-button",
                                    ),
                                    html.Button(
                                        "Cancel", id="structural-cancel", n_clicks=0,
                                        disabled=True, className="estimation-cancel-button",
                                    ),
                                ], className="estimation-actions",
                            ),
                        ], className="estimation-run-row",
                    ),
                    html.Div(
                        [
                            html.Progress(id="structural-progress", value=0, max=100, className="estimation-progress"),
                            html.Div(
                                [
                                    html.Div("Idle", id="structural-phase", className="estimation-phase"),
                                    html.Div(
                                        "Opening Structural automatically loads the default volatility, IRF, FEVD and HD once. Results remain frozen until a structural parameter changes or Refresh structural analysis is pressed.",
                                        id="structural-progress-detail", className="estimation-progress-detail",
                                    ),
                                ], className="estimation-progress-text",
                            ),
                        ], className="estimation-progress-wrap",
                    ),
                ],
                className="panel",
            ),

            # Joint structural-volatility state selector.
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Reference structural-volatility state", className="panel-title"),
                                    html.P(
                                        "There is no single BVAR volatility parameter. Each structural shock has its own "
                                        "λⱼ,ₜ. A reference date selects all of them simultaneously. The cards show √λ in "
                                        "native units and a scale-free ratio to each series' own sample median.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Exact reference date", className="control-label"),
                                    dcc.Dropdown(
                                        id="structural-reference-date", options=[], value=None,
                                        clearable=False, className="compact-dropdown wide-control",
                                    ),
                                ], className="control-block wide-control",
                            ),
                        ], className="panel-heading",
                    ),
                    html.Div(
                        [
                            html.Div("Regime", className="control-label"),
                            dcc.RadioItems(
                                id="structural-reference-regime",
                                options=[
                                    {"label": "Latest", "value": "latest"},
                                    {"label": "Calm · P10", "value": "p10"},
                                    {"label": "Median · P50", "value": "p50"},
                                    {"label": "Stressed · P90", "value": "p90"},
                                    {"label": "Peak 2022", "value": "peak_2022"},
                                ],
                                value="latest", inline=True, className="fan-radio",
                            ),
                        ],
                        style={"padding": "0 2px 12px 2px"},
                    ),
                    html.Div(
                        id="structural-reference-status",
                        style={"marginBottom": "12px"},
                    ),
                    html.Div(
                        id="structural-volatility-cards",
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                            "gap": "12px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Relative structural-variance state", style={"fontWeight": 700, "fontSize": "13px"}),
                                    html.Div(
                                        "λⱼ,ₜ divided by that shock's own sample-median λⱼ. This is a dimensionless "
                                        "state diagnostic, not a FEVD share.",
                                        style={"fontSize": "11px", "color": "#64748B", "marginTop": "2px"},
                                    ),
                                ],
                                style={"padding": "14px 14px 0 14px"},
                            ),
                            dcc.Graph(
                                id="structural-volatility-relative-graph",
                                config={"displayModeBar": False, "responsive": True},
                            ),
                        ],
                        style={
                            "marginTop": "14px", "border": "1px solid #E2E8F0",
                            "borderRadius": "12px", "background": "#FBFCFE",
                        },
                    ),
                    html.Div(
                        [
                            html.Div("What the selected date changes", style={"fontWeight": 700, "fontSize": "13px", "marginBottom": "8px"}),
                            html.Div(id="structural-reference-effects", style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
                        ],
                        style={"marginTop": "14px"},
                    ),
                ],
                className="panel chart-panel",
            ),

            html.Div(
                [
                    _stat_card("Identification", "structural-stat-identification", "structural-stat-ordering"),
                    _stat_card("Computed reference", "structural-stat-reference", "structural-stat-frequency"),
                    _stat_card("Posterior draws", "structural-stat-draws", "structural-stat-draws-total"),
                    _stat_card("HD reconstruction", "structural-stat-hd-error", "structural-stat-hd-error-note"),
                    _stat_card("FEVD sum error", "structural-stat-fevd-error", "structural-stat-fevd-error-note"),
                    _stat_card("IRF shock scale", "structural-stat-shock-scale", "structural-stat-shock-scale-note"),
                ], className="stats-grid",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Impulse responses", className="panel-title"),
                                    html.P(
                                        "A 1σ IRF uses the selected shock's √λ at the computed reference date. "
                                        "Unit-level IRFs are exactly invariant to that date because the impact normalisation "
                                        "cancels the shock scale.",
                                        className="panel-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div([html.Label("Shock", className="control-label"), dcc.Dropdown(id="structural-shock", options=[], clearable=False, className="compact-dropdown wide-control")], className="control-block wide-control"),
                                    html.Div([html.Label("Response", className="control-label"), dcc.Dropdown(id="structural-response", options=[], clearable=False, className="compact-dropdown wide-control")], className="control-block wide-control"),
                                    html.Div([html.Label("IRF object", className="control-label"), dcc.RadioItems(id="structural-irf-metric", options=[{"label": "Cumulative level", "value": "cumulative"}, {"label": "Period change", "value": "change"}], value="cumulative", inline=True, className="fan-radio")], className="control-block"),
                                    html.Div([html.Label("Fan", className="control-label"), dcc.RadioItems(id="structural-irf-fan", options=[{"label": "68%", "value": "68"}, {"label": "90%", "value": "90"}, {"label": "Both", "value": "both"}], value="68", inline=True, className="fan-radio")], className="control-block"),
                                ], className="chart-controls",
                            ),
                        ], className="panel-heading",
                    ),
                    readable_table("structural-irf-table", IRF_COLUMNS, page_size=7),
                    dcc.Loading(dcc.Graph(id="structural-irf-graph", config=_AGG_GRAPH_CONFIG if "_AGG_GRAPH_CONFIG" in globals() else _GRAPH_CONFIG), type="circle"),
                ], className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Forecast error variance decomposition", className="panel-title"),
                            html.P(
                                "Posterior-median shares, shown as 100% stacked bars. The selected reference date changes "
                                "FEVD through the joint relative structural-variance state; shares still depend on the full "
                                "VAR dynamics and contemporaneous impact matrix.",
                                className="panel-subtitle",
                            ),
                        ], className="panel-heading",
                    ),
                    readable_table("structural-fevd-table", [{"name":"Shock","id":"shock"}], page_size=12),
                    dcc.Loading(dcc.Graph(id="structural-fevd-graph", config=_AGG_GRAPH_CONFIG if "_AGG_GRAPH_CONFIG" in globals() else _GRAPH_CONFIG), type="circle"),
                ], className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Historical decomposition", className="panel-title"),
                                    html.P(
                                        "Posterior-mean structural contributions preserve the exact additive reconstruction. "
                                        "Under recursive identification the selected reference date does not change the HD: "
                                        "each historical observation uses its realised λₜ and outlier scale.",
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
                                        ], value=156, clearable=False, className="compact-dropdown",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Compute historical decomposition",
                                                id="structural-hd-run",
                                                n_clicks=0,
                                                className="refresh-button",
                                            ),
                                            html.Button(
                                                "Cancel",
                                                id="structural-hd-cancel",
                                                n_clicks=0,
                                                disabled=True,
                                                className="estimation-cancel-button",
                                            ),
                                        ],
                                        style={"display": "flex", "gap": "8px", "marginTop": "8px"},
                                    ),
                                ], className="control-block",
                            ),
                        ], className="panel-heading",
                    ),
                    html.Div(
                        "Checking HD cache…",
                        id="structural-hd-status",
                        className="selection-banner",
                    ),
                    readable_table("structural-hd-table", HD_COLUMNS, page_size=20),
                    dcc.Loading(dcc.Graph(id="structural-hd-graph", config=_AGG_GRAPH_CONFIG if "_AGG_GRAPH_CONFIG" in globals() else _GRAPH_CONFIG), type="circle"),
                    html.Div(
                        "Structural ≠ scenario. This page asks what an identified innovation does. Observable commodity-path assumptions belong in Scenarios.",
                        className="estimation-required-note",
                    ),
                ], className="panel chart-panel",
            ),
        ],
        className="page-body",
    )



# ---------------------------------------------------------------------------
# Shared Data preparation / production-vintage page
# ---------------------------------------------------------------------------


def _interior_missing_cells(frame: pd.DataFrame) -> int:
    """Count NaNs strictly between each series' first and last observation."""
    if frame is None or frame.empty:
        return 0
    total = 0
    for column in frame.columns:
        series = frame[column]
        valid = series.notna()
        if not valid.any():
            total += int(len(series))
            continue
        first = int(np.flatnonzero(valid.to_numpy())[0])
        last = int(np.flatnonzero(valid.to_numpy())[-1])
        total += int(series.iloc[first : last + 1].isna().sum())
    return int(total)


def _last_complete_date(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame is None or frame.empty:
        return None
    complete = frame.notna().all(axis=1)
    if not complete.any():
        return None
    return pd.Timestamp(frame.index[np.flatnonzero(complete.to_numpy())[-1]])


def _calendar_ok(frame: pd.DataFrame, frequency: str) -> bool:
    if frame is None or frame.empty or len(frame.index) < 2:
        return bool(frame is not None and not frame.empty)
    index = pd.DatetimeIndex(frame.index)
    rule = "MS" if str(frequency).lower() == "monthly" else "W-MON"
    expected = pd.date_range(index.min(), index.max(), freq=rule)
    return bool(index.equals(expected))


def _production_vintage_inventory() -> list[dict]:
    """All processed vintages, annotated by Energy/Headline readiness."""
    processed = PROJECT_ROOT / "data" / "processed"
    if not processed.is_dir():
        return []
    directories = sorted(
        (path for path in processed.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    try:
        energy_coverage = vintage_coverage(
            models=ENERGY_SUITE_MODEL_IDS,
            project_root=PROJECT_ROOT,
        )
    except Exception:
        energy_coverage = pd.DataFrame()
    try:
        headline_ready = set(
            map(str, headline_available_vintages(project_root=PROJECT_ROOT))
        )
    except Exception:
        headline_ready = set()

    rows = []
    for directory in directories:
        vintage = str(directory.name)
        energy_ready = bool(
            not energy_coverage.empty
            and vintage in energy_coverage.index
            and bool(energy_coverage.loc[vintage].all())
        )
        headline_ok = vintage in headline_ready
        linked = energy_ready and headline_ok
        rows.append(
            {
                "vintage": vintage,
                "energy_ready": energy_ready,
                "headline_ready": headline_ok,
                "linked_ready": linked,
            }
        )
    return rows


def _production_vintage_label(item: Mapping[str, Any]) -> str:
    if item.get("linked_ready"):
        suffix = "linked ready"
    elif item.get("energy_ready"):
        suffix = "Energy ready"
    elif item.get("headline_ready"):
        suffix = "Headline ready"
    else:
        suffix = "incomplete"
    return f"{item['vintage']} · {suffix}"


def _data_diagnostic_row(
    *,
    domain: str,
    dataset: str,
    frequency: str,
    frame: pd.DataFrame | None,
    source: str,
    status: str,
    note: str = "",
) -> dict:
    if frame is None or frame.empty:
        return {
            "domain": domain,
            "dataset": dataset,
            "frequency": frequency,
            "series": 0,
            "start": "—",
            "end": "—",
            "last_complete": "—",
            "rows": 0,
            "missing_cells": "—",
            "interior_missing": "—",
            "edge_missing": "—",
            "calendar": "—",
            "status": status,
            "source": source,
            "note": note,
        }
    start = pd.Timestamp(frame.index.min()).strftime("%Y-%m-%d")
    end = pd.Timestamp(frame.index.max()).strftime("%Y-%m-%d")
    last = _last_complete_date(frame)
    missing = int(frame.isna().sum().sum())
    interior = _interior_missing_cells(frame)
    edge_missing = max(int(missing - interior), 0)
    return {
        "domain": domain,
        "dataset": dataset,
        "frequency": frequency,
        "series": int(frame.shape[1]),
        "start": start,
        "end": end,
        "last_complete": "—" if last is None else last.strftime("%Y-%m-%d"),
        "rows": int(len(frame)),
        "missing_cells": missing,
        "interior_missing": interior,
        "edge_missing": edge_missing,
        "calendar": "OK" if _calendar_ok(frame, frequency) else "GAP",
        "status": status,
        "source": source,
        "note": note,
    }


def _validation_artifact_rows(vintage: str) -> tuple[list[dict], list[str]]:
    """Summarise persisted processed-vintage audit artefacts without model work."""
    directory = PROJECT_ROOT / "data" / "processed" / str(vintage)
    specs = (
        ("manifest.json", "Build manifest"),
        ("source_vintages.csv", "Source vintages"),
        ("model_datasets_diagnostics.csv", "Model dataset diagnostics"),
        ("aggregation_coverage.csv", "Energy aggregation coverage"),
        ("hicp_validation_failures.csv", "HICP validation failures"),
        ("hicp_weight_identity_diagnostics.csv", "HICP weight identity"),
        ("headline_hicp_weight_identity_diagnostics.csv", "Headline HICP weight identity"),
        ("headline_joint_weight_identity_diagnostics.csv", "Headline joint weight identity"),
    )
    rows: list[dict] = []
    warnings: list[str] = []

    for filename, label in specs:
        path = directory / filename
        if not path.is_file():
            rows.append(
                {
                    "artefact": label,
                    "file": filename,
                    "rows": "—",
                    "columns": "—",
                    "status": "MISSING",
                    "note": "Not present in this processed vintage.",
                }
            )
            continue

        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "artefact": label,
                        "file": filename,
                        "rows": 1,
                        "columns": len(payload) if isinstance(payload, dict) else "—",
                        "status": "PRESENT",
                        "note": "Processed-vintage provenance manifest.",
                    }
                )
            except Exception as exc:
                warnings.append(f"{filename}: unreadable JSON ({exc})")
                rows.append(
                    {
                        "artefact": label,
                        "file": filename,
                        "rows": "—",
                        "columns": "—",
                        "status": "ERROR",
                        "note": str(exc),
                    }
                )
            continue

        try:
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            n_rows = int(len(frame))
            n_cols = int(frame.shape[1])
            if filename == "hicp_validation_failures.csv":
                status = "PASS" if n_rows == 0 else "WARN"
                note = (
                    "No recorded HICP validation failures."
                    if n_rows == 0
                    else f"{n_rows} recorded validation failure row(s); inspect before estimation."
                )
                if n_rows:
                    warnings.append(f"{filename}: {n_rows} validation failure row(s)")
            else:
                status = "PRESENT"
                note = f"{n_rows} row(s), {n_cols} column(s)."
            rows.append(
                {
                    "artefact": label,
                    "file": filename,
                    "rows": n_rows,
                    "columns": n_cols,
                    "status": status,
                    "note": note,
                }
            )
        except Exception as exc:
            warnings.append(f"{filename}: unreadable CSV ({exc})")
            rows.append(
                {
                    "artefact": label,
                    "file": filename,
                    "rows": "—",
                    "columns": "—",
                    "status": "ERROR",
                    "note": str(exc),
                }
            )

    return rows, warnings


def _production_vintage_diagnostics(vintage: str) -> dict:
    """Filesystem/data-contract diagnostics only. Never estimates a model."""
    vintage = str(vintage)
    processed_dir = PROJECT_ROOT / "data" / "processed" / vintage
    rows: list[dict] = []
    energy_ready_count = 0

    for model_id in ENERGY_SUITE_MODEL_IDS:
        spec = model_spec(model_id)
        try:
            panel = build_panel(model_id, vintage, project_root=PROJECT_ROOT)
            energy_ready_count += 1
            missing = int(panel.levels.isna().sum().sum())
            interior = _interior_missing_cells(panel.levels)
            note = (
                f"{missing} total missing level cells / {interior} interior. "
                "Treatment is chosen separately in Estimation."
            )
            rows.append(
                _data_diagnostic_row(
                    domain="Energy",
                    dataset=spec.label,
                    frequency=panel.frequency,
                    frame=panel.levels,
                    source=panel.dataset_path.name,
                    status="READY",
                    note=note,
                )
            )
        except Exception as exc:
            rows.append(
                _data_diagnostic_row(
                    domain="Energy",
                    dataset=spec.label,
                    frequency=spec.frequency,
                    frame=None,
                    source=spec.dataset_file,
                    status="ERROR",
                    note=str(exc),
                )
            )

    headline_ready = False
    try:
        inputs = build_headline_inputs(vintage, project_root=PROJECT_ROOT)
        headline_ready = True
        rows.append(
            _data_diagnostic_row(
                domain="Headline",
                dataset="Joint HICP components",
                frequency="monthly",
                frame=inputs.native_levels,
                source=inputs.dataset_path.name,
                status="READY",
                note=(
                    f"weights {inputs.weights_path.name}; official Headline "
                    f"{inputs.official_indices_path.name}"
                ),
            )
        )
    except Exception as exc:
        rows.append(
            _data_diagnostic_row(
                domain="Headline",
                dataset="Joint HICP components",
                frequency="monthly",
                frame=None,
                source="headline_joint_monthly.csv",
                status="ERROR",
                note=str(exc),
            )
        )

    aggregate_required = (
        "aggregation_coverage.csv",
        "hicp_indices_monthly.csv",
        "hicp_weights_annual.csv",
        "hicp_series_metadata.csv",
        "hicp_flags.csv",
        "hicp_weight_identity_diagnostics.csv",
        "manifest.json",
    )
    aggregate_missing = [
        name for name in aggregate_required if not (processed_dir / name).is_file()
    ]
    aggregate_ready = not aggregate_missing
    rows.append(
        {
            "domain": "Energy",
            "dataset": "HICP Energy aggregation inputs",
            "frequency": "mixed",
            "series": "—",
            "start": "—",
            "end": "—",
            "last_complete": "—",
            "rows": "—",
            "missing_cells": "—",
            "interior_missing": "—",
            "edge_missing": "—",
            "calendar": "—",
            "status": "READY" if aggregate_ready else "ERROR",
            "source": ", ".join(aggregate_required),
            "note": (
                "All aggregation sidecars present"
                if aggregate_ready
                else "Missing: " + ", ".join(aggregate_missing)
            ),
        }
    )

    artifact_rows, artifact_warnings = _validation_artifact_rows(vintage)
    energy_ready = energy_ready_count == len(ENERGY_SUITE_MODEL_IDS)
    linked_ready = energy_ready and headline_ready and aggregate_ready
    return {
        "vintage": vintage,
        "processed_ready": processed_dir.is_dir(),
        "energy_ready": energy_ready,
        "energy_ready_count": int(energy_ready_count),
        "headline_ready": headline_ready,
        "aggregate_ready": aggregate_ready,
        "linked_ready": linked_ready,
        "rows": rows,
        "artifact_rows": artifact_rows,
        "artifact_warnings": artifact_warnings,
    }



def _headline_run_artifact_state(vintage: str) -> dict:
    """Inspect saved locked-Headline artefacts without estimating anything."""
    vintage = str(vintage)
    base = RESULTS_ROOT / str(HEADLINE_MODEL_ID) / vintage
    rows = []
    if base.is_dir():
        for directory in sorted(p for p in base.iterdir() if p.is_dir()):
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
            if str(metadata.get("model_id", HEADLINE_MODEL_ID)) != str(HEADLINE_MODEL_ID):
                continue
            forecast_dir = directory / "forecasts" / "unconditional"
            checks = {
                "draws": (directory / "draws.npz").is_file(),
                "forecast": (
                    (forecast_dir / "forecast_metadata.json").is_file()
                    and (forecast_dir / "forecast_draws.npz").is_file()
                ),
                "headline": (
                    (forecast_dir / "headline_metadata.json").is_file()
                    and (forecast_dir / "headline_draws.npz").is_file()
                ),
            }
            rows.append(
                {
                    "run_id": directory.name,
                    "directory": directory,
                    "checks": checks,
                    "complete": all(checks.values()),
                }
            )

    promoted_ids: set[str] = set()
    try:
        registry_rows = _registry_table("runs")
        if not registry_rows.empty:
            registry_rows = registry_rows.loc[
                registry_rows["model_id"].astype(str).eq(str(HEADLINE_MODEL_ID))
                & registry_rows["vintage"].astype(str).eq(str(vintage))
                & registry_rows["status"].astype(str).eq("complete")
            ].copy()
        if not registry_rows.empty and "promoted" in registry_rows:
            promoted_ids = set(
                registry_rows.loc[
                    registry_rows["promoted"].fillna(0).astype(int).eq(1),
                    "run_id",
                ].astype(str)
            )
    except Exception:
        promoted_ids = set()

    complete = [row for row in rows if row["complete"]]
    promoted_complete = [
        row for row in complete if str(row["run_id"]) in promoted_ids
    ]
    if len(promoted_complete) == 1:
        chosen = promoted_complete[0]
        status = "COMPLETE"
        note = "promoted locked Headline run"
    elif len(complete) == 1:
        chosen = complete[0]
        status = "COMPLETE"
        note = "unique complete locked Headline run"
    elif len(complete) > 1:
        chosen = None
        status = "AMBIGUOUS"
        note = f"{len(complete)} complete runs; promote/select one before linked production"
    elif rows:
        chosen = None
        status = "PARTIAL"
        note = f"{len(rows)} saved run director{'y' if len(rows) == 1 else 'ies'}, none production-complete"
    else:
        chosen = None
        status = "MISSING"
        note = "no saved Headline run for this vintage"

    return {
        "status": status,
        "note": note,
        "run_id": "" if chosen is None else str(chosen["run_id"]),
        "directory": None if chosen is None else chosen["directory"],
        "n_complete": len(complete),
        "n_saved": len(rows),
    }


def _production_model_readiness(
    vintage: str,
    *,
    selected_model_id: str | None,
    config_store: dict | None,
) -> dict:
    """Exact saved-model readiness for the current production vintage.

    Energy is compared against the deterministic run IDs implied by the current
    Estimation configuration. This is exactly the identity used by the suite
    callback, so the table is also the next-run resume plan.
    """
    vintage = str(vintage)
    selected_model_id = selected_model_id or ENERGY_SUITE_MODEL_IDS[0]
    if not config_store or not config_store.get("valid"):
        config_store = {**_baseline_config_payload(), "valid": True}

    rows: list[dict] = []
    energy_states: dict[str, dict] = {}
    energy_run_ids: dict[str, str] = {}

    for model_id in ENERGY_SUITE_MODEL_IDS:
        spec = model_spec(model_id)
        try:
            state = _planned_run_state(
                model_id,
                vintage,
                selected_model_id=selected_model_id,
                config_store=config_store,
            )
            energy_states[model_id] = state
            energy_run_ids[str(spec.model_id)] = str(state["run_id"])

            reusable = _component_run_reusable(state)
            any_saved = bool(
                Path(state["directory"]).is_dir()
                or state.get("has_draws")
                or state.get("has_forecast")
                or state.get("has_hicp")
            )
            status = "COMPLETE" if reusable else ("PARTIAL" if any_saved else "MISSING")
            missing_parts = [
                label
                for label, ok in (
                    ("draws", bool(state.get("has_draws"))),
                    ("forecast", bool(state.get("has_forecast"))),
                    ("HICP", bool(state.get("has_hicp"))),
                )
                if not ok
            ]
            artifacts = " · ".join(
                [
                    f"draws {'✓' if state.get('has_draws') else '—'}",
                    f"forecast {'✓' if state.get('has_forecast') else '—'}",
                    f"HICP {'✓' if state.get('has_hicp') else '—'}",
                    f"display {'✓' if state.get('has_display') else '—'}",
                ]
            )
            note = (
                "Exact planned run will be reused; Gibbs skipped."
                if reusable
                else (
                    "Exact planned run is partial; missing " + ", ".join(missing_parts) + "."
                    if any_saved
                    else "Exact planned run has not been estimated."
                )
            )
            rows.append(
                {
                    "domain": "Energy",
                    "model": spec.label,
                    "run_id": _short_run(state["run_id"]),
                    "artifacts": artifacts,
                    "status": status,
                    "next_run": "REUSE" if reusable else "ESTIMATE",
                    "note": note,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "domain": "Energy",
                    "model": spec.label,
                    "run_id": "—",
                    "artifacts": "—",
                    "status": "ERROR",
                    "next_run": "BLOCKED",
                    "note": str(exc),
                }
            )

    energy_complete = sum(
        _component_run_reusable(state) for state in energy_states.values()
    )
    all_energy_complete = (
        len(energy_states) == len(ENERGY_SUITE_MODEL_IDS)
        and energy_complete == len(ENERGY_SUITE_MODEL_IDS)
    )

    aggregate_status = "BLOCKED"
    aggregate_run_id = ""
    aggregate_note = "Complete the exact seven Energy component runs first."
    aggregate_reused = False
    if all_energy_complete:
        try:
            existing = _matching_saved_energy_aggregate(vintage, energy_run_ids)
            if existing is None:
                aggregate_status = "BUILD PENDING"
                aggregate_note = (
                    "All exact component runs are complete; the next suite run "
                    "can build the HICP Energy aggregate without rerunning Gibbs."
                )
            else:
                aggregate_run_id, _aggregate_dir = existing
                aggregate_status = "COMPLETE"
                aggregate_reused = True
                aggregate_note = (
                    "Exact aggregate already matches the seven planned component run IDs."
                )
        except Exception as exc:
            aggregate_status = "ERROR"
            aggregate_note = str(exc)

    rows.append(
        {
            "domain": "Energy",
            "model": "HICP Energy aggregate",
            "run_id": _short_run(aggregate_run_id) if aggregate_run_id else "—",
            "artifacts": "aggregate_draws + metadata" if aggregate_reused else "—",
            "status": aggregate_status,
            "next_run": (
                "REUSE"
                if aggregate_status == "COMPLETE"
                else "BUILD"
                if aggregate_status == "BUILD PENDING"
                else "BLOCKED"
            ),
            "note": aggregate_note,
        }
    )

    headline = _headline_run_artifact_state(vintage)
    rows.append(
        {
            "domain": "Headline",
            "model": "Headline joint BVAR",
            "run_id": _short_run(headline["run_id"]) if headline["run_id"] else "—",
            "artifacts": (
                "draws ✓ · forecast ✓ · Headline paths ✓"
                if headline["status"] == "COMPLETE"
                else "inspect saved run artefacts"
            ),
            "status": headline["status"],
            "next_run": "REUSE" if headline["status"] == "COMPLETE" else "ESTIMATE",
            "note": headline["note"],
        }
    )

    linked_ready = bool(
        all_energy_complete
        and aggregate_status == "COMPLETE"
        and headline["status"] == "COMPLETE"
    )

    return {
        "rows": rows,
        "energy_complete": int(energy_complete),
        "energy_total": len(ENERGY_SUITE_MODEL_IDS),
        "energy_ready": bool(all_energy_complete),
        "aggregate_status": aggregate_status,
        "aggregate_run_id": aggregate_run_id,
        "headline_status": headline["status"],
        "headline_run_id": headline["run_id"],
        "linked_ready": linked_ready,
    }


def data_page() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Data preparation & validation", className="page-title"),
                            html.P(
                                "One production information set shared by Energy and Headline. "
                                "Build or select a processed vintage here, inspect data diagnostics, "
                                "then estimate either model family without choosing a second vintage.",
                                className="page-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Production vintage", className="control-label"),
                            dcc.Dropdown(
                                id="production-vintage-select",
                                options=[],
                                value=None,
                                clearable=False,
                                persistence=True,
                                persistence_type="session",
                                className="compact-dropdown",
                            ),
                        ],
                        className="control-block",
                    ),
                ],
                className="page-heading-row",
            ),
            html.Div(
                [
                    _stat_card("Processed vintage", "data-processed-status", "data-processed-note"),
                    _stat_card("Energy inputs", "data-energy-status", "data-energy-note"),
                    _stat_card("Headline inputs", "data-headline-status", "data-headline-note"),
                    _stat_card("Linked suite", "data-linked-status", "data-linked-note"),
                ],
                className="stats-grid",
            ),
            html.Div(id="data-readiness-banner", className="selection-banner"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Build / refresh processed vintage", className="panel-title"),
                                    html.P(
                                        "Refresh and save the canonical Excel workbook, then materialise the "
                                        "shared processed vintage. This writes only below data/processed/<vintage> "
                                        "and never modifies saved BVAR results.",
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
                    html.Div(id="dataset-build-environment", className="selection-banner"),
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
                                value=[],
                                className="estimation-checklist",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Build / refresh processed vintage",
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
                                    html.Div("Idle", id="dataset-build-phase", className="estimation-phase"),
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
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Data diagnostics", className="eyebrow"),
                            html.H3("Coverage, calendar and missing observations", className="panel-title"),
                            html.P(
                                "These are data diagnostics only. The model's treatment of missing data "
                                "(linear or Durbin–Koopman) remains an Estimation setting.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    dash_table.DataTable(
                        id="data-diagnostics-table",
                        columns=[
                            {"name": "Domain", "id": "domain"},
                            {"name": "Dataset / model", "id": "dataset"},
                            {"name": "Frequency", "id": "frequency"},
                            {"name": "Series", "id": "series"},
                            {"name": "Start", "id": "start"},
                            {"name": "End", "id": "end"},
                            {"name": "Last complete", "id": "last_complete"},
                            {"name": "Rows", "id": "rows"},
                            {"name": "Missing cells", "id": "missing_cells"},
                            {"name": "Interior missing", "id": "interior_missing"},
                            {"name": "Edge missing", "id": "edge_missing"},
                            {"name": "Calendar", "id": "calendar"},
                            {"name": "Status", "id": "status"},
                            {"name": "Source", "id": "source"},
                            {"name": "Note", "id": "note"},
                        ],
                        data=[],
                        sort_action="native",
                        filter_action="native",
                        page_action="none",
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontSize": "11px",
                            "padding": "7px 8px",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "minWidth": "80px",
                            "maxWidth": "260px",
                        },
                        style_header={"fontWeight": "600"},
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Production model readiness", className="eyebrow"),
                            html.H3("Saved runs & smart-resume plan", className="panel-title"),
                            html.P(
                                "Read-only status for the selected production vintage. "
                                "Energy rows use the exact deterministic run IDs implied by the current "
                                "Estimation configuration. COMPLETE rows are reused on the next suite run; "
                                "Gibbs is skipped. A component is production-complete only when posterior "
                                "draws, raw forecast and component HICP bridge all exist.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(
                        [
                            _stat_card("Energy models", "data-model-energy-status", "data-model-energy-note"),
                            _stat_card("Energy aggregate", "data-model-aggregate-status", "data-model-aggregate-note"),
                            _stat_card("Headline BVAR", "data-model-headline-status", "data-model-headline-note"),
                            _stat_card("Linked production", "data-model-linked-status", "data-model-linked-note"),
                        ],
                        className="stats-grid",
                    ),
                    html.Div(id="data-model-readiness-banner", className="selection-banner"),
                    dash_table.DataTable(
                        id="data-model-readiness-table",
                        columns=[
                            {"name": "Domain", "id": "domain"},
                            {"name": "Model", "id": "model"},
                            {"name": "Planned / saved run", "id": "run_id"},
                            {"name": "Artefacts", "id": "artifacts"},
                            {"name": "Status", "id": "status"},
                            {"name": "Next suite action", "id": "next_run"},
                            {"name": "Note", "id": "note"},
                        ],
                        data=[],
                        sort_action="native",
                        page_action="none",
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontSize": "11px",
                            "padding": "7px 8px",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "minWidth": "90px",
                            "maxWidth": "360px",
                        },
                        style_header={"fontWeight": "600"},
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Persisted validation artefacts", className="eyebrow"),
                            html.H3("Builder diagnostics & lineage", className="panel-title"),
                            html.P(
                                "Read-only summary of the audit files already written under "
                                "data/processed/<vintage>. These checks do not estimate any model.",
                                className="panel-subtitle",
                            ),
                        ],
                        className="panel-heading",
                    ),
                    html.Div(id="data-validation-banner", className="selection-banner"),
                    dash_table.DataTable(
                        id="data-validation-artifacts-table",
                        columns=[
                            {"name": "Artefact", "id": "artefact"},
                            {"name": "File", "id": "file"},
                            {"name": "Rows", "id": "rows"},
                            {"name": "Columns", "id": "columns"},
                            {"name": "Status", "id": "status"},
                            {"name": "Note", "id": "note"},
                        ],
                        data=[],
                        sort_action="native",
                        page_action="none",
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontSize": "11px",
                            "padding": "7px 8px",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "minWidth": "90px",
                            "maxWidth": "320px",
                        },
                        style_header={"fontWeight": "600"},
                    ),
                ],
                className="panel",
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
                                    html.Label("Production vintage", className="control-label"),
                                    html.Div(
                                        id="est-production-vintage-value",
                                        children="Select it in Data",
                                        className="stat-value",
                                    ),
                                    dcc.Link(
                                        "Open Data →",
                                        href="/data",
                                        className="stat-subtitle",
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
                    html.Strong("Production data is managed centrally in Data."),
                    html.Span(
                        " Energy estimation reads the shared production vintage; "
                        "dataset building and coverage diagnostics no longer live on this page."
                    ),
                    dcc.Link(" Open Data →", href="/data", className="refresh-button"),
                ],
                className="selection-banner",
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
                                    html.Div("Select a model and production vintage.", id="estimation-progress-detail", className="estimation-progress-detail"),
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
                    html.Div("Inflation Dashboard", className="brand-title"),
                ],
                className="brand-row",
            ),
            html.Nav(
                [
                    _nav_link("Overview", "/overview", "◎"),
                    _nav_link("Data", "/data", "▦"),
                    *(
                        [_nav_link("Economic Data", "/economic-data", "▤")]
                        if ECONOMIC_DATA_ENABLED
                        else []
                    ),
                ],
                id="global-nav",
                className="nav-stack",
            ),
            html.Div(
                [
                    dcc.Link("Energy", href="/forecast", className="domain-pill"),
                    dcc.Link(
                        "Headline HICP",
                        href="/headline/forecast",
                        className="domain-pill",
                    ),
                    dcc.Link(
                        "Core",
                        href="/core/forecast",
                        className="domain-pill",
                    ),
                ],
                className="domain-switch",
            ),
            html.Nav(
                [
                    _nav_link("Forecast", "/forecast", "↗"),
                    _nav_link("Aggregate", "/aggregate", "Σ"),
                    _nav_link("Scenarios", "/scenarios", "△"),
                    _nav_link("Structural", "/structural", "ψ"),
                    _nav_link("Estimation", "/estimation", "⚙"),
                ],
                id="energy-nav",
                className="nav-stack nav-stack-energy",
            ),
            html.Nav(
                [
                    _nav_link("Forecast & Contributions", "/headline/forecast", "↗"),
                    _nav_link("Scenarios", "/headline/scenarios", "△"),
                    _nav_link("Structural", "/headline/structural", "ψ"),
                    _nav_link("Estimation", "/headline/estimation", "⚙"),
                ],
                id="headline-nav",
                className="nav-stack nav-stack-headline",
                style={"display": "none"},
            ),
            html.Nav(
                [
                    _nav_link("Forecast & Contributions", "/core/forecast", "↗"),
                    _nav_link("Scenarios", "/core/scenarios", "△"),
                ],
                id="core-nav",
                className="nav-stack nav-stack-core",
                style={"display": "none"},
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
                    html.Button("Refresh snapshot", id="refresh-registry", n_clicks=0, className="refresh-button"),
                    html.Div(id="registry-status", className="registry-status"),
                ],
                className="refresh-area",
            ),
        ],
        id="global-topbar",
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
    title="Inflation Dashboard",
    update_title="Updating…",
)
server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="registry-store", data=_scan_registry()),
        dcc.Store(id="ctx-store", storage_type="session"),
        dcc.Store(id="production-vintage-store", storage_type="session"),
        dcc.Store(id="data-store", storage_type="memory"),
        dcc.Store(id="agg-store", storage_type="memory"),
        dcc.Store(id="scenario-store", storage_type="memory"),
        dcc.Store(id="conditional-store", storage_type="memory"),
        dcc.Store(id="scenario-tax-selected-agg-store", storage_type="memory"),
        dcc.Store(id="agg-scenario-store", storage_type="memory"),
        dcc.Store(id="estimation-result-store", storage_type="memory"),
        dcc.Store(id="dataset-build-result-store", storage_type="memory"),
        dcc.Store(id="structural-volatility-store", storage_type="memory"),
        dcc.Store(id="structural-store", storage_type="memory"),
        dcc.Store(id="structural-hd-key-store", storage_type="memory"),
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
                        html.Div(overview_page(), id="page-overview", style={"display": "none"}),
                        html.Div(data_page(), id="page-data", style={"display": "none"}),
                        html.Div(
                            (
                                _economic_data_page()
                                if ECONOMIC_DATA_ENABLED
                                and _economic_data_page is not None
                                else None
                            ),
                            id="page-economic-data",
                            style={"display": "none"},
                        ),
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
                    html.Div(
                        headline_forecast_v2_page(),
                        id="page-headline-forecast",
                        style={"display": "none"},
                    ),
                    html.Div(
                        headline_scenarios_page(),
                        id="page-headline-scenarios",
                        style={"display": "none"},
                    ),
                    html.Div(
                        core_forecast_page(),
                        id="page-core-forecast",
                        style={"display": "none"},
                    ),
                    html.Div(
                        core_scenarios_page(),
                        id="page-core-scenarios",
                        style={"display": "none"},
                    ),
                        html.Div(
                            headline_structural_page(),
                            id="page-headline-structural",
                            style={"display": "none"},
                        ),
                        html.Div(
                        [
                            html.Div(
                                [
                                    html.Strong("Production vintage: "),
                                    html.Span(id="headline-production-vintage-value", children="Select it in Data"),
                                    dcc.Link(" · Open Data →", href="/data", className="refresh-button"),
                                ],
                                className="selection-banner",
                            ),
                            headline_estimation_v2_page(),
                        ],
                        id="page-headline-diagnostics",
                        style={"display": "none"},
                    )
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
    Input("dataset-build-result-store", "data"),
    Input("estimation-result-store", "data"),
    prevent_initial_call=True,
)
def refresh_registry(
    _manual_clicks: int | None,
    dataset_result: dict | None,
    estimation_result: dict | None,
) -> dict:
    """Refresh the frozen registry after explicit refreshes or successful writes.

    Background callbacks run in worker processes, so calling ``scan_results``
    inside them cannot refresh this server process' in-memory snapshot.  The
    result-store transition is the safe foreground hand-off point.
    """
    trigger = ctx.triggered_id
    if trigger == "dataset-build-result-store":
        if not (dataset_result or {}).get("ok"):
            raise PreventUpdate
    elif trigger == "estimation-result-store":
        if not (estimation_result or {}).get("ok"):
            raise PreventUpdate
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
    Input("url", "pathname"),
    State("vintage-select", "value"),
)
def vintage_options(
    _: dict | None,
    pathname: str | None,
    current: str | None,
):
    forecasts = _registry_table("forecasts")
    if forecasts.empty:
        return [], None

    domain = "headline" if (pathname or "").startswith("/core") else domain_from_path(pathname)
    allowed_models = set(
        models_for_domain(
            forecasts["model_id"].astype(str).unique(),
            domain,
        )
    )
    forecasts = forecasts.loc[
        forecasts["model_id"].astype(str).isin(allowed_models)
    ]
    if forecasts.empty:
        return [], None

    vintages = sorted(
        forecasts["vintage"].astype(str).unique(),
        reverse=True,
    )
    options = [{"label": value, "value": value} for value in vintages]
    return options, current if current in vintages else vintages[0]


@callback(
    Output("model-select", "options"),
    Output("model-select", "value"),
    Input("vintage-select", "value"),
    Input("registry-store", "data"),
    Input("url", "pathname"),
    State("model-select", "value"),
)
def model_options(
    vintage: str | None,
    _: dict | None,
    pathname: str | None,
    current: str | None,
):
    if not vintage:
        return [], None
    forecasts = _registry_table("forecasts")
    if not forecasts.empty:
        forecasts = forecasts.loc[
            forecasts["vintage"].astype(str).eq(str(vintage))
        ].copy()
    if forecasts.empty:
        return [], None

    domain = "headline" if (pathname or "").startswith("/core") else domain_from_path(pathname)
    models = models_for_domain(
        forecasts["model_id"].astype(str).unique(),
        domain,
    )
    options = [
        {"label": model_label(model_id), "value": model_id}
        for model_id in models
    ]
    values = [item["value"] for item in options]
    return (
        options,
        current if current in values else (values[0] if values else None),
    )


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
    frame = _registry_table("forecasts")
    if not frame.empty:
        frame = frame.loc[
            frame["model_id"].astype(str).eq(str(model_id))
            & frame["vintage"].astype(str).eq(str(vintage))
        ].copy()
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
    forecasts = _registry_table("forecasts")
    if not forecasts.empty:
        forecasts = forecasts.loc[
            forecasts["model_id"].astype(str).eq(str(model_id))
            & forecasts["vintage"].astype(str).eq(str(vintage))
            & forecasts["forecast_name"].astype(str).eq(str(forecast_name))
        ].copy()
    runs = _registry_table("runs")
    if not runs.empty:
        runs = runs.loc[
            runs["model_id"].astype(str).eq(str(model_id))
            & runs["vintage"].astype(str).eq(str(vintage))
            & runs["status"].astype(str).eq("complete")
        ].copy()
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
    Input("url", "pathname"),
    State("ctx-store", "data"),
)
def update_context(
    vintage: str | None,
    model_id: str | None,
    run_id: str | None,
    forecast_name: str | None,
    pathname: str | None,
    previous: dict | None,
):
    previous = dict(previous or {})
    proposed = dict(previous)
    proposed.update(
        {
            "domain": (
                "core"
                if (pathname or "").startswith("/core")
                else domain_from_path(pathname)
            ),
            "vintage": vintage,
            "model_id": model_id,
            "run_id": run_id,
            "forecast_name": forecast_name,
            "draw_mode": previous.get("draw_mode", "posterior"),
        }
    )
    # Navigation inside the same domain must not rewrite the economic context.
    # This keeps already-loaded page stores frozen when a user leaves a page and
    # later returns without changing vintage/model/run/forecast.
    if proposed == previous:
        raise PreventUpdate
    return proposed


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

    forecasts = _registry_table("forecasts")
    if not forecasts.empty:
        forecasts = forecasts.loc[
            forecasts["model_id"].astype(str).eq(str(context["model_id"]))
            & forecasts["vintage"].astype(str).eq(str(context["vintage"]))
            & forecasts["forecast_name"].astype(str).eq(str(context["forecast_name"]))
        ].copy()
    selected = forecasts.loc[forecasts["run_id"].astype(str) == str(context["run_id"])]
    if selected.empty:
        return None, html.Div("Selected forecast store is no longer present in the registry.")

    row = selected.iloc[0]
    forecast_dir = Path(str(row["directory"]))
    display_path = forecast_dir / DISPLAY_FILENAME
    snapshot_ref = (
        f"{_registry_snapshot_id()}::component-display::"
        f"{context['model_id']}::{context['vintage']}::{context['run_id']}::"
        f"{context['forecast_name']}"
    )
    cached_result = snapshot_get(
        "component_display_result",
        snapshot_ref,
    )
    if (
        isinstance(cached_result, tuple)
        and len(cached_result) == 2
    ):
        return cached_result

    cached_display = snapshot_get("component_display", snapshot_ref)
    materialised = False

    if isinstance(cached_display, dict):
        frame = cached_display["frame"]
        meta = dict(cached_display["meta"])
        snapshot_put_frame(snapshot_ref, frame)
    else:
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
            # valid for the raw BVAR while still lacking component HICP output.
            fan_mask = frame["record_type"].astype(str) == "fan"
            metrics_present = set(
                frame.loc[fan_mask, "metric"].dropna().astype(str)
            )

            is_headline = context.get("model_id") == HEADLINE_MODEL_ID
            required_metrics = (
                {"level", "yoy"}
                if is_headline
                else {"hicp_level", "yoy"}
            )
            if (
                not is_headline
                and AUTO_BUILD_DISPLAY
                and not required_metrics.issubset(metrics_present)
            ):
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
            fan_rows = int(
                (frame["record_type"].astype(str) == "fan").sum()
            )
            if fan_rows == 0:
                raise ValueError(
                    f"{DISPLAY_FILENAME} loaded successfully but contains no forecast fan rows."
                )
            final_metrics = set(
                frame.loc[
                    frame["record_type"].astype(str) == "fan",
                    "metric",
                ].dropna().astype(str)
            )
            if not required_metrics.issubset(final_metrics):
                if is_headline:
                    raise ValueError(
                        "Headline display is incomplete: expected both "
                        "'level' and 'yoy' fan metrics in display_v1."
                    )
                raise ValueError(
                    "Component HICP post-processing is incomplete: expected both "
                    "'hicp_level' and 'yoy' in display_v1."
                )
            snapshot_put(
                "component_display",
                snapshot_ref,
                {"frame": frame, "meta": dict(meta)},
            )
            snapshot_put_frame(snapshot_ref, frame)
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
            html.Span(
                " · Headline aggregate: ready"
                if context.get("model_id") == HEADLINE_MODEL_ID
                else " · component HICP: ready"
            ),
            html.Span(built_note),
        ]
    )
    component_store = {
        "snapshot_ref": snapshot_ref,
        "frame_json": _json_frame(frame),
        "meta": meta,
        "context": context,
        "display_path": str(display_path),
    }
    result = (component_store, banner)
    snapshot_put(
        "component_display_result",
        snapshot_ref,
        result,
    )
    return result


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
    Output("production-vintage-select", "options"),
    Output("production-vintage-select", "value"),
    Input("registry-store", "data"),
    State("production-vintage-select", "value"),
)
def production_vintage_options(_registry, current):
    inventory = _frozen_production_inventory()
    options = [
        {"label": _production_vintage_label(item), "value": item["vintage"]}
        for item in inventory
    ]
    values = [item["vintage"] for item in inventory]

    if current in values:
        value = current
    else:
        linked = [item["vintage"] for item in inventory if item.get("linked_ready")]
        value = linked[0] if linked else (values[0] if values else None)
    return options, value


@callback(
    Output("production-vintage-store", "data"),
    Output("data-processed-status", "children"),
    Output("data-processed-note", "children"),
    Output("data-energy-status", "children"),
    Output("data-energy-note", "children"),
    Output("data-headline-status", "children"),
    Output("data-headline-note", "children"),
    Output("data-linked-status", "children"),
    Output("data-linked-note", "children"),
    Output("data-readiness-banner", "children"),
    Output("data-diagnostics-table", "data"),
    Output("data-validation-banner", "children"),
    Output("data-validation-artifacts-table", "data"),
    Input("production-vintage-select", "value"),
    Input("dataset-build-result-store", "data"),
)
def render_production_vintage_diagnostics(vintage, _build_result):
    if not vintage:
        empty_store = {"vintage": None, "linked_ready": False}
        return (
            empty_store,
            "—", "Select a production vintage",
            "—", "0/7",
            "—", "not checked",
            "—", "not checked",
            html.Div("Select a production vintage."),
            [],
            html.Div("Select a production vintage."),
            [],
        )

    diagnostics = snapshot_get_or_build(
        "production_vintage_diagnostics",
        (_registry_snapshot_id(), str(vintage)),
        lambda: _production_vintage_diagnostics(str(vintage)),
    )
    energy_count = int(diagnostics["energy_ready_count"])
    processed = bool(diagnostics["processed_ready"])
    energy_ready = bool(diagnostics["energy_ready"])
    headline_ready = bool(diagnostics["headline_ready"])
    aggregate_ready = bool(diagnostics["aggregate_ready"])
    linked_ready = bool(diagnostics["linked_ready"])

    store = {
        "vintage": str(vintage),
        "processed_ready": processed,
        "energy_ready": energy_ready,
        "energy_ready_count": energy_count,
        "headline_ready": headline_ready,
        "aggregate_ready": aggregate_ready,
        "linked_ready": linked_ready,
    }

    if linked_ready:
        banner = html.Div(
            [
                html.Strong("Production vintage ready"),
                html.Span(f" · {vintage}"),
                html.Span(" · Energy 7/7"),
                html.Span(" · Headline ready"),
                html.Span(" · HICP Energy aggregation inputs ready"),
            ]
        )
    else:
        issues = []
        if not energy_ready:
            issues.append(f"Energy {energy_count}/7")
        if not headline_ready:
            issues.append("Headline incomplete")
        if not aggregate_ready:
            issues.append("Energy aggregation inputs incomplete")
        banner = html.Div(
            [
                html.Strong("Production vintage incomplete"),
                html.Span(f" · {vintage}"),
                html.Span(" · " + "; ".join(issues)),
            ],
            className="estimation-error-text",
        )

    artifact_warnings = list(diagnostics.get("artifact_warnings", []) or [])
    if artifact_warnings:
        validation_banner = html.Div(
            [
                html.Strong("Validation artefacts need attention"),
                html.Span(" · " + "; ".join(artifact_warnings[:4])),
            ],
            className="estimation-error-text",
        )
    else:
        validation_banner = html.Div(
            [
                html.Strong("Persisted validation artefacts"),
                html.Span(" · no recorded HICP validation failures"),
            ]
        )

    return (
        store,
        "READY" if processed else "MISSING",
        f"data/processed/{vintage}",
        "READY" if energy_ready else f"{energy_count}/7",
        "7 canonical component datasets" if energy_ready else "See diagnostics below",
        "READY" if headline_ready else "INCOMPLETE",
        "joint components + weights + official Headline",
        "READY" if linked_ready else "INCOMPLETE",
        "same production information set for Energy + Headline",
        banner,
        diagnostics["rows"],
        validation_banner,
        diagnostics.get("artifact_rows", []),
    )


@callback(
    Output("data-model-energy-status", "children"),
    Output("data-model-energy-note", "children"),
    Output("data-model-aggregate-status", "children"),
    Output("data-model-aggregate-note", "children"),
    Output("data-model-headline-status", "children"),
    Output("data-model-headline-note", "children"),
    Output("data-model-linked-status", "children"),
    Output("data-model-linked-note", "children"),
    Output("data-model-readiness-banner", "children"),
    Output("data-model-readiness-table", "data"),
    Input("production-vintage-select", "value"),
    Input("estimation-config-store", "data"),
    Input("est-model-select", "value"),
    Input("registry-store", "data"),
    Input("estimation-result-store", "data"),
)
def render_production_model_readiness(
    vintage,
    config_store,
    selected_model_id,
    _registry,
    _estimation_result,
):
    if not vintage:
        return (
            "—", "Select a production vintage",
            "—", "not checked",
            "—", "not checked",
            "—", "not checked",
            html.Div("Select a production vintage."),
            [],
        )

    try:
        readiness_key = (
            _registry_snapshot_id(),
            str(vintage),
            str(selected_model_id or ""),
            json.dumps(config_store or {}, sort_keys=True, default=str),
        )
        snapshot = snapshot_get_or_build(
            "production_model_readiness",
            readiness_key,
            lambda: _production_model_readiness(
                str(vintage),
                selected_model_id=selected_model_id,
                config_store=config_store,
            ),
        )
    except Exception as exc:
        return (
            "ERROR", str(exc),
            "ERROR", "readiness unavailable",
            "ERROR", "readiness unavailable",
            "ERROR", "readiness unavailable",
            html.Div(
                [html.Strong("Model readiness failed: "), html.Span(str(exc))],
                className="estimation-error-text",
            ),
            [],
        )

    energy_complete = int(snapshot["energy_complete"])
    energy_total = int(snapshot["energy_total"])
    aggregate_status = str(snapshot["aggregate_status"])
    headline_status = str(snapshot["headline_status"])
    linked_ready = bool(snapshot["linked_ready"])

    if linked_ready:
        banner = html.Div(
            [
                html.Strong("Production model suite ready"),
                html.Span(f" · {vintage}"),
                html.Span(" · Energy 7/7"),
                html.Span(" · aggregate complete"),
                html.Span(" · Headline complete"),
            ]
        )
    else:
        actions = []
        if energy_complete < energy_total:
            actions.append(
                f"Energy next run will reuse {energy_complete}/{energy_total} and estimate "
                f"{energy_total - energy_complete}"
            )
        elif aggregate_status == "BUILD PENDING":
            actions.append("Energy aggregate can be built without Gibbs")
        elif aggregate_status != "COMPLETE":
            actions.append(f"Energy aggregate {aggregate_status.lower()}")
        if headline_status != "COMPLETE":
            actions.append(f"Headline {headline_status.lower()}")
        banner = html.Div(
            [
                html.Strong("Production model suite incomplete"),
                html.Span(f" · {vintage}"),
                html.Span(" · " + "; ".join(actions)),
            ]
        )

    return (
        f"{energy_complete}/{energy_total}",
        (
            "all exact runs reusable"
            if energy_complete == energy_total
            else f"{energy_total - energy_complete} exact run(s) still require estimation"
        ),
        aggregate_status,
        (
            "exact seven-run aggregate reusable"
            if aggregate_status == "COMPLETE"
            else "built only after the exact 7/7 component set is complete"
        ),
        headline_status,
        (
            _short_run(snapshot["headline_run_id"])
            if snapshot.get("headline_run_id")
            else "locked Headline production run"
        ),
        "READY" if linked_ready else "INCOMPLETE",
        "same vintage + complete Energy aggregate + complete Headline",
        banner,
        snapshot["rows"],
    )


@callback(
    Output("est-production-vintage-value", "children"),
    Output("headline-production-vintage-value", "children"),
    Input("production-vintage-store", "data"),
)
def render_production_vintage_badges(store):
    vintage = str((store or {}).get("vintage") or "")
    if not vintage:
        return "Select it in Data", "Select it in Data"
    suffix = " ✓ validated" if (store or {}).get("linked_ready") else " · incomplete"
    label = vintage + suffix
    return label, label




@callback(
    Output("dataset-build-environment", "children"),
    Input("url", "pathname"),
    Input("dataset-build-raw-path", "value"),
    Input("dataset-build-result-store", "data"),
)
def dataset_build_environment(pathname, raw_path, build_result):
    if (pathname or "") != "/data":
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
            require_headline=True,
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
        headline_count = len(dict(store.get("headline_files", {}) or {}))
        banner = html.Div(
            [
                html.Strong("Processed vintage ready: "),
                html.Span(vintage),
                html.Span(
                    f" · {model_count} canonical Energy dataset files"
                    f" · {hicp_count} HICP aggregation sidecars"
                    f" · {headline_count} joint Headline files"
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
    Input("production-vintage-select", "value"),
    Input("estimation-config-store", "data"),
)
def estimation_preview(model_id, vintage, config_store):
    if not model_id or not vintage:
        return (
            "—", "", "—", "", "—", "", "—", "",
            "Select a production vintage in Data.",
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
        if _component_run_reusable(state):
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
        complete = _component_run_reusable(state)
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
        State("production-vintage-select", "value"),
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
        (Output("production-vintage-select", "disabled"), True, False),
    ],
    cancel=[Input("estimation-cancel", "n_clicks")],
    progress=[
        Output("estimation-progress", "value"),
        Output("estimation-phase", "children"),
        Output("estimation-progress-detail", "children"),
        Output("estimation-suite-progress-store", "data"),
    ],
    progress_default=(0, "Idle", "Select a model and production vintage.", {}),
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

        if _component_run_reusable(state):
            try:
                _push_estimation_progress(
                    set_progress,
                    92,
                    "Existing run",
                    "Full posterior, forecast and HICP bridge already exist; rebuilding display if needed.",
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
                    "Checking existing posterior, forecast and HICP production contract.",
                    suite_payload,
                )

                # Reuse deterministic complete run. We still rematerialise its
                # display so the dashboard contract is current.
                if _component_run_reusable(state):
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
    runs = _registry_table("runs")
    if not runs.empty:
        runs = runs.loc[
            runs["model_id"].astype(str).eq(str(model_id))
            & runs["vintage"].astype(str).eq(str(vintage))
            & runs["status"].astype(str).eq("complete")
        ].copy()
    if runs.empty:
        return None
    block = runs.loc[runs["run_id"].astype(str).eq(str(run_id))]
    return None if block.empty else block.iloc[0]


@callback(
    Output("est-diag-run-select", "options"),
    Output("est-diag-run-select", "value"),
    Input("est-model-select", "value"),
    Input("production-vintage-select", "value"),
    Input("registry-store", "data"),
    Input("estimation-result-store", "data"),
    State("est-diag-run-select", "value"),
)
def estimation_diagnostic_run_options(model_id, vintage, _registry, result_store, current):
    if not model_id or not vintage:
        return [], None
    runs = _registry_table("runs")
    if not runs.empty:
        runs = runs.loc[
            runs["model_id"].astype(str).eq(str(model_id))
            & runs["vintage"].astype(str).eq(str(vintage))
            & runs["status"].astype(str).eq("complete")
        ].copy()
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
    Input("production-vintage-select", "value"),
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
        metadata, diagnostics = snapshot_get_or_build(
            "component_diagnostics",
            (_registry_snapshot_id(), str(directory.resolve())),
            lambda: load_component_diagnostics(directory),
        )
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
    Input("production-vintage-select", "value"),
    Input("registry-store", "data"),
    Input("estimation-result-store", "data"),
)
def render_estimation_aggregate_validation(vintage, _registry, _result_store):
    if not vintage:
        return "Select a vintage.", "—", "", "—", "", "—", "", "—", ""
    rows = _registry_table("aggregates")
    if not rows.empty:
        rows = rows.loc[
            rows["vintage"].astype(str).eq(str(vintage))
            & rows["forecast_name"].astype(str).eq("unconditional")
            & rows["status"].astype(str).eq("complete")
        ].copy()
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
        diag = snapshot_get_or_build(
            "aggregate_validation",
            (_registry_snapshot_id(), str(directory.resolve())),
            lambda: load_aggregate_validation(directory),
        )
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
    Output("structural-shock", "options"),
    Output("structural-shock", "value"),
    Output("structural-response", "options"),
    Output("structural-response", "value"),
    Input("ctx-store", "data"),
    Input("url", "pathname"),
    State("structural-shock", "value"),
    State("structural-response", "value"),
)
def structural_controls(context, pathname, current_shock, current_response):
    if (pathname or "") != "/structural":
        raise PreventUpdate
    try:
        directory = _selected_structural_run_directory(context)
        contract = snapshot_get_or_build(
            "structural_run_contract",
            (
                _registry_snapshot_id(),
                str(context.get("model_id")),
                str(context.get("vintage")),
                str(context.get("run_id")),
            ),
            lambda: structural_run_contract(
                directory,
                project_root=PROJECT_ROOT,
            ),
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
            {"label": name.replace("_", " ").title(), "value": name}
            for name in variables
        ]
        banner_children = [
            html.Strong(contract["model_label"]),
            html.Span(f" · vintage {contract['vintage']}"),
            html.Span(f" · run {_short_run(contract['run_id'])}"),
            html.Span(f" · {contract['available_posterior_draws']:,} posterior draws"),
            html.Span(
                " · recursive ordering: "
                + " → ".join(name.replace("_", " ").title() for name in variables)
            ),
            html.Span(f" · missing data: {str(contract['missing_data_method']).upper()}"),
        ]
        if variables and target == variables[0]:
            banner_children.append(
                html.Span(
                    " · Identification note: this saved BVAR is target-first; recursive "
                    "contemporaneous interpretation follows the estimated order.",
                    className="estimation-error-text",
                )
            )
        return (
            html.Div(banner_children),
            date_options,
            variable_options,
            current_shock if current_shock in variables else variables[0],
            variable_options,
            (
                current_response
                if current_response in variables
                else (target if target in variables else variables[-1])
            ),
        )
    except Exception as exc:
        message = html.Div(
            [html.Strong("Structural analysis unavailable: "), html.Span(str(exc))],
            className="estimation-error-text",
        )
        return message, [], [], None, [], None


@callback(
    Output("structural-reference-date", "value"),
    Input("structural-reference-regime", "value"),
    Input("ctx-store", "data"),
    State("structural-volatility-store", "data"),
    State("structural-reference-date", "options"),
    State("structural-reference-date", "value"),
    prevent_initial_call=False,
)
def structural_reference_regime(regime, context, store, options, current):
    values = [item.get("value") for item in (options or []) if item.get("value")]
    if not values:
        return None
    trigger = ctx.triggered_id
    if trigger == "ctx-store":
        return values[-1]
    if trigger == "structural-reference-regime":
        same_run = bool(
            store
            and store.get("ok")
            and context
            and str(store.get("model_id")) == str(context.get("model_id"))
            and str(store.get("run_id")) == str(context.get("run_id"))
        )
        if str(regime or "latest") == "latest" or not same_run:
            return values[-1]
        chosen = reference_regime_date(store, str(regime))
        return chosen if chosen in set(values) else values[-1]
    return current if current in set(values) else values[-1]

@callback(
    Output("structural-shock-size-label", "children"),
    Output("structural-shock-size-unit", "children"),
    Output("structural-shock-interpretation", "children"),
    Input("structural-shock-unit", "value"),
    Input("structural-shock-size", "value"),
    Input("structural-shock", "value"),
    Input("structural-volatility-store", "data"),
)
def structural_shock_definition(shock_unit, shock_size, shock, store):
    mode = str(shock_unit or DEFAULT_SHOCK_UNIT)
    try:
        size = float(shock_size)
    except (TypeError, ValueError):
        size = float(DEFAULT_SHOCK_SIZE)
    shock_name = str(shock or "selected shock").replace("_", " ").title()
    units = dict((store or {}).get("units", {}) or {})
    native_unit = str(units.get(shock, "native units") or "native units")

    if mode == "level":
        label = "Impact size"
        unit = native_unit
        interpretation = html.Div(
            [
                html.Strong("Level-normalised IRF · "),
                html.Span(
                    f"{shock_name} moves by +{size:g} {native_unit} on impact. "
                    "This magnitude applies to the IRF only. FEVD remains defined from "
                    "one-standard-deviation structural shocks; recursive HD is unchanged."
                ),
            ]
        )
    else:
        label = "Shock magnitude"
        unit = "σ"
        interpretation = html.Div(
            [
                html.Strong("Structural-SD IRF · "),
                html.Span(
                    f"{size:g}σ at the selected joint SV reference state. "
                    "The IRF scale depends on the shocked equation's λ at that date. "
                    "FEVD always uses 1σ shocks; recursive HD is unchanged."
                ),
            ]
        )
    return label, unit, interpretation


_STRUCTURAL_CACHE_TTL_SECONDS = 6 * 60 * 60
_STRUCTURAL_FAST_CACHE_NAMESPACE = "structural-fast-v2"
_STRUCTURAL_VOL_CACHE_NAMESPACE = "structural-volatility-v2"
_STRUCTURAL_HD_CACHE_NAMESPACE = "structural-hd-lazy-v1"


def _structural_cache_identity(
    context: dict,
    *,
    posterior_draws: int,
) -> dict:
    return {
        "model_id": str(context.get("model_id") or ""),
        "vintage": str(context.get("vintage") or ""),
        "run_id": str(context.get("run_id") or ""),
        "posterior_draws": int(posterior_draws),
    }


def _structural_vol_cache_key(
    context: dict,
    *,
    posterior_draws: int,
) -> str:
    identity = {
        "namespace": _STRUCTURAL_VOL_CACHE_NAMESPACE,
        **_structural_cache_identity(
            context,
            posterior_draws=posterior_draws,
        ),
    }
    return _STRUCTURAL_VOL_CACHE_NAMESPACE + "::" + json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    )


def _structural_fast_cache_key(
    context: dict,
    *,
    posterior_draws: int,
    reference_date,
    horizon: int,
    shock_unit: str,
    shock_size: float,
) -> str:
    identity = {
        "namespace": _STRUCTURAL_FAST_CACHE_NAMESPACE,
        **_structural_cache_identity(
            context,
            posterior_draws=posterior_draws,
        ),
        "reference_date": (
            None
            if reference_date is None
            else pd.Timestamp(reference_date).isoformat()
        ),
        "horizon": int(horizon),
        "shock_unit": str(shock_unit),
        "shock_size": float(shock_size),
    }
    return _STRUCTURAL_FAST_CACHE_NAMESPACE + "::" + json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    )


def _structural_hd_cache_key(
    identity_payload: dict,
    *,
    split_outlier_amplification: bool,
) -> str:
    identity = {
        "namespace": _STRUCTURAL_HD_CACHE_NAMESPACE,
        "model_id": str(identity_payload.get("model_id") or ""),
        "vintage": str(identity_payload.get("vintage") or ""),
        "run_id": str(identity_payload.get("run_id") or ""),
        "posterior_draws": int(identity_payload.get("posterior_draws") or 0),
        "selected_draw_indices": [
            int(value)
            for value in identity_payload.get("selected_draw_indices", [])
        ],
        "split_outlier_amplification": bool(split_outlier_amplification),
    }
    return _STRUCTURAL_HD_CACHE_NAMESPACE + "::" + json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    )


def _valid_structural_payload(payload: object, context: dict, kind: str) -> bool:
    return bool(
        isinstance(payload, dict)
        and payload.get("ok")
        and str(payload.get("payload_kind") or "") == kind
        and str(payload.get("model_id")) == str(context.get("model_id"))
        and str(payload.get("vintage")) == str(context.get("vintage"))
        and str(payload.get("run_id")) == str(context.get("run_id"))
    )


def _structural_hd_placeholder(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.52,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font={"size": 13, "color": "#6B7280"},
    )
    fig.update_layout(
        template="plotly_white",
        height=560,
        margin={"l": 48, "r": 24, "t": 40, "b": 48},
        xaxis={"visible": False},
        yaxis={"visible": False},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@callback(
    output=Output("structural-volatility-store", "data"),
    inputs=[
        Input("structural-run", "n_clicks"),
        Input("url", "pathname"),
        Input("structural-draws", "value"),
    ],
    state=[
        State("ctx-store", "data"),
        State("structural-volatility-store", "data"),
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
        "Opening Structural automatically activates the frozen default analysis.",
    ),
    prevent_initial_call=False,
)
def load_structural_volatility(
    set_progress,
    _n_clicks,
    pathname,
    posterior_draws,
    context,
    current_store,
):
    if (pathname or "") != "/structural":
        raise PreventUpdate
    if not context or not all(context.get(key) for key in ("model_id", "vintage", "run_id")):
        raise PreventUpdate

    draws_requested = int(posterior_draws or DEFAULT_STRUCTURAL_DRAWS)
    request_signature = {
        "model_id": str(context.get("model_id") or ""),
        "vintage": str(context.get("vintage") or ""),
        "run_id": str(context.get("run_id") or ""),
        "posterior_draws": draws_requested,
    }
    manual_refresh = ctx.triggered_id == "structural-run"
    if (
        not manual_refresh
        and isinstance(current_store, dict)
        and current_store.get("ok")
        and current_store.get("dashboard_request_signature") == request_signature
    ):
        # A route revisit with the same state is intentionally a true no-op.
        raise PreventUpdate

    try:
        directory = _selected_structural_run_directory(context)
        key = _structural_vol_cache_key(
            context,
            posterior_draws=draws_requested,
        )
        cached = _diskcache.get(key)
        if _valid_structural_payload(cached, context, "volatility") and not manual_refresh:
            cached = dict(cached)
            cached["dashboard_request_signature"] = request_signature
            cached.setdefault("cache_usage", {}).update(
                {
                    "server_volatility_cache_hit": True,
                    "cache_namespace": _STRUCTURAL_VOL_CACHE_NAMESPACE,
                }
            )
            set_progress(
                (
                    100,
                    "Volatility ready · cache hit",
                    "Reference-state cards are available; IRF/FEVD resolve automatically.",
                )
            )
            return cached

        set_progress(
            (
                25,
                "Loading posterior volatility",
                f"Reading persisted draws for {_short_run(str(context.get('run_id', '')))}.",
            )
        )
        payload = compute_structural_volatility_v1(
            directory,
            project_root=PROJECT_ROOT,
            posterior_draws=draws_requested,
        )
        payload["dashboard_request_signature"] = request_signature
        payload["cache_usage"] = {
            "server_volatility_cache_hit": False,
            "cache_namespace": _STRUCTURAL_VOL_CACHE_NAMESPACE,
        }
        _diskcache.set(
            key,
            payload,
            expire=_STRUCTURAL_CACHE_TTL_SECONDS,
        )
        set_progress(
            (
                100,
                "Volatility ready",
                "Reference-state cards are available; IRF/FEVD resolve automatically.",
            )
        )
        return payload
    except Exception as exc:
        set_progress((0, "Volatility load failed", str(exc).splitlines()[0]))
        return {
            "ok": False,
            "payload_kind": "volatility",
            "dashboard_request_signature": request_signature,
            "error": f"{type(exc).__name__}: {exc}",
        }


@callback(
    output=Output("structural-store", "data"),
    inputs=[
        Input("structural-volatility-store", "data"),
        Input("structural-reference-date", "value"),
        Input("structural-horizon", "value"),
        Input("structural-shock-unit", "value"),
        Input("structural-shock-size", "value"),
    ],
    state=[
        State("url", "pathname"),
        State("ctx-store", "data"),
        State("structural-draws", "value"),
        State("structural-store", "data"),
    ],
    background=True,
    prevent_initial_call=False,
)
def compute_structural_analysis(
    volatility_store,
    reference_date,
    horizon,
    shock_unit,
    shock_size,
    pathname,
    context,
    posterior_draws,
    current_store,
):
    if (pathname or "") != "/structural":
        raise PreventUpdate
    if not volatility_store or not volatility_store.get("ok"):
        raise PreventUpdate

    draws_requested = int(posterior_draws or DEFAULT_STRUCTURAL_DRAWS)
    horizon_requested = int(horizon or STRUCTURAL_DEFAULT_HORIZON)
    shock_unit_requested = str(shock_unit or DEFAULT_SHOCK_UNIT)
    shock_size_requested = float(shock_size or DEFAULT_SHOCK_SIZE)
    effective_reference = reference_date
    request_signature = {
        "model_id": str(context.get("model_id") or ""),
        "vintage": str(context.get("vintage") or ""),
        "run_id": str(context.get("run_id") or ""),
        "posterior_draws": draws_requested,
        "reference_date": None if effective_reference is None else pd.Timestamp(effective_reference).isoformat(),
        "horizon": horizon_requested,
        "shock_unit": shock_unit_requested,
        "shock_size": shock_size_requested,
    }
    if (
        isinstance(current_store, dict)
        and current_store.get("ok")
        and current_store.get("dashboard_request_signature") == request_signature
    ):
        raise PreventUpdate

    try:
        directory = _selected_structural_run_directory(context)
        key = _structural_fast_cache_key(
            context,
            posterior_draws=draws_requested,
            reference_date=effective_reference,
            horizon=horizon_requested,
            shock_unit=shock_unit_requested,
            shock_size=shock_size_requested,
        )
        cached = _diskcache.get(key)
        if _valid_structural_payload(cached, context, "irf_fevd"):
            cached = dict(cached)
            cached["dashboard_request_signature"] = request_signature
            cached.setdefault("cache_usage", {}).update(
                {
                    "server_fast_cache_hit": True,
                    "cache_namespace": _STRUCTURAL_FAST_CACHE_NAMESPACE,
                }
            )
            return cached

        payload = compute_structural_irf_fevd_v1(
            directory,
            project_root=PROJECT_ROOT,
            reference_date=effective_reference,
            horizon=horizon_requested,
            posterior_draws=draws_requested,
            shock_unit=shock_unit_requested,
            shock_size=shock_size_requested,
        )
        payload["dashboard_request_signature"] = request_signature
        payload["cache_usage"] = {
            "server_fast_cache_hit": False,
            "cache_namespace": _STRUCTURAL_FAST_CACHE_NAMESPACE,
        }
        _diskcache.set(
            key,
            payload,
            expire=_STRUCTURAL_CACHE_TTL_SECONDS,
        )
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "payload_kind": "irf_fevd",
            "dashboard_request_signature": request_signature,
            "error": f"{type(exc).__name__}: {exc}",
        }


@callback(
    output=Output("structural-hd-key-store", "data"),
    inputs=[
        Input("structural-hd-run", "n_clicks"),
        Input("structural-volatility-store", "data"),
        Input("structural-hd-options", "value"),
    ],
    state=[
        State("url", "pathname"),
        State("ctx-store", "data"),
        State("structural-hd-key-store", "data"),
    ],
    background=True,
    running=[
        (Output("structural-hd-run", "disabled"), True, False),
        (Output("structural-hd-cancel", "disabled"), False, True),
    ],
    cancel=[Input("structural-hd-cancel", "n_clicks")],
    prevent_initial_call=False,
)
def resolve_or_compute_structural_hd(
    _n_clicks,
    volatility_store,
    hd_options,
    pathname,
    context,
    current_state,
):
    if (pathname or "") != "/structural":
        raise PreventUpdate
    if not volatility_store or not volatility_store.get("ok"):
        raise PreventUpdate

    split_outliers = "split_outliers" in set(hd_options or [])
    key = _structural_hd_cache_key(
        volatility_store,
        split_outlier_amplification=split_outliers,
    )
    manual_refresh = ctx.triggered_id == "structural-hd-run"
    if (
        not manual_refresh
        and isinstance(current_state, dict)
        and current_state.get("ok")
        and current_state.get("ready")
        and str(current_state.get("cache_key")) == str(key)
    ):
        raise PreventUpdate

    cached = _diskcache.get(key)
    if (
        isinstance(cached, dict)
        and cached.get("ok")
        and str(cached.get("payload_kind")) == "historical_decomposition"
        and str(cached.get("run_id")) == str(volatility_store.get("run_id"))
        and not manual_refresh
    ):
        return {
            "ok": True,
            "ready": True,
            "cache_key": key,
            "cache_hit": True,
            "run_id": cached.get("run_id"),
            "posterior_draws": cached.get("posterior_draws"),
            "diagnostics": cached.get("diagnostics", {}),
        }

    try:
        directory = _selected_structural_run_directory(context)
        payload = compute_structural_hd_v1(
            directory,
            project_root=PROJECT_ROOT,
            posterior_draws=int(
                volatility_store.get("posterior_draws")
                or DEFAULT_STRUCTURAL_DRAWS
            ),
            split_outlier_amplification=split_outliers,
        )
        if [
            int(value)
            for value in payload.get("selected_draw_indices", [])
        ] != [
            int(value)
            for value in volatility_store.get("selected_draw_indices", [])
        ]:
            raise StructuralDashboardError(
                "HD deterministic draw subset differs from the active volatility block."
            )
        _diskcache.set(key, payload, expire=None)
        return {
            "ok": True,
            "ready": True,
            "cache_key": key,
            "cache_hit": False,
            "run_id": payload.get("run_id"),
            "posterior_draws": payload.get("posterior_draws"),
            "diagnostics": payload.get("diagnostics", {}),
        }
    except Exception as exc:
        return {
            "ok": False,
            "ready": False,
            "cache_key": key,
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
    Input("structural-hd-key-store", "data"),
)
def structural_stats(store, hd_state):
    if not store or not store.get("ok"):
        return "—", "", "—", "", "—", "", "—", "", "—", "", "—", ""
    diag = dict(store.get("diagnostics", {}) or {})
    hd_diag = dict((hd_state or {}).get("diagnostics", {}) or {})
    ordering = " → ".join(
        str(name).replace("_", " ").title()
        for name in store.get("recursive_ordering", [])
    )
    ref = pd.Timestamp(store["reference_date"]).strftime(
        "%Y-%m-%d" if store.get("frequency") == "weekly" else "%Y-%m"
    )
    hd_error = hd_diag.get("hd_max_reconstruction_error_drawwise")
    fevd_error = diag.get("fevd_max_share_sum_error")
    return (
        "Recursive",
        ordering,
        ref,
        str(store.get("frequency", "")).title(),
        f"{int(store.get('posterior_draws', 0)):,}",
        f"of {int(store.get('available_posterior_draws', 0)):,} saved draws",
        "—" if hd_error is None else f"{float(hd_error):.3e}",
        (
            "max draw-wise |reconstructed − observed|"
            if hd_error is not None
            else "HD not computed for the active run/settings"
        ),
        "—" if fevd_error is None else f"{float(fevd_error):.3e}",
        "max draw-wise |sum FEVD shares − 1|",
        (
            f"{float(store.get('shock_size', 1.0)):g} native units"
            if str(store.get("shock_unit")) == "level"
            else f"{float(store.get('shock_size', 1.0)):g}σ"
        ),
        (
            "IRF impact-normalised in each shocked variable's native unit; FEVD remains 1σ"
            if str(store.get("shock_unit")) == "level"
            else "IRF in structural standard deviations; FEVD remains 1σ"
        ),
    )


@callback(
    Output("structural-volatility-cards", "children"),
    Output("structural-volatility-relative-graph", "figure"),
    Output("structural-reference-effects", "children"),
    Output("structural-reference-status", "children"),
    Input("structural-volatility-store", "data"),
    Input("structural-reference-date", "value"),
    Input("structural-shock-unit", "value"),
)
def structural_volatility_state(store, reference_date, shock_unit):
    if not store or not store.get("ok"):
        return [], relative_volatility_state_figure(store, reference_date=reference_date), [], ""

    snapshot = reference_volatility_snapshot(store, reference_date)
    cards = []
    for j, item in enumerate(snapshot.get("cards", [])):
        unit = str(item.get("unit") or "")
        sv = item.get("persistent_q50")
        ratio = item.get("relative_sd")
        percentile = item.get("percentile")
        outlier_p = item.get("outlier_probability")
        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(item["label"], style={"fontWeight": 700, "fontSize": "13px", "color": "#0F172A"}),
                            html.Div(
                                "—" if not np.isfinite(percentile) else f"P{int(round(percentile))}",
                                style={"fontSize": "11px", "fontWeight": 700, "color": "#64748B"},
                            ),
                        ],
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    ),
                    dcc.Graph(
                        figure=volatility_sparkline_figure(store, variable=item["variable"], reference_date=reference_date),
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "105px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("√λ ", style={"color": "#64748B"}),
                                    html.Strong("—" if not np.isfinite(sv) else f"{float(sv):.4g}"),
                                    html.Span(f" {unit}" if unit else ""),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Strong("—" if not np.isfinite(ratio) else f"×{float(ratio):.2f}"),
                                    html.Span(" own median", style={"color": "#64748B"}),
                                ]
                            ),
                            html.Div(
                                "" if not np.isfinite(outlier_p) else f"P(outlier) {float(outlier_p):.0%}",
                                style={"color": "#94A3B8", "fontSize": "10px"},
                            ),
                        ],
                        style={"display": "flex", "justifyContent": "space-between", "gap": "10px", "fontSize": "11px", "alignItems": "baseline"},
                    ),
                ],
                style={
                    "border": "1px solid #E2E8F0", "borderRadius": "12px",
                    "background": "#FFFFFF", "padding": "12px 14px 10px 14px",
                    "boxShadow": "0 1px 2px rgba(15,23,42,.04)",
                },
            )
        )

    chip_base = {
        "display": "inline-flex", "alignItems": "center", "padding": "7px 10px",
        "borderRadius": "999px", "fontSize": "11px", "fontWeight": 700,
        "border": "1px solid #DCE3EC", "background": "#F8FAFC", "color": "#334155",
    }
    irf_text = (
        "IRF 1σ · date-dependent"
        if str(shock_unit or "structural_std") == "structural_std"
        else "IRF unit-level · invariant"
    )
    effects = [
        html.Span("✓ " + irf_text, style={**chip_base, "color": "#166534", "background": "#F0FDF4", "borderColor": "#BBF7D0"}),
        html.Span("✓ FEVD · date-dependent", style={**chip_base, "color": "#166534", "background": "#F0FDF4", "borderColor": "#BBF7D0"}),
        html.Span("— Recursive HD · invariant", style=chip_base),
        html.Span("— Outlier multiplier · excluded from IRF / FEVD", style=chip_base),
    ]

    selected = pd.Timestamp(reference_date) if reference_date else None
    computed = pd.Timestamp(store.get("reference_date")) if store.get("reference_date") else None
    frequency = str(store.get("frequency", "monthly"))
    fmt = "%Y-%m-%d" if frequency == "weekly" else "%Y-%m"
    if selected is not None and computed is not None and selected != computed:
        status = html.Div(
            [
                html.Strong(f"Selected {selected.strftime(fmt)} · "),
                html.Span(
                    f"charts are still computed at {computed.strftime(fmt)}. Click Recompute structural analysis to apply this SV state to 1σ IRFs and FEVD."
                ),
            ],
            style={
                "padding": "9px 11px", "borderRadius": "9px", "border": "1px solid #FCD34D",
                "background": "#FFFBEB", "color": "#92400E", "fontSize": "11px",
            },
        )
    else:
        ref_text = computed.strftime(fmt) if computed is not None else "—"
        joint_p = snapshot.get("joint_percentile")
        stress = snapshot.get("joint_stress")
        status = html.Div(
            [
                html.Strong(f"Active state · {ref_text}"),
                html.Span(
                    " · joint stress "
                    + ("—" if stress is None or not np.isfinite(stress) else f"{float(stress):.2f}×")
                    + ("" if joint_p is None or not np.isfinite(joint_p) else f" · P{int(round(float(joint_p)))}")
                ),
            ],
            style={"fontSize": "11px", "color": "#475569"},
        )

    return (
        cards,
        relative_volatility_state_figure(store, reference_date=reference_date),
        effects,
        status,
    )

@callback(
    Output("structural-irf-graph", "figure"),
    Output("structural-irf-table", "data"),
    Input("structural-store", "data"),
    Input("structural-response", "value"),
    Input("structural-shock", "value"),
    Input("structural-irf-metric", "value"),
    Input("structural-irf-fan", "value"),
)
def structural_irf_graph(store, response, shock, metric, fan_mode):
    resolved_metric = metric or "cumulative"
    return (
        irf_figure(
            store, response=response, shock=shock,
            metric=resolved_metric, fan_mode=fan_mode or "68",
        ),
        structural_irf_records(
            store, response=response, shock=shock, metric=resolved_metric
        ),
    )


@callback(
    Output("structural-fevd-graph", "figure"),
    Output("structural-fevd-table", "data"),
    Output("structural-fevd-table", "columns"),
    Input("structural-store", "data"),
    Input("structural-response", "value"),
)
def structural_fevd_graph(store, response):
    rows, columns = structural_fevd_table(store, response=response)
    return fevd_figure(store, response=response), rows, columns


@callback(
    Output("structural-hd-status", "children"),
    Output("structural-hd-status", "className"),
    Input("structural-hd-key-store", "data"),
)
def structural_hd_status(hd_state):
    if not hd_state:
        return (
            "Historical decomposition · waiting for active Structural run.",
            "selection-banner",
        )
    if not hd_state.get("ok"):
        return (
            "Historical decomposition unavailable · "
            + str(hd_state.get("error") or "unknown error"),
            "banner-error",
        )
    if not hd_state.get("ready"):
        return (
            "Historical decomposition not cached for this run/draw/outlier setting. "
            "The rest of Structural is already usable; compute HD only if needed.",
            "selection-banner",
        )
    hit = bool(hd_state.get("cache_hit"))
    draws = int(hd_state.get("posterior_draws") or 0)
    return (
        "Historical decomposition ready"
        + (" · cache hit" if hit else " · newly computed")
        + (f" · {draws:,} draws" if draws else ""),
        "selection-banner",
    )


@callback(
    Output("structural-hd-graph", "figure"),
    Output("structural-hd-table", "data"),
    Input("structural-hd-key-store", "data"),
    Input("structural-response", "value"),
    Input("structural-hd-window", "value"),
    Input("structural-hd-graph", "relayoutData"),
)
def structural_hd_graph(hd_state, response, window, relayout_data):
    if not hd_state or not hd_state.get("ready"):
        return (
            _structural_hd_placeholder(
                "Historical decomposition is not computed for the active run/settings."
            ),
            [],
        )
    key = str(hd_state.get("cache_key") or "")
    payload = _diskcache.get(key) if key else None
    if not isinstance(payload, dict) or not payload.get("ok"):
        return (
            _structural_hd_placeholder(
                "Historical decomposition cache entry is unavailable or expired."
            ),
            [],
        )
    last_obs = None if window is None or int(window) < 0 else int(window)
    return (
        historical_decomposition_figure(
            payload, response=response, last_obs=last_obs,
            relayout_data=relayout_data,
        ),
        structural_hd_records(payload, response=response),
    )



# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

@callback(
    Output("energy-nav", "style"),
    Output("headline-nav", "style"),
    Output("core-nav", "style"),
    Input("url", "pathname"),
)
def domain_navigation_styles(pathname: str | None):
    path = pathname or ""
    hidden = {"display": "none"}
    if path in {"/", "/data", "/overview"}:
        return hidden, hidden, hidden
    if path.startswith("/core"):
        return hidden, hidden, {}
    if domain_from_path(pathname) == "headline":
        return hidden, {}, hidden
    return {}, hidden, hidden



@callback(
    Output("global-topbar", "style"),
    Output("selection-banner", "style"),
    Input("url", "pathname"),
)
def data_shell_visibility(pathname: str | None):
    if (pathname or "") in {
        "/",
        "/overview",
        "/data",
        "/economic-data",
        "/estimation",
        "/headline/estimation",
        "/headline/diagnostics",
    }:
        return {"display": "none"}, {"display": "none"}
    return {}, {}



@callback(
    Output("page-overview", "style"),
    Output("page-data", "style"),
    Output("page-economic-data", "style"),
    Output("page-forecast", "style"),
    Output("page-aggregate", "style"),
    Output("page-scenarios", "style"),
    Output("page-structural", "style"),
    Output("page-estimation", "style"),
    Output("page-headline-forecast", "style"),
    Output("page-headline-scenarios", "style"),
    Output("page-headline-structural", "style"),
    Output("page-headline-diagnostics", "style"),
    Output("page-core-forecast", "style"),
    Output("page-core-scenarios", "style"),
    Input("url", "pathname"),
)
def route(pathname: str | None):
    pathname = pathname or "/overview"
    route_name = {
        "/": "overview",
        "/overview": "overview",
        "/data": "data",
        "/economic-data": (
            "economic-data" if ECONOMIC_DATA_ENABLED else "overview"
        ),
        "/forecast": "forecast",
        "/aggregate": "aggregate",
        "/scenarios": "scenarios",
        "/structural": "structural",
        "/estimation": "estimation",
        "/headline": "headline-forecast",
        "/headline/overview": "headline-forecast",
        "/headline/forecast": "headline-forecast",
        "/headline/contributions": "headline-forecast",
        "/headline/components": "headline-forecast",
        "/headline/scenarios": "headline-scenarios",
        "/headline/structural": "headline-structural",
        "/headline/diagnostics": "headline-estimation",
        "/headline/estimation": "headline-estimation",
        "/core": "core-forecast",
        "/core/forecast": "core-forecast",
        "/core/scenarios": "core-scenarios",
    }.get(pathname, "overview")

    visible = {"display": "block"}
    hidden = {"display": "none"}
    names = (
        "overview",
        "data",
        "economic-data",
        "forecast",
        "aggregate",
        "scenarios",
        "structural",
        "estimation",
        "headline-forecast",
        "headline-scenarios",
        "headline-structural",
        "headline-estimation",
        "core-forecast",
        "core-scenarios",
    )
    return tuple(visible if name == route_name else hidden for name in names)


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
    Output("forecast-summary-table", "data"),
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
        return empty, [], [], "—", "", "—", "", "—", "", "—", ""

    context = dict((store or {}).get("context") or {})
    fig = forecast_figure(frame, metric=metric, series=series, fan_mode=fan_mode, context=context)
    history = _history_rows(frame, metric, series)
    fan = _forecast_rows(frame, metric, series)
    future = fan.loc[fan["segment"] == "forecast"].copy()

    table = fan[["date", "segment", "q05", "q16", "value", "q50", "q84", "q95"]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    for column in ("q05", "q16", "value", "q50", "q84", "q95"):
        table[column] = table[column].round(4)
    table_data = table.to_dict("records")
    summary_data = forecast_summary_records(
        frame, series=series, metric=metric, max_months=None
    )

    observed_value = observed_date = first_value = first_date = terminal_value = terminal_date = "—"
    if not history.empty:
        last = history.iloc[-1]
        observed_value = _format_number(last["value"])
        observed_date = pd.Timestamp(last["date"]).strftime("%Y-%m-%d")
    if not future.empty:
        first = future.iloc[0]
        last = future.iloc[-1]
        first_value = _format_number(first["value"])
        first_date = pd.Timestamp(first["date"]).strftime("%Y-%m-%d")
        terminal_value = _format_number(last["value"])
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
        summary_data,
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

TAX_SCENARIO_INPUT_UX_VERSION = "human-excise-units-v1"


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


def _scenario_excise_display_spec(source_unit: str | None) -> dict[str, object]:
    """Return a human-facing excise unit without changing the model contract.

    Model/source units remain authoritative internally:
      * Electricity: EUR/kWh  -> c€/kWh
      * Gas:         EUR/MWh  -> c€/kWh
      * WOB fuels:   EUR per 1,000 litres -> c€/litre

    ``display_factor`` is defined as:
        displayed_value = source_value * display_factor
    """
    raw = str(source_unit or "source unit")
    norm = (
        raw.casefold()
        .replace("€", "eur")
        .replace(",", "")
        .replace(" ", "")
        .replace("liters", "litres")
    )

    if "eur/kwh" in norm or "eurperkwh" in norm:
        return {
            "source_unit": raw,
            "display_unit": "c€/kWh",
            "display_factor": 100.0,
            "step": 0.1,
            "scaled": True,
        }
    if "eur/mwh" in norm or "eurpermwh" in norm:
        # 1 EUR/MWh = 0.1 c€/kWh.
        return {
            "source_unit": raw,
            "display_unit": "c€/kWh",
            "display_factor": 0.1,
            "step": 0.1,
            "scaled": True,
        }
    if (
        "eurper1000litres" in norm
        or "eur/1000litres" in norm
        or "eurper1000litre" in norm
        or "eur/1000litre" in norm
    ):
        # 1 EUR / 1,000 litres = 0.1 c€/litre.
        return {
            "source_unit": raw,
            "display_unit": "c€/litre",
            "display_factor": 0.1,
            "step": 0.1,
            "scaled": True,
        }

    return {
        "source_unit": raw,
        "display_unit": raw,
        "display_factor": 1.0,
        "step": 0.1,
        "scaled": False,
    }


def _scenario_excise_to_display(value, source_unit: str | None) -> float:
    spec = _scenario_excise_display_spec(source_unit)
    return float(value or 0.0) * float(spec["display_factor"])


def _scenario_excise_from_display(value, source_unit: str | None) -> float:
    spec = _scenario_excise_display_spec(source_unit)
    factor = float(spec["display_factor"])
    if factor <= 0:
        raise ValueError("Excise display factor must be positive.")
    return float(value or 0.0) / factor


def _scenario_excise_plausibility_error(
    *,
    displayed_delta: float,
    source_delta: float,
    source_baseline: float,
    source_unit: str | None,
) -> str | None:
    """Reject only unmistakable unit mistakes; normal stress tests remain valid."""
    spec = _scenario_excise_display_spec(source_unit)
    scenario_source = float(source_baseline) + float(source_delta)

    if scenario_source < -1e-12:
        return (
            "The requested excise change would make the excise level negative. "
            f"Baseline is {_scenario_excise_to_display(source_baseline, source_unit):.2f} "
            f"{spec['display_unit']}."
        )

    baseline_abs = abs(float(source_baseline))
    if bool(spec["scaled"]) and baseline_abs > 1e-12:
        multiple = abs(float(source_delta)) / baseline_abs
        if multiple > 10.0:
            return (
                f"Excise change {float(displayed_delta):+.2f} {spec['display_unit']} "
                f"is {multiple:.1f}× the baseline excise. This is likely a unit mistake. "
                f"The editor expects {spec['display_unit']}, while the model stores "
                f"{spec['source_unit']} internally."
            )
    return None


def _scenario_component_contract_for_row(
    vintage: str,
    model_id: str,
    row,
):
    """Load the saved-run tax-scenario contract once per frozen snapshot."""
    from energy_bvar_io import load_energy_bvar_forecast

    directory = Path(str(row["directory"]))
    key = (
        _registry_snapshot_id(),
        str(vintage),
        str(model_id),
        str(directory.resolve()),
    )

    def _build():
        forecast = load_energy_bvar_forecast(directory)
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

    return snapshot_get_or_build(
        "tax_scenario_contract",
        key,
        _build,
    )



def _cached_conditional_aggregate_contract(directory: Path) -> dict:
    directory = Path(directory)
    return snapshot_get_or_build(
        "conditional_aggregate_contract",
        (_registry_snapshot_id(), str(directory.resolve())),
        lambda: conditional_aggregate_contract(directory),
    )


def _cached_conditional_component_contract(
    directory: Path,
    *,
    model_id: str,
) -> dict:
    directory = Path(directory)
    return snapshot_get_or_build(
        "conditional_component_contract",
        (
            _registry_snapshot_id(),
            str(directory.resolve()),
            str(model_id),
        ),
        lambda: conditional_component_contract(
            directory,
            model_id=str(model_id),
            project_root=PROJECT_ROOT,
        ),
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
    frame = _registry_table("aggregates")
    if not frame.empty:
        frame = frame.loc[
            frame["vintage"].astype(str).eq(str(vintage))
        ].copy()
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
            contract = _cached_conditional_aggregate_contract(directory)
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
        contract = _cached_conditional_aggregate_contract(Path(str(row["directory"])))
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
        contract = _cached_conditional_component_contract(
            Path(str(row["directory"])),
            model_id=str(model_id),
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
        contract = _cached_conditional_component_contract(
            Path(str(row["directory"])),
            model_id=str(model_id),
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
    Output("conditional-effect-table", "data"),
    Input("conditional-store", "data"),
    Input("conditional-component-select", "value"),
)
def conditional_stats(store, model_id):
    payload = conditional_set_payload(store, model_id)
    values = conditional_kpis(payload)
    if values.get("condition_level") is None:
        return "—", "", "—", "", "—", "", "—", "", []

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
        dual_impact_records(
            payload.get("component_impact_yoy"),
            payload.get("aggregate_impact_yoy"),
            max_months=None,
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
    forecasts = _registry_table("forecasts")
    if not forecasts.empty:
        forecasts = forecasts.loc[
            forecasts["model_id"].astype(str).eq(str(model_id))
            & forecasts["vintage"].astype(str).eq(str(vintage))
            & forecasts["forecast_name"].astype(str).eq(str(forecast_name))
        ].copy()
    if forecasts.empty:
        return None
    runs = _registry_table("runs")
    if not runs.empty:
        runs = runs.loc[
            runs["model_id"].astype(str).eq(str(model_id))
            & runs["vintage"].astype(str).eq(str(vintage))
            & runs["status"].astype(str).eq("complete")
        ].copy()
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


@callback(
    Output("scenario-start-date","date"),
    Output("scenario-start-date","min_date_allowed"),
    Output("scenario-start-date","max_date_allowed"),
    Output("scenario-vat-delta","value"),
    Output("scenario-excise-delta","value"),
    Output("scenario-excise-label","children"),
    Output("scenario-excise-delta","step"),
    Output("scenario-support-note","children"),
    Output("scenario-start-date","disabled"),
    Output("scenario-vat-delta","disabled"),
    Output("scenario-excise-delta","disabled"),
    Input("scenario-component-select","value"),
    Input("vintage-select","value"),
    Input("forecast-select","value"),
    Input("registry-store","data"),
    Input("scenario-store","data"),
)
def scenario_control_defaults(model_id, vintage, forecast_name, _, scenario_store):
    if not model_id or not vintage or not forecast_name:
        return (
            None, None, None, 0.0, 0.0, "Excise change", 0.1,
            "No scenario-capable component forecast is available.",
            True, True, True,
        )

    row = _scenario_component_forecast_row(vintage, model_id, forecast_name)
    if row is None:
        return (
            None, None, None, 0.0, 0.0, "Excise change", 0.1,
            f"{model_spec(model_id).label}: no unique saved forecast. "
            "Promote one run if several coexist.",
            True, True, True,
        )

    try:
        contract = _scenario_component_contract_for_row(
            str(vintage), model_id, row
        )
    except Exception as exc:
        return (
            None, None, None, 0.0, 0.0, "Excise change", 0.1,
            f"Scenario controls unavailable: {exc}",
            True, True, True,
        )

    existing = scenario_set_payload(scenario_store, model_id)
    em = dict((existing or {}).get("meta", {}) or {})
    source_unit = str(contract.get("excise_unit", "source unit"))
    display = _scenario_excise_display_spec(source_unit)

    frequency = str(contract.get("frequency", "monthly"))
    min_start = _normalise_scenario_period(contract["min_start"], frequency)
    max_start = _normalise_scenario_period(contract["max_start"], frequency)
    default_start = _normalise_scenario_period(contract["default_start"], frequency)
    start = _normalise_scenario_period(
        em.get("scenario_start") or default_start,
        frequency,
    )
    if start < min_start or start > max_start:
        start = default_start

    source_baseline = float(contract["baseline_excise_at_start"])
    display_baseline = _scenario_excise_to_display(
        source_baseline, source_unit
    )
    stored_source_delta = float(em.get("excise_delta", 0.0) or 0.0)
    display_delta = _scenario_excise_to_display(
        stored_source_delta, source_unit
    )

    freq_label = (
        "monthly periods"
        if frequency == "monthly"
        else "weekly periods (Monday)"
    )
    tax_source_note = (
        "WOB VAT + excise tax block · "
        if frequency == "weekly"
        else "VAT + excise tax bridge · "
    )

    conversion_note = ""
    if bool(display["scaled"]):
        conversion_note = (
            f" Editor unit: {display['display_unit']}; model/source unit: "
            f"{display['source_unit']}."
        )

    note = (
        f"{model_spec(model_id).label} · run {str(row['run_id'])[:12]} · "
        f"available scenario window {min_start.date().isoformat()} — "
        f"{max_start.date().isoformat()} ({freq_label}) · "
        f"{tax_source_note}baseline at first future period: VAT "
        f"{contract['baseline_vat_at_start']:.2f}% · excise "
        f"{display_baseline:.2f} {display['display_unit']}."
        f"{conversion_note} Enter a non-zero VAT and/or excise change; "
        "zero/zero removes the component from the active set."
    )

    return (
        start.date(),
        min_start.date(),
        max_start.date(),
        float(em.get("vat_delta_pp", 0.0) or 0.0),
        display_delta,
        f"Excise change ({display['display_unit']})",
        float(display["step"]),
        note,
        False,
        False,
        False,
    )


@callback(
    Output("scenario-tax-preview", "children"),
    Input("scenario-component-select", "value"),
    Input("vintage-select", "value"),
    Input("forecast-select", "value"),
    Input("scenario-vat-delta", "value"),
    Input("scenario-excise-delta", "value"),
    Input("registry-store", "data"),
)
def scenario_tax_input_preview(
    model_id,
    vintage,
    forecast_name,
    vat_delta,
    excise_delta_display,
    _,
):
    if not model_id or not vintage or not forecast_name:
        return "Select a scenario-capable component to preview the tax change."

    row = _scenario_component_forecast_row(vintage, model_id, forecast_name)
    if row is None:
        return "No unique saved component forecast is available."

    try:
        contract = _scenario_component_contract_for_row(
            str(vintage), model_id, row
        )
        source_unit = str(contract.get("excise_unit", "source unit"))
        display = _scenario_excise_display_spec(source_unit)
        baseline_source = float(contract["baseline_excise_at_start"])
        baseline_display = _scenario_excise_to_display(
            baseline_source, source_unit
        )
        delta_display = float(excise_delta_display or 0.0)
        delta_source = _scenario_excise_from_display(
            delta_display, source_unit
        )
        scenario_display = _scenario_excise_to_display(
            baseline_source + delta_source,
            source_unit,
        )
        base_vat = float(contract["baseline_vat_at_start"])
        delta_vat = float(vat_delta or 0.0)
        scenario_vat = base_vat + delta_vat
        guard = _scenario_excise_plausibility_error(
            displayed_delta=delta_display,
            source_delta=delta_source,
            source_baseline=baseline_source,
            source_unit=source_unit,
        )
    except Exception as exc:
        return html.Span(f"Tax-input preview unavailable: {exc}")

    pieces = [
        html.Strong("Scenario preview · "),
        html.Span(
            f"VAT {base_vat:.2f}% {delta_vat:+.2f} pp → "
            f"{scenario_vat:.2f}%"
        ),
        html.Span(
            f" · Excise {baseline_display:.2f} "
            f"{display['display_unit']} {delta_display:+.2f} → "
            f"{scenario_display:.2f} {display['display_unit']}"
        ),
    ]
    if bool(display["scaled"]):
        pieces.append(
            html.Span(
                f" · internal model change "
                f"{delta_source:+.6g} {display['source_unit']}",
                style={"color": "#64748b"},
            )
        )
    if guard:
        pieces.append(
            html.Span(
                " · WARNING: " + guard,
                style={"color": "#B42318", "fontWeight": "700"},
            )
        )
    return pieces


@callback(
    Output("scenario-store","data"),
    Output("scenario-banner","children"),
    Input("scenario-apply","n_clicks"),
    Input("scenario-remove","n_clicks"),
    Input("scenario-reset-all","n_clicks"),
    State("vintage-select","value"),
    State("forecast-select","value"),
    State("scenario-component-select","value"),
    State("scenario-start-date","date"),
    State("scenario-vat-delta","value"),
    State("scenario-excise-delta","value"),
    State("scenario-store","data"),
    prevent_initial_call=True,
)
def mutate_scenario_set(
    _apply,
    _remove,
    _reset,
    vintage,
    forecast_name,
    model_id,
    start_date,
    vat_delta,
    excise_delta_display,
    current_store,
):
    trigger = ctx.triggered_id

    if trigger == "scenario-reset-all":
        return (
            clear_scenario_set(
                vintage=vintage,
                forecast_name=forecast_name,
            ),
            html.Div(
                "All component tax scenarios cleared.",
                className="selection-banner",
            ),
        )

    if not model_id:
        return no_update, html.Div(
            "Select a component.",
            className="banner-error",
        )

    if trigger == "scenario-remove":
        return (
            remove_scenario_component(current_store, model_id),
            html.Div(
                f"{model_spec(model_id).label}: scenario removed.",
                className="selection-banner",
            ),
        )

    if trigger != "scenario-apply":
        raise PreventUpdate

    if not vintage or not forecast_name or start_date is None:
        return no_update, html.Div(
            "Vintage, forecast and start date are required.",
            className="banner-error",
        )

    row = _scenario_component_forecast_row(
        vintage, model_id, forecast_name
    )
    if row is None:
        return no_update, html.Div(
            f"{model_spec(model_id).label}: no unique promoted/saved "
            "run is available.",
            className="banner-error",
        )

    vat_delta = float(vat_delta or 0.0)
    excise_delta_display = float(excise_delta_display or 0.0)

    try:
        from energy_bvar_io import load_energy_bvar_forecast

        forecast_dir = Path(str(row["directory"]))
        run_dir = forecast_dir.parent.parent
        forecast = load_energy_bvar_forecast(forecast_dir)
        dataset = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / str(vintage)
            / model_spec(model_id).dataset_file
        )
        contract = component_tax_scenario_contract(
            model_id,
            forecast,
            dataset_path=dataset,
        )

        source_unit = str(
            contract.get("excise_unit", "source unit")
        )
        display = _scenario_excise_display_spec(source_unit)
        excise_delta = _scenario_excise_from_display(
            excise_delta_display,
            source_unit,
        )

        if (
            abs(vat_delta) < 1e-15
            and abs(excise_delta) < 1e-15
        ):
            return (
                remove_scenario_component(
                    current_store, model_id
                ),
                html.Div(
                    f"{model_spec(model_id).label}: zero changes, "
                    "component removed from the active set.",
                    className="selection-banner",
                ),
            )

        guard = _scenario_excise_plausibility_error(
            displayed_delta=excise_delta_display,
            source_delta=excise_delta,
            source_baseline=float(
                contract["baseline_excise_at_start"]
            ),
            source_unit=source_unit,
        )
        if guard:
            return no_update, html.Div(
                [
                    html.Strong("Scenario not applied: "),
                    html.Span(guard),
                ],
                className="banner-error",
            )

        frequency = str(contract.get("frequency", "monthly"))
        effective_start = _normalise_scenario_period(
            start_date, frequency
        )
        min_start = _normalise_scenario_period(
            contract["min_start"], frequency
        )
        max_start = _normalise_scenario_period(
            contract["max_start"], frequency
        )
        if effective_start < min_start or effective_start > max_start:
            return no_update, html.Div(
                f"Scenario start must lie inside the selected forecast "
                f"horizon: {min_start.date().isoformat()} — "
                f"{max_start.date().isoformat()}.",
                className="banner-error",
            )

        result = build_saved_component_tax_scenario(
            run_dir,
            project_root=PROJECT_ROOT,
            forecast_name=str(forecast_name),
            start_date=effective_start,
            vat_delta_pp=vat_delta,
            excise_delta=excise_delta,
            max_draws=500,
        )
        payload = scenario_payload(result)
        payload.setdefault("meta", {}).update(
            {
                "vintage": str(vintage),
                "run_id": str(row["run_id"]),
                "forecast_name": str(forecast_name),
            }
        )
        updated = upsert_scenario_component(
            current_store,
            payload,
            vintage=str(vintage),
            forecast_name=str(forecast_name),
        )

    except Exception as exc:
        return no_update, html.Div(
            [
                html.Strong("Scenario calculation failed: "),
                html.Span(str(exc)),
            ],
            className="banner-error",
        )

    meta = payload["meta"]
    count = len(scenario_set_components(updated))
    display_delta = _scenario_excise_to_display(
        float(meta.get("excise_delta", 0.0) or 0.0),
        meta.get("excise_unit"),
    )
    display_unit = _scenario_excise_display_spec(
        meta.get("excise_unit")
    )["display_unit"]

    return updated, html.Div(
        [
            html.Strong(str(result["hicp_label"])),
            html.Span(f" · VAT {vat_delta:+.2f} pp"),
            html.Span(
                f" · excise {display_delta:+.2f} {display_unit}"
            ),
            html.Span(
                f" · {meta['n_draws_effective']} paired draws"
            ),
            html.Span(
                f" · {count} active component scenario"
                + ("s" if count != 1 else "")
            ),
        ],
        className="selection-banner",
    )


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
        excise_display_spec = _scenario_excise_display_spec(row.get("excise_unit"))
        excise_display_delta = _scenario_excise_to_display(
            row.get("excise_delta", 0.0),
            row.get("excise_unit"),
        )
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
                            html.Span(
                                f" · excise {excise_display_delta:+.2f} "
                                f"{excise_display_spec['display_unit']}"
                            ),
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


@callback(Output("scenario-baseline-terminal","children"),Output("scenario-terminal-date","children"),Output("scenario-scenario-terminal","children"),Output("scenario-terminal-date-2","children"),Output("scenario-impact-terminal","children"),Output("scenario-impact-interval","children"),Output("scenario-draws","children"),Output("scenario-draws-note","children"),Output("scenario-effect-table","data"),Input("scenario-store","data"),Input("scenario-component-select","value"))
def scenario_stat_cards(store,model_id):
    values=scenario_kpis(scenario_set_payload(store,model_id))
    if values.get("terminal_date") is None: return "—","","—","","—","","—","",[]
    date=pd.Timestamp(values["terminal_date"]).date().isoformat(); low=values.get("impact_low"); high=values.get("impact_high"); interval="" if low is None or high is None else f"68% [{low:+.2f}, {high:+.2f}] pp"; draws=values.get("n_draws")
    payload=scenario_set_payload(store,model_id)
    frames=scenario_frames(payload) if payload else {}
    fans=frames.get("fans", pd.DataFrame())
    def _tax_block(name):
        if fans is None or fans.empty or "name" not in fans:
            return pd.DataFrame()
        return fans.loc[fans["name"].astype(str).eq(name)].copy()
    table=paired_effect_records(
        _tax_block("baseline_yoy"),
        _tax_block("scenario_yoy"),
        _tax_block("yoy_impact_pp"),
        max_months=None,
    )
    return f"{values['baseline_terminal']:.2f}%",date,f"{values['scenario_terminal']:.2f}%",date,f"{values['impact_terminal']:+.2f} pp",interval,("—" if draws is None else f"{int(draws):,}"),"matched baseline/scenario",table


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
    State("conditional-agg-select", "value"),
    State("vintage-select", "value"),
)
def selected_tax_aggregate_marginal(
    tax_store,
    aggregate_run_id,
    vintage,
):
    """Precompute every active component's marginal Energy effect once.

    The only Input is ``scenario-store``; that store changes after the explicit
    Apply/Remove/Reset scenario action. Selecting a component afterwards merely
    chooses among these already-computed results.
    """
    summaries = scenario_set_summary(tax_store)
    if not summaries or not aggregate_run_id or not vintage:
        return None

    aggregate_row = _selected_aggregate_row(vintage, aggregate_run_id)
    if aggregate_row is None:
        return None
    directory = Path(str(aggregate_row["directory"]))
    metadata = _aggregate_metadata(directory)
    run_ids = _run_ids_from_aggregate_metadata(metadata)
    if not run_ids:
        return {
            "by_model": {},
            "errors": {
                "*": "Selected aggregate lacks exact component-run provenance."
            },
        }

    forecast_name = str(
        metadata.get("forecast_name") or "unconditional"
    )
    by_model: dict[str, dict] = {}
    errors: dict[str, str] = {}

    for item in summaries:
        model_id = str(item.get("model_id") or "")
        payload = scenario_set_payload(tax_store, model_id)
        if not model_id or not payload:
            continue

        single = clear_scenario_set(
            vintage=str(vintage),
            forecast_name=forecast_name,
        )
        single = upsert_scenario_component(
            single,
            payload,
            vintage=str(vintage),
            forecast_name=forecast_name,
        )
        tax_scenarios = scenario_set_to_tax_scenarios(single)
        signature = scenario_set_signature(single)
        cache_key = "selected-tax-agg-v3::" + json.dumps(
            {
                "aggregate": str(aggregate_run_id),
                "model_id": model_id,
                "signature": signature,
            },
            sort_keys=True,
            default=str,
        )
        cached = _diskcache.get(cache_key)
        if isinstance(cached, dict):
            by_model[model_id] = cached
            continue

        try:
            outcome = run_aggregate(
                str(vintage),
                project_root=PROJECT_ROOT,
                results_root=RESULTS_ROOT,
                forecast_name=forecast_name,
                run_ids=run_ids,
                n_aggregate_draws=min(
                    500,
                    max(
                        1,
                        int(
                            metadata.get("n_aggregate_draws_requested")
                            or 500
                        ),
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
                    "scenario_components": [model_id],
                    "scenario_signature": signature,
                },
            )
            result.setdefault("meta", {}).update(
                {
                    "aggregate_run_id": str(aggregate_run_id),
                    "selected_tax_model_id": model_id,
                }
            )
            _diskcache.set(cache_key, result, expire=3600)
            by_model[model_id] = result
        except Exception as exc:
            errors[model_id] = f"{type(exc).__name__}: {exc}"

    return {
        "by_model": by_model,
        "errors": errors,
        "aggregate_run_id": str(aggregate_run_id),
        "vintage": str(vintage),
    }


@callback(
    Output("scenario-tax-energy-graph", "figure"),
    Input("scenario-tax-selected-agg-store", "data"),
    Input("scenario-fan", "value"),
    Input("scenario-component-select", "value"),
)
def selected_tax_energy_figure(store, fan_mode, model_id):
    if not store or not model_id:
        return empty_aggregate_figure(
            "Add/select a tax scenario to display its marginal HICP Energy effect."
        )
    errors = dict(store.get("errors") or {})
    if model_id in errors:
        return empty_aggregate_figure(str(errors[model_id]))
    payload = dict((store.get("by_model") or {}).get(str(model_id)) or {})
    if not payload:
        return empty_aggregate_figure(
            "The selected component has no precomputed marginal Energy effect."
        )
    return tidy_aggregate_figure(
        aggregate_live_impact_figure(
            payload,
            fan_mode=fan_mode or "68",
            uirevision=(
                f"{payload.get('meta',{}).get('aggregate_run_id','agg')}::"
                f"{payload.get('meta',{}).get('selected_tax_model_id','tax')}::marginal"
            ),
        )
    )


def _selected_aggregate_row(vintage: str | None, aggregate_run_id: str | None):
    if not vintage or not aggregate_run_id:
        return None
    frame = _registry_table("aggregates")
    if not frame.empty:
        frame = frame.loc[
            frame["vintage"].astype(str).eq(str(vintage))
        ].copy()
    if frame.empty:
        return None
    selected = frame.loc[frame["aggregate_run_id"].astype(str) == str(aggregate_run_id)]
    return None if selected.empty else selected.iloc[0]


def _aggregate_metadata(directory: Path) -> dict:
    directory = Path(directory)
    key = (_registry_snapshot_id(), str(directory.resolve()))

    def _build():
        path = directory / "metadata.json"
        if not path.is_file():
            fallback = directory / "aggregate_config.json"
            path = fallback if fallback.is_file() else path
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    return snapshot_get_or_build(
        "aggregate_metadata",
        key,
        _build,
    )


def _aggregate_forecast_origin(directory: Path) -> pd.Timestamp | None:
    """Latest first-future month across component stores, loaded once."""
    directory = Path(directory)
    key = (_registry_snapshot_id(), str(directory.resolve()))

    def _build():
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
                first = (
                    pd.Timestamp(future[0])
                    .to_period("M")
                    .to_timestamp(how="start")
                )
                starts.append(first)
            except Exception:
                continue
        return max(starts) if starts else None

    return snapshot_get_or_build(
        "aggregate_forecast_origin",
        key,
        _build,
    )


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
    frame = _registry_table("aggregates")
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
    frame = _registry_table("aggregates")
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

    frame_rows = _registry_table("aggregates")
    if not frame_rows.empty:
        frame_rows = frame_rows.loc[
            frame_rows["vintage"].astype(str).eq(str(vintage))
        ].copy()
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

    aggregate_snapshot_ref = (
        f"{_registry_snapshot_id()}::aggregate-display::"
        f"{vintage}::{aggregate_run_id}"
    )
    cached_aggregate = snapshot_get(
        "aggregate_display",
        aggregate_snapshot_ref,
    )
    if (
        isinstance(cached_aggregate, tuple)
        and len(cached_aggregate) == 2
    ):
        return cached_aggregate

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
        aggregate_metadata = _aggregate_metadata(directory)
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

    # Tax-layer accounting is derived once from the exact selected aggregate,
    # its saved pairing provenance and the same processed tax contexts used by
    # production.  Presentation callbacks below consume only the frozen frame.
    tax_decomposition_frame = pd.DataFrame()
    tax_decomposition_meta: dict = {"available": False}
    try:
        tax_decomposition_frame, tax_contract = build_tax_contribution_decomposition(
            directory,
            project_root=PROJECT_ROOT,
        )
        tax_decomposition_meta = {"available": True, **dict(tax_contract)}
        meta["tax_decomposition_available"] = True
        meta["tax_decomposition_contract_version"] = tax_contract.get(
            "contract_version"
        )
        meta["tax_decomposition_additivity_max_error"] = tax_contract.get(
            "aggregate_additivity_max_abs_error"
        )
    except Exception as exc:
        tax_decomposition_meta = {
            "available": False,
            "reason": str(exc),
        }
        meta["tax_decomposition_available"] = False
        meta["tax_decomposition_reason"] = str(exc)

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
    fitted_snapshot_ref = aggregate_snapshot_ref + "::fitted"
    contribution_snapshot_ref = aggregate_snapshot_ref + "::history-contrib"
    tax_decomposition_snapshot_ref = aggregate_snapshot_ref + "::tax-decomposition"
    snapshot_put_frame(aggregate_snapshot_ref, frame)
    if not fitted_frame.empty:
        snapshot_put_frame(fitted_snapshot_ref, fitted_frame)
    if not historical_contribution_frame.empty:
        snapshot_put_frame(
            contribution_snapshot_ref,
            historical_contribution_frame,
        )
    if not tax_decomposition_frame.empty:
        snapshot_put_frame(
            tax_decomposition_snapshot_ref,
            tax_decomposition_frame,
        )

    aggregate_store = {
        "snapshot_ref": aggregate_snapshot_ref,
        "frame_json": _json_frame(frame),
        "meta": meta,
        "fitted_snapshot_ref": fitted_snapshot_ref,
        "fitted_frame_json": (
            None if fitted_frame.empty else _json_frame(fitted_frame)
        ),
        "fitted_meta": fitted_meta,
        "historical_contribution_snapshot_ref": contribution_snapshot_ref,
        "historical_contribution_frame_json": (
            None
            if historical_contribution_frame.empty
            else _json_frame(historical_contribution_frame)
        ),
        # Heavy tax-accounting data remain server-side; the browser carries
        # only the immutable reference plus compact audit metadata.
        "tax_decomposition_snapshot_ref": tax_decomposition_snapshot_ref,
        "tax_decomposition_meta": tax_decomposition_meta,
    }
    result = (aggregate_store, banner)
    snapshot_put(
        "aggregate_display",
        aggregate_snapshot_ref,
        result,
    )
    return result


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
    return snapshot_frame_from_store(
        store,
        json_key="fitted_frame_json",
        reference_key="fitted_snapshot_ref",
    )


def _historical_contribution_frame_from_store(store: dict | None) -> pd.DataFrame:
    return snapshot_frame_from_store(
        store,
        json_key="historical_contribution_frame_json",
        reference_key="historical_contribution_snapshot_ref",
    )


def _tax_decomposition_frame_from_store(store: dict | None) -> pd.DataFrame:
    # Intentionally no JSON fallback: this accounting object is built once at
    # aggregate bootstrap and kept server-side.  If it is gone, require an
    # explicit aggregate/snapshot reload rather than silently touching disk.
    return snapshot_frame_from_store(
        store,
        json_key="__tax_decomposition_no_browser_payload__",
        reference_key="tax_decomposition_snapshot_ref",
    )


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
            "paths over each component's full natural valid history after the same "
            "tax/frequency adapters. Aggregated fitted begins only at the natural "
            "common six-component overlap and is aggregated draw-by-draw with "
            "Laspeyres weights. The deterministic accounting reconstruction is "
            "audit-only and is not plotted as model fit."
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
    Output("agg-summary-table", "data"),
    Input("agg-store", "data"),
    Input("agg-metric", "value"),
)
def aggregate_stat_cards(store: dict | None, metric: str | None):
    frame = _frame_from_store(store)
    if frame.empty or not metric:
        return ("—", "", "—", "", "—", "", "—", "", [])
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
        forecast_summary_records(frame, series=None, metric=metric, max_months=None),
    )


@callback(
    Output("agg-graph", "figure"),
    Input("agg-store", "data"),
    Input("agg-metric", "value"),
    Input("agg-fan", "value"),
    Input("agg-historical-overlays", "value"),
)
def aggregate_graph(
    store: dict | None,
    metric: str | None,
    fan_mode: str | None,
    historical_overlays: list[str] | None,
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
            part = part.dropna(subset=["date", "posterior_mean"])
            if part.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=part["date"],
                    y=part["posterior_mean"].astype(float),
                    mode="lines",
                    name=f"{label} BVAR fitted · posterior mean",
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
        part = part.dropna(subset=["date", "posterior_mean"])
        if not part.empty:
            fig.add_trace(
                go.Scatter(
                    x=part["date"],
                    y=part["posterior_mean"].astype(float),
                    mode="lines",
                    name="Aggregated BVAR fitted HICP Energy · posterior mean",
                    line={"width": 2.7, "dash": "dash", "color": "#7C3AED"},
                    hovertemplate=(
                        "%{y:.2f}<extra>Aggregated BVAR fitted HICP Energy</extra>"
                    ),
                )
            )

    # Freeze the server-rendered economic figure. Browser pan/zoom remains
    # interactive via Plotly, but relayoutData no longer calls Python or
    # automatically changes the y-axis.
    return tidy_aggregate_figure(
        fig,
        height=680,
        uirevision=revision,
    )


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
    Output("agg-tax-horizon", "options"),
    Output("agg-tax-horizon", "value"),
    Input("agg-store", "data"),
    State("agg-tax-horizon", "value"),
)
def aggregate_tax_horizon_options(
    store: dict | None, current: str | None
):
    tax_frame = _tax_decomposition_frame_from_store(store)
    meta = dict((store or {}).get("meta", {}) or {})
    options = tax_horizon_options(tax_frame, meta.get("forecast_origin"))
    values = [str(item.get("value")) for item in options]
    if current in values:
        return options, current
    # Priority contract: Latest observed, then M+1 / M+2 / M+3.
    preferred = next(
        (key for key in ("latest", "m1", "m2", "m3", "m6", "m12") if key in values),
        None,
    )
    return options, preferred


@callback(
    Output("agg-tax-contrib", "figure"),
    Input("agg-store", "data"),
    Input("agg-tax-component", "value"),
    Input("agg-tax-mode", "value"),
)
def aggregate_tax_contribution_chart(
    store: dict | None,
    component: str | None,
    display_mode: str | None,
):
    tax_frame = _tax_decomposition_frame_from_store(store)
    meta = dict((store or {}).get("meta", {}) or {})
    tax_meta = dict((store or {}).get("tax_decomposition_meta", {}) or {})
    if tax_frame.empty:
        reason = str(tax_meta.get("reason") or "")
        if bool(tax_meta.get("available")) and not reason:
            reason = (
                "Tax decomposition snapshot expired. Reload the aggregate or use "
                "Refresh snapshot explicitly."
            )
        return empty_aggregate_figure(
            reason or "Tax-layer contribution decomposition unavailable"
        )
    return tax_contribution_figure(
        tax_frame,
        component=str(component or "all"),
        forecast_origin=meta.get("forecast_origin"),
        display_mode=str(display_mode or "bars"),
        uirevision=f"{meta.get('aggregate_run_id', 'agg')}::tax-layers",
    )


@callback(
    Output("agg-tax-table", "data"),
    Output("agg-tax-note", "children"),
    Output("agg-tax-table-title", "children"),
    Input("agg-store", "data"),
    Input("agg-tax-horizon", "value"),
)
def aggregate_tax_contribution_table(
    store: dict | None, horizon: str | None
):
    tax_frame = _tax_decomposition_frame_from_store(store)
    tax_meta = dict((store or {}).get("tax_decomposition_meta", {}) or {})
    meta = dict((store or {}).get("meta", {}) or {})
    if tax_frame.empty:
        reason = str(tax_meta.get("reason") or "")
        if bool(tax_meta.get("available")) and not reason:
            reason = (
                "Tax decomposition snapshot expired. Reload the aggregate or use "
                "Refresh snapshot explicitly."
            )
        return [], html.Div(
            reason or "Tax-layer contribution decomposition unavailable.",
            className="banner-error",
        ), "Tax-layer contribution — unavailable"
    if not horizon:
        return [], html.Div(
            "No readable tax-decomposition period is available.",
            className="selection-banner",
        ), "Tax-layer contribution — no period available"
    records, note = tax_matrix_records(
        tax_frame,
        horizon=str(horizon),
        forecast_origin=meta.get("forecast_origin"),
    )
    period_title = tax_matrix_period_title(
        tax_frame,
        horizon=str(horizon),
        forecast_origin=meta.get("forecast_origin"),
    )
    # Human-readable signed pp values; the underlying frozen frame remains
    # numeric and is shared with the chart.
    numeric_columns = (
        "Pre-tax / market",
        "Excise",
        "VAT",
        "Unsplit",
        "Tax total",
        "Total contribution",
    )
    formatted = []
    for row in records:
        item = dict(row)
        for key in numeric_columns:
            value = item.get(key)
            item[key] = (
                "—"
                if value is None or not np.isfinite(float(value))
                else f"{float(value):+.3f}"
            )
        formatted.append(item)

    audit = tax_meta.get("aggregate_additivity_max_abs_error")
    audit_text = (
        ""
        if audit is None or not np.isfinite(float(audit))
        else f" · additivity max error {float(audit):.2e}"
    )
    origin = meta.get("forecast_origin")
    origin_text = ""
    if origin is not None:
        try:
            origin_text = f" · forecast origin {pd.Timestamp(origin).strftime('%B %Y')}"
        except Exception:
            origin_text = ""
    return formatted, html.Div(
        [
            html.Strong(note),
            html.Span(origin_text),
            html.Span(audit_text),
            html.Span(
                " · VAT includes VAT-on-excise; tax splits require comparable t and t-12 bridges; "
                "otherwise the exact contribution remains Unsplit."
            ),
        ],
        className="selection-banner",
    ), period_title


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


def _conditional_headline_bridge_data(
    conditional_store: Mapping[str, Any] | None,
    *,
    selected_vintage: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Build the Headline bridge from one exact active Energy conditional.

    The conditional computation already produced and replay-validated the
    draw-wise HICP Energy level paths. This function only compacts their saved
    posterior summaries; it does not rerun a BVAR or a conditional forecast.
    """
    summaries = conditional_set_summary(conditional_store)
    if not summaries:
        return None, None
    if len(summaries) != 1:
        return None, (
            "Headline transfer requires exactly one active Energy conditional scenario. "
            "The current Energy conditionals are standalone marginal scenarios and are "
            "not jointly propagated or summed."
        )

    summary = summaries[0]
    model_id = str(summary.get("model_id") or "")
    payload = conditional_set_payload(conditional_store, model_id)
    if not payload:
        return None, "The active Energy conditional payload is unavailable."
    if str(payload.get("contract_version") or "") != CONDITIONAL_CONTRACT_VERSION:
        return None, (
            "The active Energy conditional predates the Headline bridge payload. "
            "Return to Energy Scenarios and click Add / update conditional once."
        )

    meta = dict(payload.get("meta", {}) or {})
    vintage = str(meta.get("vintage") or "")
    aggregate_run_id = str(meta.get("aggregate_run_id") or "")
    forecast_name = str(meta.get("forecast_name") or "unconditional")
    if not vintage or not aggregate_run_id or not model_id:
        return None, "The active Energy conditional lacks vintage/aggregate/model lineage."
    if selected_vintage and str(selected_vintage) != vintage:
        return None, (
            f"The active Energy conditional belongs to vintage {vintage}; "
            f"Headline currently selects {selected_vintage}."
        )

    baseline = pd.DataFrame(list(payload.get("aggregate_baseline_level", []) or []))
    scenario = pd.DataFrame(list(payload.get("aggregate_scenario_level", []) or []))
    required = {"date", "mean", "q16", "q50", "q84"}
    if baseline.empty or scenario.empty or not required.issubset(baseline.columns) or not required.issubset(scenario.columns):
        return None, (
            "The active Energy conditional does not contain bridge-ready HICP Energy "
            "level summaries. Recompute it once with Add / update conditional."
        )

    try:
        baseline["date"] = pd.to_datetime(baseline["date"]).dt.to_period("M").dt.to_timestamp()
        scenario["date"] = pd.to_datetime(scenario["date"]).dt.to_period("M").dt.to_timestamp()
    except Exception as exc:
        return None, f"The Energy conditional bridge calendar is unreadable: {exc}"
    baseline = baseline.sort_values("date").reset_index(drop=True)
    scenario = scenario.sort_values("date").reset_index(drop=True)
    if baseline["date"].duplicated().any() or scenario["date"].duplicated().any():
        return None, "The Energy conditional bridge contains duplicate months."
    if not baseline["date"].equals(scenario["date"]):
        return None, "Energy conditional baseline/scenario bridge calendars differ."
    if len(scenario) < 2:
        return None, "The Energy conditional bridge needs at least two monthly levels."

    baseline_level: dict[str, list[float]] = {}
    scenario_level: dict[str, list[float]] = {}
    for statistic in ("mean", "q16", "q50", "q84"):
        b = pd.to_numeric(baseline[statistic], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(scenario[statistic], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(b).all() or not np.isfinite(s).all() or np.any(b <= 0) or np.any(s <= 0):
            return None, f"Energy conditional {statistic} level path contains invalid values."
        baseline_level[statistic] = b.tolist()
        scenario_level[statistic] = s.tolist()

    condition_variable = str(meta.get("condition_variable") or "")
    condition_description = str(meta.get("condition_description") or "")
    scenario_start = meta.get("scenario_start")
    signature = [{
        "model_id": model_id,
        "condition_variable": condition_variable,
        "condition_description": condition_description,
        "aggregate_run_id": aggregate_run_id,
    }]
    bridge = {
        "contract": ENERGY_BRIDGE_CONTRACT_VERSION,
        "vintage": vintage,
        "aggregate_run_id": aggregate_run_id,
        "forecast_name": forecast_name,
        "scenario_active": True,
        "n_draws": int(meta.get("n_aggregate_draws_paired") or 0),
        "dates": [pd.Timestamp(x).isoformat() for x in scenario["date"]],
        "baseline_level": baseline_level,
        "scenario_level": scenario_level,
        "scenario_signature": signature,
        "scenario_components": [model_id],
        "scenario_starts": ({model_id: scenario_start} if scenario_start else {}),
        "transfer_contract": "month_to_month_growth_reanchored_on_headline_energy",
        "interpretation": (
            "one deterministic Energy HICP conditional summary path is converted to "
            "monthly growth; full Energy-path uncertainty is not integrated out inside "
            "the Headline BVAR"
        ),
    }
    store = {
        "meta": {
            "vintage": vintage,
            "aggregate_run_id": aggregate_run_id,
            "aggregate_forecast_name": forecast_name,
            "n_draws": bridge["n_draws"],
            "scenario_type": "conditional",
            "scenario_components": [model_id],
            "scenario_count": 1,
            "scenario_starts": bridge["scenario_starts"],
        },
        "headline_bridge": bridge,
    }
    return store, None


@callback(
    Output("agg-scenario-store", "data"),
    Output("agg-scenario-banner", "children"),
    Input("scenario-store", "data"),
    Input("conditional-store", "data"),
    State("agg-select", "value"),
    State("vintage-select", "value"),
)
def compute_live_aggregate_tax_scenario(
    scenario_store,
    conditional_store,
    aggregate_run_id,
    vintage,
):
    # This callback is downstream only of explicit scenario actions. Route,
    # aggregate-dropdown and vintage navigation are deliberately States so
    # browsing the dashboard can never launch a new aggregate calculation.
    conditional_summaries = conditional_set_summary(conditional_store)
    tax_summaries = scenario_set_summary(scenario_store)
    if conditional_summaries and tax_summaries:
        return None, html.Div(
            [
                html.Strong("Energy → Headline bridge blocked: "),
                html.Span(
                    "both a conditional scenario and a tax scenario are active. "
                    "Clear one set first; the current contracts do not define an exact joint propagation."
                ),
            ],
            className="banner-error",
        )
    if conditional_summaries:
        bridge_store, error = _conditional_headline_bridge_data(
            conditional_store,
            selected_vintage=vintage,
        )
        if error:
            return None, html.Div(
                [html.Strong("Energy → Headline bridge blocked: "), html.Span(error)],
                className="banner-error",
            )
        bridge = dict((bridge_store or {}).get("headline_bridge") or {})
        component = (bridge.get("scenario_components") or ["conditional"])[0]
        return bridge_store, html.Div(
            [
                html.Strong("Conditional Energy scenario linked"),
                html.Span(
                    f" · {str(component).replace('_', ' ')} · vintage {bridge.get('vintage')} "
                    f"· {int(bridge.get('n_draws') or 0):,} paired aggregate draws"
                ),
            ],
            className="selection-banner",
        )

    if not aggregate_run_id or not vintage:
        return None, "Select an aggregate run."
    summaries = tax_summaries
    if not summaries:
        return None, "No active component tax scenario. Configure one or more components in Scenarios."
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
    if (
        isinstance(cached, dict)
        and str(((cached.get("headline_bridge") or {}).get("contract")))
        == ENERGY_BRIDGE_CONTRACT_VERSION
    ):
        payload = cached
    else:
        try:
            outcome=run_aggregate(str(vintage),project_root=PROJECT_ROOT,results_root=RESULTS_ROOT,forecast_name=forecast_name,run_ids=run_ids,n_aggregate_draws=n_draws,pairing_seed=pairing_seed,tax_scenarios=tax_scenarios,weekly_tax_mode="strict",persist=False)
            payload=aggregate_live_scenario_payload(outcome,combined_meta); payload.setdefault("meta",{}).update({"aggregate_run_id":str(aggregate_run_id),"aggregate_forecast_name":forecast_name}); payload["headline_bridge"]=energy_outcome_headline_bridge(outcome,combined_meta); _diskcache.set(cache_key,payload,expire=3600)
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


# ---------------------------------------------------------------------------
# Headline slice-3 callbacks — display_v1 only, no filesystem I/O
# ---------------------------------------------------------------------------


def _headline_pct(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value:.2f}%"


def _headline_date(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")









# ---------------------------------------------------------------------------
# Headline slice-4 callbacks — structural + diagnostics from display_v1 only
# ---------------------------------------------------------------------------


def _headline_diag_number(value, *, percent=False, scientific=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    if scientific:
        return f"{number:.2e}"
    if percent:
        return f"{100.0 * number:.2f}%"
    return f"{number:.3f}"


@callback(
    Output("headline-root-table", "data"),
    Output("headline-mcmc-table", "data"),
    Output("headline-validation-table", "data"),
    Output("headline-diag-radius", "children"),
    Output("headline-diag-rejection", "children"),
    Output("headline-diag-hd-error", "children"),
    Output("headline-diag-fevd-error", "children"),
    Input("data-store", "data"),
)
def headline_diagnostics_callback(store: dict | None):
    frame = _frame_from_store(store)
    kpis = stability_diagnostic_kpis(frame)
    return (
        root_diagnostic_records(frame),
        mcmc_diagnostic_records(frame),
        validation_diagnostic_records(frame),
        _headline_diag_number(kpis.get("median_spectral_radius")),
        _headline_diag_number(kpis.get("B_instability_rejection_rate"), percent=True),
        _headline_diag_number(kpis.get("hd_max_reconstruction_error"), scientific=True),
        _headline_diag_number(kpis.get("fevd_max_share_sum_error"), scientific=True),
    )




# ---------------------------------------------------------------------------
# Headline Dashboard Slice 5 — callbacks on the real unified Dash app
# ---------------------------------------------------------------------------

register_headline_slice5_callbacks(
    app,
    results_root=RESULTS_ROOT,
    registry_path=REGISTRY_PATH,
    background_manager=background_callback_manager,
    store_id="data-store",
    vintage_selector_id="production-vintage-select",
)



# ---------------------------------------------------------------------------
# Headline Dashboard Slice 6 — conditional/scenario callbacks
# ---------------------------------------------------------------------------

register_headline_slice6_callbacks(
    app,
    results_root=RESULTS_ROOT,
    project_root=PROJECT_ROOT,
    store_id="data-store",
    energy_scenario_store_id="agg-scenario-store",
)

# Headline Structural — live saved-posterior analysis with the same UX contract
# as Energy Structural. No BVAR re-estimation occurs in these callbacks.
register_headline_structural_callbacks(
    app,
    results_root=RESULTS_ROOT,
    project_root=PROJECT_ROOT,
    store_id="data-store",
)

register_core_dashboard_callbacks(
    app,
    results_root=RESULTS_ROOT,
    project_root=PROJECT_ROOT,
    store_id="data-store",
)

register_overview_callbacks(
    app,
    results_root=RESULTS_ROOT,
    registry_path=REGISTRY_PATH,
    project_root=PROJECT_ROOT,
    production_vintage_store_id="production-vintage-store",
    registry_store_id="registry-store",
)

if (
    ECONOMIC_DATA_ENABLED
    and _register_economic_data_callbacks is not None
):
    _register_economic_data_callbacks(
        app,
        project_root=PROJECT_ROOT,
        registry_store_id="registry-store",
    )


if __name__ == "__main__":
    host = os.getenv("ENERGY_BVAR_DASH_HOST", "127.0.0.1")
    port = int(os.getenv("ENERGY_BVAR_DASH_PORT", "8050"))
    debug = os.getenv("ENERGY_BVAR_DASH_DEBUG", "0") in {"1", "true", "True"}
    app.run(host=host, port=port, debug=debug)
