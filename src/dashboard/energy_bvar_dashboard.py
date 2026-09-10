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
CONDITIONAL_WINDOWS_HARMONIZED_V1_ENERGY_DASH = True

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
from dashboard_xlsx_export import register_xlsx_export

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
    ConditionalScenarioError,
    clear_conditional_set,
    compute_conditional_scenario,
    compute_joint_energy_scenario,
    conditional_path_signature_fields,
    joint_energy_contribution_impact_figure,
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
    SELECTED_ENERGY_BASELINE_CONTRACT_VERSION,
    energy_outcome_headline_bridge,
    run_saved_headline_energy_marginal,
    run_saved_headline_selected_energy_contribution,
)
from energy_bvar_theme import TOKENS, apply_theme, fan_traces  # noqa: E402
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
from energy_bvar_model import (  # noqa: E402
    BVARSVOPriorConfig,
    SamplerConfig,
    default_bvar_svo_prior_config,
    outlier_mean_frequency_from_years,
    outlier_prior_observations_from_years,
    periods_per_year,
)
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
    headline_scenario_impact_figure,
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


# ENERGY_ESTIMATION_MISSING_METHOD_INDEPENDENT_V1
def _paper_baseline_configs(model_id: str):
    """Frequency-aware literature/project baseline for one Energy model."""
    spec = model_spec(model_id)
    return (
        default_bvar_svo_prior_config(spec.frequency, outlier_interval_years=4.0),
        SamplerConfig(seed=spec.seed),
    )


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
                "missing_data_method": str(spec.missing_data_method),
            }
        )
    return rows


def _baseline_config_payload() -> dict:
    prior = default_bvar_svo_prior_config("monthly", outlier_interval_years=4.0)
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
            "outlier_interval_years": 4.0,
            "outlier_interval_unit": "years",
            "outlier_prior_strength_years": 10.0,
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
        "missing_data_method": "baseline",
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
    prior = dict(payload.get("prior", {}) or {})
    if "outlier_interval_years" not in prior and "outlier_every" in prior:
        # Legacy profiles stored a denominator in model periods but did not
        # record which frequency it referred to. The old baseline was 48, so
        # migrate using the monthly convention and surface a warning in the UI.
        prior["outlier_interval_years"] = float(prior["outlier_every"]) / 12.0
        prior["outlier_interval_unit"] = "years"
        payload["legacy_outlier_profile_migration"] = (
            "Legacy period-based outlier interval converted using 12 periods/year; "
            "review this value before re-saving the profile."
        )
    if "outlier_prior_strength_years" not in prior:
        if "outlier_prior_observations" in prior:
            prior["outlier_prior_strength_years"] = float(
                prior["outlier_prior_observations"]
            ) / 12.0
            payload["legacy_outlier_strength_migration"] = (
                "Legacy period-based outlier prior strength converted using "
                "12 periods/year; review before re-saving the profile."
            )
        else:
            prior["outlier_prior_strength_years"] = 10.0
    payload.setdefault("missing_data_method", "baseline")
    payload["prior"] = prior
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


def _prior_from_mapping(
    values: dict,
    *,
    frequency: str = "monthly",
) -> BVARSVOPriorConfig:
    interval_years = _safe_float(
        values.get("outlier_interval_years"), "mean outlier interval in years"
    )
    outlier_frequency = outlier_mean_frequency_from_years(
        interval_years, frequency
    )
    prior_strength_years = _safe_float(
        values.get("outlier_prior_strength_years"),
        "outlier prior confidence in years",
    )
    prior = BVARSVOPriorConfig(
        lambda1=_safe_float(values.get("lambda1"), "lambda1"),
        lambda2=_safe_float(values.get("lambda2"), "lambda2"),
        lambda3=_safe_float(values.get("lambda3"), "lambda3"),
        lambda4=_safe_float(values.get("lambda4"), "lambda4"),
        a_prior_var=_safe_float(values.get("a_prior_var"), "A prior variance"),
        phi_prior_mean=_safe_float(values.get("phi_prior_mean"), "phi prior mean"),
        phi_prior_df=_safe_float(values.get("phi_prior_df"), "phi prior df"),
        h0_var=_safe_float(values.get("h0_var"), "initial log-vol variance"),
        outlier_mean_frequency=outlier_frequency,
        outlier_prior_observations=outlier_prior_observations_from_years(
            prior_strength_years, frequency
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


def _missing_method_override(value) -> str | None:
    key = str(value or "baseline").strip().lower()
    if key in {"", "baseline", "model_baseline", "default"}:
        return None
    if key not in {"linear", "dk"}:
        raise ValueError("Missing-data method must be baseline, linear or dk.")
    return key


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
    # Missing-data treatment is independent from the prior/MCMC profile.
    # Per-model scope remains authoritative for row-specific missing methods.
    if not config_store or not config_store.get("valid"):
        raise ValueError(
            "Estimation configuration is invalid. Correct the highlighted settings first."
        )

    profile_mode = str(config_store.get("profile_mode", "paper_baseline"))
    scope = str(config_store.get("scope", "selected_only"))
    spec = model_spec(model_id)
    shared_missing = _missing_method_override(
        config_store.get("missing_data_method", "baseline")
    )

    if profile_mode == "paper_baseline":
        prior, sampler = _paper_baseline_configs(model_id)
        spec_overrides = {}
        if shared_missing is not None:
            spec_overrides["missing_data_method"] = shared_missing
        return prior, sampler, spec_overrides

    if scope == "selected_only" and model_id != selected_model_id:
        prior, sampler = _paper_baseline_configs(model_id)
        spec_overrides = {}
        if shared_missing is not None:
            spec_overrides["missing_data_method"] = shared_missing
        return prior, sampler, spec_overrides

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
        method = _missing_method_override(
            row.get("missing_data_method", "baseline")
        )
        if method is not None:
            spec_overrides["missing_data_method"] = method
    else:
        spec_overrides = {}
        if shared_missing is not None:
            spec_overrides["missing_data_method"] = shared_missing

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
                        structure["reference_month"],
                        "electricity reference month",
                    )
                    if reference not in range(1, 13):
                        raise ValueError(
                            "Electricity reference month must lie in 1..12."
                        )
                    spec_overrides["reference_month"] = reference

    prior = _prior_from_mapping(
        shared_prior,
        frequency=spec.frequency,
    )
    sampler = _sampler_from_mapping(
        shared_sampler,
        seed_fallback=spec.seed,
    )
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
        # PORTABLE_PRODUCTION_STATE_P1_V1
        from production_manifest import restore_production_state
        restore_production_state(
            project_root=PROJECT_ROOT,
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

# GRAPH_EXPORT_READABILITY_G3_FORECAST_V1
# FORECAST_ENERGY_PRESENTATION_LOT2_AXES_V1
def forecast_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    series: str,
    fan_mode: str,
    context: dict,
) -> go.Figure:
    """Continuous observed -> nowcast -> forecast chart with bounded visible axes."""
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

    # Presentation-only axis contract.  Autoscale can be distorted by dated
    # rows whose plotted values are NaN and by quantile columns that are not
    # visible in the selected fan mode.  Bound the axes from visible finite
    # points only, then add a symmetric 5% margin.
    fan_axis_columns = ["value"]
    if fan_mode in {"68", "both"}:
        fan_axis_columns.extend(["q16", "q84"])
    if fan_mode in {"90", "both"}:
        fan_axis_columns.extend(["q05", "q95"])
    fan_axis_columns = list(dict.fromkeys(fan_axis_columns))

    x_parts = []
    y_parts = []

    if not history.empty:
        history_values = pd.to_numeric(history["value"], errors="coerce").to_numpy(dtype=float)
        history_visible = np.isfinite(history_values)
        if history_visible.any():
            x_parts.append(
                pd.Series(
                    pd.to_datetime(
                        history.loc[history_visible, "date"],
                        errors="coerce",
                    )
                )
            )
            y_parts.append(history_values[history_visible])

    if not fan.empty:
        fan_visible = np.zeros(len(fan), dtype=bool)
        for column in fan_axis_columns:
            if column not in fan.columns:
                continue
            values = pd.to_numeric(fan[column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            fan_visible |= finite
            if finite.any():
                y_parts.append(values[finite])
        if fan_visible.any():
            x_parts.append(
                pd.Series(
                    pd.to_datetime(
                        fan.loc[fan_visible, "date"],
                        errors="coerce",
                    )
                )
            )

    x_range = None
    if x_parts:
        visible_dates = pd.concat(x_parts, ignore_index=True).dropna()
        if not visible_dates.empty:
            x_min = pd.Timestamp(visible_dates.min())
            x_max = pd.Timestamp(visible_dates.max())
            span = x_max - x_min
            pad = span * 0.05 if span > pd.Timedelta(0) else pd.Timedelta(days=1)
            x_range = [x_min - pad, x_max + pad]

    y_range = None
    if y_parts:
        visible_values = np.concatenate(y_parts)
        visible_values = visible_values[np.isfinite(visible_values)]
        if visible_values.size:
            y_min = float(np.min(visible_values))
            y_max = float(np.max(visible_values))
            span = y_max - y_min
            if span > 0:
                pad = 0.05 * span
            else:
                pad = max(abs(y_min) * 0.05, 1e-6)
            y_range = [y_min - pad, y_max + pad]

    fig.update_layout(
        template="plotly_white",
        margin={"l": 54, "r": 24, "t": 82, "b": 42},
        height=520,
        title={"text": label, "x": 0.01, "xanchor": "left", "font": {"size": 18, "color": "#111827"}},
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#374151", "size": 12},
        xaxis_title=None, yaxis_title=unit, hovermode="x unified", dragmode="pan",
        hoverlabel={"bgcolor": "white", "bordercolor": "#e5e7eb", "font": {"color": "#111827"}},
        legend={"orientation": "h", "y": 1.13, "x": 1, "xanchor": "right", "font": {"size": 11}},
        uirevision=f"forecast-axes-v1::{context.get('model_id')}::{series}::{metric}::{fan_mode}",
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#e5e7eb",
        tickfont={"color": "#6b7280"},
        range=x_range,
        autorange=x_range is None,
    )
    fig.update_yaxes(
        gridcolor="#eef0f3",
        zerolinecolor="#d1d5db",
        tickfont={"color": "#6b7280"},
        range=y_range,
        autorange=y_range is None,
    )
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
        paper_bgcolor="white",
        plot_bgcolor="white",
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

_RESULT_BLOCK_STATES = frozenset({"fresh", "computing", "stale"})


def _result_block(
    *,
    eyebrow: Any,
    value: Any = None,
    unit_date: Any = None,
    uncertainty: Any = None,
    provenance: Any,
    state: str = "fresh",
    state_text: str | None = None,
) -> html.Div:
    """Canonical compact result card for scenario propagation outputs.

    The helper is intentionally id-less. Dynamic IDs/stores are introduced only
    by the owning layout/callback block. ``provenance`` is mandatory so every
    materialised result carries its source contract on screen.

    States
    ------
    fresh
        Current result for the active scenario.
    computing
        Stable card geometry. If a previous value exists it remains visible in
        soft ink; otherwise a same-height skeleton occupies the value slot.
    stale
        Previous result retained with reduced emphasis and an explicit stale
        message until the user requests a recomputation.
    """
    state = str(state or "fresh").strip().lower()
    if state not in _RESULT_BLOCK_STATES:
        raise ValueError(
            f"Unknown result-block state {state!r}; "
            f"expected one of {sorted(_RESULT_BLOCK_STATES)}."
        )
    if provenance is None or (
        isinstance(provenance, str) and not provenance.strip()
    ):
        raise ValueError("result-block provenance is mandatory.")

    has_value = not (
        value is None
        or (isinstance(value, str) and not value.strip())
    )
    value_classes = ["result-block-value"]
    if state == "computing" and has_value:
        value_classes.append("result-block-value--previous")

    if state == "computing" and not has_value:
        value_node = html.Div(
            html.Span(className="result-block-skeleton-bar"),
            className="result-block-value-frame",
        )
    else:
        value_node = html.Div(
            value if has_value else "—",
            className=" ".join(value_classes),
        )

    status_node = None
    if state == "computing":
        status_node = html.Div(
            state_text or "Updating…",
            className="result-block-state-text",
        )
    elif state == "stale":
        status_node = html.Div(
            state_text or "Stale — scenario changed",
            className="result-block-state-text",
        )

    children = [
        html.Div(eyebrow, className="result-block-eyebrow"),
        value_node,
        html.Div(
            unit_date if unit_date not in (None, "") else "—",
            className="result-block-unit-date",
        ),
        html.Div(
            uncertainty if uncertainty not in (None, "") else "—",
            className="result-block-uncertainty",
        ),
    ]
    if status_node is not None:
        children.append(status_node)
    children.extend(
        [
            html.Div(className="result-block-rule"),
            html.Div(provenance, className="result-block-provenance"),
        ]
    )
    return html.Div(
        children,
        className=f"result-block result-block--{state}",
        **{
            "data-result-state": state,
            "aria-live": "polite",
        },
    )


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


# FORECAST_ENERGY_PRESENTATION_LOT3_CONTROLS_V1
# FORECAST_ENERGY_POST_LOT4_UI_CLEANUP_V1_1
def _energy_forecast_control_bar(view: str) -> html.Div:
    """Stable six-slot control bar shared by both Energy Forecast views."""
    view = str(view or "").strip().lower()
    if view not in {"components", "aggregate"}:
        raise ValueError(f"Unknown Energy Forecast view {view!r}.")

    model_component_id = "forecast-model-select" if view == "components" else "agg-model-select"
    model_title = (
        "Select the component forecast shown in the Components view."
        if view == "components"
        else (
            "Selected component context only. This selector does not change or "
            "recompute the saved HICP Energy aggregate."
        )
    )
    model_control = html.Div(
        [
            html.Label("Model", className="control-label"),
            dcc.Dropdown(
                id=model_component_id,
                options=[
                    {"label": model_spec(model_id).label, "value": model_id}
                    for model_id in ENERGY_SUITE_MODEL_IDS
                ],
                value="gas",
                clearable=False,
                persistence=True,
                persistence_type="session",
                className="compact-dropdown",
                style={"width": "100%", "minWidth": 0},
            ),
        ],
        className="control-block",
        style={"minWidth": 0},
        title=model_title,
    )

    if view == "aggregate":
        run_control = html.Div(
            [
                html.Label("Aggregate run", className="control-label"),
                dcc.Dropdown(
                    id="agg-select",
                    placeholder="Select an aggregate",
                    clearable=False,
                    className="compact-dropdown",
                    style={"width": "100%", "minWidth": 0},
                ),
            ],
            className="control-block",
            style={"minWidth": 0},
            title="Saved HICP Energy aggregate displayed in this view.",
        )
    else:
        run_control = html.Div(
            [
                html.Label("Aggregate run", className="control-label"),
                dcc.Dropdown(
                    id="forecast-aggregate-run-disabled",
                    options=[{"label": "Not used in Components", "value": "not_applicable"}],
                    value="not_applicable",
                    clearable=False,
                    disabled=True,
                    className="compact-dropdown",
                    style={"width": "100%", "minWidth": 0},
                ),
            ],
            className="control-block",
            style={"minWidth": 0},
            title=(
                "Not applicable in Components. The component view uses the "
                "selected saved component run from the page context."
            ),
        )

    metric_control = html.Div(
        [
            html.Label("Metric", className="control-label"),
            dcc.Dropdown(
                id="forecast-metric" if view == "components" else "agg-metric",
                clearable=False,
                className="compact-dropdown",
                style={"width": "100%", "minWidth": 0},
            ),
        ],
        className="control-block",
        style={"minWidth": 0},
    )

    if view == "components":
        series_control = html.Div(
            [
                html.Label("Series", className="control-label"),
                dcc.Dropdown(
                    id="forecast-series",
                    clearable=False,
                    className="compact-dropdown",
                    style={"width": "100%", "minWidth": 0},
                ),
            ],
            className="control-block",
            style={"minWidth": 0},
            title="Series from the selected component's saved display artefact.",
        )
    else:
        series_control = html.Div(
            [
                html.Label("Series", className="control-label"),
                dcc.Dropdown(
                    id="agg-series-disabled",
                    options=[{"label": "HICP Energy", "value": "hicp_energy"}],
                    value="hicp_energy",
                    clearable=False,
                    disabled=True,
                    className="compact-dropdown",
                    style={"width": "100%", "minWidth": 0},
                ),
            ],
            className="control-block",
            style={"minWidth": 0},
            title=(
                "The Aggregate view is fixed to HICP Energy. Choose Components "
                "to inspect an individual component series."
            ),
        )

    fan_control = html.Div(
        [
            html.Label("Fan", className="control-label"),
            dcc.RadioItems(
                id="forecast-fan" if view == "components" else "agg-fan",
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
        style={"minWidth": 0},
    )

    if view == "aggregate":
        overlays_control = html.Div(
            [
                html.Label("Model fit overlays", className="control-label"),
                dcc.Checklist(
                    id="agg-historical-overlays",
                    options=[],
                    value=[],
                    inline=True,
                    className="fan-radio",
                ),
                # Callback anchor retained, explanatory body hidden.
                html.Div(id="agg-overlay-note", style={"display": "none"}),
            ],
            className="control-block",
            style={"minWidth": 0},
            title="Optional fitted-history overlays for the HICP Energy aggregate.",
        )
    else:
        overlays_control = html.Div(
            [
                html.Label("Model fit overlays", className="control-label"),
                dcc.Checklist(
                    id="forecast-overlays-disabled",
                    options=[
                        {
                            "label": "Aggregate view only",
                            "value": "aggregate_only",
                            "disabled": True,
                        }
                    ],
                    value=[],
                    inline=True,
                    className="fan-radio",
                ),
            ],
            className="control-block",
            style={"minWidth": 0},
            title=(
                "Model-fit overlays are an Aggregate-view diagnostic and do not "
                "apply to the component fan chart."
            ),
        )

    return html.Div(
        [model_control, run_control, metric_control, series_control, fan_control, overlays_control],
        className="panel chart-controls forecast-energy-control-bar",
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
            "columnGap": "20px",
            "rowGap": "14px",
            "alignItems": "start",
            "width": "100%",
            "boxSizing": "border-box",
            "marginBottom": "14px",
            "overflow": "visible",
        },
    )

# FORECAST_ENERGY_PRESENTATION_LOT4_TABLE_V1
def forecast_page() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Forecast", className="page-title"),
                    html.P(
                        "Observed data, ragged-edge nowcast and posterior predictive forecast from the selected saved run.",
                        className="page-subtitle",
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            _energy_forecast_control_bar("components"),
            html.Div(
                [
                    _stat_card("Latest observed", "stat-observed", "stat-observed-date"),
                    _stat_card("First future mean", "stat-first", "stat-first-date"),
                    _stat_card("Terminal mean", "stat-terminal", "stat-terminal-date"),
                    _stat_card("Forecast-only horizon", "stat-horizon", "stat-horizon-unit"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H3("Full path", className="panel-title")],
                        className="panel-heading",
                    ),
                    dash_table.DataTable(
                        export_format="xlsx",
                        export_headers="display",
                        export_columns="all",
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
                [readable_table("forecast-summary-table", FORECAST_COLUMNS, page_size=7)],
                style={"display": "none"},
            ),
        ],
        className="page-body",
    )




# FORECAST_ENERGY_AGGREGATE_SCENARIO_UI_CLEANUP_V1
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
            _energy_forecast_control_bar("aggregate"),
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
            html.Div([html.Button("", id="agg-headline-contribution-run", n_clicks=0),html.Div(id="agg-headline-contribution-status"),html.Div(id="agg-headline-contribution-conditioned-result"),html.Div(id="agg-headline-contribution-terminal-result"),dcc.Graph(id="agg-headline-contribution-graph")], style={"display": "none"}),
            html.Div([html.Div(id="agg-all-scenarios-summary"),dcc.Graph(id="agg-conditional-scenarios-impact")], style={"display": "none"}),
            html.Div(
                [
                    html.Div([html.Div(id="agg-scenario-banner"),html.Div(id="agg-scen-baseline"),html.Div(id="agg-scen-date"),html.Div(id="agg-scen-scenario"),html.Div(id="agg-scen-date-2"),html.Div(id="agg-scen-impact"),html.Div(id="agg-scen-interval"),html.Div(id="agg-scen-draws"),html.Div(id="agg-scen-draws-note"),dcc.Graph(id="agg-scenario-graph"),dcc.Graph(id="agg-scenario-impact"),dcc.Graph(id="agg-scenario-contrib-impact")], style={"display": "none"})
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
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
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
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
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

# ENERGY_SCENARIO_UI_CLARITY_AND_JOINT_WORKSPACE_V1
# REMOVE_REDUNDANT_JOINT_ENERGY_UI_E_V1_1
# Joint Energy is resolved internally by the unified Energy -> Headline button when required.
# SCENARIOS_PRESENTATION_CLEANUP_V1_1
def scenario_page() -> html.Div:
    """Conditional observable paths plus ex-post VAT/excise scenarios."""
    conditional_controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Model", className="control-label"),
                    dcc.Dropdown(
                        id="scenario-model-select",
                        options=[
                            {"label": model_spec(model_id).label, "value": model_id}
                            for model_id in ENERGY_SUITE_MODEL_IDS
                        ],
                        value="gas",
                        clearable=False,
                        persistence=True,
                        persistence_type="session",
                        className="compact-dropdown wide-control",
                    ),
                ],
                className="control-block wide-control",
            ),
            html.Div(
                [
                    html.Label("Energy component", className="control-label"),
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
                    html.Label("Scenario driver", className="control-label"),
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
                    html.Label("Assumption", className="control-label"),
                    dcc.Dropdown(
                        id="conditional-path-mode",
                        options=[
                            {"label": "% vs last observed", "value": "percent"}
                        ],
                        value="percent",
                        clearable=False,
                        searchable=False,
                        disabled=True,
                        className="compact-dropdown",
                    ),
                ],
                className="control-block",
            ),
            html.Div(
                [
                    html.Label("Condition start", className="selector-label"),
                    dcc.Dropdown(
                        id="conditional-window-start",
                        options=[],
                        value=None,
                        clearable=False,
                    ),
                ],
                className="selector-block",
            ),
            html.Div(
                [
                    html.Label("Condition end", className="selector-label"),
                    dcc.Dropdown(
                        id="conditional-window-end",
                        options=[],
                        value=None,
                        clearable=False,
                    ),
                ],
                className="selector-block",
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
            html.Div(
                [
                    html.Div("CONDITIONED VALUES", className="eyebrow"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Set all (%)",
                                        id="conditional-path-value-label",
                                        className="control-label",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="conditional-path-value",
                                                type="number",
                                                value=10.0,
                                                step=0.1,
                                                debounce=0.4,
                                                className="compact-dropdown",
                                                style={"width": "120px"},
                                            ),
                                            html.Button(
                                                "Apply",
                                                id="conditional-path-set-all",
                                                n_clicks=0,
                                                className="refresh-button",
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "gap": "8px",
                                            "alignItems": "end",
                                            "flexWrap": "wrap",
                                        },
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Linear path (%)",
                                        className="control-label",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="conditional-linear-start",
                                                type="number",
                                                value=0.0,
                                                step=0.1,
                                                debounce=0.4,
                                                placeholder="Start",
                                                className="compact-dropdown",
                                                style={"width": "110px"},
                                            ),
                                            dcc.Input(
                                                id="conditional-linear-end",
                                                type="number",
                                                value=10.0,
                                                step=0.1,
                                                debounce=0.4,
                                                placeholder="End",
                                                className="compact-dropdown",
                                                style={"width": "110px"},
                                            ),
                                            html.Button(
                                                "Fill",
                                                id="conditional-path-linear-fill",
                                                n_clicks=0,
                                                className="refresh-button",
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "gap": "8px",
                                            "alignItems": "end",
                                            "flexWrap": "wrap",
                                        },
                                    ),
                                ],
                                className="control-block",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "18px",
                            "alignItems": "end",
                            "flexWrap": "wrap",
                            "marginTop": "8px",
                        },
                    ),
                    html.Div(
                        id="conditional-path-editor",
                        style={"marginTop": "12px"},
                    ),
                ],
                style={
                    "gridColumn": "1 / -1",
                    "borderTop": "1px solid #e2e8f0",
                    "paddingTop": "12px",
                },
            ),
        ],
        className="chart-controls",
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, minmax(190px, 1fr))",
            "gap": "14px",
            "alignItems": "start",
            "marginTop": "10px",
            "marginBottom": "10px",
            "overflowX": "auto",
        },
    )
    conditional_advanced = html.Details([html.Summary('Advanced / source run', style={'fontWeight': '650', 'cursor': 'pointer', 'color': '#475569'}), html.Div([html.Label('HICP Energy aggregate run', className='control-label'), dcc.Dropdown(id='conditional-agg-select', options=[], value=None, clearable=False, className='compact-dropdown')], style={'maxWidth': '520px', 'paddingTop': '10px'})], style={'marginTop': '10px', 'marginBottom': '10px'})
    tax_controls = html.Div([html.Div([html.Label('Component', className='control-label'), dcc.Dropdown(id='scenario-component-select', options=[], value=None, clearable=False, className='compact-dropdown')], className='control-block'), html.Div([html.Label('Scenario start', className='control-label'), dcc.DatePickerSingle(id='scenario-start-date', display_format='YYYY-MM-DD', first_day_of_week=1, clearable=False)], className='control-block'), html.Div([html.Label('VAT change (pp; − = cut)', className='control-label'), dcc.Input(id='scenario-vat-delta', type='number', value=0.0, step="any", debounce=0.4, className='compact-dropdown')], className='control-block'), html.Div([html.Label(id='scenario-excise-label', children='Excise change (− = cut)', className='control-label'), dcc.Input(id='scenario-excise-delta', type='number', value=0.0, step="any", debounce=0.4, className='compact-dropdown')], className='control-block'), html.Div([html.Label('Fan', className='control-label'), dcc.RadioItems(id='scenario-fan', options=[{'label': '68%', 'value': '68'}, {'label': '90%', 'value': '90'}, {'label': 'Both', 'value': 'both'}], value='68', inline=True, className='fan-radio')], className='control-block'), html.Button('Add / update', id='scenario-apply', n_clicks=0, className='refresh-button'), html.Button('Remove', id='scenario-remove', n_clicks=0, className='refresh-button'), html.Button('Reset all', id='scenario-reset-all', n_clicks=0, className='refresh-button')], className='chart-controls')
    return html.Div([html.Div([html.H2('Scenarios', className='page-title')], className='page-heading-row'), html.Div([html.Div([html.H3('Conditional commodity / observable path', className='panel-title'), html.P('Each active cell is a hard % condition versus the same latest observed driver value; periods outside the window remain unconstrained but jointly smoothed.', className='panel-subtitle')]), conditional_controls, html.Div(id='conditional-question', style={'display': 'none'}), conditional_advanced, html.Div(id='conditional-support-note', className='selection-banner'), html.Div(id='conditional-banner'), html.Div(id='conditional-run-diagnostic', className='selection-banner', style={'marginTop': '8px', 'fontFamily': 'monospace'}), html.Div([html.Button('Add / update conditional', id='conditional-run', n_clicks=0, className='estimation-run-button'), html.Button('Remove selected', id='conditional-remove', n_clicks=0, className='refresh-button'), html.Button('Reset all conditionals', id='conditional-reset-all', n_clicks=0, className='refresh-button'), html.Button('Cancel', id='conditional-cancel', n_clicks=0, disabled=True, className='estimation-cancel-button'), dcc.Link('Open Headline scenarios →', href='/headline/scenarios', className='refresh-button', style={'textDecoration': 'none', 'display': 'inline-flex', 'alignItems': 'center'})], className='estimation-actions'), html.Div([html.Progress(id='conditional-progress', value=0, max=100, className='estimation-progress'), html.Div([html.Div('Idle', id='conditional-phase', className='estimation-phase'), html.Div('Choose a saved aggregate, component and observable path, then compute.', id='conditional-progress-detail', className='estimation-progress-detail')], className='estimation-progress-text')], className='estimation-progress-wrap')], className='panel'), html.Div([html.Div([html.Div('Active conditional scenario set', className='eyebrow'), html.Div(id='conditional-set-summary')], className='panel', style={'padding': '16px 18px', 'marginBottom': '16px'})]), html.Div([_stat_card('Imposed observable path', 'conditional-stat-level', 'conditional-stat-level-note'), _stat_card('Component terminal effect', 'conditional-stat-component', 'conditional-stat-component-note'), _stat_card('HICP Energy terminal effect', 'conditional-stat-aggregate', 'conditional-stat-aggregate-note'), _stat_card('Paired aggregate draws', 'conditional-stat-draws', 'conditional-stat-draws-note')], className='stats-grid', id='energy-conditional-results-kpis', style={'display': 'none'}), html.Div([html.Div([html.Div('Key scenario results', className='eyebrow'), html.H3('Conditional effect by horizon', className='panel-title'), html.P('Component and HICP Energy effects from the same paired conditional draws used by the charts.', className='panel-subtitle')], className='panel-heading'), readable_table('conditional-effect-table', DUAL_IMPACT_COLUMNS, page_size=7)], className='panel table-panel', id='energy-conditional-results-table', style={'display': 'none'}), html.Div([dcc.Loading(dcc.Graph(id='conditional-path-graph', config=_GRAPH_CONFIG), type='circle')], className='panel chart-panel', id='energy-conditional-results-path', style={'display': 'none'}), html.Div([html.Div([dcc.Loading(dcc.Graph(id='conditional-target-graph', config=_GRAPH_CONFIG), type='circle')], className='panel chart-panel'), html.Div([dcc.Loading(dcc.Graph(id='conditional-component-graph', config=_GRAPH_CONFIG), type='circle')], className='panel chart-panel')], style={'display': 'none', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px'}, id='energy-conditional-results-component-grid'), html.Div([html.Div([dcc.Loading(dcc.Graph(id='conditional-aggregate-graph', config=_GRAPH_CONFIG), type='circle')], className='panel chart-panel'), html.Div([dcc.Loading(dcc.Graph(id='conditional-aggregate-impact-graph', config=_GRAPH_CONFIG), type='circle')], className='panel chart-panel')], style={'display': 'none', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px', 'marginBottom': '24px'}, id='energy-conditional-results-aggregate-grid'), html.Div([html.Div([html.H3('VAT / excise scenarios', className='panel-title'), html.P('Build several ex-post VAT/excise shocks on different energy components. Saved BVAR price draws remain fixed; the active tax set is propagated draw-by-draw to HICP Energy.', className='panel-subtitle')]), tax_controls], className='panel'), html.Div(id='scenario-support-note', className='selection-banner'), html.Div(id='scenario-tax-preview', className='selection-banner'), html.Div(id='scenario-banner'), html.Div([html.Div([html.Div('Active tax scenario set', className='eyebrow'), html.Div(id='scenario-set-summary')], className='panel', style={'padding': '16px 18px', 'marginBottom': '16px'})]), html.Div([_stat_card('Baseline terminal YoY', 'scenario-baseline-terminal', 'scenario-terminal-date'), _stat_card('Scenario terminal YoY', 'scenario-scenario-terminal', 'scenario-terminal-date-2'), _stat_card('Terminal tax impact', 'scenario-impact-terminal', 'scenario-impact-interval'), _stat_card('Paired posterior draws', 'scenario-draws', 'scenario-draws-note')], className='stats-grid', id='energy-tax-results-kpis', style={'display': 'none'}), html.Div([html.Div([html.Div('Key scenario results', className='eyebrow'), html.H3('Tax scenario effect by horizon', className='panel-title'), html.P('Legacy tax artefacts persist quantiles rather than a posterior mean; central values here are therefore the saved posterior medians.', className='panel-subtitle')], className='panel-heading'), readable_table('scenario-effect-table', PAIRED_EFFECT_COLUMNS, page_size=7)], className='panel table-panel', id='energy-tax-results-table', style={'display': 'none'}), html.Div([dcc.Loading(dcc.Graph(id='scenario-main-graph', config=_GRAPH_CONFIG), type='circle')], className='panel chart-panel', id='energy-tax-results-main', style={'display': 'none'}), html.Div([html.Div([dcc.Loading(dcc.Graph(id='scenario-level-impact', config=_GRAPH_CONFIG), type='circle')], className='panel chart-panel'), html.Div([dcc.Loading(dcc.Graph(id='scenario-yoy-impact', config=_GRAPH_CONFIG), type='circle')], className='panel chart-panel')], style={'display': 'none', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px'}, id='energy-tax-results-impact-grid'), html.Div([dcc.Loading(dcc.Graph(id='scenario-tax-graph', config=_GRAPH_CONFIG), type='circle')], className='panel chart-panel', id='energy-tax-results-tax', style={'display': 'none'}), html.Div([html.Div([html.H3('Selected tax scenario — HICP Energy marginal effect', className='panel-title'), html.P('The selected VAT/excise scenario is propagated alone through the same saved HICP Energy aggregate. The underlying pre-tax BVAR target is unchanged by construction.', className='panel-subtitle')], className='panel-heading'), html.Div('BVAR target effect = 0 by construction for a pure tax scenario.', className='selection-banner'), dcc.Loading(dcc.Graph(id='scenario-tax-energy-graph', config=_GRAPH_CONFIG), type='circle')], className='panel chart-panel', id='energy-tax-results-energy', style={'display': 'none'}), html.Div([html.Div([html.Div([html.Div('INTER-DOMAIN PROPAGATION', className='eyebrow'), html.H3('Propagation to HICP Energy and Headline', className='panel-title'), html.P('One chart for both domains. Impact shows scenario minus baseline in percentage points; Paths shows observed context plus the baseline and scenario forecasts. Headline remains an explicit saved-posterior conditional calculation.', className='panel-subtitle')]), html.Div([html.Div([html.Label('View', className='control-label'), dcc.RadioItems(id='scenario-propagation-view-mode', options=[{'label': 'Impact', 'value': 'impact'}, {'label': 'Paths', 'value': 'paths'}], value='impact', inline=True, className='fan-radio')], className='control-block'), html.Div([html.Label('Display horizon', className='control-label'), dcc.Dropdown(id='scenario-propagation-display-horizon', options=[{'label': '3 months', 'value': 3}, {'label': '6 months', 'value': 6}, {'label': '9 months', 'value': 9}, {'label': '12 months', 'value': 12}], value=6, clearable=False, searchable=False, className='compact-dropdown', style={'minWidth': '145px'})], className='control-block'), html.Button('Calculate Headline impact (~23–40 s + Joint if needed)', id='scenario-propagation-headline-run', n_clicks=0, className='estimation-run-button')], style={'display': 'flex', 'gap': '14px', 'alignItems': 'end', 'flexWrap': 'wrap'})], className='panel-heading', style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '18px', 'alignItems': 'flex-start', 'flexWrap': 'wrap'}), html.Div('Display only · M+1 is the first monthly forecast/nowcast point; changing View or Display horizon never recomputes a model or scenario.', className='selection-banner'), html.Div('Ready · build an HICP Energy scenario; calculate Headline explicitly when required.', id='scenario-propagation-headline-status', className='propagation-headline-status'), html.Div(id='scenario-propagation-joint-vs-standalone', className='selection-banner', style={'marginTop': '8px', 'marginBottom': '8px'}), html.Div(id='scenario-propagation-energy-result', style={'display': 'none'}), html.Div(id='scenario-propagation-headline-result', style={'display': 'none'}), dcc.Loading(dcc.Graph(id='scenario-propagation-headline-graph', figure=empty_scenario_figure('Build an HICP Energy scenario; calculate Headline explicitly when required.'), config=_GRAPH_CONFIG), type='circle')], className='panel', style={'marginTop': '24px', 'marginBottom': '24px', 'display': 'none'}, id='energy-propagation-results'), html.Div(), html.Div(), html.Div([html.Div(), html.Div()], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px', 'marginBottom': '24px'})], className='page-body')



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
                                    html.Label("Model", className="control-label"),
                                    dcc.Dropdown(
                                        id="structural-model-select",
                                        options=[
                                            {"label": model_spec(model_id).label, "value": model_id}
                                            for model_id in ENERGY_SUITE_MODEL_IDS
                                        ],
                                        value="gas",
                                        clearable=False,
                                        persistence=True,
                                        persistence_type="session",
                                        className="compact-dropdown wide-control",
                                    ),
                                ],
                                className="control-block wide-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Computation horizon", className="control-label"),
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
                                    html.Label("Display horizon", className="control-label"),
                                    dcc.Input(
                                        id="structural-display-horizon",
                                        type="number",
                                        min=1,
                                        max=STRUCTURAL_MAX_HORIZON,
                                        step=1,
                                        value=6,
                                        className="est-profile-name-input",
                                    ),
                                ],
                                className="control-block",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Structural IRF / FEVD ready",
                                        id="structural-analysis-status",
                                        style={"fontWeight": 600, "fontSize": "12px"},
                                    ),
                                    html.Div(id='structural-display-status', style={'fontSize': '11px', 'color': '#64748B', 'marginTop': '4px', 'display': 'none'}),
                                    html.Div(
                                        id="structural-freshness-status",
                                        style={"fontSize": "11px", "marginTop": "6px"},
                                    ),
                                ],
                                className="control-block",
                                style={"minWidth": "280px"},
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
                                                step="any",
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
                            "gridTemplateColumns": "minmax(180px,.8fr) minmax(110px,.45fr) minmax(130px,.5fr) minmax(250px,1fr) minmax(180px,.75fr)",
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


def _vintage_display_text(vintage: str | None) -> str:
    value = str(vintage or "").strip()
    if re.fullmatch(r"\d{8}", value):
        return f"{value} · {value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def _production_vintage_label(item: Mapping[str, Any]) -> str:
    if item.get("linked_ready"):
        suffix = "processed data ready"
    elif item.get("energy_ready"):
        suffix = "Energy ready"
    elif item.get("headline_ready"):
        suffix = "Headline ready"
    else:
        suffix = "incomplete"
    return f"{_vintage_display_text(item['vintage'])} · {suffix}"


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
        "complete_run_ids": [str(row["run_id"]) for row in complete],
        "promoted_complete_run_ids": [
            str(row["run_id"]) for row in promoted_complete
        ],
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
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
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
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
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
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
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
                                        "The locked baseline follows the ECB model architecture and project-standard calibrations where the paper does not report numerical hyperparameters; it is not presented as empirically optimal. Choose Custom or a saved profile to change priors and MCMC settings; every effective configuration enters the deterministic run identity.",
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
                        [
                            html.Label("Missing values", className="control-label"),
                            dcc.Dropdown(
                                id="est-missing-method",
                                options=[
                                    {"label": "Model baseline", "value": "baseline"},
                                    {"label": "Linear interpolation · fast approximation", "value": "linear"},
                                    {"label": "Durbin–Koopman · exact augmentation", "value": "dk"},
                                ],
                                value="baseline",
                                clearable=False,
                                className="compact-dropdown",
                            ),
                            html.Div(
                                "Independent of the prior profile. Model baseline keeps each model's native treatment; Linear or Durbin–Koopman applies to every model in suite runs. Per-model scope uses the table below.",
                                className="est-editor-note",
                            ),
                        ],
                        className="control-block",
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
                                            html.Label(["A variance", html.Span("Standardized; scaled by σᵢ/σⱼ")], className="est-field-label"),
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
                                            html.Label(["Outlier interval", html.Span("Mean calendar years")], className="est-field-label"),
                                            dcc.Input(id="est-outlier-years", type="number", value=4.0, step=0.25, min=0.05, className="est-number-input"),
                                            html.Div(id="est-outlier-effective", className="est-editor-note"),
                                            html.Label(["Prior confidence", html.Span("Equivalent calendar years of prior information")], className="est-field-label"),
                                            dcc.Input(id="est-outlier-prior-years", type="number", value=10.0, step=0.5, min=0.1, className="est-number-input"),
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
                                        "Editable when scope is Per-model overrides. These rows control λ1–λ4, p, missing-value treatment and the core MCMC settings; SV/outlier/numerical settings remain shared above.",
                                        className="est-editor-note",
                                    ),
                                ],
                                className="est-per-model-heading",
                            ),
                            dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
                                id="est-per-model-table",
                                data=_baseline_per_model_rows(),
                                columns=[
                                    {"name": "Model", "id": "model", "editable": False},
                                    {"name": "p", "id": "p", "type": "numeric"},
                                    {"name": "Missing", "id": "missing_data_method", "presentation": "dropdown"},
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
                                dropdown={
                                    "missing_data_method": {
                                        "options": [
                                            {"label": "Linear", "value": "linear"},
                                            {"label": "DK", "value": "dk"},
                                        ]
                                    }
                                },
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
                        id="estimation-run-vintage-banner",
                        className="selection-banner",
                        style={
                            "marginBottom": "12px",
                            "borderWidth": "2px",
                            "fontSize": "13px",
                        },
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
                                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
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
                            dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
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


def _domain_section_href(section: str, domain: str) -> str:
    section = str(section).strip().lower()
    domain = str(domain).strip().lower()
    if section not in {"forecast", "scenarios", "structural", "estimation"}:
        raise ValueError(f"Unknown dashboard section {section!r}.")
    if domain == "energy":
        return f"/{section}"
    if domain in {"headline", "core"}:
        return f"/{section}/{domain}"
    raise ValueError(f"Unknown dashboard domain {domain!r}.")


def _domain_switch(section: str) -> html.Div:
    """Id-less, URL-driven domain segmented control shared by all domain pages."""
    core_unavailable = section in {"structural", "estimation"}
    core_title = (
        f"{section.title()} is not yet available for Core. "
        "Use Core Forecast or Scenarios instead."
        if core_unavailable
        else None
    )
    return html.Div(
        [
            dcc.Link(
                "Energy",
                href=_domain_section_href(section, "energy"),
                className="domain-pill",
            ),
            dcc.Link(
                "Headline HICP",
                href=_domain_section_href(section, "headline"),
                className="domain-pill",
            ),
            dcc.Link(
                "Core",
                href=_domain_section_href(section, "core"),
                className=(
                    "domain-pill domain-pill-disabled"
                    if core_unavailable
                    else "domain-pill"
                ),
                title=core_title,
                style=(
                    {"opacity": "0.45", "cursor": "help"}
                    if core_unavailable
                    else None
                ),
            ),
        ],
        className="domain-switch page-domain-switch",
        style={"width": "420px", "maxWidth": "100%", "margin": "8px 0 14px"},
    )


def _forecast_view_switch(active_view: str) -> html.Div:
    """Energy Forecast view toggle; URL-driven, id-less, callback-free."""
    base_style = {
        "display": "block",
        "padding": "7px 10px",
        "border": "1px solid #d1d5db",
        "borderRadius": "8px",
        "textAlign": "center",
        "textDecoration": "none",
        "fontSize": "13px",
    }

    def _style(active: bool) -> dict:
        style = dict(base_style)
        style.update({
            "fontWeight": 700 if active else 500,
            "background": "#f3f4f6" if active else "transparent",
        })
        return style

    return html.Div(
        [
            dcc.Link(
                "Components",
                href="/forecast",
                className="forecast-view-pill",
                style=_style(active_view == "components"),
            ),
            dcc.Link(
                "HICP Energy Aggregate",
                href="/forecast/aggregate",
                className="forecast-view-pill",
                style=_style(active_view == "aggregate"),
            ),
        ],
        className="forecast-view-switch",
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "4px",
            "width": "420px",
            "maxWidth": "100%",
            "margin": "0 0 16px",
        },
    )


def _insert_page_navigation(
    component,
    additions: list,
    *,
    audit_branch: list[int] | None = None,
) -> bool:
    """Insert after the principal page heading using three strictly bounded shapes."""
    children = getattr(component, "children", None)
    if isinstance(children, tuple):
        children = list(children)
        component.children = children
    if not isinstance(children, list):
        return False

    # Branch 1: a direct page-heading-row child owns the title/subtitle.
    for index, child in enumerate(children):
        class_name = str(getattr(child, "className", "") or "")
        if "page-heading-row" in class_name.split():
            children[index + 1:index + 1] = additions
            if audit_branch is not None:
                audit_branch.append(1)
            return True

    # Branch 2: compact pages expose H2.page-title + optional P.page-subtitle directly.
    for index, child in enumerate(children):
        class_name = str(getattr(child, "className", "") or "")
        if type(child).__name__ == "H2" and "page-title" in class_name.split():
            insert_at = index + 1
            if insert_at < len(children):
                next_child = children[insert_at]
                next_class = str(getattr(next_child, "className", "") or "")
                if (
                    type(next_child).__name__ == "P"
                    and "page-subtitle" in next_class.split()
                ):
                    insert_at += 1
            children[insert_at:insert_at] = additions
            if audit_branch is not None:
                audit_branch.append(2)
            return True

    # Branch 3: exactly one bounded level for an anonymous Div heading wrapper.
    # This covers core_scenarios_page without reopening recursive H2 discovery.
    for index, child in enumerate(children):
        if type(child).__name__ != "Div":
            continue
        class_name = str(getattr(child, "className", "") or "").strip()
        if class_name:
            continue
        inner = getattr(child, "children", None)
        if isinstance(inner, tuple):
            inner = list(inner)
            child.children = inner
        if not isinstance(inner, list) or not inner:
            continue
        first = inner[0]
        first_class = str(getattr(first, "className", "") or "")
        if type(first).__name__ == "H2" and "page-title" in first_class.split():
            children[index + 1:index + 1] = additions
            if audit_branch is not None:
                audit_branch.append(3)
            return True

    # No recursion beyond one anonymous heading wrapper: nested H2s may be sub-cards.
    return False


def _with_page_navigation(
    page,
    *,
    section: str,
    domain: str,
    forecast_view: str | None = None,
):
    """Keep the three-domain switch as a direct sticky child of every domain page."""
    children = getattr(page, "children", None)
    if not isinstance(children, list):
        raise RuntimeError(
            f"Domain page has no direct children list for section={section!r}, "
            f"domain={domain!r}."
        )

    # Direct child of page-body: CSS sticky is then bounded by the whole page,
    # not by a short heading row, so Energy / Headline HICP / Core remain
    # accessible while the user scrolls.
    children.insert(0, _domain_switch(section))

    # Keep the existing Energy Forecast sub-view switch near the page heading;
    # only the three-domain switch becomes page-sticky.
    if section == "forecast" and domain == "energy":
        if not _insert_page_navigation(
            page,
            [_forecast_view_switch(forecast_view or "components")],
        ):
            raise RuntimeError(
                f"Could not find the Energy Forecast page title for "
                f"forecast_view={forecast_view!r}."
            )
    return page


def _core_unavailable_page(section: str) -> html.Div:
    title = section.title()
    return html.Div(
        [
            html.H2(title, className="page-title"),
            html.P(
                f"{title} is not yet available for Core.",
                className="page-subtitle",
            ),
            html.Div(
                [
                    html.Div("Core", className="eyebrow"),
                    html.H3(
                        f"{title} is not yet available for Core",
                        className="placeholder-title",
                    ),
                    html.P(
                        "No redirect is performed. Core Forecast and Scenarios remain "
                        "available from the section-first shell.",
                        className="placeholder-text",
                    ),
                    dcc.Link(
                        "Open Core → Forecast",
                        href="/forecast/core",
                        className="refresh-button",
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
                ],
                id="global-nav",
                className="nav-stack",
            ),
            html.Nav(
                [
                    _nav_link("Forecast", "/forecast", "↗"),
                    _nav_link("Scenarios", "/scenarios", "△"),
                    _nav_link("Structural", "/structural", "ψ"),
                    _nav_link("Estimation", "/estimation", "⚙"),
                ],
                id="energy-nav",
                className="nav-stack nav-stack-energy",
            ),
            html.Nav(
                [
                    _nav_link("Forecast", "/forecast/headline", "↗"),
                    _nav_link("Scenarios", "/scenarios/headline", "△"),
                    _nav_link("Structural", "/structural/headline", "ψ"),
                    _nav_link("Estimation", "/estimation/headline", "⚙"),
                ],
                id="headline-nav",
                className="nav-stack nav-stack-headline",
                style={"display": "none"},
            ),
            html.Nav(
                [
                    _nav_link("Forecast", "/forecast/core", "↗"),
                    _nav_link("Scenarios", "/scenarios/core", "△"),
                    _nav_link("Structural", "/structural/core", "ψ"),
                    _nav_link("Estimation", "/estimation/core", "⚙"),
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
                    _selector("Production vintage", "vintage-select", "Select production vintage"),
                    html.Div([_selector("Model", "model-select", "Select model")], style={"display": "none"}),
                    html.Div([_selector("Forecast", "forecast-select", "Forecast contract")], style={"display": "none"}),
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
register_xlsx_export(server)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="registry-store", data=_scan_registry()),
        dcc.Store(id="ctx-store", storage_type="session"),
        dcc.Store(id="production-vintage-store", storage_type="session"),
        html.Div(id="xlsx-export-vintage", style={"display": "none"}),
        dcc.Store(id="data-store", storage_type="memory"),
        dcc.Store(id="agg-store", storage_type="memory"),
        dcc.Store(id="agg-headline-contribution-store", storage_type="memory"),
        dcc.Store(id="scenario-store", storage_type="memory"),
        dcc.Store(id="conditional-store", storage_type="memory"),
        dcc.Store(id="joint-energy-scenario-store", storage_type="memory"),
        dcc.Store(id="scenario-tax-selected-agg-store", storage_type="memory"),
        dcc.Store(id="agg-scenario-store", storage_type="memory"),
        dcc.Store(id="scenario-propagation-headline-store", storage_type="memory"),
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
                html.Div(
                    id="global-production-results-warning",
                    className="selection-banner global-production-results-warning",
                    style={"display": "none"},
                ),
                html.Div(id="selection-banner", className="selection-banner"),
                html.Div(
                    [
                        html.Div(overview_page(), id="page-overview", style={"display": "none"}),
                        html.Div(data_page(), id="page-data", style={"display": "none"}),
                        html.Div(_with_page_navigation(forecast_page(), section="forecast", domain="energy", forecast_view="components"), id="page-forecast"),
                        html.Div(
                            _with_page_navigation(aggregate_page(), section="forecast", domain="energy", forecast_view="aggregate"),
                            id="page-aggregate",
                            style={"display": "none"},
                        ),
                        html.Div(
                            _with_page_navigation(scenario_page(), section="scenarios", domain="energy"),
                            id="page-scenarios",
                            style={"display": "none"},
                        ),
                        html.Div(
                            _with_page_navigation(structural_page(), section="structural", domain="energy"),
                            id="page-structural",
                            style={"display": "none"},
                        ),
                        html.Div(
                            _with_page_navigation(estimation_page(), section="estimation", domain="energy"),
                            id="page-estimation",
                            style={"display": "none"},
                        ),
                    html.Div(
                        _with_page_navigation(headline_forecast_v2_page(), section="forecast", domain="headline"),
                        id="page-headline-forecast",
                        style={"display": "none"},
                    ),
                    html.Div(
                        _with_page_navigation(headline_scenarios_page(), section="scenarios", domain="headline"),
                        id="page-headline-scenarios",
                        style={"display": "none"},
                    ),
                    html.Div(
                        _with_page_navigation(core_forecast_page(), section="forecast", domain="core"),
                        id="page-core-forecast",
                        style={"display": "none"},
                    ),
                    html.Div(
                        _with_page_navigation(core_scenarios_page(), section="scenarios", domain="core"),
                        id="page-core-scenarios",
                        style={"display": "none"},
                    ),
                        html.Div(
                            _with_page_navigation(headline_structural_page(), section="structural", domain="headline"),
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
                            _with_page_navigation(headline_estimation_v2_page(), section="estimation", domain="headline"),
                        ],
                        id="page-headline-diagnostics",
                        style={"display": "none"},
                    )
,
                        html.Div(
                            _with_page_navigation(
                                _core_unavailable_page("structural"),
                                section="structural",
                                domain="core",
                            ),
                            id="page-core-structural",
                            style={"display": "none"},
                        ),
                        html.Div(
                            _with_page_navigation(
                                _core_unavailable_page("estimation"),
                                section="estimation",
                                domain="core",
                            ),
                            id="page-core-estimation",
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


# DASHBOARD_REGISTRY_VINTAGE_SYNC_V1
# Data's production vintage is authoritative across all dashboard domains.
# Successful Headline estimation also refreshes the frozen registry snapshot.

# ---------------------------------------------------------------------------
# Registry and global context callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("registry-store", "data"),
    Input("refresh-registry", "n_clicks"),
    Input("dataset-build-result-store", "data"),
    Input("estimation-result-store", "data"),
    Input("h5-estimate-result", "data"),
    prevent_initial_call=True,
)
def refresh_registry(
    _manual_clicks: int | None,
    dataset_result: dict | None,
    estimation_result: dict | None,
    headline_estimation_result: dict | None,
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
    elif trigger == "h5-estimate-result":
        if not (headline_estimation_result or {}).get("ok"):
            raise PreventUpdate
    return _scan_registry()


# GLOBAL_PRODUCTION_RESULTS_WARNING_V1

def _production_results_warning_state(
    production_store: dict | None,
) -> dict:
    """Summarise saved-result readiness from the frozen registry snapshot only."""
    vintage = str((production_store or {}).get("vintage") or "").strip()
    if not vintage:
        return {
            "show": False,
            "vintage": None,
            "energy_ready_count": 0,
            "energy_expected": len(ENERGY_SUITE_MODEL_IDS),
            "aggregate_ready": False,
            "headline_ready": False,
            "completed_run_count": 0,
            "usable_forecast_count": 0,
        }

    runs = _registry_table("runs")
    forecasts = _registry_table("forecasts")

    if runs.empty or "vintage" not in runs.columns:
        complete_runs = runs.iloc[0:0].copy()
    else:
        complete_runs = runs.loc[
            runs["vintage"].astype(str).eq(vintage)
        ].copy()
        if "status" in complete_runs.columns:
            complete_runs = complete_runs.loc[
                complete_runs["status"].astype(str).eq("complete")
            ].copy()
        else:
            complete_runs = complete_runs.iloc[0:0].copy()

    complete_run_ids = (
        set(complete_runs["run_id"].astype(str))
        if not complete_runs.empty and "run_id" in complete_runs.columns
        else set()
    )

    if forecasts.empty or "vintage" not in forecasts.columns:
        usable_forecasts = forecasts.iloc[0:0].copy()
    else:
        usable_forecasts = forecasts.loc[
            forecasts["vintage"].astype(str).eq(vintage)
        ].copy()
        if "run_id" in usable_forecasts.columns:
            usable_forecasts = usable_forecasts.loc[
                usable_forecasts["run_id"].astype(str).isin(complete_run_ids)
            ].copy()
        else:
            usable_forecasts = usable_forecasts.iloc[0:0].copy()

    ready_model_ids = (
        set(usable_forecasts["model_id"].astype(str))
        if not usable_forecasts.empty and "model_id" in usable_forecasts.columns
        else set()
    )
    energy_ready_count = sum(
        1
        for model_id in ENERGY_SUITE_MODEL_IDS
        if str(model_id) in ready_model_ids
    )
    headline_ready = str(HEADLINE_MODEL_ID) in ready_model_ids

    aggregates = _registry_table("aggregates")
    aggregate_ready = False
    if not aggregates.empty and "vintage" in aggregates.columns:
        here = aggregates.loc[
            aggregates["vintage"].astype(str).eq(vintage)
        ].copy()
        if not here.empty:
            if "status" in here.columns:
                aggregate_ready = bool(
                    here["status"].astype(str).eq("complete").any()
                )
            else:
                aggregate_ready = True

    all_ready = bool(
        energy_ready_count == len(ENERGY_SUITE_MODEL_IDS)
        and aggregate_ready
        and headline_ready
    )
    return {
        "show": not all_ready,
        "vintage": vintage,
        "energy_ready_count": int(energy_ready_count),
        "energy_expected": int(len(ENERGY_SUITE_MODEL_IDS)),
        "aggregate_ready": bool(aggregate_ready),
        "headline_ready": bool(headline_ready),
        "completed_run_count": int(len(complete_runs)),
        "usable_forecast_count": int(len(usable_forecasts)),
    }


def _production_results_warning_component(
    production_store: dict | None,
):
    state = _production_results_warning_state(production_store)
    if not state["show"] or not state["vintage"]:
        return "", {"display": "none"}

    vintage = str(state["vintage"])
    energy = (
        f"Energy saved results {state['energy_ready_count']}/"
        f"{state['energy_expected']}"
    )
    aggregate = (
        "HICP Energy aggregate ready"
        if state["aggregate_ready"]
        else "HICP Energy aggregate pending"
    )
    headline = (
        "Headline saved result ready"
        if state["headline_ready"]
        else "Headline saved result pending"
    )

    if state["completed_run_count"] == 0:
        lead = (
            f" · vintage {vintage} is processed, but no completed model run exists yet."
        )
    elif state["usable_forecast_count"] == 0:
        lead = (
            f" · vintage {vintage} has completed run metadata, "
            "but no usable saved forecast is available yet."
        )
    else:
        lead = f" · vintage {vintage} has only a partial saved result set."

    children = [
        html.Strong("Production data ready; results pending"),
        html.Span(lead),
        html.Span(f" · {energy} · {aggregate} · {headline}."),
        html.Span(
            " Navigation remains available. Result-dependent panels will populate "
            "after the missing estimations/forecasts complete and the registry "
            "snapshot refreshes."
        ),
    ]
    return children, {}

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
    Output("global-production-results-warning", "children"),
    Output("global-production-results-warning", "style"),
    Input("production-vintage-store", "data"),
    Input("registry-store", "data"),
)
def global_production_results_warning(
    production_store: dict | None,
    _registry: dict | None,
):
    # registry-store is only a re-trigger after explicit/successful refreshes.
    # The lookup itself stays on the already-frozen in-memory snapshot.
    return _production_results_warning_component(production_store)

@callback(
    Output("vintage-select", "options"),
    Output("vintage-select", "value"),
    Output("vintage-select", "disabled"),
    Input("registry-store", "data"),
    Input("url", "pathname"),
    Input("production-vintage-store", "data"),
    State("vintage-select", "value"),
)
def vintage_options(
    _: dict | None,
    pathname: str | None,
    production_store: dict | None,
    current: str | None,
):
    production_vintage = str((production_store or {}).get("vintage") or "")
    if production_vintage:
        # Data owns the production information set. Keep the legacy global
        # dropdown as a read-only compatibility control for downstream callbacks,
        # but never let it drift to a different historical vintage.
        return (
            [{"label": f"{_vintage_display_text(production_vintage)} · production", "value": production_vintage}],
            production_vintage,
            True,
        )

    forecasts = _registry_table("forecasts")
    if forecasts.empty:
        return [], None, True

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
        return [], None, True

    vintages = sorted(
        forecasts["vintage"].astype(str).unique(),
        reverse=True,
    )
    options = [{"label": value, "value": value} for value in vintages]
    return options, current if current in vintages else vintages[0], False


@callback(
    Output("model-select", "options"),
    Output("model-select", "value"),
    Input("vintage-select", "value"),
    Input("registry-store", "data"),
    Input("url", "pathname"),
    State("model-select", "value"),
    State("forecast-model-select", "value"),
    State("scenario-model-select", "value"),
    State("structural-model-select", "value"),
    State("est-model-select", "value"),
)
def model_options(
    vintage: str | None,
    _: dict | None,
    pathname: str | None,
    current: str | None,
    forecast_model: str | None,
    scenario_model: str | None,
    structural_model: str | None,
    estimation_model: str | None,
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

    path = pathname or "/forecast"
    domain = "headline" if path.startswith("/core") else domain_from_path(path)
    models = models_for_domain(
        forecasts["model_id"].astype(str).unique(),
        domain,
    )
    options = [
        {"label": model_label(model_id), "value": model_id}
        for model_id in models
    ]
    values = [item["value"] for item in options]

    local_candidate = None
    if domain == "energy":
        if path in {"/forecast", "/forecast/aggregate", "/aggregate"}:
            local_candidate = forecast_model
        elif path == "/scenarios":
            local_candidate = scenario_model
        elif path == "/structural":
            local_candidate = structural_model
        elif path == "/estimation":
            local_candidate = estimation_model

    if local_candidate in values:
        selected = local_candidate
    elif current in values:
        selected = current
    else:
        selected = values[0] if values else None
    return options, selected


@callback(
    Output("forecast-select", "options"),
    Output("forecast-select", "value"),
    Input("vintage-select", "value"),
    Input("model-select", "value"),
    Input("registry-store", "data"),
    Input("url", "pathname"),
    Input("conditional-agg-select", "value"),
    State("forecast-select", "value"),
)
def forecast_options(
    vintage: str | None,
    model_id: str | None,
    _: dict | None,
    pathname: str | None,
    scenario_aggregate_run_id: str | None,
    current: str | None,
):
    """Resolve the hidden canonical forecast, with Scenario source alignment.

    Outside Energy Scenarios the historical production policy is preserved:
    prefer/lock ``unconditional`` when it exists.  On ``/scenarios`` the
    visible HICP Energy source aggregate is authoritative, so the hidden
    forecast follows that aggregate's exact saved ``forecast_name``.
    """
    _ = current
    if not vintage or not model_id:
        return [], None

    frame = _registry_table("forecasts")
    if not frame.empty:
        frame = frame.loc[
            frame["model_id"].astype(str).eq(str(model_id))
            & frame["vintage"].astype(str).eq(str(vintage))
        ].copy()
    names = sorted(
        frame["forecast_name"].astype(str).unique()
    ) if not frame.empty else []

    if (pathname or "") == "/scenarios" and scenario_aggregate_run_id:
        row = _selected_aggregate_row(vintage, scenario_aggregate_run_id)
        if row is not None:
            try:
                contract = _cached_conditional_aggregate_contract(
                    Path(str(row["directory"]))
                )
                desired = str(contract.get("forecast_name") or "").strip()
            except Exception:
                desired = ""
            if desired and desired in names:
                return (
                    [
                        {
                            "label": desired.replace("_", " ").title(),
                            "value": desired,
                        }
                    ],
                    desired,
                )

    if "unconditional" in names:
        return (
            [{"label": "Unconditional", "value": "unconditional"}],
            "unconditional",
        )
    options = [
        {"label": name.replace("_", " ").title(), "value": name}
        for name in names
    ]
    return options, (names[0] if names else None)


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
        raise PreventUpdate
    if global_model in ENERGY_SUITE_MODEL_IDS:
        return global_model
    return current if current in ENERGY_SUITE_MODEL_IDS else "gas"

# ENERGY_LOCAL_MODEL_SELECTORS_V2
@callback(
    Output("model-select", "value", allow_duplicate=True),
    Output("scenario-model-select", "value", allow_duplicate=True),
    Output("structural-model-select", "value", allow_duplicate=True),
    Output("est-model-select", "value", allow_duplicate=True),
    Input("forecast-model-select", "value"),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def sync_forecast_model(local_model, pathname):
    if (pathname or "") not in {"/forecast", "/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
    if local_model not in ENERGY_SUITE_MODEL_IDS:
        raise PreventUpdate
    return local_model, local_model, local_model, local_model

@callback(
    Output("agg-model-select", "value"),
    Input("url", "pathname"),
    State("forecast-model-select", "value"),
    State("agg-model-select", "value"),
)
def sync_aggregate_forecast_model_on_route(pathname, forecast_model, current):
    """Copy component context into Aggregate when the user opens that view."""
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
    if forecast_model in ENERGY_SUITE_MODEL_IDS:
        return forecast_model
    return current if current in ENERGY_SUITE_MODEL_IDS else "gas"


@callback(
    Output("model-select", "value", allow_duplicate=True),
    Output("forecast-model-select", "value", allow_duplicate=True),
    Output("scenario-model-select", "value", allow_duplicate=True),
    Output("structural-model-select", "value", allow_duplicate=True),
    Output("est-model-select", "value", allow_duplicate=True),
    Input("agg-model-select", "value"),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def sync_aggregate_model(local_model, pathname):
    """Aggregate Model is component context only; it never selects an aggregate."""
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
    if local_model not in ENERGY_SUITE_MODEL_IDS:
        raise PreventUpdate
    return local_model, local_model, local_model, local_model, local_model


@callback(
    Output("model-select", "value", allow_duplicate=True),
    Output("forecast-model-select", "value", allow_duplicate=True),
    Output("structural-model-select", "value", allow_duplicate=True),
    Output("est-model-select", "value", allow_duplicate=True),
    Output("conditional-component-select", "value", allow_duplicate=True),
    Output("scenario-component-select", "value", allow_duplicate=True),
    Input("scenario-model-select", "value"),
    State("url", "pathname"),
    State("conditional-component-select", "options"),
    State("scenario-component-select", "options"),
    prevent_initial_call=True,
)
def sync_scenario_model(
    local_model,
    pathname,
    conditional_options,
    tax_options,
):
    if (pathname or "") != "/scenarios":
        raise PreventUpdate
    if local_model not in ENERGY_SUITE_MODEL_IDS:
        raise PreventUpdate

    def _listed(options):
        return any(
            str(item.get("value")) == str(local_model)
            for item in (options or [])
            if isinstance(item, dict)
        )

    conditional_value = local_model if _listed(conditional_options) else no_update
    tax_value = local_model if _listed(tax_options) else no_update
    return (
        local_model,
        local_model,
        local_model,
        local_model,
        conditional_value,
        tax_value,
    )


@callback(
    Output("model-select", "value", allow_duplicate=True),
    Output("forecast-model-select", "value", allow_duplicate=True),
    Output("scenario-model-select", "value", allow_duplicate=True),
    Output("est-model-select", "value", allow_duplicate=True),
    Input("structural-model-select", "value"),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def sync_structural_model(local_model, pathname):
    if (pathname or "") != "/structural":
        raise PreventUpdate
    if local_model not in ENERGY_SUITE_MODEL_IDS:
        raise PreventUpdate
    return local_model, local_model, local_model, local_model


@callback(
    Output("model-select", "value", allow_duplicate=True),
    Output("forecast-model-select", "value", allow_duplicate=True),
    Output("scenario-model-select", "value", allow_duplicate=True),
    Output("structural-model-select", "value", allow_duplicate=True),
    Input("est-model-select", "value"),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def push_estimation_model(local_model, pathname):
    if (pathname or "") != "/estimation":
        raise PreventUpdate
    if local_model not in ENERGY_SUITE_MODEL_IDS:
        raise PreventUpdate
    return local_model, local_model, local_model, local_model


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
    Input("registry-store", "data"),
)
def render_production_vintage_diagnostics(vintage, _build_result, _registry):
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
    Input("url", "pathname"),
)
def render_production_model_readiness(
    vintage,
    config_store,
    selected_model_id,
    _registry,
    _estimation_result, pathname,
):
    if (pathname or "") != "/data":
        raise PreventUpdate
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
    label = _vintage_display_text(vintage) + suffix
    return label, label




# DASHBOARD_UI_CLARITY_CONTRACT_V2
@callback(
    Output("estimation-run-vintage-banner", "children"),
    Input("production-vintage-store", "data"),
)
def render_estimation_run_vintage_banner(store):
    payload = dict(store or {})
    vintage = str(payload.get("vintage") or "")
    if not vintage:
        return html.Div(
            [
                html.Strong("RUN TARGET · no production vintage selected"),
                html.Span(
                    " · choose the production vintage in Data before starting Energy estimation."
                ),
            ],
            style={"color": "#991b1b"},
        )

    energy_ready = bool(payload.get("energy_ready"))
    return html.Div(
        [
            html.Strong("RUN TARGET"),
            html.Span(
                f" · VINTAGE {vintage}",
                style={
                    "fontWeight": "800",
                    "fontSize": "15px",
                    "color": "#1d4ed8",
                },
            ),
            html.Span(
                " · Every Energy action below uses production vintage "
                f"{vintage}."
            ),
            html.Span(
                " · Energy inputs READY"
                if energy_ready
                else " · Energy inputs INCOMPLETE",
                style={
                    "fontWeight": "700",
                    "color": "#166534" if energy_ready else "#991b1b",
                },
            ),
        ]
    )


# DASHBOARD_XLSX_EXPORT_CONTRACT_V1
@callback(
    Output("xlsx-export-vintage", "children"),
    Input("production-vintage-store", "data"),
)
def render_xlsx_export_vintage(store):
    return str((store or {}).get("vintage") or "")



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
        banner = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            "✓ BUILD COMPLETE",
                            style={
                                "fontWeight": "800",
                                "fontSize": "16px",
                                "letterSpacing": "0.03em",
                                "color": "#166534",
                            },
                        ),
                        html.Div(
                            f"Vintage {vintage} is fully materialised and ready for estimation.",
                            style={
                                "fontWeight": "650",
                                "fontSize": "14px",
                                "color": "#14532d",
                                "marginTop": "3px",
                            },
                        ),
                    ]
                ),
                banner,
            ],
            style={
                "border": "2px solid #22c55e",
                "background": "#f0fdf4",
                "borderRadius": "10px",
                "padding": "12px 14px",
                "marginTop": "12px",
                "marginBottom": "10px",
            },
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
    Output("est-outlier-years", "value"),
    Output("est-outlier-prior-years", "value"),
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
            if payload.get("legacy_outlier_profile_migration"):
                message += " · " + str(payload["legacy_outlier_profile_migration"])
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
        prior["outlier_interval_years"],
        prior["outlier_prior_strength_years"],
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
    Output("est-missing-method", "value"),
    Input("est-profile-select", "value"),
)
def load_estimation_missing_method(profile_value):
    if profile_value in (None, "paper_baseline", "custom"):
        return "baseline"
    try:
        payload = _load_saved_profile(profile_value)
        value = str(payload.get("missing_data_method", "baseline")).lower()
        return value if value in {"baseline", "linear", "dk"} else "baseline"
    except Exception:
        return "baseline"


@callback(
    Output("est-missing-method", "disabled"),
    Input("est-profile-select", "value"),
)
def estimation_missing_method_editability(profile_value):
    # Missing-data treatment is independent from the prior/MCMC profile.
    return False


@callback(
    Output("est-lambda1", "disabled"),
    Output("est-lambda2", "disabled"),
    Output("est-lambda3", "disabled"),
    Output("est-lambda4", "disabled"),
    Output("est-a-prior-var", "disabled"),
    Output("est-phi-mean", "disabled"),
    Output("est-phi-df", "disabled"),
    Output("est-h0-var", "disabled"),
    Output("est-outlier-years", "disabled"),
    Output("est-outlier-prior-years", "disabled"),
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
    Output("est-outlier-effective", "children"),
    Input("est-model-select", "value"),
    Input("est-outlier-years", "value"),
    Input("est-outlier-prior-years", "value"),
)
def estimation_outlier_calendar_note(model_id, interval_years, prior_strength_years):
    if not model_id or interval_years in (None, ""):
        return "Select a model to see the frequency conversion."
    try:
        spec = model_spec(model_id)
        years = float(interval_years)
        periods = years * periods_per_year(spec.frequency)
        unit = "months" if spec.frequency == "monthly" else "weeks"
        strength_years = float(prior_strength_years)
        prior_periods = outlier_prior_observations_from_years(
            strength_years, spec.frequency
        )
        return (
            f"{spec.frequency.capitalize()} model · mean interval {years:g} years "
            f"= {periods:g} {unit} · prior confidence {strength_years:g} years "
            f"= {prior_periods:g} {unit}."
        )
    except Exception as exc:
        return f"Invalid outlier interval: {exc}"


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
    Input("est-outlier-years", "value"),
    Input("est-outlier-prior-years", "value"),
    Input("est-missing-method", "value"),
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
    outlier_interval_years,
    outlier_prior_strength_years,
    missing_data_method,
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
            "outlier_interval_years": outlier_interval_years,
            "outlier_prior_strength_years": outlier_prior_strength_years,
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
        "missing_data_method": str(missing_data_method or "baseline").lower(),
        "selected_structure": {
            "p": custom_p,
            "exog_prior_scale": exog_prior_scale,
            "reference_month": reference_month,
        },
        "per_model": per_model_rows or _baseline_per_model_rows(),
    }
    errors = []
    try:
        _prior_from_mapping(payload["prior"], frequency="monthly")
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
    try:
        _missing_method_override(payload.get("missing_data_method", "baseline"))
    except Exception as exc:
        errors.append(f"Missing values: {exc}")

    if payload["scope"] == "per_model":
        for row in payload["per_model"]:
            model_id = str(row.get("model_id", "unknown"))
            try:
                if _safe_int(row.get("p"), f"{model_id} p") < 1:
                    raise ValueError("p must be positive")
                row_method = str(
                    row.get("missing_data_method", model_spec(model_id).missing_data_method)
                ).lower()
                if row_method not in {"linear", "dk"}:
                    raise ValueError("missing_data_method must be linear or dk")
                row_prior = dict(payload["prior"])
                for key in ("lambda1", "lambda2", "lambda3", "lambda4"):
                    row_prior[key] = row.get(key)
                _prior_from_mapping(row_prior, frequency=model_spec(model_id).frequency)
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
    Input("url", "pathname"),
)
def estimation_preview(model_id, vintage, config_store, pathname):
    if (pathname or "") != "/estimation":
        raise PreventUpdate
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
    """Extract aggregate-key -> run_id from legacy or logical store refs."""
    from inflation_path_portability import run_id_from_forecast_store

    out: dict[str, str] = {}
    for key, raw in dict(metadata.get("component_forecast_stores", {}) or {}).items():
        run_id = run_id_from_forecast_store(raw)
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
    Input("url", "pathname"),
    State("est-diag-run-select", "value"),
)
def estimation_diagnostic_run_options(model_id, vintage, _registry, result_store, pathname, current):
    if (pathname or "") != "/estimation":
        raise PreventUpdate
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
    State("url", "pathname"),
)
def render_estimation_diagnostics(model_id, vintage, run_id, pathname):
    if (pathname or "") != "/estimation":
        raise PreventUpdate
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
    Input("url", "pathname"),
)
def render_estimation_aggregate_validation(vintage, _registry, _result_store, pathname):
    if (pathname or "") != "/estimation":
        raise PreventUpdate
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
    State("url", "pathname"),
    prevent_initial_call=False,
)
def structural_reference_regime(regime, context, store, options, current, pathname):
    if (pathname or "") != "/structural":
        raise PreventUpdate
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


# GRAPH_EXPORT_READABILITY_G2_ENERGY_STRUCTURAL_HD_PLACEHOLDER_V1
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
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


@callback(
    output=Output("structural-volatility-store", "data"),
    inputs=[
        Input("structural-run", "n_clicks"),
        Input("structural-run-banner", "children"),
        Input("structural-draws", "value"),
    ],
    state=[
        State("url", "pathname"),
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
    prevent_initial_call=True,
)
def load_structural_volatility(
    set_progress,
    _n_clicks,
    _activation,
    posterior_draws,
    pathname,
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
    running=[
        (Output("structural-analysis-status", "children"),
         "Computing structural IRF / FEVD…",
         "Structural IRF / FEVD ready"),
    ],
    prevent_initial_call=True,
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
    # STRUCTURAL_NUMERIC_SHOCK_S1B_V1_ENERGY
    try:
        if shock_size is None:
            raise ValueError("shock magnitude is empty or invalid")
        shock_size_requested = float(shock_size)
        if not np.isfinite(shock_size_requested) or shock_size_requested <= 0.0:
            raise ValueError(
                f"shock magnitude must be finite and strictly positive, got {shock_size!r}"
            )
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            "ok": False,
            "payload_kind": "irf_fevd",
            "dashboard_request_signature": None,
            "error": (
                "Invalid structural shock magnitude. Enter a finite positive number; "
                "no structural calculation was run. "
                f"({exc})"
            ),
        }
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
    prevent_initial_call=True,
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

# ENERGY_STRUCTURAL_DISPLAY_HORIZON_S2_V1
def _structural_display_payload(
    store: Mapping[str, Any] | None,
    display_horizon: int | None,
) -> Mapping[str, Any] | None:
    # Presentation-only shallow slice of an already-computed IRF/FEVD payload.
    if not isinstance(store, Mapping):
        return store
    if not store.get("ok"):
        return store

    signature = dict(store.get("dashboard_request_signature") or {})
    try:
        computed_h = int(
            store.get("horizon")
            or signature.get("horizon")
            or STRUCTURAL_DEFAULT_HORIZON
        )
    except (TypeError, ValueError):
        computed_h = int(STRUCTURAL_DEFAULT_HORIZON)
    computed_h = max(1, computed_h)

    try:
        requested_h = int(display_horizon or computed_h)
    except (TypeError, ValueError):
        requested_h = computed_h
    requested_h = max(1, requested_h)
    effective_h = min(requested_h, computed_h)

    payload = dict(store)
    for key in ("irf", "fevd"):
        rows = store.get(key)
        if not isinstance(rows, list):
            continue
        kept = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                h = int(row.get("horizon"))
            except (TypeError, ValueError):
                continue
            if h <= effective_h:
                kept.append(dict(row))
        payload[key] = kept

    payload["display_horizon"] = int(effective_h)
    payload["display_horizon_requested"] = int(requested_h)
    payload["computed_horizon"] = int(computed_h)
    return payload


# ENERGY_STRUCTURAL_STALE_CURRENT_S3_V1
def _structural_freshness_signature(
    context,
    posterior_draws,
    reference_date,
    horizon,
    shock_unit,
    shock_size,
):
    # Reconstruct exactly the compute-relevant dashboard request signature.
    if not isinstance(context, Mapping):
        context = {}

    draws_requested = int(posterior_draws or DEFAULT_STRUCTURAL_DRAWS)
    horizon_requested = int(horizon or STRUCTURAL_DEFAULT_HORIZON)
    shock_unit_requested = str(shock_unit or DEFAULT_SHOCK_UNIT)

    if shock_size is None:
        raise ValueError("shock magnitude is empty or invalid")
    shock_size_requested = float(shock_size)
    if not np.isfinite(shock_size_requested) or shock_size_requested <= 0.0:
        raise ValueError("shock magnitude must be finite and strictly positive")

    effective_reference = reference_date
    return {
        "model_id": str(context.get("model_id") or ""),
        "vintage": str(context.get("vintage") or ""),
        "run_id": str(context.get("run_id") or ""),
        "posterior_draws": draws_requested,
        "reference_date": (
            None
            if effective_reference is None
            else pd.Timestamp(effective_reference).isoformat()
        ),
        "horizon": horizon_requested,
        "shock_unit": shock_unit_requested,
        "shock_size": shock_size_requested,
    }


@callback(
    Output("structural-freshness-status", "children"),
    Output("structural-freshness-status", "style"),
    Input("structural-store", "data"),
    Input("ctx-store", "data"),
    Input("structural-draws", "value"),
    Input("structural-reference-date", "value"),
    Input("structural-horizon", "value"),
    Input("structural-shock-unit", "value"),
    Input("structural-shock-size", "value"),
)
def structural_freshness_status(
    structural_store,
    context,
    posterior_draws,
    reference_date,
    horizon,
    shock_unit,
    shock_size,
):
    base_style = {
        "fontSize": "11px",
        "fontWeight": 700,
        "marginTop": "6px",
        "padding": "6px 8px",
        "borderRadius": "6px",
    }

    try:
        desired = _structural_freshness_signature(
            context,
            posterior_draws,
            reference_date,
            horizon,
            shock_unit,
            shock_size,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        style = dict(base_style)
        style.update(
            {
                "color": "#991B1B",
                "backgroundColor": "#FEF2F2",
                "border": "1px solid #FECACA",
            }
        )
        return (
            "INVALID SETTINGS · The structural result below is not current. "
            f"{exc}.",
            style,
        )

    if not isinstance(structural_store, Mapping) or not structural_store.get("ok"):
        style = dict(base_style)
        style.update(
            {
                "color": "#92400E",
                "backgroundColor": "#FFFBEB",
                "border": "1px solid #FDE68A",
            }
        )
        return (
            "NO CURRENT RESULT · Run or refresh Structural Analysis for the "
            "selected computation settings.",
            style,
        )

    actual = structural_store.get("dashboard_request_signature")
    if not isinstance(actual, Mapping):
        style = dict(base_style)
        style.update(
            {
                "color": "#92400E",
                "backgroundColor": "#FFFBEB",
                "border": "1px solid #FDE68A",
            }
        )
        return (
            "STALE · The displayed structural result has no comparable request "
            "signature. Refresh Structural Analysis.",
            style,
        )

    actual = dict(actual)
    if actual == desired:
        style = dict(base_style)
        style.update(
            {
                "color": "#166534",
                "backgroundColor": "#F0FDF4",
                "border": "1px solid #BBF7D0",
            }
        )
        return (
            "CURRENT · IRF / FEVD match the computation settings above.",
            style,
        )

    keys = (
        "model_id",
        "vintage",
        "run_id",
        "posterior_draws",
        "reference_date",
        "horizon",
        "shock_unit",
        "shock_size",
    )
    changed = [key for key in keys if actual.get(key) != desired.get(key)]
    labels = {
        "model_id": "model",
        "vintage": "vintage",
        "run_id": "run",
        "posterior_draws": "posterior draws",
        "reference_date": "reference date",
        "horizon": "computation horizon",
        "shock_unit": "shock unit",
        "shock_size": "shock magnitude",
    }
    changed_text = ", ".join(labels[key] for key in changed) or "structural settings"

    style = dict(base_style)
    style.update(
        {
            "color": "#92400E",
            "backgroundColor": "#FFFBEB",
            "border": "1px solid #FDE68A",
        }
    )
    return (
        "STALE · Results shown below were computed with previous settings "
        f"({changed_text}). Wait for the current recalculation or use Refresh "
        "Structural Analysis.",
        style,
    )

@callback(
    Output("structural-display-status", "children"),
    Input("structural-store", "data"),
    Input("structural-display-horizon", "value"),
)
def structural_display_status(store, display_horizon):
    if not isinstance(store, dict) or not store.get("ok"):
        return "No computed IRF / FEVD result is available yet."

    payload = _structural_display_payload(store, display_horizon)
    computed_h = int(payload.get("computed_horizon", STRUCTURAL_DEFAULT_HORIZON))
    requested_h = int(payload.get("display_horizon_requested", computed_h))
    effective_h = int(payload.get("display_horizon", computed_h))
    frequency = str(payload.get("frequency") or "period").lower()
    unit = (
        "weeks" if frequency == "weekly"
        else "months" if frequency == "monthly"
        else "periods"
    )

    if requested_h > computed_h:
        return (
            f"Computed H={computed_h} {unit} · requested display H={requested_h} "
            f"is clipped to H={effective_h}. Increase Computation horizon to extend the result."
        )
    return (
        f"Computed H={computed_h} {unit} · displaying H={effective_h}. "
        "Display changes are presentation-only."
    )

@callback(
    Output("structural-irf-graph", "figure"),
    Output("structural-irf-table", "data"),
    Input("structural-store", "data"),
    Input("structural-display-horizon", "value"),
    Input("structural-response", "value"),
    Input("structural-shock", "value"),
    Input("structural-irf-metric", "value"),
    Input("structural-irf-fan", "value"),
)
def structural_irf_graph(store, display_horizon, response, shock, metric, fan_mode):
    resolved_metric = metric or "cumulative"
    display_store = _structural_display_payload(store, display_horizon)
    return (
        irf_figure(
            display_store,
            response=response,
            shock=shock,
            metric=resolved_metric,
            fan_mode=fan_mode or "68",
        ),
        structural_irf_records(
            display_store,
            response=response,
            shock=shock,
            metric=resolved_metric,
        ),
    )


@callback(
    Output("structural-fevd-graph", "figure"),
    Output("structural-fevd-table", "data"),
    Output("structural-fevd-table", "columns"),
    Input("structural-store", "data"),
    Input("structural-display-horizon", "value"),
    Input("structural-response", "value"),
)
def structural_fevd_graph(store, display_horizon, response):
    display_store = _structural_display_payload(store, display_horizon)
    rows, columns = structural_fevd_table(display_store, response=response)
    return fevd_figure(display_store, response=response), rows, columns


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
    if path in {"/", "/overview", "/data"}:
        # Global pages have no domain of their own; keep the section tabs visible.
        return {}, hidden, hidden
    domain = domain_from_path(pathname)
    if domain == "headline":
        return hidden, {}, hidden
    if domain == "core":
        return hidden, hidden, {}
    return {}, hidden, hidden




@callback(
    Output("global-topbar", "style"),
    Output("selection-banner", "style"),
    Input("url", "pathname"),
)
def data_shell_visibility(pathname: str | None):
    path = pathname or ""
    if path in {
        "/", "/overview", "/data",
        "/estimation", "/estimation/headline", "/estimation/core",
        "/headline/estimation", "/headline/diagnostics", "/core/estimation",
    }:
        return {"display": "none"}, {"display": "none"}
    return {}, {}




@callback(
    Output("page-overview", "style"),
    Output("page-data", "style"),
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
    Output("page-core-structural", "style"),
    Output("page-core-estimation", "style"),
    Input("url", "pathname"),
)
def route(pathname: str | None):
    pathname = pathname or "/overview"
    route_name = {
        "/": "overview",
        "/overview": "overview",
        "/data": "data",
        "/forecast": "forecast",
        "/forecast/aggregate": "aggregate",
        "/scenarios": "scenarios",
        "/structural": "structural",
        "/estimation": "estimation",

        # Canonical section-first Headline routes.
        "/forecast/headline": "headline-forecast",
        "/scenarios/headline": "headline-scenarios",
        "/structural/headline": "headline-structural",
        "/estimation/headline": "headline-estimation",

        # Canonical section-first Core routes.
        "/forecast/core": "core-forecast",
        "/scenarios/core": "core-scenarios",
        "/structural/core": "core-structural",
        "/estimation/core": "core-estimation",

        # Existing aliases remain functional.
        "/aggregate": "aggregate",
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
        "/core/structural": "core-structural",
        "/core/estimation": "core-estimation",
    }.get(pathname, "overview")

    visible = {"display": "block"}
    hidden = {"display": "none"}
    # Positional order MUST stay aligned with the Output list above.
    names = (
        "overview",             # 1  page-overview
        "data",                 # 2  page-data
        "forecast",             # 4  page-forecast
        "aggregate",            # 5  page-aggregate
        "scenarios",            # 6  page-scenarios
        "structural",           # 7  page-structural
        "estimation",           # 8  page-estimation
        "headline-forecast",    # 9  page-headline-forecast
        "headline-scenarios",   # 10 page-headline-scenarios
        "headline-structural",  # 11 page-headline-structural
        "headline-estimation",  # 12 page-headline-diagnostics
        "core-forecast",        # 13 page-core-forecast
        "core-scenarios",       # 14 page-core-scenarios
        "core-structural",      # 15 page-core-structural
        "core-estimation",      # 16 page-core-estimation
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
    Input("url", "pathname"),
    State("forecast-metric", "value"),
)
def metric_options(store: dict | None, pathname: str | None, current: str | None):
    if (pathname or "") != "/forecast":
        raise PreventUpdate
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
    Input("url", "pathname"),
    State("forecast-series", "value"),
)
def series_options(
    store: dict | None,
    metric: str | None,
    pathname: str | None,
    current: str | None,
):
    if (pathname or "") != "/forecast":
        raise PreventUpdate
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
    Input("url", "pathname"),
)
def update_forecast_view(
    store: dict | None,
    metric: str | None,
    series: str | None,
    fan_mode: str,
    pathname: str | None,
):
    if (pathname or "") != "/forecast":
        raise PreventUpdate
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
# ENERGY_SCENARIO_WORKSPACE_V1
def _joint_energy_dashboard_recipe(
    *,
    conditional_store,
    tax_store,
    vintage,
    selected_aggregate_run_id,
):
    vintage = str(vintage or "")
    if not vintage:
        raise ConditionalScenarioError(
            "No production vintage is selected."
        )

    conditional_specs = []
    aggregate_ids = set()
    labels = []

    for model_id in conditional_set_components(conditional_store):
        payload = conditional_set_payload(conditional_store, model_id)
        meta = dict((payload or {}).get("meta") or {})
        if not meta:
            continue

        item_vintage = str(meta.get("vintage") or vintage)
        if item_vintage != vintage:
            raise ConditionalScenarioError(
                f"{model_id}: conditional vintage {item_vintage} "
                f"!= production vintage {vintage}."
            )

        aggregate_run_id = str(meta.get("aggregate_run_id") or "")
        if not aggregate_run_id:
            raise ConditionalScenarioError(
                f"{model_id}: active conditional does not record aggregate_run_id."
            )
        aggregate_ids.add(aggregate_run_id)

        path_fields = conditional_path_signature_fields(
            path_value=meta.get("path_value"),
            path_values=meta.get("path_values"),
        )
        conditional_specs.append(
            {
                "vintage": vintage,
                "aggregate_run_id": aggregate_run_id,
                "model_id": str(model_id),
                "condition_variable": str(
                    meta.get("condition_variable") or ""
                ),
                "path_mode": str(meta.get("path_mode") or ""),
                **path_fields,
                "condition_start": int(meta.get("condition_start") or 1),
                "condition_end": (
                    None
                    if meta.get("condition_end") is None
                    else int(meta.get("condition_end"))
                ),
                "condition_mask_hash": meta.get("condition_mask_hash"),
            }
        )
        labels.append(
            {
                "kind": "Conditional",
                "component": str(meta.get("model_label") or model_id),
                "detail": str(
                    meta.get("condition_description")
                    or meta.get("condition_variable")
                    or ""
                ),
            }
        )

    tax_scenarios = scenario_set_to_tax_scenarios(tax_store)
    for row in scenario_set_summary(tax_store):
        vat = float(row.get("vat_delta_pp", 0.0) or 0.0)
        excise = float(row.get("excise_delta", 0.0) or 0.0)
        if abs(vat) < 1e-15 and abs(excise) < 1e-15:
            continue
        detail = f"VAT {vat:+.2f} pp"
        if abs(excise) >= 1e-15:
            detail += (
                f" · excise {excise:+.3f} "
                + str(row.get("excise_unit") or "")
            )
        labels.append(
            {
                "kind": "Tax",
                "component": str(
                    row.get("label")
                    or row.get("model_id")
                    or "Energy component"
                ),
                "detail": detail,
            }
        )

    count = len(conditional_specs) + len(tax_scenarios)
    if count < 1:
        raise ConditionalScenarioError(
            "No active conditional or tax Energy scenario."
        )

    selected = str(selected_aggregate_run_id or "")
    if not aggregate_ids:
        if not selected:
            raise ConditionalScenarioError(
                "Select the HICP Energy aggregate run used by the joint package."
            )
        aggregate_ids.add(selected)
    elif selected and selected not in aggregate_ids:
        raise ConditionalScenarioError(
            "The selected aggregate run differs from the aggregate "
            "used by the active conditional scenarios."
        )

    if len(aggregate_ids) != 1:
        raise ConditionalScenarioError(
            "Active conditional scenarios refer to different HICP Energy "
            "aggregate runs: "
            + ", ".join(sorted(aggregate_ids))
        )

    aggregate_run_id = next(iter(aggregate_ids))
    signature = json.dumps(
        {
            "vintage": vintage,
            "aggregate_run_id": aggregate_run_id,
            "conditional": conditional_specs,
            "tax": tax_scenarios,
        },
        sort_keys=True,
        default=str,
    )
    return {
        "vintage": vintage,
        "aggregate_run_id": aggregate_run_id,
        "conditional_specs": conditional_specs,
        "tax_scenarios": tax_scenarios,
        "labels": labels,
        "scenario_count": count,
        "signature": signature,
    }



def _resolve_canonical_joint_energy_scenario(
    *,
    conditional_store: Mapping[str, Any] | None,
    tax_store: Mapping[str, Any] | None,
    vintage: str | None,
    selected_aggregate_run_id: str | None,
    cached_payload: Mapping[str, Any] | None = None,
    force_recompute: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Resolve one exact Joint Energy package for the CURRENT assumptions.

    Resolution order:
    1. exact browser/session Joint payload supplied by the caller;
    2. exact diskcache payload keyed by the current dashboard recipe;
    3. replay the saved Energy posteriors through compute_joint_energy_scenario.

    This helper never estimates a BVAR and never writes a Dash store.  The
    returned payload is normalized to the canonical downstream propagation
    shape consumed by the existing Energy -> Headline bridge.
    """
    recipe = _joint_energy_dashboard_recipe(
        conditional_store=conditional_store,
        tax_store=tax_store,
        vintage=vintage,
        selected_aggregate_run_id=selected_aggregate_run_id,
    )

    def _valid(payload: Mapping[str, Any] | None) -> bool:
        item = dict(payload or {})
        meta = dict(item.get("meta") or {})
        bridge = dict(item.get("headline_bridge") or {})
        return bool(
            item
            and item.get("ok")
            and str(meta.get("dashboard_input_signature") or "")
            == str(recipe["signature"])
            and str(meta.get("vintage") or "") == str(recipe["vintage"])
            and str(meta.get("aggregate_run_id") or "")
            == str(recipe["aggregate_run_id"])
            and str(bridge.get("contract") or "")
            == ENERGY_BRIDGE_CONTRACT_VERSION
            and bool(bridge.get("scenario_active", False))
        )

    def _canonicalize(
        payload: Mapping[str, Any],
        *,
        source_label: str,
    ) -> dict[str, Any]:
        item = dict(payload or {})
        meta = dict(item.get("meta") or {})
        bridge = dict(item.get("headline_bridge") or {})
        if not _valid(item):
            raise ConditionalScenarioError(
                "Resolved Joint Energy payload failed the canonical "
                "vintage / aggregate / signature / bridge contract."
            )
        meta.update(
            {
                "dashboard_input_signature": recipe["signature"],
                "dashboard_active_labels": recipe["labels"],
                "scenario_type": "joint",
                "aggregate_run_id": str(recipe["aggregate_run_id"]),
                "aggregate_forecast_name": str(
                    meta.get("forecast_name")
                    or bridge.get("forecast_name")
                    or "unconditional"
                ),
                "n_draws": int(
                    meta.get("n_aggregate_draws_paired")
                    or bridge.get("n_draws")
                    or 0
                ),
                "scenario_components": list(
                    bridge.get("scenario_components")
                    or meta.get("active_models")
                    or []
                ),
                "scenario_signature": bridge.get("scenario_signature"),
                "scenario_starts": dict(
                    bridge.get("scenario_starts")
                    or meta.get("scenario_starts")
                    or {}
                ),
                "canonical_propagation_source": str(source_label),
            }
        )
        item["meta"] = meta
        return item

    if not force_recompute and _valid(cached_payload):
        return (
            _canonicalize(
                dict(cached_payload or {}),
                source_label="joint-energy-scenario-store",
            ),
            recipe,
            "session-cache",
        )

    cache_key = (
        "canonical-joint-energy-v1::"
        + __import__("hashlib").sha256(
            str(recipe["signature"]).encode("utf-8")
        ).hexdigest()
    )
    if not force_recompute:
        disk_payload = _diskcache.get(cache_key)
        if _valid(disk_payload):
            return (
                _canonicalize(
                    dict(disk_payload or {}),
                    source_label="diskcache",
                ),
                recipe,
                "disk-cache",
            )

    row = _selected_aggregate_row(
        recipe["vintage"],
        recipe["aggregate_run_id"],
    )
    if row is None:
        raise ConditionalScenarioError(
            "The current Joint Energy recipe aggregate run is not present "
            "in the frozen registry snapshot."
        )

    payload = compute_joint_energy_scenario(
        Path(str(row["directory"])),
        project_root=PROJECT_ROOT,
        conditional_specs=recipe["conditional_specs"],
        tax_scenarios=recipe["tax_scenarios"],
    )
    payload = dict(payload or {})
    meta = dict(payload.get("meta") or {})
    meta.update(
        {
            "dashboard_input_signature": recipe["signature"],
            "dashboard_active_labels": recipe["labels"],
        }
    )
    payload["meta"] = meta

    if not _valid(payload):
        raise ConditionalScenarioError(
            "Newly computed Joint Energy package failed the current "
            "dashboard recipe / bridge contract."
        )

    # Cache only the engine package. Dash stores remain owned by their
    # pre-existing callbacks; this helper creates no second store writer.
    _diskcache.set(cache_key, payload, expire=3600)
    return (
        _canonicalize(
            payload,
            source_label="joint-energy-engine",
        ),
        recipe,
        "recomputed",
    )




# ENERGY_SCENARIO_TAX_CUTS_V1
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


def _scenario_vat_delta_error(
    *,
    baseline_vat: float,
    delta_vat: float,
) -> str | None:
    """Return an actionable error when a VAT delta leaves [0, 100]."""
    baseline = float(baseline_vat)
    delta = float(delta_vat)
    result = baseline + delta
    if result < -1e-12:
        return (
            f"VAT cut too large: baseline {baseline:.2f}% + "
            f"({delta:+.2f} pp) = {result:.2f}%. Minimum allowed change is "
            f"{-baseline:+.2f} pp (VAT = 0%)."
        )
    if result > 100.0 + 1e-12:
        return (
            f"VAT increase too large: baseline {baseline:.2f}% + "
            f"({delta:+.2f} pp) = {result:.2f}%. Maximum allowed change is "
            f"{100.0 - baseline:+.2f} pp (VAT = 100%)."
        )
    return None


def _scenario_component_contract_for_row(
    vintage: str,
    model_id: str,
    row,
    *,
    start_date=None,
):
    """Load the saved-run tax contract, optionally at a selected start date."""
    from energy_bvar_io import load_energy_bvar_forecast

    directory = Path(str(row["directory"]))
    start_key = "default" if start_date is None else str(pd.Timestamp(start_date))
    key = (
        _registry_snapshot_id(),
        str(vintage),
        str(model_id),
        str(directory.resolve()),
        start_key,
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
            start_date=start_date,
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
    """Rank usable scenario aggregates by available calendar horizon first."""
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

    candidates = []
    for _idx, row in frame.iterrows():
        directory = Path(str(row["directory"]))
        try:
            contract = _cached_conditional_aggregate_contract(directory)
            if not contract.get("model_ids"):
                continue
            raw_dates = pd.to_datetime(
                list(contract.get("dates") or []),
                errors="coerce",
            )
            valid_dates = [pd.Timestamp(x) for x in raw_dates if not pd.isna(x)]
            horizon_months = len(
                {stamp.to_period("M") for stamp in valid_dates}
            )
            if horizon_months < 1:
                continue
            forecast_name = str(contract.get("forecast_name") or "").strip()
        except Exception:
            continue

        raw_promoted = row.get("promoted")
        promoted = bool(
            pd.notna(raw_promoted) and int(raw_promoted) == 1
        )
        created = pd.to_datetime(
            row.get("created_at_utc"),
            errors="coerce",
            utc=True,
        )
        created_rank = -1 if pd.isna(created) else int(pd.Timestamp(created).value)
        run_id = str(row["aggregate_run_id"])
        candidates.append(
            {
                "run_id": run_id,
                "horizon_months": int(horizon_months),
                "forecast_name": forecast_name,
                "promoted": promoted,
                "created_rank": created_rank,
            }
        )

    if not candidates:
        return [], None

    candidates.sort(
        key=lambda item: (
            -int(item["horizon_months"]),
            -int(bool(item["promoted"])),
            -int(item["created_rank"]),
            str(item["run_id"]),
        )
    )
    max_horizon = int(candidates[0]["horizon_months"])
    preferred_ids = {
        str(item["run_id"])
        for item in candidates
        if int(item["horizon_months"]) == max_horizon
    }

    options = []
    for item in candidates:
        tags = [f"{int(item['horizon_months'])}m"]
        if item["forecast_name"]:
            tags.append(str(item["forecast_name"]))
        if item["promoted"]:
            tags.append("PROMOTED")
        options.append(
            {
                "label": f"{_short_run(item['run_id'])} · " + " · ".join(tags),
                "value": str(item["run_id"]),
            }
        )

    selected = (
        str(current)
        if current is not None and str(current) in preferred_ids
        else str(candidates[0]["run_id"])
    )
    return options, selected


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
    Output("conditional-question", "children"),
    Output("conditional-support-note", "children"),
    Output("conditional-window-start", "options"),
    Output("conditional-window-start", "value"),
    Output("conditional-window-end", "options"),
    Output("conditional-window-end", "value"),
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
    if str(path_mode or "percent") != "percent":
        path_mode = "percent"

    row = _selected_aggregate_row(vintage, aggregate_run_id)
    if row is None or not model_id or not condition_variable:
        return (
            "Set all (%)",
            10.0,
            "Choose an Energy component and a scenario driver.",
            "The source aggregate is selected automatically from the production vintage.",
            [],
            None,
            [],
            None,
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
        first_date = (
            pd.Timestamp(future[0]).date().isoformat()
            if len(future)
            else "—"
        )
        last_future_date = (
            pd.Timestamp(future[-1]).date().isoformat()
            if len(future)
            else "—"
        )
        H = int(contract.get("H") or len(future))
        if H < 1:
            raise ConditionalScenarioError(
                "The saved forecast has no future period."
            )
        period_prefix = (
            "W"
            if str(contract.get("frequency") or "").lower() == "weekly"
            else "M"
        )
        options = [
            {"label": f"{period_prefix}+{x}", "value": x}
            for x in range(1, H + 1)
        ]

        existing = conditional_set_payload(conditional_store, model_id)
        meta = dict((existing or {}).get("meta", {}) or {})
        same_variable = (
            str(meta.get("condition_variable") or "")
            == str(condition_variable)
        )
        if same_variable:
            window_start = int(meta.get("condition_start") or 1)
            raw_end = meta.get("condition_end")
            window_end = H if raw_end is None else int(raw_end)
        else:
            window_start = 1
            window_end = H if period_prefix == "W" else min(3, H)

        window_start = min(max(window_start, 1), H)
        window_end = min(max(window_end, window_start), H)

        stored_mode = str(
            meta.get("path_mode") or "percent"
        ).strip().lower()
        same_percent = same_variable and stored_mode == "percent"
        legacy_level = same_variable and stored_mode == "level"

        if same_percent and "path_value" in meta:
            set_all_value = float(meta["path_value"])
            assumption_text = (
                f"is {set_all_value:+.2f}% versus its latest observed level"
            )
        elif same_percent and meta.get("path_values"):
            values = [float(x) for x in list(meta.get("path_values") or [])]
            set_all_value = values[0] if values else 10.0
            assumption_text = (
                "follows the active custom percentage path versus "
                "its latest observed level"
            )
        elif legacy_level:
            set_all_value = 10.0
            assumption_text = (
                "follows the percentage path entered below; the currently "
                "stored legacy absolute-level scenario remains unchanged "
                "until it is removed or explicitly overwritten"
            )
        else:
            set_all_value = 10.0
            assumption_text = (
                "follows the percentage path entered below versus "
                "its latest observed level"
            )

        driver_label = str(condition_variable).replace("_", " ").title()
        hicp_component = str(
            contract.get("affected_hicp_component") or model_id
        ).replace("_", " ").title()
        question = html.Div(
            [
                html.Strong("You are asking: "),
                html.Span(
                    f"What happens to HICP {hicp_component} if "
                    f"{driver_label} {assumption_text} during "
                    f"{period_prefix}+{window_start}…"
                    f"{period_prefix}+{window_end}?"
                ),
            ]
        )
        note = (
            f"{contract['model_label']} · run {_short_run(contract['run_id'])} · "
            f"{condition_variable.replace('_', ' ')} last observed "
            f"{last_value:.6g}{(' ' + unit if unit else '')} on {last_date} · "
            f"saved forecast spans {first_date} through {last_future_date} "
            f"({period_prefix}+1…{period_prefix}+{H}) · periods outside the "
            "active hard-condition window are unconstrained but may move under "
            "joint DK conditioning · "
            f"{int(contract['n_forecast_draws']):,} saved forecast draws · "
            "one active conditional is allowed per component BVAR."
        )
        if legacy_level:
            note += (
                " · Legacy absolute-level scenario currently stored: "
                "preserved for existing results / Joint propagation, but "
                "not loaded into this percent-only editor."
            )
        return (
            "Set all (%)",
            set_all_value,
            question,
            note,
            options,
            window_start,
            options,
            window_end,
        )
    except Exception as exc:
        return (
            "Set all (%)",
            10.0,
            "Scenario question unavailable.",
            html.Div(
                [
                    html.Strong("Conditional controls unavailable: "),
                    html.Span(str(exc)),
                ],
                className="banner-error",
            ),
            [],
            None,
            [],
            None,
        )


# CONDITIONAL_RUNTIME_DIAGNOSTIC_V1
# ENERGY_CONDITIONAL_PATH_EDITOR_E2_V1
@callback(
    Output("conditional-path-editor", "children"),
    Input("conditional-agg-select", "value"),
    Input("conditional-component-select", "value"),
    Input("conditional-variable-select", "value"),
    Input("vintage-select", "value"),
    Input("conditional-store", "data"),
)
def conditional_path_editor(
    aggregate_run_id,
    model_id,
    condition_variable,
    vintage,
    conditional_store,
):
    row = _selected_aggregate_row(vintage, aggregate_run_id)
    if row is None or not model_id or not condition_variable:
        return html.Div(
            "Choose an Energy component and scenario driver.",
            className="placeholder-text",
        )

    try:
        contract = _cached_conditional_component_contract(
            Path(str(row["directory"])),
            model_id=str(model_id),
        )
        H = int(
            contract.get("H")
            or len(contract.get("future_dates", []) or [])
        )
        if H < 1:
            raise ConditionalScenarioError(
                "The saved forecast has no future period."
            )
        prefix = (
            "W"
            if str(contract.get("frequency") or "").lower() == "weekly"
            else "M"
        )

        existing = conditional_set_payload(conditional_store, model_id)
        meta = dict((existing or {}).get("meta", {}) or {})
        same_variable = (
            str(meta.get("condition_variable") or "")
            == str(condition_variable)
        )
        stored_mode = str(
            meta.get("path_mode") or "percent"
        ).strip().lower()
        same_percent = same_variable and stored_mode == "percent"

        values = [10.0] * H
        active_start = 1
        active_end = H if prefix == "W" else min(3, H)
        if same_variable:
            active_start = min(
                max(int(meta.get("condition_start") or 1), 1),
                H,
            )
            raw_end = meta.get("condition_end")
            active_end = H if raw_end is None else int(raw_end)
            active_end = min(max(active_end, active_start), H)

            if same_percent and "path_value" in meta:
                scalar = float(meta["path_value"])
                values = [scalar] * H
            elif same_percent and meta.get("path_values"):
                stored = [
                    float(x)
                    for x in list(meta.get("path_values") or [])
                ]
                expected = active_end - active_start + 1
                if len(stored) == expected:
                    for offset, value in enumerate(
                        stored,
                        start=active_start,
                    ):
                        values[offset - 1] = value

        cells = []
        for offset in range(1, H + 1):
            active = active_start <= offset <= active_end
            cells.append(
                html.Div(
                    [
                        html.Label(
                            f"{prefix}+{offset}",
                            className="control-label",
                            style={"marginBottom": "4px"},
                        ),
                        dcc.Input(
                            id={
                                "type": "conditional-path-cell",
                                "index": offset,
                            },
                            type="number",
                            value=float(values[offset - 1]),
                            step=0.1,
                            debounce=0.35,
                            disabled=not active,
                            className="compact-dropdown",
                            style={"width": "100%"},
                        ),
                    ],
                    id={"type": "conditional-path-wrap", "index": offset},
                    style={"minWidth": "82px", "display": "block" if active else "none"},
                )
            )

        return html.Div(
            [
                html.Div(
                    f"{prefix}+1…{prefix}+{H} · "
                    "% change versus the same latest observed driver value",
                    className="control-help",
                    style={"marginBottom": "8px"},
                ),
                html.Div(
                    cells,
                    style={
                        "display": "grid",
                        "gridTemplateColumns": (
                            "repeat(auto-fit, minmax(82px, 1fr))"
                        ),
                        "gap": "8px",
                    },
                ),
            ]
        )
    except Exception as exc:
        return html.Div(
            [
                html.Strong("Conditional path editor unavailable: "),
                html.Span(str(exc)),
            ],
            className="banner-error",
        )


@callback(
    Output(
        {"type": "conditional-path-cell", "index": ALL},
        "disabled",
    ),
    Output(
        {"type": "conditional-path-wrap", "index": ALL},
        "style",
    ),
    Input("conditional-window-start", "value"),
    Input("conditional-window-end", "value"),
    State(
        {"type": "conditional-path-cell", "index": ALL},
        "id",
    ),
)
def conditional_path_window_state(
    condition_start,
    condition_end,
    cell_ids,
):
    ids = list(cell_ids or [])
    if not ids:
        return [], []
    start = int(condition_start or 1)
    end = int(condition_end or start)
    return [not start <= int((item or {}).get('index') or 0) <= end for item in ids], [{"minWidth": "82px", "display": "none" if (not start <= int((item or {}).get('index') or 0) <= end) else "block"} for item in ids]


@callback(
    Output(
        {"type": "conditional-path-cell", "index": ALL},
        "value",
    ),
    Input("conditional-path-set-all", "n_clicks"),
    Input("conditional-path-linear-fill", "n_clicks"),
    State("conditional-path-value", "value"),
    State("conditional-linear-start", "value"),
    State("conditional-linear-end", "value"),
    State("conditional-window-start", "value"),
    State("conditional-window-end", "value"),
    State(
        {"type": "conditional-path-cell", "index": ALL},
        "id",
    ),
    State(
        {"type": "conditional-path-cell", "index": ALL},
        "value",
    ),
    prevent_initial_call=True,
)
def conditional_path_fill(
    _set_all_clicks,
    _linear_clicks,
    set_all_value,
    linear_start,
    linear_end,
    condition_start,
    condition_end,
    cell_ids,
    current_values,
):
    trigger = ctx.triggered_id
    if trigger not in {
        "conditional-path-set-all",
        "conditional-path-linear-fill",
    }:
        raise PreventUpdate

    ids = list(cell_ids or [])
    values = list(current_values or [])
    if not ids or len(ids) != len(values):
        raise PreventUpdate

    start = int(condition_start or 1)
    end = int(condition_end or start)
    active_positions = [
        j
        for j, item in enumerate(ids)
        if start
        <= int((item or {}).get("index") or 0)
        <= end
    ]
    if not active_positions:
        raise PreventUpdate

    updated = list(values)
    if trigger == "conditional-path-set-all":
        if set_all_value is None:
            raise PreventUpdate
        fill = float(set_all_value)
        if not np.isfinite(fill):
            raise PreventUpdate
        for j in active_positions:
            updated[j] = fill
        return updated

    if linear_start is None or linear_end is None:
        raise PreventUpdate
    first = float(linear_start)
    last = float(linear_end)
    if not np.isfinite(first) or not np.isfinite(last):
        raise PreventUpdate

    linear = np.linspace(first, last, len(active_positions))
    for j, value in zip(active_positions, linear):
        updated[j] = float(value)
    return updated

@callback(
    output=[
        Output("conditional-store", "data"),
        Output("conditional-run-diagnostic", "children"),
    ],
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
        State(
            {"type": "conditional-path-cell", "index": ALL},
            "id",
        ),
        State(
            {"type": "conditional-path-cell", "index": ALL},
            "value",
        ),
        State("conditional-window-start", "value"),
        State("conditional-window-end", "value"),
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
    cell_ids,
    cell_values,
    condition_start,
    condition_end,
    vintage,
    current_store,
):
    trigger = ctx.triggered_id

    def _emit(message):
        text = str(message)
        print("[Conditional runtime diagnostic] " + text, flush=True)
        return text

    if trigger == "conditional-reset-all":
        set_progress(
            (
                100,
                "Conditional set cleared",
                "No active conditional scenarios.",
            )
        )
        updated = clear_conditional_set(
            vintage=str(vintage) if vintage else None,
            aggregate_run_id=(
                str(aggregate_run_id)
                if aggregate_run_id
                else None
            ),
        )
        return updated, _emit(
            "RESET · conditional set cleared · "
            f"vintage={vintage or '—'} · "
            f"aggregate={aggregate_run_id or '—'}"
        )

    if trigger == "conditional-remove":
        if not model_id:
            return current_store, _emit(
                "REMOVE-SKIP · no component selected"
            )
        updated = remove_conditional_component(
            current_store,
            model_id,
        )
        set_progress(
            (
                100,
                "Conditional removed",
                f"{model_spec(model_id).label} removed from the "
                "active conditional set.",
            )
        )
        return updated, _emit(
            f"REMOVE · model={model_id} · "
            f"remaining={len(conditional_set_components(updated))}"
        )

    if trigger != "conditional-run":
        raise PreventUpdate

    if not aggregate_run_id or not model_id or not condition_variable:
        set_progress(
            (
                0,
                "Conditional forecast failed",
                "Aggregate, component and variable are required.",
            )
        )
        return current_store, _emit(
            "FAIL-STATE · "
            f"aggregate={aggregate_run_id!r} · "
            f"model={model_id!r} · "
            f"variable={condition_variable!r}"
        )

    if str(path_mode or "percent") != "percent":
        set_progress(
            (
                0,
                "Conditional forecast failed",
                "Energy path assumptions are fixed to % vs last observed.",
            )
        )
        return current_store, _emit(
            f"FAIL-STATE · unsupported path_mode={path_mode!r}"
        )

    try:
        start = int(condition_start or 1)
        end = int(condition_end or start)
        ids = list(cell_ids or [])
        values = list(cell_values or [])
        if not ids or len(ids) != len(values):
            raise ConditionalScenarioError(
                "Visible Energy path cells are unavailable."
            )

        by_offset = {}
        for item, value in zip(ids, values):
            offset = int((item or {}).get("index") or 0)
            if offset < start or offset > end:
                continue
            if value is None:
                raise ConditionalScenarioError(
                    f"Condition value is missing at forecast offset {offset}."
                )
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ConditionalScenarioError(
                    f"Condition value is non-finite at forecast offset {offset}."
                )
            by_offset[offset] = numeric

        expected = list(range(start, end + 1))
        missing = [x for x in expected if x not in by_offset]
        if missing:
            raise ConditionalScenarioError(
                "Condition path is missing forecast offsets: "
                + ", ".join(str(x) for x in missing)
            )
        active_values = [float(by_offset[x]) for x in expected]
        path_fields = conditional_path_signature_fields(
            path_values=active_values
        )
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        set_progress((0, "Conditional forecast failed", first))
        return current_store, _emit(
            f"FAIL-PATH · {type(exc).__name__}: {first}"
        )

    row = _selected_aggregate_row(vintage, aggregate_run_id)
    if row is None:
        set_progress(
            (
                0,
                "Conditional forecast failed",
                "Selected aggregate run is unavailable.",
            )
        )
        return current_store, _emit(
            "FAIL-AGGREGATE-LOOKUP · "
            f"vintage={vintage!r} · "
            f"aggregate={aggregate_run_id!r}"
        )

    directory = Path(str(row["directory"]))
    cache_key = "conditional-v2::" + json.dumps(
        {
            "contract": CONDITIONAL_CONTRACT_VERSION,
            "aggregate_run_id": str(aggregate_run_id),
            "model_id": str(model_id),
            "condition_variable": str(condition_variable),
            "path_mode": "percent",
            **path_fields,
            "condition_start": start,
            "condition_end": end,
        },
        sort_keys=True,
    )

    cached = _diskcache.get(cache_key)
    cache_state = "disk-cache"
    if isinstance(cached, dict) and cached.get("ok"):
        payload = cached
    else:
        cache_state = "recomputed"
        try:
            set_progress(
                (
                    10,
                    "Loading saved posterior",
                    "Reading the exact component forecast/run recorded "
                    "by the selected aggregate.",
                )
            )
            set_progress(
                (
                    30,
                    "Paired conditional forecast",
                    "Replaying the saved unconditional path and "
                    "imposing the observable future path with the "
                    "same random stream.",
                )
            )
            payload = compute_conditional_scenario(
                directory,
                project_root=PROJECT_ROOT,
                model_id=str(model_id),
                condition_variable=str(condition_variable),
                path_mode="percent",
                path_value=path_fields.get("path_value"),
                path_values=path_fields.get("path_values"),
                condition_start=start,
                condition_end=end,
            )
            _diskcache.set(cache_key, payload, expire=3600)
        except Exception as exc:
            first = (
                str(exc).splitlines()[0]
                if str(exc)
                else type(exc).__name__
            )
            set_progress((0, "Conditional forecast failed", first))
            return current_store, _emit(
                f"FAIL-ENGINE · {type(exc).__name__}: {first}"
            )

    if not isinstance(payload, dict) or not payload.get("ok"):
        set_progress(
            (
                0,
                "Conditional forecast failed",
                "Engine returned a non-ready payload.",
            )
        )
        return current_store, _emit(
            "FAIL-PAYLOAD-CONTRACT · "
            f"type={type(payload).__name__} · "
            f"ok={getattr(payload, 'get', lambda *_: None)('ok')}"
        )

    try:
        payload_json = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        payload_bytes = len(payload_json.encode("utf-8"))
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        set_progress(
            (
                0,
                "Conditional forecast failed",
                "Payload is not JSON serializable: " + first,
            )
        )
        return current_store, _emit(
            f"FAIL-PAYLOAD-SERIALIZE · {type(exc).__name__}: {first}"
        )

    try:
        updated = upsert_conditional_component(
            current_store,
            payload,
        )
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        set_progress(
            (
                0,
                "Conditional forecast failed",
                "Conditional store upsert failed: " + first,
            )
        )
        return current_store, _emit(
            f"FAIL-UPSERT · {type(exc).__name__}: {first}"
        )

    try:
        store_json = json.dumps(
            updated,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        store_bytes = len(store_json.encode("utf-8"))
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        set_progress(
            (
                0,
                "Conditional forecast failed",
                "Final conditional store is not JSON serializable: "
                + first,
            )
        )
        return current_store, _emit(
            f"FAIL-STORE-SERIALIZE · {type(exc).__name__}: {first}"
        )

    components = len(conditional_set_components(updated))
    meta = dict(payload.get("meta") or {})
    prefix = (
        "W"
        if str(meta.get("frequency") or "").lower() == "weekly"
        else "M"
    )
    profile = "constant" if "path_value" in path_fields else "custom"
    diagnostic = (
        "RETURNING · PASS-ENGINE · PASS-UPSERT · "
        f"source={cache_state} · model={model_id} · "
        f"aggregate={aggregate_run_id} · "
        f"path={profile} · "
        f"window={prefix}+{start}→{prefix}+{end} · "
        f"payload_json={payload_bytes / 1024.0:.1f} KiB · "
        f"store_json={store_bytes / 1024.0:.1f} KiB · "
        f"components={components}"
    )
    _emit(diagnostic)

    set_progress(
        (
            100,
            "Conditional scenario active",
            f"{model_spec(model_id).label} added/updated. "
            f"{components} conditional component(s) active. "
            f"Returning {store_bytes / 1024.0:.1f} KiB JSON store.",
        )
    )
    return updated, diagnostic


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
                            html.Strong(str(row["label"]).upper()),
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
                        "Driver: "
                        f"{row['condition_variable'].replace('_',' ').title()} · "
                        "Assumption: "
                        f"{row['condition_description']} · "
                        f"{impact_label}",
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
    Output('energy-conditional-results-kpis', 'style'), Output('energy-conditional-results-table', 'style'), Output('energy-conditional-results-path', 'style'), Output('energy-conditional-results-component-grid', 'style'), Output('energy-conditional-results-aggregate-grid', 'style'), Input("conditional-store", "data"),
    Input("conditional-component-select", "value"),
    Input("conditional-fan", "value"),
)
def conditional_figures(store, model_id, fan_mode):
    payload = conditional_set_payload(store, model_id)
    if not payload:
        empty = empty_conditional_figure(
            "Add/update a conditional scenario for the selected component."
        )
        return (empty, empty, empty, empty, empty, ({} if bool(payload) else {'display': 'none'}), ({} if bool(payload) else {'display': 'none'}), ({} if bool(payload) else {'display': 'none'}), ({'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px'} if bool(payload) else {'display': 'none', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px'}), ({'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px', 'marginBottom': '24px'} if bool(payload) else {'display': 'none', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px', 'marginBottom': '24px'}))

    meta = dict(payload.get("meta", {}) or {})
    if "path_value" in meta:
        path_revision = str(meta.get("path_value"))
    else:
        path_revision = json.dumps(
            conditional_path_signature_fields(
                path_value=meta.get("path_value"),
                path_values=meta.get("path_values"),
            ),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    revision = (
        f"{meta.get('aggregate_run_id','agg')}::"
        f"{meta.get('model_id','model')}::"
        f"{meta.get('condition_variable','condition')}::"
        f"{meta.get('path_mode','mode')}::"
        f"{path_revision}"
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
    ({} if bool(payload) else {'display': 'none'}), ({} if bool(payload) else {'display': 'none'}), ({} if bool(payload) else {'display': 'none'}), ({'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px'} if bool(payload) else {'display': 'none', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px'}), ({'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px', 'marginBottom': '24px'} if bool(payload) else {'display': 'none', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px', 'marginBottom': '24px'}))


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


@callback(Output("scenario-component-select","options"), Output("scenario-component-select","value"), Input("vintage-select","value"), Input("forecast-select","value"), Input("registry-store","data"), Input("url","pathname"), State("scenario-component-select","value"))
def scenario_component_options(vintage, forecast_name, _, pathname, current):
    if (pathname or "") != "/scenarios":
        raise PreventUpdate
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


# VAT_SCENARIO_EDITOR_TRIGGER_POLICY_V1
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
    State("scenario-start-date","date"),
    State("scenario-vat-delta","value"),
    State("scenario-excise-delta","value"),
)
def scenario_control_defaults(
    model_id,
    vintage,
    forecast_name,
    _registry,
    scenario_store,
    current_start_date,
    current_vat_delta,
    current_excise_delta_display,
):
    # Use the complete trigger set; do not rely on a single-trigger shortcut.
    # more than one Input can change in the same Dash resolution cycle.
    try:
        triggered_props = {
            str(item.get("prop_id") or "")
            for item in (ctx.triggered or [])
            if str(item.get("prop_id") or "") not in {"", "."}
        }
    except Exception:
        # Direct unit/audit calls have no Dash callback context.  Preserve the
        # historical behaviour: load committed/default values.
        triggered_props = set()

    context_props = {
        "scenario-component-select.value",
        "vintage-select.value",
        "forecast-select.value",
    }
    context_changed = bool(triggered_props & context_props)
    scenario_store_changed = "scenario-store.data" in triggered_props
    registry_changed = "registry-store.data" in triggered_props

    # Ranked causes, evaluated from the full set:
    # context > explicit scenario-store commit > passive registry refresh.
    trigger_mode = next(
        mode
        for mode, active in (
            ("context", context_changed),
            ("commit", scenario_store_changed),
            ("registry", registry_changed),
            ("initial", True),
        )
        if active
    )
    preserve_registry_draft = trigger_mode == "registry"

    if not model_id or not vintage or not forecast_name:
        return (
            None, None, None, 0.0, 0.0, "Excise change (− = cut)", "any",
            "No scenario-capable component forecast is available.",
            True, True, True,
        )

    row = _scenario_component_forecast_row(vintage, model_id, forecast_name)
    if row is None:
        return (
            None, None, None, 0.0, 0.0, "Excise change (− = cut)", "any",
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
            None, None, None, 0.0, 0.0, "Excise change (− = cut)", "any",
            f"Scenario controls unavailable: {exc}",
            True, True, True,
        )

    # Resolve committed state only for the EXACT model/vintage/forecast
    # context.  scenario_set_payload() intentionally falls back to the first
    # component when a requested component is absent, which is unsuitable for
    # an editor default callback after a component switch.
    store_item = dict(scenario_store or {})
    legacy_meta = dict(store_item.get("meta", {}) or {})
    store_vintage = store_item.get("vintage", legacy_meta.get("vintage"))
    store_forecast = store_item.get(
        "forecast_name", legacy_meta.get("forecast_name")
    )
    store_context_matches = (
        store_vintage in (None, "", str(vintage))
        and store_forecast in (None, "", str(forecast_name))
    )
    components = (
        scenario_set_components(scenario_store)
        if store_context_matches
        else {}
    )
    existing = components.get(str(model_id))
    em = dict((existing or {}).get("meta", {}) or {})

    frequency = str(contract.get("frequency", "monthly"))
    min_start = _normalise_scenario_period(contract["min_start"], frequency)
    max_start = _normalise_scenario_period(contract["max_start"], frequency)
    default_start = _normalise_scenario_period(
        contract["default_start"], frequency
    )

    committed_start = _normalise_scenario_period(
        em.get("scenario_start") or default_start,
        frequency,
    )
    if committed_start < min_start or committed_start > max_start:
        committed_start = default_start

    start = committed_start
    date_output = committed_start.date()

    # A registry-only refresh is passive.  Keep the user's unsaved date exactly
    # as typed/displayed when its model-period normalisation remains admissible.
    if preserve_registry_draft and current_start_date not in (None, ""):
        try:
            draft_start = _normalise_scenario_period(
                current_start_date, frequency
            )
        except Exception:
            draft_start = None
        if (
            draft_start is not None
            and min_start <= draft_start <= max_start
        ):
            start = draft_start
            date_output = current_start_date

    committed_vat = float(em.get("vat_delta_pp", 0.0) or 0.0)
    if preserve_registry_draft and current_vat_delta is not None:
        try:
            vat_output = float(current_vat_delta)
        except (TypeError, ValueError):
            vat_output = committed_vat
    else:
        vat_output = committed_vat

    pre_source_unit = str(contract.get("excise_unit", "source unit"))
    pre_display = _scenario_excise_display_spec(pre_source_unit)
    stored_source_delta = float(em.get("excise_delta", 0.0) or 0.0)
    committed_excise_display = _scenario_excise_to_display(
        stored_source_delta, pre_source_unit
    )
    if preserve_registry_draft and current_excise_delta_display is not None:
        try:
            excise_output = float(current_excise_delta_display)
        except (TypeError, ValueError):
            excise_output = committed_excise_display
    else:
        excise_output = committed_excise_display

    try:
        contract = _scenario_component_contract_for_row(
            str(vintage), model_id, row, start_date=start
        )
    except Exception as exc:
        # On a passive registry refresh, do not replace valid numeric draft
        # inputs by zeros merely because the refreshed contract cannot currently
        # be evaluated at the chosen period.
        return (
            date_output,
            min_start.date(),
            max_start.date(),
            vat_output,
            excise_output,
            f"Excise change ({pre_display['display_unit']}; − = cut)",
            "any",
            f"Scenario controls unavailable at {start.date()}: {exc}",
            False, True, True,
        )

    source_unit = str(contract.get("excise_unit", "source unit"))
    display = _scenario_excise_display_spec(source_unit)
    source_baseline = float(contract["baseline_excise_at_start"])
    display_baseline = _scenario_excise_to_display(source_baseline, source_unit)

    # Commit/context/initial modes render the committed source-unit value using
    # the current display unit. Registry-only mode keeps the browser draft.
    if not preserve_registry_draft:
        excise_output = _scenario_excise_to_display(
            stored_source_delta, source_unit
        )

    min_excise_display = _scenario_excise_to_display(
        float(contract["excise_delta_min"]), source_unit
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
        f"{tax_source_note}baseline at selected start {start.date().isoformat()}: "
        f"VAT {contract['baseline_vat_at_start']:.2f}% · excise "
        f"{display_baseline:.2f} {display['display_unit']}. "
        f"Negative changes are tax cuts. VAT may be cut by at most "
        f"{abs(float(contract['vat_delta_min'])):.2f} pp; excise may be cut by "
        f"at most {abs(min_excise_display):.2f} {display['display_unit']} (to zero)."
        f"{conversion_note} Zero/zero removes the component from the active set."
    )

    return (
        date_output,
        min_start.date(),
        max_start.date(),
        vat_output,
        excise_output,
        f"Excise change ({display['display_unit']}; − = cut)",
        "any",
        note,
        False,
        False,
        False,
    )

@callback(
    Output("scenario-tax-preview", "children"),
    Output("scenario-vat-delta", "min"),
    Output("scenario-vat-delta", "max"),
    Output("scenario-excise-delta", "min"),
    Input("scenario-component-select", "value"),
    Input("vintage-select", "value"),
    Input("forecast-select", "value"),
    Input("scenario-start-date", "date"),
    Input("scenario-vat-delta", "value"),
    Input("scenario-excise-delta", "value"),
    Input("registry-store", "data"),
)
def scenario_tax_input_preview(
    model_id,
    vintage,
    forecast_name,
    start_date,
    vat_delta,
    excise_delta_display,
    _,
):
    if not model_id or not vintage or not forecast_name:
        return (
            "Select a scenario-capable component to preview the tax change.",
            None, None, None,
        )

    row = _scenario_component_forecast_row(vintage, model_id, forecast_name)
    if row is None:
        return (
            "No unique saved component forecast is available.",
            None, None, None,
        )

    try:
        contract = _scenario_component_contract_for_row(
            str(vintage), model_id, row, start_date=start_date
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
        vat_guard = _scenario_vat_delta_error(
            baseline_vat=base_vat,
            delta_vat=delta_vat,
        )
        excise_guard = _scenario_excise_plausibility_error(
            displayed_delta=delta_display,
            source_delta=delta_source,
            source_baseline=baseline_source,
            source_unit=source_unit,
        )
        vat_min = float(contract["vat_delta_min"])
        vat_max = float(contract["vat_delta_max"])
        excise_min_display = _scenario_excise_to_display(
            float(contract["excise_delta_min"]), source_unit
        )
        selected_start = pd.Timestamp(contract["selected_start"]).date().isoformat()
    except Exception as exc:
        return (
            html.Span(f"Tax-input preview unavailable: {exc}"),
            None, None, None,
        )

    pieces = [
        html.Strong("Scenario preview · "),
        html.Span(f"start {selected_start} · "),
        html.Span(
            f"VAT baseline {base_vat:.2f}% · change {delta_vat:+.2f} pp → "
            f"{scenario_vat:.2f}%"
        ),
        html.Span(
            f" · allowed VAT change [{vat_min:+.2f}, {vat_max:+.2f}] pp"
        ),
        html.Span(
            f" · Excise baseline {baseline_display:.2f} "
            f"{display['display_unit']} · change {delta_display:+.2f} → "
            f"{scenario_display:.2f} {display['display_unit']}"
        ),
        html.Span(
            f" · minimum excise change {excise_min_display:+.2f} "
            f"{display['display_unit']} (tax = 0)"
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
    for guard in (vat_guard, excise_guard):
        if guard:
            pieces.append(
                html.Span(
                    " · BLOCKED: " + guard,
                    style={"color": "#B42318", "fontWeight": "700"},
                )
            )
    if not vat_guard and not excise_guard and (
        delta_vat < 0.0 or delta_display < 0.0
    ):
        pieces.append(
            html.Span(
                " · Tax cut is valid.",
                style={"color": "#067647", "fontWeight": "700"},
            )
        )
    return pieces, None, None, None
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
            clear_scenario_set(vintage=vintage, forecast_name=forecast_name),
            html.Div(
                "All component tax scenarios cleared.",
                className="selection-banner",
            ),
        )

    if not model_id:
        return no_update, html.Div("Select a component.", className="banner-error")

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

    row = _scenario_component_forecast_row(vintage, model_id, forecast_name)
    if row is None:
        return no_update, html.Div(
            f"{model_spec(model_id).label}: no unique promoted/saved run is available.",
            className="banner-error",
        )

    if vat_delta is None or excise_delta_display is None:
        return no_update, html.Div(
            "VAT or excise input was rejected by the browser as an invalid number. "
            "Re-enter both values.",
            className="banner-error",
        )
    vat_delta = float(vat_delta)
    excise_delta_display = float(excise_delta_display)
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
        base_contract = component_tax_scenario_contract(
            model_id,
            forecast,
            dataset_path=dataset,
        )
        frequency = str(base_contract.get("frequency", "monthly"))
        effective_start = _normalise_scenario_period(start_date, frequency)
        min_start = _normalise_scenario_period(base_contract["min_start"], frequency)
        max_start = _normalise_scenario_period(base_contract["max_start"], frequency)
        if effective_start < min_start or effective_start > max_start:
            return no_update, html.Div(
                f"Scenario start must lie inside the selected forecast horizon: "
                f"{min_start.date().isoformat()} — {max_start.date().isoformat()}.",
                className="banner-error",
            )

        contract = component_tax_scenario_contract(
            model_id,
            forecast,
            dataset_path=dataset,
            start_date=effective_start,
        )
        source_unit = str(contract.get("excise_unit", "source unit"))
        excise_delta = _scenario_excise_from_display(
            excise_delta_display,
            source_unit,
        )

        if abs(vat_delta) < 1e-15 and abs(excise_delta) < 1e-15:
            return (
                remove_scenario_component(current_store, model_id),
                html.Div(
                    f"{model_spec(model_id).label}: zero changes, component removed "
                    "from the active set.",
                    className="selection-banner",
                ),
            )

        vat_guard = _scenario_vat_delta_error(
            baseline_vat=float(contract["baseline_vat_at_start"]),
            delta_vat=vat_delta,
        )
        if vat_guard:
            return no_update, html.Div(
                [html.Strong("Scenario not applied: "), html.Span(vat_guard)],
                className="banner-error",
            )

        excise_guard = _scenario_excise_plausibility_error(
            displayed_delta=excise_delta_display,
            source_delta=excise_delta,
            source_baseline=float(contract["baseline_excise_at_start"]),
            source_unit=source_unit,
        )
        if excise_guard:
            suffix = (
                " A negative final excise would be a subsidy, not an excise tax cut."
                if float(contract["baseline_excise_at_start"]) + excise_delta < 0.0
                else ""
            )
            return no_update, html.Div(
                [
                    html.Strong("Scenario not applied: "),
                    html.Span(excise_guard + suffix),
                ],
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
            [html.Strong("Scenario calculation failed: "), html.Span(str(exc))],
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
    resulting_vat = float(contract["baseline_vat_at_start"]) + vat_delta
    resulting_excise = _scenario_excise_to_display(
        float(contract["baseline_excise_at_start"]) + float(meta.get("excise_delta", 0.0) or 0.0),
        meta.get("excise_unit"),
    )

    return updated, html.Div(
        [
            html.Strong(str(result["hicp_label"])),
            html.Span(
                f" · VAT change {vat_delta:+.2f} pp → {resulting_vat:.2f}%"
            ),
            html.Span(
                f" · excise change {display_delta:+.2f} {display_unit} → "
                f"{resulting_excise:.2f} {display_unit}"
            ),
            html.Span(f" · {meta['n_draws_effective']} paired draws"),
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


@callback(Output("scenario-main-graph","figure"),Output("scenario-level-impact","figure"),Output("scenario-yoy-impact","figure"),Output("scenario-tax-graph","figure"),Output('energy-tax-results-kpis', 'style'), Output('energy-tax-results-table', 'style'), Output('energy-tax-results-main', 'style'), Output('energy-tax-results-impact-grid', 'style'), Output('energy-tax-results-tax', 'style'), Input("scenario-store","data"),Input("scenario-component-select","value"),Input("scenario-fan","value"))
def scenario_figures(store,model_id,fan_mode):
    payload=scenario_set_payload(store,model_id)
    if not payload:
        empty=empty_scenario_figure("Add a tax scenario for the selected component"); return (empty,empty,empty,empty, ({} if bool(payload) else {'display': 'none'}), ({} if bool(payload) else {'display': 'none'}), ({} if bool(payload) else {'display': 'none'}), ({'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px'} if bool(payload) else {'display': 'none', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px'}), ({} if bool(payload) else {'display': 'none'}))
    meta=dict(payload.get("meta",{}) or {}); revision=f"{meta.get('run_id','run')}::{meta.get('forecast_name','forecast')}::{model_id}::scenario"; mode=fan_mode or "68"
    return (scenario_main_figure(payload,fan_mode=mode,uirevision=revision+"::main"),scenario_impact_figure(payload,metric="level",fan_mode=mode,uirevision=revision+"::level"),scenario_impact_figure(payload,metric="yoy",fan_mode=mode,uirevision=revision+"::yoy"),scenario_tax_figure(payload,uirevision=revision+"::tax"), ({} if bool(payload) else {'display': 'none'}), ({} if bool(payload) else {'display': 'none'}), ({} if bool(payload) else {'display': 'none'}), ({'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px'} if bool(payload) else {'display': 'none', 'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))', 'gap': '16px'}), ({} if bool(payload) else {'display': 'none'}))


def _scenario_propagation_provenance(store: dict | None) -> str:
    """Compact exact lineage for the inter-domain propagation cards."""
    meta = dict((store or {}).get("meta", {}) or {})
    bridge = dict((store or {}).get("headline_bridge", {}) or {})

    vintage = bridge.get("vintage") or meta.get("vintage") or "—"
    source_aggregate_run_id = meta.get("aggregate_run_id") or "—"
    scenario_aggregate_run_id = bridge.get("aggregate_run_id") or "—"
    bridge_contract = bridge.get("contract") or "—"

    signature = (
        meta.get("scenario_signature")
        or bridge.get("scenario_signature")
        or []
    )
    component_runs: list[str] = []
    if isinstance(signature, list):
        for item in signature:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("model_id") or "").strip()
            run_id = str(item.get("run_id") or "").strip()
            if run_id:
                component_runs.append(
                    f"{model_id}:{run_id}" if model_id else run_id
                )
    component_run_text = (
        ", ".join(component_runs)
        if component_runs
        else "—"
    )

    return (
        f"vintage {vintage} · component run {component_run_text} · "
        f"aggregate source run {source_aggregate_run_id} · "
        f"scenario aggregate {scenario_aggregate_run_id} · "
        f"bridge contract {bridge_contract}"
    )


@callback(
    Output("scenario-propagation-energy-result", "children"),
    Input("agg-scenario-store", "data"),
)
def scenario_propagation_result_cards(store: dict | None):
    """Render HICP Energy strictly from the already-computed aggregate store."""
    if not store:
        return _result_block(
            eyebrow="HICP Energy impact",
            value=None,
            unit_date="Awaiting propagated aggregate scenario",
            uncertainty=(
                "Posterior mean and 68% interval appear after "
                "agg-scenario-store is materialized."
            ),
            provenance=(
                "source · agg-scenario-store · no materialized propagation"
            ),
        )

    provenance = _scenario_propagation_provenance(store)
    meta = dict((store or {}).get("meta", {}) or {})
    if str(meta.get("scenario_type") or "").strip().lower() == "conditional":
        result = dict((store or {}).get("energy_result", {}) or {})
        if not result:
            return _result_block(
                eyebrow="HICP Energy impact · conditional",
                value=None,
                unit_date="Conditional Energy result unavailable",
                uncertainty=(
                    "No statistic fabricated. Re-run Add / update conditional once "
                    "to refresh the bridge-ready Energy result."
                ),
                provenance=provenance,
            )

        date = pd.Timestamp(result["date"]).date().isoformat()
        impact = float(result["value"])
        low = result.get("q16")
        high = result.get("q84")
        draws = result.get("n_draws")

        interval = "68% interval unavailable"
        if low is not None and high is not None:
            interval = f"68% [{float(low):+.3f}, {float(high):+.3f}] pp"

        draw_text = (
            f"{int(draws):,} paired aggregate draws"
            if draws is not None
            else "paired aggregate draws"
        )

        return _result_block(
            eyebrow="HICP Energy impact · conditional",
            value=f"{impact:+.3f} pp",
            unit_date=f"{date} · posterior median",
            uncertainty=f"{interval} · {draw_text}",
            provenance=provenance,
        )
    values = aggregate_live_scenario_kpis(store)
    terminal_date = values.get("date")

    if terminal_date is None:
        return _result_block(
            eyebrow="HICP Energy impact",
            value=None,
            unit_date="No tax aggregate impact fan in current store",
            uncertainty=(
                "No statistic fabricated. Conditional HICP Energy results remain "
                "in the conditional scenario block above."
            ),
            provenance=provenance,
        )

    date = pd.Timestamp(terminal_date).date().isoformat()
    impact = float(values["impact"])
    low = float(values["low"])
    high = float(values["high"])
    draws = values.get("n_draws")
    draw_text = (
        f"{int(draws):,} paired aggregate draws"
        if draws is not None
        else "paired aggregate draws"
    )
    return _result_block(
        eyebrow="HICP Energy impact",
        value=f"{impact:+.2f} pp",
        unit_date=f"{date} · posterior mean",
        uncertainty=f"68% [{low:+.2f}, {high:+.2f}] pp · {draw_text}",
        provenance=provenance,
    )


_HEADLINE_PROPAGATION_CONDITION_HORIZON = 3
_HEADLINE_PROPAGATION_STATISTIC = "mean"


def _scenario_headline_source_contract(store: dict | None) -> dict[str, str]:
    """Resolve the exact Energy scenario identity consumed by Headline.

    Scenario identity is part of the source key. This is essential for Joint
    Energy because the source aggregate run stays fixed while the combination
    of conditional/tax assumptions can change.
    """
    item = dict(store or {})
    meta = dict(item.get("meta") or {})
    bridge = dict(item.get("headline_bridge") or {})
    if not item:
        raise ValueError("No agg-scenario-store is materialized.")
    if not bool(bridge.get("scenario_active", False)):
        raise ValueError("The current HICP Energy aggregate has no active scenario.")

    vintage = str(bridge.get("vintage") or meta.get("vintage") or "").strip()
    source_aggregate_run_id = str(meta.get("aggregate_run_id") or "").strip()
    scenario_aggregate_run_id = str(bridge.get("aggregate_run_id") or "").strip()
    forecast_name = str(bridge.get("forecast_name") or "").strip()
    bridge_contract = str(bridge.get("contract") or "").strip()
    scenario_identity = str(meta.get("dashboard_input_signature") or "").strip()
    if not scenario_identity:
        signature = bridge.get("scenario_signature")
        if signature is not None:
            scenario_identity = json.dumps(
                signature,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
    if not scenario_identity:
        scenario_identity = scenario_aggregate_run_id

    missing = [
        name
        for name, value in (
            ("vintage", vintage),
            ("source_aggregate_run_id", source_aggregate_run_id),
            ("scenario_aggregate_run_id", scenario_aggregate_run_id),
            ("forecast_name", forecast_name),
            ("bridge_contract", bridge_contract),
            ("scenario_identity", scenario_identity),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "agg-scenario-store lacks Headline propagation lineage: "
            + ", ".join(missing)
        )

    source_key = "|".join(
        (
            vintage,
            source_aggregate_run_id,
            scenario_aggregate_run_id,
            forecast_name,
            bridge_contract,
            scenario_identity,
        )
    )
    return {
        "vintage": vintage,
        "source_aggregate_run_id": source_aggregate_run_id,
        "scenario_aggregate_run_id": scenario_aggregate_run_id,
        "forecast_name": forecast_name,
        "bridge_contract": bridge_contract,
        "scenario_identity": scenario_identity,
        "source_key": source_key,
    }


# UNIFIED_SCENARIO_12M_RUNTIME_LOTC1_V1
def _scenario_headline_condition_horizon(
    store: Mapping[str, Any] | None,
) -> int:
    """Use every available monthly Energy bridge date, capped at 12 months."""
    bridge = dict((store or {}).get("headline_bridge") or {})
    raw_dates = list(bridge.get("dates") or [])
    dates = pd.to_datetime(raw_dates, errors="coerce")
    months = {
        pd.Timestamp(value).to_period("M")
        for value in dates
        if not pd.isna(value)
    }
    available = int(len(months))
    if available < 1:
        raise ValueError(
            "The current Energy scenario exposes no monthly Headline bridge dates."
        )
    return int(min(12, available))

def _compute_scenario_headline_propagation(store: dict | None) -> dict:
    """Run the explicit saved-posterior Headline scenario-minus-baseline calculation."""
    source = _scenario_headline_source_contract(store)
    headline_state = _headline_run_artifact_state(source["vintage"])
    if str(headline_state.get("status") or "") != "COMPLETE":
        raise RuntimeError(
            "Headline saved-run selection is not production-complete: "
            + str(headline_state.get("note") or headline_state.get("status") or "unknown")
        )
    run_directory = headline_state.get("directory")
    headline_run_id = str(headline_state.get("run_id") or "").strip()
    if run_directory is None or not headline_run_id:
        raise RuntimeError("Resolved Headline run has no directory/run_id.")

    condition_horizon = _scenario_headline_condition_horizon(store)
    started = time.perf_counter()
    payload = run_saved_headline_energy_marginal(
        run_directory,
        energy_store=store,
        source_aggregate_run_id=source["source_aggregate_run_id"],
        forecast_name=source["forecast_name"],
        H=condition_horizon,
        statistic=_HEADLINE_PROPAGATION_STATISTIC,
        lineage={
            "source_type": "energy_scenario",
            "impact_definition": "energy_scenario_minus_energy_baseline",
            "impact_baseline_label": "Headline conditioned on Energy baseline",
            "impact_scenario_label": "Headline conditioned on Energy scenario",
            "impact_interpretation": "conditional-forecast effect",
            "dashboard_surface": "energy_scenarios_inter_domain_propagation",
        },
        project_root=PROJECT_ROOT,
        persist=False,
    )
    elapsed = float(time.perf_counter() - started)
    computed_at = datetime.now(timezone.utc).isoformat()
    meta = dict(payload.get("meta") or {})
    if meta.get("bvar_reestimated") is not False:
        raise RuntimeError("Headline propagation violated bvar_reestimated=False.")
    if meta.get("paired_energy_baseline_scenario") is not True:
        raise RuntimeError("Headline propagation is not paired Energy scenario-minus-baseline.")
    if int(meta.get("conditioned_horizon_months", -1)) != int(condition_horizon):
        raise RuntimeError("Headline conditioned-horizon contract changed.")
    if int(meta.get("free_propagation_horizon_months", -1)) != (
        int(meta.get("computational_horizon", 0)) - int(condition_horizon)
    ):
        raise RuntimeError("Headline free-propagation horizon contract changed.")

    meta.update(
        {
            "dashboard_computed_at_utc": computed_at,
            "dashboard_elapsed_seconds": elapsed,
            "dashboard_source_key": source["source_key"],
            "dashboard_source_aggregate_run_id": source["source_aggregate_run_id"],
            "dashboard_scenario_aggregate_run_id": source["scenario_aggregate_run_id"],
            "dashboard_bridge_contract": source["bridge_contract"],
            "dashboard_condition_horizon_months": int(condition_horizon),
        }
    )
    payload = dict(payload)
    payload["meta"] = meta
    return {
        "state": "fresh",
        "source_key": source["source_key"],
        "computed_at_utc": computed_at,
        "elapsed_seconds": elapsed,
        "headline_run_id": headline_run_id,
        "provenance": _scenario_propagation_provenance(store),
        "payload": payload,
    }


def _scenario_headline_terminal_summary(payload: dict | None) -> dict | None:
    """Terminal Headline B−A posterior summary; no statistic substitution."""
    item = dict(payload or {})
    dates = pd.DatetimeIndex(pd.to_datetime(item.get("dates") or []))
    impact = dict(item.get("headline_yoy_impact") or {})
    arrays = {}
    for key in ("mean", "q16", "q84"):
        values = np.asarray(impact.get(key) or [], dtype=float)
        arrays[key] = values
    n = min([len(dates), *(len(values) for values in arrays.values())])
    if n < 1:
        return None
    finite = np.isfinite(arrays["mean"][:n])
    if not finite.any():
        return None
    i = int(np.where(finite)[0][-1])
    if not all(np.isfinite(arrays[key][i]) for key in arrays):
        return None
    return {
        "date": pd.Timestamp(dates[i]),
        "mean": float(arrays["mean"][i]),
        "q16": float(arrays["q16"][i]),
        "q84": float(arrays["q84"][i]),
    }


def _format_small_pp(value: float) -> str:
    """Keep tiny economically meaningful Headline effects visible."""
    value = float(value)
    precision = 4 if abs(value) < 0.01 else 2
    return f"{value:+.{precision}f}"


def _scenario_headline_result_card(
    result_store: dict | None,
    *,
    stale: bool,
) -> html.Div:
    result = dict(result_store or {})
    payload = dict(result.get("payload") or {})
    meta = dict(payload.get("meta") or {})
    summary = _scenario_headline_terminal_summary(payload)
    if summary is None:
        raise ValueError("Headline propagation payload has no terminal YoY impact.")

    date = pd.Timestamp(summary["date"]).date().isoformat()
    draws = meta.get("n_draws")
    conditioned = int(meta.get("conditioned_horizon_months", 0))
    propagated = int(meta.get("free_propagation_horizon_months", 0))
    computed_at = str(
        result.get("computed_at_utc")
        or meta.get("dashboard_computed_at_utc")
        or "—"
    )
    elapsed = float(
        result.get("elapsed_seconds")
        or meta.get("dashboard_elapsed_seconds")
        or 0.0
    )
    headline_run_id = str(
        result.get("headline_run_id")
        or meta.get("headline_run_id")
        or "—"
    )
    provenance = (
        str(result.get("provenance") or "source · agg-scenario-store")
        + f" · headline run {headline_run_id}"
        + f" · computed {computed_at}"
    )
    uncertainty = (
        f"68% [{_format_small_pp(summary['q16'])}, "
        f"{_format_small_pp(summary['q84'])}] pp"
        + (
            f" · {int(draws):,} paired Headline draws"
            if draws is not None
            else " · paired Headline draws"
        )
        + f" · conditioned {conditioned}m / freely propagated {propagated}m"
        + " · conditional-forecast effect (B − A)"
        + " · Energy input = posterior-mean path; full Energy path uncertainty not propagated"
    )
    return _result_block(
        eyebrow="Headline impact",
        value=f"{_format_small_pp(summary['mean'])} pp",
        unit_date=(
            f"{date} · posterior mean · {elapsed:.1f} s"
        ),
        uncertainty=uncertainty,
        provenance=provenance,
        state="stale" if stale else "fresh",
    )


@callback(
    Output("scenario-propagation-headline-store", "data"),
    Input("scenario-propagation-headline-run", "n_clicks"),
    State("agg-scenario-store", "data"),
    State("conditional-store", "data"),
    State("scenario-store", "data"),
    State("joint-energy-scenario-store", "data"),
    State("vintage-select", "value"),
    State("conditional-agg-select", "value"),
    prevent_initial_call=True,
    running=[
        (
            Output("scenario-propagation-headline-run", "disabled"),
            True,
            False,
        ),
        (
            Output("scenario-propagation-headline-run", "children"),
            "Resolving Energy + calculating Headline…",
            "Calculate Headline impact (~23–40 s + Joint if needed)",
        ),
    ],
)
def compute_scenario_headline_propagation(
    run_clicks: int | None,
    aggregate_store: dict | None,
    conditional_store: dict | None,
    tax_store: dict | None,
    joint_store: dict | None,
    vintage: str | None,
    selected_aggregate_run_id: str | None,
):
    "Explicit click: resolve CURRENT Energy assumptions, then run Headline A/B."
    if not run_clicks:
        raise PreventUpdate

    source_key = ""
    recipe_signature = ""
    cache_state = "not-required"
    requires_joint = False
    try:
        conditional_summaries = conditional_set_summary(conditional_store)
        tax_summaries = scenario_set_summary(tax_store)
        requires_joint = bool(
            (conditional_summaries and tax_summaries)
            or len(conditional_summaries) > 1
        )

        recipe = _joint_energy_dashboard_recipe(
            conditional_store=conditional_store,
            tax_store=tax_store,
            vintage=vintage,
            selected_aggregate_run_id=selected_aggregate_run_id,
        )
        recipe_signature = str(recipe["signature"])

        effective_store = aggregate_store
        if requires_joint:
            effective_store, recipe, cache_state = (
                _resolve_canonical_joint_energy_scenario(
                    conditional_store=conditional_store,
                    tax_store=tax_store,
                    vintage=vintage,
                    selected_aggregate_run_id=selected_aggregate_run_id,
                    cached_payload=joint_store,
                    force_recompute=False,
                )
            )
            recipe_signature = str(recipe["signature"])

        source = _scenario_headline_source_contract(effective_store)
        source_key = str(source["source_key"])
        result = dict(
            _compute_scenario_headline_propagation(effective_store)
        )

        result["energy_interdomain_store"] = (
            _scenario_interdomain_compact_energy_store(effective_store)
        )

        payload = dict(result.get("payload") or {})
        # HEADLINE_HISTORY_NAMESPACE_D_V1
        # Headline forecast namespace is independent from Energy forecast_name.
        payload["headline_history_yoy"] = (
            _scenario_interdomain_saved_headline_history(
                vintage=str(source["vintage"]),
                headline_run_id=str(result.get("headline_run_id") or ""),
                forecast_name="",
            )
        )
        result["payload"] = payload

        result.update(
            {
                "energy_recipe_signature": recipe_signature,
                "energy_recipe_requires_joint": bool(requires_joint),
                "joint_resolution": cache_state,
                "energy_recipe_scenario_count": int(
                    recipe.get("scenario_count") or 0
                ),
            }
        )
        return result
    except TimeoutError as exc:
        return {
            "state": "timeout",
            "source_key": source_key,
            "energy_recipe_signature": recipe_signature,
            "energy_recipe_requires_joint": bool(requires_joint),
            "joint_resolution": cache_state,
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"Headline conditional calculation timed out: {exc}",
        }
    except Exception as exc:
        return {
            "state": "error",
            "source_key": source_key,
            "energy_recipe_signature": recipe_signature,
            "energy_recipe_requires_joint": bool(requires_joint),
            "joint_resolution": cache_state,
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }


# UNIFIED_SCENARIO_INTERDOMAIN_UI_LOTC2_V1

def _scenario_interdomain_energy_rows(
    store: Mapping[str, Any] | None,
    kind: str,
) -> pd.DataFrame:
    """Return monthly HICP Energy scenario summaries from any canonical store.

    Conditional/Joint stores carry direct row lists. Tax-only aggregate stores
    carry the same summaries in fans_json. This helper only parses the browser
    payload already in memory; it never computes an aggregate.
    """
    item = dict(store or {})
    kind = str(kind or "").strip().lower()
    direct_keys = {
        "history": "aggregate_history_yoy",
        "baseline": "aggregate_baseline_yoy",
        "scenario": "aggregate_scenario_yoy",
        "impact": "aggregate_impact_yoy",
    }
    if kind not in direct_keys:
        raise ValueError(f"Unknown Energy inter-domain series kind {kind!r}.")

    rows = item.get(direct_keys[kind])
    if isinstance(rows, list) and rows:
        frame = pd.DataFrame(list(rows)).copy()
    elif kind != "history" and item.get("fans_json"):
        try:
            frame = pd.read_json(
                __import__("io").StringIO(str(item.get("fans_json"))),
                orient="split",
            )
        except Exception:
            return pd.DataFrame()
        names = {
            "baseline": "baseline_yoy",
            "scenario": "scenario_yoy",
            "impact": "impact_yoy_pp",
        }
        if "name" not in frame:
            return pd.DataFrame()
        frame = frame.loc[
            frame["name"].astype(str).eq(names[kind])
        ].copy()
    else:
        return pd.DataFrame()

    if frame.empty or "date" not in frame:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if kind == "history":
        if "value" not in frame:
            return pd.DataFrame()
        frame["mean"] = pd.to_numeric(frame["value"], errors="coerce")
    elif "mean" not in frame and "value" in frame:
        frame["mean"] = pd.to_numeric(frame["value"], errors="coerce")
    for key in ("mean", "q16", "q50", "q84"):
        if key in frame:
            frame[key] = pd.to_numeric(frame[key], errors="coerce")
    return frame.reset_index(drop=True)


def _scenario_interdomain_headline_rows(
    payload: Mapping[str, Any] | None,
    kind: str,
) -> pd.DataFrame:
    """Return Headline YoY baseline/scenario/impact summaries by calendar month."""
    item = dict(payload or {})
    dates = pd.DatetimeIndex(pd.to_datetime(item.get("dates") or [], errors="coerce"))
    dates = dates[~dates.isna()]
    if len(dates) < 1:
        return pd.DataFrame()

    kind = str(kind or "").strip().lower()
    if kind == "impact":
        block = dict(item.get("headline_yoy_impact") or {})
    elif kind in {"baseline", "scenario"}:
        yoy = dict(
            (((item.get("fans") or {}).get("hicp_total") or {}).get("yoy") or {})
        )
        key = "baseline" if kind == "baseline" else "conditional"
        block = dict(yoy.get(key) or {})
    else:
        raise ValueError(f"Unknown Headline inter-domain series kind {kind!r}.")

    arrays: dict[str, np.ndarray] = {}
    for key in ("mean", "q16", "q50", "q84"):
        values = np.asarray(block.get(key) or [], dtype=float)
        arrays[key] = values
    available = [len(dates), *(len(values) for values in arrays.values() if len(values))]
    n = min(available) if available else 0
    if n < 1 or len(arrays["mean"]) < n:
        return pd.DataFrame()

    data: dict[str, Any] = {
        "date": dates[:n],
        "mean": arrays["mean"][:n],
    }
    for key in ("q16", "q50", "q84"):
        if len(arrays[key]) >= n:
            data[key] = arrays[key][:n]
    frame = pd.DataFrame(data)
    frame = frame.loc[np.isfinite(frame["mean"].to_numpy(dtype=float))].copy()
    return frame.reset_index(drop=True)


def _scenario_interdomain_energy_history(
    aggregate_store: Mapping[str, Any] | None,
    aggregate_display_store: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Observed HICP Energy YoY context, preferring the scenario's own history."""
    direct = _scenario_interdomain_energy_rows(aggregate_store, "history")
    if not direct.empty:
        return direct[["date", "mean"]].dropna().sort_values("date")

    display = dict(aggregate_display_store or {})
    scenario_meta = dict((aggregate_store or {}).get("meta") or {})
    display_meta = dict(display.get("meta") or {})
    wanted = str(scenario_meta.get("aggregate_run_id") or "")
    observed = str(
        display_meta.get("aggregate_run_id")
        or display_meta.get("run_id")
        or ""
    )
    if wanted and observed and wanted != observed:
        return pd.DataFrame()
    try:
        frame = _frame_from_store(display)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return pd.DataFrame()
    required = {"record_type", "metric", "series", "date", "value"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    mask = (
        frame["record_type"].astype(str).eq("history")
        & frame["metric"].astype(str).eq("yoy")
        & frame["series"].astype(str).eq("hicp_energy")
    )
    out = frame.loc[mask, ["date", "value"]].copy()
    if out.empty:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["mean"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date", "mean"])[["date", "mean"]].sort_values("date")


# INTERDOMAIN_TRUE_MERGE_V1
def _scenario_interdomain_compact_energy_store(
    store: Mapping[str, Any] | None,
) -> dict[str, Any]:
    "Keep only the Energy rows required by the unified presentation chart."
    item = dict(store or {})
    if not item:
        return {}
    keys = (
        "meta",
        "headline_bridge",
        "aggregate_history_yoy",
        "aggregate_baseline_yoy",
        "aggregate_scenario_yoy",
        "aggregate_impact_yoy",
        "fans_json",
    )
    return {key: item[key] for key in keys if key in item}


def _scenario_interdomain_saved_headline_history(
    *,
    vintage: str,
    headline_run_id: str,
    forecast_name: str,
) -> list[dict[str, Any]]:
    "Read compact observed Headline YoY history during the explicit click only."
    vintage = str(vintage or "").strip()
    run_id = str(headline_run_id or "").strip()
    forecast = str(forecast_name or "unconditional").strip() or "unconditional"
    if not vintage or not run_id:
        return []

    path = (
        RESULTS_ROOT
        / "headline_joint"
        / vintage
        / run_id
        / "forecasts"
        / forecast
        / DISPLAY_FILENAME
    )
    if not path.is_file():
        print("WARNING · Headline observed history file not found: " + str(path))
        return []

    try:
        frame = pd.read_parquet(
            path,
            columns=["record_type", "metric", "series", "date", "value"],
        )
    except Exception:
        return []
    if frame.empty:
        return []

    mask = (
        frame["record_type"].astype(str).eq("history")
        & frame["metric"].astype(str).eq("yoy")
        & frame["series"].astype(str).eq("hicp_total")
    )
    out = frame.loc[mask, ["date", "value"]].copy()
    if out.empty:
        return []
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date").tail(36)
    return [
        {
            "date": pd.Timestamp(row.date).isoformat(),
            "value": float(row.value),
        }
        for row in out.itertuples(index=False)
        if np.isfinite(float(row.value))
    ]


def _scenario_interdomain_headline_history(
    payload: Mapping[str, Any] | None,
) -> pd.DataFrame:
    "Observed Headline YoY rows embedded by the explicit calculation."
    rows = list((payload or {}).get("headline_history_yoy") or [])
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).copy()
    if frame.empty or not {"date", "value"}.issubset(frame.columns):
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["mean"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "mean"]).sort_values("date")
    return frame[["date", "mean"]].reset_index(drop=True)


# INTERDOMAIN_CAPABILITY_AWARE_SOURCE_B_V1
def _scenario_interdomain_has_energy_paths(
    store: Mapping[str, Any] | None,
) -> bool:
    item = dict(store or {})
    direct_keys = (
        "aggregate_baseline_yoy",
        "aggregate_scenario_yoy",
        "aggregate_impact_yoy",
    )
    if all(
        isinstance(item.get(key), list) and bool(item.get(key))
        for key in direct_keys
    ):
        return True
    return bool(item.get("fans_json"))

def _scenario_interdomain_effective_energy_store(
    aggregate_store: Mapping[str, Any] | None,
    joint_store: Mapping[str, Any] | None,
    result_store: Mapping[str, Any] | None,
    *,
    requires_joint: bool,
    current_recipe_signature: str,
) -> tuple[dict[str, Any], bool, str]:
    aggregate = dict(aggregate_store or {})
    joint = dict(joint_store or {})
    result = dict(result_store or {})
    snapshot = dict(result.get("energy_interdomain_store") or {})
    current = str(current_recipe_signature or "")

    def signature(store: Mapping[str, Any] | None) -> str:
        meta = dict((store or {}).get("meta") or {})
        return str(meta.get("dashboard_input_signature") or "")

    result_signature = str(result.get("energy_recipe_signature") or "")

    if requires_joint:
        if (
            _scenario_interdomain_has_energy_paths(joint)
            and current
            and signature(joint) == current
        ):
            return joint, False, "current Joint Energy store"
        if (
            _scenario_interdomain_has_energy_paths(snapshot)
            and current
            and result_signature == current
        ):
            return snapshot, False, "exact Energy package used by Headline"
        if (
            _scenario_interdomain_has_energy_paths(aggregate)
            and current
            and signature(aggregate) == current
        ):
            return aggregate, False, "current canonical aggregate store"

        # JOINT_REQUIRED_NO_STALE_FALLBACK_F_V1
        # Never display an older marginal/Joint/snapshot package as the current
        # combined scenario when multiple active assumptions require Joint.
        return {}, True, "current Joint Energy package not materialised"

    if _scenario_interdomain_has_energy_paths(aggregate):
        return aggregate, False, "current aggregate scenario store"
    if (
        _scenario_interdomain_has_energy_paths(snapshot)
        and (not current or result_signature == current)
    ):
        return snapshot, False, "exact Energy package used by Headline"
    if _scenario_interdomain_has_energy_paths(joint):
        return joint, False, "Joint Energy store"
    return {}, False, "Energy scenario not materialised"

# INTERDOMAIN_JOINT_VS_STANDALONE_G2_V1
def _scenario_joint_vs_standalone_summary_data(
    result_store: Mapping[str, Any] | None,
    aggregate_store: Mapping[str, Any] | None,
    joint_store: Mapping[str, Any] | None,
    conditional_store: Mapping[str, Any] | None,
    tax_store: Mapping[str, Any] | None,
    tax_marginal_store: Mapping[str, Any] | None,
    *,
    vintage: str | None,
    selected_aggregate_run_id: str | None,
    display_horizon: int | None,
) -> dict[str, Any]:
    """Presentation-only Joint-vs-standalone posterior-mean comparison.

    Every number is read from an already materialised browser payload.  No
    aggregate, conditional scenario, Headline forecast, or BVAR is computed.
    """
    conditional_summaries = conditional_set_summary(conditional_store)
    tax_summaries = scenario_set_summary(tax_store)
    requires_joint = bool(
        (conditional_summaries and tax_summaries)
        or len(conditional_summaries) > 1
    )
    if not requires_joint:
        return {
            "state": "not_applicable",
            "reason": "Joint comparison requires multiple active Energy assumptions.",
        }

    try:
        horizon = int(display_horizon or 12)
    except (TypeError, ValueError):
        horizon = 12
    if horizon not in {3, 6, 9, 12}:
        horizon = 12

    try:
        recipe = _joint_energy_dashboard_recipe(
            conditional_store=conditional_store,
            tax_store=tax_store,
            vintage=vintage,
            selected_aggregate_run_id=selected_aggregate_run_id,
        )
        recipe_signature = str(recipe["signature"])
    except Exception as exc:
        return {
            "state": "unavailable",
            "reason": f"Current Joint recipe unavailable: {exc}",
        }

    energy_store, energy_stale, source_label = (
        _scenario_interdomain_effective_energy_store(
            aggregate_store,
            joint_store,
            result_store,
            requires_joint=True,
            current_recipe_signature=recipe_signature,
        )
    )
    if not energy_store or energy_stale:
        return {
            "state": "awaiting_joint",
            "reason": (
                "Current Joint Energy package is not materialised. "
                "Click Calculate Headline impact first."
            ),
        }

    joint_frame = _scenario_interdomain_energy_rows(
        energy_store,
        "impact",
    )
    if joint_frame.empty or "mean" not in joint_frame:
        return {
            "state": "unavailable",
            "reason": "Current Joint Energy impact has no posterior-mean path.",
        }

    joint_frame = (
        joint_frame.dropna(subset=["date", "mean"])
        .sort_values("date")
        .iloc[:horizon]
        .copy()
    )
    if joint_frame.empty:
        return {
            "state": "unavailable",
            "reason": "Current Joint Energy impact has no finite displayed horizon.",
        }

    joint_row = joint_frame.iloc[-1]
    terminal_date = pd.Timestamp(joint_row["date"])
    joint_mean = float(joint_row["mean"])
    displayed_months = int(len(joint_frame))

    def _mean_at_date(
        payload: Mapping[str, Any] | None,
    ) -> float | None:
        frame = _scenario_interdomain_energy_rows(payload, "impact")
        if frame.empty or "mean" not in frame:
            return None
        block = frame.dropna(subset=["date", "mean"]).copy()
        if block.empty:
            return None
        block["date"] = pd.to_datetime(block["date"], errors="coerce")
        block = block.loc[
            block["date"].dt.to_period("M")
            == terminal_date.to_period("M")
        ]
        if block.empty:
            return None
        value = pd.to_numeric(
            pd.Series([block.iloc[-1]["mean"]]),
            errors="coerce",
        ).iloc[0]
        return None if pd.isna(value) else float(value)

    standalone: list[dict[str, Any]] = []
    missing: list[str] = []

    for summary in conditional_summaries:
        model_id = str(summary.get("model_id") or "")
        if not model_id:
            continue
        payload = conditional_set_payload(
            conditional_store,
            model_id,
        )
        label = str(
            summary.get("label")
            or model_id.replace("_", " ").title()
        )
        value = _mean_at_date(payload)
        if value is None:
            missing.append(f"Conditional · {label}")
            continue
        standalone.append(
            {
                "kind": "Conditional",
                "model_id": model_id,
                "label": label,
                "mean": value,
            }
        )

    tax_marginal = dict(tax_marginal_store or {})
    tax_context_ok = bool(
        tax_marginal
        and str(tax_marginal.get("vintage") or "") == str(vintage or "")
        and str(tax_marginal.get("aggregate_run_id") or "")
        == str(selected_aggregate_run_id or "")
    )
    tax_by_model = (
        dict(tax_marginal.get("by_model") or {})
        if tax_context_ok
        else {}
    )
    tax_errors = (
        dict(tax_marginal.get("errors") or {})
        if tax_context_ok
        else {}
    )

    for summary in tax_summaries:
        model_id = str(summary.get("model_id") or "")
        if not model_id:
            continue
        label = str(
            summary.get("label")
            or model_id.replace("_", " ").title()
        )
        payload = dict(tax_by_model.get(model_id) or {})
        value = _mean_at_date(payload)
        if value is None:
            suffix = (
                f" ({tax_errors[model_id]})"
                if model_id in tax_errors
                else ""
            )
            missing.append(f"Tax · {label}{suffix}")
            continue
        standalone.append(
            {
                "kind": "Tax",
                "model_id": model_id,
                "label": label,
                "mean": value,
            }
        )

    expected_count = int(
        len(conditional_summaries) + len(tax_summaries)
    )
    if missing or len(standalone) != expected_count:
        return {
            "state": "incomplete",
            "date": terminal_date.isoformat(),
            "displayed_months": displayed_months,
            "joint_mean": joint_mean,
            "available_count": int(len(standalone)),
            "expected_count": expected_count,
            "missing": missing,
            "source_label": source_label,
            "reason": (
                "Standalone posterior means are incomplete; "
                "no q50 substitution was made."
            ),
        }

    standalone_sum = float(
        sum(float(item["mean"]) for item in standalone)
    )
    difference = float(joint_mean - standalone_sum)
    return {
        "state": "ready",
        "date": terminal_date.isoformat(),
        "displayed_months": displayed_months,
        "joint_mean": joint_mean,
        "standalone_sum": standalone_sum,
        "difference": difference,
        "standalone": standalone,
        "source_label": source_label,
        "statistic": "posterior mean",
        "comparison_contract": (
            "same calendar month; posterior mean on Joint and every "
            "standalone HICP Energy impact"
        ),
    }

def _scenario_interdomain_figure(
    aggregate_store: Mapping[str, Any] | None,
    headline_payload: Mapping[str, Any] | None,
    *,
    mode: str,
    horizon: int,
    headline_stale: bool = False,
    energy_stale: bool = False,
    aggregate_display_store: Mapping[str, Any] | None = None,
) -> go.Figure:
    # INTERDOMAIN_FINAL_RENDERER_G1_V1
    # INTERDOMAIN_INTERPRETATION_G1_1_V1
    # INTERDOMAIN_IMPACT_STACKED_H1_V1
    # INTERDOMAIN_HEADLINE_GAP_INSET_G1_2_V1
    # INTERDOMAIN_READABILITY_H2_V1
    # INTERDOMAIN_REMOVE_HEADLINE_GAP_H2_2_V1
    # Presentation only: no scenario/model computation is allowed here.
    mode = str(mode or "impact").strip().lower()
    if mode not in {"impact", "paths"}:
        mode = "impact"
    try:
        horizon = int(horizon)
    except (TypeError, ValueError):
        horizon = 12
    horizon = 12 if horizon not in {3, 6, 9, 12} else horizon

    energy_colour = TOKENS["cyan_deep"]
    headline_colour = TOKENS["navy"]
    muted = TOKENS["muted"]
    observed_colour = "#94A3B8"

    def _rgba(hex_colour: str, alpha: float) -> str:
        raw = str(hex_colour).lstrip("#")
        if len(raw) != 6:
            return f"rgba(107,110,114,{float(alpha):.3f})"
        rgb = tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{float(alpha):.3f})"

    def _slice(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        block = frame.copy()
        block["date"] = pd.to_datetime(block["date"], errors="coerce")
        return block.dropna(subset=["date"]).sort_values("date").copy()
    
    def _domain_months(frames: tuple[pd.DataFrame, ...]) -> set[pd.Period]:
        month_sets: list[set[pd.Period]] = []
        for frame in frames:
            if frame is None or frame.empty:
                continue
            months = set(pd.DatetimeIndex(frame["date"]).to_period("M"))
            if months:
                month_sets.append(months)
        if not month_sets:
            return set()
        return set.intersection(*month_sets)
    
    def _restrict_months(frame: pd.DataFrame, months: list[pd.Period]) -> pd.DataFrame:
        if frame is None or frame.empty or not months:
            return pd.DataFrame()
        month_key = pd.DatetimeIndex(frame["date"]).to_period("M")
        mask = np.asarray([period in months for period in month_key], dtype=bool)
        return frame.loc[mask].sort_values("date").copy()

    def _finite_values(
        frame: pd.DataFrame,
        columns: tuple[str, ...],
    ) -> list[float]:
        values: list[float] = []
        if frame is None or frame.empty:
            return values
        for column in columns:
            if column not in frame:
                continue
            arr = pd.to_numeric(
                frame[column], errors="coerce"
            ).to_numpy(dtype=float)
            values.extend(float(x) for x in arr[np.isfinite(arr)])
        return values

    def _axis_range(
        frames: tuple[pd.DataFrame, ...],
        *,
        columns: tuple[str, ...] = ("mean",),
        include_zero: bool = False,
        padding: float = 0.10,
        floor_span: float = 0.01,
    ) -> list[float] | None:
        values: list[float] = []
        for frame in frames:
            values.extend(_finite_values(frame, columns))
        finite = [x for x in values if np.isfinite(x)]
        if include_zero:
            finite.append(0.0)
        if not finite:
            return None
        lo = min(finite)
        hi = max(finite)
        span = max(
            hi - lo,
            max(abs(lo), abs(hi)) * 0.05,
            float(floor_span),
        )
        pad = float(padding) * span
        return [lo - pad, hi + pad]

    def _terminal_summary(
        frame: pd.DataFrame,
        *,
        label: str,
    ) -> str:
        if frame is None or frame.empty:
            return f"{label}: unavailable"
        block = frame.dropna(subset=["date", "mean"]).sort_values("date")
        if block.empty:
            return f"{label}: unavailable"
        row = block.iloc[-1]
        effect = float(row["mean"])

        low = pd.to_numeric(
            pd.Series([row.get("q16", np.nan)]),
            errors="coerce",
        ).iloc[0]
        high = pd.to_numeric(
            pd.Series([row.get("q84", np.nan)]),
            errors="coerce",
        ).iloc[0]

        if np.isfinite(low) and np.isfinite(high):
            zero_text = (
                "68% interval includes zero"
                if float(low) <= 0.0 <= float(high)
                else "68% interval excludes zero"
            )
            return (
                f"{label} terminal effect: {effect:+.3f} pp · "
                f"68% interval [{float(low):+.3f}, {float(high):+.3f}] · "
                f"{zero_text}"
            )
        return (
            f"{label} terminal effect: {effect:+.3f} pp · "
            "point estimate only"
        )

    def _has_interval(frame: pd.DataFrame) -> bool:
        if frame is None or frame.empty:
            return False
        if not {"q16", "q84"}.issubset(frame.columns):
            return False
        block = frame[["q16", "q84"]].apply(
            pd.to_numeric, errors="coerce"
        )
        return bool(
            np.isfinite(block["q16"].to_numpy(dtype=float)).any()
            and np.isfinite(block["q84"].to_numpy(dtype=float)).any()
        )

    def _add_band(
        fig: go.Figure,
        frame: pd.DataFrame,
        *,
        colour: str,
        label: str,
        yaxis: str | None = None,
        anchor_date: pd.Timestamp | None = None,
        anchor_value: float | None = None,
    ) -> None:
        if frame.empty or not {"q16", "q84"}.issubset(frame.columns):
            return
        block = frame.dropna(subset=["date", "q16", "q84"]).copy()
        if block.empty:
            return

        x = list(block["date"])
        lo = list(block["q16"].astype(float))
        hi = list(block["q84"].astype(float))
        horizons = [f"M+{i}" for i in range(1, len(x) + 1)]

        if anchor_date is not None and anchor_value is not None:
            x = [pd.Timestamp(anchor_date)] + x
            lo = [float(anchor_value)] + lo
            hi = [float(anchor_value)] + hi
            horizons = ["Start"] + horizons

        polygon_x = x + x[::-1]
        polygon_y = hi + lo[::-1]
        custom = [
            [horizons[i], float(lo[i]), float(hi[i])]
            for i in range(len(x))
        ]
        polygon_custom = custom + custom[::-1]

        kwargs: dict[str, Any] = {}
        if yaxis:
            kwargs["yaxis"] = yaxis

        fig.add_trace(
            go.Scatter(
                x=polygon_x,
                y=polygon_y,
                customdata=polygon_custom,
                mode="lines",
                line={"width": 0},
                fill="toself",
                fillcolor=_rgba(colour, 0.13),
                name=f"68% interval · {label}",
                showlegend=True,
                hovertemplate=(
                    "%{x|%b %Y}<br>%{customdata[0]}<br>"
                    + label
                    + " · 68% interval: [%{customdata[1]:+.3f}, "
                    "%{customdata[2]:+.3f}] pp"
                    "<extra></extra>"
                ),
                **kwargs,
            )
        )

    def _impact_trace(
        fig: go.Figure,
        frame: pd.DataFrame,
        *,
        colour: str,
        label: str,
        yaxis: str | None,
        anchor_date: pd.Timestamp | None,
        stale: bool,
    ) -> None:
        if frame.empty:
            return
        dates = list(frame["date"])
        values = list(frame["mean"].astype(float))
        horizon_labels = [f"M+{i}" for i in range(1, len(dates) + 1)]

        if anchor_date is not None:
            dates = [pd.Timestamp(anchor_date)] + dates
            values = [0.0] + values
            horizon_labels = ["Start"] + horizon_labels

        kwargs: dict[str, Any] = {}
        if yaxis:
            kwargs["yaxis"] = yaxis

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                customdata=horizon_labels,
                mode="lines+markers",
                name=label,
                line={"color": colour, "width": 2.4},
                marker={"color": colour, "size": 5},
                opacity=0.50 if stale else 1.0,
                hovertemplate=(
                    "%{x|%b %Y}<br>%{customdata}<br>%{y:+.3f} pp"
                    "<extra>" + label + "</extra>"
                ),
                **kwargs,
            )
        )

    def _last_point(
        frame: pd.DataFrame,
    ) -> tuple[pd.Timestamp, float] | None:
        if frame is None or frame.empty:
            return None
        block = frame.dropna(subset=["date", "mean"]).sort_values("date")
        if block.empty:
            return None
        row = block.iloc[-1]
        return pd.Timestamp(row["date"]), float(row["mean"])

    def _direct_label(
        fig: go.Figure,
        frame: pd.DataFrame,
        *,
        text: str,
        colour: str,
        yref: str,
        yshift: int,
    ) -> None:
        point = _last_point(frame)
        if point is None:
            return
        date, value = point
        fig.add_annotation(
            x=date,
            y=value,
            xref="x",
            yref=yref,
            text=text,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            xshift=8,
            yshift=yshift,
            font={"size": 10, "color": colour},
            bgcolor="rgba(255,255,255,0.82)",
            borderpad=2,
        )

    energy_impact = _slice(_scenario_interdomain_energy_rows(aggregate_store, "impact"))
    energy_base = _slice(_scenario_interdomain_energy_rows(aggregate_store, "baseline"))
    energy_scen = _slice(_scenario_interdomain_energy_rows(aggregate_store, "scenario"))
    headline_impact = _slice(_scenario_interdomain_headline_rows(headline_payload, "impact"))
    headline_base = _slice(_scenario_interdomain_headline_rows(headline_payload, "baseline"))
    headline_scen = _slice(_scenario_interdomain_headline_rows(headline_payload, "scenario"))
    
    # Align comparable future observations by calendar month.
    # If both domains exist, only genuinely common months are displayed.
    if mode == "impact":
        energy_months = _domain_months((energy_impact,))
        headline_months = _domain_months((headline_impact,))
    else:
        energy_months = _domain_months((energy_base, energy_scen))
        headline_months = _domain_months((headline_base, headline_scen))
    both_domains = bool(energy_months and headline_months)
    if both_domains:
        display_months = sorted(energy_months & headline_months)[:horizon]
        if not display_months:
            return empty_scenario_figure("HICP Energy and Headline future calendars do not overlap.")
    else:
        display_months = sorted(energy_months or headline_months)[:horizon]
    
    energy_impact = _restrict_months(energy_impact, display_months)
    energy_base = _restrict_months(energy_base, display_months)
    energy_scen = _restrict_months(energy_scen, display_months)
    headline_impact = _restrict_months(headline_impact, display_months)
    headline_base = _restrict_months(headline_base, display_months)
    headline_scen = _restrict_months(headline_scen, display_months)
    unique_future = [period.to_timestamp(how="start") for period in display_months]

    fig = go.Figure()

    if mode == "impact":
        # Common future months are already aligned across domains.
        # Do not add an Energy-only pre-period zero anchor.
        energy_anchor = None

        _add_band(
            fig,
            energy_impact,
            colour=energy_colour,
            label="HICP Energy",
            yaxis=None,
            anchor_date=energy_anchor,
            anchor_value=0.0 if energy_anchor is not None else None,
        )
        _impact_trace(
            fig,
            energy_impact,
            colour=energy_colour,
            label="HICP Energy impact",
            yaxis=None,
            anchor_date=energy_anchor,
            stale=energy_stale,
        )

        _add_band(
            fig,
            headline_impact,
            colour=headline_colour,
            label="Headline",
            yaxis="y2",
            anchor_date=None,
            anchor_value=None,
        )
        _impact_trace(
            fig,
            headline_impact,
            colour=headline_colour,
            label="Headline impact",
            yaxis="y2",
            anchor_date=None,
            stale=headline_stale,
        )

        if not fig.data:
            return empty_scenario_figure(
                "Calculate Headline impact to materialise the current "
                "combined Energy scenario."
            )

        energy_range = _axis_range(
            (energy_impact,),
            columns=("mean", "q16", "q84"),
            include_zero=True,
            padding=0.10,
            floor_span=0.05,
        )
        headline_range = _axis_range(
            (headline_impact,),
            columns=("mean", "q16", "q84"),
            include_zero=True,
            padding=0.10,
            floor_span=0.002,
        )

        energy_summary = _terminal_summary(
            energy_impact,
            label="HICP Energy",
        )
        headline_summary = _terminal_summary(
            headline_impact,
            label="Headline",
        )
        headline_band_note = (
            ""
            if _has_interval(headline_impact)
            else "<br>Headline shown as point estimate only."
        )

        fig = apply_theme(
            fig,
            uirevision=f"scenario-interdomain::{mode}::{horizon}",
            height=760,
            y_title=None,
        )
        fig.update_layout(
            title=None,
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="x unified",
            margin={"l": 64, "r": 52, "t": 72, "b": 118},
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.01,
                "xanchor": "left",
                "x": 0,
            },
            xaxis={
                "anchor": "y2",
                "showgrid": False,
                "tickformat": "%b %Y",
                "dtick": "M3",
                "ticks": "outside",
            },
            xaxis2={
                "overlaying": "x",
                "matches": "x",
                "anchor": "free",
                "side": "bottom",
                "position": 0.56,
                "showgrid": False,
                "showticklabels": True,
                "tickformat": "%b %Y",
                "dtick": "M3",
                "ticks": "outside",
            },
            yaxis={
                "title": "pp",
                "domain": [0.56, 1.0],
                "anchor": "x",
                "range": energy_range,
                "showgrid": True,
                "gridcolor": TOKENS["hairline_soft"],
                "zeroline": True,
                "zerolinecolor": TOKENS["hairline"],
            },
            yaxis2={
                "title": "pp",
                "domain": [0.18, 0.42],
                "anchor": "x",
                "range": headline_range,
                "showgrid": True,
                "gridcolor": TOKENS["hairline_soft"],
                "zeroline": True,
                "zerolinecolor": TOKENS["hairline"],
            },
        )

        fig.add_annotation(
            x=0,
            y=1.0,
            xref="paper",
            yref="paper",
            text="HICP Energy impact",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 11, "color": energy_colour},
        )
        fig.add_annotation(
            x=0,
            y=0.42,
            xref="paper",
            yref="paper",
            text="Headline impact",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 11, "color": headline_colour},
        )
        fig.add_shape(
            type="rect",
            x0=0.0,
            x1=1.0,
            y0=0.025,
            y1=0.125,
            xref="paper",
            yref="paper",
            line={"color": TOKENS["hairline"], "width": 1},
            fillcolor="rgba(255,255,255,1.0)",
            layer="below",
        )
        fig.add_annotation(
            x=0.015,
            y=0.075,
            xref="paper",
            yref="paper",
            text=(
                "<b>Interpretation</b> · The Headline interval reflects only "
                "Headline-BVAR uncertainty. The Energy path is fixed at its "
                "posterior mean and Energy-path uncertainty is not propagated. "
                "<b>The Energy and Headline intervals are not directly comparable.</b>"
            ),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font={"size": 10, "color": muted},
        )
        fig.add_annotation(
            x=0,
            y=-0.105,
            xref="paper",
            yref="paper",
            text=(
                energy_summary
                + "<br>"
                + headline_summary
                + headline_band_note
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 10, "color": muted},
        )

    else:
        energy_history = _scenario_interdomain_energy_history(
            aggregate_store,
            aggregate_display_store,
        )
        headline_history = _scenario_interdomain_headline_history(
            headline_payload
        )
        first = unique_future[0] if unique_future else None

        if first is not None and not energy_history.empty:
            energy_history = energy_history.loc[
                energy_history["date"] < pd.Timestamp(first)
            ].tail(18)
        if first is not None and not headline_history.empty:
            headline_history = headline_history.loc[
                headline_history["date"] < pd.Timestamp(first)
            ].tail(18)

        hmeta = dict((headline_payload or {}).get("meta") or {})
        observed_headline_date = pd.to_datetime(
            hmeta.get("headline_observed_anchor_date"),
            errors="coerce",
        )
        try:
            observed_headline_value = float(
                hmeta.get("headline_observed_anchor_yoy")
            )
        except (TypeError, ValueError):
            observed_headline_value = float("nan")
        has_headline_anchor = (
            not pd.isna(observed_headline_date)
            and np.isfinite(observed_headline_value)
        )

        def _future_with_anchor(
            future: pd.DataFrame,
            *,
            observed: pd.DataFrame | None = None,
            fallback_date: pd.Timestamp | None = None,
            fallback_value: float | None = None,
        ) -> pd.DataFrame:
            if future is None or future.empty:
                return pd.DataFrame()

            block = future[["date", "mean"]].copy()
            anchor_date = None
            anchor_value = None

            if observed is not None and not observed.empty:
                obs = observed.dropna(
                    subset=["date", "mean"]
                ).sort_values("date")
                if not obs.empty:
                    anchor_date = pd.Timestamp(obs.iloc[-1]["date"])
                    anchor_value = float(obs.iloc[-1]["mean"])
            elif (
                fallback_date is not None
                and fallback_value is not None
            ):
                anchor_date = pd.Timestamp(fallback_date)
                anchor_value = float(fallback_value)

            if anchor_date is not None and (
                block.empty
                or anchor_date < pd.Timestamp(block.iloc[0]["date"])
            ):
                block = pd.concat(
                    [
                        pd.DataFrame(
                            [{
                                "date": anchor_date,
                                "mean": anchor_value,
                            }]
                        ),
                        block,
                    ],
                    ignore_index=True,
                )
            return block

        if not energy_history.empty:
            fig.add_trace(
                go.Scatter(
                    x=energy_history["date"],
                    y=energy_history["mean"],
                    mode="lines",
                    name="HICP Energy observed",
                    line={"color": observed_colour, "width": 1.6},
                    showlegend=False,
                    hovertemplate=(
                        "%{x|%b %Y}<br>%{y:.2f}%"
                        "<extra>HICP Energy observed</extra>"
                    ),
                )
            )

        if not headline_history.empty:
            fig.add_trace(
                go.Scatter(
                    x=headline_history["date"],
                    y=headline_history["mean"],
                    mode="lines",
                    name="Headline observed",
                    line={"color": observed_colour, "width": 1.6},
                    yaxis="y2",
                    showlegend=False,
                    hovertemplate=(
                        "%{x|%b %Y}<br>%{y:.2f}%"
                        "<extra>Headline observed</extra>"
                    ),
                )
            )

        energy_base_plot = _future_with_anchor(
            energy_base,
            observed=energy_history,
        )
        energy_scen_plot = _future_with_anchor(
            energy_scen,
            observed=energy_history,
        )
        headline_base_plot = _future_with_anchor(
            headline_base,
            observed=headline_history,
            fallback_date=(
                pd.Timestamp(observed_headline_date)
                if has_headline_anchor
                else None
            ),
            fallback_value=(
                observed_headline_value
                if has_headline_anchor
                else None
            ),
        )
        headline_scen_plot = _future_with_anchor(
            headline_scen,
            observed=headline_history,
            fallback_date=(
                pd.Timestamp(observed_headline_date)
                if has_headline_anchor
                else None
            ),
            fallback_value=(
                observed_headline_value
                if has_headline_anchor
                else None
            ),
        )

        def _path_trace(
            frame: pd.DataFrame,
            *,
            colour: str,
            label: str,
            yaxis: str | None,
            dash: str,
            stale: bool,
        ) -> None:
            if frame.empty:
                return
            kwargs: dict[str, Any] = {}
            if yaxis:
                kwargs["yaxis"] = yaxis
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame["mean"],
                    mode="lines",
                    name=label,
                    line={
                        "color": colour,
                        "width": 2.4,
                        "dash": dash,
                    },
                    opacity=0.50 if stale else 1.0,
                    showlegend=False,
                    hovertemplate=(
                        "%{x|%b %Y}<br>%{y:.2f}%"
                        "<extra>" + label + "</extra>"
                    ),
                    **kwargs,
                )
            )

        _path_trace(
            energy_base_plot,
            colour=energy_colour,
            label="HICP Energy baseline",
            yaxis=None,
            dash='solid',
            stale=energy_stale,
        )
        _path_trace(
            energy_scen_plot,
            colour=energy_colour,
            label="HICP Energy scenario",
            yaxis=None,
            dash='dash',
            stale=energy_stale,
        )
        _path_trace(
            headline_base_plot,
            colour=headline_colour,
            label="Headline baseline",
            yaxis="y2",
            dash='solid',
            stale=headline_stale,
        )
        _path_trace(
            headline_scen_plot,
            colour=headline_colour,
            label="Headline scenario",
            yaxis="y2",
            dash='dash',
            stale=headline_stale,
        )

        energy_range = _axis_range(
            (
                energy_history,
                energy_base_plot,
                energy_scen_plot,
            ),
            columns=("mean",),
            include_zero=False,
            padding=0.08,
            floor_span=0.10,
        )
        headline_range = _axis_range(
            (
                headline_history,
                headline_base_plot,
                headline_scen_plot,
            ),
            columns=("mean",),
            include_zero=False,
            padding=0.08,
            floor_span=0.10,
        )

        fig = apply_theme(
            fig,
            uirevision=f"scenario-interdomain::{mode}::{horizon}",
            height=650,
            y_title=None,
        )
        fig.update_layout(
            title=None,
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="x unified",
            showlegend=False,
            margin={"l": 68, "r": 112, "t": 66, "b": 62},
            xaxis={
                "anchor": "y2",
                "showgrid": False,
                "tickformat": "%b %Y",
                "dtick": "M3",
                "ticks": "outside",
            },
            xaxis2={
                "overlaying": "x",
                "matches": "x",
                "anchor": "free",
                "side": "bottom",
                "position": 0.56,
                "showgrid": False,
                "showticklabels": True,
                "tickformat": "%b %Y",
                "dtick": "M3",
                "ticks": "outside",
            },
            yaxis={
                "title": "HICP Energy · % y/y",
                "domain": [0.56, 1.0],
                "anchor": "x",
                "range": energy_range,
                "showgrid": True,
                "gridcolor": TOKENS["hairline_soft"],
                "zeroline": True,
                "zerolinecolor": TOKENS["hairline"],
            },
            yaxis2={
                "title": "Headline HICP · % y/y",
                "domain": [0.0, 0.42],
                "anchor": "x",
                "range": headline_range,
                "showgrid": True,
                "gridcolor": TOKENS["hairline_soft"],
                "zeroline": False,
            },
        )

        if first is not None:
            fig.add_shape(
                type="line",
                x0=pd.Timestamp(first),
                x1=pd.Timestamp(first),
                xref="x",
                y0=0,
                y1=1,
                yref="paper",
                line={
                    "color": muted,
                    "width": 1,
                    "dash": "dot",
                },
            )
            fig.add_annotation(
                x=pd.Timestamp(first),
                y=1.01,
                xref="x",
                yref="paper",
                text="Forecast",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font={"size": 10, "color": muted},
            )

        _direct_label(
            fig,
            energy_base_plot,
            text="Energy baseline",
            colour=energy_colour,
            yref="y",
            yshift=-10,
        )
        _direct_label(
            fig,
            energy_scen_plot,
            text="Energy scenario",
            colour=energy_colour,
            yref="y",
            yshift=10,
        )
        _direct_label(
            fig,
            headline_base_plot,
            text="Headline baseline",
            colour=headline_colour,
            yref="y2",
            yshift=-10,
        )
        _direct_label(
            fig,
            headline_scen_plot,
            text="Headline scenario",
            colour=headline_colour,
            yref="y2",
            yshift=10,
        )

        fig.add_annotation(
            x=0,
            y=1.0,
            xref="paper",
            yref="paper",
            text="HICP Energy",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 11, "color": energy_colour},
        )
        fig.add_annotation(
            x=0,
            y=0.42,
            xref="paper",
            yref="paper",
            text="Headline HICP",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 11, "color": headline_colour},
        )

    if headline_payload is None or not headline_payload:
        fig.add_annotation(
            x=0,
            y=1.08,
            xref="paper",
            yref="paper",
            text=(
                "Headline not calculated · click Calculate Headline impact "
                "to add the Headline series."
            ),
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 10, "color": muted},
        )
    elif headline_stale:
        fig.add_annotation(
            x=0,
            y=1.08,
            xref="paper",
            yref="paper",
            text=(
                "STALE · Headline belongs to the previous Energy assumptions; "
                "recalculate to refresh it."
            ),
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 10, "color": TOKENS["warning"]},
        )
    elif energy_stale:
        fig.add_annotation(
            x=0,
            y=1.08,
            xref="paper",
            yref="paper",
            text=(
                "Current combined Energy package is not materialised · "
                "click Calculate Headline impact."
            ),
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 10, "color": TOKENS["warning"]},
        )

    return fig

@callback(
    Output("scenario-propagation-headline-result", "children"),
    Output("scenario-propagation-headline-graph", "figure"),
    Output("scenario-propagation-headline-status", "children"),
    Output('energy-propagation-results', 'style'), Input("scenario-propagation-headline-store", "data"),
    Input("agg-scenario-store", "data"),
    Input("joint-energy-scenario-store", "data"),
    Input("conditional-store", "data"),
    Input("scenario-store", "data"),
    Input("vintage-select", "value"),
    Input("conditional-agg-select", "value"),
    Input("scenario-propagation-view-mode", "value"),
    Input("scenario-propagation-display-horizon", "value"),
    Input("agg-store", "data"),
)
def render_scenario_headline_propagation(
    result_store: dict | None,
    aggregate_store: dict | None,
    joint_store: dict | None,
    conditional_store: dict | None,
    tax_store: dict | None,
    vintage: str | None,
    selected_aggregate_run_id: str | None,
    view_mode: str | None,
    display_horizon: int | None,
    aggregate_display_store: dict | None,
):
    "Presentation-only unified CURRENT Energy + Headline renderer."
    result = dict(result_store or {})
    mode = str(view_mode or "impact").strip().lower()
    if mode not in {"impact", "paths"}:
        mode = "impact"
    try:
        horizon = int(display_horizon or 12)
    except (TypeError, ValueError):
        horizon = 12
    if horizon not in {3, 6, 9, 12}:
        horizon = 12

    current_recipe = None
    current_recipe_signature = ""
    current_recipe_error = ""
    try:
        current_recipe = _joint_energy_dashboard_recipe(
            conditional_store=conditional_store,
            tax_store=tax_store,
            vintage=vintage,
            selected_aggregate_run_id=selected_aggregate_run_id,
        )
        current_recipe_signature = str(current_recipe["signature"])
    except Exception as exc:
        current_recipe_error = str(exc)

    conditional_summaries = conditional_set_summary(conditional_store)
    tax_summaries = scenario_set_summary(tax_store)
    requires_joint = bool(
        (conditional_summaries and tax_summaries)
        or len(conditional_summaries) > 1
    )

    energy_display_store, energy_stale, energy_source_label = (
        _scenario_interdomain_effective_energy_store(
            aggregate_store,
            joint_store,
            result,
            requires_joint=requires_joint,
            current_recipe_signature=current_recipe_signature,
        )
    )

    current_key = ""
    current_provenance = ""
    current_source_error = ""
    try:
        current_source = _scenario_headline_source_contract(
            energy_display_store
        )
        current_key = str(current_source["source_key"])
        current_provenance = _scenario_propagation_provenance(
            energy_display_store
        )
    except Exception as exc:
        current_source_error = str(exc)

    result_recipe_signature = str(result.get("energy_recipe_signature") or "")
    result_key = str(result.get("source_key") or "")
    headline_stale = bool(
        result
        and (
            current_recipe is None
            or (
                current_recipe_signature
                and result_recipe_signature
                and result_recipe_signature != current_recipe_signature
            )
            or (
                current_key
                and result_key
                and result_key != current_key
            )
        )
    )

    state = str(result.get("state") or "")
    payload = dict(result.get("payload") or {})
    if state in {"error", "timeout"}:
        payload = {}

    figure = _scenario_interdomain_figure(
        energy_display_store,
        payload,
        mode=mode,
        horizon=horizon,
        headline_stale=headline_stale,
        energy_stale=energy_stale,
        aggregate_display_store=aggregate_display_store,
    )

    if not result:
        card = _result_block(
            eyebrow="Headline impact",
            value="Not calculated",
            unit_date="Explicit conditional run required",
            uncertainty=(
                "All current Energy assumptions are resolved on click when required; "
                "View and Display horizon are presentation-only."
            ),
            provenance=(
                current_provenance
                or "current Energy recipe · Headline result not materialized"
            ),
        )
    elif state in {"error", "timeout"}:
        label = "Timed out" if state == "timeout" else "Calculation failed"
        card = _result_block(
            eyebrow="Headline impact",
            value=label,
            unit_date=str(result.get("computed_at_utc") or "—"),
            uncertainty=str(result.get("error") or "Unknown error"),
            provenance=(current_provenance or "Headline calculation")
            + (" · previous Energy assumptions" if headline_stale else ""),
            state="stale" if headline_stale else "fresh",
        )
    elif not payload:
        card = _result_block(
            eyebrow="Headline impact",
            value="Not calculated",
            unit_date="Stored Headline result is incomplete",
            uncertainty="No point estimate fabricated.",
            provenance=current_provenance or "Headline result incomplete",
            state="stale" if headline_stale else "fresh",
        )
    else:
        card = _scenario_headline_result_card(
            result,
            stale=headline_stale,
        )

    active_labels = list((current_recipe or {}).get("labels") or [])
    active_text = (
        " · active: " + " | ".join(str(x) for x in active_labels)
        if active_labels
        else ""
    )
    source_text = f" · Energy source: {energy_source_label}"

    if current_recipe is None:
        status = (
            "Current Energy assumptions are not buildable"
            + (f" · {current_recipe_error}" if current_recipe_error else "")
            + f" · {mode.title()} view · display H={horizon}m"
            + source_text
        )
    elif not result:
        count = int(current_recipe.get("scenario_count") or 0)
        status = (
            f"Ready · {count} active Energy assumption"
            + ("s" if count != 1 else "")
            + active_text
            + " · Headline not calculated"
            + f" · {mode.title()} view · display H={horizon}m"
            + source_text
        )
    elif headline_stale:
        status = (
            "STALE · Energy assumptions changed; previous Headline result retained "
            "for reference. Click Calculate Headline impact to refresh it"
            + active_text
            + f" · {mode.title()} view · display H={horizon}m"
            + source_text
        )
    elif state in {"error", "timeout"}:
        status = (
            f"{state.upper()} · {result.get('error') or 'Headline calculation failed'}"
            + active_text
            + f" · {mode.title()} view · display H={horizon}m"
            + source_text
        )
    elif energy_stale:
        status = (
            "STALE · current Joint Energy scenario is not materialized in the "
            "unified chart yet"
            + active_text
            + f" · {mode.title()} view · display H={horizon}m"
            + source_text
        )
    else:
        status = (
            "Current Energy + Headline scenario"
            + active_text
            + f" · {mode.title()} view · display H={horizon}m"
            + source_text
        )

    if current_source_error and not current_provenance:
        status += f" · {current_source_error}"

    return (card, figure, status, ({'marginTop': '24px', 'marginBottom': '24px'} if bool(conditional_summaries or tax_summaries) else {'marginTop': '24px', 'marginBottom': '24px', 'display': 'none'}))


@callback(
    Output("scenario-propagation-joint-vs-standalone", "children"),
    Input("scenario-propagation-headline-store", "data"),
    Input("agg-scenario-store", "data"),
    Input("joint-energy-scenario-store", "data"),
    Input("conditional-store", "data"),
    Input("scenario-store", "data"),
    Input("scenario-tax-selected-agg-store", "data"),
    Input("vintage-select", "value"),
    Input("conditional-agg-select", "value"),
    Input("scenario-propagation-display-horizon", "value"),
)
def render_scenario_joint_vs_standalone(
    result_store: dict | None,
    aggregate_store: dict | None,
    joint_store: dict | None,
    conditional_store: dict | None,
    tax_store: dict | None,
    tax_marginal_store: dict | None,
    vintage: str | None,
    selected_aggregate_run_id: str | None,
    display_horizon: int | None,
):
    """Render only already-computed posterior-mean scenario summaries."""
    data = _scenario_joint_vs_standalone_summary_data(
        result_store,
        aggregate_store,
        joint_store,
        conditional_store,
        tax_store,
        tax_marginal_store,
        vintage=vintage,
        selected_aggregate_run_id=selected_aggregate_run_id,
        display_horizon=display_horizon,
    )
    state = str(data.get("state") or "")

    if state == "not_applicable":
        return ""

    if state in {"awaiting_joint", "unavailable"}:
        return html.Span(
            str(data.get("reason") or "Joint comparison unavailable."),
            style={"color": TOKENS["muted"]},
        )

    if state == "incomplete":
        missing = list(data.get("missing") or [])
        detail = " · ".join(str(item) for item in missing)
        return [
            html.Strong("Joint vs standalone · incomplete"),
            html.Span(
                " · "
                + str(data.get("reason") or "")
                + (f" Missing: {detail}" if detail else "")
            ),
        ]

    if state != "ready":
        return ""

    date = pd.Timestamp(data["date"]).date().isoformat()
    joint_mean = float(data["joint_mean"])
    standalone_sum = float(data["standalone_sum"])
    difference = float(data["difference"])
    months = int(data.get("displayed_months") or 0)

    detail_parts = []
    for item in list(data.get("standalone") or []):
        detail_parts.append(
            f"{item['kind']} · {item['label']} {float(item['mean']):+.3f} pp"
        )

    return [
        html.Strong(
            f"Joint terminal effect: {joint_mean:+.3f} pp"
        ),
        html.Span(
            f" · Sum of standalone effects: {standalone_sum:+.3f} pp"
        ),
        html.Span(
            f" · Difference: {difference:+.3f} pp"
        ),
        html.Br(),
        html.Span(
            f"{date} · M+{months} displayed terminal · posterior mean on both sides"
        ),
        html.Br(),
        html.Span(
            "Standalone: " + " · ".join(detail_parts),
            style={"color": TOKENS["muted"]},
        ),
        html.Br(),
        html.Span(
            "Descriptive non-additivity only: standalone effects need not sum "
            "to the jointly solved scenario because Joint applies all active "
            "assumptions together; admissible/pairing subsets can also differ. "
            "This difference is not labelled as a structural interaction.",
            style={"color": TOKENS["muted"]},
        ),
    ]

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
    Output('energy-tax-results-energy', 'style'), Input("scenario-tax-selected-agg-store", "data"),
    Input("scenario-fan", "value"),
    Input("scenario-component-select", "value"),
)
def selected_tax_energy_figure(store, fan_mode, model_id):
    if not store or not model_id:
        return (empty_aggregate_figure(
            "Add/select a tax scenario to display its marginal HICP Energy effect."
        ), ({} if bool(store and model_id) else {'display': 'none'}))
    errors = dict(store.get("errors") or {})
    if model_id in errors:
        return (empty_aggregate_figure(str(errors[model_id])), ({} if bool(store and model_id) else {'display': 'none'}))
    payload = dict((store.get("by_model") or {}).get(str(model_id)) or {})
    if not payload:
        return (empty_aggregate_figure(
            "The selected component has no precomputed marginal Energy effect."
        ), ({} if bool(store and model_id) else {'display': 'none'}))
    return (tidy_aggregate_figure(
        aggregate_live_impact_figure(
            payload,
            fan_mode=fan_mode or "68",
            uirevision=(
                f"{payload.get('meta',{}).get('aggregate_run_id','agg')}::"
                f"{payload.get('meta',{}).get('selected_tax_model_id','tax')}::marginal"
            ),
        )
    ), ({} if bool(store and model_id) else {'display': 'none'}))


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
    """Latest first-future month across portable component stores, loaded once."""
    from inflation_path_portability import resolve_forecast_store_reference

    directory = Path(directory)
    key = (_registry_snapshot_id(), str(directory.resolve()))

    def _build():
        metadata = _aggregate_metadata(directory)
        stores = dict(metadata.get("component_forecast_stores", {}) or {})
        starts: list[pd.Timestamp] = []
        for raw in stores.values():
            try:
                forecast_directory = resolve_forecast_store_reference(
                    raw,
                    project_root_value=PROJECT_ROOT,
                    results_root_value=RESULTS_ROOT,
                    must_exist=True,
                )
            except Exception:
                continue
            path = forecast_directory / "forecast_metadata.json"
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

    return snapshot_get_or_build("aggregate_forecast_origin", key, _build)


def _run_ids_from_aggregate_metadata(metadata: dict) -> dict[str, str]:
    """Extract exact component run IDs independent of OS/path prefix."""
    from inflation_path_portability import run_id_from_forecast_store

    out: dict[str, str] = {}
    for key, raw in dict(metadata.get("component_forecast_stores", {}) or {}).items():
        run_id = run_id_from_forecast_store(raw)
        if run_id:
            out[str(key)] = str(run_id)
    return out


def _tax_scenarios_from_payload(payload: dict | None) -> dict[str, dict]:
    return scenario_set_to_tax_scenarios(payload)


# ---------------------------------------------------------------------------
# Aggregate page callbacks
# ---------------------------------------------------------------------------


_AGG_HEADLINE_CONTRIBUTION_STATISTIC = "mean"


def _aggregate_headline_contribution_source_contract(
    aggregate_run_id: str | None,
    vintage: str | None,
) -> dict[str, str]:
    """Resolve the exact baseline Aggregate -> Headline calculation identity."""
    aggregate_run_id = str(aggregate_run_id or "").strip()
    vintage = str(vintage or "").strip()
    if not vintage:
        raise ValueError("Select a vintage before calculating Headline contribution.")
    if not aggregate_run_id:
        raise ValueError("Select an HICP Energy aggregate run first.")

    row = _selected_aggregate_row(vintage, aggregate_run_id)
    if row is None:
        raise ValueError("The selected HICP Energy aggregate is unavailable.")
    if str(row.get("status") or "") != "complete":
        raise ValueError(
            "The selected HICP Energy aggregate is not production-complete."
        )
    aggregate_directory = Path(str(row["directory"])).resolve()
    metadata = _aggregate_metadata(aggregate_directory)
    stored_vintage = str(metadata.get("vintage") or vintage).strip()
    if stored_vintage != vintage:
        raise ValueError(
            "Selected aggregate vintage mismatch: "
            f"metadata={stored_vintage}, selected={vintage}."
        )
    stored_run = str(
        metadata.get("aggregate_run_id") or aggregate_directory.name
    ).strip()
    if stored_run != aggregate_run_id or aggregate_directory.name != aggregate_run_id:
        raise ValueError(
            "Selected aggregate directory/metadata/run identity mismatch."
        )
    forecast_name = str(metadata.get("forecast_name") or "unconditional").strip()

    headline_state = _headline_run_artifact_state(vintage)
    if str(headline_state.get("status") or "") != "COMPLETE":
        raise ValueError(
            "Headline saved-run selection is not production-complete: "
            + str(
                headline_state.get("note")
                or headline_state.get("status")
                or "unknown"
            )
        )
    headline_run_id = str(headline_state.get("run_id") or "").strip()
    headline_directory = headline_state.get("directory")
    if not headline_run_id or headline_directory is None:
        raise ValueError("Resolved Headline run has no run_id/directory.")

    source_key = "|".join(
        (
            vintage,
            aggregate_run_id,
            headline_run_id,
            forecast_name,
            SELECTED_ENERGY_BASELINE_CONTRACT_VERSION,
        )
    )
    return {
        "vintage": vintage,
        "aggregate_run_id": aggregate_run_id,
        "aggregate_directory": str(aggregate_directory),
        "forecast_name": forecast_name,
        "headline_run_id": headline_run_id,
        "headline_directory": str(Path(headline_directory).resolve()),
        "contract": SELECTED_ENERGY_BASELINE_CONTRACT_VERSION,
        "source_key": source_key,
    }


def _aggregate_headline_contribution_provenance(
    source: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> str:
    meta = dict((payload or {}).get("meta") or {})
    conditioned = int(meta.get("conditioned_horizon_months") or 0)
    free = int(meta.get("free_propagation_horizon_months") or 0)
    return (
        f"vintage {source.get('vintage', '—')} · "
        f"aggregate run {source.get('aggregate_run_id', '—')} · "
        f"headline run {source.get('headline_run_id', '—')} · "
        f"forecast {source.get('forecast_name', '—')} · "
        f"contract {source.get('contract', '—')}"
        + (f" · conditioned {conditioned}m / free {free}m" if conditioned else "")
    )


def _compute_aggregate_headline_contribution(
    aggregate_run_id: str | None,
    vintage: str | None,
) -> dict[str, Any]:
    """Run one explicit selected-Aggregate baseline Headline contribution."""
    source = _aggregate_headline_contribution_source_contract(
        aggregate_run_id, vintage
    )
    started = time.perf_counter()
    payload = run_saved_headline_selected_energy_contribution(
        source["headline_directory"],
        aggregate_directory=source["aggregate_directory"],
        source_aggregate_run_id=source["aggregate_run_id"],
        forecast_name=source["forecast_name"],
        statistic=_AGG_HEADLINE_CONTRIBUTION_STATISTIC,
        lineage={
            "dashboard_surface": "energy_aggregate_headline_contribution",
            "dashboard_source": "agg-select",
        },
        project_root=PROJECT_ROOT,
    )
    elapsed = float(time.perf_counter() - started)
    computed_at = datetime.now(timezone.utc).isoformat()

    if str(payload.get("contract") or "") != SELECTED_ENERGY_BASELINE_CONTRACT_VERSION:
        raise RuntimeError("Selected-Energy contribution contract changed.")
    meta = dict(payload.get("meta") or {})
    required = {
        "source_aggregate_run_id": source["aggregate_run_id"],
        "headline_run_id": source["headline_run_id"],
        "forecast_name": source["forecast_name"],
        "condition_statistic": _AGG_HEADLINE_CONTRIBUTION_STATISTIC,
    }
    for key, expected in required.items():
        if str(meta.get(key) or "") != str(expected):
            raise RuntimeError(
                f"Selected-Energy contribution lineage mismatch for {key}: "
                f"{meta.get(key)!r} != {expected!r}."
            )
    conditioned = int(meta.get("conditioned_horizon_months") or 0)
    computational = int(meta.get("computational_horizon") or 0)
    free = int(meta.get("free_propagation_horizon_months") or -1)
    if conditioned < 1 or computational != 12 or free != computational - conditioned:
        raise RuntimeError("Selected-Energy contribution horizon contract changed.")
    if meta.get("scenario_bridge_used") is not False:
        raise RuntimeError("Aggregate baseline contribution invoked a scenario bridge.")
    if int(meta.get("upstream_energy_draws_propagated") or 0) != 0:
        raise RuntimeError("Upstream Energy aggregate draws were unexpectedly propagated.")
    if meta.get("bvar_reestimated") is not False:
        raise RuntimeError("Selected-Energy contribution re-estimated the BVAR.")
    additivity = float(meta.get("contribution_additivity_max_abs_error") or 0.0)
    if not np.isfinite(additivity) or additivity > 1e-9:
        raise RuntimeError(
            "Selected-Energy contribution additivity gate failed: "
            f"{additivity:.3e} pp."
        )

    payload = dict(payload)
    meta.update(
        {
            "dashboard_computed_at_utc": computed_at,
            "dashboard_elapsed_seconds": elapsed,
            "dashboard_source_key": source["source_key"],
        }
    )
    payload["meta"] = meta
    return {
        "state": "fresh",
        "source_key": source["source_key"],
        "computed_at_utc": computed_at,
        "elapsed_seconds": elapsed,
        "provenance": _aggregate_headline_contribution_provenance(source, payload),
        "payload": payload,
    }


def _aggregate_headline_contribution_summaries(
    payload: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item = dict(payload or {})
    dates = pd.DatetimeIndex(pd.to_datetime(item.get("dates") or []))
    conditioned = np.asarray(item.get("conditioned") or [], dtype=bool)
    summary = dict(item.get("energy_contribution_yoy") or {})
    arrays = {
        key: np.asarray(summary.get(key) or [], dtype=float)
        for key in ("mean", "q16", "q50", "q84")
    }
    n = min(
        [len(dates), len(conditioned), *(len(values) for values in arrays.values())]
    )
    if n < 1:
        raise ValueError("Selected-Energy contribution payload has no usable path.")
    dates = dates[:n]
    conditioned = conditioned[:n]
    arrays = {key: values[:n] for key, values in arrays.items()}
    complete = np.ones(n, dtype=bool)
    for values in arrays.values():
        complete &= np.isfinite(values)
    if not complete.any():
        raise ValueError("Selected-Energy contribution payload has no finite summary.")
    conditioned_idx = np.where(complete & conditioned)[0]
    if len(conditioned_idx) < 1:
        raise ValueError("Selected-Energy contribution has no finite conditioned month.")
    terminal_idx = np.where(complete)[0][-1]
    last_conditioned_idx = conditioned_idx[-1]

    def _row(i: int) -> dict[str, Any]:
        return {
            "date": pd.Timestamp(dates[i]),
            "mean": float(arrays["mean"][i]),
            "q16": float(arrays["q16"][i]),
            "q50": float(arrays["q50"][i]),
            "q84": float(arrays["q84"][i]),
            "conditioned": bool(conditioned[i]),
            "index": int(i),
        }

    return _row(last_conditioned_idx), _row(terminal_idx)


def _aggregate_headline_contribution_card(
    result_store: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    terminal: bool,
    stale: bool,
) -> html.Div:
    result = dict(result_store or {})
    payload = dict(result.get("payload") or {})
    meta = dict(payload.get("meta") or {})
    date = pd.Timestamp(row["date"]).date().isoformat()
    elapsed = float(result.get("elapsed_seconds") or 0.0)
    upstream = meta.get("upstream_energy_draws_available")
    headline_draws = meta.get("headline_conditional_draws")
    conditioned = int(meta.get("conditioned_horizon_months") or 0)
    band_width = abs(float(row["q84"]) - float(row["q16"]))

    if terminal:
        uncertainty = (
            f"68% [{float(row['q16']):+.2f}, {float(row['q84']):+.2f}] pp · "
            f"{int(headline_draws):,} Headline draws"
            if headline_draws is not None
            else f"68% [{float(row['q16']):+.2f}, {float(row['q84']):+.2f}] pp"
        )
        uncertainty += (
            f" · Energy free after {conditioned} conditioned months"
            f" · upstream Energy uncertainty excluded"
        )
        eyebrow = "Contribution · H=12 terminal"
    else:
        if band_width <= 1e-12:
            uncertainty = (
                "Degenerate across Headline draws while Energy is conditioned"
                " · deterministic selected-aggregate posterior-mean input"
            )
        else:
            uncertainty = (
                f"68% [{float(row['q16']):+.2f}, {float(row['q84']):+.2f}] pp"
            )
        uncertainty += " · upstream Energy uncertainty excluded"
        eyebrow = "Contribution · selected Energy path"

    provenance = str(result.get("provenance") or "source · agg-select")
    if upstream is not None:
        provenance += f" · upstream Energy draws {int(upstream):,} available / 0 propagated"
    return _result_block(
        eyebrow=eyebrow,
        value=f"{float(row['mean']):+.2f} pp",
        unit_date=f"{date} · posterior mean · {elapsed:.1f} s",
        uncertainty=uncertainty,
        provenance=provenance,
        state="stale" if stale else "fresh",
    )


def _aggregate_headline_contribution_figure(
    payload: Mapping[str, Any] | None,
    *,
    source_key: str,
    stale: bool,
) -> go.Figure:
    item = dict(payload or {})
    dates = pd.DatetimeIndex(pd.to_datetime(item.get("dates") or []))
    conditioned = np.asarray(item.get("conditioned") or [], dtype=bool)
    summary = dict(item.get("energy_contribution_yoy") or {})
    meta = dict(item.get("meta") or {})
    q16 = np.asarray(summary.get("q16") or [], dtype=float)
    q50 = np.asarray(summary.get("q50") or [], dtype=float)
    q84 = np.asarray(summary.get("q84") or [], dtype=float)
    mean = np.asarray(summary.get("mean") or [], dtype=float)
    n = min(len(dates), len(conditioned), len(q16), len(q50), len(q84), len(mean))
    if n < 1:
        return empty_aggregate_figure(
            "Calculate the selected HICP Energy contribution to Headline."
        )
    dates = dates[:n]
    conditioned = conditioned[:n]
    q16, q50, q84, mean = q16[:n], q50[:n], q84[:n], mean[:n]
    if not (
        np.isfinite(q16).all()
        and np.isfinite(q50).all()
        and np.isfinite(q84).all()
        and np.isfinite(mean).all()
    ):
        return empty_aggregate_figure(
            "Selected-Energy Headline contribution contains non-finite summaries."
        )

    figure = go.Figure()
    for trace in fan_traces(
        dates,
        {"q16": q16, "q84": q84, "q50": q50},
        bands=("68",),
        name="Headline conditional",
        show_median=False,
    ):
        if str(getattr(trace, "name", "") or "") == "68% interval":
            upstream = meta.get("upstream_energy_draws_available")
            trace.name = (
                "68% Headline draws · upstream Energy uncertainty excluded"
                + (
                    f" (0/{int(upstream):,} Energy draws propagated)"
                    if upstream is not None
                    else ""
                )
            )
        figure.add_trace(trace)
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=mean,
            mode="lines+markers",
            line={"color": TOKENS["cyan"], "width": 2.2},
            marker={"size": 5},
            name="Selected HICP Energy contribution · posterior mean",
            hovertemplate="%{x|%b %Y}<br>%{y:+.3f} pp<extra></extra>",
        )
    )
    conditioned_idx = np.where(conditioned)[0]
    if len(conditioned_idx):
        boundary = pd.Timestamp(dates[conditioned_idx[-1]])
        figure.add_vline(
            x=boundary,
            line_width=1,
            line_dash="dot",
            line_color=TOKENS["hairline"],
        )
        figure.add_annotation(
            x=boundary,
            y=1.02,
            xref="x",
            yref="paper",
            text="Selected Energy conditioning ends",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 10, "color": TOKENS["muted"]},
        )
    if stale:
        figure.add_annotation(
            x=0,
            y=1.15,
            xref="paper",
            yref="paper",
            text="STALE · Aggregate/Headline source changed; recalculate.",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 10, "color": TOKENS["warning"]},
        )
    figure.update_layout(
        title="Selected HICP Energy contribution to Headline YoY",
        legend={"orientation": "h", "y": 1.10, "x": 0},
    )
    return apply_theme(
        figure,
        uirevision=f"{source_key}::selected-energy-headline-contribution",
        height=460,
        y_title="pp contribution to Headline YoY",
    )


@callback(
    Output("agg-headline-contribution-store", "data"),
    Input("agg-headline-contribution-run", "n_clicks"),
    State("agg-select", "value"),
    State("vintage-select", "value"),
    prevent_initial_call=True,
    running=[
        (
            Output("agg-headline-contribution-run", "disabled"),
            True,
            False,
        ),
        (
            Output("agg-headline-contribution-run", "children"),
            "Calculating Headline contribution…",
            "Calculate Headline contribution (~13 s)",
        ),
    ],
)
def compute_aggregate_headline_contribution(
    run_clicks: int | None,
    aggregate_run_id: str | None,
    vintage: str | None,
):
    """Explicit expensive calculation; agg-select/vintage are States only."""
    if not run_clicks:
        raise PreventUpdate
    source_key = ""
    try:
        source_key = _aggregate_headline_contribution_source_contract(
            aggregate_run_id, vintage
        )["source_key"]
        return _compute_aggregate_headline_contribution(
            aggregate_run_id, vintage
        )
    except TimeoutError as exc:
        return {
            "state": "timeout",
            "source_key": source_key,
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"Headline contribution calculation timed out: {exc}",
        }
    except Exception as exc:
        return {
            "state": "error",
            "source_key": source_key,
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }


@callback(
    Output("agg-headline-contribution-conditioned-result", "children"),
    Output("agg-headline-contribution-terminal-result", "children"),
    Output("agg-headline-contribution-graph", "figure"),
    Output("agg-headline-contribution-status", "children"),
    Input("agg-headline-contribution-store", "data"),
    Input("agg-select", "value"),
    Input("vintage-select", "value"),
    Input("registry-store", "data"),
)
def render_aggregate_headline_contribution(
    result_store: dict | None,
    aggregate_run_id: str | None,
    vintage: str | None,
    _registry: dict | None,
):
    """Render fresh/stale/error state without launching a conditional forecast."""
    try:
        source = _aggregate_headline_contribution_source_contract(
            aggregate_run_id, vintage
        )
        current_key = source["source_key"]
        current_provenance = _aggregate_headline_contribution_provenance(source)
    except Exception as exc:
        placeholder = _result_block(
            eyebrow="Contribution · selected Energy path",
            value="Not calculated",
            unit_date="Select a production-complete aggregate",
            uncertainty="No Headline conditional run is launched automatically.",
            provenance="source · agg-select · contribution unavailable",
        )
        terminal = _result_block(
            eyebrow="Contribution · H=12 terminal",
            value="Not calculated",
            unit_date="Terminal DK month",
            uncertainty="No point estimate fabricated.",
            provenance="source · agg-select · contribution unavailable",
        )
        return (
            placeholder,
            terminal,
            empty_aggregate_figure(
                "Select a production-complete HICP Energy aggregate first."
            ),
            str(exc),
        )

    result = dict(result_store or {})
    if not result:
        return (
            _result_block(
                eyebrow="Contribution · selected Energy path",
                value="Not calculated",
                unit_date="Last conditioned month",
                uncertainty=(
                    "Explicit saved-posterior conditional run required · "
                    "selected Energy input = posterior mean."
                ),
                provenance=current_provenance,
            ),
            _result_block(
                eyebrow="Contribution · H=12 terminal",
                value="Not calculated",
                unit_date="Terminal DK month",
                uncertainty=(
                    "Remaining DK months are freely propagated after the "
                    "selected Energy path ends."
                ),
                provenance=current_provenance,
            ),
            empty_aggregate_figure(
                "Calculate Headline contribution explicitly to materialize the path."
            ),
            (
                "Ready · click Calculate Headline contribution · all consecutive "
                "selected-aggregate future months will be conditioned; remaining "
                "months stay latent through H=12 · upstream Energy draws are excluded."
            ),
        )

    result_key = str(result.get("source_key") or "")
    stale = bool(result_key and result_key != current_key)
    state = str(result.get("state") or "")
    if state in {"error", "timeout"}:
        label = "Timed out" if state == "timeout" else "Calculation failed"
        provenance = current_provenance
        if stale:
            provenance += " · error belongs to a previous source selection"
        error_card = _result_block(
            eyebrow="Contribution · selected Energy path",
            value=label,
            unit_date=str(result.get("computed_at_utc") or "—"),
            uncertainty=str(result.get("error") or "Unknown error"),
            provenance=provenance,
            state="stale" if stale else "fresh",
        )
        return (
            error_card,
            _result_block(
                eyebrow="Contribution · H=12 terminal",
                value=label,
                unit_date="No valid contribution path",
                uncertainty="No point estimate fabricated.",
                provenance=provenance,
                state="stale" if stale else "fresh",
            ),
            empty_aggregate_figure(str(result.get("error") or label)),
            str(result.get("error") or label),
        )

    payload = dict(result.get("payload") or {})
    if not payload:
        return (
            _result_block(
                eyebrow="Contribution · selected Energy path",
                value="Not calculated",
                unit_date="Stored result is incomplete",
                uncertainty="No point estimate fabricated.",
                provenance=current_provenance,
            ),
            _result_block(
                eyebrow="Contribution · H=12 terminal",
                value="Not calculated",
                unit_date="Stored result is incomplete",
                uncertainty="No point estimate fabricated.",
                provenance=current_provenance,
            ),
            empty_aggregate_figure("Stored Headline contribution result is incomplete."),
            "Stored Headline contribution result is incomplete; recalculate.",
        )

    last_conditioned, terminal = _aggregate_headline_contribution_summaries(payload)
    conditioned_card = _aggregate_headline_contribution_card(
        result, row=last_conditioned, terminal=False, stale=stale
    )
    terminal_card = _aggregate_headline_contribution_card(
        result, row=terminal, terminal=True, stale=stale
    )
    figure = _aggregate_headline_contribution_figure(
        payload, source_key=result_key or current_key, stale=stale
    )
    meta = dict(payload.get("meta") or {})
    conditioned = int(meta.get("conditioned_horizon_months") or 0)
    free = int(meta.get("free_propagation_horizon_months") or 0)
    upstream = meta.get("upstream_energy_draws_available")
    status = (
        "Stale · selected aggregate or Headline source changed; previous result "
        "is shown only for reference. Recalculate explicitly."
        if stale
        else (
            f"Fresh · computed {result.get('computed_at_utc', '—')} · "
            f"{float(result.get('elapsed_seconds') or 0.0):.1f} s · "
            f"conditioned {conditioned}m / freely propagated {free}m · "
            + (
                f"0/{int(upstream):,} upstream Energy draws propagated."
                if upstream is not None
                else "upstream Energy path uncertainty not propagated."
            )
        )
    )
    return conditioned_card, terminal_card, figure, status


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
    Input("url", "pathname"),
)
def load_aggregate_display(
    aggregate_run_id: str | None,
    vintage: str | None,
    _: dict | None,
    pathname: str | None,
):
    """Read the aggregate forecast display and its independent BVAR-fit cache.

    The accounting display is the primary artefact.  Historical BVAR fitted
    paths are a diagnostic cache built from persisted posterior draws; failure
    to materialise that cache must never make the aggregate forecast unusable.
    """
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate", "/scenarios"}:
        raise PreventUpdate
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
    if (pathname or "") == "/scenarios":
        meta = display_metadata(frame)
        meta["aggregate_run_id"] = str(aggregate_run_id)
        meta["vintage"] = str(vintage)
        meta["dashboard_store_profile"] = "scenario-history-display-only-v1"
        snapshot_put_frame(aggregate_snapshot_ref, frame)
        aggregate_store = {
            "snapshot_ref": aggregate_snapshot_ref,
            "frame_json": _json_frame(frame),
            "meta": meta,
        }
        return (
            aggregate_store,
            html.Div(
                [
                    html.Strong("HICP Energy"),
                    html.Span(f" · vintage {vintage}"),
                    html.Span(f" · aggregate {_short_run(aggregate_run_id)}"),
                    html.Span(" · scenario history display only"),
                ],
                className="selection-banner",
            ),
        )

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
    tax_decomposition_error_detail = ""
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
        tax_decomposition_error_detail = f"{type(exc).__name__}: {exc}"
        tax_decomposition_meta = {
            "available": False,
            "reason": "Tax-layer contribution decomposition is unavailable for this aggregate.",
            "technical_detail": tax_decomposition_error_detail,
        }
        meta["tax_decomposition_available"] = False
        meta["tax_decomposition_reason"] = (
            "Tax-layer contribution decomposition is unavailable for this aggregate."
        )
        meta["tax_decomposition_technical_detail"] = tax_decomposition_error_detail

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
    banner_children = [
        html.Strong("HICP Energy"),
        html.Span(f" · vintage {vintage}"),
        html.Span(f" · aggregate {_short_run(aggregate_run_id)}"),
        html.Span(f" · weekly tax: {meta.get('weekly_tax_mode_effective', '—')}"),
        html.Span(" · " + " · ".join(status_bits) if status_bits else ""),
    ]
    if tax_decomposition_error_detail:
        banner_children.append(
            html.Details(
                [
                    html.Summary("Tax-layer technical details"),
                    html.Code(
                        tax_decomposition_error_detail,
                        style={
                            "whiteSpace": "pre-wrap",
                            "wordBreak": "break-all",
                        },
                    ),
                ],
                title=(
                    "Technical diagnostic details for the unavailable "
                    "tax-layer decomposition."
                ),
                style={"marginTop": "6px"},
            )
        )
    banner = html.Div(banner_children, className="selection-banner")
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
    State("url", "pathname"),
)
def aggregate_metric_dropdown(store: dict | None, current: str | None, pathname):
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
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
    State("url", "pathname"),
)
def aggregate_overlay_options(store: dict | None, current: list[str] | None, pathname):
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
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
    State("url", "pathname"),
)
def aggregate_stat_cards(store: dict | None, metric: str | None, pathname):
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
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
    State("url", "pathname"),
)
def aggregate_graph(
    store: dict | None,
    metric: str | None,
    fan_mode: str | None,
    historical_overlays: list[str] | None, pathname,
):
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
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
    State("url", "pathname"),
)
def aggregate_contributions(
    store: dict | None,
    display_mode: str | None,
    label_options: list[str] | None, pathname,
):
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
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
    State("url", "pathname"),
)
def aggregate_tax_horizon_options(
    store: dict | None, current: str | None, pathname
):
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
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
    State("url", "pathname"),
)
def aggregate_tax_contribution_chart(
    store: dict | None,
    component: str | None,
    display_mode: str | None, pathname,
):
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
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
    State("url", "pathname"),
)
def aggregate_tax_contribution_table(
    store: dict | None, horizon: str | None, pathname
):
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
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


# CONDITIONAL_ENERGY_INTERDOMAIN_DISPLAY_V1_1
def _conditional_interdomain_energy_result(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    # Reuse the existing conditional dashboard KPI contract exactly.
    values = conditional_kpis(payload)
    impact = values.get("aggregate_impact")
    terminal_date = values.get("terminal_date")

    if impact is None or terminal_date in (None, ""):
        raise ValueError(
            "Conditional HICP Energy terminal impact is unavailable. "
            "Re-run Add / update conditional once."
        )

    low = values.get("aggregate_low")
    high = values.get("aggregate_high")
    draws = values.get("n_draws")

    return {
        "date": pd.Timestamp(terminal_date).isoformat(),
        "statistic": "q50",
        "value": float(impact),
        "q16": None if low is None else float(low),
        "q84": None if high is None else float(high),
        "n_draws": None if draws is None else int(draws),
    }

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
    store["energy_result"] = _conditional_interdomain_energy_result(payload)
    # CONDITIONAL_INTERDOMAIN_SERIES_PROJECTION_A_V1
    # Presentation-only projection: these monthly rows were already computed
    # upstream by the conditional engine. Keep the terminal KPI under
    # energy_result, and expose the canonical monthly Energy rows at top level.
    for key in (
        "aggregate_history_yoy",
        "aggregate_baseline_yoy",
        "aggregate_scenario_yoy",
        "aggregate_impact_yoy",
    ):
        rows = list((payload or {}).get(key, []) or [])
        if not rows:
            raise ValueError(
                f"Conditional HICP Energy payload is missing required {key!r} rows."
            )
        store[key] = rows
    return store, None


@callback(
    Output("agg-scenario-store", "data"),
    Output("agg-scenario-banner", "children"),
    Input("scenario-store", "data"),
    Input("conditional-store", "data"),
    Input("joint-energy-scenario-store", "data"),
    State("conditional-agg-select", "value"),
    State("vintage-select", "value"),
)
def compute_live_aggregate_tax_scenario(
    scenario_store,
    conditional_store,
    joint_store,
    aggregate_run_id,
    vintage,
):
    """Materialize one canonical Energy scenario for downstream propagation.

    Single conditional and tax-only behavior is preserved. When the active
    assumptions require the Joint Energy engine (conditional + tax, or more
    than one conditional), this callback reuses the already-built Joint Energy
    payload after validating its exact dashboard input signature. It never
    recomputes the joint package here.
    """
    conditional_summaries = conditional_set_summary(conditional_store)
    tax_summaries = scenario_set_summary(scenario_store)
    requires_joint = bool(
        (conditional_summaries and tax_summaries)
        or len(conditional_summaries) > 1
    )

    if requires_joint:
        try:
            canonical, recipe, cache_state = (
                _resolve_canonical_joint_energy_scenario(
                    conditional_store=conditional_store,
                    tax_store=scenario_store,
                    vintage=vintage,
                    selected_aggregate_run_id=aggregate_run_id,
                    cached_payload=joint_store,
                    force_recompute=False,
                )
            )
        except Exception as exc:
            return None, html.Div(
                [
                    html.Strong("Joint Energy propagation failed: "),
                    html.Span(str(exc)),
                ],
                className="banner-error",
            )

        meta = dict(canonical.get("meta") or {})
        return canonical, html.Div(
            [
                html.Strong("Joint Energy scenario materialised"),
                html.Span(
                    f" · {int(meta.get('scenario_count') or recipe['scenario_count'])} "
                    f"assumptions · vintage {recipe['vintage']} · aggregate "
                    f"{_short_run(recipe['aggregate_run_id'])} · "
                    f"{int(meta.get('n_draws') or meta.get('n_aggregate_draws_paired') or 0):,} "
                    "paired aggregate draws"
                ),
                html.Span(
                    f" · resolution={cache_state} · HICP Energy ready; "
                    "Headline not calculated · BVAR re-estimation: NO"
                ),
            ],
            className="selection-banner",
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
    set_vintage = None if not scenario_store else scenario_store.get("vintage")
    if set_vintage is None:
        fp = scenario_set_payload(scenario_store, summaries[0]["model_id"])
        set_vintage = dict((fp or {}).get("meta", {}) or {}).get("vintage")
    if set_vintage is not None and str(set_vintage) != str(vintage):
        return None, (
            f"The active scenario set belongs to vintage {set_vintage}; "
            f"selected aggregate is {vintage}."
        )
    tax_scenarios = _tax_scenarios_from_payload(scenario_store)
    if not tax_scenarios:
        return None, "The active scenario set contains no non-zero VAT or excise change."
    row = _selected_aggregate_row(vintage, aggregate_run_id)
    if row is None:
        return None, "The selected aggregate is unavailable."
    directory = Path(str(row["directory"]))
    metadata = _aggregate_metadata(directory)
    run_ids = _run_ids_from_aggregate_metadata(metadata)
    if not run_ids:
        return None, (
            "This aggregate does not record its component forecast stores, so an "
            "exact live propagation cannot be reproduced."
        )
    n_requested = int(metadata.get("n_aggregate_draws_requested") or 500)
    n_draws = min(500, max(1, n_requested))
    pairing_seed = int(metadata.get("pairing_seed") or 2026)
    forecast_name = str(metadata.get("forecast_name") or "unconditional")
    set_forecast = None if not scenario_store else scenario_store.get("forecast_name")
    if set_forecast and str(set_forecast) != forecast_name:
        return None, (
            f"The active scenario set was built on {set_forecast!r}, while this "
            f"aggregate uses {forecast_name!r}."
        )
    signature = scenario_set_signature(scenario_store)
    starts = {
        str(x["label"]): x.get("scenario_start")
        for x in summaries
        if x.get("scenario_start") is not None
    }
    parsed = [pd.Timestamp(x) for x in starts.values()]
    combined_meta = {
        "scenario_start": min(parsed).isoformat() if parsed else None,
        "scenario_starts": starts,
        "scenario_components": [x["model_id"] for x in summaries],
        "scenario_count": len(summaries),
        "scenario_signature": signature,
    }
    cache_key = "agg-tax-scenario-v2::" + json.dumps(
        {
            "aggregate_run_id": str(aggregate_run_id),
            "vintage": str(vintage),
            "scenario_set": signature,
            "forecast_name": forecast_name,
            "n_draws": n_draws,
            "pairing_seed": pairing_seed,
        },
        sort_keys=True,
        default=str,
    )
    cached = _diskcache.get(cache_key)
    if (
        isinstance(cached, dict)
        and str(((cached.get("headline_bridge") or {}).get("contract")))
        == ENERGY_BRIDGE_CONTRACT_VERSION
    ):
        payload = cached
    else:
        try:
            outcome = run_aggregate(
                str(vintage),
                project_root=PROJECT_ROOT,
                results_root=RESULTS_ROOT,
                forecast_name=forecast_name,
                run_ids=run_ids,
                n_aggregate_draws=n_draws,
                pairing_seed=pairing_seed,
                tax_scenarios=tax_scenarios,
                weekly_tax_mode="strict",
                persist=False,
            )
            payload = aggregate_live_scenario_payload(outcome, combined_meta)
            payload.setdefault("meta", {}).update(
                {
                    "aggregate_run_id": str(aggregate_run_id),
                    "aggregate_forecast_name": forecast_name,
                }
            )
            payload["headline_bridge"] = energy_outcome_headline_bridge(
                outcome, combined_meta
            )
            _diskcache.set(cache_key, payload, expire=3600)
        except Exception as exc:
            return None, html.Div(
                [
                    html.Strong("HICP Energy scenario propagation failed: "),
                    html.Span(str(exc)),
                ],
                className="banner-error",
            )
    parts = [
        html.Strong(
            f"Combined scenario · {len(summaries)} component"
            + ("s" if len(summaries) != 1 else "")
        )
    ]
    for item in summaries:
        parts.append(
            html.Span(
                f" · {item['label']}: VAT {item['vat_delta_pp']:+.2f} pp, "
                f"excise {item['excise_delta']:+.3f} {item['excise_unit']}"
            )
        )
    parts.append(
        html.Span(
            f" · propagated with {payload.get('meta', {}).get('n_draws', '—')} "
            "paired aggregate draws"
        )
    )
    return payload, html.Div(parts, className="selection-banner")


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
    State("url", "pathname"),
)
def aggregate_scenario_figures(
    scenario_store: dict | None, aggregate_store: dict | None, fan_mode: str | None, pathname
):
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
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
    State("url", "pathname"),
)
def aggregate_table(store: dict | None, pathname):
    if (pathname or "") not in {"/forecast/aggregate", "/aggregate"}:
        raise PreventUpdate
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
    production_vintage_store_id="production-vintage-store",
    headline_run_resolver=_headline_run_artifact_state,
    energy_scenario_store_id="agg-scenario-store",
    energy_conditional_store_id="conditional-store",
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
    production_vintage_store_id="production-vintage-store",
    headline_run_resolver=_headline_run_artifact_state,
)

register_overview_callbacks(
    app,
    results_root=RESULTS_ROOT,
    registry_path=REGISTRY_PATH,
    project_root=PROJECT_ROOT,
    production_vintage_store_id="production-vintage-store",
    registry_store_id="registry-store",
)



if __name__ == "__main__":
    host = os.getenv("ENERGY_BVAR_DASH_HOST", "127.0.0.1")
    port = int(os.getenv("ENERGY_BVAR_DASH_PORT", "8050"))
    debug = os.getenv("ENERGY_BVAR_DASH_DEBUG", "0") in {"1", "true", "True"}
    app.run(host=host, port=port, debug=debug)
