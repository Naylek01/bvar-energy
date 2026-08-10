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
    promote_run,
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
    """
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
    for model_id in CANONICAL_MODEL_IDS:
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
        {"label": "Paper baseline", "value": "paper_baseline"},
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
        The other six models retain the paper baseline in Estimate-all-7.
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
    """HICP Energy aggregate: the only place year-on-year HICP inflation lives.

    Component runs for gas, electricity and the fuels forecast a pre-tax price,
    so their own year-on-year rate is not HICP inflation. The tax bridge, the
    weekly-to-monthly conversion and the HICP rebasing are applied only in the
    aggregation layer, and this page reads its saved output.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Aggregate", className="page-title"),
                            html.P(
                                "Draw-wise chain-linked HICP Energy, six-component "
                                "contributions and aggregation diagnostics.",
                                className="page-subtitle",
                            ),
                        ]
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
                                        options=[
                                            {
                                                "label": "Show six-model reconstruction",
                                                "value": "components",
                                            },
                                            {
                                                "label": "Show BVAR model reconstruction",
                                                "value": "bvar_reconstruction",
                                            },
                                        ],
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
                className="page-heading-row",
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
                [dcc.Loading(dcc.Graph(id="agg-graph", config=_GRAPH_CONFIG), type="circle")],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Active component tax scenarios", className="panel-title"),
                                    html.P(
                                        "The active VAT/excise scenario set is propagated jointly, draw-by-draw, through the selected HICP Energy aggregate. No BVAR is re-estimated.",
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
                                style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "16px"},
                            ),
                        ]
                    )
                ],
                className="panel chart-panel",
            ),
            html.Div(
                [dcc.Loading(dcc.Graph(id="agg-contrib", config=_GRAPH_CONFIG), type="circle")],
                className="panel chart-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H3("Aggregation provenance", className="panel-title")],
                        className="panel-heading",
                    ),
                    dash_table.DataTable(
                        id="agg-table",
                        columns=[{"name": "Item", "id": "Item"}, {"name": "Value", "id": "Value"}],
                        data=[],
                        page_size=12,
                        style_as_list_view=True,
                        style_cell={
                            "fontFamily": "Inter, Segoe UI, sans-serif",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                    ),
                ],
                className="panel table-panel",
            ),
        ],
        className="page-body",
    )



def scenario_page() -> html.Div:
    """Multi-component ex-post VAT/excise scenario builder."""
    controls = html.Div([
        html.Div([html.Label("Component", className="control-label"), dcc.Dropdown(id="scenario-component-select", options=[], value=None, clearable=False, className="compact-dropdown")], className="control-block"),
        html.Div([html.Label("Scenario start", className="control-label"), dcc.DatePickerSingle(id="scenario-start-date", display_format="YYYY-MM-DD", first_day_of_week=1, clearable=False)], className="control-block"),
        html.Div([html.Label("VAT change (pp)", className="control-label"), dcc.Input(id="scenario-vat-delta", type="number", value=0.0, step=0.1, debounce=0.4, className="compact-dropdown")], className="control-block"),
        html.Div([html.Label(id="scenario-excise-label", children="Excise change", className="control-label"), dcc.Input(id="scenario-excise-delta", type="number", value=0.0, step=0.1, debounce=0.4, className="compact-dropdown")], className="control-block"),
        html.Div([html.Label("Fan", className="control-label"), dcc.RadioItems(id="scenario-fan", options=[{"label":"68%","value":"68"},{"label":"90%","value":"90"},{"label":"Both","value":"both"}], value="68", inline=True, className="fan-radio")], className="control-block"),
        html.Button("Add / update", id="scenario-apply", n_clicks=0, className="refresh-button"),
        html.Button("Remove", id="scenario-remove", n_clicks=0, className="refresh-button"),
        html.Button("Reset all", id="scenario-reset-all", n_clicks=0, className="refresh-button"),
    ], className="chart-controls")
    return html.Div([
        html.Div([html.Div([html.H2("Scenarios", className="page-title"), html.P("Build several VAT/excise shocks on different energy components. Saved BVAR draws remain fixed; the combined set is propagated draw-by-draw to HICP Energy.", className="page-subtitle")]), controls], className="page-heading-row"),
        html.Div(id="scenario-support-note", className="selection-banner"), html.Div(id="scenario-banner"),
        html.Div([html.Div([html.Div("Active scenario set", className="eyebrow"), html.Div(id="scenario-set-summary")], className="panel", style={"padding":"16px 18px","marginBottom":"16px"})]),
        html.Div([_stat_card("Baseline terminal YoY","scenario-baseline-terminal","scenario-terminal-date"), _stat_card("Scenario terminal YoY","scenario-scenario-terminal","scenario-terminal-date-2"), _stat_card("Terminal tax impact","scenario-impact-terminal","scenario-impact-interval"), _stat_card("Paired posterior draws","scenario-draws","scenario-draws-note")], className="stats-grid"),
        html.Div([dcc.Loading(dcc.Graph(id="scenario-main-graph", config=_GRAPH_CONFIG), type="circle")], className="panel chart-panel"),
        html.Div([html.Div([dcc.Loading(dcc.Graph(id="scenario-level-impact", config=_GRAPH_CONFIG), type="circle")], className="panel chart-panel"), html.Div([dcc.Loading(dcc.Graph(id="scenario-yoy-impact", config=_GRAPH_CONFIG), type="circle")], className="panel chart-panel")], style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(0, 1fr))","gap":"16px"}),
        html.Div([dcc.Loading(dcc.Graph(id="scenario-tax-graph", config=_GRAPH_CONFIG), type="circle")], className="panel chart-panel"),
    ], className="page-body")


def estimation_page() -> html.Div:
    model_options = [
        {"label": model_spec(model_id).label, "value": model_id}
        for model_id in CANONICAL_MODEL_IDS
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Estimation", className="page-title"),
                            html.P(
                                "Production BVAR estimation from processed vintages. Full posterior draws are always persisted so Structural analysis remains available.",
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
                                        "Paper baseline is immutable. Choose Custom or a saved profile to change priors and MCMC settings; every effective configuration enters the deterministic run identity.",
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
                                "A global lock prevents two estimations from running at the same time. An identical complete run is reused instead of recomputed.",
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
                                        "Estimate all 7 models",
                                        id="estimation-run-all",
                                        n_clicks=0,
                                        className="refresh-button estimation-run-all-button",
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
                    _nav_link("Aggregate", "/aggregate", "Σ"),
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
        dcc.Store(id="agg-scenario-store", storage_type="memory"),
        dcc.Store(id="estimation-result-store", storage_type="memory"),
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
                            placeholder_page(
                                "Structural analysis",
                                "Impulse responses, FEVD and historical decomposition using saved Gibbs draws.",
                                "Next: recursive/sign identification, reference date and shock-unit controls.",
                            ),
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
    if global_model in CANONICAL_MODEL_IDS:
        return global_model
    return current if current in CANONICAL_MODEL_IDS else "gas"


@callback(
    Output("est-vintage-select", "options"),
    Output("est-vintage-select", "value"),
    Input("est-model-select", "value"),
    State("vintage-select", "value"),
    State("est-vintage-select", "value"),
)
def estimation_vintage_options(model_id, global_vintage, current):
    if not model_id:
        return [], None
    try:
        vintages = available_vintages(model_id, project_root=PROJECT_ROOT)
    except Exception:
        return [], None
    vintages = sorted(map(str, vintages), reverse=True)
    options = [{"label": value, "value": value} for value in vintages]
    if global_vintage in vintages:
        value = global_vintage
    elif current in vintages:
        value = current
    else:
        value = vintages[0] if vintages else None
    return options, value



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
        message = "Paper baseline loaded · controls are locked until you choose Custom or a saved profile."
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
            "Paper baseline"
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
            models=CANONICAL_MODEL_IDS,
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
            missing_models = [model_spec(name).label for name in CANONICAL_MODEL_IDS]
            ready = 0

        if ready == len(CANONICAL_MODEL_IDS):
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


def _suite_progress_payload(
    vintage: str,
    statuses: dict[str, dict],
    *,
    current_model: str | None = None,
) -> dict:
    """Serializable live status for the seven sequential component runs."""
    models = []
    for model_id in CANONICAL_MODEL_IDS:
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
) -> dict:
    models = []
    for model_id in CANONICAL_MODEL_IDS:
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
    for model_id in CANONICAL_MODEL_IDS:
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
    model_id,
    vintage,
    promote_values,
    config_store,
):
    trigger = ctx.triggered_id
    if trigger not in {"estimation-run", "estimation-run-all"}:
        raise PreventUpdate
    if not vintage:
        raise PreventUpdate
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

    started = time.monotonic()
    promote_after = "promote" in (promote_values or [])

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
            models=CANONICAL_MODEL_IDS,
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
            total_models = len(CANONICAL_MODEL_IDS)

            for model_index, suite_model_id in enumerate(CANONICAL_MODEL_IDS):
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

            # Only after all seven components are complete do we update the
            # registry and, optionally, promote the coherent suite. This avoids
            # intentionally promoting a partial estimation round.
            _push_estimation_progress(
                set_progress,
                91,
                "Registering suite",
                "Scanning all component stores and validating forecast provenance.",
                _suite_progress_payload(common_vintage, statuses),
            )
            scan_results(results_root=RESULTS_ROOT, registry_path=REGISTRY_PATH)

            if promote_after:
                _push_estimation_progress(
                    set_progress,
                    96,
                    "Promoting suite",
                    "Promoting all seven completed deterministic runs.",
                    _suite_progress_payload(common_vintage, statuses),
                )
                for suite_model_id in CANONICAL_MODEL_IDS:
                    promote_run(
                        REGISTRY_PATH,
                        suite_model_id,
                        common_vintage,
                        statuses[suite_model_id]["run_id"],
                        required_forecast="unconditional",
                        require_draws=True,
                    )

            estimated = sum(
                item.get("status") == "complete" and not item.get("reused")
                for item in statuses.values()
            )
            reused = sum(
                item.get("status") == "complete" and item.get("reused")
                for item in statuses.values()
            )
            _push_estimation_progress(
                set_progress,
                100,
                "Seven-model suite complete",
                f"{estimated} estimated · {reused} reused · "
                + ("all seven promoted" if promote_after else "promotion unchanged"),
                _suite_progress_payload(common_vintage, statuses),
            )
            return _estimation_suite_result_payload(
                ok=True,
                status="complete",
                vintage=common_vintage,
                message=(
                    f"Seven-model suite complete: {estimated} estimated, "
                    f"{reused} reused."
                ),
                elapsed=time.monotonic() - started,
                statuses=statuses,
                promoted=promote_after,
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
    for item in store.get("models", []):
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
                        f"Common vintage {store.get('vintage', '—')} · sequential execution",
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

    banner = html.Div(
        [
            html.Strong(
                "Completed: " if ok and not is_suite
                else "Suite completed: " if ok
                else "Estimation failed: " if not is_suite
                else "Suite failed: "
            ),
            html.Span(message),
        ],
        className="estimation-success-banner" if ok else "estimation-error-banner",
    )
    elapsed = float(store.get("elapsed_seconds", 0.0) or 0.0)

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
        return (
            banner,
            status.title(),
            f"{completed}/7 complete · {estimated} estimated · {reused} reused",
            f"{completed} / 7",
            "7 PROMOTED" if store.get("promoted") and completed == 7 else "not promoted as suite",
            draws_text,
            "each completed run has mandatory draws.npz",
            f"{elapsed / 60.0:.1f} min" if elapsed >= 60 else f"{elapsed:.1f} s",
            f"{store.get('vintage', '')} · {store.get('directory', '')}",
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
    if not vintage or not forecast_name: return [], None
    available=[]; options=[]
    for model_id in TAX_SCENARIO_MODEL_IDS:
        if _scenario_component_forecast_row(vintage, model_id, forecast_name) is not None:
            available.append(model_id); options.append({"label":model_spec(model_id).label,"value":model_id})
    return options, (current if current in available else (available[0] if available else None))


@callback(Output("scenario-start-date","date"), Output("scenario-start-date","min_date_allowed"), Output("scenario-start-date","max_date_allowed"), Output("scenario-vat-delta","value"), Output("scenario-excise-delta","value"), Output("scenario-excise-label","children"), Output("scenario-support-note","children"), Output("scenario-start-date","disabled"), Output("scenario-vat-delta","disabled"), Output("scenario-excise-delta","disabled"), Input("scenario-component-select","value"), Input("vintage-select","value"), Input("forecast-select","value"), Input("registry-store","data"), Input("scenario-store","data"))
def scenario_control_defaults(model_id, vintage, forecast_name, _, scenario_store):
    if not model_id or not vintage or not forecast_name:
        return None,None,None,0.0,0.0,"Excise change","No scenario-capable component forecast is available.",True,True,True
    row=_scenario_component_forecast_row(vintage,model_id,forecast_name)
    if row is None:
        return None,None,None,0.0,0.0,"Excise change",f"{model_spec(model_id).label}: no unique saved forecast. Promote one run if several coexist.",True,True,True
    try:
        from energy_bvar_io import load_energy_bvar_forecast
        forecast=load_energy_bvar_forecast(Path(str(row["directory"])))
        dataset=PROJECT_ROOT/"data"/"processed"/str(vintage)/model_spec(model_id).dataset_file
        contract=component_tax_scenario_contract(model_id,forecast,dataset_path=dataset)
    except Exception as exc:
        return None,None,None,0.0,0.0,"Excise change",f"Scenario controls unavailable: {exc}",True,True,True
    existing=scenario_set_payload(scenario_store,model_id); em=dict((existing or {}).get("meta",{}) or {})
    unit=str(contract.get("excise_unit","source unit"))
    min_start=pd.Timestamp(contract["min_start"])
    max_start=pd.Timestamp(contract["max_start"])
    start=pd.Timestamp(em.get("scenario_start") or contract["default_start"])
    if start < min_start or start > max_start:
        start=pd.Timestamp(contract["default_start"])
    freq_label="monthly periods" if str(contract.get("frequency"))=="monthly" else "weekly periods (Monday)"
    note=(
        f"{model_spec(model_id).label} · run {str(row['run_id'])[:12]} · "
        f"available scenario window {min_start.date().isoformat()} — {max_start.date().isoformat()} "
        f"({freq_label}) · baseline at first future period: VAT "
        f"{contract['baseline_vat_at_start']:.2f}% · excise "
        f"{contract['baseline_excise_at_start']:.3f} {unit}. "
        "Changes are additive to the baseline tax path."
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
        requested_start=pd.Timestamp(start_date)
        if frequency=="weekly":
            effective_start=requested_start.to_period("W-SUN").start_time
        else:
            effective_start=requested_start.to_period("M").to_timestamp(how="start")
        min_start=pd.Timestamp(contract["min_start"])
        max_start=pd.Timestamp(contract["max_start"])
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
                    "Nothing was found under results/hicp_energy_aggregate/. Run "
                    "notebook 10 or run_aggregate, then press Refresh."
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
    """Read one aggregate display artefact, materialising it if needed."""
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
    materialised = False
    try:
        if not display_path.is_file():
            if not AUTO_BUILD_DISPLAY:
                raise FileNotFoundError(
                    f"{display_path} is missing and ENERGY_BVAR_AUTO_BUILD_DISPLAY=0."
                )
            build_aggregate_display(directory, project_root=PROJECT_ROOT)
            materialised = True
        frame = load_display_artifact(display_path)
        if not (frame["record_type"].astype(str) == "fan").any():
            raise ValueError(
                f"{DISPLAY_FILENAME} loaded but carries no aggregate fan rows."
            )

        # Upgrade display artefacts created before the true posterior BVAR-fitted
        # contract.  A negative availability result is persisted in metadata so
        # legacy runs without posterior B draws are not retried on every load.
        current_meta = display_metadata(frame)
        needs_fitted_upgrade = "bvar_fitted_available" not in current_meta
        if current_meta.get("bvar_fitted_available") is False:
            # Displays built by the first fitted-reconstruction patch can carry
            # a false-negative caused only by the petrol/diesel metadata-key
            # mismatch (aggregate keys ``petrol``/``diesel`` versus canonical
            # run ids ``car_fuels_petrol``/``car_fuels_diesel``). Rebuild those
            # artefacts once so the backward-compatible resolver can retry.
            reason = str(current_meta.get("bvar_fitted_reason") or "")
            resolver_contract_upgraded = (
                "Aggregate metadata does not record forecast stores" in reason
            )

            # A same-run re-estimation with the updated IO layer can add the
            # compact fit cache later without changing the deterministic run_id.
            # Detect that transition so an old "unavailable" display is rebuilt.
            try:
                aggregate_meta = json.loads(
                    (directory / "metadata.json").read_text(encoding="utf-8")
                )
                stores = dict(aggregate_meta.get("component_forecast_stores", {}))
                fit_ready = bool(stores) and all(
                    (Path(str(path)).parent.parent / "fit_draws.npz").is_file()
                    or (Path(str(path)).parent.parent / "draws.npz").is_file()
                    for path in stores.values()
                )
            except Exception:
                fit_ready = False
            needs_fitted_upgrade = resolver_contract_upgraded or fit_ready
        if needs_fitted_upgrade and AUTO_BUILD_DISPLAY:
            build_aggregate_display(
                directory, project_root=PROJECT_ROOT, overwrite=True
            )
            frame = load_display_artifact(display_path)
            materialised = True
    except Exception as exc:
        return None, html.Div(
            [html.Strong("Aggregate load failed: "), html.Span(str(exc))],
            className="banner-error",
        )

    meta = display_metadata(frame)
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
            meta["last_observed_hicp_energy"] = pd.Timestamp(history["date"].max()).isoformat()
    banner = html.Div(
        [
            html.Strong("HICP Energy"),
            html.Span(f" · vintage {vintage}"),
            html.Span(f" · aggregate {_short_run(aggregate_run_id)}"),
            html.Span(f" · weekly tax: {meta.get('weekly_tax_mode_effective', '—')}"),
            html.Span(" · display_v1 materialised now" if materialised else ""),
        ],
        className="selection-banner",
    )
    return {"frame_json": _json_frame(frame), "meta": meta}, banner


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
    meta = (store or {}).get("meta", {})
    revision = f"{meta.get('aggregate_run_id', 'agg')}::{metric}"
    overlays = set(historical_overlays or [])
    return aggregate_fan_figure(
        frame, metric=metric, fan_mode=fan_mode or "68", uirevision=revision,
        forecast_origin=meta.get("forecast_origin"),
        last_observed=meta.get("last_observed_hicp_energy"),
        show_six_model_components="components" in overlays,
        show_bvar_reconstruction="bvar_reconstruction" in overlays,
    )


@callback(
    Output("agg-contrib", "figure"),
    Input("agg-store", "data"),
)
def aggregate_contributions(store: dict | None):
    frame = _frame_from_store(store)
    if frame.empty:
        return empty_aggregate_figure("Select a saved aggregate run")
    meta = (store or {}).get("meta", {})
    revision = f"{meta.get('aggregate_run_id', 'agg')}::contrib"
    return aggregate_contribution_figure(
        frame, basis="baseline", uirevision=revision,
        forecast_origin=meta.get("forecast_origin"),
        last_observed=meta.get("last_observed_hicp_energy"),
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
    return (
        aggregate_live_scenario_figure(
            scenario_store, frame, fan_mode=fan_mode or "68",
            forecast_origin=meta.get("forecast_origin"),
            uirevision=revision + "::main",
        ),
        aggregate_live_impact_figure(
            scenario_store, fan_mode=fan_mode or "68", uirevision=revision + "::impact"
        ),
        aggregate_live_contribution_impact_figure(
            scenario_store, uirevision=revision + "::contrib"
        ),
    )


@callback(
    Output("agg-table", "data"),
    Input("agg-store", "data"),
)
def aggregate_table(store: dict | None):
    frame = _frame_from_store(store)
    if frame.empty:
        return []
    return aggregate_diagnostics_table(frame).to_dict("records")


if __name__ == "__main__":
    host = os.getenv("ENERGY_BVAR_DASH_HOST", "127.0.0.1")
    port = int(os.getenv("ENERGY_BVAR_DASH_PORT", "8050"))
    debug = os.getenv("ENERGY_BVAR_DASH_DEBUG", "0") in {"1", "true", "True"}
    app.run(host=host, port=port, debug=debug)
