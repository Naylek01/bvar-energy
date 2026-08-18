"""Declarative orchestration layer for the seven energy BVAR components.

This module holds the per-model constants that were previously scattered across
notebooks 03-09, so that a notebook run and a dashboard run are byte-identical:
same ``model_id``, same ``vintage``, same config hash, therefore same ``run_id``.

Design rules
------------
1. Nothing econometric lives here. Variables, targets, units and dataset file
   names are delegated to the existing spec functions
   (:func:`monthly_hicp_component_spec`, :func:`weekly_fuel_spec`) wherever they
   exist. They are duplicated here only for ``gas`` and ``electricity``, which
   have no spec function in the library -- that duplication is a known gap, not
   a design choice.
2. ``model_id`` is the on-disk directory name under ``results/`` and
   ``PROCESSED_ROOT``. It is NOT always the short name a human would use:
   petrol -> ``car_fuels_petrol``, diesel -> ``car_fuels_diesel``. Short aliases
   are resolved by :func:`resolve_model_id`.
3. Vintage resolution is centralised. Each notebook independently took
   ``candidates[-1]`` for its own dataset, which silently permits a mixed-vintage
   estimation round. :func:`resolve_common_vintage` forbids that.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from energy_bvar_monthly_hicp import (
    load_monthly_hicp_component_panel,
    monthly_hicp_component_spec,
)
from energy_bvar_weekly_fuels import load_weekly_fuel_panel, weekly_fuel_spec
from energy_bvar_gas import load_gas_panel
from energy_bvar_electricity import (
    load_electricity_panel,
    make_electricity_seasonal_dummies,
)

__all__ = [
    "MODEL_SPECS",
    "CANONICAL_MODEL_IDS",
    "PipelineError",
    "VintageError",
    "ModelSpec",
    "Panel",
    "RunRef",
    "RunOutcome",
    "SuiteOutcome",
    "model_spec",
    "model_contract",
    "resolve_model_id",
    "find_project_root",
    "available_vintages",
    "resolve_vintage",
    "resolve_common_vintage",
    "vintage_coverage",
    "build_panel",
    "forecast_horizon",
    "forecast_draws",
    "discover_runs",
    "resolve_run",
    "resolve_forecast_stores",
    "planned_run_metadata",
    "run_component",
    "run_all_components",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PipelineError(RuntimeError):
    """Base class for orchestration failures."""


class VintageError(PipelineError):
    """Raised when a vintage is missing, incomplete or ambiguous."""


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------


def find_project_root(start: Path | str | None = None) -> Path:
    """Walk upwards until a directory containing ``data`` or ``src`` is found.

    Mirrors the helper used at the top of every notebook so that the pipeline
    and the notebooks agree on ``PROJECT_ROOT`` without importing each other.
    """
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() or (candidate / "src").exists():
            return candidate
    return start


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to turn a (model_id, vintage) pair into an estimation.

    Attributes
    ----------
    model_id
        On-disk directory name under ``results/``. Authoritative.
    aliases
        Human-facing short names accepted by :func:`resolve_model_id`.
    family
        Dispatch key for :func:`build_panel`. One of ``monthly_hicp``,
        ``weekly_fuel``, ``gas``, ``electricity``.
    dataset_file
        File name inside ``data/processed/<vintage>/``.
    spec_key
        Argument passed to the library spec function, when one exists.
    p
        Lag order. 12 for monthly models, 24 for weekly models.
    horizon_rule
        ``to_year_end`` (monthly: ``max(3, months to 31 December)``) or
        ``fixed`` (weekly: :attr:`horizon_fixed` periods).
    exog_prior_scale
        Only meaningful when the model carries deterministic regressors.
    """

    model_id: str
    aggregate_key: str
    aliases: tuple[str, ...]
    family: str
    dataset_file: str
    label: str
    frequency: str
    p: int
    horizon_rule: str
    horizon_fixed: int | None = None
    horizon_floor: int = 3
    spec_key: str | None = None
    seed: int = 42
    missing_data_method: str = "linear"
    simulate_future_outliers: bool = True
    code_version: str = "energy_bvar_model-v2"
    forecast_seed: int = 2026
    forecast_draws_rule: str = "min_1000"
    forecast_draws: int = 500
    forecast_draws_cap: int = 1_000
    structural_horizon: int = 12
    exog_prior_scale: float | None = None
    reference_month: int | None = None
    # Only populated for models with no library spec function (gas, electricity).
    inline_variables: tuple[str, ...] = ()
    inline_units: Mapping[str, str] = field(default_factory=dict)
    inline_target: str | None = None
    notes: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "gas": ModelSpec(
        model_id="gas",
        aggregate_key="gas",
        aliases=("gas",),
        family="gas",
        dataset_file="gas_monthly.csv",
        label="Natural gas",
        frequency="monthly",
        p=12,
        horizon_rule="to_year_end",
        inline_variables=("natural_gas_wholesale", "gas_pre_tax"),
        inline_units={
            "natural_gas_wholesale": "EUR/MWh",
            "gas_pre_tax": "source unit",
        },
        inline_target="gas_pre_tax",
        notes=(
            "No library spec function exists for gas; the variable list is "
            "duplicated here from notebook 03. The constructed pre-tax price "
            "unit is resolved from the processed-vintage manifest at runtime."
        ),
    ),
    "electricity": ModelSpec(
        model_id="electricity",
        aggregate_key="electricity",
        aliases=("electricity",),
        family="electricity",
        dataset_file="electricity_monthly.csv",
        label="Electricity",
        frequency="monthly",
        p=12,
        horizon_rule="to_year_end",
        exog_prior_scale=10.0,
        reference_month=1,
        inline_variables=("natural_gas_wholesale", "electricity_pre_tax"),
        inline_units={
            "natural_gas_wholesale": "EUR/MWh",
            "electricity_pre_tax": "source unit",
        },
        inline_target="electricity_pre_tax",
        notes=(
            "Carries eleven monthly seasonal dummies as deterministic "
            "regressors, with January as the reference month."
        ),
    ),
    "heat_energy": ModelSpec(
        model_id="heat_energy",
        aggregate_key="heat_energy",
        aliases=("heat_energy", "heat"),
        family="monthly_hicp",
        dataset_file="heat_energy_monthly.csv",
        label="Heat energy",
        frequency="monthly",
        p=12,
        horizon_rule="to_year_end",
        spec_key="heat_energy",
    ),
    "solid_fuels": ModelSpec(
        model_id="solid_fuels",
        aggregate_key="solid_fuels",
        aliases=("solid_fuels", "solid"),
        family="monthly_hicp",
        dataset_file="solid_fuels_monthly.csv",
        label="Solid fuels",
        frequency="monthly",
        p=12,
        horizon_rule="to_year_end",
        spec_key="solid_fuels",
    ),
    "car_fuels_petrol": ModelSpec(
        model_id="car_fuels_petrol",
        aggregate_key="petrol",
        aliases=("petrol", "car_fuels_petrol", "gasoline"),
        family="weekly_fuel",
        dataset_file="car_fuels_weekly.csv",
        label="Car fuels - petrol",
        frequency="weekly",
        p=24,
        horizon_rule="fixed",
        horizon_fixed=26,
        spec_key="petrol",
        forecast_seed=123,
        forecast_draws_rule="fixed",
        forecast_draws=500,
        missing_data_method="dk",
    ),
    "car_fuels_diesel": ModelSpec(
        model_id="car_fuels_diesel",
        aggregate_key="diesel",
        aliases=("diesel", "car_fuels_diesel"),
        family="weekly_fuel",
        dataset_file="car_fuels_weekly.csv",
        label="Car fuels - diesel",
        frequency="weekly",
        p=24,
        horizon_rule="fixed",
        horizon_fixed=26,
        spec_key="diesel",
        forecast_seed=123,
        forecast_draws_rule="fixed",
        forecast_draws=500,
        missing_data_method="dk",
    ),
    "liquid_fuels": ModelSpec(
        model_id="liquid_fuels",
        aggregate_key="liquid_fuels",
        aliases=("liquid_fuels", "heating_oil"),
        family="weekly_fuel",
        dataset_file="liquid_fuels_weekly.csv",
        label="Liquid fuels - heating oil",
        frequency="weekly",
        p=24,
        horizon_rule="fixed",
        horizon_fixed=26,
        spec_key="liquid_fuels",
        forecast_seed=123,
        forecast_draws_rule="fixed",
        forecast_draws=500,
        missing_data_method="dk",
    ),
}


CANONICAL_MODEL_IDS: tuple[str, ...] = tuple(MODEL_SPECS)


_ALIAS_INDEX: dict[str, str] = {}
for _model_id, _spec in MODEL_SPECS.items():
    for _alias in (_model_id, *_spec.aliases):
        _key = _alias.strip().lower()
        if _key in _ALIAS_INDEX and _ALIAS_INDEX[_key] != _model_id:
            raise PipelineError(
                f"Alias {_alias!r} is claimed by both {_ALIAS_INDEX[_key]!r} "
                f"and {_model_id!r}."
            )
        _ALIAS_INDEX[_key] = _model_id
del _model_id, _spec, _alias, _key


def resolve_model_id(name: str) -> str:
    """Map a short alias (``petrol``) to the on-disk ``model_id``."""
    key = str(name).strip().lower()
    if key not in _ALIAS_INDEX:
        raise PipelineError(
            f"Unknown model {name!r}. Known ids: {CANONICAL_MODEL_IDS}; "
            f"known aliases: {sorted(_ALIAS_INDEX)}."
        )
    return _ALIAS_INDEX[key]


def model_spec(name: str, **overrides) -> ModelSpec:
    """Return the specification for one model, optionally with overrides.

    Overrides are how the dashboard's hyperparameter panel varies a run without
    mutating the canonical table. Because every override feeds the estimation
    config, it also changes the resulting ``run_id``.
    """
    spec = MODEL_SPECS[resolve_model_id(name)]
    if not overrides:
        return spec
    unknown = set(overrides).difference(spec.__dataclass_fields__)
    if unknown:
        raise PipelineError(f"Unknown ModelSpec fields: {sorted(unknown)}.")
    return replace(spec, **overrides)


# ---------------------------------------------------------------------------
# Vintage resolution
# ---------------------------------------------------------------------------


def _processed_root(project_root: Path | str | None = None) -> Path:
    root = find_project_root() if project_root is None else Path(project_root)
    return root / "data" / "processed"


def available_vintages(
    name: str,
    *,
    project_root: Path | str | None = None,
) -> list[str]:
    """Sorted vintage directory names that contain this model's dataset."""
    spec = model_spec(name)
    processed = _processed_root(project_root)
    if not processed.is_dir():
        raise VintageError(f"Processed root does not exist: {processed}")
    return sorted(
        path.parent.name
        for path in processed.glob(f"*/{spec.dataset_file}")
        if path.is_file()
    )


def resolve_vintage(
    name: str,
    vintage: str | None = None,
    *,
    project_root: Path | str | None = None,
) -> str:
    """Resolve one model's vintage, defaulting to the most recent available.

    Reproduces the notebooks' ``candidates[-1]`` behaviour, but raises instead
    of returning silently when the requested vintage has no dataset.
    """
    spec = model_spec(name)
    processed = _processed_root(project_root)
    if vintage is not None:
        dataset = processed / str(vintage) / spec.dataset_file
        if not dataset.is_file():
            raise VintageError(
                f"{spec.model_id}: dataset not found for vintage "
                f"{vintage!r} at {dataset}."
            )
        return str(vintage)
    candidates = available_vintages(name, project_root=project_root)
    if not candidates:
        raise VintageError(
            f"{spec.model_id}: no {spec.dataset_file} found below {processed}."
        )
    return candidates[-1]


def vintage_coverage(
    *,
    models: Sequence[str] = CANONICAL_MODEL_IDS,
    project_root: Path | str | None = None,
) -> pd.DataFrame:
    """Boolean matrix of vintage x model dataset availability."""
    per_model = {
        resolve_model_id(name): set(
            available_vintages(name, project_root=project_root)
        )
        for name in models
    }
    all_vintages = sorted(set().union(*per_model.values())) if per_model else []
    return pd.DataFrame(
        {
            model_id: [vintage in seen for vintage in all_vintages]
            for model_id, seen in per_model.items()
        },
        index=pd.Index(all_vintages, name="vintage"),
    )


def resolve_common_vintage(
    vintage: str | None = None,
    *,
    models: Sequence[str] = CANONICAL_MODEL_IDS,
    project_root: Path | str | None = None,
) -> str:
    """Return one vintage for which *every* requested model has a dataset.

    Each notebook resolved its own latest vintage independently, so a partially
    written vintage directory silently produced a mixed-vintage estimation
    round that only surfaced in the notebook 10 audit. This fails immediately
    instead, and names the models that are missing.
    """
    coverage = vintage_coverage(models=models, project_root=project_root)
    if coverage.empty:
        raise VintageError("No processed vintages were found for any model.")

    if vintage is not None:
        vintage = str(vintage)
        if vintage not in coverage.index:
            raise VintageError(f"Vintage {vintage!r} does not exist.")
        row = coverage.loc[vintage]
        missing = sorted(row.index[~row.to_numpy(dtype=bool)])
        if missing:
            raise VintageError(
                f"Vintage {vintage!r} is incomplete; missing datasets for "
                f"{missing}."
            )
        return vintage

    complete = coverage.index[coverage.all(axis=1)]
    if len(complete) == 0:
        latest = coverage.index[-1]
        row = coverage.loc[latest]
        missing = sorted(row.index[~row.to_numpy(dtype=bool)])
        raise VintageError(
            "No vintage carries a dataset for every model. The most recent "
            f"vintage {latest!r} is missing {missing}."
        )
    return str(complete[-1])


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """A model-ready estimation input set."""

    model_id: str
    vintage: str
    dataset_path: Path
    levels: pd.DataFrame
    exog: pd.DataFrame | None
    variables: list[str]
    units: dict[str, str]
    target: str
    frequency: str
    p: int
    exog_prior_scale: float | None
    spec: ModelSpec

    @property
    def n_observations(self) -> int:
        return int(len(self.levels))

    def describe(self) -> pd.Series:
        return pd.Series(
            {
                "model_id": self.model_id,
                "vintage": self.vintage,
                "dataset": str(self.dataset_path),
                "frequency": self.frequency,
                "p": self.p,
                "variables": ", ".join(self.variables),
                "target": self.target,
                "n_observations": self.n_observations,
                "sample_start": self.levels.index.min(),
                "sample_end": self.levels.index.max(),
                "n_exog": 0 if self.exog is None else int(self.exog.shape[1]),
            }
        )


def _library_spec(spec: ModelSpec) -> dict:
    """Fetch variables/target/units from the library, or from the inline copy.

    ``weekly_fuel_spec`` exposes the dataset name under ``dataset`` while
    ``monthly_hicp_component_spec`` uses ``dataset_file``. That inconsistency is
    absorbed here rather than propagated.
    """
    if spec.family == "monthly_hicp":
        raw = monthly_hicp_component_spec(spec.spec_key)
        dataset_file = raw["dataset_file"]
    elif spec.family == "weekly_fuel":
        raw = weekly_fuel_spec(spec.spec_key)
        dataset_file = raw["dataset"]
    else:
        if not spec.inline_variables or spec.inline_target is None:
            raise PipelineError(
                f"{spec.model_id}: no library spec function and no inline "
                "variable list."
            )
        return {
            "variables": list(spec.inline_variables),
            "units": dict(spec.inline_units),
            "target": spec.inline_target,
            "dataset_file": spec.dataset_file,
        }

    if dataset_file != spec.dataset_file:
        raise PipelineError(
            f"{spec.model_id}: MODEL_SPECS declares dataset_file="
            f"{spec.dataset_file!r} but the library spec says "
            f"{dataset_file!r}. Reconcile before estimating."
        )
    return {
        "variables": list(raw["variables"]),
        "units": dict(raw["units"]),
        "target": raw["target"],
        "dataset_file": dataset_file,
    }


def _manifest_pre_tax_unit(dataset: Path, component: str, fallback: str) -> str:
    """Resolve the constructed consumer-price unit from the vintage manifest.

    Gas/electricity units are part of the processed-vintage construction
    contract and must not be hard-coded in the orchestration table.  If an old
    manifest does not expose ``output_price_unit``, the provided fallback is
    retained.
    """
    import json

    manifest_path = dataset.parent / "manifest.json"
    if not manifest_path.is_file():
        return fallback
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = (
            manifest.get("construction", {})
            .get("pre_tax", {})
            .get(component, {})
            .get("output_price_unit")
        )
    except (OSError, ValueError, TypeError):
        return fallback
    value = "" if value is None else str(value).strip()
    return value or fallback


def model_contract(
    name: str,
    vintage: str | None = None,
    *,
    project_root: Path | str | None = None,
    spec_overrides: Mapping | None = None,
) -> dict:
    """Return the display/estimation contract for one model.

    This is the public counterpart to ``_library_spec``.  It resolves the
    canonical variables, target, dataset and units for a concrete vintage.
    Gas and electricity constructed-price units come from ``manifest.json``
    when available, preventing a silent EUR/kWh vs EUR/MWh mismatch.
    """
    spec = model_spec(name, **(spec_overrides or {}))
    detail = _library_spec(spec)
    units = dict(detail["units"])
    resolved_vintage = None
    dataset = None
    if vintage is not None:
        # Contract inspection must remain possible for a portable saved run even
        # when data/processed is not mounted. Estimation itself still resolves
        # and validates the vintage in build_panel().
        resolved_vintage = str(vintage)
        dataset = _processed_root(project_root) / resolved_vintage / spec.dataset_file
        if spec.family == "gas":
            units["gas_pre_tax"] = _manifest_pre_tax_unit(
                dataset, "gas", units.get("gas_pre_tax", "source unit")
            )
        elif spec.family == "electricity":
            units["electricity_pre_tax"] = _manifest_pre_tax_unit(
                dataset,
                "electricity",
                units.get("electricity_pre_tax", "source unit"),
            )
    return {
        **detail,
        "model_id": spec.model_id,
        "label": spec.label,
        "family": spec.family,
        "frequency": spec.frequency,
        "p": spec.p,
        "units": units,
        "vintage": resolved_vintage,
        "dataset_path": dataset,
    }


def build_panel(
    name: str,
    vintage: str | None = None,
    *,
    project_root: Path | str | None = None,
    spec_overrides: Mapping | None = None,
) -> Panel:
    """Load levels and deterministic regressors for one model and vintage.

    Dispatches on ``spec.family`` because the seven models are not homogeneous:
    gas is bivariate with no exog, electricity adds eleven seasonal dummies,
    the weekly models use a different loader and lag order.
    """
    spec = model_spec(name, **(spec_overrides or {}))
    vintage = resolve_vintage(spec.model_id, vintage, project_root=project_root)
    dataset = _processed_root(project_root) / vintage / spec.dataset_file
    detail = model_contract(
        spec.model_id, vintage, project_root=project_root, spec_overrides=spec_overrides
    )
    variables = detail["variables"]

    exog: pd.DataFrame | None = None
    if spec.family == "monthly_hicp":
        levels = load_monthly_hicp_component_panel(
            dataset, spec.spec_key, variables=variables
        )
    elif spec.family == "weekly_fuel":
        levels = load_weekly_fuel_panel(dataset, spec.spec_key)
    elif spec.family == "gas":
        levels = load_gas_panel(dataset, variables=variables)
    elif spec.family == "electricity":
        levels = load_electricity_panel(dataset, variables=variables)
        exog = make_electricity_seasonal_dummies(
            levels.index, reference_month=spec.reference_month
        )
    else:
        raise PipelineError(f"Unknown model family {spec.family!r}.")

    missing = [column for column in variables if column not in levels.columns]
    if missing:
        raise PipelineError(
            f"{spec.model_id}: loader returned a panel missing {missing}."
        )

    return Panel(
        model_id=spec.model_id,
        vintage=vintage,
        dataset_path=dataset,
        levels=levels[variables],
        exog=exog,
        variables=variables,
        units=detail["units"],
        target=detail["target"],
        frequency=spec.frequency,
        p=spec.p,
        exog_prior_scale=spec.exog_prior_scale,
        spec=spec,
    )


# ---------------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------------


def forecast_horizon(name: str, last_date, *, spec_overrides: Mapping | None = None) -> int:
    """Horizon in native periods, reproducing the notebooks' conventions.

    Monthly models forecast to the end of the current calendar year with a
    floor of three months; weekly models use a fixed 26-week horizon.
    """
    spec = model_spec(name, **(spec_overrides or {}))
    if spec.horizon_rule == "fixed":
        if spec.horizon_fixed is None:
            raise PipelineError(f"{spec.model_id}: horizon_fixed is not set.")
        return int(spec.horizon_fixed)
    if spec.horizon_rule != "to_year_end":
        raise PipelineError(f"Unknown horizon rule {spec.horizon_rule!r}.")
    stamp = pd.Timestamp(last_date)
    months_to_year_end = 12 - int(stamp.month)
    # A December origin has zero months left in the year; the notebooks roll it
    # forward to a full twelve rather than collapsing onto the floor of three.
    if months_to_year_end == 0:
        months_to_year_end = 12
    return int(max(spec.horizon_floor, months_to_year_end))


def forecast_draws(name: str, n_posterior_draws: int, *, spec_overrides: Mapping | None = None) -> int:
    """Number of predictive paths to simulate, per the notebooks' conventions.

    Monthly models take ``min(1000, n_draws)``; weekly models use a fixed 500.
    """
    spec = model_spec(name, **(spec_overrides or {}))
    if spec.forecast_draws_rule == "fixed":
        return int(min(spec.forecast_draws, n_posterior_draws))
    if spec.forecast_draws_rule != "min_1000":
        raise PipelineError(f"Unknown draw rule {spec.forecast_draws_rule!r}.")
    return int(min(spec.forecast_draws_cap, n_posterior_draws))


# ---------------------------------------------------------------------------
# Run and forecast-store resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRef:
    """A pointer to one saved estimation run."""

    model_id: str
    vintage: str
    run_id: str
    directory: Path
    has_draws: bool
    forecast_names: tuple[str, ...]

    def forecast_dir(self, forecast_name: str = "unconditional") -> Path:
        return self.directory / "forecasts" / forecast_name


def discover_runs(
    name: str,
    vintage: str,
    *,
    results_root: Path | str,
) -> list[RunRef]:
    """List every saved run for one model and vintage, newest last.

    ``has_draws`` records whether ``draws.npz`` was persisted. Runs saved
    through ``save_energy_bvar_forecast_for_result`` do not carry the Gibbs
    arrays, so impulse responses, FEVD and historical decompositions are
    impossible for them without re-estimating.
    """
    spec = model_spec(name)
    base = Path(results_root) / spec.model_id / str(vintage)
    if not base.is_dir():
        return []
    runs: list[RunRef] = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "metadata.json").is_file():
            continue
        forecasts_dir = run_dir / "forecasts"
        names = tuple(
            sorted(
                child.name
                for child in (
                    forecasts_dir.iterdir() if forecasts_dir.is_dir() else []
                )
                if (child / "forecast_metadata.json").is_file()
                and (child / "forecast_draws.npz").is_file()
            )
        )
        runs.append(
            RunRef(
                model_id=spec.model_id,
                vintage=str(vintage),
                run_id=run_dir.name,
                directory=run_dir,
                has_draws=(run_dir / "draws.npz").is_file(),
                forecast_names=names,
            )
        )
    return runs


def resolve_run(
    name: str,
    vintage: str,
    *,
    results_root: Path | str,
    run_id: str | None = None,
    forecast_name: str | None = "unconditional",
) -> RunRef:
    """Select exactly one run, refusing to guess when several qualify.

    Notebook 10 resolved this with ``max(stores, key=st_mtime_ns)``, i.e. the
    most recently written directory. Once several prior configurations coexist
    for one vintage, that silently makes the aggregate depend on file
    timestamps: a copy, a backup restore or a partial rewrite changes the
    selected model without any diagnostic. This raises instead, and the caller
    (registry promotion, or an explicit ``run_id``) must disambiguate.
    """
    runs = discover_runs(name, vintage, results_root=results_root)
    if forecast_name is not None:
        runs = [run for run in runs if forecast_name in run.forecast_names]

    spec = model_spec(name)
    if run_id is not None:
        matches = [run for run in runs if run.run_id == run_id]
        if not matches:
            raise PipelineError(
                f"{spec.model_id}/{vintage}: no run {run_id!r} with forecast "
                f"{forecast_name!r}."
            )
        return matches[0]

    if not runs:
        raise PipelineError(
            f"{spec.model_id}/{vintage}: no saved run carries a "
            f"{forecast_name!r} forecast store. Estimate and save it first."
        )
    if len(runs) > 1:
        raise PipelineError(
            f"{spec.model_id}/{vintage}: {len(runs)} runs carry a "
            f"{forecast_name!r} forecast store "
            f"({[run.run_id[:12] for run in runs]}). Pass run_id explicitly, "
            "or promote one run in the registry. Refusing to pick by "
            "modification time."
        )
    return runs[0]


def resolve_forecast_stores(
    vintage: str,
    *,
    results_root: Path | str,
    forecast_name: str = "unconditional",
    run_ids: Mapping[str, str] | None = None,
    models: Sequence[str] = CANONICAL_MODEL_IDS,
) -> dict[str, Path]:
    """Return ``{aggregate_key: forecast_store_path}`` for the aggregation layer.

    Keys match notebook 10's ``MODEL_IDS`` mapping (``petrol``, ``diesel``, ...)
    so the result is a drop-in replacement for its ``FORECAST_STORES`` dict.
    All errors are collected before raising, so one call reports every missing
    or ambiguous component rather than only the first.
    """
    run_ids = dict(run_ids or {})
    stores: dict[str, Path] = {}
    errors: list[str] = []
    for name in models:
        spec = model_spec(name)
        try:
            run = resolve_run(
                spec.model_id,
                vintage,
                results_root=results_root,
                run_id=run_ids.get(spec.model_id) or run_ids.get(spec.aggregate_key),
                forecast_name=forecast_name,
            )
        except PipelineError as exc:
            errors.append(str(exc))
            continue
        stores[spec.aggregate_key] = run.forecast_dir(forecast_name)
    if errors:
        raise PipelineError(
            "Cannot assemble the aggregate forecast stores:\n  - "
            + "\n  - ".join(errors)
        )
    return stores


# ---------------------------------------------------------------------------
# Component estimation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    """What one full estimate-and-forecast cycle produced."""

    model_id: str
    vintage: str
    run_id: str
    directory: Path | None
    forecast_directory: Path | None
    hicp_directory: Path | None
    forecast_name: str
    horizon: int
    n_posterior_draws: int
    n_forecast_draws: int
    persisted_draws: bool
    panel: "Panel"
    result: Mapping
    forecast: Mapping
    hicp_forecast: Mapping

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "model_id": self.model_id,
                "vintage": self.vintage,
                "run_id": self.run_id,
                "horizon": self.horizon,
                "posterior_draws": self.n_posterior_draws,
                "forecast_draws": self.n_forecast_draws,
                "persisted_draws": self.persisted_draws,
                "directory": "" if self.directory is None else str(self.directory),
                "hicp_store": "" if self.hicp_directory is None else str(self.hicp_directory),
            }
        )


def _future_dates(panel: "Panel", last_date, horizon: int) -> pd.DatetimeIndex:
    """Calendar dates for the forecast window, in the model's own frequency."""
    stamp = pd.Timestamp(last_date)
    if panel.frequency == "monthly":
        return pd.date_range(
            stamp + pd.DateOffset(months=1), periods=horizon, freq="MS"
        )
    if panel.frequency == "weekly":
        return pd.date_range(
            stamp + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON"
        )
    raise PipelineError(f"Unknown frequency {panel.frequency!r}.")


def _future_exog(panel: "Panel", last_date, horizon: int) -> pd.DataFrame | None:
    """Deterministic regressors over the forecast window.

    Only electricity carries any; the dispatch mirrors :func:`build_panel` so
    that in-sample and out-of-sample regressors cannot drift apart.
    """
    if panel.exog is None:
        return None
    if panel.spec.family != "electricity":
        raise PipelineError(
            f"{panel.model_id}: in-sample exog exist but no future-exog rule "
            "is defined for this family."
        )
    future = make_electricity_seasonal_dummies(
        _future_dates(panel, last_date, horizon),
        reference_month=panel.spec.reference_month,
    )
    if list(future.columns) != list(panel.exog.columns):
        raise PipelineError(
            f"{panel.model_id}: future exog columns {list(future.columns)} "
            f"differ from in-sample {list(panel.exog.columns)}."
        )
    return future


def planned_run_metadata(
    name: str,
    vintage: str | None = None,
    *,
    project_root: Path | str | None = None,
    prior_config=None,
    sampler_config=None,
    spec_overrides: Mapping | None = None,
) -> dict:
    """Return the exact metadata identity a dashboard run would receive.

    This performs no Gibbs sampling.  It is used by the Estimation page to
    detect an already-complete deterministic run before spending several
    minutes re-estimating the same model.  The progress hook is intentionally
    excluded from the identity and therefore never changes ``run_id``.
    """
    from energy_bvar_model import (
        BVARSVOPriorConfig,
        SamplerConfig,
        build_run_metadata,
    )

    spec = model_spec(name, **(spec_overrides or {}))
    panel = build_panel(
        spec.model_id,
        vintage,
        project_root=project_root,
        spec_overrides=spec_overrides,
    )
    prior_config = BVARSVOPriorConfig() if prior_config is None else prior_config
    sampler_config = (
        SamplerConfig(seed=spec.seed) if sampler_config is None else sampler_config
    )
    return build_run_metadata(
        model_id=spec.model_id,
        vintage=panel.vintage,
        levels=panel.levels,
        p=panel.p,
        variables=panel.variables,
        frequency=spec.frequency,
        exog=panel.exog,
        exog_prior_scale=(
            10.0 if panel.exog_prior_scale is None else panel.exog_prior_scale
        ),
        prior_config=prior_config,
        sampler_config=sampler_config,
        missing_data_method=spec.missing_data_method,
        code_version=spec.code_version,
    )


def run_component(
    name: str,
    vintage: str | None = None,
    *,
    project_root: Path | str | None = None,
    results_root: Path | str | None = None,
    prior_config=None,
    sampler_config=None,
    spec_overrides: Mapping | None = None,
    forecast_name: str = "unconditional",
    persist: bool = True,
    persist_draws: bool = True,
    overwrite: bool = False,
    progress_callback=None,
) -> RunOutcome:
    """Estimate one component and save its unconditional predictive paths.

    This is the single entry point shared by the notebooks and the dashboard.
    Given the same model, vintage and configuration, both callers must obtain
    the same ``run_id``; if they do not, a constant is still living outside
    :data:`MODEL_SPECS`.

    ``persist_draws`` defaults to ``True``, unlike the notebooks'
    ``SAVE_RESULTS = False``. Without ``draws.npz`` the Gibbs arrays are gone
    once the kernel dies, and impulse responses, FEVD and historical
    decompositions become impossible without re-estimating.

    Only the unconditional forecast is produced. Conditional paths are cheap to
    regenerate from the persisted draws and are better computed on demand than
    frozen into a fixed scenario set.
    """
    # Imported lazily so that the declarative half of this module stays usable
    # (and testable) without pulling in the sampler.
    from energy_bvar_model import (
        BVARSVOPriorConfig,
        SamplerConfig,
        forecast_bvar_sv_outlier,
        run_energy_bvar,
    )
    from energy_bvar_io import (
        save_energy_bvar_forecast,
        save_energy_bvar_forecast_for_result,
        save_energy_bvar_hicp_forecast,
        save_energy_bvar_result,
    )
    from energy_bvar_component_hicp import build_component_hicp_forecast

    spec = model_spec(name, **(spec_overrides or {}))
    panel = build_panel(
        spec.model_id,
        vintage,
        project_root=project_root,
        spec_overrides=spec_overrides,
    )
    if results_root is None:
        results_root = find_project_root(project_root) / "results"
    results_root = Path(results_root)

    prior_config = BVARSVOPriorConfig() if prior_config is None else prior_config
    if sampler_config is None:
        sampler_config = SamplerConfig(seed=spec.seed)

    result = run_energy_bvar(
        levels=panel.levels,
        model_id=spec.model_id,
        vintage=panel.vintage,
        p=panel.p,
        variables=panel.variables,
        frequency=spec.frequency,
        exog=panel.exog,
        exog_prior_scale=(
            10.0 if panel.exog_prior_scale is None else panel.exog_prior_scale
        ),
        prior_config=prior_config,
        sampler_config=sampler_config,
        missing_data_method=spec.missing_data_method,
        code_version=spec.code_version,
        progress_callback=progress_callback,
    )

    run_id = str(result["metadata"]["run_id"])
    run_directory = results_root / spec.model_id / panel.vintage / run_id
    forecast_directory = run_directory / "forecasts" / forecast_name
    if persist and not overwrite and forecast_directory.is_dir():
        raise PipelineError(
            f"{spec.model_id}/{panel.vintage}/{run_id[:12]}: a "
            f"{forecast_name!r} forecast store already exists at "
            f"{forecast_directory}. Pass overwrite=True to replace it."
        )

    prep = result["prep"]
    last_date = prep["last_calendar_date"]
    horizon = forecast_horizon(
        spec.model_id, last_date, spec_overrides=spec_overrides
    )
    n_forecast_draws = forecast_draws(
        spec.model_id, int(result["n_draws"]), spec_overrides=spec_overrides
    )

    forecast = forecast_bvar_sv_outlier(
        result,
        H=horizon,
        future_exog=_future_exog(panel, last_date, horizon),
        n_draws=n_forecast_draws,
        simulate_future_outliers=spec.simulate_future_outliers,
        seed=spec.forecast_seed,
    )

    # The component HICP product is part of the production contract, not a
    # dashboard-side reconstruction.  This calls the same post-processing
    # functions used by the component notebooks (tax re-attribution where
    # required, weekly->monthly + HICP rebasing for fuels, direct extraction
    # for HICP-target models).
    hicp_forecast = build_component_hicp_forecast(
        spec.model_id,
        forecast,
        result=result,
        dataset_path=panel.dataset_path,
        project_root=project_root,
        forecast_name=forecast_name,
    )

    directory: Path | None = None
    saved_forecast: Path | None = None
    saved_hicp: Path | None = None
    if persist:
        if persist_draws:
            saved = save_energy_bvar_result(result, results_root)
            directory = Path(saved["directory"])
            saved_forecast = Path(
                save_energy_bvar_forecast(forecast, directory, forecast_name)[
                    "directory"
                ]
            )
        else:
            saved_forecast = Path(
                save_energy_bvar_forecast_for_result(
                    forecast, result, results_root, forecast_name
                )["directory"]
            )
            directory = saved_forecast.parent.parent

        # Save HICP beside the raw predictive store.  The dashboard can now
        # display HICP level / HICP YoY without calling the aggregate pipeline.
        saved_hicp = Path(
            save_energy_bvar_hicp_forecast(
                hicp_forecast, saved_forecast, overwrite=overwrite
            )["directory"]
        )

    return RunOutcome(
        model_id=spec.model_id,
        vintage=panel.vintage,
        run_id=run_id,
        directory=directory,
        forecast_directory=saved_forecast,
        hicp_directory=saved_hicp,
        forecast_name=forecast_name,
        horizon=horizon,
        n_posterior_draws=int(result["n_draws"]),
        n_forecast_draws=n_forecast_draws,
        persisted_draws=bool(persist and persist_draws),
        panel=panel,
        result=result,
        forecast=forecast,
        hicp_forecast=hicp_forecast,
    )

# ---------------------------------------------------------------------------
# Seven-model suite orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteOutcome:
    """Results of one coherent seven-model estimation round.

    The vintage is resolved once before any model starts. This prevents a
    partially-written processed directory from producing a mixed-vintage suite.
    """

    vintage: str
    outcomes: Mapping[str, RunOutcome]

    def summary(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {model_id: outcome.summary() for model_id, outcome in self.outcomes.items()}
        ).T
        frame.index.name = "model_id"
        return frame


def _mapping_value(mapping: Mapping | None, spec: ModelSpec):
    """Resolve a per-model mapping by canonical id first, then aggregate key."""
    if mapping is None:
        return None
    return mapping.get(spec.model_id, mapping.get(spec.aggregate_key))


def run_all_components(
    vintage: str | None = None,
    *,
    project_root: Path | str | None = None,
    results_root: Path | str | None = None,
    prior_configs: Mapping | None = None,
    sampler_configs: Mapping | None = None,
    spec_overrides: Mapping[str, Mapping] | None = None,
    forecast_name: str = "unconditional",
    persist: bool = True,
    persist_draws: bool = True,
    overwrite: bool = False,
) -> SuiteOutcome:
    """Estimate all seven BVARs on one common processed vintage.

    ``vintage=None`` means the latest vintage for which every canonical model
    dataset exists. Per-model overrides may be keyed by canonical ``model_id``
    (preferred) or by the aggregate key (``petrol``, ``diesel``, ...).

    The function deliberately runs sequentially. Background execution and
    cancellation belong to the Dash layer; the econometric pipeline stays
    deterministic and single-process.
    """
    common = resolve_common_vintage(
        vintage, models=CANONICAL_MODEL_IDS, project_root=project_root
    )
    overrides = dict(spec_overrides or {})
    outcomes: dict[str, RunOutcome] = {}

    for model_id in CANONICAL_MODEL_IDS:
        base_spec = model_spec(model_id)
        model_overrides = _mapping_value(overrides, base_spec) or {}
        outcome = run_component(
            model_id,
            common,
            project_root=project_root,
            results_root=results_root,
            prior_config=_mapping_value(prior_configs, base_spec),
            sampler_config=_mapping_value(sampler_configs, base_spec),
            spec_overrides=model_overrides,
            forecast_name=forecast_name,
            persist=persist,
            persist_draws=persist_draws,
            overwrite=overwrite,
        )
        if outcome.vintage != common:
            raise PipelineError(
                f"{model_id}: resolved vintage {outcome.vintage!r} differs from "
                f"suite vintage {common!r}."
            )
        outcomes[model_id] = outcome

    return SuiteOutcome(vintage=common, outcomes=outcomes)

