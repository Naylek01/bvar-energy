"""Headline-component adapters for the shared BVAR-SV-outlier engine.

The project already has a validated statistical engine in ``energy_bvar_model``.
This module does *not* duplicate that sampler.  It provides the headline-suite
adapter layer that keeps three spaces explicit:

    native levels -> model state levels -> first differences used by the engine

For Energy, native and state levels coincide, so the generic engine estimates
absolute differences.  For the non-energy HICP components considered here,
the paper specification is based on first log differences; the adapter therefore
maps positive native levels ``x_t`` to ``log(x_t)`` before calling the same
engine.  Forecast state paths are mapped back to native units afterwards.

Economic specification
----------------------
The component contracts follow Benalal, Diaz del Hoyo, Landau, Roma and
Skudelny, ECB Working Paper 374 (2004):

* unprocessed food: own lags 1, 10 and 12 + monthly seasonal dummies;
* processed food: dynamic single equation with target lags 1-4, food commodity
  prices lag 2, compensation per employee lags 0-2, contemporaneous VAT, and
  seasonal dummies;
* NEIG: dynamic single equation with target lags 1, 6 and 12, PPI consumer
  goods lag 1, compensation per employee lag 2, contemporaneous VAT, and
  seasonal dummies;
* services: BVAR with services, compensation per employee, PPI consumer goods
  and unprocessed food, using lags 1-5 and 12.

Bayesian machinery
------------------
The production sampler is reused from ``energy_bvar_model``: Minnesota
shrinkage centred on white noise, stochastic volatility, model-based outliers,
and Durbin--Koopman missing/ragged-edge handling.  This mirrors the machinery
used by the project's implementation of ECB Working Paper 3062.

Important implementation boundary
---------------------------------
``unprocessed_food``, ``processed_food``, ``neig`` and ``services`` are
production-enabled in this adapter version.  Processed food and NEIG are implemented as the exact
WP374 dynamic single equations in lag structure: the HICP target is the sole
endogenous state while transformed external drivers enter through the shared
engine's exogenous block.  The generic Energy engine now accepts an optional
``active_lags`` restriction, so NEIG can estimate exactly lags 1, 6 and 12 while
retaining a full 12-lag companion state for stability, DK and forecasting.

The unprocessed-food production model still uses a dense Bayesian AR(12) as an
explicit shrinkage approximation to the paper's sparse AR(1,10,12). Services is
implemented as the selected four-variable WP374 BVAR with the common sparse lag
set (1,2,3,4,5,12).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from energy_bvar_model import (
    BVARSVOPriorConfig,
    SamplerConfig,
    forecast_bvar_sv_outlier,
    load_energy_panel,
    monthly_seasonal_dummies,
    run_energy_bvar,
)


# -----------------------------------------------------------------------------
# Public contract constants
# -----------------------------------------------------------------------------

STATE_PREFIX = "state__"
HEADLINE_FORECAST_CONTRACT_VERSION = "headline-hicp-component-v1"
HEADLINE_ADAPTER_VERSION = "2026-08-11-v4"
SUPPORTED_TRANSFORMS = ("diff", "logdiff", "percent_rate_diff")


# -----------------------------------------------------------------------------
# Component specifications
# -----------------------------------------------------------------------------

# ``paper_active_lags`` describes the selected WP374 specification. ``p`` is
# the maximum companion lag. ``active_lags`` optionally fixes all omitted
# endogenous lags exactly at zero in the shared Energy engine.
HEADLINE_COMPONENT_SPECS: dict[str, dict[str, object]] = {
    "unprocessed_food": {
        "label": "Unprocessed food",
        "dataset_file": "unprocessed_food_monthly.csv",
        "frequency": "monthly",
        "target": "hicp_unprocessed_food",
        "variables": ["hicp_unprocessed_food"],
        "transforms": {"hicp_unprocessed_food": "logdiff"},
        "units": {"hicp_unprocessed_food": "HICP index"},
        "p": 12,
        "seasonal_dummies": True,
        "seasonal_reference_month": 12,
        "paper_model_class": "dynamic single equation / autoregression",
        "paper_active_lags": {"hicp_unprocessed_food": [1, 10, 12]},
        "paper_lag_structure": "HICP unprocessed food L1, L10, L12 + 11 monthly seasonal dummies",
        "implementation_mode": "dense_bayesian_ar12_shrinkage_approximation",
        "production_ready": True,
        "production_blocker": None,
    },
    "processed_food": {
        "label": "Processed food",
        "dataset_file": "processed_food_monthly.csv",
        "frequency": "monthly",
        "target": "hicp_processed_food",
        "variables": [
            "hicp_processed_food",
            "food_commodity_price",
            "compensation_per_employee",
            "vat_standard_ea",
        ],
        # Price/wage levels enter in log differences.  VAT is supplied by the
        # builder in percentage points (e.g. 20.5); the model state is the rate
        # as a share (0.205), so a +1 pp VAT move is a +0.01 model change.
        "transforms": {
            "hicp_processed_food": "logdiff",
            "food_commodity_price": "logdiff",
            "compensation_per_employee": "logdiff",
            "vat_standard_ea": "percent_rate_diff",
        },
        "units": {
            "hicp_processed_food": "HICP index",
            "food_commodity_price": "index",
            "compensation_per_employee": "compensation per employee",
            "vat_standard_ea": "% standard VAT rate",
        },
        "p": 4,
        "active_lags": [1, 2, 3, 4],
        "seasonal_dummies": True,
        "seasonal_reference_month": 12,
        "paper_model_class": "dynamic single equation",
        "paper_active_lags": {
            "hicp_processed_food": [1, 2, 3, 4],
            "food_commodity_price": [2],
            "compensation_per_employee": [0, 1, 2],
            "vat_standard_ea": [0],
        },
        "single_equation_predictors": {
            "food_commodity_price": [2],
            "compensation_per_employee": [0, 1, 2],
            "vat_standard_ea": [0],
        },
        "paper_lag_structure": "HICP PF L1-L4; food commodities L2; wages L0-L2; VAT L0; seasonal dummies",
        "implementation_mode": "bayesian_arx4_sv_outlier_exact_wp374_lag_structure",
        "forecast_driver_policy": "flat_native_level",
        "production_ready": True,
        "production_blocker": None,
    },
    "neig": {
        "label": "Non-energy industrial goods",
        "dataset_file": "neig_monthly.csv",
        "frequency": "monthly",
        "target": "hicp_neig",
        "variables": [
            "hicp_neig",
            "ppi_consumer_goods",
            "compensation_per_employee",
            "vat_standard_ea",
        ],
        "transforms": {
            "hicp_neig": "logdiff",
            "ppi_consumer_goods": "logdiff",
            "compensation_per_employee": "logdiff",
            "vat_standard_ea": "percent_rate_diff",
        },
        "units": {
            "hicp_neig": "HICP index",
            "ppi_consumer_goods": "PPI consumer-goods index",
            "compensation_per_employee": "compensation per employee",
            "vat_standard_ea": "% standard VAT rate",
        },
        "p": 12,
        "active_lags": [1, 6, 12],
        "seasonal_dummies": True,
        "seasonal_reference_month": 12,
        "paper_model_class": "dynamic single equation",
        "paper_active_lags": {
            "hicp_neig": [1, 6, 12],
            "ppi_consumer_goods": [1],
            "compensation_per_employee": [2],
            "vat_standard_ea": [0],
        },
        "single_equation_predictors": {
            "ppi_consumer_goods": [1],
            "compensation_per_employee": [2],
            "vat_standard_ea": [0],
        },
        "paper_lag_structure": "HICP NEIG L1,L6,L12; PPI consumer goods L1; wages L2; VAT L0; seasonal dummies",
        "implementation_mode": "bayesian_sparse_arx12_sv_outlier_exact_wp374_lag_structure",
        "forecast_driver_policy": "flat_native_level",
        "production_ready": True,
        "production_blocker": None,
    },
    "services": {
        "label": "Services",
        "dataset_file": "services_monthly.csv",
        "frequency": "monthly",
        "target": "hicp_services",
        "variables": [
            "hicp_services",
            "compensation_per_employee",
            "ppi_consumer_goods",
            "hicp_unprocessed_food",
        ],
        "transforms": {
            "hicp_services": "logdiff",
            "compensation_per_employee": "logdiff",
            "ppi_consumer_goods": "logdiff",
            "hicp_unprocessed_food": "logdiff",
        },
        "units": {
            "hicp_services": "HICP index",
            "compensation_per_employee": "index",
            "ppi_consumer_goods": "PPI index",
            "hicp_unprocessed_food": "HICP index",
        },
        "p": 12,
        "active_lags": [1, 2, 3, 4, 5, 12],
        "seasonal_dummies": False,
        "seasonal_reference_month": 12,
        "paper_model_class": "BVAR",
        "paper_active_lags": {
            "hicp_services": [1, 2, 3, 4, 5, 12],
            "compensation_per_employee": [1, 2, 3, 4, 5, 12],
            "ppi_consumer_goods": [1, 2, 3, 4, 5, 12],
            "hicp_unprocessed_food": [1, 2, 3, 4, 5, 12],
        },
        "paper_lag_structure": "Services/wages/PPI consumer goods/unprocessed food: L1-L5 and L12",
        "implementation_mode": "bayesian_sparse_bvar12_sv_outlier_exact_wp374_lag_structure",
        "paper_bvar_hyperparameters": {
            "tightness": 0.1,
            "cross_variable_weight": 0.9,
            "lag_decay": 0.1,
        },
        "production_bayesian_policy": (
            "Use the validated Energy-suite BVAR-SV-outlier prior calibration; "
            "WP374 hyperparameters are retained only as a historical reference."
        ),
        "production_ready": True,
        "production_blocker": None,
    },
}


# -----------------------------------------------------------------------------
# Specification helpers
# -----------------------------------------------------------------------------


def headline_component_spec(component: str) -> dict[str, object]:
    """Return a defensive copy of one headline-component specification."""
    key = str(component).strip().lower()
    try:
        return deepcopy(HEADLINE_COMPONENT_SPECS[key])
    except KeyError as exc:
        raise ValueError(
            f"Unknown headline component {component!r}. Choose one of "
            f"{tuple(HEADLINE_COMPONENT_SPECS)}."
        ) from exc


def headline_component_contract_table() -> pd.DataFrame:
    """Summarise the paper and implementation contract for all components."""
    rows = []
    for component, spec in HEADLINE_COMPONENT_SPECS.items():
        rows.append(
            {
                "component": component,
                "label": spec["label"],
                "dataset_file": spec["dataset_file"],
                "target": spec["target"],
                "paper_model_class": spec["paper_model_class"],
                "paper_lag_structure": spec["paper_lag_structure"],
                "active_lags": spec.get("active_lags"),
                "implementation_mode": spec["implementation_mode"],
                "production_ready": bool(spec["production_ready"]),
                "production_blocker": spec["production_blocker"],
            }
        )
    return pd.DataFrame(rows).set_index("component")


def _validate_transform(kind: str) -> str:
    value = str(kind).strip().lower()
    if value not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"Unknown transform {kind!r}; supported transforms are "
            f"{SUPPORTED_TRANSFORMS}."
        )
    return value


def _state_name(native_name: str) -> str:
    return f"{STATE_PREFIX}{native_name}"


def state_variable_map(component: str) -> dict[str, str]:
    """Return ``native_name -> state_name`` for one component."""
    spec = headline_component_spec(component)
    return {name: _state_name(name) for name in spec["variables"]}


def resolve_headline_component_dataset(
    project_root: str | Path,
    component: str,
    *,
    vintage: str | None = None,
    data_root: str | Path | None = None,
) -> Path:
    """Resolve one headline-component CSV deterministically.

    By default this searches ``<project_root>/data/processed/<vintage>/``.
    Passing ``data_root`` lets notebooks use an isolated scratch-preview root
    without changing any model logic.
    """
    root = Path(project_root).expanduser().resolve()
    spec = headline_component_spec(component)
    filename = str(spec["dataset_file"])
    source_root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else root / "data" / "processed"
    )
    if not source_root.exists():
        raise FileNotFoundError(f"Headline data root does not exist: {source_root}")

    if vintage is not None:
        path = source_root / str(vintage) / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = sorted(path for path in source_root.glob(f"*/{filename}") if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No {filename} found below {source_root}. Run the headline-capable "
            "dataset builder first or set data_root/vintage explicitly."
        )
    return candidates[-1]


def headline_panel_audit(
    native_levels: pd.DataFrame,
    component: str,
) -> pd.DataFrame:
    """Return a compact, dashboard-friendly audit of native input series."""
    spec = headline_component_spec(component)
    expected = list(spec["variables"])
    missing = [name for name in expected if name not in native_levels.columns]
    if missing:
        raise KeyError(f"{component}: native panel is missing {missing}.")
    frame = native_levels[expected].copy().sort_index()
    rows = []
    for name in expected:
        series = pd.to_numeric(frame[name], errors="coerce").astype(float)
        observed = series.dropna()
        if name == spec["target"]:
            role = "target"
        elif spec.get("single_equation_predictors"):
            role = "exogenous_driver"
        else:
            role = "endogenous_auxiliary"
        rows.append(
            {
                "series": name,
                "role": role,
                "unit": spec["units"].get(name, "source unit"),
                "first_observed": None if observed.empty else pd.Timestamp(observed.index.min()),
                "last_observed": None if observed.empty else pd.Timestamp(observed.index.max()),
                "n_calendar": int(len(series)),
                "n_observed": int(series.notna().sum()),
                "n_missing": int(series.isna().sum()),
                "missing_share": float(series.isna().mean()) if len(series) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("series")


# -----------------------------------------------------------------------------
# Data loading and transform layer
# -----------------------------------------------------------------------------


def load_headline_component_panel(
    path: str | Path,
    component: str,
    variables: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load native monthly levels from a processed headline-component CSV.

    Missing values are retained exactly as in the Energy loaders.  No log,
    differencing, seasonal adjustment, interpolation, or edge filling occurs
    here; those operations belong either to the builder or to this adapter.
    """
    spec = headline_component_spec(component)
    selected = list(spec["variables"] if variables is None else variables)
    unknown = [name for name in selected if name not in spec["variables"]]
    if unknown:
        raise ValueError(
            f"{component}: requested variables {unknown} are outside the "
            f"component contract {spec['variables']}."
        )
    return load_energy_panel(
        path,
        variables=selected,
        frequency=str(spec["frequency"]),
    )


def native_to_state_levels(
    native_levels: pd.DataFrame,
    transforms: Mapping[str, str],
    *,
    rename_state: bool = True,
) -> pd.DataFrame:
    """Map native levels to levels whose first differences the engine models.

    ``diff``
        state = native; engine difference = x_t - x_{t-1}.

    ``logdiff``
        state = log(native); engine difference = log(x_t)-log(x_{t-1}).

    NaNs are preserved.  Every observed value transformed with ``logdiff`` must
    be strictly positive.
    """
    if not isinstance(native_levels, pd.DataFrame):
        raise TypeError("native_levels must be a pandas DataFrame.")
    missing_rules = [name for name in native_levels.columns if name not in transforms]
    if missing_rules:
        raise KeyError(f"Missing transform rule(s) for {missing_rules}.")
    extra_rules = [name for name in transforms if name not in native_levels.columns]
    if extra_rules:
        raise KeyError(
            f"Transform rule(s) supplied for absent native columns {extra_rules}."
        )

    state = pd.DataFrame(index=native_levels.index)
    for name in native_levels.columns:
        kind = _validate_transform(transforms[name])
        series = pd.to_numeric(native_levels[name], errors="coerce").astype(float)
        if np.isinf(series.to_numpy(dtype=float)).any():
            raise ValueError(f"{name}: native levels contain infinite values.")
        if kind == "logdiff":
            bad = series.notna() & (series <= 0)
            if bad.any():
                first = bad[bad].index[0]
                raise ValueError(
                    f"{name}: logdiff requires strictly positive observed "
                    f"levels; first invalid value is at {pd.Timestamp(first).date()}."
                )
            transformed = np.log(series)
        elif kind == "percent_rate_diff":
            # Native VAT is stored in percentage points, e.g. 20.5.  The state
            # is the corresponding share, 0.205, so a +1 pp change is +0.01.
            transformed = series / 100.0
        else:  # diff
            transformed = series
        output_name = _state_name(name) if rename_state else name
        state[output_name] = transformed.to_numpy(dtype=float)

    state.index = native_levels.index.copy()
    state.index.name = native_levels.index.name or "date"
    return state


def state_to_native_levels(
    state_levels: pd.DataFrame,
    transforms: Mapping[str, str],
    *,
    state_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Invert :func:`native_to_state_levels` for a DataFrame."""
    mapping = (
        {name: _state_name(name) for name in transforms}
        if state_names is None
        else dict(state_names)
    )
    missing = [state for state in mapping.values() if state not in state_levels.columns]
    if missing:
        raise KeyError(f"State-level frame is missing columns {missing}.")

    native = pd.DataFrame(index=state_levels.index)
    for name, kind_raw in transforms.items():
        kind = _validate_transform(kind_raw)
        values = pd.to_numeric(state_levels[mapping[name]], errors="coerce").astype(float)
        if kind == "logdiff":
            with np.errstate(over="ignore", invalid="ignore"):
                restored = np.exp(values)
        elif kind == "percent_rate_diff":
            restored = 100.0 * values
        else:
            restored = values
        native[name] = restored.to_numpy(dtype=float)

    if np.isinf(native.to_numpy(dtype=float)).any():
        raise FloatingPointError("Inverse state transformation produced infinite native levels.")
    native.index.name = state_levels.index.name or "date"
    return native


def state_paths_to_native(
    state_paths: np.ndarray,
    *,
    native_variables: Sequence[str],
    state_variables: Sequence[str],
    transforms: Mapping[str, str],
) -> np.ndarray:
    """Invert posterior state-level paths to native units draw by draw."""
    values = np.asarray(state_paths, dtype=float)
    if values.ndim != 3:
        raise ValueError("state_paths must have shape (draws, periods, variables).")
    state_variables = list(state_variables)
    native_variables = list(native_variables)
    if values.shape[2] != len(state_variables):
        raise ValueError("state_paths variable dimension does not match state_variables.")
    if set(native_variables) != set(transforms):
        raise ValueError("native_variables and transforms must describe the same variables.")

    output = np.empty((values.shape[0], values.shape[1], len(native_variables)))
    for j, native_name in enumerate(native_variables):
        state_name = _state_name(native_name)
        if state_name not in state_variables:
            raise ValueError(
                f"State variable {state_name!r} for {native_name!r} is absent "
                f"from forecast variables {state_variables}."
            )
        source_j = state_variables.index(state_name)
        kind = _validate_transform(transforms[native_name])
        source = values[:, :, source_j]
        if kind == "logdiff":
            with np.errstate(over="ignore", invalid="ignore"):
                restored = np.exp(source)
        elif kind == "percent_rate_diff":
            restored = 100.0 * source
        else:
            restored = source
        output[:, :, j] = restored

    if not np.isfinite(output).all():
        raise FloatingPointError(
            "Inverse forecast transformation produced non-finite native paths. "
            "Inspect extreme posterior state draws."
        )
    return output


def native_model_differences(
    native_levels: pd.DataFrame,
    component: str,
) -> pd.DataFrame:
    """Return the transformed first differences actually used by the engine."""
    spec = headline_component_spec(component)
    expected = list(spec["variables"])
    if list(native_levels.columns) != expected:
        missing = [name for name in expected if name not in native_levels.columns]
        extra = [name for name in native_levels.columns if name not in expected]
        if missing or extra:
            raise ValueError(
                f"{component}: native panel differs from contract; "
                f"missing={missing}, extra={extra}."
            )
        native_levels = native_levels[expected]
    state = native_to_state_levels(native_levels, spec["transforms"], rename_state=False)
    differences = state.diff()
    differences.columns = expected
    return differences


# -----------------------------------------------------------------------------
# Deterministic regressors and preparation
# -----------------------------------------------------------------------------


def make_headline_seasonal_dummies(
    index: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    component: str,
) -> pd.DataFrame | None:
    """Return the component's 11 monthly dummies, or ``None`` if not used."""
    spec = headline_component_spec(component)
    if not bool(spec["seasonal_dummies"]):
        return None
    return monthly_seasonal_dummies(
        index,
        reference_month=int(spec["seasonal_reference_month"]),
        prefix="month",
    )


def _normalise_native_monthly_panel(
    native_levels: pd.DataFrame,
    variables: Sequence[str],
    component: str,
) -> pd.DataFrame:
    missing = [name for name in variables if name not in native_levels.columns]
    if missing:
        raise KeyError(f"{component}: native panel is missing {missing}.")
    native = native_levels[list(variables)].copy().astype(float).sort_index()
    if not isinstance(native.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("native_levels must have a DatetimeIndex or PeriodIndex.")
    if isinstance(native.index, pd.PeriodIndex):
        native.index = native.index.to_timestamp(how="start")
    native.index = native.index.to_period("M").to_timestamp(how="start")
    native.index = pd.DatetimeIndex(native.index, name="date")
    if native.index.has_duplicates:
        raise ValueError(f"{component}: native panel contains duplicate months.")
    full_index = pd.date_range(native.index.min(), native.index.max(), freq="MS", name="date")
    native = native.reindex(full_index)
    if np.isinf(native.to_numpy(dtype=float)).any():
        raise ValueError(f"{component}: native panel contains infinite values.")
    return native


def _carry_trailing_state_edge(
    state_levels: pd.DataFrame,
    variables: Sequence[str],
    *,
    component: str,
    policy: str = "flat_native_level",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill only trailing driver gaps under an explicit conditioning policy.

    Interior gaps are never filled.  The current single-equation implementation
    treats regressors as exogenous, so the generic DK sampler cannot draw their
    missing values.  At the ragged edge we therefore hold the last native/state
    level fixed, equivalent to zero future model change.  Every filled month is
    surfaced in the returned audit table.
    """
    if policy != "flat_native_level":
        raise ValueError("Only policy='flat_native_level' is currently supported.")

    out = state_levels.copy()
    records = []
    for name in variables:
        series = out[name].copy()
        observed = series.dropna()
        if observed.empty:
            raise ValueError(f"{component}: driver {name!r} has no observations.")
        first = observed.index.min()
        last = observed.index.max()
        interior = series.loc[first:last]
        if interior.isna().any():
            first_bad = interior.index[interior.isna()][0]
            raise ValueError(
                f"{component}: driver {name!r} has an interior missing state at "
                f"{pd.Timestamp(first_bad).date()}; only trailing-edge conditioning "
                "is allowed for exact single-equation regressors."
            )
        trailing = series.index > last
        n_filled = int(trailing.sum())
        if n_filled:
            out.loc[trailing, name] = float(observed.iloc[-1])
        records.append(
            {
                "driver": name,
                "first_observed": pd.Timestamp(first),
                "last_observed": pd.Timestamp(last),
                "trailing_months_conditioned": n_filled,
                "edge_policy": policy,
            }
        )
    return out, pd.DataFrame(records).set_index("driver")


def build_headline_single_equation_design(
    native_levels: pd.DataFrame,
    component: str,
    *,
    predictor_edge_policy: str = "flat_native_level",
) -> dict[str, object]:
    """Build the exact WP374 ARX design for a supported single equation.

    The shared Energy engine receives only the transformed target state as its
    endogenous variable.  Variable-specific driver lags and seasonal dummies are
    passed through ``exog``.  The endogenous lag structure itself is supplied
    separately through ``spec["active_lags"]`` and enforced as exact fixed-zero
    restrictions by the shared engine.  This supports both processed food
    (target L1-L4) and NEIG (target L1,L6,L12) without creating a dense BVAR.

    External predictor lags and monthly dummies remain ordinary exogenous rows.
    """
    component = str(component).strip().lower()
    spec = headline_component_spec(component)
    predictors = spec.get("single_equation_predictors")
    if not predictors:
        raise ValueError(f"{component}: no sparse single-equation predictor contract.")

    variables = list(spec["variables"])
    target = str(spec["target"])
    native = _normalise_native_monthly_panel(native_levels, variables, component)
    all_state = native_to_state_levels(native, spec["transforms"], rename_state=False)

    driver_names = list(predictors)
    design_state, edge_audit = _carry_trailing_state_edge(
        all_state,
        driver_names,
        component=component,
        policy=predictor_edge_policy,
    )
    differences = design_state.diff()

    exog = pd.DataFrame(index=native.index)
    predictor_columns = []
    for name, lags in predictors.items():
        for lag in lags:
            lag = int(lag)
            if lag < 0:
                raise ValueError("Predictor lags must be non-negative.")
            column = f"{name}_L{lag}"
            exog[column] = differences[name].shift(lag)
            predictor_columns.append(column)

    seasonals = make_headline_seasonal_dummies(native.index, component)
    if seasonals is not None:
        exog = pd.concat([exog, seasonals], axis=1)

    valid = exog.notna().all(axis=1)
    if not valid.any():
        raise ValueError(f"{component}: no complete sparse-regressor date is available.")
    first_regression_date = pd.Timestamp(exog.index[valid][0])
    p = int(spec["p"])
    model_start = first_regression_date - pd.DateOffset(months=p)
    model_start = max(pd.Timestamp(native.index.min()), pd.Timestamp(model_start))

    # The first p exog rows are never used by the VAR regression.  Fill only
    # those warm-up rows with zero so the shared engine's strict exog validator
    # sees a complete calendar without altering the likelihood.
    engine_index = native.loc[model_start:].index
    engine_exog = exog.reindex(engine_index).copy()
    regression_mask = engine_exog.index >= first_regression_date
    if engine_exog.loc[regression_mask].isna().any().any():
        bad = engine_exog.loc[regression_mask].index[
            engine_exog.loc[regression_mask].isna().any(axis=1)
        ][0]
        raise ValueError(
            f"{component}: sparse exogenous design is incomplete at "
            f"{pd.Timestamp(bad).date()} after edge conditioning."
        )
    engine_exog.loc[~regression_mask] = engine_exog.loc[~regression_mask].fillna(0.0)
    if engine_exog.isna().any().any():
        raise RuntimeError(f"{component}: warm-up exog normalization failed.")

    target_state_name = _state_name(target)
    target_state = native_to_state_levels(
        native[[target]],
        {target: spec["transforms"][target]},
        rename_state=True,
    ).reindex(engine_index)

    # Keep native driver levels used by the conditioning policy for dashboard
    # transparency.  State->native inversion is exact for all supported rules.
    design_native = state_to_native_levels(
        design_state,
        spec["transforms"],
        state_names={name: name for name in spec["transforms"]},
    )

    return {
        "component": component,
        "spec": spec,
        "native_levels": native,
        "all_state_levels": all_state,
        "design_state_levels": design_state,
        "design_native_levels": design_native,
        "state_levels": target_state,
        "state_variable_map": {target: target_state_name},
        "engine_native_variables": [target],
        "exog": engine_exog,
        "predictor_columns": predictor_columns,
        "first_regression_date": first_regression_date,
        "model_start": pd.Timestamp(engine_index.min()),
        "predictor_edge_policy": predictor_edge_policy,
        "predictor_edge_audit": edge_audit,
    }


def prepare_headline_component(
    native_levels: pd.DataFrame,
    component: str,
    *,
    predictor_edge_policy: str = "flat_native_level",
) -> dict[str, object]:
    """Build the adapter context consumed by :func:`run_headline_bvar`."""
    component = str(component).strip().lower()
    spec = headline_component_spec(component)

    if spec.get("single_equation_predictors"):
        return build_headline_single_equation_design(
            native_levels,
            component,
            predictor_edge_policy=predictor_edge_policy,
        )

    variables = list(spec["variables"])
    native = _normalise_native_monthly_panel(native_levels, variables, component)
    state = native_to_state_levels(native, spec["transforms"], rename_state=True)
    exog = make_headline_seasonal_dummies(native.index, component)
    return {
        "component": component,
        "spec": spec,
        "native_levels": native,
        "state_levels": state,
        "all_state_levels": state.rename(
            columns={_state_name(name): name for name in variables}
        ),
        "state_variable_map": state_variable_map(component),
        "engine_native_variables": variables,
        "exog": exog,
        "predictor_edge_policy": None,
        "predictor_edge_audit": pd.DataFrame(),
    }

# -----------------------------------------------------------------------------
# Production Bayesian run wrapper
# -----------------------------------------------------------------------------


def _require_production_ready(spec: Mapping[str, object], component: str) -> None:
    if bool(spec.get("production_ready", False)):
        return
    blocker = spec.get("production_blocker") or "component adapter is not production-ready"
    raise NotImplementedError(f"{component}: {blocker}")


def run_headline_bvar(
    prepared: Mapping[str, object] | pd.DataFrame,
    *,
    component: str,
    vintage: str,
    prior_config: BVARSVOPriorConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    exog_prior_scale: float = 10.0,
    code_version: str = HEADLINE_ADAPTER_VERSION,
) -> dict:
    """Run one production-ready headline component through the shared engine.

    The shared engine itself is unchanged.  It receives *state levels* and
    therefore continues to estimate ordinary first differences internally.
    This adapter makes the native/log transformation explicit and reversible.
    """
    component = str(component).strip().lower()
    spec = headline_component_spec(component)
    _require_production_ready(spec, component)

    context = (
        prepare_headline_component(prepared, component)
        if isinstance(prepared, pd.DataFrame)
        else dict(prepared)
    )
    if str(context.get("component")) != component:
        raise ValueError(
            f"Prepared context is for {context.get('component')!r}, not {component!r}."
        )
    state_levels = context.get("state_levels")
    native_levels = context.get("native_levels")
    exog = context.get("exog")
    if not isinstance(state_levels, pd.DataFrame) or not isinstance(native_levels, pd.DataFrame):
        raise TypeError("Prepared context must contain DataFrame native_levels and state_levels.")

    engine_native_variables = list(
        context.get("engine_native_variables", spec["variables"])
    )
    state_variables = [_state_name(name) for name in engine_native_variables]
    if list(state_levels.columns) != state_variables:
        raise ValueError(
            f"{component}: engine state-level columns must be {state_variables}, "
            f"found {list(state_levels.columns)}."
        )

    result = run_energy_bvar(
        levels=state_levels,
        model_id=component,
        vintage=str(vintage),
        p=int(spec["p"]),
        variables=state_variables,
        frequency=str(spec["frequency"]),
        exog=exog,
        exog_prior_scale=float(exog_prior_scale),
        prior_config=prior_config,
        sampler_config=sampler_config,
        active_lags=spec.get("active_lags"),
        code_version=code_version,
    )

    # Non-serialized context used by notebooks and immediate dashboard runs.
    # The existing energy_bvar_io deliberately serializes only metadata and
    # posterior arrays, so adding this context does not alter Energy artifacts.
    result["headline_context"] = {
        "component": component,
        "native_levels": native_levels,
        "state_variable_map": dict(context["state_variable_map"]),
        "transforms": dict(spec["transforms"]),
        "target": str(spec["target"]),
        "label": str(spec["label"]),
        "engine_native_variables": engine_native_variables,
        "design_native_levels": context.get("design_native_levels"),
        "design_state_levels": context.get("design_state_levels"),
        "predictor_edge_policy": context.get("predictor_edge_policy"),
        "predictor_edge_audit": context.get("predictor_edge_audit"),
        "predictor_columns": list(context.get("predictor_columns", [])),
    }

    metadata = result.setdefault("metadata", {})
    metadata.update(
        {
            "suite": "headline",
            "component": component,
            "component_label": str(spec["label"]),
            "native_target": str(spec["target"]),
            "native_variables": list(spec["variables"]),
            "engine_native_variables": engine_native_variables,
            "state_variables": state_variables,
            "state_variable_map": dict(context["state_variable_map"]),
            "transform_map": dict(spec["transforms"]),
            "transformation_contract": (
                "native levels -> state levels (identity or log) -> shared "
                "engine first difference -> inverse state transform"
            ),
            "paper_reference": "ECB Working Paper 374 (2004)",
            "bayesian_engine_reference": "ECB Working Paper 3062 project implementation",
            "paper_model_class": str(spec["paper_model_class"]),
            "paper_lag_structure": str(spec["paper_lag_structure"]),
            "paper_active_lags": deepcopy(spec["paper_active_lags"]),
            "active_lags": (
                None if spec.get("active_lags") is None else list(spec["active_lags"])
            ),
            "implementation_mode": str(spec["implementation_mode"]),
            "predictor_edge_policy": context.get("predictor_edge_policy"),
            "predictor_columns": list(context.get("predictor_columns", [])),
            "headline_adapter_version": HEADLINE_ADAPTER_VERSION,
            "forecast_contract_version": HEADLINE_FORECAST_CONTRACT_VERSION,
        }
    )
    return result


# -----------------------------------------------------------------------------
# Forecast adapter and dashboard-compatible HICP contract
# -----------------------------------------------------------------------------


def _native_conditions_to_state(
    conditions: Mapping[str, float | Sequence[float]] | None,
    spec: Mapping[str, object],
    *,
    allowed_native_variables: Sequence[str] | None = None,
) -> dict[str, float | np.ndarray] | None:
    if conditions is None:
        return None
    transforms = dict(spec["transforms"])
    allowed = set(transforms if allowed_native_variables is None else allowed_native_variables)
    state_conditions: dict[str, float | np.ndarray] = {}
    for native_name, values in conditions.items():
        if native_name not in transforms or native_name not in allowed:
            raise KeyError(
                f"Unknown/endogenous-ineligible native condition variable "
                f"{native_name!r}; choose from {sorted(allowed)}."
            )
        kind = _validate_transform(transforms[native_name])
        arr = np.asarray(values, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"Condition for {native_name!r} contains non-finite values.")
        if kind == "logdiff":
            if np.any(arr <= 0):
                raise ValueError(
                    f"Native level conditions for {native_name!r} must be positive "
                    "under logdiff."
                )
            transformed = np.log(arr)
        elif kind == "percent_rate_diff":
            transformed = arr / 100.0
        else:
            transformed = arr
        state_conditions[_state_name(native_name)] = (
            float(transformed) if transformed.ndim == 0 else transformed
        )
    return state_conditions


def _native_values_to_state_array(
    native_name: str,
    values: np.ndarray,
    transform: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    kind = _validate_transform(transform)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Future native path for {native_name!r} must be finite.")
    if kind == "logdiff":
        if np.any(arr <= 0):
            raise ValueError(
                f"Future native path for {native_name!r} must be positive under logdiff."
            )
        return np.log(arr)
    if kind == "percent_rate_diff":
        return arr / 100.0
    return arr


def build_headline_future_exog(
    result: Mapping,
    *,
    component: str,
    H: int,
    native_driver_paths: Mapping[str, float | Sequence[float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Construct future ARX regressors from native driver assumptions.

    Default policy is a random walk / flat native level for every exogenous
    driver.  A user/dashboard scenario can instead supply a scalar or H-length
    native path for any driver.  Lagged predictor columns are then built from
    transformed *changes*, so inherited historical lags are handled correctly
    during the first forecast months.
    """
    component = str(component).strip().lower()
    spec = headline_component_spec(component)
    predictors = spec.get("single_equation_predictors")
    if not predictors:
        raise ValueError(f"{component}: no single-equation predictor contract.")
    if H < 1:
        raise ValueError("H must be at least 1.")

    context = result.get("headline_context", {})
    design_state = context.get("design_state_levels")
    design_native = context.get("design_native_levels")
    if not isinstance(design_state, pd.DataFrame) or not isinstance(design_native, pd.DataFrame):
        raise ValueError(
            "The fitted single-equation result is missing its driver design context. "
            "Supply a live result from run_headline_bvar; disk reconstruction will "
            "require persisted driver assumptions in a later registry extension."
        )

    last_date = pd.Timestamp(result["prep"]["last_calendar_date"])
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=int(H),
        freq="MS",
        name="date",
    )
    supplied = {} if native_driver_paths is None else dict(native_driver_paths)
    unknown = [name for name in supplied if name not in predictors]
    if unknown:
        raise KeyError(
            f"{component}: future driver path(s) {unknown} are not in "
            f"{list(predictors)}."
        )

    combined_state = design_state.copy()
    native_paths_used: dict[str, np.ndarray] = {}
    for name in predictors:
        history_native = pd.to_numeric(design_native[name], errors="coerce").dropna()
        if history_native.empty:
            raise ValueError(f"{component}: no usable history for driver {name!r}.")
        if name in supplied:
            arr = np.asarray(supplied[name], dtype=float)
            if arr.ndim == 0:
                arr = np.repeat(float(arr), H)
            if arr.ndim != 1 or len(arr) != H:
                raise ValueError(
                    f"Future driver path for {name!r} must be scalar or length H={H}."
                )
        else:
            arr = np.repeat(float(history_native.iloc[-1]), H)
        state_arr = _native_values_to_state_array(
            name,
            arr,
            str(spec["transforms"][name]),
        )
        native_paths_used[name] = arr.astype(float)
        extension = pd.Series(state_arr, index=future_dates, name=name)
        combined_state = combined_state.reindex(
            combined_state.index.union(future_dates).sort_values()
        )
        combined_state.loc[future_dates, name] = extension

    differences = combined_state.diff()
    future_exog = pd.DataFrame(index=future_dates)
    for name, lags in predictors.items():
        for lag in lags:
            lag = int(lag)
            column = f"{name}_L{lag}"
            future_exog[column] = differences[name].shift(lag).reindex(future_dates)

    seasonals = make_headline_seasonal_dummies(future_dates, component)
    if seasonals is not None:
        future_exog = pd.concat([future_exog, seasonals], axis=1)

    fitted_names = list(result["prep"].get("exog_names", []))
    missing = [name for name in fitted_names if name not in future_exog.columns]
    extra = [name for name in future_exog.columns if name not in fitted_names]
    if missing or extra:
        raise ValueError(
            f"{component}: future exog columns differ from fitted contract; "
            f"missing={missing}, extra={extra}."
        )
    future_exog = future_exog[fitted_names]
    if future_exog.isna().any().any():
        bad = future_exog.index[future_exog.isna().any(axis=1)][0]
        raise ValueError(
            f"{component}: future exogenous design is incomplete at "
            f"{pd.Timestamp(bad).date()}."
        )
    return future_exog.astype(float), native_paths_used

def _future_seasonals_for_result(
    result: Mapping,
    spec: Mapping[str, object],
    H: int,
) -> pd.DataFrame | None:
    names = list(result["prep"].get("exog_names", []))
    if not names:
        return None
    if not bool(spec["seasonal_dummies"]):
        raise ValueError(
            "The fitted result contains exogenous regressors, but the component "
            "specification does not declare seasonal dummies. Supply future_exog "
            "explicitly instead."
        )
    last_date = pd.Timestamp(result["prep"]["last_calendar_date"])
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1), periods=int(H), freq="MS", name="date"
    )
    dummies = monthly_seasonal_dummies(
        future_dates,
        reference_month=int(spec["seasonal_reference_month"]),
        prefix="month",
    )
    if list(dummies.columns) != names:
        raise ValueError(
            f"Automatic seasonal columns {list(dummies.columns)} do not match "
            f"fitted exogenous columns {names}."
        )
    return dummies


def _resolve_native_levels(
    result: Mapping,
    native_levels: pd.DataFrame | None,
    spec: Mapping[str, object],
) -> pd.DataFrame:
    if native_levels is not None:
        frame = native_levels.copy()
    else:
        context = result.get("headline_context", {})
        frame = context.get("native_levels")
        if frame is None:
            raise ValueError(
                "Native history is required to build HICP/YoY output. Supply "
                "native_levels when forecasting a result reconstructed from disk."
            )
        frame = frame.copy()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("native_levels must be a pandas DataFrame.")
    variables = list(spec["variables"])
    missing = [name for name in variables if name not in frame.columns]
    if missing:
        raise KeyError(f"Native history is missing {missing}.")
    frame = frame[variables].sort_index().astype(float)
    return frame


def headline_forecast_to_hicp(
    state_forecast: Mapping,
    result: Mapping,
    *,
    component: str,
    native_levels: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Convert a generic state-space forecast to the dashboard HICP contract."""
    component = str(component).strip().lower()
    spec = headline_component_spec(component)
    native_history = _resolve_native_levels(result, native_levels, spec)

    state_variables = list(state_forecast["variables"])
    reverse_state_map = {
        _state_name(name): name for name in spec["variables"]
    }
    unknown_states = [name for name in state_variables if name not in reverse_state_map]
    if unknown_states:
        raise KeyError(
            f"{component}: forecast contains unknown state variables {unknown_states}."
        )
    engine_native_variables = [reverse_state_map[name] for name in state_variables]
    native_paths = state_paths_to_native(
        np.asarray(state_forecast["level_paths"], dtype=float),
        native_variables=engine_native_variables,
        state_variables=state_variables,
        transforms={name: spec["transforms"][name] for name in engine_native_variables},
    )

    target = str(spec["target"])
    if target not in engine_native_variables:
        raise ValueError(f"{component}: target {target!r} is not an endogenous forecast state.")
    target_j = engine_native_variables.index(target)
    hicp_level_paths = native_paths[:, :, target_j]
    dates = pd.DatetimeIndex(state_forecast["path_dates"], name="date")
    if hicp_level_paths.shape[1] != len(dates):
        raise ValueError("Converted target paths and path_dates are inconsistent.")

    actual_hicp = native_history[target].astype(float).sort_index().rename(target)
    balanced_end = pd.Timestamp(result["prep"]["balanced_end"])
    history = actual_hicp.loc[actual_hicp.index <= balanced_end].dropna()
    if history.empty:
        raise ValueError("No observed target HICP history exists before balanced_end.")

    full_index = pd.date_range(
        min(history.index.min(), dates.min()),
        max(history.index.max(), dates.max()),
        freq="MS",
        name="date",
    )
    yoy_paths = np.full_like(hicp_level_paths, np.nan, dtype=float)
    for draw in range(len(hicp_level_paths)):
        series = history.reindex(full_index)
        series.loc[dates] = hicp_level_paths[draw]
        yoy = 100.0 * (series / series.shift(12) - 1.0)
        yoy_paths[draw] = yoy.reindex(dates).to_numpy(dtype=float)

    tail_length = int(state_forecast.get("tail_length", 0))
    future_dates = pd.DatetimeIndex(state_forecast["future_dates"], name="date")
    return {
        # Existing dashboard-facing HICP contract.
        "hicp_level_paths": hicp_level_paths,
        "hicp_yoy_paths": yoy_paths,
        "path_dates": dates,
        "future_dates": future_dates,
        "tail_length": tail_length,
        "future_hicp_level_paths": hicp_level_paths[:, tail_length:],
        "future_hicp_yoy_paths": yoy_paths[:, tail_length:],
        "actual_hicp": actual_hicp,
        "target_variable": target,
        "component": component,
        "component_label": str(spec["label"]),
        # General native-space contract for later dashboard/model reuse.
        "native_variables": engine_native_variables,
        "input_native_variables": list(spec["variables"]),
        "native_level_paths": native_paths,
        "future_native_level_paths": native_paths[:, tail_length:],
        "state_variables": state_variables,
        "state_forecast": state_forecast,
        "transform_map": dict(spec["transforms"]),
        "forecast_contract_version": HEADLINE_FORECAST_CONTRACT_VERSION,
    }


def forecast_headline_component(
    result: Mapping,
    *,
    component: str,
    H: int = 18,
    native_level_conditions: Mapping[str, float | Sequence[float]] | None = None,
    native_driver_paths: Mapping[str, float | Sequence[float]] | None = None,
    future_exog: pd.DataFrame | None = None,
    native_levels: pd.DataFrame | None = None,
    n_draws: int | None = None,
    simulate_future_outliers: bool = True,
    seed: int = 123,
) -> dict[str, object]:
    """Forecast a headline component and return native HICP level/YoY paths.

    For BVAR/AR components with only seasonal deterministics, future dummies are
    generated automatically.

    For exact WP374 dynamic single equations, exogenous driver paths are
    required conceptually.  The default baseline is a flat native level (random
    walk) for every driver; ``native_driver_paths`` can override any path in
    dashboard/scenario units.  The resulting lagged transformed changes are
    constructed automatically and passed as ``future_exog``.
    """
    component = str(component).strip().lower()
    spec = headline_component_spec(component)
    _require_production_ready(spec, component)
    if H < 1:
        raise ValueError("H must be at least 1 month.")

    driver_paths_used = None
    if future_exog is None:
        if spec.get("single_equation_predictors"):
            future_exog, driver_paths_used = build_headline_future_exog(
                result,
                component=component,
                H=int(H),
                native_driver_paths=native_driver_paths,
            )
        else:
            if native_driver_paths:
                raise ValueError(
                    f"{component}: native_driver_paths are only valid for a "
                    "single-equation component with exogenous drivers."
                )
            future_exog = _future_seasonals_for_result(result, spec, H)
    elif native_driver_paths is not None:
        raise ValueError(
            "Supply either future_exog or native_driver_paths, not both."
        )

    engine_native_variables = list(
        result.get("metadata", {}).get(
            "engine_native_variables",
            result.get("headline_context", {}).get(
                "engine_native_variables",
                spec["variables"],
            ),
        )
    )
    state_conditions = _native_conditions_to_state(
        native_level_conditions,
        spec,
        allowed_native_variables=engine_native_variables,
    )
    state_forecast = forecast_bvar_sv_outlier(
        result,
        H=int(H),
        level_conditions=state_conditions,
        future_exog=future_exog,
        n_draws=n_draws,
        simulate_future_outliers=bool(simulate_future_outliers),
        seed=int(seed),
    )
    output = headline_forecast_to_hicp(
        state_forecast,
        result,
        component=component,
        native_levels=native_levels,
    )
    output["native_level_conditions"] = (
        None if native_level_conditions is None else dict(native_level_conditions)
    )
    output["native_driver_paths"] = driver_paths_used
    output["driver_forecast_policy"] = (
        spec.get("forecast_driver_policy")
        if spec.get("single_equation_predictors")
        else None
    )
    return output

# -----------------------------------------------------------------------------
# Tables for notebooks/dashboard documentation
# -----------------------------------------------------------------------------


def headline_nowcast_summary_table(
    forecast: Mapping,
    *,
    kind: str = "level",
    quantiles: Sequence[float] = (5, 16, 50, 84, 95),
    missing_only: bool = True,
) -> pd.DataFrame:
    """Summarise the ragged-edge HICP portion of a headline forecast."""
    if kind == "level":
        paths = np.asarray(forecast["hicp_level_paths"], dtype=float)
        actual = pd.to_numeric(forecast["actual_hicp"], errors="coerce").astype(float)
    elif kind == "yoy":
        paths = np.asarray(forecast["hicp_yoy_paths"], dtype=float)
        actual_level = pd.to_numeric(forecast["actual_hicp"], errors="coerce").astype(float)
        actual = 100.0 * (actual_level / actual_level.shift(12) - 1.0)
    else:
        raise ValueError("kind must be 'level' or 'yoy'.")

    tail_length = int(forecast.get("tail_length", 0))
    columns = ["status", "observed", "q05", "q16", "median", "q84", "q95"]
    if tail_length == 0:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    dates = pd.DatetimeIndex(forecast["path_dates"][:tail_length], name="date")
    tail_paths = paths[:, :tail_length]
    q = np.nanpercentile(tail_paths, quantiles, axis=0)
    observed = actual.reindex(dates)
    status = np.where(observed.notna(), "observed / conditioned", "nowcast")
    table = pd.DataFrame(
        {
            "status": status,
            "observed": observed.to_numpy(dtype=float),
            "q05": q[0],
            "q16": q[1],
            "median": q[2],
            "q84": q[3],
            "q95": q[4],
        },
        index=dates,
    )
    return table.loc[table["status"] == "nowcast"] if missing_only else table


def headline_forecast_horizon_table(
    forecast: Mapping,
    *,
    kind: str = "yoy",
    horizons: Sequence[int] = (1, 3, 6, 12, 18),
    quantiles: Sequence[float] = (5, 16, 50, 84, 95),
) -> pd.DataFrame:
    """Return draw-wise HICP forecast summaries at selected monthly horizons."""
    if kind == "level":
        paths = np.asarray(forecast["future_hicp_level_paths"], dtype=float)
    elif kind == "yoy":
        paths = np.asarray(forecast["future_hicp_yoy_paths"], dtype=float)
    else:
        raise ValueError("kind must be 'level' or 'yoy'.")

    dates = pd.DatetimeIndex(forecast["future_dates"], name="date")
    if paths.ndim != 2 or paths.shape[1] != len(dates):
        raise ValueError("Future HICP paths and future_dates are inconsistent.")

    rows = []
    for horizon in horizons:
        h = int(horizon)
        if h < 1:
            raise ValueError("Forecast horizons must be positive integers.")
        if h > len(dates):
            continue
        values = paths[:, h - 1]
        q = np.nanpercentile(values, quantiles)
        rows.append(
            {
                "months_ahead": h,
                "target_date": dates[h - 1],
                "q05": q[0],
                "q16": q[1],
                "median": q[2],
                "q84": q[3],
                "q95": q[4],
            }
        )
    return pd.DataFrame(rows).set_index("months_ahead")


def headline_transformation_table(component: str) -> pd.DataFrame:
    """Describe native, state and model spaces for one component."""
    spec = headline_component_spec(component)
    rows = []
    target = str(spec["target"])
    active_lags = dict(spec["paper_active_lags"])
    for name in spec["variables"]:
        kind = str(spec["transforms"][name])
        rows.append(
            {
                "native_series": name,
                "role": "target" if name == target else "driver",
                "native_unit": spec["units"].get(name, "source unit"),
                "transform": kind,
                "state_level": (
                    f"log({name})"
                    if kind == "logdiff"
                    else (f"{name}/100" if kind == "percent_rate_diff" else name)
                ),
                "model_change": (
                    f"Δlog({name})"
                    if kind == "logdiff"
                    else (
                        f"Δ({name}/100)"
                        if kind == "percent_rate_diff"
                        else f"Δ{name}"
                    )
                ),
                "state_variable": _state_name(name),
                "paper_active_lags": active_lags.get(name, []),
            }
        )
    return pd.DataFrame(rows).set_index("native_series")


def headline_series_construction_table(
    dataset_path: str | Path,
    component: str,
) -> pd.DataFrame:
    """Document processed inputs and their model-layer transformations."""
    spec = headline_component_spec(component)
    path = Path(dataset_path)
    table = headline_transformation_table(component).copy()
    table.insert(0, "dataset_file", path.name)
    table["paper_model_class"] = str(spec["paper_model_class"])
    table["paper_lag_structure"] = str(spec["paper_lag_structure"])
    table["implementation_mode"] = str(spec["implementation_mode"])
    return table


# -----------------------------------------------------------------------------
# Exact WP374 benchmark for the first notebook
# -----------------------------------------------------------------------------


def wp374_unprocessed_food_sparse_benchmark(
    native_levels: pd.DataFrame | pd.Series,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    reference_month: int = 12,
):
    """Estimate the WP374 euro-area unprocessed-food sparse OLS benchmark.

    Specification:

        Δlog(HICP_UF)_t = c
                         + b1  Δlog(HICP_UF)_{t-1}
                         + b10 Δlog(HICP_UF)_{t-10}
                         + b12 Δlog(HICP_UF)_{t-12}
                         + eleven monthly dummies + error_t

    The function returns the statsmodels fit, the estimation frame, the three
    coefficients reported in WP374 Appendix A4, and a compact comparison table.
    """
    if isinstance(native_levels, pd.DataFrame):
        if "hicp_unprocessed_food" not in native_levels.columns:
            raise KeyError("native_levels must contain 'hicp_unprocessed_food'.")
        series = native_levels["hicp_unprocessed_food"]
    elif isinstance(native_levels, pd.Series):
        series = native_levels
    else:
        raise TypeError("native_levels must be a Series or DataFrame.")

    series = pd.to_numeric(series, errors="coerce").astype(float).sort_index()
    bad = series.notna() & (series <= 0)
    if bad.any():
        raise ValueError("Unprocessed-food HICP must be positive for log differences.")
    dlog = np.log(series).diff().rename("dlog_hicp_unprocessed_food")
    frame = pd.DataFrame({"y": dlog})
    for lag in (1, 10, 12):
        frame[f"L{lag}"] = dlog.shift(lag)

    dummies = monthly_seasonal_dummies(
        frame.index,
        reference_month=int(reference_month),
        prefix="SD",
    )
    dummy_rename = {
        col: f"SD{int(col.rsplit('_', 1)[1])}" for col in dummies.columns
    }
    dummies = dummies.rename(columns=dummy_rename)
    frame = pd.concat([frame, dummies], axis=1)
    if start is not None:
        frame = frame.loc[pd.Timestamp(start):]
    if end is not None:
        frame = frame.loc[:pd.Timestamp(end)]
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("No complete observations remain for the sparse benchmark.")

    regressors = ["L1", "L10", "L12", *list(dummies.columns)]
    X = sm.add_constant(frame[regressors], has_constant="add")
    fit = sm.OLS(frame["y"], X).fit()

    reported = pd.Series(
        {"L1": 0.2254, "L10": 0.2325, "L12": -0.1670},
        name="WP374_reported",
        dtype=float,
    )
    estimated = fit.params[["L1", "L10", "L12"]].rename("current_sample")
    comparison = pd.concat([reported, estimated], axis=1)
    return {
        "fit": fit,
        "data": frame,
        "reported_coefficients": reported,
        "comparison": comparison,
        "reference_month": int(reference_month),
        "active_lags": (1, 10, 12),
    }


def wp374_processed_food_sparse_benchmark(
    native_levels: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    reference_month: int = 12,
):
    """Estimate the exact WP374 processed-food dynamic single-equation OLS.

    Model-space normalisation used by this project:
      * HICP processed food, food commodities, wages: first log differences;
      * VAT: first difference of the rate expressed as a share, so +1 pp = +0.01.

    The lag structure is exactly the selected euro-area equation in WP374:
    target L1-L4, food commodity L2, wages L0-L2, VAT L0, seasonal dummies.
    """
    component = "processed_food"
    spec = headline_component_spec(component)
    native = _normalise_native_monthly_panel(
        native_levels,
        spec["variables"],
        component,
    )
    state = native_to_state_levels(native, spec["transforms"], rename_state=False)
    d = state.diff()

    frame = pd.DataFrame({"y": d["hicp_processed_food"]}, index=native.index)
    for lag in (1, 2, 3, 4):
        frame[f"HICP_L{lag}"] = d["hicp_processed_food"].shift(lag)
    frame["COMFD_L2"] = d["food_commodity_price"].shift(2)
    for lag in (0, 1, 2):
        frame[f"WAGES_L{lag}"] = d["compensation_per_employee"].shift(lag)
    frame["VAT_L0"] = d["vat_standard_ea"]

    dummies = monthly_seasonal_dummies(
        frame.index,
        reference_month=int(reference_month),
        prefix="SD",
    ).rename(
        columns=lambda col: f"SD{int(str(col).rsplit('_', 1)[1])}"
    )
    frame = pd.concat([frame, dummies], axis=1)
    if start is not None:
        frame = frame.loc[pd.Timestamp(start):]
    if end is not None:
        frame = frame.loc[:pd.Timestamp(end)]
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("No complete observations remain for processed-food benchmark.")

    regressors = [
        "HICP_L1", "HICP_L2", "HICP_L3", "HICP_L4",
        "COMFD_L2",
        "WAGES_L0", "WAGES_L1", "WAGES_L2",
        "VAT_L0",
        *list(dummies.columns),
    ]
    X = sm.add_constant(frame[regressors], has_constant="add")
    fit = sm.OLS(frame["y"], X).fit()

    reported = pd.Series(
        {
            "HICP_L1": 0.2654,
            "HICP_L2": 0.0165,
            "HICP_L3": 0.2690,
            "HICP_L4": 0.0078,
            "COMFD_L2": 0.0055,
            "WAGES_L0": 0.1827,
            "WAGES_L1": -0.2024,
            "WAGES_L2": 0.1710,
            "VAT_L0": 0.0453,
        },
        name="WP374_reported",
        dtype=float,
    )
    estimated = fit.params[reported.index].rename("current_sample")
    return {
        "fit": fit,
        "data": frame,
        "reported_coefficients": reported,
        "comparison": pd.concat([reported, estimated], axis=1),
        "reference_month": int(reference_month),
        "active_lags": deepcopy(spec["paper_active_lags"]),
        "vat_normalisation": "native percentage points / 100 before first difference",
    }


def headline_driver_assumption_table(
    result: Mapping,
) -> pd.DataFrame:
    """Return the exogenous-driver edge-conditioning audit for a fitted model."""
    context = result.get("headline_context", {})
    table = context.get("predictor_edge_audit")
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.DataFrame(
        columns=[
            "first_observed",
            "last_observed",
            "trailing_months_conditioned",
            "edge_policy",
        ]
    )

def wp374_processed_food_posterior_comparison(result: Mapping) -> pd.DataFrame:
    """Compare processed-food posterior coefficients with WP374 Appendix A4."""
    from energy_bvar_model import coefficient_posterior_table

    table = coefficient_posterior_table(result)
    eq = _state_name("hicp_processed_food")
    label_map = {
        "HICP_L1": f"{eq}: {eq} L1",
        "HICP_L2": f"{eq}: {eq} L2",
        "HICP_L3": f"{eq}: {eq} L3",
        "HICP_L4": f"{eq}: {eq} L4",
        "COMFD_L2": f"{eq}: food_commodity_price_L2",
        "WAGES_L0": f"{eq}: compensation_per_employee_L0",
        "WAGES_L1": f"{eq}: compensation_per_employee_L1",
        "WAGES_L2": f"{eq}: compensation_per_employee_L2",
        "VAT_L0": f"{eq}: vat_standard_ea_L0",
    }
    missing = [label for label in label_map.values() if label not in table.index]
    if missing:
        raise KeyError(f"Processed-food posterior table is missing {missing}.")
    wp = pd.Series(
        {
            "HICP_L1": 0.2654,
            "HICP_L2": 0.0165,
            "HICP_L3": 0.2690,
            "HICP_L4": 0.0078,
            "COMFD_L2": 0.0055,
            "WAGES_L0": 0.1827,
            "WAGES_L1": -0.2024,
            "WAGES_L2": 0.1710,
            "VAT_L0": 0.0453,
        },
        name="WP374_reported",
    )
    rows=[]
    for short, full in label_map.items():
        r=table.loc[full]
        rows.append({
            "term": short,
            "WP374_reported": float(wp[short]),
            "posterior_median": float(r["posterior_median"]),
            "q16": float(r["q16"]),
            "q84": float(r["q84"]),
            "posterior_sd": float(r["posterior_sd"]),
            "ESS": float(r["ESS"]),
        })
    return pd.DataFrame(rows).set_index("term")


def wp374_neig_sparse_benchmark(
    native_levels: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    reference_month: int = 12,
):
    """Estimate the exact WP374 euro-area NEIG dynamic single equation by OLS.

    Appendix A4 reports the selected 1990:01--2002:06 equation with coefficients
    approximately:
      HICP L1 -0.0488, L6 0.4536, L12 0.3302,
      PPI consumer goods L1 0.1301, wages L2 0.0682, VAT L0 0.0579.
    """
    component = "neig"
    spec = headline_component_spec(component)
    native = _normalise_native_monthly_panel(native_levels, spec["variables"], component)
    state = native_to_state_levels(native, spec["transforms"], rename_state=False)
    d = state.diff()

    frame = pd.DataFrame({"y": d["hicp_neig"]}, index=native.index)
    for lag in (1, 6, 12):
        frame[f"HICP_L{lag}"] = d["hicp_neig"].shift(lag)
    frame["PPI_L1"] = d["ppi_consumer_goods"].shift(1)
    frame["WAGES_L2"] = d["compensation_per_employee"].shift(2)
    frame["VAT_L0"] = d["vat_standard_ea"]

    dummies = monthly_seasonal_dummies(
        frame.index,
        reference_month=int(reference_month),
        prefix="SD",
    ).rename(columns=lambda col: f"SD{int(str(col).rsplit('_', 1)[1])}")
    frame = pd.concat([frame, dummies], axis=1)
    if start is not None:
        frame = frame.loc[pd.Timestamp(start):]
    if end is not None:
        frame = frame.loc[:pd.Timestamp(end)]
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("No complete observations remain for the NEIG benchmark.")

    regressors = [
        "HICP_L1", "HICP_L6", "HICP_L12",
        "PPI_L1", "WAGES_L2", "VAT_L0",
        *list(dummies.columns),
    ]
    X = sm.add_constant(frame[regressors], has_constant="add")
    fit = sm.OLS(frame["y"], X).fit()
    reported = pd.Series(
        {
            "HICP_L1": -0.0488,
            "HICP_L6": 0.4536,
            "HICP_L12": 0.3302,
            "PPI_L1": 0.1301,
            "WAGES_L2": 0.0682,
            "VAT_L0": 0.0579,
        },
        name="WP374_reported",
        dtype=float,
    )
    estimated = fit.params[reported.index].rename("current_sample")
    return {
        "fit": fit,
        "data": frame,
        "reported_coefficients": reported,
        "comparison": pd.concat([reported, estimated], axis=1),
        "reference_month": int(reference_month),
        "active_lags": [1, 6, 12],
        "vat_normalisation": "native percentage points / 100 before first difference",
    }


def wp374_neig_posterior_comparison(result: Mapping) -> pd.DataFrame:
    """Compare the exact sparse NEIG Bayesian posterior with Appendix A4."""
    from energy_bvar_model import coefficient_posterior_table

    table = coefficient_posterior_table(result)
    eq = _state_name("hicp_neig")
    label_map = {
        "HICP_L1": f"{eq}: {eq} L1",
        "HICP_L6": f"{eq}: {eq} L6",
        "HICP_L12": f"{eq}: {eq} L12",
        "PPI_L1": f"{eq}: ppi_consumer_goods_L1",
        "WAGES_L2": f"{eq}: compensation_per_employee_L2",
        "VAT_L0": f"{eq}: vat_standard_ea_L0",
    }
    missing = [label for label in label_map.values() if label not in table.index]
    if missing:
        raise KeyError(f"NEIG posterior table is missing {missing}.")
    wp = pd.Series(
        {
            "HICP_L1": -0.0488,
            "HICP_L6": 0.4536,
            "HICP_L12": 0.3302,
            "PPI_L1": 0.1301,
            "WAGES_L2": 0.0682,
            "VAT_L0": 0.0579,
        },
        name="WP374_reported",
    )
    rows = []
    for short, full in label_map.items():
        r = table.loc[full]
        rows.append(
            {
                "term": short,
                "WP374_reported": float(wp[short]),
                "posterior_median": float(r["posterior_median"]),
                "q16": float(r["q16"]),
                "q84": float(r["q84"]),
                "posterior_sd": float(r["posterior_sd"]),
                "ESS": float(r["ESS"]),
            }
        )
    return pd.DataFrame(rows).set_index("term")


def sparse_lag_restriction_table(result: Mapping) -> pd.DataFrame:
    """Audit exact zero restrictions on omitted endogenous lags."""
    p = int(result["p"])
    active = result.get("active_lags")
    if active is None:
        active = result.get("metadata", {}).get("active_lags")
    active_set = set(range(1, p + 1)) if active is None else set(map(int, active))
    B = np.asarray(result["B"], dtype=float)
    n = len(result["variables"])
    rows = []
    for lag in range(1, p + 1):
        block = B[:, 1 + (lag - 1) * n : 1 + lag * n, :]
        rows.append(
            {
                "lag": lag,
                "status": "estimated" if lag in active_set else "fixed_zero",
                "max_abs_posterior_coefficient": float(np.max(np.abs(block))),
            }
        )
    return pd.DataFrame(rows).set_index("lag")


def headline_scenario_impact_table(
    baseline: Mapping,
    scenario: Mapping,
    *,
    kind: str = "yoy",
    horizons: Sequence[int] = (1, 2, 3, 6, 12, 18),
) -> pd.DataFrame:
    """Summarise draw-wise scenario-minus-baseline effects at selected horizons."""
    if kind == "yoy":
        base = np.asarray(baseline["future_hicp_yoy_paths"], dtype=float)
        alt = np.asarray(scenario["future_hicp_yoy_paths"], dtype=float)
    elif kind == "level":
        base = np.asarray(baseline["future_hicp_level_paths"], dtype=float)
        alt = np.asarray(scenario["future_hicp_level_paths"], dtype=float)
    else:
        raise ValueError("kind must be 'yoy' or 'level'.")
    if base.shape != alt.shape:
        raise ValueError("Baseline and scenario path arrays must have identical shapes.")
    base_dates = pd.DatetimeIndex(baseline["future_dates"], name="date")
    alt_dates = pd.DatetimeIndex(scenario["future_dates"], name="date")
    if not base_dates.equals(alt_dates):
        raise ValueError("Baseline and scenario future dates differ.")
    delta = alt - base
    rows=[]
    for horizon in horizons:
        h=int(horizon)
        if h<1 or h>len(base_dates):
            continue
        values=delta[:,h-1]
        q05,q16,q50,q84,q95=np.nanpercentile(values,[5,16,50,84,95])
        rows.append({
            "months_ahead":h,
            "target_date":base_dates[h-1],
            "q05":q05,"q16":q16,"median":q50,"q84":q84,"q95":q95,
        })
    return pd.DataFrame(rows).set_index("months_ahead")

def headline_native_nowcast_summary_table(
    forecast: Mapping,
    native_levels: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (5, 16, 50, 84, 95),
    missing_only: bool = True,
) -> pd.DataFrame:
    """Summarise native-space ragged-edge draws for every endogenous variable.

    This is useful for multivariate Headline models such as Services, where the
    HICP target can be observed through the latest month while quarterly wages
    or a PPI series still require DK data augmentation at the ragged edge.
    """
    paths = np.asarray(forecast["native_level_paths"], dtype=float)
    variables = list(forecast["native_variables"])
    dates = pd.DatetimeIndex(forecast["path_dates"], name="date")
    tail_length = int(forecast.get("tail_length", 0))
    if paths.ndim != 3 or paths.shape[1] != len(dates) or paths.shape[2] != len(variables):
        raise ValueError("native_level_paths have incompatible dimensions.")
    if tail_length < 1:
        return pd.DataFrame(
            columns=[
                "date", "variable", "observed_native",
                *[f"q{int(q):02d}" for q in quantiles],
            ]
        ).set_index(["date", "variable"])

    observed = native_levels.reindex(dates[:tail_length])
    rows = []
    for t, date in enumerate(dates[:tail_length]):
        for j, variable in enumerate(variables):
            obs = (
                float(observed.loc[date, variable])
                if variable in observed.columns
                and pd.notna(observed.loc[date, variable])
                else np.nan
            )
            if missing_only and np.isfinite(obs):
                continue
            qvals = np.nanpercentile(paths[:, t, j], quantiles)
            row = {
                "date": pd.Timestamp(date),
                "variable": variable,
                "observed_native": obs,
            }
            row.update(
                {
                    f"q{int(q):02d}": float(value)
                    for q, value in zip(quantiles, qvals)
                }
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "date", "variable", "observed_native",
                *[f"q{int(q):02d}" for q in quantiles],
            ]
        ).set_index(["date", "variable"])
    return pd.DataFrame(rows).set_index(["date", "variable"]).sort_index()


def headline_native_condition_path(
    forecast: Mapping,
    variable: str,
    *,
    quantile: float = 0.50,
    future_only: bool = True,
) -> pd.Series:
    """Extract one deterministic native path from posterior forecast draws.

    The helper is deliberately generic so a dashboard can pass the median path
    from one component model as a level condition to another model.  For
    example, the Services BVAR can be conditioned on the median unprocessed-food
    path produced by the dedicated Unprocessed Food model, reproducing the
    hierarchy used in WP374 without exposing log/state transformations to the UI.
    """
    q = float(quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    variables = list(forecast["native_variables"])
    if variable not in variables:
        raise KeyError(
            f"{variable!r} is not in forecast native_variables={variables}."
        )
    j = variables.index(variable)
    if future_only:
        paths = np.asarray(forecast["future_native_level_paths"], dtype=float)
        dates = pd.DatetimeIndex(forecast["future_dates"], name="date")
    else:
        paths = np.asarray(forecast["native_level_paths"], dtype=float)
        dates = pd.DatetimeIndex(forecast["path_dates"], name="date")
    if paths.ndim != 3 or paths.shape[1] != len(dates):
        raise ValueError("Forecast native paths and dates are inconsistent.")
    values = np.nanquantile(paths[:, :, j], q, axis=0)
    return pd.Series(values, index=dates, name=variable)


def headline_native_condition_paths(
    forecast: Mapping,
    variables: Sequence[str],
    *,
    quantile: float = 0.50,
) -> dict[str, np.ndarray]:
    """Return future native conditioning arrays for several variables."""
    return {
        str(variable): headline_native_condition_path(
            forecast,
            str(variable),
            quantile=quantile,
            future_only=True,
        ).to_numpy(dtype=float)
        for variable in variables
    }


def wp374_services_reference_table() -> pd.DataFrame:
    """Return the selected euro-area Services BVAR contract from WP374."""
    return pd.DataFrame(
        [
            {
                "item": "model_class",
                "WP374": "BVAR",
                "production": "BVAR-SV-outlier",
            },
            {
                "item": "variables",
                "WP374": (
                    "HICP services; compensation per employee; "
                    "PPI consumer goods; HICP unprocessed food"
                ),
                "production": (
                    "HICP services; compensation per employee; "
                    "PPI consumer goods; HICP unprocessed food"
                ),
            },
            {
                "item": "common_lags",
                "WP374": "1,2,3,4,5,12",
                "production": "1,2,3,4,5,12 (exact fixed-zero restriction otherwise)",
            },
            {
                "item": "tightness",
                "WP374": 0.1,
                "production": 0.20,
            },
            {
                "item": "cross_variable_weight",
                "WP374": 0.9,
                "production": 0.50,
            },
            {
                "item": "lag_decay",
                "WP374": 0.1,
                "production": 1.00,
            },
            {
                "item": "volatility",
                "WP374": "constant",
                "production": "stochastic volatility + transitory outliers",
            },
        ]
    ).set_index("item")


def wp374_services_response_benchmark_table() -> pd.DataFrame:
    """Published cumulative Services-inflation responses reported in WP374."""
    return pd.DataFrame(
        [
            {
                "driver": "compensation_per_employee",
                "change": "+1% level",
                "horizon_months": 2,
                "published_service_inflation_effect_pp": 0.09,
            },
            {
                "driver": "compensation_per_employee",
                "change": "+1% level",
                "horizon_months": 6,
                "published_service_inflation_effect_pp": 0.24,
            },
            {
                "driver": "ppi_consumer_goods",
                "change": "+1% level",
                "horizon_months": 6,
                "published_service_inflation_effect_pp": 0.09,
            },
            {
                "driver": "hicp_unprocessed_food",
                "change": "+1% level",
                "horizon_months": 6,
                "published_service_inflation_effect_pp": 0.07,
            },
        ]
    ).set_index(["driver", "horizon_months"])


def compare_services_scenario_to_wp374(
    baseline: Mapping,
    scenario: Mapping,
    *,
    driver: str,
    horizons: Sequence[int] = (2, 6),
) -> pd.DataFrame:
    """Compare modern conditional-path effects with WP374 published responses.

    This is a diagnostic comparison, not an equality test: the modern model has
    a different sample, stochastic volatility/outliers and Energy-suite priors.
    """
    modern = headline_scenario_impact_table(
        baseline,
        scenario,
        kind="yoy",
        horizons=horizons,
    ).copy()
    reference = wp374_services_response_benchmark_table()
    rows = []
    for h, row in modern.iterrows():
        key = (str(driver), int(h))
        published = (
            float(reference.loc[key, "published_service_inflation_effect_pp"])
            if key in reference.index
            else np.nan
        )
        rows.append(
            {
                "months_ahead": int(h),
                "target_date": row["target_date"],
                "modern_median_pp": float(row["median"]),
                "modern_q16_pp": float(row["q16"]),
                "modern_q84_pp": float(row["q84"]),
                "WP374_published_pp": published,
            }
        )
    return pd.DataFrame(rows).set_index("months_ahead")

__all__ = [
    "BVARSVOPriorConfig",
    "SamplerConfig",
    "HEADLINE_ADAPTER_VERSION",
    "HEADLINE_COMPONENT_SPECS",
    "HEADLINE_FORECAST_CONTRACT_VERSION",
    "SUPPORTED_TRANSFORMS",
    "headline_component_spec",
    "headline_component_contract_table",
    "state_variable_map",
    "resolve_headline_component_dataset",
    "headline_panel_audit",
    "load_headline_component_panel",
    "native_to_state_levels",
    "state_to_native_levels",
    "state_paths_to_native",
    "native_model_differences",
    "make_headline_seasonal_dummies",
    "build_headline_single_equation_design",
    "prepare_headline_component",
    "run_headline_bvar",
    "build_headline_future_exog",
    "headline_forecast_to_hicp",
    "forecast_headline_component",
    "headline_nowcast_summary_table",
    "headline_forecast_horizon_table",
    "headline_transformation_table",
    "headline_series_construction_table",
    "wp374_unprocessed_food_sparse_benchmark",
    "wp374_processed_food_sparse_benchmark",
    "headline_driver_assumption_table",
    "wp374_processed_food_posterior_comparison",
    "wp374_neig_sparse_benchmark",
    "wp374_neig_posterior_comparison",
    "sparse_lag_restriction_table",
    "headline_scenario_impact_table",
    "headline_native_nowcast_summary_table",
    "headline_native_condition_path",
    "headline_native_condition_paths",
    "wp374_services_reference_table",
    "wp374_services_response_benchmark_table",
    "compare_services_scenario_to_wp374",
]
