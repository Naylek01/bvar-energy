"""Joint monthly BVAR for euro-area Headline HICP components.

Current production model
------------------------
One joint monthly BVAR over four aggregate HICP indices:

    HICP Energy
    HICP Food
    HICP NEIG
    HICP Services

All four native positive indices are mapped to log state levels.  The shared
``energy_bvar_model`` engine then works on first differences of those states,
i.e. monthly log changes.

Canonical specification
-----------------------
* baseline: dense BVAR(12), ``active_lags=None``;
* only lag sensitivity: dense BVAR(6);
* Bayesian machinery: the validated Energy-suite Minnesota/SV/outlier/DK
  implementation.

This module deliberately does NOT contain the deferred component-decomposition
logic (VAT, wages, PPI, food commodities, PF/UF ARX models).  That logic remains
in ``headline_bvar_model.py``.

Aggregation
-----------
The four joint posterior component-index paths are aggregated draw-by-draw with
annual HICP weights using a December-rebased chain-link formula.  Food's annual
weight is constructed upstream as Processed Food + Unprocessed Food.  Forecast
years beyond the latest published weight year carry forward the latest complete
weight vector, matching the established project policy.

The official total HICP is not endogenous in the BVAR.  It is used to anchor
the first aggregation year and to validate historical reconstruction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from energy_bvar_model import (
    BVARSVOPriorConfig,
    SamplerConfig,
    forecast_bvar_sv_outlier,
    run_energy_bvar,
)


HEADLINE_JOINT_VERSION = "2026-08-11-v1.1"

MODEL_ID = "headline_joint"
FREQUENCY = "monthly"
BASELINE_P = 12
SENSITIVITY_P = 6
ALLOWED_LAG_ORDERS = (6, 12)

NATIVE_VARIABLES = [
    "hicp_energy",
    "hicp_food",
    "hicp_neig",
    "hicp_services",
]
STATE_VARIABLES = [f"state__{name}" for name in NATIVE_VARIABLES]
STATE_VARIABLE_MAP = dict(zip(NATIVE_VARIABLES, STATE_VARIABLES))

DATASET_FILE = "headline_joint_monthly.csv"
WEIGHTS_FILE = "headline_joint_weights_annual.csv"
HEADLINE_INDICES_FILE = "headline_hicp_indices_monthly.csv"

WEIGHT_TOLERANCE_PER_THOUSAND = 0.10


def headline_joint_spec() -> dict[str, object]:
    """Return a defensive copy of the current production model contract."""
    return {
        "model_id": MODEL_ID,
        "frequency": FREQUENCY,
        "native_variables": list(NATIVE_VARIABLES),
        "state_variables": list(STATE_VARIABLES),
        "state_variable_map": dict(STATE_VARIABLE_MAP),
        "transforms": {name: "logdiff" for name in NATIVE_VARIABLES},
        "baseline_p": BASELINE_P,
        "sensitivity_p": SENSITIVITY_P,
        "allowed_lag_orders": list(ALLOWED_LAG_ORDERS),
        "active_lags": None,
        "dataset_file": DATASET_FILE,
        "weights_file": WEIGHTS_FILE,
        "official_total_file": HEADLINE_INDICES_FILE,
        "bayesian_engine": "energy_bvar_model",
        "headline_joint_version": HEADLINE_JOINT_VERSION,
    }


def _normalise_monthly_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("Expected a DatetimeIndex or PeriodIndex.")
    if isinstance(index, pd.PeriodIndex):
        idx = index.to_timestamp(how="start")
    else:
        idx = pd.DatetimeIndex(index)
    idx = idx.to_period("M").to_timestamp(how="start")
    return pd.DatetimeIndex(idx, name="date")


def _read_monthly_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path.name}: required 'date' column is missing.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError(f"{path.name}: invalid date value found.")
    frame = frame.set_index("date").sort_index()
    frame.index = _normalise_monthly_index(frame.index)
    if frame.index.has_duplicates:
        raise ValueError(f"{path.name}: duplicate monthly dates.")
    return frame


def resolve_headline_joint_dataset(
    project_root: str | Path,
    *,
    vintage: str | None = None,
    data_root: str | Path | None = None,
) -> Path:
    """Resolve ``headline_joint_monthly.csv`` deterministically."""
    root = Path(project_root).resolve()
    processed = (
        Path(data_root).resolve()
        if data_root is not None
        else root / "data" / "processed"
    )
    if vintage is not None:
        path = processed / str(vintage) / DATASET_FILE
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(p for p in processed.glob(f"*/{DATASET_FILE}") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No data/processed/<vintage>/{DATASET_FILE} found below {processed}."
        )
    return candidates[-1]


def resolve_headline_joint_weights(
    project_root: str | Path,
    *,
    vintage: str | None = None,
    data_root: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    processed = (
        Path(data_root).resolve()
        if data_root is not None
        else root / "data" / "processed"
    )
    if vintage is not None:
        path = processed / str(vintage) / WEIGHTS_FILE
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(p for p in processed.glob(f"*/{WEIGHTS_FILE}") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No data/processed/<vintage>/{WEIGHTS_FILE} found below {processed}."
        )
    return candidates[-1]


def resolve_headline_official_indices(
    project_root: str | Path,
    *,
    vintage: str | None = None,
    data_root: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    processed = (
        Path(data_root).resolve()
        if data_root is not None
        else root / "data" / "processed"
    )
    if vintage is not None:
        path = processed / str(vintage) / HEADLINE_INDICES_FILE
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(
        p for p in processed.glob(f"*/{HEADLINE_INDICES_FILE}") if p.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No data/processed/<vintage>/{HEADLINE_INDICES_FILE} found below "
            f"{processed}."
        )
    return candidates[-1]


def load_headline_joint_panel(path: str | Path) -> pd.DataFrame:
    """Load and validate the four native HICP aggregate indices."""
    frame = _read_monthly_csv(path)
    missing = [name for name in NATIVE_VARIABLES if name not in frame.columns]
    if missing:
        raise KeyError(f"{Path(path).name}: missing model columns {missing}.")
    panel = frame[NATIVE_VARIABLES].apply(pd.to_numeric, errors="coerce").astype(float)

    expected = pd.date_range(panel.index.min(), panel.index.max(), freq="MS", name="date")
    if not panel.index.equals(expected):
        raise ValueError("headline_joint: calendar index is not monthly regular.")
    if np.isinf(panel.to_numpy(dtype=float)).any():
        raise ValueError("headline_joint: infinite value found.")
    for name in NATIVE_VARIABLES:
        observed = panel[name].dropna()
        if observed.empty:
            raise ValueError(f"headline_joint: {name} is entirely missing.")
        bad = observed <= 0
        if bad.any():
            first = bad[bad].index[0]
            raise ValueError(
                f"headline_joint: {name} must be positive; first invalid value "
                f"is at {pd.Timestamp(first).date()}."
            )
        inside = panel[name].loc[observed.index.min() : observed.index.max()]
        if inside.isna().any():
            first = inside.index[inside.isna()][0]
            raise ValueError(
                f"headline_joint: {name} has an interior missing native HICP "
                f"observation at {pd.Timestamp(first).date()}. Official aggregate "
                "HICP inputs must be calendar-continuous; only a trailing ragged "
                "edge is allowed."
            )
    return panel


def load_headline_joint_weights(path: str | Path) -> pd.DataFrame:
    """Load annual four-component weights in per-thousand units."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    if "year" not in raw.columns:
        # tolerate pandas writing an unnamed integer index
        first = raw.columns[0]
        raw = raw.rename(columns={first: "year"})
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    if raw["year"].isna().any():
        raise ValueError(f"{path.name}: invalid annual weight year.")
    raw["year"] = raw["year"].astype(int)
    if raw["year"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate annual weight year.")

    missing = [name for name in NATIVE_VARIABLES if name not in raw.columns]
    if missing:
        raise KeyError(f"{path.name}: missing joint weight columns {missing}.")
    weights = (
        raw.set_index("year")[NATIVE_VARIABLES]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .sort_index()
    )
    if np.isinf(weights.to_numpy(dtype=float)).any():
        raise ValueError(f"{path.name}: infinite annual weight found.")
    if (weights.dropna(how="all") < 0).any().any():
        raise ValueError(f"{path.name}: negative HICP weight found.")

    complete = weights.dropna(how="any")
    if complete.empty:
        raise ValueError(f"{path.name}: no complete four-component weight year.")
    error = complete.sum(axis=1) - 1000.0
    bad = error.abs() > WEIGHT_TOLERANCE_PER_THOUSAND
    if bad.any():
        year = int(bad[bad].index[0])
        raise ValueError(
            f"{path.name}: four-component weights sum to "
            f"{complete.loc[year].sum():.6f}, not 1000, in {year}."
        )
    return weights


def load_official_headline_total(path: str | Path) -> pd.Series:
    """Load official total HICP from ``headline_hicp_indices_monthly.csv``."""
    frame = _read_monthly_csv(path)
    if "hicp_total" not in frame.columns:
        raise KeyError(f"{Path(path).name}: hicp_total column is missing.")
    series = pd.to_numeric(frame["hicp_total"], errors="coerce").astype(float)
    series.name = "hicp_total"
    observed = series.dropna()
    if observed.empty:
        raise ValueError("Official HICP Total is entirely missing.")
    if (observed <= 0).any():
        raise ValueError("Official HICP Total must be strictly positive.")
    return series


def headline_joint_panel_audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Return per-series coverage/missingness diagnostics."""
    panel = panel[NATIVE_VARIABLES]
    rows = []
    for name in NATIVE_VARIABLES:
        s = panel[name]
        observed = s.dropna()
        inside = (
            s.loc[observed.index.min() : observed.index.max()]
            if not observed.empty
            else s.iloc[:0]
        )
        rows.append(
            {
                "series": name,
                "first_observed": (
                    observed.index.min() if not observed.empty else pd.NaT
                ),
                "last_observed": (
                    observed.index.max() if not observed.empty else pd.NaT
                ),
                "observations": int(observed.size),
                "leading_missing": (
                    int((s.index < observed.index.min()).sum())
                    if not observed.empty else int(len(s))
                ),
                "interior_missing": int(inside.isna().sum()),
                "trailing_missing": (
                    int((s.index > observed.index.max()).sum())
                    if not observed.empty else int(len(s))
                ),
                "min": float(observed.min()) if not observed.empty else np.nan,
                "max": float(observed.max()) if not observed.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("series")


def native_to_state_levels(native_levels: pd.DataFrame) -> pd.DataFrame:
    """Map four positive native HICP indices to log state levels."""
    missing = [name for name in NATIVE_VARIABLES if name not in native_levels.columns]
    if missing:
        raise KeyError(f"native_levels is missing {missing}.")
    native = native_levels[NATIVE_VARIABLES].copy().astype(float)
    for name in NATIVE_VARIABLES:
        bad = native[name].notna() & (native[name] <= 0)
        if bad.any():
            first = bad[bad].index[0]
            raise ValueError(
                f"{name}: log state requires positive HICP levels; first invalid "
                f"value at {pd.Timestamp(first).date()}."
            )
    state = np.log(native)
    state.columns = STATE_VARIABLES
    state.index = _normalise_monthly_index(state.index)
    return state


def state_paths_to_native(state_paths: np.ndarray) -> np.ndarray:
    """Exponentiate state-level forecast paths back to native HICP indices."""
    arr = np.asarray(state_paths, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != len(NATIVE_VARIABLES):
        raise ValueError(
            "state_paths must have shape (draws, dates, 4) in the canonical "
            "Headline-joint variable order."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        native = np.exp(arr)
    if np.any(native <= 0):
        raise RuntimeError("Inverse log transformation produced non-positive HICP.")
    return native


def _validate_lag_order(p: int) -> int:
    p = int(p)
    if p not in ALLOWED_LAG_ORDERS:
        raise ValueError(
            f"Current Headline joint policy compares only dense BVAR(6) and "
            f"dense BVAR(12); received p={p}."
        )
    return p


def run_headline_joint_bvar(
    native_levels: pd.DataFrame,
    *,
    vintage: str,
    p: int = BASELINE_P,
    prior_config: BVARSVOPriorConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    code_version: str = "headline-joint-v1",
) -> dict:
    """Estimate the dense joint BVAR(6) or BVAR(12).

    ``active_lags`` is intentionally ``None``: every lag 1..p is estimated and
    Minnesota shrinkage controls dimensionality.
    """
    p = _validate_lag_order(p)
    state = native_to_state_levels(native_levels)
    prior_config = BVARSVOPriorConfig() if prior_config is None else prior_config
    sampler_config = SamplerConfig() if sampler_config is None else sampler_config

    result = run_energy_bvar(
        levels=state,
        model_id=MODEL_ID,
        vintage=str(vintage),
        p=p,
        variables=STATE_VARIABLES,
        frequency=FREQUENCY,
        exog=None,
        prior_config=prior_config,
        sampler_config=sampler_config,
        code_version=code_version,
    )
    result["headline_joint_context"] = {
        "native_variables": list(NATIVE_VARIABLES),
        "state_variables": list(STATE_VARIABLES),
        "state_variable_map": dict(STATE_VARIABLE_MAP),
        "native_levels": native_levels[NATIVE_VARIABLES].copy(),
    }
    result["metadata"].update(
        {
            "suite": "headline",
            "component": "headline_joint",
            "model_class": f"dense BVAR({p})",
            "native_variables": list(NATIVE_VARIABLES),
            "state_variables": list(STATE_VARIABLES),
            "state_variable_map": dict(STATE_VARIABLE_MAP),
            "transform_map": {name: "logdiff" for name in NATIVE_VARIABLES},
            "active_lags": None,
            "lag_policy": (
                "canonical dense BVAR(12); only sensitivity is dense BVAR(6)"
            ),
            "headline_joint_version": HEADLINE_JOINT_VERSION,
        }
    )
    return result


def run_headline_joint_lag_comparison(
    native_levels: pd.DataFrame,
    *,
    vintage: str,
    prior_config: BVARSVOPriorConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    code_version: str = "headline-joint-lag-comparison-v1",
) -> dict[int, dict]:
    """Run only the approved dense BVAR(6) versus dense BVAR(12) comparison."""
    outputs = {}
    for p in (SENSITIVITY_P, BASELINE_P):
        outputs[p] = run_headline_joint_bvar(
            native_levels,
            vintage=vintage,
            p=p,
            prior_config=prior_config,
            sampler_config=sampler_config,
            code_version=f"{code_version}-p{p}",
        )
    return outputs


def _native_conditions_to_state(
    conditions: Mapping[str, float | Sequence[float]] | None,
) -> dict[str, float | np.ndarray] | None:
    if conditions is None:
        return None
    output: dict[str, float | np.ndarray] = {}
    for native_name, values in conditions.items():
        if native_name not in STATE_VARIABLE_MAP:
            raise KeyError(
                f"Unknown native condition variable {native_name!r}; choose from "
                f"{NATIVE_VARIABLES}."
            )
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{native_name}: native conditions must be finite.")
        if np.any(arr <= 0):
            raise ValueError(f"{native_name}: HICP level conditions must be positive.")
        transformed = np.log(arr)
        output[STATE_VARIABLE_MAP[native_name]] = (
            float(transformed) if transformed.ndim == 0 else transformed
        )
    return output


def _component_yoy_paths(
    native_paths: np.ndarray,
    path_dates: pd.DatetimeIndex,
    native_history: pd.DataFrame,
) -> np.ndarray:
    """Compute component YoY paths draw-by-draw using observed pre-path history."""
    draws, _, n = native_paths.shape
    output = np.full_like(native_paths, np.nan, dtype=float)
    pos = {pd.Timestamp(date): i for i, date in enumerate(path_dates)}

    history = native_history[NATIVE_VARIABLES].copy().sort_index()
    history.index = _normalise_monthly_index(history.index)

    for t, date in enumerate(path_dates):
        lag_date = pd.Timestamp(date) - pd.DateOffset(months=12)
        for j, name in enumerate(NATIVE_VARIABLES):
            if lag_date in pos:
                denominator = native_paths[:, pos[lag_date], j]
            else:
                if lag_date not in history.index or pd.isna(history.loc[lag_date, name]):
                    continue
                denominator = np.full(
                    draws,
                    float(history.loc[lag_date, name]),
                    dtype=float,
                )
            output[:, t, j] = 100.0 * (
                native_paths[:, t, j] / denominator - 1.0
            )
    return output


def forecast_headline_joint_bvar(
    result: Mapping,
    *,
    native_levels: pd.DataFrame | None = None,
    H: int = 18,
    native_level_conditions: Mapping[
        str, float | Sequence[float]
    ] | None = None,
    n_draws: int | None = None,
    simulate_future_outliers: bool = True,
    seed: int = 123,
) -> dict[str, object]:
    """Forecast the joint system and return native component HICP paths."""
    if native_levels is None:
        context = result.get("headline_joint_context", {})
        native_levels = context.get("native_levels")
    if not isinstance(native_levels, pd.DataFrame):
        raise ValueError(
            "native_levels is required when the fitted result does not carry a "
            "live headline_joint_context."
        )

    state_conditions = _native_conditions_to_state(native_level_conditions)
    state_forecast = forecast_bvar_sv_outlier(
        result,
        H=int(H),
        level_conditions=state_conditions,
        future_exog=None,
        n_draws=n_draws,
        simulate_future_outliers=bool(simulate_future_outliers),
        seed=int(seed),
    )
    if list(state_forecast["variables"]) != STATE_VARIABLES:
        raise ValueError(
            "Joint forecast state-variable ordering changed; expected "
            f"{STATE_VARIABLES}, found {state_forecast['variables']}."
        )

    native_paths = state_paths_to_native(state_forecast["level_paths"])
    path_dates = pd.DatetimeIndex(state_forecast["path_dates"], name="date")
    component_yoy = _component_yoy_paths(
        native_paths,
        path_dates,
        native_levels,
    )
    tail = int(state_forecast["tail_length"])

    output = dict(state_forecast)
    output.update(
        {
            "component": MODEL_ID,
            "component_label": "Headline components joint BVAR",
            "native_variables": list(NATIVE_VARIABLES),
            "state_variables": list(STATE_VARIABLES),
            "native_level_paths": native_paths,
            "future_native_level_paths": native_paths[:, tail:],
            "component_yoy_paths": component_yoy,
            "future_component_yoy_paths": component_yoy[:, tail:],
            "native_level_conditions": (
                None
                if native_level_conditions is None
                else dict(native_level_conditions)
            ),
            "forecast_contract_version": "headline_joint_v1",
        }
    )
    return output


def headline_component_forecast_table(
    forecast: Mapping,
    *,
    kind: str = "yoy",
    horizons: Sequence[int] = (1, 3, 6, 12, 18),
    quantiles: Sequence[float] = (5, 16, 50, 84, 95),
) -> pd.DataFrame:
    """Summarise future component forecasts at selected horizons."""
    dates = pd.DatetimeIndex(forecast["future_dates"], name="date")
    if kind == "level":
        paths = np.asarray(forecast["future_native_level_paths"], dtype=float)
    elif kind == "yoy":
        paths = np.asarray(forecast["future_component_yoy_paths"], dtype=float)
    else:
        raise ValueError("kind must be 'level' or 'yoy'.")

    variables = list(forecast["native_variables"])
    rows = []
    for horizon in horizons:
        h = int(horizon)
        if h < 1 or h > len(dates):
            continue
        for j, variable in enumerate(variables):
            values = paths[:, h - 1, j]
            q = np.nanpercentile(values, quantiles)
            row = {
                "months_ahead": h,
                "target_date": dates[h - 1],
                "variable": variable,
            }
            row.update(
                {
                    f"q{int(prob):02d}": float(value)
                    for prob, value in zip(quantiles, q)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).set_index(["months_ahead", "variable"])


def _complete_weight_row(
    weights: pd.DataFrame,
    year: int,
    *,
    carry_forward_future: bool,
) -> tuple[pd.Series, int, bool]:
    year = int(year)
    if year in weights.index and weights.loc[year].notna().all():
        return weights.loc[year, NATIVE_VARIABLES].astype(float), year, False

    complete = weights[NATIVE_VARIABLES].dropna(how="any")
    if complete.empty:
        raise ValueError("No complete annual joint HICP weight vector exists.")
    latest_year = int(complete.index.max())
    if carry_forward_future and year > latest_year:
        return complete.loc[latest_year].astype(float), latest_year, True

    raise ValueError(
        f"No complete Energy/Food/NEIG/Services weight vector for {year}."
    )



def resolve_headline_reconstruction_start_year(
    native_levels: pd.DataFrame,
    weights: pd.DataFrame,
    official_total: pd.Series,
) -> int:
    """Return the first historically admissible Headline reconstruction year.

    A year y is admissible only if all three hard requirements hold:

    1. Energy/Food/NEIG/Services annual weights for y are all published;
    2. all four component HICP indices are observed in December y-1;
    3. official HICP Total is observed in December y-1 for the initial anchor.

    Historical weights are never backfilled or carried backward.  This is
    important because NEIG/Services annual weights begin later than some of the
    component index histories.
    """
    native = native_levels[NATIVE_VARIABLES].copy().sort_index()
    native.index = _normalise_monthly_index(native.index)
    weights = weights[NATIVE_VARIABLES].copy().sort_index()
    total = official_total.copy().sort_index()
    total.index = _normalise_monthly_index(total.index)

    complete_weight_years = [
        int(year)
        for year, row in weights.iterrows()
        if row.notna().all()
    ]
    if not complete_weight_years:
        raise ValueError(
            "No complete Energy/Food/NEIG/Services annual weight vector exists."
        )

    rejection_reasons: list[str] = []
    for year in sorted(complete_weight_years):
        dec_prev = pd.Timestamp(year=year - 1, month=12, day=1)

        if dec_prev not in native.index:
            rejection_reasons.append(
                f"{year}: component December {dec_prev.date()} is outside the panel"
            )
            continue

        component_row = native.loc[dec_prev, NATIVE_VARIABLES]
        if component_row.isna().any():
            missing = component_row.index[component_row.isna()].tolist()
            rejection_reasons.append(
                f"{year}: component December {dec_prev.date()} missing {missing}"
            )
            continue

        if dec_prev not in total.index or pd.isna(total.loc[dec_prev]):
            rejection_reasons.append(
                f"{year}: official HICP Total anchor missing at {dec_prev.date()}"
            )
            continue

        return year

    detail = "; ".join(rejection_reasons[:5])
    if len(rejection_reasons) > 5:
        detail += "; ..."
    raise ValueError(
        "No admissible historical Headline reconstruction start year. "
        f"Checked complete-weight years. {detail}"
    )



def reconstruct_headline_history(
    native_levels: pd.DataFrame,
    weights: pd.DataFrame,
    official_total: pd.Series,
    *,
    start_year: int | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Reconstruct historical Headline HICP with annual December chain linking.

    The first reconstructed year is anchored to the official HICP Total level in
    December of the preceding year. If ``start_year`` is omitted, the function
    automatically starts at the first year for which the four annual component
    weights, the preceding-December component levels and the official Total
    anchor are all available. Historical weight gaps are never filled.
    """
    native = native_levels[NATIVE_VARIABLES].copy().sort_index()
    native.index = _normalise_monthly_index(native.index)
    total = official_total.copy().sort_index()
    total.index = _normalise_monthly_index(total.index)
    weights = weights[NATIVE_VARIABLES].copy().sort_index()

    common_complete = native.dropna(how="any")
    if common_complete.empty:
        raise ValueError("No fully observed four-component historical month.")

    if start_year is None:
        start_year = resolve_headline_reconstruction_start_year(
            native,
            weights,
            total,
        )
    else:
        start_year = int(start_year)
        # Explicit historical start years are held to the same hard contract.
        if start_year not in weights.index or weights.loc[start_year].isna().any():
            raise ValueError(
                f"Requested historical reconstruction start_year={start_year}, "
                "but no complete four-component weight vector exists for that year."
            )
        dec_prev = pd.Timestamp(year=start_year - 1, month=12, day=1)
        if dec_prev not in native.index:
            raise ValueError(
                f"Requested start_year={start_year} requires component December "
                f"{dec_prev.date()}, which is outside the panel."
            )
        missing_dec = native.loc[dec_prev, NATIVE_VARIABLES].isna()
        if missing_dec.any():
            missing = missing_dec.index[missing_dec].tolist()
            raise ValueError(
                f"Requested start_year={start_year} has missing component levels "
                f"at {dec_prev.date()}: {missing}."
            )
        if dec_prev not in total.index or pd.isna(total.loc[dec_prev]):
            raise ValueError(
                f"Requested start_year={start_year} requires official HICP Total "
                f"anchor at {dec_prev.date()}."
            )

    if end_date is None:
        last_date = common_complete.index.max()
    else:
        last_date = min(pd.Timestamp(end_date), common_complete.index.max())

    reconstructed = pd.Series(dtype=float, name="hicp_total_reconstructed")
    previous_december_aggregate = None

    for year in range(int(start_year), int(last_date.year) + 1):
        weight_row, _, carried = _complete_weight_row(
            weights,
            year,
            carry_forward_future=False,
        )
        if carried:
            raise RuntimeError("Historical reconstruction cannot carry weights.")

        dec_prev = pd.Timestamp(year=year - 1, month=12, day=1)
        if dec_prev not in native.index:
            raise ValueError(
                f"Historical reconstruction needs component December {dec_prev.date()}."
            )
        dec_component = native.loc[dec_prev, NATIVE_VARIABLES]
        if dec_component.isna().any():
            raise ValueError(
                f"Historical reconstruction has missing component level at "
                f"{dec_prev.date()}."
            )

        if previous_december_aggregate is None:
            if dec_prev not in total.index or pd.isna(total.loc[dec_prev]):
                raise ValueError(
                    f"Official HICP Total anchor missing at {dec_prev.date()}."
                )
            anchor = float(total.loc[dec_prev])
        else:
            anchor = float(previous_december_aggregate)

        year_dates = native.index[
            (native.index.year == year) & (native.index <= last_date)
        ]
        for date in year_dates:
            current = native.loc[date, NATIVE_VARIABLES]
            if current.isna().any():
                continue
            ratio = current / dec_component
            reconstructed.loc[date] = float(
                anchor * np.sum((weight_row / 1000.0) * ratio)
            )

        december = pd.Timestamp(year=year, month=12, day=1)
        if december in reconstructed.index:
            previous_december_aggregate = reconstructed.loc[december]
        elif year < last_date.year:
            raise ValueError(
                f"Cannot chain into {year+1}: reconstructed December {year} is missing."
            )

    reconstructed.index = pd.DatetimeIndex(reconstructed.index, name="date")
    return reconstructed.sort_index()


def headline_reconstruction_diagnostics(
    reconstructed: pd.Series,
    official_total: pd.Series,
) -> pd.Series:
    """Level and YoY errors of bottom-up historical reconstruction."""
    joined = pd.concat(
        [
            reconstructed.rename("reconstructed"),
            official_total.rename("official"),
        ],
        axis=1,
    ).dropna()
    if joined.empty:
        raise ValueError("No overlap between reconstructed and official HICP Total.")

    level_error = joined["reconstructed"] - joined["official"]
    reconstructed_yoy = 100.0 * (
        joined["reconstructed"] / joined["reconstructed"].shift(12) - 1.0
    )
    official_yoy = 100.0 * (
        joined["official"] / joined["official"].shift(12) - 1.0
    )
    yoy_error = (reconstructed_yoy - official_yoy).dropna()

    return pd.Series(
        {
            "n_level_months": int(level_error.size),
            "level_mae": float(level_error.abs().mean()),
            "level_rmse": float(np.sqrt(np.mean(np.square(level_error)))),
            "level_max_abs_error": float(level_error.abs().max()),
            "n_yoy_months": int(yoy_error.size),
            "yoy_mae_pp": float(yoy_error.abs().mean()) if len(yoy_error) else np.nan,
            "yoy_rmse_pp": (
                float(np.sqrt(np.mean(np.square(yoy_error))))
                if len(yoy_error)
                else np.nan
            ),
            "yoy_max_abs_error_pp": (
                float(yoy_error.abs().max()) if len(yoy_error) else np.nan
            ),
        },
        name="value",
    )


def aggregate_headline_draws(
    component_forecast: Mapping,
    *,
    native_levels: pd.DataFrame,
    weights: pd.DataFrame,
    official_total: pd.Series,
    future_weight_policy: str = "carry_forward_latest",
) -> dict[str, object]:
    """Aggregate joint component forecast draws into Headline HICP draw-by-draw.

    For each calendar year y:

        H_t = H_Dec(y-1) * sum_i [w_i^y/1000 * I_i,t/I_i,Dec(y-1)].

    The first forecast/path year is anchored to official total HICP in the
    preceding December. Later December anchors are draw-specific reconstructed
    values. The latest complete annual weight vector is carried forward only
    for forecast years beyond the latest published weight year.
    """
    if future_weight_policy != "carry_forward_latest":
        raise ValueError(
            "Only future_weight_policy='carry_forward_latest' is supported."
        )

    paths = np.asarray(component_forecast["native_level_paths"], dtype=float)
    variables = list(component_forecast["native_variables"])
    if variables != NATIVE_VARIABLES:
        raise ValueError(
            f"Expected native variable order {NATIVE_VARIABLES}, found {variables}."
        )
    dates = pd.DatetimeIndex(component_forecast["path_dates"], name="date")
    if paths.ndim != 3 or paths.shape[1] != len(dates) or paths.shape[2] != 4:
        raise ValueError("Component forecast path dimensions are inconsistent.")

    native = native_levels[NATIVE_VARIABLES].copy().sort_index()
    native.index = _normalise_monthly_index(native.index)
    total = official_total.copy().sort_index()
    total.index = _normalise_monthly_index(total.index)
    weights = weights[NATIVE_VARIABLES].copy().sort_index()

    n_draws = paths.shape[0]
    headline = np.full((n_draws, len(dates)), np.nan, dtype=float)
    contributions = np.full((n_draws, len(dates), 4), np.nan, dtype=float)
    weight_year_used = np.full(len(dates), -1, dtype=int)
    carried_weights = np.zeros(len(dates), dtype=bool)
    date_to_pos = {pd.Timestamp(date): i for i, date in enumerate(dates)}

    first_year = int(dates.min().year)
    first_dec_prev = pd.Timestamp(first_year - 1, 12, 1)
    if first_dec_prev not in total.index or pd.isna(total.loc[first_dec_prev]):
        raise ValueError(
            f"Official HICP Total anchor missing at {first_dec_prev.date()}."
        )
    current_anchor = np.full(
        n_draws,
        float(total.loc[first_dec_prev]),
        dtype=float,
    )

    for year in sorted(set(int(y) for y in dates.year)):
        year_positions = np.flatnonzero(dates.year == year)
        if len(year_positions) == 0:
            continue

        weight_row, source_year, carried = _complete_weight_row(
            weights,
            year,
            carry_forward_future=True,
        )
        dec_prev = pd.Timestamp(year - 1, 12, 1)

        if dec_prev in date_to_pos:
            dec_component = paths[:, date_to_pos[dec_prev], :]
            # From the second path year onward, the draw-specific reconstructed
            # December level is the next annual anchor.
            current_anchor = headline[:, date_to_pos[dec_prev]]
        else:
            if dec_prev not in native.index:
                raise ValueError(
                    f"Component December anchor missing at {dec_prev.date()}."
                )
            observed_dec = native.loc[dec_prev, NATIVE_VARIABLES]
            if observed_dec.isna().any():
                raise ValueError(
                    f"Component December anchor has missing values at "
                    f"{dec_prev.date()}."
                )
            dec_component = np.broadcast_to(
                observed_dec.to_numpy(dtype=float),
                (n_draws, 4),
            )

        if not np.all(np.isfinite(current_anchor)):
            raise ValueError(
                f"Aggregate December anchor is unavailable entering year {year}."
            )

        w = weight_row.to_numpy(dtype=float) / 1000.0
        for pos in year_positions:
            ratios = paths[:, pos, :] / dec_component
            contrib = current_anchor[:, None] * ratios * w[None, :]
            contributions[:, pos, :] = contrib
            headline[:, pos] = np.sum(contrib, axis=1)
            weight_year_used[pos] = source_year
            carried_weights[pos] = carried

    # Headline YoY: official pre-path history, draw-specific aggregate thereafter.
    headline_yoy = np.full_like(headline, np.nan)
    for pos, date in enumerate(dates):
        lag_date = pd.Timestamp(date) - pd.DateOffset(months=12)
        if lag_date in date_to_pos:
            denominator = headline[:, date_to_pos[lag_date]]
        else:
            if lag_date not in total.index or pd.isna(total.loc[lag_date]):
                continue
            denominator = np.full(
                n_draws,
                float(total.loc[lag_date]),
                dtype=float,
            )
        headline_yoy[:, pos] = 100.0 * (headline[:, pos] / denominator - 1.0)

    tail = int(component_forecast["tail_length"])
    additivity_error = np.nanmax(
        np.abs(np.sum(contributions, axis=2) - headline)
    )
    if not np.isfinite(additivity_error) or additivity_error > 1e-10:
        raise RuntimeError(
            f"Headline draw-wise contribution additivity failed: "
            f"{additivity_error:.3e}."
        )

    return {
        **dict(component_forecast),
        "headline_level_paths": headline,
        "headline_yoy_paths": headline_yoy,
        "future_headline_level_paths": headline[:, tail:],
        "future_headline_yoy_paths": headline_yoy[:, tail:],
        "headline_component_index_contributions": contributions,
        "future_headline_component_index_contributions": contributions[:, tail:],
        "headline_weight_year_used": pd.Series(
            weight_year_used,
            index=dates,
            name="weight_year_used",
        ),
        "headline_weight_carried_forward": pd.Series(
            carried_weights,
            index=dates,
            name="weight_carried_forward",
        ),
        "headline_aggregation_method": "annual December-rebased chain link",
        "future_weight_policy": future_weight_policy,
        "headline_drawwise_additivity_max_abs_error": float(additivity_error),
        "headline_forecast_contract_version": "headline_joint_aggregate_v1",
    }


def headline_forecast_table(
    aggregate_forecast: Mapping,
    *,
    kind: str = "yoy",
    horizons: Sequence[int] = (1, 3, 6, 12, 18),
    quantiles: Sequence[float] = (5, 16, 50, 84, 95),
) -> pd.DataFrame:
    """Summarise aggregated Headline HICP forecasts."""
    dates = pd.DatetimeIndex(aggregate_forecast["future_dates"], name="date")
    if kind == "level":
        paths = np.asarray(
            aggregate_forecast["future_headline_level_paths"],
            dtype=float,
        )
    elif kind == "yoy":
        paths = np.asarray(
            aggregate_forecast["future_headline_yoy_paths"],
            dtype=float,
        )
    else:
        raise ValueError("kind must be 'level' or 'yoy'.")

    rows = []
    for horizon in horizons:
        h = int(horizon)
        if h < 1 or h > len(dates):
            continue
        q = np.nanpercentile(paths[:, h - 1], quantiles)
        row = {
            "months_ahead": h,
            "target_date": dates[h - 1],
        }
        row.update(
            {
                f"q{int(prob):02d}": float(value)
                for prob, value in zip(quantiles, q)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("months_ahead")


def lag_comparison_table(results: Mapping[int, Mapping]) -> pd.DataFrame:
    """Compact BVAR(6) versus BVAR(12) in-sample/MCMC bookkeeping table.

    This is deliberately NOT an automatic model-selection rule. Forecast
    comparison must use a common evaluation window later.
    """
    rows = []
    for p in (SENSITIVITY_P, BASELINE_P):
        if p not in results:
            continue
        result = results[p]
        radius = np.asarray(result["spectral_radius"], dtype=float)
        loglik = np.asarray(result.get("log_likelihood", []), dtype=float)
        rows.append(
            {
                "p": p,
                "model": f"dense BVAR({p})",
                "regression_observations": int(
                    result["prep"]["n_regression_observations"]
                ),
                "posterior_draws": int(result["n_draws"]),
                "spectral_radius_median": float(np.nanmedian(radius)),
                "spectral_radius_q95": float(np.nanpercentile(radius, 95)),
                "mean_log_likelihood": (
                    float(np.nanmean(loglik)) if loglik.size else np.nan
                ),
                "canonical_baseline": bool(p == BASELINE_P),
            }
        )
    return pd.DataFrame(rows).set_index("p")


__all__ = [
    "BVARSVOPriorConfig",
    "SamplerConfig",
    "HEADLINE_JOINT_VERSION",
    "MODEL_ID",
    "BASELINE_P",
    "SENSITIVITY_P",
    "ALLOWED_LAG_ORDERS",
    "NATIVE_VARIABLES",
    "STATE_VARIABLES",
    "headline_joint_spec",
    "resolve_headline_joint_dataset",
    "resolve_headline_joint_weights",
    "resolve_headline_official_indices",
    "load_headline_joint_panel",
    "load_headline_joint_weights",
    "load_official_headline_total",
    "headline_joint_panel_audit",
    "native_to_state_levels",
    "state_paths_to_native",
    "run_headline_joint_bvar",
    "run_headline_joint_lag_comparison",
    "forecast_headline_joint_bvar",
    "headline_component_forecast_table",
    "resolve_headline_reconstruction_start_year",
    "reconstruct_headline_history",
    "headline_reconstruction_diagnostics",
    "aggregate_headline_draws",
    "headline_forecast_table",
    "lag_comparison_table",
]
