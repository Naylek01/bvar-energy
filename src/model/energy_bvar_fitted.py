"""Historical posterior fitted paths for the Energy BVAR dashboard.

This module materialises the diagnostic the Aggregate page actually needs:

1. one-step conditional fitted values from each of the seven saved BVARs;
2. the same tax / frequency / HICP adapters used by the forecast aggregation;
3. petrol + diesel -> car fuels;
4. six HICP component fitted paths -> HICP Energy, draw by draw.

It intentionally does *not* use ``historical_model_component_reconstruction``
as a model-fit overlay.  That object is an accounting/chain-linking validation
based on published HICP component indices, not a BVAR fitted value.

No model is re-estimated.  Only persisted posterior coefficient draws and the
exact processed vintage are read.  The resulting compact cache is stored beside
the aggregate result so the dashboard never has to reopen heavy ``draws.npz``
files when a checkbox is toggled.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from energy_bvar_aggregate import (
    MODEL_AGGREGATE_COMPONENTS,
    aggregate_component_draw_paths_laspeyres,
    aggregate_draw_paths_laspeyres,
    historical_model_component_reconstruction,
    independent_draw_pairing,
    load_aggregation_inputs,
    rebase_price_paths_to_hicp_index,
    weekly_paths_with_history_to_monthly_mean,
)
from energy_bvar_electricity import (
    expand_semester_series as expand_electricity_semester_series,
    load_electricity_tax_context,
    reattribute_electricity_taxes,
)
from energy_bvar_gas import (
    expand_semester_series as expand_gas_semester_series,
    load_gas_tax_context,
    reattribute_gas_taxes,
)
from energy_bvar_model import prepare_bvar_panel
from energy_bvar_pipeline import build_panel, model_spec
from energy_bvar_weekly_fuels import load_weekly_tax_context

try:  # Available in the production model module; kept optional for portability.
    from energy_bvar_model import hash_model_data as _hash_model_data
except ImportError:  # pragma: no cover - only for very old model modules.
    _hash_model_data = None


FITTED_CACHE_VERSION = "energy-bvar-fitted-v8"
FITTED_CACHE_BASENAME = "bvar_fitted_v3"
FITTED_META_FILENAME = f"{FITTED_CACHE_BASENAME}_metadata.json"
# No dashboard-imposed display floor. Each component starts at the first
# date its saved BVAR fit and required HICP/tax bridge are genuinely valid.
DEFAULT_START_DATE = None
# Car fuels is different: the maintained six-component/transport Laspeyres
# reconstruction is defined from the post-2017 classification. December 2016
# is therefore the PREVIOUS-DECEMBER LASPEYRES CHAIN-LINK anchor for the
# car-fuels sub-aggregate. It is NOT the WOB/HICP proportional-rebase anchor
# used by petrol, diesel or liquid fuels below.
TRANSPORT_ANCHOR = pd.Timestamp("2016-12-01")
TRANSPORT_FITTED_START = pd.Timestamp("2017-01-01")
DEFAULT_MAX_DRAWS = 300
DEFAULT_PAIRING_SEED = 2026
QUANTILES = (0.05, 0.16, 0.50, 0.84, 0.95)

_COMPONENT_RUN_IDS = {
    "gas": "gas",
    "electricity": "electricity",
    "heat_energy": "heat_energy",
    "solid_fuels": "solid_fuels",
    "petrol": "car_fuels_petrol",
    "diesel": "car_fuels_diesel",
    "liquid_fuels": "liquid_fuels",
}

_WEEKLY_TARGETS = {
    "petrol": "wob_petrol_pre_tax",
    "diesel": "wob_diesel_pre_tax",
    "liquid_fuels": "wob_heating_oil_pre_tax",
}

_WEEKLY_HICP_COLUMNS = {
    "petrol": "hicp_petrol",
    "diesel": "hicp_diesel",
    "liquid_fuels": "hicp_liquid_fuels",
}

_DIRECT_TARGETS = {
    "heat_energy": "hicp_heat_energy",
    "solid_fuels": "hicp_solid_fuels",
}

_TRANSPORT_COMPONENTS = (
    "hicp_diesel",
    "hicp_petrol",
    "hicp_other_transport_fuels",
)


class FittedMaterialisationError(RuntimeError):
    """Raised when a saved aggregate cannot be given a valid fitted history."""


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return value


def _aggregate_metadata(aggregate_directory: Path) -> dict:
    path = aggregate_directory / "metadata.json"
    if not path.is_file():
        fallback = aggregate_directory / "aggregate_config.json"
        path = fallback if fallback.is_file() else path
    return _read_json(path)


def _results_root_from_aggregate(aggregate_directory: Path) -> Path:
    # <results>/hicp_energy_aggregate/<vintage>/<aggregate_run_id>
    try:
        return aggregate_directory.parents[2]
    except IndexError as exc:  # pragma: no cover - defensive only.
        raise FittedMaterialisationError(
            f"Cannot infer results root from {aggregate_directory}."
        ) from exc


def _source_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _component_run_directories(
    aggregate_directory: Path,
    aggregate_metadata: Mapping,
) -> dict[str, Path]:
    stores = dict(aggregate_metadata.get("component_forecast_stores", {}) or {})
    if not stores:
        raise FittedMaterialisationError(
            "Aggregate metadata does not record component_forecast_stores."
        )

    vintage = str(
        aggregate_metadata.get("vintage")
        or aggregate_directory.parent.name
    )
    results_root = _results_root_from_aggregate(aggregate_directory)
    out: dict[str, Path] = {}
    missing: list[str] = []

    for aggregate_key, canonical_model_id in _COMPONENT_RUN_IDS.items():
        raw = stores.get(aggregate_key)
        if raw is None:
            # Backward compatibility: a few experimental stores used canonical
            # model ids rather than aggregation keys for petrol/diesel.
            raw = stores.get(canonical_model_id)
        if raw is None:
            missing.append(aggregate_key)
            continue

        store_path = Path(str(raw))
        try:
            run_id = store_path.parents[1].name
        except IndexError:
            run_id = ""

        candidate = (
            store_path.parents[1] if len(store_path.parents) >= 2 else None
        )
        if (candidate is None or not candidate.is_dir()) and run_id:
            candidate = results_root / canonical_model_id / vintage / run_id

        if candidate is None or not candidate.is_dir():
            missing.append(
                f"{aggregate_key} (run directory not found from {raw!s})"
            )
            continue
        if not (candidate / "metadata.json").is_file():
            missing.append(f"{aggregate_key} (metadata.json missing)")
            continue
        if not (candidate / "draws.npz").is_file():
            missing.append(f"{aggregate_key} (draws.npz missing)")
            continue
        out[aggregate_key] = candidate

    if missing:
        raise FittedMaterialisationError(
            "Historical BVAR fitted paths require persisted posterior draws for "
            "all seven component models; missing: " + ", ".join(missing)
        )
    return out


def _draw_subset_indices(n_draws: int, max_draws: int) -> np.ndarray:
    n = int(n_draws)
    m = min(n, int(max_draws))
    if n < 1 or m < 1:
        raise FittedMaterialisationError("No posterior coefficient draws are available.")
    if m == n:
        return np.arange(n, dtype=int)
    # Spread retained draws across the full stored chain rather than taking a
    # contiguous prefix.  This is deterministic and cache-reproducible.
    return np.unique(np.linspace(0, n - 1, m, dtype=int))


def _prepare_saved_run(panel, metadata: Mapping) -> tuple[dict, pd.DataFrame, str]:
    """Recreate the exact design convention used by a persisted run.

    Older Energy runs stored ``missing_data_method='linear'``.  The newer
    ``prepare_bvar_panel`` API removed that keyword and makes interior missing
    observations latent by default.  For those old runs we reproduce the old
    effective panel *before* calling the new API instead of passing an invalid
    keyword or silently changing the historical design.
    """
    variables = list(metadata.get("variables") or panel.variables)
    missing = [name for name in variables if name not in panel.levels.columns]
    if missing:
        raise FittedMaterialisationError(
            f"{metadata.get('model_id')}: current processed panel is missing {missing}."
        )
    levels = panel.levels[variables].copy().astype(float)

    exog_names = list(metadata.get("exog_names") or [])
    exog = panel.exog
    if exog_names:
        if exog is None:
            raise FittedMaterialisationError(
                f"{metadata.get('model_id')}: saved run expects exogenous columns {exog_names}."
            )
        missing_exog = [name for name in exog_names if name not in exog.columns]
        if missing_exog:
            raise FittedMaterialisationError(
                f"{metadata.get('model_id')}: current exog is missing {missing_exog}."
            )
        exog = exog[exog_names]
    elif exog is not None and exog.shape[1]:
        # The saved coefficient matrix has no deterministic block: do not let a
        # later model-spec change alter the fitted design retrospectively.
        exog = None

    if _hash_model_data is not None:
        expected_hash = metadata.get("source_data_hash") or metadata.get("data_hash")
        if expected_hash:
            actual_hash = _hash_model_data(levels, exog)
            if str(actual_hash) != str(expected_hash):
                raise FittedMaterialisationError(
                    f"{metadata.get('model_id')}: processed-vintage data no longer "
                    "match the source data hash of the saved posterior run."
                )

    missing_method = str(metadata.get("missing_data_method") or "dk").lower()
    params = inspect.signature(prepare_bvar_panel).parameters
    kwargs = {
        "p": int(metadata.get("p") or panel.p),
        "variables": variables,
    }
    if "frequency" in params:
        kwargs["frequency"] = str(metadata.get("frequency") or panel.frequency)
    if "exog" in params:
        kwargs["exog"] = exog

    effective_levels = levels
    if "missing_data_method" in params:
        kwargs["missing_data_method"] = missing_method
    elif missing_method == "linear":
        effective_levels = levels.interpolate(method="time", limit_area="inside")
    elif missing_method not in {"dk", "durbin_koopman", "durbin-koopman", ""}:
        raise FittedMaterialisationError(
            f"Unsupported saved missing_data_method={missing_method!r}."
        )

    prep = prepare_bvar_panel(effective_levels, **kwargs)

    # When we manually recreated the old linear branch, verify the *effective*
    # panel too when that historical hash is available.
    if _hash_model_data is not None and missing_method == "linear":
        expected_effective = metadata.get("effective_estimation_data_hash")
        if expected_effective:
            actual_effective = _hash_model_data(prep["levels"][variables], prep.get("exog"))
            if str(actual_effective) != str(expected_effective):
                raise FittedMaterialisationError(
                    f"{metadata.get('model_id')}: recreated linear effective panel "
                    "does not match the saved estimation-data hash."
                )

    return prep, effective_levels, missing_method


def _one_step_fitted_target(
    panel,
    metadata: Mapping,
    draws: Mapping[str, np.ndarray],
    *,
    target: str,
    max_draws: int,
) -> dict:
    """Posterior one-step conditional mean for one saved target equation."""
    if "B" not in draws:
        raise FittedMaterialisationError(
            f"{metadata.get('model_id')}: draws.npz has no coefficient array B."
        )
    B_all = np.asarray(draws["B"], dtype=float)
    if B_all.ndim != 3:
        raise FittedMaterialisationError(
            f"{metadata.get('model_id')}: B must have shape (draw, k, n), found {B_all.shape}."
        )

    variables = list(metadata.get("variables") or panel.variables)
    if target not in variables:
        raise FittedMaterialisationError(
            f"{metadata.get('model_id')}: target {target!r} is not in {variables}."
        )
    j = variables.index(target)
    chosen = _draw_subset_indices(B_all.shape[0], max_draws)
    B = B_all[chosen]

    prep, _, missing_method = _prepare_saved_run(panel, metadata)
    if B.shape[1] != int(prep["k"]) or B.shape[2] != len(variables):
        raise FittedMaterialisationError(
            f"{metadata.get('model_id')}: saved B shape {B.shape[1:]} is incompatible "
            f"with recreated design (k={prep['k']}, n={len(variables)})."
        )

    fit_dates = pd.DatetimeIndex(prep["dates"], name="date")
    p = int(prep["p"])
    n = len(variables)

    if not bool(prep.get("requires_data_augmentation", False)):
        X = np.asarray(prep["X"], dtype=float)
        if X.shape != (len(fit_dates), B.shape[1]):
            raise FittedMaterialisationError(
                f"{metadata.get('model_id')}: recreated X has shape {X.shape}, "
                f"expected {(len(fit_dates), B.shape[1])}."
            )
        fitted_change = np.einsum("tk,dk->dt", X, B[:, :, j], optimize=True)
        previous = prep["levels"][target].shift(1).reindex(fit_dates).to_numpy(dtype=float)
        if not np.all(np.isfinite(previous)):
            raise FittedMaterialisationError(
                f"{metadata.get('model_id')}: non-finite lagged target levels on fitted dates."
            )
        fitted_level = previous[None, :] + fitted_change
        completed_mode = "observed_or_linear_effective_history"
    else:
        if "completed_differences_draws" not in draws:
            raise FittedMaterialisationError(
                f"{metadata.get('model_id')}: DK fitted values require "
                "completed_differences_draws in draws.npz."
            )
        completed_all = np.asarray(draws["completed_differences_draws"], dtype=float)
        if completed_all.ndim != 4 and completed_all.ndim != 3:
            raise FittedMaterialisationError(
                f"{metadata.get('model_id')}: unexpected completed_differences_draws "
                f"shape {completed_all.shape}."
            )
        # Production schema is (draw, p+T, n).  A singleton chain dimension was
        # briefly used in one experimental branch; squeeze it defensively.
        if completed_all.ndim == 4 and completed_all.shape[1] == 1:
            completed_all = completed_all[:, 0]
        if completed_all.ndim != 3:
            raise FittedMaterialisationError(
                f"{metadata.get('model_id')}: cannot interpret completed differences "
                f"shape {completed_all.shape}."
            )
        completed = completed_all[chosen]
        expected_rows = p + len(fit_dates)
        if completed.shape[1:] != (expected_rows, n):
            raise FittedMaterialisationError(
                f"{metadata.get('model_id')}: completed differences shape "
                f"{completed.shape[1:]} != {(expected_rows, n)}."
            )

        # Compute only the target equation, lag block by lag block.  This avoids
        # constructing a potentially >300 MB draw x time x regressor tensor for
        # weekly VAR(24) models.
        fitted_change = np.repeat(B[:, 0, j][:, None], len(fit_dates), axis=1)
        for lag in range(1, p + 1):
            row0 = 1 + (lag - 1) * n
            coeff = B[:, row0 : row0 + n, j]
            lag_values = completed[:, p - lag : p - lag + len(fit_dates), :]
            fitted_change += np.einsum(
                "dti,di->dt", lag_values, coeff, optimize=True
            )

        exog = prep.get("exog_regression")
        if exog is not None and exog.shape[1]:
            exog_values = exog.reindex(fit_dates).to_numpy(dtype=float)
            row0 = 1 + p * n
            coeff = B[:, row0 : row0 + exog_values.shape[1], j]
            fitted_change += np.einsum(
                "tm,dm->dt", exog_values, coeff, optimize=True
            )

        anchor = np.asarray(prep["augmentation_anchor_level"], dtype=float)
        estimation_diffs = completed[:, p:, :]
        previous_target = np.empty((len(B), len(fit_dates)), dtype=float)
        previous_target[:, 0] = float(anchor[j])
        if len(fit_dates) > 1:
            previous_target[:, 1:] = (
                float(anchor[j])
                + np.cumsum(estimation_diffs[:, :-1, j], axis=1)
            )
        fitted_level = previous_target + fitted_change
        completed_mode = "draw_specific_DK_completed_history"

    if np.any(~np.isfinite(fitted_level)):
        raise FittedMaterialisationError(
            f"{metadata.get('model_id')}: non-finite one-step fitted target levels."
        )

    return {
        "dates": fit_dates,
        "level_paths": fitted_level,
        "draw_indices": chosen,
        "target": target,
        "variables": variables,
        "missing_data_method": missing_method,
        "conditioning_history": completed_mode,
    }


def _slice_object(obj: Mapping, dates: pd.DatetimeIndex) -> np.ndarray:
    source_dates = pd.DatetimeIndex(obj["dates"], name="date")
    positions = source_dates.get_indexer(dates)
    if (positions < 0).any():
        missing = dates[positions < 0]
        raise FittedMaterialisationError(
            f"Requested fitted dates are absent from a component path: {list(missing[:5])}."
        )
    return np.asarray(obj["paths"], dtype=float)[:, positions]


def _normalise_optional_start_date(
    value: str | pd.Timestamp | None,
) -> pd.Timestamp | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    stamp = pd.Timestamp(value)
    return stamp.to_period("M").to_timestamp(how="start")


def _common_monthly_dates(
    objects: Sequence[Mapping],
    start_date: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Natural common monthly overlap, with an optional explicit lower bound."""
    if not objects:
        raise FittedMaterialisationError("No component fitted paths were supplied.")
    starts = [pd.DatetimeIndex(obj["dates"]).min() for obj in objects]
    ends = [pd.DatetimeIndex(obj["dates"]).max() for obj in objects]
    candidates = list(map(pd.Timestamp, starts))
    if start_date is not None:
        candidates.append(pd.Timestamp(start_date))
    start = max(candidates)
    end = min(map(pd.Timestamp, ends))
    if end < start:
        suffix = (
            ""
            if start_date is None
            else f" after {pd.Timestamp(start_date).date()}"
        )
        raise FittedMaterialisationError(
            f"No common fitted monthly window{suffix}."
        )
    dates = pd.date_range(
        start.to_period("M").to_timestamp(how="start"),
        end.to_period("M").to_timestamp(how="start"),
        freq="MS",
        name="date",
    )
    for obj in objects:
        source = pd.DatetimeIndex(obj["dates"], name="date")
        if (source.get_indexer(dates) < 0).any():
            raise FittedMaterialisationError(
                "A component fitted path has an interior monthly calendar gap."
            )
    return dates


def _drop_bad_draws(paths: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(paths, dtype=float)
    keep = np.all(np.isfinite(values) & (values > 0.0), axis=1)
    if not keep.any():
        raise FittedMaterialisationError(
            f"{label}: no fitted draw remains finite and positive over the display window."
        )
    return values[keep]


def _monthly_pretax_to_hicp(
    fitted: Mapping,
    *,
    tax_context: Mapping,
    expand_function,
    reattribute_function,
    start_date: pd.Timestamp | None = None,
) -> dict:
    dates = pd.DatetimeIndex(fitted["dates"], name="date")
    pre_tax = np.asarray(fitted["level_paths"], dtype=float)
    if start_date is not None:
        mask = dates >= pd.Timestamp(start_date)
        dates = dates[mask]
        pre_tax = pre_tax[:, mask]
    if len(dates) == 0:
        raise FittedMaterialisationError(
            "No monthly fitted dates survive the requested lower bound."
        )

    vat = expand_function(tax_context["vat_percent"], dates).to_numpy(dtype=float)
    excise = expand_function(tax_context["excise"], dates).to_numpy(dtype=float)
    finite_dates = np.isfinite(vat) & np.isfinite(excise)
    if not finite_dates.any():
        raise FittedMaterialisationError("No tax observations overlap the fitted history.")
    # Tax series can start after the statistical model.  Retain one contiguous
    # suffix rather than silently punching holes in the monthly path.
    first = int(np.flatnonzero(finite_dates)[0])
    if not finite_dates[first:].all():
        raise FittedMaterialisationError("Tax context has an interior gap in the fitted window.")
    dates = dates[first:]
    pre_tax = pre_tax[:, first:]
    vat = vat[first:]
    excise = excise[first:]
    hicp = reattribute_function(
        pre_tax,
        float(tax_context["gamma"]),
        vat[None, :],
        excise[None, :],
    )
    hicp = _drop_bad_draws(hicp, label="monthly pre-tax -> HICP")
    return {"dates": dates, "paths": hicp}


def _carry_weekly_series(series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    observed = series.astype(float).dropna().sort_index()
    observed.index = pd.DatetimeIndex(observed.index).to_period("W-SUN").start_time
    union = observed.index.union(dates).sort_values()
    carried = observed.reindex(union).ffill().reindex(dates)
    if carried.isna().any():
        first_bad = carried.index[carried.isna()][0]
        raise FittedMaterialisationError(
            f"No weekly tax observation exists on or before {first_bad.date()}."
        )
    return carried.to_numpy(dtype=float)


def _monthly_mean_history(series: pd.Series) -> pd.Series:
    s = series.astype(float).dropna().sort_index()
    idx = pd.DatetimeIndex(s.index).to_period("W-SUN").start_time
    s.index = idx
    if s.index.has_duplicates:
        s = s.groupby(level=0).last()
    out = s.groupby(s.index.to_period("M")).mean()
    out.index = out.index.to_timestamp(how="start")
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out.sort_index()


def _normalise_monthly_positive_history(series: pd.Series) -> pd.Series:
    """Month-start, finite, strictly-positive observed history."""
    out = series.astype(float).copy().sort_index()
    out.index = (
        pd.DatetimeIndex(out.index)
        .to_period("M")
        .to_timestamp(how="start")
    )
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    out = out.where(np.isfinite(out) & (out > 0.0)).dropna()
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out.sort_index()


def _earliest_rebasable_fitted_month(
    fitted_dates: Sequence[pd.Timestamp],
    historical_monthly_price: pd.Series,
    hicp_history: pd.Series,
    *,
    label: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return earliest fitted month with an honest prior WOB/HICP anchor.

    ``rebase_price_paths_to_hicp_index`` requires a month ``a`` strictly before
    the simulated path such that BOTH observed monthly WOB and HICP are
    available. Removing the old 2017 dashboard floor exposed fitted BVAR dates
    that precede the first such overlap. Those dates cannot be converted to a
    HICP proxy and must not be labelled as fitted HICP observations.

    The returned pair is ``(fitted_start_month, anchor_month)``. The anchor is
    data-driven and component-specific.
    """
    dates = pd.DatetimeIndex(fitted_dates)
    if dates.empty:
        raise FittedMaterialisationError(
            f"{label}: fitted weekly calendar is empty."
        )

    fitted_months = pd.DatetimeIndex(
        sorted(
            pd.DatetimeIndex(dates)
            .to_period("M")
            .to_timestamp(how="start")
            .unique()
        ),
        name="date",
    )

    price = _normalise_monthly_positive_history(historical_monthly_price)
    hicp = _normalise_monthly_positive_history(hicp_history)
    common = price.index.intersection(hicp.index).sort_values()
    if common.empty:
        raise FittedMaterialisationError(
            f"{label}: observed WOB and HICP histories have no common positive month."
        )

    for month in fitted_months:
        prior = common[common < pd.Timestamp(month)]
        if len(prior):
            return pd.Timestamp(month), pd.Timestamp(prior.max())

    raise FittedMaterialisationError(
        f"{label}: no fitted month has a common observed WOB/HICP anchor "
        "strictly before it."
    )


def _weekly_pretax_to_hicp(
    fitted: Mapping,
    *,
    context: Mapping,
    hicp_history: pd.Series,
    start_date: pd.Timestamp | None = None,
    label: str,
) -> dict:
    """Convert weekly fitted pre-tax WOB prices to an honest HICP proxy path.

    Full-history means the earliest date that can actually be represented under
    the production bridge. It does NOT mean forcing the BVAR fitted path before
    a WOB/HICP proportional-rebase anchor exists.
    """
    dates = pd.DatetimeIndex(fitted["dates"], name="date")
    pre_tax = np.asarray(fitted["level_paths"], dtype=float)

    if start_date is not None:
        lower = pd.Timestamp(start_date).to_period("M").to_timestamp(how="start")
        mask = dates >= lower
        dates = dates[mask]
        pre_tax = pre_tax[:, mask]

    data = context["data"]

    # Taxes must already exist on/before every fitted weekly price date.
    tax_starts = []
    for tax_name in ("excise", "vat_percent"):
        tax_series = data[tax_name].astype(float).dropna().sort_index()
        if tax_series.empty:
            raise FittedMaterialisationError(
                f"{label}: {tax_name} contains no finite historical observations."
            )
        tax_index = (
            pd.DatetimeIndex(tax_series.index)
            .to_period("W-SUN")
            .start_time
        )
        tax_starts.append(pd.Timestamp(tax_index.min()))

    natural_tax_start = max(tax_starts)
    mask = dates >= natural_tax_start
    dates = dates[mask]
    pre_tax = pre_tax[:, mask]
    if len(dates) == 0:
        raise FittedMaterialisationError(
            f"{label}: no fitted weekly dates overlap the historical tax bridge."
        )

    historical_monthly_price = _monthly_mean_history(data["after_tax"])

    # Critical full-history guard: find the EARLIEST fitted month for which
    # rebase_price_paths_to_hicp_index has a genuine observed WOB/HICP month
    # strictly before the simulated path.
    fitted_start_month, expected_anchor = _earliest_rebasable_fitted_month(
        dates,
        historical_monthly_price,
        hicp_history,
        label=label,
    )

    fitted_month_index = (
        dates.to_period("M").to_timestamp(how="start")
    )
    mask = fitted_month_index >= fitted_start_month
    dates = dates[mask]
    pre_tax = pre_tax[:, mask]

    excise = _carry_weekly_series(data["excise"], dates)
    vat = _carry_weekly_series(data["vat_percent"], dates)
    after_tax = (pre_tax + excise[None, :]) * (1.0 + vat[None, :] / 100.0)

    # Apply admissibility only on the economically representable fitted window;
    # an earlier unanchorable BVAR month must not reject an otherwise valid draw.
    keep = np.all(np.isfinite(after_tax) & (after_tax > 0.0), axis=1)
    if not keep.any():
        raise FittedMaterialisationError(
            f"{label}: all fitted after-tax price draws are invalid."
        )
    after_tax = after_tax[keep]

    monthly, monthly_dates = weekly_paths_with_history_to_monthly_mean(
        after_tax,
        dates,
        data["after_tax"],
    )
    if len(monthly_dates) == 0:
        raise FittedMaterialisationError(
            f"{label}: weekly-to-monthly conversion produced no complete month."
        )
    monthly = _drop_bad_draws(
        monthly,
        label=f"{label} monthly after-tax",
    )

    rebased = rebase_price_paths_to_hicp_index(
        monthly,
        monthly_dates,
        historical_monthly_price,
        hicp_history,
    )
    anchor_date = pd.Timestamp(rebased["anchor_date"])
    if anchor_date != pd.Timestamp(expected_anchor):
        raise FittedMaterialisationError(
            f"{label}: dynamic anchor mismatch; expected "
            f"{pd.Timestamp(expected_anchor).date()}, got {anchor_date.date()}."
        )

    # The production rebase helper can prepend OBSERVED WOB months between the
    # anchor and simulated path. That is correct for forecasting continuity,
    # but these are historical BVAR-FITTED overlays. Keep only months for which
    # the fitted WOB path actually supplied the monthly price.
    rebased_dates = pd.DatetimeIndex(rebased["dates"], name="date")
    monthly_dates = pd.DatetimeIndex(monthly_dates, name="date")
    positions = rebased_dates.get_indexer(monthly_dates)
    if (positions < 0).any():
        raise FittedMaterialisationError(
            f"{label}: rebased HICP proxy does not contain every fitted month."
        )
    fitted_hicp = np.asarray(rebased["index_paths"], dtype=float)[:, positions]

    return {
        "dates": monthly_dates,
        "paths": _drop_bad_draws(
            fitted_hicp,
            label=f"{label} HICP proxy",
        ),
        "anchor_date": anchor_date,
        "fitted_start_month": pd.Timestamp(monthly_dates.min()),
        "bridge_method": rebased["method"],
    }


def _historical_actual_paths(
    series: pd.Series,
    dates: pd.DatetimeIndex,
    n_draws: int,
    *,
    label: str = "observed residual HICP component",
) -> dict:
    """Observed residual levels on the maximal honest historical overlap.

    ``hicp_other_transport_fuels`` has no dedicated BVAR.  It is therefore an
    observed residual input when Petrol and Diesel fitted paths are combined
    into historical fitted Car fuels.  A trailing publication ragged edge is
    not an interior-data failure: the fitted diagnostic is trimmed to the last
    published positive residual month.  We never forward-fill an unpublished
    *historical* residual.  Missing/non-positive observations inside the
    overlap remain a hard error.
    """
    requested = pd.DatetimeIndex(dates, name="date")
    if requested.empty:
        raise FittedMaterialisationError(f"{label}: requested fitted window is empty.")

    observed = series.astype(float).copy().sort_index()
    observed.index = (
        pd.DatetimeIndex(observed.index)
        .to_period("M")
        .to_timestamp(how="start")
    )
    observed.index = pd.DatetimeIndex(observed.index, name="date")
    if observed.index.has_duplicates:
        observed = observed.groupby(level=0).last()

    valid_observed = observed.where(
        np.isfinite(observed) & (observed > 0.0)
    ).dropna()
    if valid_observed.empty:
        raise FittedMaterialisationError(
            f"{label}: no finite positive published HICP observations are available."
        )

    start = max(pd.Timestamp(requested.min()), pd.Timestamp(valid_observed.index.min()))
    end = min(pd.Timestamp(requested.max()), pd.Timestamp(valid_observed.index.max()))
    if end < start:
        raise FittedMaterialisationError(
            f"{label}: no overlap between fitted dates "
            f"{requested.min().date()}–{requested.max().date()} and published "
            f"HICP history {valid_observed.index.min().date()}–"
            f"{valid_observed.index.max().date()}."
        )

    overlap = pd.date_range(
        start.to_period("M").to_timestamp(how="start"),
        end.to_period("M").to_timestamp(how="start"),
        freq="MS",
        name="date",
    )
    if (requested.get_indexer(overlap) < 0).any():
        raise FittedMaterialisationError(
            f"{label}: fitted source has an interior monthly calendar gap."
        )

    aligned = observed.reindex(overlap)
    bad = aligned.isna() | ~np.isfinite(aligned) | (aligned <= 0.0)
    if bad.any():
        first_bad = pd.Timestamp(aligned.index[bad][0])
        raise FittedMaterialisationError(
            f"{label}: published HICP has an interior missing/non-positive "
            f"observation at {first_bad.date()} inside the fitted overlap."
        )

    paths = np.repeat(
        aligned.to_numpy(dtype=float)[None, :],
        int(n_draws),
        axis=0,
    )
    return {
        "dates": overlap,
        "paths": paths,
        "published_start": pd.Timestamp(valid_observed.index.min()),
        "published_end": pd.Timestamp(valid_observed.index.max()),
        "trimmed_leading_months": int((requested < overlap.min()).sum()),
        "trimmed_trailing_months": int((requested > overlap.max()).sum()),
    }


def _yoy_paths(
    level_paths: np.ndarray,
    dates: pd.DatetimeIndex,
    actual_history: pd.Series,
) -> np.ndarray:
    values = np.asarray(level_paths, dtype=float)
    dates = pd.DatetimeIndex(dates, name="date")
    out = np.full(values.shape, np.nan, dtype=float)
    lookup = {pd.Timestamp(date): j for j, date in enumerate(dates)}
    history = actual_history.astype(float).sort_index()
    history.index = pd.DatetimeIndex(history.index).to_period("M").to_timestamp(how="start")
    if history.index.has_duplicates:
        history = history.groupby(level=0).last()

    for t, date in enumerate(dates):
        lag_date = pd.Timestamp(date) - pd.DateOffset(months=12)
        lag_date = lag_date.to_period("M").to_timestamp(how="start")
        if lag_date in lookup:
            denominator = values[:, lookup[lag_date]]
        elif lag_date in history.index and np.isfinite(history.loc[lag_date]):
            denominator = np.repeat(float(history.loc[lag_date]), len(values))
        else:
            continue
        valid = np.isfinite(denominator) & (denominator > 0.0)
        out[valid, t] = 100.0 * (values[valid, t] / denominator[valid] - 1.0)
    return out


def _summary_frame(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    series: str,
    metric: str,
    basis: str,
) -> pd.DataFrame:
    values = np.asarray(paths, dtype=float)
    rows = []
    for t, date in enumerate(pd.DatetimeIndex(dates, name="date")):
        x = values[:, t]
        x = x[np.isfinite(x)]
        if len(x):
            q = np.quantile(x, QUANTILES)
            row = {
                "posterior_mean": float(np.mean(x)),
                "q05": float(q[0]),
                "q16": float(q[1]),
                "q50": float(q[2]),
                "q84": float(q[3]),
                "q95": float(q[4]),
                "n_draws": int(len(x)),
            }
        else:
            row = {
                "posterior_mean": np.nan,
                "q05": np.nan, "q16": np.nan, "q50": np.nan,
                "q84": np.nan, "q95": np.nan, "n_draws": 0,
            }
        rows.append(
            {
                "date": pd.Timestamp(date),
                "basis": basis,
                "series": series,
                "metric": metric,
                **row,
            }
        )
    return pd.DataFrame(rows)


def validate_fitted_frame(frame: pd.DataFrame) -> dict:
    """Validate the compact fitted-overlay contract before it reaches Dash.

    ``basis`` is the canonical discriminator.  ``record_type`` is deliberately
    absent from this cache contract so a display artefact cannot become a
    second source of truth for fitted availability.
    """
    required = {
        "date", "basis", "series", "metric",
        "posterior_mean", "q05", "q16", "q50", "q84", "q95", "n_draws",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FittedMaterialisationError(
            f"Fitted cache is missing required column(s): {missing}."
        )
    if frame.empty:
        raise FittedMaterialisationError("Fitted cache is empty.")

    allowed_basis = {"source_bvar_fitted", "component_bvar_fitted", "aggregate_bvar_fitted"}
    basis = set(frame["basis"].dropna().astype(str))
    if basis != allowed_basis:
        raise FittedMaterialisationError(
            f"Unexpected fitted basis contract {sorted(basis)}; "
            f"expected {sorted(allowed_basis)}."
        )

    source_rows = frame.loc[frame["basis"].astype(str) == "source_bvar_fitted"]
    source_series = set(source_rows["series"].dropna().astype(str))
    expected_source_series = {
        "gas", "electricity", "heat_energy", "solid_fuels",
        "petrol", "diesel", "liquid_fuels",
    }
    if source_series != expected_source_series:
        raise FittedMaterialisationError(
            "Source-BVAR fitted cache does not contain exactly the seven Energy "
            f"models; got {sorted(source_series)}."
        )

    component_rows = frame.loc[frame["basis"].astype(str) == "component_bvar_fitted"]
    component_series = set(component_rows["series"].dropna().astype(str))
    expected_components = set(MODEL_AGGREGATE_COMPONENTS)
    if component_series != expected_components:
        raise FittedMaterialisationError(
            "Component fitted cache does not contain exactly the six HICP Energy "
            f"components; got {sorted(component_series)}."
        )

    aggregate_rows = frame.loc[frame["basis"].astype(str) == "aggregate_bvar_fitted"]
    aggregate_series = set(aggregate_rows["series"].dropna().astype(str))
    if aggregate_series != {"hicp_energy"}:
        raise FittedMaterialisationError(
            "Aggregate fitted cache must contain only series='hicp_energy'."
        )

    metrics = set(frame["metric"].dropna().astype(str))
    if metrics != {"level", "yoy"}:
        raise FittedMaterialisationError(
            f"Unexpected fitted metrics {sorted(metrics)}; expected ['level', 'yoy']."
        )

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise FittedMaterialisationError("Fitted cache contains invalid dates.")

    qcols = ["q05", "q16", "q50", "q84", "q95"]
    numeric = frame[qcols].apply(pd.to_numeric, errors="coerce")
    posterior_mean = pd.to_numeric(frame["posterior_mean"], errors="coerce")
    finite_rows = numeric.notna().all(axis=1)
    if not finite_rows.any():
        raise FittedMaterialisationError("Fitted cache contains no finite posterior summaries.")
    q = numeric.loc[finite_rows].to_numpy(dtype=float)
    if np.any(np.diff(q, axis=1) < -1e-12):
        raise FittedMaterialisationError("Fitted posterior quantiles are not ordered.")

    level_mask = frame["metric"].astype(str) == "level"
    level_mean = posterior_mean.loc[level_mask].dropna()
    if level_mean.empty or (level_mean <= 0.0).any():
        raise FittedMaterialisationError(
            "Fitted HICP level posterior means must be finite and strictly positive."
        )

    # ``n_draws == 0`` is legitimate only for leading YoY dates for which a
    # 12-month denominator does not yet exist.  The previous validator rejected
    # those structurally unavailable YoY rows and therefore failed even when
    # the fitted level paths had been materialised correctly.
    draw_counts = pd.to_numeric(frame["n_draws"], errors="coerce")
    if draw_counts.isna().any() or (draw_counts < 0).any():
        raise FittedMaterialisationError(
            "Fitted cache contains an invalid/negative draw count."
        )

    any_quantile = numeric.notna().any(axis=1)
    all_quantiles = numeric.notna().all(axis=1)
    partial_quantiles = any_quantile & ~all_quantiles
    if partial_quantiles.any():
        raise FittedMaterialisationError(
            "Fitted cache contains a row with only a partial posterior summary."
        )

    positive_draws = draw_counts > 0
    if (positive_draws & ~posterior_mean.notna()).any():
        raise FittedMaterialisationError(
            "Fitted cache has positive n_draws but missing posterior_mean."
        )
    if ((draw_counts == 0) & posterior_mean.notna()).any():
        raise FittedMaterialisationError(
            "Fitted cache has n_draws=0 but finite posterior_mean."
        )
    if not positive_draws.any():
        raise FittedMaterialisationError(
            "Fitted cache contains no row with a positive posterior draw count."
        )
    if (positive_draws & ~all_quantiles).any():
        raise FittedMaterialisationError(
            "Fitted cache has positive n_draws but missing posterior quantiles."
        )
    if ((draw_counts == 0) & any_quantile).any():
        raise FittedMaterialisationError(
            "Fitted cache has n_draws=0 but finite posterior quantiles."
        )

    level_mask = frame["metric"].astype(str) == "level"
    if (draw_counts.loc[level_mask] <= 0).any():
        raise FittedMaterialisationError(
            "Fitted HICP level rows must always have a positive draw count."
        )

    # For YoY, zero-draw rows are permitted only as a contiguous leading block
    # within each fitted series.  Once a 12-month denominator exists, a later
    # zero-draw month would represent a genuine interior contract failure.
    yoy_mask = frame["metric"].astype(str) == "yoy"
    for keys, block in frame.loc[yoy_mask].groupby(
        ["basis", "series"], dropna=False
    ):
        block = block.sort_values("date")
        counts = pd.to_numeric(block["n_draws"], errors="coerce").to_numpy(dtype=float)
        positive_positions = np.flatnonzero(counts > 0)
        if len(positive_positions) == 0:
            raise FittedMaterialisationError(
                f"Fitted YoY block {keys} contains no computable posterior month."
            )
        first_positive = int(positive_positions[0])
        if np.any(counts[first_positive:] <= 0):
            raise FittedMaterialisationError(
                f"Fitted YoY block {keys} contains an interior unavailable month "
                "after YoY becomes computable."
            )

    # Every basis/series/metric block must use a complete, strictly increasing
    # monthly calendar. Component blocks may have different natural starts:
    # that is the full-history display contract. The aggregate block is the
    # natural common overlap required by Laspeyres aggregation.
    calendars: dict[tuple[str, str, str], pd.DatetimeIndex] = {}
    for keys, block in frame.groupby(["basis", "series", "metric"], dropna=False):
        block_dates = pd.DatetimeIndex(pd.to_datetime(block["date"])).sort_values()
        if block_dates.has_duplicates:
            raise FittedMaterialisationError(f"Duplicate fitted dates in block {keys}.")
        if len(block_dates) > 1:
            expected = pd.date_range(block_dates.min(), block_dates.max(), freq="MS")
            if not block_dates.equals(expected):
                raise FittedMaterialisationError(
                    f"Interior monthly date gap in fitted block {keys}."
                )
        calendars[tuple(map(str, keys))] = block_dates

    source_windows: dict[str, dict] = {}
    for series in (
        "gas", "electricity", "heat_energy", "solid_fuels",
        "petrol", "diesel", "liquid_fuels",
    ):
        level_key = ("source_bvar_fitted", str(series), "level")
        yoy_key = ("source_bvar_fitted", str(series), "yoy")
        level_dates = calendars.get(level_key)
        yoy_dates = calendars.get(yoy_key)
        if level_dates is None or yoy_dates is None:
            raise FittedMaterialisationError(
                f"Missing fitted level/YoY calendar for source BVAR {series!r}."
            )
        if not level_dates.equals(yoy_dates):
            raise FittedMaterialisationError(
                f"Fitted level and YoY calendars differ for source BVAR {series!r}."
            )
        source_windows[str(series)] = {
            "fit_start": pd.Timestamp(level_dates.min()),
            "fit_end": pd.Timestamp(level_dates.max()),
            "n_months": int(len(level_dates)),
        }

    component_windows: dict[str, dict] = {}
    for series in MODEL_AGGREGATE_COMPONENTS:
        level_key = ("component_bvar_fitted", str(series), "level")
        yoy_key = ("component_bvar_fitted", str(series), "yoy")
        level_dates = calendars.get(level_key)
        yoy_dates = calendars.get(yoy_key)
        if level_dates is None or yoy_dates is None:
            raise FittedMaterialisationError(
                f"Missing fitted level/YoY calendar for component {series!r}."
            )
        if not level_dates.equals(yoy_dates):
            raise FittedMaterialisationError(
                f"Fitted level and YoY calendars differ for {series!r}."
            )
        component_windows[str(series)] = {
            "fit_start": pd.Timestamp(level_dates.min()),
            "fit_end": pd.Timestamp(level_dates.max()),
            "n_months": int(len(level_dates)),
        }

    aggregate_level_key = ("aggregate_bvar_fitted", "hicp_energy", "level")
    aggregate_yoy_key = ("aggregate_bvar_fitted", "hicp_energy", "yoy")
    aggregate_dates = calendars.get(aggregate_level_key)
    aggregate_yoy_dates = calendars.get(aggregate_yoy_key)
    if aggregate_dates is None or aggregate_yoy_dates is None:
        raise FittedMaterialisationError(
            "Missing fitted level/YoY calendar for HICP Energy aggregate."
        )
    if not aggregate_dates.equals(aggregate_yoy_dates):
        raise FittedMaterialisationError(
            "Aggregate fitted level and YoY calendars differ."
        )

    # The aggregate must be a subset of every component calendar, but the
    # components are intentionally allowed to extend further back.
    for series, window in component_windows.items():
        component_dates = calendars[
            ("component_bvar_fitted", series, "level")
        ]
        if (component_dates.get_indexer(aggregate_dates) < 0).any():
            raise FittedMaterialisationError(
                f"Aggregate fitted calendar is not contained in component {series!r}."
            )

    return {
        "basis": sorted(basis),
        "source_bvars": [
            "gas", "electricity", "heat_energy", "solid_fuels",
            "petrol", "diesel", "liquid_fuels",
        ],
        "source_fit_windows": source_windows,
        "components": list(MODEL_AGGREGATE_COMPONENTS),
        "metrics": sorted(metrics),
        # Backward-compatible aggregate aliases.
        "fit_start": pd.Timestamp(aggregate_dates.min()),
        "fit_end": pd.Timestamp(aggregate_dates.max()),
        "n_months": int(len(aggregate_dates)),
        "aggregate_fit_start": pd.Timestamp(aggregate_dates.min()),
        "aggregate_fit_end": pd.Timestamp(aggregate_dates.max()),
        "aggregate_n_months": int(len(aggregate_dates)),
        "component_fit_windows": component_windows,
        "minimum_positive_summary_draws": int(draw_counts.loc[draw_counts > 0].min()),
        "leading_unavailable_yoy_rows": int(
            ((frame["metric"].astype(str) == "yoy") & (draw_counts == 0)).sum()
        ),
    }


def _write_cache(frame: pd.DataFrame, aggregate_directory: Path) -> Path:
    parquet = aggregate_directory / f"{FITTED_CACHE_BASENAME}.parquet"
    csv = aggregate_directory / f"{FITTED_CACHE_BASENAME}.csv"
    try:
        frame.to_parquet(parquet, index=False)
        if csv.exists():
            csv.unlink()
        return parquet
    except (ImportError, ModuleNotFoundError):
        frame.to_csv(csv, index=False)
        return csv


def _read_cache(aggregate_directory: Path) -> tuple[pd.DataFrame, Path]:
    parquet = aggregate_directory / f"{FITTED_CACHE_BASENAME}.parquet"
    csv = aggregate_directory / f"{FITTED_CACHE_BASENAME}.csv"
    if parquet.is_file():
        frame = pd.read_parquet(parquet)
        path = parquet
    elif csv.is_file():
        frame = pd.read_csv(csv, parse_dates=["date"])
        path = csv
    else:
        raise FileNotFoundError(parquet)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame, path


def _cache_signature(
    aggregate_directory: Path,
    aggregate_metadata: Mapping,
    run_directories: Mapping[str, Path],
    *,
    max_draws: int,
    pairing_seed: int,
    start_date: pd.Timestamp | None,
) -> dict:
    components = {}
    for key, directory in sorted(run_directories.items()):
        components[key] = {
            "metadata": _source_signature(directory / "metadata.json"),
            "draws": _source_signature(directory / "draws.npz"),
        }
    return {
        "cache_version": FITTED_CACHE_VERSION,
        "aggregate_run_id": str(
            aggregate_metadata.get("aggregate_run_id") or aggregate_directory.name
        ),
        "vintage": str(
            aggregate_metadata.get("vintage") or aggregate_directory.parent.name
        ),
        "max_draws": int(max_draws),
        "pairing_seed": int(pairing_seed),
        "start_date": (
            None
            if start_date is None
            else pd.Timestamp(start_date).date().isoformat()
        ),
        "history_policy": "full_natural_valid_history",
        "components": components,
    }


def _cache_is_current(meta: Mapping, signature: Mapping) -> bool:
    return dict(meta.get("source_signature", {}) or {}) == dict(signature)


def build_energy_aggregate_fitted(
    aggregate_directory: str | Path,
    *,
    project_root: str | Path,
    max_draws: int = DEFAULT_MAX_DRAWS,
    pairing_seed: int = DEFAULT_PAIRING_SEED,
    start_date: str | pd.Timestamp | None = DEFAULT_START_DATE,
    overwrite: bool = False,
    persist_cache: bool = True,
) -> tuple[pd.DataFrame, dict, bool]:
    """Build or load the compact historical BVAR fitted cache.

    Returns ``(frame, metadata, materialised_now)``.
    """
    aggregate_directory = Path(aggregate_directory)
    project_root = Path(project_root)
    start_date = _normalise_optional_start_date(start_date)
    aggregate_metadata = _aggregate_metadata(aggregate_directory)
    run_directories = _component_run_directories(aggregate_directory, aggregate_metadata)
    signature = _cache_signature(
        aggregate_directory,
        aggregate_metadata,
        run_directories,
        max_draws=max_draws,
        pairing_seed=pairing_seed,
        start_date=start_date,
    )
    meta_path = aggregate_directory / FITTED_META_FILENAME

    if persist_cache and not overwrite and meta_path.is_file():
        try:
            old_meta = _read_json(meta_path)
            if _cache_is_current(old_meta, signature):
                frame, cache_path = _read_cache(aggregate_directory)
                old_meta["cache_path"] = str(cache_path)
                old_meta["available"] = True
                return frame, old_meta, False
        except Exception:
            # A corrupt/stale fitted cache is safe to rebuild; the underlying
            # aggregate forecast remains untouched.
            pass

    processed_dir = project_root / "data" / "processed" / str(
        aggregate_metadata.get("vintage") or aggregate_directory.parent.name
    )
    inputs = load_aggregation_inputs(processed_dir)
    indices = inputs["indices"]
    weights = inputs["weights"]
    model_history = historical_model_component_reconstruction(processed_dir)

    fitted_raw: dict[str, dict] = {}
    preparation_audit: dict[str, dict] = {}
    for aggregate_key, run_directory in run_directories.items():
        metadata = _read_json(run_directory / "metadata.json")
        canonical_model_id = _COMPONENT_RUN_IDS[aggregate_key]
        panel = build_panel(
            canonical_model_id,
            str(metadata.get("vintage") or aggregate_directory.parent.name),
            project_root=project_root,
        )
        target = (
            _WEEKLY_TARGETS[aggregate_key]
            if aggregate_key in _WEEKLY_TARGETS
            else _DIRECT_TARGETS.get(aggregate_key, panel.target)
        )
        with np.load(run_directory / "draws.npz", allow_pickle=False) as archive:
            draws = {name: archive[name] for name in archive.files if name in {"B", "completed_differences_draws"}}
        fitted = _one_step_fitted_target(
            panel,
            metadata,
            draws,
            target=target,
            max_draws=max_draws,
        )
        fitted_raw[aggregate_key] = fitted
        preparation_audit[aggregate_key] = {
            "model_id": canonical_model_id,
            "run_id": str(metadata.get("run_id") or run_directory.name),
            "target": target,
            "posterior_draws_used": int(len(fitted["draw_indices"])),
            "missing_data_method": fitted["missing_data_method"],
            "conditioning_history": fitted["conditioning_history"],
            "fit_start": pd.Timestamp(fitted["dates"].min()).isoformat(),
            "fit_end": pd.Timestamp(fitted["dates"].max()).isoformat(),
        }

    # --- Same HICP adapters as forecast aggregation ---------------------
    gas_context = load_gas_tax_context(processed_dir / model_spec("gas").dataset_file)
    electricity_context = load_electricity_tax_context(
        processed_dir / model_spec("electricity").dataset_file
    )
    components: dict[str, dict] = {}
    components["gas"] = _monthly_pretax_to_hicp(
        fitted_raw["gas"],
        tax_context=gas_context,
        expand_function=expand_gas_semester_series,
        reattribute_function=reattribute_gas_taxes,
        start_date=start_date,
    )
    components["electricity"] = _monthly_pretax_to_hicp(
        fitted_raw["electricity"],
        tax_context=electricity_context,
        expand_function=expand_electricity_semester_series,
        reattribute_function=reattribute_electricity_taxes,
        start_date=start_date,
    )

    for name in ("heat_energy", "solid_fuels"):
        obj = fitted_raw[name]
        dates = pd.DatetimeIndex(obj["dates"], name="date")
        paths = np.asarray(obj["level_paths"], dtype=float)
        if start_date is not None:
            mask = dates >= start_date
            dates = dates[mask]
            paths = paths[:, mask]
        if len(dates) == 0:
            raise FittedMaterialisationError(
                f"{name}: no fitted HICP dates survive the requested lower bound."
            )
        components[name] = {
            "dates": dates,
            "paths": _drop_bad_draws(
                paths,
                label=f"{name} fitted HICP",
            ),
        }

    weekly_components: dict[str, dict] = {}
    for name in ("petrol", "diesel", "liquid_fuels"):
        context = load_weekly_tax_context(
            processed_dir / model_spec(name).dataset_file,
            name,
        )
        weekly_components[name] = _weekly_pretax_to_hicp(
            fitted_raw[name],
            context=context,
            hicp_history=indices[_WEEKLY_HICP_COLUMNS[name]],
            start_date=start_date,
            label=name,
        )
    components["liquid_fuels"] = weekly_components["liquid_fuels"]
    weekly_bridge_audit = {
        name: {
            "anchor_date": pd.Timestamp(obj["anchor_date"]).isoformat(),
            "fitted_start_month": pd.Timestamp(obj["fitted_start_month"]).isoformat(),
            "bridge_method": str(obj["bridge_method"]),
        }
        for name, obj in weekly_components.items()
    }

    # --- petrol + diesel + OBSERVED other-transport residual -> car fuels --
    #
    # Other transport fuels has no dedicated BVAR.  For a historical fitted
    # diagnostic it is an observed residual input.  Unlike the future forecast
    # convention (last published level held flat), an unpublished historical
    # residual month is never invented: the transport fitted window is trimmed
    # to the maximal published overlap.
    transport_start = TRANSPORT_FITTED_START
    if start_date is not None:
        transport_start = max(transport_start, pd.Timestamp(start_date))
    transport_candidate_dates = _common_monthly_dates(
        [weekly_components["petrol"], weekly_components["diesel"]],
        transport_start,
    )
    petrol_candidate = _slice_object(
        weekly_components["petrol"], transport_candidate_dates
    )
    diesel_candidate = _slice_object(
        weekly_components["diesel"], transport_candidate_dates
    )
    n_transport = min(
        int(max_draws), len(petrol_candidate), len(diesel_candidate)
    )
    other_obj = _historical_actual_paths(
        indices["hicp_other_transport_fuels"],
        transport_candidate_dates,
        n_transport,
        label="HICP other transport fuels",
    )
    transport_dates = pd.DatetimeIndex(other_obj["dates"], name="date")
    petrol = _slice_object(weekly_components["petrol"], transport_dates)
    diesel = _slice_object(weekly_components["diesel"], transport_dates)
    other = np.asarray(other_obj["paths"], dtype=float)
    transport_paired, _ = independent_draw_pairing(
        {
            "hicp_petrol": petrol,
            "hicp_diesel": diesel,
            "hicp_other_transport_fuels": other,
        },
        n_draws=n_transport,
        seed=int(pairing_seed),
    )
    transport_history = pd.concat(
        [
            pd.Series(
                [100.0],
                index=pd.DatetimeIndex([TRANSPORT_ANCHOR], name="date"),
            ),
            model_history["transport_energy"]["index"],
        ]
    ).sort_index()
    transport_history.name = "car_fuels"
    car_fuels = aggregate_component_draw_paths_laspeyres(
        indices[list(_TRANSPORT_COMPONENTS)],
        transport_paired,
        transport_dates,
        weights[list(_TRANSPORT_COMPONENTS)],
        transport_history,
        components=list(_TRANSPORT_COMPONENTS),
    )
    components["car_fuels"] = {
        "dates": pd.DatetimeIndex(car_fuels["path_dates"], name="date"),
        "paths": np.asarray(car_fuels["level_paths"], dtype=float),
    }

    # --- Six model fitted HICP paths -> HICP Energy, DRAW BY DRAW --------
    ordered_components = list(MODEL_AGGREGATE_COMPONENTS)
    common_dates = _common_monthly_dates(
        [components[name] for name in ordered_components], start_date
    )
    unpaired = {
        name: _slice_object(components[name], common_dates)
        for name in ordered_components
    }
    n_aggregate = min(int(max_draws), *(len(value) for value in unpaired.values()))
    paired, pairing_indices = independent_draw_pairing(
        unpaired,
        n_draws=n_aggregate,
        seed=int(pairing_seed) + 1,
    )
    aggregate = aggregate_draw_paths_laspeyres(
        model_history["component_history"],
        paired,
        common_dates,
        weights,
        indices["hicp_energy"],
    )
    aggregate_levels = np.asarray(aggregate["level_paths"], dtype=float)

    # Build compact long-form summaries. Each displayed component keeps its
    # FULL natural valid history. On the aggregate overlap we still use the
    # exact draw indices selected by the maintained cross-model pairing, so the
    # marginal fitted summaries remain tied to the draws feeding aggregation.
    frames: list[pd.DataFrame] = []

    # Seven source BVAR fitted HICP targets. Petrol and Diesel are kept
    # separate here; Car fuels below remains the six-component accounting
    # aggregate used by HICP Energy and is therefore not counted as a BVAR.
    source_objects = {
        "gas": components["gas"],
        "electricity": components["electricity"],
        "heat_energy": components["heat_energy"],
        "solid_fuels": components["solid_fuels"],
        "petrol": weekly_components["petrol"],
        "diesel": weekly_components["diesel"],
        "liquid_fuels": weekly_components["liquid_fuels"],
    }
    source_history_map = {
        "gas": indices["hicp_gas"],
        "electricity": indices["hicp_electricity"],
        "heat_energy": indices["hicp_heat_cooling_energy"],
        "solid_fuels": indices["hicp_solid_fuels"],
        "petrol": indices["hicp_petrol"],
        "diesel": indices["hicp_diesel"],
        "liquid_fuels": indices["hicp_liquid_fuels"],
    }
    source_fit_windows: dict[str, dict] = {}
    for name, obj in source_objects.items():
        source_dates = pd.DatetimeIndex(obj["dates"], name="date")
        source_levels = np.asarray(obj["paths"], dtype=float)
        source_yoy = _yoy_paths(source_levels, source_dates, source_history_map[name])
        frames.append(
            _summary_frame(
                source_levels, source_dates, series=name, metric="level",
                basis="source_bvar_fitted",
            )
        )
        frames.append(
            _summary_frame(
                source_yoy, source_dates, series=name, metric="yoy",
                basis="source_bvar_fitted",
            )
        )
        source_fit_windows[name] = {
            "fit_start": pd.Timestamp(source_dates.min()).isoformat(),
            "fit_end": pd.Timestamp(source_dates.max()).isoformat(),
            "n_months": int(len(source_dates)),
        }

    component_history_map = {
        "car_fuels": model_history["component_history"]["car_fuels"],
        "liquid_fuels": indices["hicp_liquid_fuels"],
        "gas": indices["hicp_gas"],
        "electricity": indices["hicp_electricity"],
        "heat_energy": indices["hicp_heat_cooling_energy"],
        "solid_fuels": indices["hicp_solid_fuels"],
    }
    component_fit_windows: dict[str, dict] = {}
    for name in ordered_components:
        component_dates = pd.DatetimeIndex(components[name]["dates"], name="date")
        component_paths = np.asarray(components[name]["paths"], dtype=float)
        selected_indices = np.asarray(pairing_indices[name], dtype=int)
        if selected_indices.size == 0 or selected_indices.max() >= len(component_paths):
            raise FittedMaterialisationError(
                f"{name}: aggregate pairing indices are incompatible with the "
                "full component fitted draw pool."
            )
        levels = component_paths[selected_indices]
        yoy = _yoy_paths(levels, component_dates, component_history_map[name])
        frames.append(
            _summary_frame(
                levels,
                component_dates,
                series=name,
                metric="level",
                basis="component_bvar_fitted",
            )
        )
        frames.append(
            _summary_frame(
                yoy,
                component_dates,
                series=name,
                metric="yoy",
                basis="component_bvar_fitted",
            )
        )
        component_fit_windows[name] = {
            "fit_start": pd.Timestamp(component_dates.min()).isoformat(),
            "fit_end": pd.Timestamp(component_dates.max()).isoformat(),
            "n_months": int(len(component_dates)),
        }

    aggregate_yoy = _yoy_paths(
        aggregate_levels,
        common_dates,
        indices["hicp_energy"],
    )
    frames.append(
        _summary_frame(
            aggregate_levels,
            common_dates,
            series="hicp_energy",
            metric="level",
            basis="aggregate_bvar_fitted",
        )
    )
    frames.append(
        _summary_frame(
            aggregate_yoy,
            common_dates,
            series="hicp_energy",
            metric="yoy",
            basis="aggregate_bvar_fitted",
        )
    )
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["basis", "series", "metric", "date"]).reset_index(drop=True)
    contract_audit = validate_fitted_frame(frame)

    cache_path = _write_cache(frame, aggregate_directory) if persist_cache else None
    meta = {
        "available": True,
        "cache_version": FITTED_CACHE_VERSION,
        "cache_path": None if cache_path is None else str(cache_path),
        "persisted": bool(persist_cache),
        "canonical_discriminator": "basis",
        "fitted_contract": contract_audit,
        "definition": (
            "posterior one-step conditional fitted target levels; no in-sample "
            "innovation is added; tax/frequency/HICP adapters match aggregate forecasting"
        ),
        "source_bvar_overlay_definition": (
            "seven source Energy BVAR posterior one-step fitted HICP targets; "
            "Petrol and Diesel remain separate source models; Car fuels is not "
            "counted as a source BVAR"
        ),
        "source_bvar_fit_windows": source_fit_windows,
        "component_overlay_definition": (
            "six HICP component posterior fitted summaries over each component's "
            "full natural valid history; the same deterministic aggregate-pairing "
            "draw indices are used on every component; Car fuels combines fitted "
            "Petrol/Diesel with observed Other transport fuels"
        ),
        "aggregate_overlay_definition": (
            "draw-wise Laspeyres aggregation of the six fitted HICP component paths; "
            "quantiles are computed only after aggregation"
        ),
        "deterministic_accounting_reconstruction": "audit_only_not_plotted_as_model_fit",
        "cross_model_dependence": "independent posterior chains; deterministic random draw pairing",
        "pairing_seed": int(pairing_seed),
        "n_aggregate_fitted_draws": int(n_aggregate),
        "history_policy": "full_natural_valid_history",
        "requested_start_date": (
            None if start_date is None else pd.Timestamp(start_date).isoformat()
        ),
        # Backward-compatible aggregate window aliases.
        "fit_start": pd.Timestamp(common_dates.min()).isoformat(),
        "fit_end": pd.Timestamp(common_dates.max()).isoformat(),
        "aggregate_fit_start": pd.Timestamp(common_dates.min()).isoformat(),
        "aggregate_fit_end": pd.Timestamp(common_dates.max()).isoformat(),
        "component_fit_windows": component_fit_windows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_signature": signature,
        "preparation_audit": preparation_audit,
        "weekly_hicp_rebase_audit": weekly_bridge_audit,
        "car_fuels_residual_policy": {
            "component": "hicp_other_transport_fuels",
            "modelled": False,
            "production_chain_link_anchor": TRANSPORT_ANCHOR.isoformat(),
            "earliest_honest_fitted_month": TRANSPORT_FITTED_START.isoformat(),
            "historical_fitted_policy": (
                "published observed HICP only; trim leading/trailing ragged edge; "
                "hard-fail on any interior missing/non-positive month"
            ),
            "forecast_policy_is_different": (
                "future aggregate forecasting may hold the latest published residual "
                "level constant; that convention is not used for historical fitted diagnostics"
            ),
            "published_start": pd.Timestamp(other_obj["published_start"]).isoformat(),
            "published_end": pd.Timestamp(other_obj["published_end"]).isoformat(),
            "trimmed_leading_months": int(other_obj["trimmed_leading_months"]),
            "trimmed_trailing_months": int(other_obj["trimmed_trailing_months"]),
            "effective_transport_fit_start": pd.Timestamp(transport_dates.min()).isoformat(),
            "effective_transport_fit_end": pd.Timestamp(transport_dates.max()).isoformat(),
        },
        "aggregate_pairing_pool_indices_recorded": {
            name: int(len(np.asarray(index)))
            for name, index in pairing_indices.items()
        },
    }
    if persist_cache:
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return frame, meta, bool(persist_cache)


def load_energy_aggregate_fitted(
    aggregate_directory: str | Path,
    *,
    project_root: str | Path,
    max_draws: int = DEFAULT_MAX_DRAWS,
    pairing_seed: int = DEFAULT_PAIRING_SEED,
    start_date: str | pd.Timestamp | None = DEFAULT_START_DATE,
    overwrite: bool = False,
    persist_cache: bool = True,
) -> tuple[pd.DataFrame, dict, bool]:
    """Public dashboard entry point; alias kept intentionally descriptive."""
    return build_energy_aggregate_fitted(
        aggregate_directory,
        project_root=project_root,
        max_draws=max_draws,
        pairing_seed=pairing_seed,
        start_date=start_date,
        overwrite=overwrite,
        persist_cache=persist_cache,
    )


__all__ = [
    "FITTED_CACHE_VERSION",
    "FITTED_CACHE_BASENAME",
    "FITTED_META_FILENAME",
    "FittedMaterialisationError",
    "validate_fitted_frame",
    "build_energy_aggregate_fitted",
    "load_energy_aggregate_fitted",
]
