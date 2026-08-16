"""Presentation-only VAT / excise decomposition of HICP Energy contributions.

The existing aggregate already provides an exact Laspeyres contribution identity.
This module does not re-estimate a BVAR and does not alter the aggregate.  It
splits each *existing aggregate term* into economically ordered tax layers:

    pre-tax + excise wedge + VAT wedge + unsplit = gross contribution.

For gas/electricity the tax shares come from the same Eurostat VAT/excise bridge
used by the production aggregate.  For petrol/diesel/liquid fuels they come
from the same Weekly Oil Bulletin (WOB) pre-tax / IDT / VAT block used by the
production aggregate.  Heat energy and solid fuels have no separate tax bridge
in the current model contract and therefore remain ``unsplit``.  Inside Car
fuels, the unmodelled Other transport fuels residual also remains ``unsplit``.

VAT is the final legal layer, so VAT-on-excise belongs to the VAT wedge.  The
construction is performed at Laspeyres-term level, which means the split remains
exactly additive through annual weight changes and across the forecast origin.

For year-on-year contributions, a tax split is shown only when the tax-layer
identity is available both in month t and in month t-12.  If either side of the
year-on-year comparison is unidentified, the exact existing component
contribution is kept entirely in ``Unsplit``.  This prevents a change in tax-data
coverage from appearing as a spurious market/VAT/excise economic shock.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TAX_DECOMPOSITION_VERSION = "energy-tax-contribution-v1.2.1"

LAYERS: tuple[str, ...] = ("pre_tax", "excise", "vat", "unsplit")
LAYER_LABELS = {
    "pre_tax": "Pre-tax / market",
    "excise": "Excise",
    "vat": "VAT",
    "unsplit": "Unsplit",
}
LAYER_COLOURS = {
    "pre_tax": "#0B4F6C",
    "excise": "#C47A00",
    "vat": "#7C3AED",
    "unsplit": "#94A3B8",
}

COMPONENTS: tuple[str, ...] = (
    "car_fuels",
    "liquid_fuels",
    "gas",
    "electricity",
    "heat_energy",
    "solid_fuels",
)
COMPONENT_LABELS = {
    "car_fuels": "Car fuels",
    "liquid_fuels": "Liquid fuels",
    "gas": "Gas",
    "electricity": "Electricity",
    "heat_energy": "Heat energy",
    "solid_fuels": "Solid fuels",
}

TRANSPORT_COMPONENTS = (
    "hicp_diesel",
    "hicp_petrol",
    "hicp_other_transport_fuels",
)
TRANSPORT_ANCHOR = pd.Timestamp("2016-12-01")

WEEKLY_SOURCE_KEY = {
    "petrol": "car_fuels_petrol",
    "diesel": "car_fuels_diesel",
    "liquid_fuels": "liquid_fuels",
}
WEEKLY_HICP_COLUMN = {
    "petrol": "hicp_petrol",
    "diesel": "hicp_diesel",
    "liquid_fuels": "hicp_liquid_fuels",
}


class TaxDecompositionError(RuntimeError):
    """Raised when a saved aggregate cannot be replayed exactly enough."""


def _metadata(directory: Path) -> dict[str, Any]:
    path = Path(directory) / "metadata.json"
    if not path.is_file():
        raise TaxDecompositionError(f"Aggregate metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _arrays(directory: Path) -> dict[str, np.ndarray]:
    path = Path(directory) / "aggregate_draws.npz"
    if not path.is_file():
        raise TaxDecompositionError(f"Aggregate draws not found: {path}")
    with np.load(path, allow_pickle=False) as store:
        return {key: np.asarray(store[key]) for key in store.files}


def _dates(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> pd.DatetimeIndex:
    if "aggregate_dates" in arrays:
        out = pd.DatetimeIndex(pd.to_datetime(arrays["aggregate_dates"]), name="date")
    else:
        out = pd.DatetimeIndex(pd.to_datetime(metadata.get("path_dates", [])), name="date")
    if len(out) == 0:
        raise TaxDecompositionError("Aggregate calendar is empty.")
    return out


def _component_names(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> list[str]:
    names = list(metadata.get("component_index_names", []) or metadata.get("components", []) or [])
    paths = np.asarray(arrays.get("component_index_paths"))
    if paths.ndim != 3:
        raise TaxDecompositionError(
            "aggregate_draws.npz must contain component_index_paths with shape "
            "(draws, dates, components)."
        )
    if len(names) != paths.shape[2]:
        if paths.shape[2] == len(COMPONENTS):
            names = list(COMPONENTS)
        else:
            raise TaxDecompositionError("Aggregate component names are absent/inconsistent.")
    return [str(x) for x in names]


def _model_hicp_names(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> list[str]:
    names = list(metadata.get("model_hicp_names", []) or [])
    paths = np.asarray(arrays.get("model_hicp_index_paths"))
    if paths.ndim != 3:
        raise TaxDecompositionError(
            "This aggregate predates model_hicp_index_paths. Rebuild the aggregate "
            "with the current production pipeline before requesting the tax split."
        )
    if len(names) != paths.shape[2]:
        raise TaxDecompositionError("model_hicp_names are absent/inconsistent.")
    return [str(x) for x in names]


def _align_2d(paths: np.ndarray, source_dates: Sequence[pd.Timestamp], target_dates: pd.DatetimeIndex) -> np.ndarray:
    source = pd.DatetimeIndex(source_dates, name="date")
    positions = source.get_indexer(target_dates)
    if (positions < 0).any():
        missing = target_dates[positions < 0].strftime("%Y-%m-%d").tolist()
        raise TaxDecompositionError(f"Source path is missing aggregate months {missing}.")
    values = np.asarray(paths, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(source):
        raise TaxDecompositionError("Expected draw x date source paths.")
    return values[:, positions]


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float)
    num_b = np.broadcast_to(num, out.shape)
    den_b = np.broadcast_to(den, out.shape)
    mask = np.broadcast_to(valid, out.shape)
    out[mask] = num_b[mask] / den_b[mask]
    return out, mask


def _normalised_shares(pre: np.ndarray, excise_inclusive: np.ndarray, gross: np.ndarray) -> dict[str, np.ndarray]:
    """Return shares that sum to one; unidentified cells become unsplit."""
    pre = np.asarray(pre, dtype=float)
    exc = np.asarray(excise_inclusive, dtype=float)
    gross = np.asarray(gross, dtype=float)
    shape = np.broadcast_shapes(pre.shape, exc.shape, gross.shape)
    pre = np.broadcast_to(pre, shape)
    exc = np.broadcast_to(exc, shape)
    gross = np.broadcast_to(gross, shape)

    valid = (
        np.isfinite(pre)
        & np.isfinite(exc)
        & np.isfinite(gross)
        & (gross > 0)
        & (pre >= 0)
    )
    s_pre = np.zeros(shape, dtype=float)
    s_exc = np.zeros(shape, dtype=float)
    s_vat = np.zeros(shape, dtype=float)
    s_unsplit = np.ones(shape, dtype=float)

    if valid.any():
        raw_pre = pre[valid] / gross[valid]
        raw_exc_inclusive = exc[valid] / gross[valid]
        # Small published tax-identity discrepancies are absorbed by the final
        # VAT residual so the legal ordering stays exact. Clamp only tiny
        # numerical excursions; materially pathological cells become unsplit.
        good = (
            np.isfinite(raw_pre)
            & np.isfinite(raw_exc_inclusive)
            & (raw_pre >= -1e-8)
            & (raw_exc_inclusive >= raw_pre - 1e-8)
            & (raw_exc_inclusive <= 1.0 + 5e-2)
        )
        idx = np.flatnonzero(valid)
        good_idx = idx[good]
        if len(good_idx):
            flat_pre = s_pre.ravel()
            flat_exc = s_exc.ravel()
            flat_vat = s_vat.ravel()
            flat_unsplit = s_unsplit.ravel()
            rp = np.clip(raw_pre[good], 0.0, 1.0)
            re = np.clip(raw_exc_inclusive[good], rp, 1.0)
            flat_pre[good_idx] = rp
            flat_exc[good_idx] = re - rp
            flat_vat[good_idx] = 1.0 - re
            flat_unsplit[good_idx] = 0.0

    total = s_pre + s_exc + s_vat + s_unsplit
    if not np.allclose(total, 1.0, atol=1e-12, rtol=0.0):
        raise TaxDecompositionError("Tax shares do not sum to one.")
    return {
        "pre_tax": s_pre,
        "excise": s_exc,
        "vat": s_vat,
        "unsplit": s_unsplit,
    }


def _monthly_mean(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if values.empty:
        return values
    grouped = values.groupby(values.index.to_period("M")).mean()
    grouped.index = pd.DatetimeIndex(grouped.index.to_timestamp(how="start"), name="date")
    return grouped


def _monthly_tax_shares_history(model: str, context: Mapping[str, Any], dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    data = pd.DataFrame(context["data"]).copy().sort_index()
    pre = _monthly_mean(data["pre_tax"]).reindex(dates).to_numpy(dtype=float)
    exc = _monthly_mean(data["pre_tax"] + data["excise"]).reindex(dates).to_numpy(dtype=float)
    gross = _monthly_mean(data["after_tax"]).reindex(dates).to_numpy(dtype=float)
    return _normalised_shares(pre, exc, gross)


def _monthly_gas_electricity_shares_history(
    gross_hicp: pd.Series,
    context: Mapping[str, Any],
    dates: pd.DatetimeIndex,
    expand_function,
) -> dict[str, np.ndarray]:
    gross_index = pd.to_numeric(gross_hicp, errors="coerce").reindex(dates).to_numpy(dtype=float)
    vat = expand_function(context["vat_percent"], dates).to_numpy(dtype=float)
    excise = expand_function(context["excise"], dates).to_numpy(dtype=float)
    gamma = float(context["gamma"])
    gross_price = gamma * gross_index
    with np.errstate(divide="ignore", invalid="ignore"):
        pre = gross_price / (1.0 + vat / 100.0) - excise
    excise_inclusive = pre + excise
    return _normalised_shares(pre, excise_inclusive, gross_price)


def _forecast_store(metadata: Mapping[str, Any], key: str) -> Path:
    stores = dict(metadata.get("component_forecast_stores", {}) or {})
    raw = stores.get(key)
    if raw is None:
        raise TaxDecompositionError(f"Aggregate metadata does not record forecast store {key!r}.")
    return Path(str(raw))


def _monthly_model_tax_shares_forecast(
    model: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    aggregate_dates: pd.DatetimeIndex,
    processed_dir: Path,
) -> dict[str, np.ndarray]:
    from energy_bvar_io import load_energy_bvar_forecast
    from energy_bvar_pipeline import model_spec

    if model == "gas":
        from energy_bvar_gas import expand_semester_series, load_gas_tax_context
        target = "gas_pre_tax"
        pair_key = "final_gas_pair_indices"
        context = load_gas_tax_context(processed_dir / model_spec("gas").dataset_file)
    elif model == "electricity":
        from energy_bvar_electricity import expand_semester_series, load_electricity_tax_context
        target = "electricity_pre_tax"
        pair_key = "final_electricity_pair_indices"
        context = load_electricity_tax_context(
            processed_dir / model_spec("electricity").dataset_file
        )
    else:
        raise TaxDecompositionError(f"Unsupported monthly taxed model {model!r}.")

    forecast = load_energy_bvar_forecast(_forecast_store(metadata, model))
    source_dates = pd.DatetimeIndex(forecast["path_dates"], name="date")
    variables = list(forecast["variables"])
    if target not in variables:
        raise TaxDecompositionError(f"{model}: target {target!r} absent from forecast.")
    j = variables.index(target)
    all_pre = np.asarray(forecast["level_paths"], dtype=float)[:, :, j]
    paired = np.asarray(arrays[pair_key], dtype=int)
    if paired.max(initial=-1) >= len(all_pre):
        raise TaxDecompositionError(f"{model}: saved pairing indices exceed source draw pool.")
    pre = _align_2d(all_pre[paired], source_dates, aggregate_dates)
    vat = expand_semester_series(context["vat_percent"], aggregate_dates).to_numpy(dtype=float)[None, :]
    exc = expand_semester_series(context["excise"], aggregate_dates).to_numpy(dtype=float)[None, :]
    excise_inclusive = pre + exc
    gross = excise_inclusive * (1.0 + vat / 100.0)
    return _normalised_shares(pre, excise_inclusive, gross)


def _weekly_source_draw_indices(model: str, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    retained = np.asarray(arrays[f"{model}_retained_original_draw_indices"], dtype=int)
    if model == "liquid_fuels":
        final = np.asarray(arrays["final_liquid_fuels_pair_indices"], dtype=int)
        if final.max(initial=-1) >= len(retained):
            raise TaxDecompositionError("liquid_fuels pairing exceeds retained draw pool.")
        return retained[final]
    if model in {"petrol", "diesel"}:
        transport = np.asarray(arrays[f"transport_{model}_pair_indices"], dtype=int)
        final_car = np.asarray(arrays["final_car_fuels_pair_indices"], dtype=int)
        if final_car.max(initial=-1) >= len(transport):
            raise TaxDecompositionError(f"{model}: final car-fuels pairing exceeds transport pool.")
        selected_retained_positions = transport[final_car]
        if selected_retained_positions.max(initial=-1) >= len(retained):
            raise TaxDecompositionError(f"{model}: transport pairing exceeds retained pool.")
        return retained[selected_retained_positions]
    raise TaxDecompositionError(f"Unknown weekly model {model!r}.")


def _weekly_model_tax_shares_forecast(
    model: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    aggregate_dates: pd.DatetimeIndex,
    processed_dir: Path,
) -> dict[str, np.ndarray]:
    from energy_bvar_aggregate import weekly_paths_with_history_to_monthly_mean
    from energy_bvar_io import load_energy_bvar_forecast
    from energy_bvar_pipeline import model_spec
    from energy_bvar_weekly_fuels import load_weekly_tax_context, reattribute_weekly_taxes

    forecast = load_energy_bvar_forecast(_forecast_store(metadata, model))
    context = load_weekly_tax_context(
        processed_dir / model_spec(model).dataset_file,
        model=model,
    )
    taxed = reattribute_weekly_taxes(forecast, context)
    weekly_dates = pd.DatetimeIndex(taxed["path_dates"], name="date")
    pre_weekly = np.asarray(taxed["pre_tax_level_paths"], dtype=float)
    exc_weekly = pre_weekly + np.asarray(taxed["baseline_excise"], dtype=float)[None, :]
    gross_weekly = np.asarray(taxed["baseline_post_tax_level_paths"], dtype=float)

    data = pd.DataFrame(context["data"]).sort_index()
    histories = {
        "pre": data["pre_tax"],
        "exc": data["pre_tax"] + data["excise"],
        "gross": data["after_tax"],
    }
    monthly = {}
    monthly_dates = None
    for name, paths, history in (
        ("pre", pre_weekly, histories["pre"]),
        ("exc", exc_weekly, histories["exc"]),
        ("gross", gross_weekly, histories["gross"]),
    ):
        values, dates, _coverage = weekly_paths_with_history_to_monthly_mean(
            paths,
            weekly_dates,
            history,
            return_coverage=True,
        )
        if monthly_dates is None:
            monthly_dates = pd.DatetimeIndex(dates, name="date")
        elif not monthly_dates.equals(pd.DatetimeIndex(dates, name="date")):
            raise TaxDecompositionError(f"{model}: tax-layer monthly calendars differ.")
        monthly[name] = np.asarray(values, dtype=float)

    assert monthly_dates is not None
    source_indices = _weekly_source_draw_indices(model, arrays)
    if source_indices.max(initial=-1) >= len(monthly["gross"]):
        raise TaxDecompositionError(f"{model}: saved pairing exceeds source monthly draw pool.")

    # Weekly WOB paths are converted to COMPLETE monthly means.  Around the
    # ragged edge this calendar can legitimately be shorter than the aggregate
    # HICP calendar (for example, a partial final month).  Missing tax-share
    # months are therefore *unidentified*, not an error and not zero-tax.
    # Keep the exact aggregate contribution on those dates by assigning the
    # whole gross term to ``Unsplit``.  Where a complete WOB tax month exists,
    # use the exact pre-tax -> excise-inclusive -> gross legal layering.
    positions = monthly_dates.get_indexer(aggregate_dates)
    available = positions >= 0
    shape = (len(source_indices), len(aggregate_dates))
    shares = {
        "pre_tax": np.zeros(shape, dtype=float),
        "excise": np.zeros(shape, dtype=float),
        "vat": np.zeros(shape, dtype=float),
        "unsplit": np.ones(shape, dtype=float),
    }
    if available.any():
        source_positions = positions[available]
        pre = monthly["pre"][source_indices][:, source_positions]
        exc = monthly["exc"][source_indices][:, source_positions]
        gross = monthly["gross"][source_indices][:, source_positions]
        identified = _normalised_shares(pre, exc, gross)
        for layer in LAYERS:
            shares[layer][:, available] = identified[layer]

    total = sum(shares[layer] for layer in LAYERS)
    if not np.allclose(total, 1.0, atol=1e-12, rtol=0.0):
        raise TaxDecompositionError(
            f"{model}: aligned weekly tax shares do not sum to one."
        )
    return shares


def _layer_terms_from_shares(gross_terms: np.ndarray, shares: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    gross = np.asarray(gross_terms, dtype=float)
    out = {}
    for layer in LAYERS:
        share = np.asarray(shares[layer], dtype=float)
        out[layer] = gross * np.broadcast_to(share, gross.shape)
    replay = sum(out[layer] for layer in LAYERS)
    error = np.nanmax(np.abs(replay - gross)) if replay.size else 0.0
    if np.isfinite(error) and error > 1e-9:
        raise TaxDecompositionError(f"Layer terms do not replay gross terms: {error:.3e}.")
    return out


def _contribution_paths(
    forecast_terms: np.ndarray,
    historical_terms: pd.Series,
    aggregate_paths: np.ndarray,
    aggregate_history: pd.Series,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    terms = np.asarray(forecast_terms, dtype=float)
    aggregate = np.asarray(aggregate_paths, dtype=float)
    if terms.ndim != 2 or aggregate.ndim != 2 or terms.shape != aggregate.shape:
        raise TaxDecompositionError("Forecast term/aggregate shapes are inconsistent.")

    hist_term = pd.to_numeric(historical_terms, errors="coerce").sort_index()
    hist_agg = pd.to_numeric(aggregate_history, errors="coerce").sort_index()
    full_index = pd.date_range(
        min(hist_term.index.min(), hist_agg.index.min(), dates.min()),
        dates.max(),
        freq="MS",
        name="date",
    )
    positions = full_index.get_indexer(dates)
    if (positions < 0).any() or (positions < 12).any():
        raise TaxDecompositionError("Insufficient 12-month history for tax contributions.")

    term_base = hist_term.reindex(full_index).to_numpy(dtype=float)
    agg_base = hist_agg.reindex(full_index).to_numpy(dtype=float)
    term_full = np.repeat(term_base[None, :], terms.shape[0], axis=0)
    agg_full = np.repeat(agg_base[None, :], aggregate.shape[0], axis=0)
    term_full[:, positions] = terms
    agg_full[:, positions] = aggregate
    lag_positions = positions - 12
    denom = agg_full[:, lag_positions]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * (term_full[:, positions] - term_full[:, lag_positions]) / denom
    return out


def _history_contribution(layer_term: pd.Series, aggregate_history: pd.Series) -> pd.Series:
    term = pd.to_numeric(layer_term, errors="coerce").sort_index()
    agg = pd.to_numeric(aggregate_history, errors="coerce").sort_index()
    denom = agg.shift(12).reindex(term.index)
    return 100.0 * (term - term.shift(12)) / denom


def _history_split_identified(
    layer_terms: Mapping[str, pd.Series],
) -> pd.Series:
    """Whether a historical gross term has an identified tax-layer split.

    ``Unsplit == gross`` means no tax bridge is available.  A partially split
    Car-fuels term remains identified because petrol/diesel can be separated
    even though Other transport fuels stays in Unsplit.
    """
    index = pd.DatetimeIndex(layer_terms["unsplit"].index, name="date")
    gross = pd.Series(0.0, index=index, dtype=float)
    for layer in LAYERS:
        gross = gross.add(
            pd.to_numeric(layer_terms[layer], errors="coerce").reindex(index),
            fill_value=0.0,
        )
    unsplit = pd.to_numeric(layer_terms["unsplit"], errors="coerce").reindex(index)
    scale = np.maximum(np.abs(gross.to_numpy(dtype=float)), 1.0)
    identified = (
        np.isfinite(gross.to_numpy(dtype=float))
        & np.isfinite(unsplit.to_numpy(dtype=float))
        & (np.abs(gross.to_numpy(dtype=float) - unsplit.to_numpy(dtype=float)) > 1e-10 * scale)
    )
    return pd.Series(identified, index=index, dtype=bool)


def _forecast_split_identified(
    layer_terms: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Draw/date mask for whether a forecast gross term is tax identified."""
    gross = sum(np.asarray(layer_terms[layer], dtype=float) for layer in LAYERS)
    unsplit = np.asarray(layer_terms["unsplit"], dtype=float)
    scale = np.maximum(np.abs(gross), 1.0)
    return (
        np.isfinite(gross)
        & np.isfinite(unsplit)
        & (np.abs(gross - unsplit) > 1e-10 * scale)
    )


def _historical_comparable_yoy_mask(
    layer_terms: Mapping[str, pd.Series],
) -> pd.Series:
    """Split only when t and t-12 are both tax identified."""
    identified = _history_split_identified(layer_terms)
    return identified & identified.shift(12, fill_value=False)


def _forecast_comparable_yoy_mask(
    history_layer_terms: Mapping[str, pd.Series],
    forecast_layer_terms: Mapping[str, np.ndarray],
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Draw/date split-eligibility using the exact t versus t-12 calendar.

    Historical identification supplies lag months before the saved model path;
    forecast identification overwrites the model-path block.  The resulting
    mask is True only when both sides of the y/y comparison are identified.
    """
    dates = pd.DatetimeIndex(dates, name="date")
    forecast_identified = _forecast_split_identified(forecast_layer_terms)
    if forecast_identified.ndim != 2 or forecast_identified.shape[1] != len(dates):
        raise TaxDecompositionError("Forecast tax-identification mask has invalid shape.")

    historical_identified = _history_split_identified(history_layer_terms)
    full_index = pd.date_range(
        min(historical_identified.index.min(), dates.min() - pd.DateOffset(months=12)),
        dates.max(),
        freq="MS",
        name="date",
    )
    base = historical_identified.reindex(full_index, fill_value=False).to_numpy(dtype=bool)
    combined = np.broadcast_to(base[None, :], (forecast_identified.shape[0], len(full_index))).copy()
    positions = full_index.get_indexer(dates)
    if (positions < 0).any() or (positions < 12).any():
        raise TaxDecompositionError("Insufficient calendar for tax comparability mask.")
    combined[:, positions] = forecast_identified
    return combined[:, positions] & combined[:, positions - 12]


def _collapse_historical_incomparable(
    contributions: Mapping[str, pd.Series],
    expected: pd.Series,
    comparable: pd.Series,
) -> dict[str, pd.Series]:
    """Collapse non-comparable y/y dates to the exact existing Unsplit total."""
    index = pd.DatetimeIndex(expected.index, name="date")
    allowed = comparable.reindex(index, fill_value=False).to_numpy(dtype=bool)
    out = {
        layer: pd.to_numeric(contributions[layer], errors="coerce").reindex(index).copy()
        for layer in LAYERS
    }
    for layer in ("pre_tax", "excise", "vat"):
        values = out[layer].to_numpy(dtype=float, copy=True)
        values[~allowed] = 0.0
        out[layer] = pd.Series(values, index=index, dtype=float)
    unsplit = out["unsplit"].to_numpy(dtype=float, copy=True)
    exp = pd.to_numeric(expected, errors="coerce").reindex(index).to_numpy(dtype=float, copy=True)
    unsplit[~allowed] = exp[~allowed]
    out["unsplit"] = pd.Series(unsplit, index=index, dtype=float)
    return out


def _collapse_forecast_incomparable(
    contributions: Mapping[str, np.ndarray],
    expected: np.ndarray,
    comparable: np.ndarray,
) -> dict[str, np.ndarray]:
    """Draw-wise equivalent of _collapse_historical_incomparable."""
    exp = np.asarray(expected, dtype=float)
    allowed = np.asarray(comparable, dtype=bool)
    if allowed.shape != exp.shape:
        raise TaxDecompositionError(
            f"Tax comparability mask {allowed.shape} != contribution shape {exp.shape}."
        )
    out = {
        layer: np.asarray(contributions[layer], dtype=float).copy()
        for layer in LAYERS
    }
    for layer in ("pre_tax", "excise", "vat"):
        out[layer][~allowed] = 0.0
    out["unsplit"][~allowed] = exp[~allowed]
    return out


def _frame_rows_from_paths(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    component: str,
    layer: str,
    segment: str,
) -> list[dict[str, Any]]:
    values = np.asarray(paths, dtype=float)
    q = np.nanquantile(values, (0.05, 0.16, 0.50, 0.84, 0.95), axis=0)
    mean = np.nanmean(values, axis=0)
    rows = []
    for pos, date in enumerate(dates):
        rows.append(
            {
                "date": pd.Timestamp(date),
                "component": component,
                "component_label": COMPONENT_LABELS[component],
                "layer": layer,
                "layer_label": LAYER_LABELS[layer],
                "value": float(mean[pos]) if np.isfinite(mean[pos]) else np.nan,
                "q05": float(q[0, pos]) if np.isfinite(q[0, pos]) else np.nan,
                "q16": float(q[1, pos]) if np.isfinite(q[1, pos]) else np.nan,
                "q50": float(q[2, pos]) if np.isfinite(q[2, pos]) else np.nan,
                "q84": float(q[3, pos]) if np.isfinite(q[3, pos]) else np.nan,
                "q95": float(q[4, pos]) if np.isfinite(q[4, pos]) else np.nan,
                "segment": segment,
                "basis": "tax_layer_accounting",
            }
        )
    return rows


def _frame_rows_from_history(
    values: pd.Series,
    *,
    component: str,
    layer: str,
) -> list[dict[str, Any]]:
    rows = []
    for date, value in pd.to_numeric(values, errors="coerce").items():
        if not np.isfinite(value):
            continue
        rows.append(
            {
                "date": pd.Timestamp(date),
                "component": component,
                "component_label": COMPONENT_LABELS[component],
                "layer": layer,
                "layer_label": LAYER_LABELS[layer],
                "value": float(value),
                "q05": float(value),
                "q16": float(value),
                "q50": float(value),
                "q84": float(value),
                "q95": float(value),
                "segment": "historical",
                "basis": "tax_layer_accounting",
            }
        )
    return rows


def build_tax_contribution_decomposition(
    aggregate_directory: str | Path,
    *,
    project_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build exact historical + saved-forecast VAT/excise contribution split.

    This function is intentionally suitable for the aggregate bootstrap callback:
    it may read the saved aggregate, source forecasts and processed tax context,
    but it writes nothing and performs no estimation.
    """
    from energy_bvar_aggregate import (
        aggregate_component_draw_paths_laspeyres,
        aggregate_draw_paths_laspeyres,
        carry_last_index_path,
        historical_model_component_reconstruction,
        load_aggregation_inputs,
        yoy_contributions_from_terms,
    )
    from energy_bvar_electricity import (
        expand_semester_series as expand_electricity_semester_series,
        load_electricity_tax_context,
    )
    from energy_bvar_gas import (
        expand_semester_series as expand_gas_semester_series,
        load_gas_tax_context,
    )
    from energy_bvar_pipeline import model_spec
    from energy_bvar_weekly_fuels import load_weekly_tax_context

    directory = Path(aggregate_directory)
    root = Path(project_root).resolve()
    metadata = _metadata(directory)
    arrays = _arrays(directory)
    vintage = str(metadata.get("vintage") or directory.parent.name)
    processed_dir = root / "data" / "processed" / vintage
    if not processed_dir.is_dir():
        raise TaxDecompositionError(f"Processed vintage not found: {processed_dir}")

    dates = _dates(metadata, arrays)
    component_names = _component_names(metadata, arrays)
    if set(component_names) != set(COMPONENTS):
        raise TaxDecompositionError(
            f"Unexpected six-component aggregate contract: {component_names}."
        )
    component_positions = {name: component_names.index(name) for name in component_names}
    model_names = _model_hicp_names(metadata, arrays)
    model_positions = {name: model_names.index(name) for name in model_names}
    required_models = {
        "car_fuels_petrol",
        "car_fuels_diesel",
        "liquid_fuels",
        "gas",
        "electricity",
        "heat_energy",
        "solid_fuels",
    }
    missing_models = sorted(required_models.difference(model_positions))
    if missing_models:
        raise TaxDecompositionError(f"Aggregate model HICP paths miss {missing_models}.")

    inputs = load_aggregation_inputs(processed_dir)
    indices = inputs["indices"]
    weights = inputs["weights"]
    model_history = historical_model_component_reconstruction(processed_dir)
    component_history = model_history["component_history"]
    energy_history = model_history["energy"]
    aggregate_history = pd.to_numeric(indices["hicp_energy"], errors="coerce")

    component_index_paths = np.asarray(arrays["component_index_paths"], dtype=float)
    n_draws = component_index_paths.shape[0]
    if component_index_paths.shape[1] != len(dates):
        raise TaxDecompositionError("component_index_paths calendar mismatch.")
    component_path_map = {
        name: component_index_paths[:, :, component_positions[name]]
        for name in COMPONENTS
    }
    replay = aggregate_draw_paths_laspeyres(
        component_history,
        component_path_map,
        dates,
        weights,
        aggregate_history,
    )
    saved_levels = np.asarray(arrays["baseline_level_paths"], dtype=float)
    level_replay_error = float(np.nanmax(np.abs(replay["level_paths"] - saved_levels)))
    if level_replay_error > 1e-8:
        raise TaxDecompositionError(
            "Saved aggregate cannot be replayed from component_index_paths; "
            f"max level error={level_replay_error:.3e}."
        )
    gross_energy_terms = np.asarray(replay["term_paths"], dtype=float)

    # Reconstruct the exact first-stage Car-fuels terms from the saved individual
    # petrol/diesel HICP paths plus the production residual rule.
    model_hicp = np.asarray(arrays["model_hicp_index_paths"], dtype=float)
    petrol_hicp = model_hicp[:, :, model_positions["car_fuels_petrol"]]
    diesel_hicp = model_hicp[:, :, model_positions["car_fuels_diesel"]]
    other_hicp = carry_last_index_path(
        indices["hicp_other_transport_fuels"], dates, n_draws=n_draws
    )
    transport_history_index = pd.concat(
        [
            pd.Series(
                [100.0],
                index=pd.DatetimeIndex([TRANSPORT_ANCHOR], name="date"),
            ),
            model_history["transport_energy"]["index"],
        ]
    ).sort_index()
    transport_history_index.name = "car_fuels"
    transport_replay = aggregate_component_draw_paths_laspeyres(
        indices[list(TRANSPORT_COMPONENTS)],
        {
            "hicp_diesel": diesel_hicp,
            "hicp_petrol": petrol_hicp,
            "hicp_other_transport_fuels": other_hicp,
        },
        dates,
        weights[list(TRANSPORT_COMPONENTS)],
        transport_history_index,
        components=TRANSPORT_COMPONENTS,
    )
    car_saved = component_path_map["car_fuels"]
    car_replay_error = float(
        np.nanmax(np.abs(transport_replay["level_paths"] - car_saved))
    )
    if car_replay_error > 1e-8:
        raise TaxDecompositionError(
            "Saved Car-fuels path cannot be replayed from petrol/diesel pairing; "
            f"max level error={car_replay_error:.3e}."
        )

    # Tax contexts and historical shares.
    gas_context = load_gas_tax_context(processed_dir / model_spec("gas").dataset_file)
    electricity_context = load_electricity_tax_context(
        processed_dir / model_spec("electricity").dataset_file
    )
    weekly_contexts = {
        model: load_weekly_tax_context(
            processed_dir / model_spec(model).dataset_file,
            model=model,
        )
        for model in ("petrol", "diesel", "liquid_fuels")
    }

    hist_dates = pd.DatetimeIndex(energy_history["terms"].index, name="date")
    historical_shares: dict[str, dict[str, np.ndarray]] = {
        "gas": _monthly_gas_electricity_shares_history(
            component_history["gas"], gas_context, hist_dates, expand_gas_semester_series
        ),
        "electricity": _monthly_gas_electricity_shares_history(
            component_history["electricity"],
            electricity_context,
            hist_dates,
            expand_electricity_semester_series,
        ),
        "liquid_fuels": _monthly_tax_shares_history(
            "liquid_fuels", weekly_contexts["liquid_fuels"], hist_dates
        ),
    }
    transport_hist_dates = pd.DatetimeIndex(
        model_history["transport_energy"]["terms"].index, name="date"
    )
    petrol_hist_shares = _monthly_tax_shares_history(
        "petrol", weekly_contexts["petrol"], transport_hist_dates
    )
    diesel_hist_shares = _monthly_tax_shares_history(
        "diesel", weekly_contexts["diesel"], transport_hist_dates
    )

    # Build deterministic historical layer terms at the *Energy term* level.
    history_layer_terms: dict[str, dict[str, pd.Series]] = {
        component: {
            layer: pd.Series(0.0, index=hist_dates, dtype=float)
            for layer in LAYERS
        }
        for component in COMPONENTS
    }
    energy_terms_hist = pd.DataFrame(energy_history["terms"]).reindex(hist_dates)
    for component in ("gas", "electricity", "liquid_fuels"):
        gross = pd.to_numeric(energy_terms_hist[component], errors="coerce").to_numpy(dtype=float)
        shares = historical_shares[component]
        for layer in LAYERS:
            history_layer_terms[component][layer] = pd.Series(
                gross * shares[layer], index=hist_dates, dtype=float
            )

    for component in ("heat_energy", "solid_fuels"):
        history_layer_terms[component]["unsplit"] = pd.to_numeric(
            energy_terms_hist[component], errors="coerce"
        ).astype(float)

    transport_terms_hist = pd.DataFrame(model_history["transport_energy"]["terms"]).reindex(
        transport_hist_dates
    )
    car_index_hist = pd.to_numeric(
        model_history["transport_energy"]["index"], errors="coerce"
    ).reindex(hist_dates)
    car_energy_term_hist = pd.to_numeric(
        energy_terms_hist["car_fuels"], errors="coerce"
    )
    car_scale_hist = car_energy_term_hist / car_index_hist

    lower_layer_hist = {
        layer: pd.Series(0.0, index=transport_hist_dates, dtype=float)
        for layer in LAYERS
    }
    for source_name, shares in (
        ("hicp_petrol", petrol_hist_shares),
        ("hicp_diesel", diesel_hist_shares),
    ):
        gross = pd.to_numeric(transport_terms_hist[source_name], errors="coerce").to_numpy(dtype=float)
        for layer in LAYERS:
            lower_layer_hist[layer] = lower_layer_hist[layer] + pd.Series(
                gross * shares[layer], index=transport_hist_dates, dtype=float
            )
    lower_layer_hist["unsplit"] = lower_layer_hist["unsplit"] + pd.to_numeric(
        transport_terms_hist["hicp_other_transport_fuels"], errors="coerce"
    ).astype(float)
    for layer in LAYERS:
        history_layer_terms["car_fuels"][layer] = (
            lower_layer_hist[layer].reindex(hist_dates) * car_scale_hist
        )

    # Hard historical term replay before contribution arithmetic.
    hist_term_error = 0.0
    for component in COMPONENTS:
        replay_term = sum(history_layer_terms[component][layer] for layer in LAYERS)
        gross_term = pd.to_numeric(energy_terms_hist[component], errors="coerce")
        diff = (replay_term - gross_term).abs().dropna()
        if len(diff):
            hist_term_error = max(hist_term_error, float(diff.max()))
    if hist_term_error > 1e-8:
        raise TaxDecompositionError(
            f"Historical layer terms fail gross-term replay: {hist_term_error:.3e}."
        )

    # Forecast tax shares on the exact saved aggregate pairing.
    forecast_shares = {
        "gas": _monthly_model_tax_shares_forecast(
            "gas", metadata, arrays, dates, processed_dir
        ),
        "electricity": _monthly_model_tax_shares_forecast(
            "electricity", metadata, arrays, dates, processed_dir
        ),
        "liquid_fuels": _weekly_model_tax_shares_forecast(
            "liquid_fuels", metadata, arrays, dates, processed_dir
        ),
        "petrol": _weekly_model_tax_shares_forecast(
            "petrol", metadata, arrays, dates, processed_dir
        ),
        "diesel": _weekly_model_tax_shares_forecast(
            "diesel", metadata, arrays, dates, processed_dir
        ),
    }

    forecast_layer_terms: dict[str, dict[str, np.ndarray]] = {
        component: {
            layer: np.zeros((n_draws, len(dates)), dtype=float)
            for layer in LAYERS
        }
        for component in COMPONENTS
    }
    for component in ("gas", "electricity", "liquid_fuels"):
        gross = gross_energy_terms[:, :, component_positions[component]]
        blocks = _layer_terms_from_shares(gross, forecast_shares[component])
        for layer in LAYERS:
            forecast_layer_terms[component][layer] = blocks[layer]

    for component in ("heat_energy", "solid_fuels"):
        forecast_layer_terms[component]["unsplit"] = gross_energy_terms[
            :, :, component_positions[component]
        ]

    # Car fuels: split its first-stage transport terms, then map those terms into
    # the second-stage Energy term with the exact saved car-fuels scaling.
    transport_terms = np.asarray(transport_replay["term_paths"], dtype=float)
    transport_names = list(TRANSPORT_COMPONENTS)
    lower_layers = {
        layer: np.zeros((n_draws, len(dates)), dtype=float) for layer in LAYERS
    }
    for source_name, share_key in (
        ("hicp_petrol", "petrol"),
        ("hicp_diesel", "diesel"),
    ):
        gross = transport_terms[:, :, transport_names.index(source_name)]
        blocks = _layer_terms_from_shares(gross, forecast_shares[share_key])
        for layer in LAYERS:
            lower_layers[layer] += blocks[layer]
    lower_layers["unsplit"] += transport_terms[
        :, :, transport_names.index("hicp_other_transport_fuels")
    ]
    gross_car_energy_term = gross_energy_terms[:, :, component_positions["car_fuels"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        car_scale = gross_car_energy_term / car_saved
    if not np.isfinite(car_scale).all():
        raise TaxDecompositionError("Car-fuels Energy-term scaling contains non-finite values.")
    for layer in LAYERS:
        forecast_layer_terms["car_fuels"][layer] = lower_layers[layer] * car_scale

    forecast_term_error = 0.0
    for component in COMPONENTS:
        replay_term = sum(forecast_layer_terms[component][layer] for layer in LAYERS)
        gross_term = gross_energy_terms[:, :, component_positions[component]]
        forecast_term_error = max(
            forecast_term_error,
            float(np.nanmax(np.abs(replay_term - gross_term))),
        )
    if forecast_term_error > 1e-8:
        raise TaxDecompositionError(
            f"Forecast layer terms fail gross-term replay: {forecast_term_error:.3e}."
        )

    # Convert layer terms to exact YoY contributions.  A layer split is shown
    # only when the tax identity is identified on BOTH sides of the y/y
    # comparison (t and t-12).  Coverage transitions are collapsed to Unsplit
    # so they cannot masquerade as large market/VAT/excise shocks.
    rows: list[dict[str, Any]] = []
    historical_existing = yoy_contributions_from_terms(
        energy_history["index"], energy_history["terms"]
    )
    historical_component_error = 0.0
    for component in COMPONENTS:
        expected = pd.to_numeric(historical_existing[component], errors="coerce")
        raw = {
            layer: _history_contribution(
                history_layer_terms[component][layer], energy_history["index"]
            )
            for layer in LAYERS
        }
        comparable = _historical_comparable_yoy_mask(
            history_layer_terms[component]
        )
        contributions = _collapse_historical_incomparable(
            raw, expected, comparable
        )
        component_sum = pd.Series(0.0, index=expected.index, dtype=float)
        for layer in LAYERS:
            contribution = contributions[layer]
            component_sum = component_sum.add(contribution, fill_value=0.0)
            rows.extend(
                _frame_rows_from_history(
                    contribution,
                    component=component,
                    layer=layer,
                )
            )
        diff = (component_sum - expected).abs().dropna()
        if len(diff):
            historical_component_error = max(
                historical_component_error, float(diff.max())
            )
    if historical_component_error > 1e-8:
        raise TaxDecompositionError(
            "Historical tax-layer contributions do not replay existing component "
            f"contributions: {historical_component_error:.3e}."
        )

    saved_component_contrib = np.asarray(arrays["baseline_contribution_paths"], dtype=float)
    saved_yoy = np.asarray(arrays["baseline_yoy_paths"], dtype=float)
    forecast_component_error = 0.0
    all_layer_sum = np.zeros_like(saved_yoy, dtype=float)
    for component in COMPONENTS:
        expected = saved_component_contrib[:, :, component_positions[component]]
        raw = {
            layer: _contribution_paths(
                forecast_layer_terms[component][layer],
                history_layer_terms[component][layer],
                saved_levels,
                energy_history["index"],
                dates,
            )
            for layer in LAYERS
        }
        comparable = _forecast_comparable_yoy_mask(
            history_layer_terms[component],
            forecast_layer_terms[component],
            dates,
        )
        contributions = _collapse_forecast_incomparable(
            raw, expected, comparable
        )
        comp_sum = np.zeros_like(saved_yoy, dtype=float)
        for layer in LAYERS:
            contribution_paths = contributions[layer]
            comp_sum += contribution_paths
            all_layer_sum += contribution_paths
            rows.extend(
                _frame_rows_from_paths(
                    contribution_paths,
                    dates,
                    component=component,
                    layer=layer,
                    segment="model_path",
                )
            )
        mask = np.isfinite(comp_sum) & np.isfinite(expected)
        if mask.any():
            forecast_component_error = max(
                forecast_component_error,
                float(np.max(np.abs(comp_sum[mask] - expected[mask]))),
            )
    if forecast_component_error > 1e-8:
        raise TaxDecompositionError(
            "Forecast tax-layer contributions do not replay saved component "
            f"contributions: {forecast_component_error:.3e}."
        )
    aggregate_mask = np.isfinite(all_layer_sum) & np.isfinite(saved_yoy)
    aggregate_additivity_error = (
        float(np.max(np.abs(all_layer_sum[aggregate_mask] - saved_yoy[aggregate_mask])))
        if aggregate_mask.any()
        else float("nan")
    )
    if np.isfinite(aggregate_additivity_error) and aggregate_additivity_error > 1e-8:
        raise TaxDecompositionError(
            "Tax-layer contributions do not add to saved HICP Energy YoY: "
            f"{aggregate_additivity_error:.3e}."
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise TaxDecompositionError("Tax decomposition produced no rows.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    first_model = dates.min()
    frame = frame.loc[
        (frame["segment"].astype(str) != "historical")
        | (frame["date"] < first_model)
    ].copy()
    frame = frame.sort_values(["date", "component", "layer"]).reset_index(drop=True)

    meta = {
        "contract_version": TAX_DECOMPOSITION_VERSION,
        "vintage": vintage,
        "aggregate_run_id": str(metadata.get("aggregate_run_id") or directory.name),
        "forecast_name": str(metadata.get("forecast_name") or "unconditional"),
        "n_draws": int(n_draws),
        "forecast_start": pd.Timestamp(dates.min()).isoformat(),
        "forecast_end": pd.Timestamp(dates.max()).isoformat(),
        "level_replay_max_abs_error": level_replay_error,
        "car_fuels_replay_max_abs_error": car_replay_error,
        "historical_term_replay_max_abs_error": hist_term_error,
        "forecast_term_replay_max_abs_error": forecast_term_error,
        "historical_component_contribution_replay_max_abs_error": historical_component_error,
        "forecast_component_contribution_replay_max_abs_error": forecast_component_error,
        "aggregate_additivity_max_abs_error": aggregate_additivity_error,
        "vat_interaction_policy": "VAT is final legal layer; VAT-on-excise belongs to VAT",
        "unidentified_tax_policy": (
            "Heat energy, solid fuels and Other transport fuels remain Unsplit; "
            "missing tax-share cells also remain Unsplit rather than being treated as zero tax"
        ),
        "weekly_share_policy": (
            "WOB pre-tax and excise-inclusive COMPLETE-month shares applied to the exact saved HICP terms; "
            "aggregate months absent from the complete-month WOB calendar remain Unsplit"
        ),
        "yoy_split_policy": (
            "Pre-tax / Excise / VAT are shown only when the tax split is identified at both t and t-12; "
            "otherwise the exact existing component contribution is assigned to Unsplit"
        ),
        "no_reestimation": True,
    }
    return frame, meta


def _selected_rows(frame: pd.DataFrame, component: str | None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if component in {None, "all", "hicp_energy"}:
        return frame.copy()
    return frame.loc[frame["component"].astype(str).eq(str(component))].copy()


def tax_contribution_figure(
    frame: pd.DataFrame,
    *,
    component: str = "all",
    forecast_origin: Any = None,
    display_mode: str = "bars",
    uirevision: str = "tax-contribution",
) -> go.Figure:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Tax contribution decomposition unavailable",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(template="plotly_white", height=560)
        return fig

    rows = _selected_rows(frame, component)
    title_subject = (
        "HICP Energy" if component in {None, "all", "hicp_energy"}
        else COMPONENT_LABELS.get(str(component), str(component))
    )
    central = (
        rows.groupby(["date", "layer"], as_index=False)["value"]
        .sum(min_count=1)
        .sort_values(["date", "layer"])
    )
    total = (
        central.groupby("date", as_index=False)["value"]
        .sum(min_count=1)
        .sort_values("date")
    )

    fig = go.Figure()
    for layer in LAYERS:
        block = central.loc[central["layer"].astype(str).eq(layer)].sort_values("date")
        if block.empty:
            continue
        if display_mode == "lines":
            fig.add_trace(
                go.Scatter(
                    x=block["date"],
                    y=block["value"],
                    mode="lines",
                    name=LAYER_LABELS[layer],
                    line={"width": 2.0, "color": LAYER_COLOURS[layer]},
                    hovertemplate=f"%{{y:+.2f}} pp<extra>{LAYER_LABELS[layer]}</extra>",
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=block["date"],
                    y=block["value"],
                    name=LAYER_LABELS[layer],
                    marker={"color": LAYER_COLOURS[layer]},
                    hovertemplate=f"%{{y:+.2f}} pp<extra>{LAYER_LABELS[layer]}</extra>",
                )
            )
    total_line_name = (
        "HICP Energy inflation"
        if component in {None, "all", "hicp_energy"}
        else f"{title_subject} total contribution"
    )
    fig.add_trace(
        go.Scatter(
            x=total["date"],
            y=total["value"],
            mode="lines",
            name=total_line_name,
            line={"color": "#111827", "width": 2.4, "dash": "dash"},
            hovertemplate=f"%{{y:+.2f}} pp<extra>{total_line_name}</extra>",
        )
    )
    if forecast_origin is not None and not pd.isna(forecast_origin):
        date = pd.Timestamp(forecast_origin)
        fig.add_shape(
            type="line", x0=date, x1=date, y0=0, y1=1,
            xref="x", yref="paper",
            line={"color": "#9CA3AF", "width": 1, "dash": "dot"},
        )
        fig.add_annotation(
            x=date, y=0.98, xref="x", yref="paper",
            text="Forecast origin", showarrow=False,
            xanchor="left", yanchor="top",
            font={"size": 10, "color": "#6B7280"},
            bgcolor="rgba(255,255,255,0.72)",
        )
    if component in {None, "all", "hicp_energy"}:
        main_title = "Drivers of HICP Energy annual inflation — tax-layer decomposition"
    else:
        main_title = (
            f"Drivers of {title_subject} contribution to HICP Energy annual inflation "
            "— tax-layer decomposition"
        )
    subtitle = (
        "Contributions in percentage points to year-on-year HICP Energy inflation. "
        "A layer is split only when its tax bridge is identified in both t and t-12."
    )
    fig.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"{main_title}<br>"
                f"<span style='font-size:11px;color:#6B7280'>{subtitle}</span>"
            ),
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 17},
        },
        barmode="relative",
        height=620,
        margin={"l": 58, "r": 28, "t": 108, "b": 96},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10},
        },
        yaxis_title="percentage points",
        hovermode="x unified",
        dragmode="pan",
        uirevision=str(uirevision),
    )
    return fig


def horizon_dates(
    frame: pd.DataFrame,
    forecast_origin: Any,
) -> dict[str, pd.Timestamp]:
    if frame.empty:
        return {}
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"].dropna().unique())).sort_values()
    if len(dates) == 0:
        return {}
    origin = pd.Timestamp(forecast_origin) if forecast_origin is not None else None
    out: dict[str, pd.Timestamp] = {}
    if origin is not None:
        hist = dates[dates < origin]
        if len(hist):
            out["latest"] = pd.Timestamp(hist[-1])
        model_dates = set(pd.Timestamp(x) for x in dates[dates >= origin])
        for n in (1, 2, 3, 6, 12):
            target = origin + pd.DateOffset(months=n - 1)
            target = target.to_period("M").to_timestamp(how="start")
            if target in model_dates:
                out[f"m{n}"] = target
    else:
        out["latest"] = pd.Timestamp(dates[-1])
    return out


def horizon_options(frame: pd.DataFrame, forecast_origin: Any) -> list[dict[str, str]]:
    resolved = horizon_dates(frame, forecast_origin)
    labels = {
        "latest": "Latest observed",
        "m1": "M+1",
        "m2": "M+2",
        "m3": "M+3",
        "m6": "M+6",
        "m12": "M+12",
    }
    return [
        {
            "label": f"{labels[key]} · {pd.Timestamp(resolved[key]).strftime('%b %Y')}",
            "value": key,
        }
        for key in ("latest", "m1", "m2", "m3", "m6", "m12")
        if key in resolved
    ]


def tax_matrix_period_title(
    frame: pd.DataFrame,
    *,
    horizon: str,
    forecast_origin: Any,
) -> str:
    resolved = horizon_dates(frame, forecast_origin)
    date = resolved.get(str(horizon))
    if date is None:
        return "Tax-layer contribution — selected period unavailable"
    labels = {
        "latest": "Observed",
        "m1": "Forecast M+1",
        "m2": "Forecast M+2",
        "m3": "Forecast M+3",
        "m6": "Forecast M+6",
        "m12": "Forecast M+12",
    }
    month = pd.Timestamp(date).strftime("%B %Y")
    return f"Tax-layer contribution — {month} · {labels.get(str(horizon), str(horizon))}"


def tax_matrix_records(
    frame: pd.DataFrame,
    *,
    horizon: str,
    forecast_origin: Any,
) -> tuple[list[dict[str, Any]], str]:
    resolved = horizon_dates(frame, forecast_origin)
    date = resolved.get(str(horizon))
    if date is None:
        return [], "Selected horizon is not available in the frozen aggregate path."
    block = frame.loc[pd.to_datetime(frame["date"]).eq(pd.Timestamp(date))].copy()
    if block.empty:
        return [], f"No tax decomposition rows are available for {pd.Timestamp(date).date()}."

    pivot = block.pivot_table(
        index=["component", "component_label"],
        columns="layer",
        values="value",
        aggfunc="sum",
        fill_value=0.0,
    )
    for layer in LAYERS:
        if layer not in pivot.columns:
            pivot[layer] = 0.0
    pivot = pivot.reset_index()
    order = {name: idx for idx, name in enumerate(COMPONENTS)}
    pivot["_order"] = pivot["component"].map(order).fillna(999)
    pivot = pivot.sort_values("_order")

    records: list[dict[str, Any]] = []
    for _, row in pivot.iterrows():
        pre = float(row["pre_tax"])
        exc = float(row["excise"])
        vat = float(row["vat"])
        unsplit = float(row["unsplit"])
        records.append(
            {
                "Component": str(row["component_label"]),
                "Pre-tax / market": pre,
                "Excise": exc,
                "VAT": vat,
                "Unsplit": unsplit,
                "Tax total": exc + vat,
                "Total contribution": pre + exc + vat + unsplit,
            }
        )
    if records:
        numeric = [
            "Pre-tax / market", "Excise", "VAT", "Unsplit", "Tax total", "Total contribution"
        ]
        total = {"Component": "HICP Energy"}
        for key in numeric:
            total[key] = float(sum(float(item[key]) for item in records))
        records.append(total)
    label = {
        "latest": "Latest observed",
        "m1": "M+1",
        "m2": "M+2",
        "m3": "M+3",
        "m6": "M+6",
        "m12": "M+12",
    }.get(str(horizon), str(horizon).upper())
    month = pd.Timestamp(date).strftime("%B %Y")
    return records, (
        f"{label} · {month} · contributions in percentage points to "
        "year-on-year HICP Energy inflation"
    )


__all__ = [
    "TAX_DECOMPOSITION_VERSION",
    "LAYERS",
    "LAYER_LABELS",
    "COMPONENTS",
    "COMPONENT_LABELS",
    "TaxDecompositionError",
    "build_tax_contribution_decomposition",
    "tax_contribution_figure",
    "horizon_options",
    "tax_matrix_period_title",
    "tax_matrix_records",
]
