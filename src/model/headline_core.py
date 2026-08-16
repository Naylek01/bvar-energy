"""Derived Core HICP utilities for the locked Headline joint BVAR.

Core definition
---------------
``hicp_core`` is Eurostat ``TOT_X_NRG_FOOD``:
all-items HICP excluding energy, food, alcohol and tobacco.

The locked Headline BVAR is NOT re-estimated.  Core is a derived basket built
draw-by-draw from the saved NEIG + Services native index paths, using their
annual HICP item weights renormalised inside the Core basket and the same
previous-December chain-link convention used by Headline.

Official Eurostat Core is used for observed history and reconstruction
validation.  The Statistics API is queried with named filters so no positional
SDMX dimension order is assumed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

EUROSTAT_STATS_API = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
)
EUROSTAT_CORE_DATASET = "prc_hicp_minr"
EUROSTAT_CORE_CODE = "TOT_X_NRG_FOOD"
EUROSTAT_CORE_UNIT = "I25"
EUROSTAT_CORE_GEO = "EA"

CORE_SERIES = "hicp_core"
CORE_COMPONENTS = ("hicp_neig", "hicp_services")
CORE_LABEL = "Core HICP"
CORE_LONG_LABEL = "Core HICP excl. energy, food, alcohol & tobacco"

CORE_OFFICIAL_FILENAME = "headline_official_core.csv"
CORE_SOURCE_METADATA_FILENAME = "headline_official_core_metadata.json"
CORE_RECONSTRUCTION_FILENAME = "headline_core_reconstructed_history.csv"
CORE_DIAGNOSTICS_FILENAME = "headline_core_reconstruction_diagnostics.csv"


class HeadlineCoreError(RuntimeError):
    pass


def _monthly_index(index) -> pd.DatetimeIndex:
    return (
        pd.DatetimeIndex(index)
        .to_period("M")
        .to_timestamp(how="start")
    )


def _normalise_weight_index(weights: pd.DataFrame) -> pd.DataFrame:
    out = weights.copy()
    out.index = pd.Index(
        pd.to_numeric(out.index, errors="raise").astype(int),
        name=out.index.name,
    )
    return out.sort_index()


def _jsonstat_ordered_codes(dimension: Mapping) -> list[str]:
    """Return JSON-stat category codes in their declared order."""
    category = dict(dimension.get("category", {}) or {})
    index = category.get("index", {})
    if isinstance(index, Mapping):
        return [
            str(code)
            for code, _ in sorted(index.items(), key=lambda item: int(item[1]))
        ]
    if isinstance(index, Sequence) and not isinstance(index, (str, bytes)):
        return [str(code) for code in index]
    raise HeadlineCoreError(
        "Unsupported Eurostat JSON-stat category.index representation."
    )


def _jsonstat_dense_values(payload: Mapping) -> np.ndarray:
    """Expand JSON-stat dense or sparse values to the declared cube shape."""
    size = [int(x) for x in payload.get("size", [])]
    if not size:
        raise HeadlineCoreError("Eurostat JSON-stat response has no dimension sizes.")
    n = int(np.prod(size))
    raw = payload.get("value", [])

    dense = np.full(n, np.nan, dtype=float)
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            pos = int(key)
            if pos < 0 or pos >= n:
                raise HeadlineCoreError(
                    f"Eurostat JSON-stat sparse value position {pos} is out of bounds."
                )
            if value is not None:
                dense[pos] = float(value)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != n:
            raise HeadlineCoreError(
                f"Eurostat JSON-stat value length {len(raw)} does not match cube size {n}."
            )
        dense = np.asarray(
            [np.nan if value is None else float(value) for value in raw],
            dtype=float,
        )
    else:
        raise HeadlineCoreError("Unsupported Eurostat JSON-stat value representation.")

    return dense.reshape(tuple(size))


def _jsonstat_single_monthly_series(
    payload: Mapping,
    *,
    expected_dimensions: Mapping[str, str],
    name: str,
) -> tuple[pd.Series, str]:
    """Extract one monthly series without assuming any dimension order."""
    ids = [str(x).lower() for x in payload.get("id", [])]
    dimensions = dict(payload.get("dimension", {}) or {})
    dimensions_lower = {str(k).lower(): v for k, v in dimensions.items()}

    if not ids:
        raise HeadlineCoreError("Eurostat JSON-stat response has no dimension ids.")

    for dim, expected in expected_dimensions.items():
        dim = str(dim).lower()
        if dim not in ids or dim not in dimensions_lower:
            raise HeadlineCoreError(
                f"Eurostat Core response is missing dimension {dim!r}; found {ids}."
            )
        observed = _jsonstat_ordered_codes(dimensions_lower[dim])
        if observed != [str(expected)]:
            raise HeadlineCoreError(
                f"Eurostat Core response: expected {dim}={expected!r}, found {observed}."
            )

    time_dim = next((dim for dim in ids if dim in {"time", "time_period"}), None)
    if time_dim is None:
        raise HeadlineCoreError(
            f"Eurostat Core response has no time dimension; found {ids}."
        )

    time_codes = _jsonstat_ordered_codes(dimensions_lower[time_dim])
    cube = _jsonstat_dense_values(payload)

    slicer = []
    for axis, dim in enumerate(ids):
        if dim == time_dim:
            slicer.append(slice(None))
        else:
            if cube.shape[axis] != 1:
                raise HeadlineCoreError(
                    f"Expected one selected category for dimension {dim!r}, "
                    f"found size={cube.shape[axis]}."
                )
            slicer.append(0)

    values = np.asarray(cube[tuple(slicer)], dtype=float).reshape(-1)
    if len(values) != len(time_codes):
        raise HeadlineCoreError(
            "Eurostat Core time/value length mismatch: "
            f"{len(time_codes)} vs {len(values)}."
        )

    dates = pd.PeriodIndex(time_codes, freq="M").to_timestamp(how="start")
    series = pd.Series(values, index=dates, name=name).dropna().astype(float)
    series.index = pd.DatetimeIndex(series.index, name="date")

    if series.index.has_duplicates:
        duplicates = series.index[series.index.duplicated(keep=False)]
        raise HeadlineCoreError(
            "Eurostat Core response contains duplicate months: "
            + ", ".join(pd.DatetimeIndex(duplicates).strftime("%Y-%m"))
        )
    if series.empty:
        raise HeadlineCoreError("Eurostat Core API returned no usable observations.")
    if (series <= 0).any():
        first_bad = series.index[(series <= 0)][0]
        raise HeadlineCoreError(
            "Eurostat Core index must be positive; first invalid month is "
            f"{pd.Timestamp(first_bad).date()}."
        )

    coicop_dim = dimensions_lower.get("coicop18", {})
    labels = dict((coicop_dim.get("category", {}) or {}).get("label", {}) or {})
    label = str(labels.get(EUROSTAT_CORE_CODE, EUROSTAT_CORE_CODE))
    return series, label


def load_eurostat_core_index(
    start_period: str,
    *,
    timeout: int = 90,
) -> tuple[pd.Series, dict[str, Any]]:
    """Download official Core HICP through named Statistics-API filters."""
    base = f"{EUROSTAT_STATS_API}/{EUROSTAT_CORE_DATASET}"
    params = {
        "format": "JSON",
        "lang": "EN",
        "freq": "M",
        "unit": EUROSTAT_CORE_UNIT,
        "coicop18": EUROSTAT_CORE_CODE,
        "geo": EUROSTAT_CORE_GEO,
        "sinceTimePeriod": str(start_period),
    }
    request_url = f"{base}?{urlencode(params)}"
    request = Request(
        request_url,
        headers={"User-Agent": "bvar-energy-headline-core/1.0"},
    )
    try:
        with urlopen(request, timeout=int(timeout)) as response:
            payload = json.loads(response.read().decode("utf-8"))
            resolved_url = response.geturl()
    except Exception as exc:
        raise HeadlineCoreError(
            f"Eurostat Core API request failed: {exc}"
        ) from exc

    if isinstance(payload, Mapping) and payload.get("error"):
        raise HeadlineCoreError(f"Eurostat Core API error: {payload['error']}")

    core, label = _jsonstat_single_monthly_series(
        payload,
        expected_dimensions={
            "freq": "M",
            "unit": EUROSTAT_CORE_UNIT,
            "coicop18": EUROSTAT_CORE_CODE,
            "geo": EUROSTAT_CORE_GEO,
        },
        name=CORE_SERIES,
    )
    metadata = {
        "provider": "Eurostat",
        "api": "statistics/1.0",
        "dataset": EUROSTAT_CORE_DATASET,
        "code": EUROSTAT_CORE_CODE,
        "label": label,
        "unit": EUROSTAT_CORE_UNIT,
        "geo": EUROSTAT_CORE_GEO,
        "request_url": resolved_url,
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "first_date": core.index.min().date().isoformat(),
        "last_date": core.index.max().date().isoformat(),
        "n_observations": int(core.notna().sum()),
        "dimension_order_returned": list(payload.get("id", [])),
    }
    return core, metadata


def load_or_fetch_official_core(
    run_directory: str | Path,
    *,
    native_history: pd.DataFrame,
    observed_end: str | pd.Timestamp | None = None,
    refresh: bool = False,
    timeout: int = 90,
) -> tuple[pd.Series, dict[str, Any]]:
    """Load a run-local official Core snapshot, fetching Eurostat if absent."""
    run_dir = Path(run_directory)
    csv_path = run_dir / CORE_OFFICIAL_FILENAME
    meta_path = run_dir / CORE_SOURCE_METADATA_FILENAME

    if csv_path.is_file() and meta_path.is_file() and not refresh:
        series = pd.read_csv(
            csv_path, index_col=0, parse_dates=True
        ).iloc[:, 0].sort_index()
        series.index = _monthly_index(series.index)
        series.name = CORE_SERIES
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        native = native_history.loc[:, list(CORE_COMPONENTS)].copy().sort_index()
        native.index = _monthly_index(native.index)
        if native.empty:
            raise HeadlineCoreError("NEIG/Services history is empty.")
        start = (
            pd.Timestamp(native.index.min()) - pd.DateOffset(months=1)
        ).strftime("%Y-%m")
        series, metadata = load_eurostat_core_index(start, timeout=timeout)

        temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
        temporary_meta = meta_path.with_name(meta_path.name + ".tmp")
        series.to_frame(CORE_SERIES).to_csv(
            temporary_csv,
            index_label="date",
            date_format="%Y-%m-%d",
        )
        temporary_meta.write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )
        temporary_csv.replace(csv_path)
        temporary_meta.replace(meta_path)

    if observed_end is not None:
        end = pd.Timestamp(observed_end).to_period("M").to_timestamp(how="start")
        series = series.loc[series.index <= end].copy()

    if series.empty:
        raise HeadlineCoreError(
            "Official Eurostat Core snapshot has no observations in the run history."
        )
    return series.astype(float), metadata


def core_weight_table(annual_weights: pd.DataFrame) -> pd.DataFrame:
    """Return NEIG/Services weights renormalised to sum to one each year."""
    weights = _normalise_weight_index(annual_weights)
    missing = [name for name in CORE_COMPONENTS if name not in weights.columns]
    if missing:
        raise HeadlineCoreError(
            f"Core annual weights are missing columns: {missing}."
        )
    raw = weights.loc[:, list(CORE_COMPONENTS)].apply(
        pd.to_numeric, errors="coerce"
    )
    denominator = raw.sum(axis=1, min_count=len(CORE_COMPONENTS))
    out = raw.div(denominator, axis=0)
    invalid = denominator.notna() & (~np.isfinite(denominator) | (denominator <= 0))
    if invalid.any():
        years = list(map(int, denominator.index[invalid]))
        raise HeadlineCoreError(
            f"Invalid NEIG+Services weight denominator for years {years}."
        )
    complete = out.dropna(how="any")
    if not complete.empty:
        error = float(np.max(np.abs(complete.sum(axis=1).to_numpy(dtype=float) - 1.0)))
        if not np.isfinite(error) or error > 1e-12:
            raise HeadlineCoreError(
                f"Core weight renormalisation failed: max sum error {error:.3e}."
            )
    return out


def _complete_core_weight_row(
    annual_weights: pd.DataFrame,
    year: int,
    *,
    carry_forward_future: bool,
) -> tuple[pd.Series, int, bool]:
    weights = core_weight_table(annual_weights)
    year = int(year)
    if year in weights.index and weights.loc[year].notna().all():
        return weights.loc[year].astype(float), year, False

    complete = weights.dropna(how="any")
    if complete.empty:
        raise HeadlineCoreError(
            "No complete NEIG/Services annual Core weight vector exists."
        )
    latest = int(complete.index.max())
    if carry_forward_future and year > latest:
        return complete.loc[latest].astype(float), latest, True
    raise HeadlineCoreError(f"No complete Core weight vector for {year}.")


def resolve_core_reconstruction_start_year(
    core_levels: pd.DataFrame,
    annual_weights: pd.DataFrame,
    *,
    anchor_series: pd.Series | None = None,
) -> int:
    """First year admissible for the NEIG+Services Core basket."""
    native = core_levels.loc[:, list(CORE_COMPONENTS)].copy().sort_index()
    native.index = _monthly_index(native.index)
    weights = core_weight_table(annual_weights)
    complete_years = list(map(int, weights.dropna(how="any").index))
    if not complete_years:
        raise HeadlineCoreError(
            "No complete NEIG/Services annual Core weight vector exists."
        )

    anchor = None
    if anchor_series is not None:
        anchor = pd.to_numeric(anchor_series, errors="coerce").dropna().sort_index()
        anchor.index = _monthly_index(anchor.index)

    rejected = []
    for year in sorted(complete_years):
        dec_prev = pd.Timestamp(year=year - 1, month=12, day=1)
        if dec_prev not in native.index:
            rejected.append(f"{year}: component December outside Core panel")
            continue
        if native.loc[dec_prev, list(CORE_COMPONENTS)].isna().any():
            rejected.append(f"{year}: missing Core component December")
            continue
        if anchor is not None and (
            dec_prev not in anchor.index or pd.isna(anchor.loc[dec_prev])
        ):
            rejected.append(f"{year}: official Core anchor missing")
            continue
        return int(year)

    detail = "; ".join(rejected[:6])
    if len(rejected) > 6:
        detail += "; ..."
    raise HeadlineCoreError(
        "No admissible Core reconstruction start year. " + detail
    )


def reconstruct_core_history(
    core_levels: pd.DataFrame,
    annual_weights: pd.DataFrame,
    *,
    start_year: int | None = None,
    end_date: str | pd.Timestamp | None = None,
    anchor_series: pd.Series | None = None,
    anchor_value: float = 100.0,
) -> pd.Series:
    """Previous-December chain-link of NEIG + Services."""
    native = core_levels.loc[:, list(CORE_COMPONENTS)].copy().sort_index()
    native.index = _monthly_index(native.index)
    weights = core_weight_table(annual_weights)

    anchor = None
    if anchor_series is not None:
        anchor = pd.to_numeric(anchor_series, errors="coerce").dropna().sort_index()
        anchor.index = _monthly_index(anchor.index)

    if start_year is None:
        start_year = resolve_core_reconstruction_start_year(
            native, annual_weights, anchor_series=anchor
        )
    start_year = int(start_year)

    last_date = pd.Timestamp(native.index.max())
    if end_date is not None:
        last_date = min(
            last_date,
            pd.Timestamp(end_date).to_period("M").to_timestamp(how="start"),
        )
    if anchor is not None:
        last_date = min(last_date, pd.Timestamp(anchor.index.max()))

    output = pd.Series(
        index=native.index[
            (native.index.year >= start_year) & (native.index <= last_date)
        ],
        dtype=float,
        name="hicp_core_reconstructed",
    )

    previous_december = None
    for year in range(start_year, int(last_date.year) + 1):
        if year not in weights.index or weights.loc[year].isna().any():
            raise HeadlineCoreError(
                f"Core reconstruction is missing annual weights for {year}."
            )
        weight_row = weights.loc[year].astype(float)
        dec_prev = pd.Timestamp(year=year - 1, month=12, day=1)
        if dec_prev not in native.index:
            raise HeadlineCoreError(
                f"Core reconstruction needs component December {dec_prev.date()}."
            )
        base = native.loc[dec_prev, list(CORE_COMPONENTS)].astype(float)
        if base.isna().any() or np.any(base.to_numpy(dtype=float) <= 0):
            raise HeadlineCoreError(
                f"Invalid Core component December anchor at {dec_prev.date()}."
            )

        if previous_december is None:
            if anchor is not None:
                if dec_prev not in anchor.index or pd.isna(anchor.loc[dec_prev]):
                    raise HeadlineCoreError(
                        f"Official Core anchor missing at {dec_prev.date()}."
                    )
                year_anchor = float(anchor.loc[dec_prev])
            else:
                year_anchor = float(anchor_value)
                if not np.isfinite(year_anchor) or year_anchor <= 0:
                    raise HeadlineCoreError(
                        "Synthetic Core anchor_value must be positive and finite."
                    )
        else:
            year_anchor = float(previous_december)

        dates = native.index[
            (native.index.year == year) & (native.index <= last_date)
        ]
        for date in dates:
            current = native.loc[date, list(CORE_COMPONENTS)].astype(float)
            if current.isna().any():
                continue
            ratios = current / base
            output.loc[date] = float(
                year_anchor
                * np.sum(
                    weight_row.to_numpy(dtype=float)
                    * ratios.to_numpy(dtype=float)
                )
            )

        december = pd.Timestamp(year=year, month=12, day=1)
        if december in output.index and pd.notna(output.loc[december]):
            previous_december = float(output.loc[december])
        elif year < last_date.year:
            raise HeadlineCoreError(
                f"Cannot chain Core into {year + 1}: reconstructed December {year} is missing."
            )

    output.index = pd.DatetimeIndex(output.index, name="date")
    return output.sort_index()


def core_reconstruction_diagnostics(
    reconstructed: pd.Series,
    official_core: pd.Series,
) -> pd.Series:
    """Level and YoY errors against official Eurostat Core."""
    official = pd.to_numeric(official_core, errors="coerce").dropna().sort_index()
    official.index = _monthly_index(official.index)
    reconstructed = pd.to_numeric(reconstructed, errors="coerce").dropna().sort_index()
    reconstructed.index = _monthly_index(reconstructed.index)

    joined = pd.concat(
        [reconstructed.rename("reconstructed"), official.rename("official")],
        axis=1,
    ).dropna()
    if joined.empty:
        raise HeadlineCoreError(
            "No overlap between reconstructed and official Core HICP."
        )

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
            "yoy_mae_pp": (
                float(yoy_error.abs().mean()) if len(yoy_error) else np.nan
            ),
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


def historical_core_component_levels(
    core_levels: pd.DataFrame,
    annual_weights: pd.DataFrame,
    official_core: pd.Series,
) -> pd.DataFrame:
    """Official-rescaled component levels that add exactly to official Core."""
    native = core_levels.loc[:, list(CORE_COMPONENTS)].copy().sort_index()
    native.index = _monthly_index(native.index)
    official = pd.to_numeric(official_core, errors="coerce").dropna().sort_index()
    official.index = _monthly_index(official.index)

    reconstructed = reconstruct_core_history(
        native,
        annual_weights,
        anchor_series=official,
    )
    if reconstructed.empty:
        raise HeadlineCoreError("Historical Core reconstruction is empty.")

    weights = core_weight_table(annual_weights)
    out = pd.DataFrame(
        index=reconstructed.index,
        columns=list(CORE_COMPONENTS),
        dtype=float,
    )
    previous_december = None
    first_year = int(reconstructed.index.min().year)
    last_year = int(reconstructed.index.max().year)

    for year in range(first_year, last_year + 1):
        weight_row = weights.loc[year].astype(float)
        dec_prev = pd.Timestamp(year=year - 1, month=12, day=1)
        base = native.loc[dec_prev, list(CORE_COMPONENTS)].astype(float)
        if previous_december is None:
            anchor = float(official.loc[dec_prev])
        else:
            anchor = float(previous_december)

        dates = out.index[out.index.year == year]
        for date in dates:
            ratios = native.loc[date, list(CORE_COMPONENTS)].astype(float) / base
            out.loc[date, list(CORE_COMPONENTS)] = (
                anchor
                * weight_row.to_numpy(dtype=float)
                * ratios.to_numpy(dtype=float)
            )
        december = pd.Timestamp(year=year, month=12, day=1)
        if december in reconstructed.index and pd.notna(reconstructed.loc[december]):
            previous_december = float(reconstructed.loc[december])

    reconstructed_sum = out.sum(axis=1)
    add_error = float(
        np.nanmax(
            np.abs(
                reconstructed_sum.to_numpy(dtype=float)
                - reconstructed.reindex(out.index).to_numpy(dtype=float)
            )
        )
    )
    if not np.isfinite(add_error) or add_error > 1e-10:
        raise HeadlineCoreError(
            f"Historical Core component reconstruction additivity failed: {add_error:.3e}."
        )

    shares = out.div(reconstructed_sum, axis=0)
    scaled = shares.mul(official.reindex(out.index), axis=0)
    comparable = official.reindex(scaled.index).notna()
    if comparable.any():
        scaled_error = float(
            np.nanmax(
                np.abs(
                    scaled.loc[comparable].sum(axis=1).to_numpy(dtype=float)
                    - official.reindex(scaled.index)[comparable].to_numpy(dtype=float)
                )
            )
        )
        if not np.isfinite(scaled_error) or scaled_error > 1e-10:
            raise HeadlineCoreError(
                f"Official Core rescaling additivity failed: {scaled_error:.3e}."
            )
    return scaled


def historical_core_yoy_contributions(
    core_levels: pd.DataFrame,
    annual_weights: pd.DataFrame,
    official_core: pd.Series,
) -> pd.DataFrame:
    """Exact observed pp contributions of NEIG and Services to official Core YoY."""
    official = pd.to_numeric(official_core, errors="coerce").dropna().sort_index()
    official.index = _monthly_index(official.index)
    components = historical_core_component_levels(
        core_levels, annual_weights, official
    )

    lagged_components = components.copy()
    lagged_components.index = lagged_components.index + pd.DateOffset(months=12)
    lagged_components = lagged_components.reindex(components.index)

    lagged_official = official.copy()
    lagged_official.index = lagged_official.index + pd.DateOffset(months=12)
    lagged_official = lagged_official.reindex(components.index)

    contribution = 100.0 * (
        components - lagged_components
    ).div(lagged_official, axis=0)
    contribution = contribution.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if contribution.empty:
        raise HeadlineCoreError(
            "No exact historical Core YoY contribution dates are available."
        )

    target = 100.0 * (
        official.reindex(contribution.index)
        / lagged_official.reindex(contribution.index)
        - 1.0
    )
    error = float(
        np.max(
            np.abs(
                contribution.sum(axis=1).to_numpy(dtype=float)
                - target.to_numpy(dtype=float)
            )
        )
    )
    if not np.isfinite(error) or error > 1e-10:
        raise HeadlineCoreError(
            f"Historical Core YoY contribution additivity failed: {error:.3e} pp."
        )
    return contribution


def aggregate_core_draws(
    native_level_paths: np.ndarray,
    path_dates: pd.DatetimeIndex,
    *,
    native_variables: Sequence[str],
    native_history: pd.DataFrame,
    annual_weights: pd.DataFrame,
    official_core: pd.Series,
    carry_forward_future_weights: bool = True,
) -> dict[str, Any]:
    """Aggregate saved NEIG + Services paths draw-by-draw into exact Core paths."""
    paths = np.asarray(native_level_paths, dtype=float)
    dates = _monthly_index(path_dates)
    variables = list(map(str, native_variables))
    if paths.ndim != 3 or paths.shape[1] != len(dates):
        raise HeadlineCoreError(
            f"Expected native paths (draws, dates, variables), got {paths.shape}."
        )
    try:
        positions = [variables.index(name) for name in CORE_COMPONENTS]
    except ValueError as exc:
        raise HeadlineCoreError(
            f"Native paths must contain {CORE_COMPONENTS}; found {variables}."
        ) from exc

    core_native = paths[:, :, positions]
    if not np.all(np.isfinite(core_native)):
        raise HeadlineCoreError("Saved NEIG/Services level paths contain non-finite values.")
    if np.any(core_native <= 0):
        raise HeadlineCoreError("Saved NEIG/Services level paths must be positive.")

    history = native_history.loc[:, list(CORE_COMPONENTS)].copy().sort_index()
    history.index = _monthly_index(history.index)
    official = pd.to_numeric(official_core, errors="coerce").dropna().sort_index()
    official.index = _monthly_index(official.index)

    hist_components = historical_core_component_levels(
        history, annual_weights, official
    )
    reconstructed_history = reconstruct_core_history(
        history, annual_weights, anchor_series=official
    )

    n_draws, n_dates, _ = core_native.shape
    core_level = np.full((n_draws, n_dates), np.nan, dtype=float)
    component_level = np.full((n_draws, n_dates, len(CORE_COMPONENTS)), np.nan)
    weight_year_used = np.full(n_dates, -1, dtype=int)
    weight_carried = np.zeros(n_dates, dtype=bool)
    pos = {pd.Timestamp(date): i for i, date in enumerate(dates)}

    for year in sorted(set(map(int, dates.year))):
        weight_row, used_year, carried = _complete_core_weight_row(
            annual_weights,
            year,
            carry_forward_future=carry_forward_future_weights,
        )
        dec_prev = pd.Timestamp(year=year - 1, month=12, day=1)
        if dec_prev in pos:
            base_pos = pos[dec_prev]
            base_components = core_native[:, base_pos, :]
            anchor = core_level[:, base_pos]
            if not np.all(np.isfinite(anchor)):
                raise HeadlineCoreError(
                    f"Core path December {dec_prev.date()} was not constructed."
                )
        else:
            if dec_prev not in history.index:
                raise HeadlineCoreError(
                    f"Core path needs historical component December {dec_prev.date()}."
                )
            base_row = history.loc[dec_prev, list(CORE_COMPONENTS)].to_numpy(dtype=float)
            if not np.all(np.isfinite(base_row)) or np.any(base_row <= 0):
                raise HeadlineCoreError(
                    f"Invalid historical Core component December {dec_prev.date()}."
                )
            base_components = np.broadcast_to(
                base_row, (n_draws, len(CORE_COMPONENTS))
            )
            if dec_prev in official.index and pd.notna(official.loc[dec_prev]):
                anchor_value = float(official.loc[dec_prev])
            elif dec_prev in reconstructed_history.index:
                anchor_value = float(reconstructed_history.loc[dec_prev])
            else:
                raise HeadlineCoreError(
                    f"No Core aggregate anchor is available at {dec_prev.date()}."
                )
            anchor = np.full(n_draws, anchor_value, dtype=float)

        year_positions = np.flatnonzero(dates.year == year)
        for t in year_positions:
            ratios = core_native[:, t, :] / base_components
            contrib = (
                anchor[:, None]
                * weight_row.to_numpy(dtype=float)[None, :]
                * ratios
            )
            component_level[:, t, :] = contrib
            core_level[:, t] = np.sum(contrib, axis=1)
            weight_year_used[t] = int(used_year)
            weight_carried[t] = bool(carried)

    level_error = float(
        np.nanmax(
            np.abs(
                np.nansum(component_level, axis=2) - core_level
            )
        )
    )
    if not np.isfinite(level_error) or level_error > 1e-10:
        raise HeadlineCoreError(
            f"Draw-wise Core level additivity failed: {level_error:.3e}."
        )

    core_yoy = np.full_like(core_level, np.nan)
    yoy_contribution = np.full_like(component_level, np.nan)

    for t, date in enumerate(dates):
        lag_date = pd.Timestamp(date) - pd.DateOffset(months=12)
        if lag_date in pos:
            lag_pos = pos[lag_date]
            denominator = core_level[:, lag_pos]
            lag_components = component_level[:, lag_pos, :]
        else:
            if lag_date not in official.index or lag_date not in hist_components.index:
                continue
            denominator = np.full(
                n_draws, float(official.loc[lag_date]), dtype=float
            )
            lag_row = hist_components.loc[
                lag_date, list(CORE_COMPONENTS)
            ].to_numpy(dtype=float)
            lag_components = np.broadcast_to(
                lag_row, (n_draws, len(CORE_COMPONENTS))
            )

        core_yoy[:, t] = 100.0 * (core_level[:, t] / denominator - 1.0)
        yoy_contribution[:, t, :] = 100.0 * (
            component_level[:, t, :] - lag_components
        ) / denominator[:, None]

    summed = np.nansum(yoy_contribution, axis=2)
    complete = np.isfinite(yoy_contribution).all(axis=2) & np.isfinite(core_yoy)
    yoy_error = (
        float(np.max(np.abs(summed[complete] - core_yoy[complete])))
        if complete.any()
        else float("nan")
    )
    if complete.any() and (
        not np.isfinite(yoy_error) or yoy_error > 1e-9
    ):
        raise HeadlineCoreError(
            f"Draw-wise Core YoY contribution additivity failed: {yoy_error:.3e} pp."
        )

    return {
        "core_level_paths": core_level,
        "core_yoy_paths": core_yoy,
        "core_component_index_contributions": component_level,
        "core_yoy_contribution_paths": yoy_contribution,
        "core_weight_year_used": weight_year_used,
        "core_weight_carried_forward": weight_carried,
        "core_drawwise_level_additivity_max_abs_error": level_error,
        "core_yoy_contribution_additivity_max_abs_error": yoy_error,
        "core_aggregation_method": (
            "annual previous-December chain-link; NEIG+Services weights "
            "renormalised within Core; draw-wise native-level aggregation"
        ),
    }


def persist_core_reconstruction_diagnostics(
    run_directory: str | Path,
    *,
    reconstructed: pd.Series,
    diagnostics: pd.Series,
) -> tuple[Path, Path]:
    run_dir = Path(run_directory)
    recon_path = run_dir / CORE_RECONSTRUCTION_FILENAME
    diag_path = run_dir / CORE_DIAGNOSTICS_FILENAME

    reconstructed.to_frame("hicp_core_reconstructed").to_csv(
        recon_path,
        index_label="date",
        date_format="%Y-%m-%d",
    )
    diagnostics.to_frame("value").to_csv(
        diag_path,
        index_label="metric",
    )
    return recon_path, diag_path


__all__ = [
    "HeadlineCoreError",
    "EUROSTAT_CORE_DATASET",
    "EUROSTAT_CORE_CODE",
    "EUROSTAT_CORE_UNIT",
    "EUROSTAT_CORE_GEO",
    "CORE_SERIES",
    "CORE_COMPONENTS",
    "CORE_LABEL",
    "CORE_LONG_LABEL",
    "CORE_OFFICIAL_FILENAME",
    "CORE_SOURCE_METADATA_FILENAME",
    "core_weight_table",
    "resolve_core_reconstruction_start_year",
    "reconstruct_core_history",
    "core_reconstruction_diagnostics",
    "historical_core_component_levels",
    "historical_core_yoy_contributions",
    "aggregate_core_draws",
    "load_eurostat_core_index",
    "load_or_fetch_official_core",
    "persist_core_reconstruction_diagnostics",
]
