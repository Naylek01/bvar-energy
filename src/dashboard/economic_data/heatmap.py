"""Economic Conditions Heatmap engine.

Adapted from the standalone Economic Conditions Heatmap supplied by the user.
This module contains no Dash app/server and performs no Haver import at module
import time. Haver is imported lazily only when an explicit dataset update is
requested by the Economic Data feature.
"""
from __future__ import annotations

import importlib
import threading
import uuid
from collections import OrderedDict
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html

SERIES_CONFIG = {
    "HICP":         {"ticker": "P025HICP@EUDATA",  "transform": "yoy_pct", "invert": False},
    "PMI Output":   {"ticker": "S023TG@MKTPMI",    "transform": "level",   "invert": False},
    "IP":           {"ticker": "S025D@G10",        "transform": "yoy_pct", "invert": False},
    "URX":          {"ticker": "S025R@EUDATA",     "transform": "level",   "invert": True},
    "Vacancy rate": {"ticker": "S0259XAX@EUDATA", "transform": "level", "invert": False,"frequency": "quarterly"},
    "PMI Delivery": {"ticker": "S023MD@MKTPMI",    "transform": "level",   "invert": True},
    "GSCPI":        {"ticker": "W1NGSCPI@TRANSPRT", "transform": "level",   "invert": False, "standardise": False},
    "NEER":         {"ticker": "FXTNBEU@USECON",   "transform": "yoy_pct", "invert": True},
    "CCI":          {"ticker": "E025C@EUDATA",     "transform": "diff12",  "invert": False},
}

VALID_TRANSFORMS = {"yoy_pct", "diff12", "level"}
VALID_FREQUENCIES = {"monthly", "quarterly"}
SERIES_NAMES = list(SERIES_CONFIG.keys())
DEFAULT_GRAPH_START_DATE = "2006-01-01"
DEFAULT_END_DATE = date.today().isoformat()
DEFAULT_END_MONTH = (
    pd.Timestamp.today().to_period("M").to_timestamp().strftime("%Y-%m-%d")
)
DEFAULT_STD_START_DATE = "2006-01-01"
DEFAULT_STD_END_DATE = DEFAULT_END_MONTH
MONTH_OPTIONS_START = "1990-01-01"
_MONTH_RANGE = pd.date_range(
    MONTH_OPTIONS_START,
    pd.Timestamp.today().to_period("M").to_timestamp(),
    freq="MS",
)
MONTH_OPTIONS = [
    {"label": month.strftime("%Y-%m"), "value": month.strftime("%Y-%m-%d")}
    for month in reversed(_MONTH_RANGE)
]
PARULA_COLOURSCALE = [
    [0.0, "rgb(53,42,135)"],
    [0.1, "rgb(30,86,197)"],
    [0.2, "rgb(6,123,209)"],
    [0.3, "rgb(17,146,197)"],
    [0.4, "rgb(21,162,178)"],
    [0.5, "rgb(43,176,153)"],
    [0.6, "rgb(94,190,125)"],
    [0.7, "rgb(155,196,94)"],
    [0.8, "rgb(213,193,66)"],
    [0.9, "rgb(250,204,42)"],
    [1.0, "rgb(249,251,21)"],
]
CUSTOM_COLOUR_SCALE = PARULA_COLOURSCALE
EXPORT_WIDTH_PX = 1050
EXPORT_SCALE = 2
MIN_RELIABLE_STD_OBS = 24

for _name, _cfg in SERIES_CONFIG.items():
    if not _cfg.get("ticker"):
        raise ValueError(f"Series '{_name}' is missing a Haver ticker.")
    if _cfg.get("transform") not in VALID_TRANSFORMS:
        raise ValueError(
            f"Series '{_name}' has unknown transform {_cfg.get('transform')!r}."
        )
    if not isinstance(_cfg.get("invert"), bool):
        raise ValueError(f"Series '{_name}' needs invert=True/False.")
    if _cfg.get("frequency", "monthly") not in VALID_FREQUENCIES:
        raise ValueError(f"Series '{_name}' has invalid frequency.")
    if not isinstance(_cfg.get("standardise", True), bool):
        raise ValueError(f"Series '{_name}' has invalid standardise flag.")


_HAVER_MODULE = None
_HAVER_LOCK = threading.RLock()


def _haver():
    """Lazy Haver import: the main dashboard never depends on Haver."""
    global _HAVER_MODULE
    with _HAVER_LOCK:
        if _HAVER_MODULE is None:
            try:
                module = importlib.import_module("Haver")
            except Exception as exc:
                raise RuntimeError(
                    "Haver is unavailable on this machine. Install/configure the "
                    "proprietary Haver Python package to use Economic Heatmap."
                ) from exc
            module.path("ini")
            _HAVER_MODULE = module
        return _HAVER_MODULE


def normalise_monthly_index(index: pd.Index) -> pd.DatetimeIndex:
    """
    Convert any Haver monthly index into a standard DatetimeIndex.

    Haver can return a PeriodIndex such as:

        2006-01
        2006-02

    Plotly, pandas date filtering and Dash work more consistently
    with timestamps such as:

        2006-01-01
        2006-02-01

    The data remain monthly. The conversion only changes the
    internal representation of the dates.
    """

    # Haver often returns a PeriodIndex for monthly series.
    if isinstance(index, pd.PeriodIndex):
        datetime_index = index.to_timestamp(how="start")

    # Otherwise, try to convert the existing index directly.
    else:
        datetime_index = pd.to_datetime(index)

    # Force every observation to the first day of its month.
    #
    # For example:
    # 2025-01-31 becomes 2025-01-01.
    datetime_index = (
        datetime_index
        .to_period("M")
        .to_timestamp(how="start")
    )

    return datetime_index

def normalise_quarterly_index(index: pd.Index) -> pd.DatetimeIndex:
    """
    Anchor each quarterly observation to the first day of its quarter.

    The quarterly analogue of normalise_monthly_index: Haver may return
    a quarterly PeriodIndex or dated observations; either way the result
    is a DatetimeIndex on quarter starts (Jan / Apr / Jul / Oct).
    """

    if isinstance(index, pd.PeriodIndex):
        datetime_index = index.to_timestamp(how="start")
    else:
        datetime_index = pd.to_datetime(index)

    return (
        datetime_index
        .to_period("Q")
        .to_timestamp(how="start")
    )

def expand_quarterly_to_monthly(
    quarterly_df: pd.DataFrame,
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Place a quarterly series onto a monthly grid.

    Haver never disaggregates a low frequency up to a higher one, so a
    quarterly series is pulled natively and expanded here.

    'ffill' repeats each quarter's value across its three months (a
    constant / step disaggregation — no fabricated intra-quarter
    movement, and the trailing incomplete quarter stays empty so it
    shows as missing on the heatmap). 'interpolate' draws a straight
    line between successive quarterly points instead.
    """

    # An empty frame has no min/max to build a grid from.
    if quarterly_df.empty:
        return quarterly_df

    monthly_index = pd.date_range(
        start=quarterly_df.index.min(),
        end=quarterly_df.index.max() + pd.DateOffset(months=2),
        freq="MS",
    )

    monthly = quarterly_df.reindex(monthly_index)

    if method == "interpolate":
        # Straight line between quarterly points, then repeat the final
        # partial quarter so the newest months are not left blank.
        monthly = monthly.interpolate(method="linear", limit_area="inside")
        monthly = monthly.ffill(limit=2)
    else:
        # Repeat each quarter's value within its own three months only.
        monthly = monthly.ffill(limit=2)

    return monthly

def fetch_haver_series(
    series_name: str,
    ticker: str,
    extraction_start: pd.Timestamp,
    extraction_end: pd.Timestamp,
    native_frequency: str = "monthly",
) -> pd.DataFrame:
    """
    Download one series from Haver and return a clean DataFrame.

    Each series is requested separately. This makes error handling
    clearer and prevents an incorrect ticker from invalidating the
    entire multi-series query.
    """

    is_quarterly = native_frequency == "quarterly"

    result = _haver().data(
        codes=[ticker],
        startdate=extraction_start.strftime("%Y-%m-%d"),
        enddate=extraction_end.strftime("%Y-%m-%d"),

        # Request the series' NATIVE frequency. Haver aggregates high
        # frequencies down but cannot disaggregate a quarterly series
        # up to monthly (it would drop it), so quarterly series are
        # pulled quarterly and expanded to monthly below.
        frequency="quarterly" if is_quarterly else "monthly",

        # A relaxed aggregation is useful when a higher-frequency
        # source series contains occasional missing observations.
        aggmode="relaxed",
    )

    # A successful Haver query returns a pandas DataFrame.
    #
    # An unsuccessful query can return a Haver error dictionary.
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError(
            f"Haver did not return a DataFrame for "
            f"{series_name} ({ticker}).\n"
            f"Haver response: {result}"
        )

    # The query should contain exactly one column because only
    # one ticker was requested.
    if result.shape[1] != 1:
        raise RuntimeError(
            f"Unexpected number of columns for {series_name}: "
            f"{result.shape[1]}"
        )

    # Replace the technical Haver ticker with a readable name.
    result.columns = [series_name]

    # Convert the index to a standard DatetimeIndex at the native
    # frequency (month-start or quarter-start).
    if is_quarterly:
        result.index = normalise_quarterly_index(result.index)
    else:
        result.index = normalise_monthly_index(result.index)

    # Remove duplicate observations if they exist.
    #
    # The last value is retained because it is normally the most
    # recently updated observation.
    result = result[
        ~result.index.duplicated(keep="last")
    ]

    # Force the values to numeric format.
    #
    # Invalid values are converted to NaN instead of causing
    # the application to crash.
    result[series_name] = pd.to_numeric(
        result[series_name],
        errors="coerce",
    )

    result = result.sort_index()

    # Expand a quarterly series onto the monthly grid so it aligns
    # with the monthly series in the downstream join.
    if is_quarterly:
        result = expand_quarterly_to_monthly(result, method="ffill")

    return result

def infer_native_frequency(raw_series: pd.Series) -> str:
    """
    Guess the native frequency of a series from the spacing of its
    non-missing observations.

    The series carries a monthly index (Haver was queried at monthly
    frequency), so a genuinely quarterly source shows observations
    three months apart, an annual source twelve months apart, and so
    on. The most common gap between consecutive observations is mapped
    to a readable label. This is inferred from the DATA AS FETCHED, so
    if Haver has already filled a coarse series onto every month it
    will legitimately read as monthly.
    """

    observed = raw_series.dropna()

    if len(observed) < 2:
        return "n/a"

    index = observed.index

    # Gap, in whole months, between consecutive observations.
    month_gaps = [
        (index[i].year - index[i - 1].year) * 12
        + (index[i].month - index[i - 1].month)
        for i in range(1, len(index))
    ]

    if not month_gaps:
        return "n/a"

    # Most frequent gap.
    modal_gap = max(set(month_gaps), key=month_gaps.count)

    labels = {
        1: "Monthly",
        2: "Bi-monthly",
        3: "Quarterly",
        6: "Semi-annual",
        12: "Annual",
    }

    return labels.get(modal_gap, f"~{modal_gap}-monthly")

def build_coverage_records(
    combined_union: pd.DataFrame,
    transformed_union: pd.DataFrame,
    std_start: pd.Timestamp,
    std_end: pd.Timestamp,
) -> tuple[list[dict], int]:
    """
    Summarise, per series, how well the standardisation window is
    populated.

    For each series it reports:
    * the native frequency (inferred);
    * how many transformed observations actually fed the mean/σ
      (non-missing points inside the window);
    * that count as a percentage of the monthly slots in the window;
    * how many monthly slots are missing, and the same as a
      percentage.

    The percentages are relative to the number of MONTHLY slots in the
    window, so for a quarterly or annual series a low percentage
    reflects its native frequency rather than a data gap — which is
    why the frequency is reported alongside. What matters most for a
    stable norm is the absolute observation count.

    Rows are produced in SERIES_CONFIG order, so the coverage table
    always lists the series in exactly the same fixed order as the
    heatmap rows.
    """

    # Monthly slots present in the index inside the window.
    window_mask = (
        (transformed_union.index >= std_start)
        & (transformed_union.index <= std_end)
    )
    window_months = int(window_mask.sum())

    records: list[dict] = []

    # Iterate in the fixed SERIES_CONFIG order (not the incidental
    # column order), so the table rows always match the heatmap rows.
    ordered_columns = [
        name for name in SERIES_NAMES if name in combined_union.columns
    ]

    for column in ordered_columns:

        declared_frequency = SERIES_CONFIG.get(column, {}).get(
            "frequency", "monthly"
        )

        if declared_frequency == "quarterly":
            # The series was forward-filled onto the monthly grid, so it
            # would otherwise infer as "Monthly" with an inflated count.
            # Label it honestly and count INDEPENDENT quarters (every
            # third month) that fed the norm.
            frequency_label = "Quarterly (filled)"
            window_slice = transformed_union.loc[window_mask, column]
            fed = int(
                window_slice[window_slice.index.month.isin([1, 4, 7, 10])]
                .notna()
                .sum()
            )
            # Expected independent quarters in the window.
            window_units = max(round(window_months / 3), 1)
        else:
            frequency_label = infer_native_frequency(combined_union[column])
            fed = int(
                transformed_union.loc[window_mask, column].notna().sum()
            )
            window_units = window_months

        # Series that bypass standardisation (published pre-standardised,
        # e.g. GSCPI) do not feed a mean/σ at all; note it in the label.
        if not SERIES_CONFIG.get(column, {}).get("standardise", True):
            frequency_label += " · pre-standardised"

        missing = max(window_units - fed, 0)

        present_pct = (
            round(100 * fed / window_units, 1)
            if window_units
            else 0.0
        )
        missing_pct = (
            round(100 * missing / window_units, 1)
            if window_units
            else 0.0
        )

        # Latest date for which the RAW Haver value exists, over the
        # whole fetched sample, so it reflects how fresh each indicator
        # is. For a monthly series this is simply its last non-missing
        # month. For a quarterly series (forward-filled onto the monthly
        # grid) the filled months are ignored and the last TRUE quarter
        # is reported instead — the most recent quarter-anchor month
        # (Jan/Apr/Jul/Oct) that Haver actually published.
        raw_column = combined_union[column]

        if declared_frequency == "quarterly":
            true_points = raw_column[
                raw_column.index.month.isin([1, 4, 7, 10])
            ].dropna()
            latest_valid = (
                true_points.index.max()
                if not true_points.empty
                else None
            )
            latest_label = (
                f"{latest_valid.year}-Q{(latest_valid.month - 1) // 3 + 1}"
                if latest_valid is not None
                else "—"
            )
        else:
            valid_points = raw_column.dropna()
            latest_valid = (
                valid_points.index.max()
                if not valid_points.empty
                else None
            )
            latest_label = (
                latest_valid.strftime("%Y-%m")
                if latest_valid is not None
                else "—"
            )

        records.append(
            {
                "series": column,
                "frequency": frequency_label,
                "obs": fed,
                "present_pct": present_pct,
                "missing": missing,
                "missing_pct": missing_pct,
                "latest": latest_label,
                "filled": declared_frequency == "quarterly",
                "thin": fed < MIN_RELIABLE_STD_OBS,
            }
        )

    return records, window_months

def build_datasets(
    graph_start_date: str,
    graph_end_date: str,
    std_start_date: str,
    std_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Download the Haver data and create the three main datasets:

    1. combined:
       raw Haver data (display window).

    2. transformed:
       YoY percentage changes, 12-month differences or levels
       (display window).

    3. zscore:
       standardised and economically oriented data (display window).

    Two independent windows are used:

    * The DISPLAY window [graph_start, graph_end] controls what the
      heatmap and grid show.

    * The STANDARDISATION window [std_start, std_end] controls the
      sample used to compute each series' mean and standard
      deviation. Because it is independent of the display window,
      the colours no longer change when the display range changes.

    Haver is queried once per series over the union of both windows,
    extended 12 months backwards to allow the year-on-year
    transformations to be computed from the first required month.
    """

    # Convert the Dash date strings into pandas timestamps.
    graph_start = pd.Timestamp(graph_start_date)
    graph_end = pd.Timestamp(graph_end_date)
    std_start = pd.Timestamp(std_start_date)
    std_end = pd.Timestamp(std_end_date)

    # Validate both requested date ranges.
    if graph_start > graph_end:
        raise ValueError(
            "The display start date must be earlier than the "
            "display end date."
        )

    if std_start > std_end:
        raise ValueError(
            "The standardisation start date must be earlier than "
            "the standardisation end date."
        )

    # Fetch over the union of the display and standardisation
    # windows, then subtract a 12-month buffer for the YoY
    # transformations.
    overall_start = min(graph_start, std_start)
    overall_end = max(graph_end, std_end)

    extraction_start = (
        overall_start
        - pd.DateOffset(months=12)
    )

    # Store the individual Haver DataFrames in this dictionary.
    raw_series = {}

    # Download each configured Haver series.
    for series_name, series_cfg in SERIES_CONFIG.items():

        series_df = fetch_haver_series(
            series_name=series_name,
            ticker=series_cfg["ticker"],
            extraction_start=extraction_start,
            extraction_end=overall_end,
            native_frequency=series_cfg.get("frequency", "monthly"),
        )

        raw_series[series_name] = series_df

    # Join all the individual DataFrames using their monthly dates.
    #
    # axis=1 means that the series are joined as columns.
    combined = pd.concat(
        raw_series.values(),
        axis=1,
    ).sort_index()

    # Preserve the intended series order.
    combined = combined[SERIES_NAMES]

    # Keep only the requested extraction range.
    combined = combined.loc[
        (combined.index >= extraction_start)
        & (combined.index <= overall_end)
    ]

    # Create an empty DataFrame with the same dates.
    transformed = pd.DataFrame(index=combined.index)

    # Apply each series' configured transformation.
    for series_name, series_cfg in SERIES_CONFIG.items():

        transform = series_cfg["transform"]
        raw_column = combined[series_name]

        if transform == "yoy_pct":
            # 100 * (X_t / X_(t-12) - 1)
            transformed[series_name] = (
                raw_column.pct_change(periods=12, fill_method=None) * 100
            )
        elif transform == "diff12":
            # X_t - X_(t-12)
            transformed[series_name] = raw_column.diff(periods=12)
        else:
            # "level": keep the raw value unchanged.
            transformed[series_name] = raw_column

    # Restore the intended column order.
    transformed = transformed[SERIES_NAMES]

    # ----- Standardisation over the fixed standardisation window ---
    #
    # The mean and standard deviation are computed ONLY on the
    # standardisation window, then applied to the whole transformed
    # sample. This decouples the colour scale from the display range.
    std_sample = transformed.loc[
        (transformed.index >= std_start)
        & (transformed.index <= std_end)
    ]

    if std_sample.dropna(how="all").empty:
        raise ValueError(
            "The standardisation window contains no usable data "
            "after transformation. Please widen it."
        )

    # Compute the mean for each series over the standardisation window.
    sample_mean = std_sample.mean()

    # Compute the population standard deviation over the same window.
    sample_std = std_sample.std(ddof=0)

    # Replace zero standard deviations with NaN.
    #
    # This avoids dividing by zero for a constant series.
    sample_std = sample_std.mask(sample_std == 0)

    # Standardise every series:
    #
    # z_t = (x_t - mean) / standard deviation
    zscore = (
        transformed - sample_mean
    ) / sample_std

    # Series flagged "standardise": False are already published in
    # standard-deviation units (e.g. GSCPI), so re-standardising them
    # over the sample window would distort their scale. Their
    # transformed value is used directly as the z-score.
    for series_name, series_cfg in SERIES_CONFIG.items():
        if not series_cfg.get("standardise", True):
            zscore[series_name] = transformed[series_name]

    # Replace possible infinite values with missing values.
    zscore = zscore.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Flip the sign of series flagged "invert", so that a positive
    # z-score consistently reads as stronger economic conditions.
    for series_name, series_cfg in SERIES_CONFIG.items():
        if series_cfg["invert"]:
            zscore[series_name] = -zscore[series_name]

    # ----- Coverage report over the standardisation window ---------
    #
    # Computed here, on the full (union-range) frames, because the
    # standardisation window can lie outside the display window and
    # would otherwise be trimmed away below.
    coverage_records, coverage_window_months = build_coverage_records(
        combined_union=combined,
        transformed_union=transformed,
        std_start=std_start,
        std_end=std_end,
    )

    # ----- Trim every output to the DISPLAY window ----------------
    combined_display = combined.loc[
        (combined.index >= graph_start)
        & (combined.index <= graph_end)
    ]

    transformed_display = transformed.loc[
        (transformed.index >= graph_start)
        & (transformed.index <= graph_end)
    ]

    zscore_display = zscore.loc[
        (zscore.index >= graph_start)
        & (zscore.index <= graph_end)
    ]

    return (
        combined_display,
        transformed_display,
        zscore_display,
        extraction_start,
        {
            "records": coverage_records,
            "window_months": coverage_window_months,
        },
    )

def build_coverage_table(
    coverage: dict | None,
) -> list:
    """
    Turn a coverage summary into a compact HTML table for the
    interface.

    Thin series (few observations feeding the norm) are highlighted so
    they can be spotted at a glance. Rows arrive from
    build_coverage_records already in the fixed SERIES_CONFIG order,
    matching the heatmap.
    """

    if not coverage or not coverage.get("records"):
        return [
            html.Div(
                "Coverage will appear here after the dataset is "
                "updated.",
                style={
                    "fontSize": "13px",
                    "color": "#98A2B3",
                    "fontStyle": "italic",
                },
            )
        ]

    records = coverage["records"]
    window_months = coverage.get("window_months", 0)

    header_cells = [
        "Series",
        "Frequency",
        "Obs feeding σ",
        "% of window",
        "Missing",
        "Missing %",
        "Latest data",
    ]

    header = html.Tr(
        children=[
            html.Th(
                text,
                style={
                    "textAlign": "left" if text in ("Series", "Frequency") else "right",
                    "padding": "6px 12px",
                    "borderBottom": "2px solid #D0D5DD",
                    "fontSize": "12px",
                    "color": "#475467",
                    "whiteSpace": "nowrap",
                },
            )
            for text in header_cells
        ]
    )

    body_rows = []

    for record in records:

        is_thin = record.get("thin", False)

        # Highlight the observation count when the norm rests on very
        # few points.
        obs_style = {
            "textAlign": "right",
            "padding": "5px 12px",
            "fontSize": "13px",
            "fontWeight": "bold" if is_thin else "normal",
            "color": "#B42318" if is_thin else "#344054",
        }

        base_cell = {
            "padding": "5px 12px",
            "fontSize": "13px",
            "color": "#344054",
        }

        # A leading asterisk marks a forward-filled (quarterly) series;
        # the note below the table explains the filling.
        series_display = (
            f"*{record['series']}"
            if record.get("filled")
            else record["series"]
        )

        cells = [
            html.Td(series_display, style={**base_cell, "textAlign": "left"}),
            html.Td(record["frequency"], style={**base_cell, "textAlign": "left"}),
            html.Td(
                (
                    f"{record['obs']}"
                    + ("  ⚠" if is_thin else "")
                ),
                style=obs_style,
            ),
            html.Td(f"{record['present_pct']}%", style={**base_cell, "textAlign": "right"}),
            html.Td(f"{record['missing']}", style={**base_cell, "textAlign": "right"}),
            html.Td(f"{record['missing_pct']}%", style={**base_cell, "textAlign": "right"}),
            html.Td(
                record.get("latest", "—"),
                style={**base_cell, "textAlign": "right", "whiteSpace": "nowrap"},
            ),
        ]

        body_rows.append(
            html.Tr(
                children=cells,
                style={
                    "borderBottom": "1px solid #EEF1F5",
                    "backgroundColor": "#FFF6F5" if is_thin else "white",
                },
            )
        )

    table = html.Table(
        children=[html.Thead(header), html.Tbody(body_rows)],
        style={
            "borderCollapse": "collapse",
            "width": "100%",
            "marginTop": "6px",
        },
    )

    caption = html.Div(
        (
            f"A \"Quarterly (filled)\" series is pulled quarterly "
            f"and forward-filled onto the monthly grid, so it is counted "
            f"in independent quarters (its % is relative to the number "
            f"of quarters in the window). Rows flagged in red have fewer "
            f"than {MIN_RELIABLE_STD_OBS} observations feeding the mean "
            f"and standard deviation."
        ),
        style={
            "fontSize": "12px",
            "color": "#667085",
            "marginTop": "10px",
            "lineHeight": "1.5",
        },
    )

    note = html.Div(
        (
            "\"Latest data\" is the most recent month for which Haver "
            "published a raw value. "
            "* This series is only published every quarter."
        ),
        style={
            "fontSize": "12px",
            "color": "#667085",
            "marginTop": "6px",
            "lineHeight": "1.5",
        },
    )

    return [table, caption, note]

def build_empty_figure(message: str) -> go.Figure:
    """
    Create a blank Plotly figure containing an information message.
    """

    figure = go.Figure()

    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 18},
    )

    figure.update_layout(
        height=500,
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )

    return figure

def parse_annotation_date(raw_date: str | None) -> pd.Timestamp | None:
    """
    Convert a raw annotation date string into a month-start timestamp.

    Returns None if the value is missing or cannot be parsed.
    """

    if not raw_date:
        return None

    try:
        return (
            pd.to_datetime(raw_date)
            .to_period("M")
            .to_timestamp(how="start")
        )
    except Exception:
        return None

def drawable_annotations(
    annotations: list[dict],
    valid_columns: set,
) -> list[dict]:
    """
    Return the annotations whose date falls inside the displayed
    months, in a deterministic order.

    This helper is shared by the figure builder and the drag callback
    so that the i-th drawn line always corresponds to the same
    annotation in both places.
    """

    drawn: list[dict] = []

    for annotation in annotations:
        if parse_annotation_date(annotation.get("date")) in valid_columns:
            drawn.append(annotation)

    return drawn

def build_heatmap_figure(
    combined: pd.DataFrame,
    transformed: pd.DataFrame,
    zscore: pd.DataFrame,
    selected_series: list[str],
    colour_limit: float,
    revision_key: str,
    annotations: list[dict] | None = None,
) -> go.Figure:
    """
    Build the interactive Plotly heatmap.

    The hover box displays:
    - the series;
    - the date;
    - the raw Haver value;
    - the transformed value;
    - the final z-score.

    Annotations are date lines, each of the form:

        {"id": int, "date": "YYYY-MM-DD", "label": str}

    Every date line is drawn as ONE draggable vertical shape. Its
    label (if any) is drawn in a bordered frame JUST ABOVE the plot
    area (inside the top margin), so it never covers the heatmap
    cells. Because the label uses paper y-coordinates and no bounding
    box, it never resizes the plot or shifts the heatmap cells either.

    Interaction model: dragging the canvas PANS along the date axis
    with the scale unchanged (the y-axis is fixed); zooming is done
    with the scroll wheel only.
    """

    annotations = annotations or []

    # Keep only valid selected series, in the fixed SERIES_CONFIG row
    # order. Iterating over SERIES_NAMES (rather than the user's
    # selection order) means the heatmap rows always appear in the same
    # order no matter how the series were picked in the selector.
    selected_lookup = set(selected_series)
    selected_series = [
        series
        for series in SERIES_NAMES
        if series in selected_lookup and series in zscore.columns
    ]

    if not selected_series:
        return build_empty_figure(
            "Select at least one series."
        )

    # Keep the selected columns in the fixed order.
    zscore_selected = zscore[selected_series]

    # Align the raw and transformed datasets with the z-score dates.
    raw_selected = (
        combined
        .reindex(zscore_selected.index)
        [selected_series]
    )

    transformed_selected = (
        transformed
        .reindex(zscore_selected.index)
        [selected_series]
    )

    # Transpose the data:
    #
    # Before:
    # rows = dates, columns = series
    #
    # After:
    # rows = series, columns = dates
    heatmap_data = zscore_selected.T
    raw_heatmap_data = raw_selected.T
    transformed_heatmap_data = transformed_selected.T

    # Build a three-dimensional array for the hover data.
    #
    # Dimension 1: series
    # Dimension 2: dates
    # Dimension 3:
    #     0 = raw value
    #     1 = transformed value
    custom_data = np.stack(
        [
            raw_heatmap_data.to_numpy(),
            transformed_heatmap_data.to_numpy(),
        ],
        axis=-1,
    )

    # Integer colour-bar ticks between -limit and +limit.
    #
    # Using ceil/floor guarantees clean integers even when the slider
    # lands on a half value such as 2.5.
    tick_low = int(np.ceil(-colour_limit))
    tick_high = int(np.floor(colour_limit))
    colour_ticks = list(range(tick_low, tick_high + 1))

    figure = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.to_numpy(),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            customdata=custom_data,

            # Use the parula-style reference colour scale.
            colorscale=CUSTOM_COLOUR_SCALE,

            # Use a symmetric colour range around zero.
            zmin=-colour_limit,
            zmax=colour_limit,
            zmid=0,

            # Missing values remain empty instead of being connected.
            connectgaps=False,
            hoverongaps=False,

            # Keep every monthly rectangle sharp.
            zsmooth=False,
            xgap=0,
            ygap=0,

            colorbar={
                # For a vertical colour bar, side="right" places the
                # title vertically beside the graduations.
                "title": {
                    "text": "Deviations from long-term mean",
                    "side": "right",
                    "font": {"size": 13},
                },
                "orientation": "v",
                "tickmode": "array",
                "tickvals": colour_ticks,
                "ticks": "outside",
                "ticklen": 5,
                "tickfont": {"size": 12},
                "thickness": 22,
                "len": 0.92,
                "x": 1.02,
                "xanchor": "left",
                "y": 0.5,
                "yanchor": "middle",
            },

            # Define exactly what appears when the user points
            # at one heatmap cell.
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Date: %{x|%Y-%m}<br>"
                "Raw value: %{customdata[0]:.3f}<br>"
                "Transformed value: %{customdata[1]:.3f}<br>"
                "Z-score: %{z:.2f}"
                "<extra></extra>"
            ),
        )
    )

    # ----------------------------------------------------------------
    # Date-line annotations.
    #
    # Each drawable annotation becomes exactly ONE draggable line
    # shape (in drawn order), plus an optional boxed label placed just
    # above the plot area. Keeping one shape per annotation, in a
    # deterministic order, lets the drag callback map "shapes[i]" back
    # to the correct annotation.
    # ----------------------------------------------------------------
    valid_columns = set(heatmap_data.columns)
    drawn = drawable_annotations(annotations, valid_columns)

    annotation_shapes: list[dict] = []

    for annotation in drawn:

        annotation_date_value = parse_annotation_date(
            annotation.get("date")
        )
        annotation_label = (annotation.get("label") or "").strip()

        # Draggable vertical reference line (thin, solid, black).
        annotation_shapes.append(
            {
                "type": "line",
                "x0": annotation_date_value,
                "x1": annotation_date_value,
                "y0": 0,
                "y1": 1,
                "xref": "x",
                "yref": "paper",
                "layer": "above",
                "opacity": 0.9,
                "line": {
                    "color": "black",
                    "width": 1.3,
                },
            }
        )

        # Horizontal boxed label placed JUST ABOVE the plot area
        # (anchored to the top of the paper area and shifted up into
        # the top margin), so it never hides the heatmap cells.
        if annotation_label:
            figure.add_annotation(
                x=annotation_date_value,
                xref="x",
                y=1.0,
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                yshift=4,
                text=annotation_label,
                showarrow=False,
                textangle=0,
                font={"size": 11, "color": "black"},
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1,
                borderpad=3,
            )

    # ----------------------------------------------------------------
    # Figure geometry.
    #
    # Height is a simple function of the number of series; labels live
    # in the top margin, so there is no caption band to account for.
    # The top margin is sized to leave room for the label boxes between
    # the subtitle and the plot area.
    # ----------------------------------------------------------------
    top_margin = 110
    bottom_margin = 82
    plot_area_px = max(320, 60 * len(selected_series))
    chart_height = top_margin + plot_area_px + bottom_margin

    # Fix the initial data range so that annotations can never make
    # Plotly extend the date axis beyond the heatmap sample.
    first_month = heatmap_data.columns.min()
    last_month = heatmap_data.columns.max()
    x_range = [
        first_month - pd.Timedelta(days=16),
        last_month + pd.Timedelta(days=16),
    ]

    figure.update_layout(
        title={
            "text": (
                "Indicators of Economic Conditions"
                "<br>"
                "<span style='font-size:13px;color:#555555'>"
                "(standard deviations)"
                "</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        height=chart_height,
        margin={
            "l": 130,
            "r": 130,
            "t": top_margin,
            "b": bottom_margin,
        },
        shapes=annotation_shapes if annotation_shapes else None,
        xaxis={
            "title": "Date",
            "tickformat": "%Y",
            "dtick": "M12",
            "tick0": first_month,
            "range": x_range,
            "autorange": False,
            "showgrid": False,
            "rangeslider": {"visible": False},

            # The date axis can be panned and (scroll-)zoomed.
            "fixedrange": False,
        },
        yaxis={
            "title": "",
            "type": "category",
            "categoryorder": "array",
            "categoryarray": selected_series,
            "autorange": "reversed",
            "showgrid": False,

            # Lock the vertical axis so panning and zooming only affect
            # the date axis: the heatmap rows keep the same scale.
            "fixedrange": True,
        },

        # Dragging the canvas PANS along the date axis (the vertical
        # scale never changes because the y-axis is fixed). Zooming is
        # done with the scroll wheel only; the box-zoom tool is removed
        # from the mode bar in the Graph config.
        dragmode="pan",

        # Preserve the user's pan/zoom while non-date controls change.
        #
        # The view is reset when the loaded date range changes,
        # because revision_key then changes.
        uirevision=revision_key,

        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel={
            "bgcolor": "white",
            "font": {"color": "black"},
        },
    )

    return figure

def style_figure_for_export(figure: go.Figure) -> go.Figure:
    """
    Restyle a copy of the on-screen figure into an Excel/MATLAB-style
    report figure:

    * no chart title and no "(standard deviations)" subtitle;
    * no "Date" axis title;
    * a boxed plot with tick graduations on all four sides.

    The heatmap, colours, colour bar and any above-plot labels are
    kept.
    """

    figure.update_layout(
        # Remove title and subtitle.
        title={"text": ""},
        width=EXPORT_WIDTH_PX,
        font={"size": 13},
        margin={
            "l": 95,
            "r": 120,
            "t": 34,
            "b": 58,
        },
        xaxis={
            # Remove the "Date" axis title.
            "title": {"text": ""},
            # Boxed axis with inward tick graduations on both sides.
            "showline": True,
            "linecolor": "black",
            "linewidth": 1,
            "mirror": "ticks",
            "ticks": "inside",
            "ticklen": 4,
            "tickcolor": "black",
            "tickwidth": 1,
        },
        yaxis={
            "showline": True,
            "linecolor": "black",
            "linewidth": 1,
            "mirror": "ticks",
            "ticks": "inside",
            "ticklen": 4,
            "tickcolor": "black",
            "tickwidth": 1,
        },
    )

    return figure


# ---------------------------------------------------------------------------
# Server-side immutable heatmap snapshots
# ---------------------------------------------------------------------------

_HEATMAP_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_HEATMAP_CACHE_LOCK = threading.RLock()
_HEATMAP_CACHE_MAX = 6


def _trim_cache() -> None:
    while len(_HEATMAP_CACHE) > _HEATMAP_CACHE_MAX:
        _HEATMAP_CACHE.popitem(last=False)


def create_heatmap_snapshot(
    *,
    graph_start_date: str,
    graph_end_date: str,
    std_start_date: str,
    std_end_date: str,
) -> dict[str, Any]:
    combined, transformed, zscore, extraction_start, coverage = build_datasets(
        graph_start_date=graph_start_date,
        graph_end_date=graph_end_date,
        std_start_date=std_start_date,
        std_end_date=std_end_date,
    )
    snapshot_id = uuid.uuid4().hex
    updated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "snapshot_id": snapshot_id,
        "combined": combined,
        "transformed": transformed,
        "zscore": zscore,
        "coverage": coverage,
        "graph_start_date": graph_start_date,
        "graph_end_date": graph_end_date,
        "std_start_date": std_start_date,
        "std_end_date": std_end_date,
        "extraction_start_date": extraction_start.strftime("%Y-%m-%d"),
        "updated_at": updated_at,
    }
    with _HEATMAP_CACHE_LOCK:
        _HEATMAP_CACHE[snapshot_id] = payload
        _HEATMAP_CACHE.move_to_end(snapshot_id)
        _trim_cache()
    return {
        "snapshot_id": snapshot_id,
        "graph_start_date": graph_start_date,
        "graph_end_date": graph_end_date,
        "std_start_date": std_start_date,
        "std_end_date": std_end_date,
        "extraction_start_date": extraction_start.strftime("%Y-%m-%d"),
        "updated_at": updated_at,
        "rows": int(len(combined)),
    }


def get_heatmap_snapshot(snapshot_id: str | None) -> dict[str, Any] | None:
    if not snapshot_id:
        return None
    with _HEATMAP_CACHE_LOCK:
        payload = _HEATMAP_CACHE.get(str(snapshot_id))
        if payload is None:
            return None
        _HEATMAP_CACHE.move_to_end(str(snapshot_id))
        return payload


def raw_table_payload(combined: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    frame = (
        combined.sort_index(ascending=False)
        .reset_index()
        .rename(columns={"index": "Date"})
    )
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    frame = frame.astype(object).where(pd.notna(frame), None)
    records = frame.to_dict("records")
    columns = [
        {"name": str(column), "id": str(column)}
        for column in frame.columns
    ]
    return records, columns


__all__ = [
    "SERIES_CONFIG",
    "SERIES_NAMES",
    "MONTH_OPTIONS",
    "DEFAULT_GRAPH_START_DATE",
    "DEFAULT_END_MONTH",
    "DEFAULT_STD_START_DATE",
    "DEFAULT_STD_END_DATE",
    "EXPORT_WIDTH_PX",
    "EXPORT_SCALE",
    "build_empty_figure",
    "build_coverage_table",
    "build_heatmap_figure",
    "style_figure_for_export",
    "drawable_annotations",
    "create_heatmap_snapshot",
    "get_heatmap_snapshot",
    "raw_table_payload",
]
