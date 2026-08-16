"""Design tokens and the Plotly template for the energy BVAR dashboard.

The palette is sampled from the SPX Capital site, not approximated. Keeping the
values here as well as in ``assets/dashboard.css`` is deliberate duplication:
Plotly figures are drawn in Python and cannot read CSS custom properties, so the
two must be kept in step by hand. :func:`assert_tokens_match_css` checks that
they are.

Accessibility constraint that shapes the palette: SPX cyan (#009FE3) scores
2.44:1 against the paper grey and fails WCAG AA for text. It is used for rules,
marks, focus and the primary data series; never for axis labels or annotations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "TOKENS",
    "SERIES",
    "FAN_BANDS",
    "TEMPLATE_NAME",
    "register_template",
    "apply_theme",
    "fan_traces",
    "forecast_origin_marker",
    "assert_tokens_match_css",
    "GRAPH_CONFIG",
    "graph_config",
    "INFLATION_COLORS",
    "apply_inflation_figure_style",
]


TOKENS: dict[str, str] = {
    "paper": "#EAE8E9",
    "surface": "#F7F6F7",
    "surface_sunk": "#E3E1E2",
    "ink": "#040506",
    "ink_soft": "#3A3D40",
    "muted": "#6B6E72",
    "cyan": "#009FE3",
    "cyan_deep": "#0079AE",
    "navy": "#004B99",
    "navy_deep": "#003570",
    "hairline": "#D2D0D1",
    "hairline_soft": "#DEDCDD",
    "positive": "#0B7A5C",
    "negative": "#B3341F",
    "warning": "#A66A00",
}

# Categorical order for component series. Cyan first because the selected
# component is the subject of the chart; everything else is deliberately
# desaturated so one line reads as foreground.
SERIES: tuple[str, ...] = (
    TOKENS["cyan"],
    TOKENS["navy"],
    TOKENS["positive"],
    TOKENS["warning"],
    TOKENS["negative"],
    "#6E4B9E",
    TOKENS["muted"],
)

# Fan opacities. The 90% band must stay light enough that the 68% band inside
# it remains distinguishable when both are shown.
FAN_BANDS: dict[str, float] = {"90": 0.13, "68": 0.26}

TEMPLATE_NAME = "spx_light"

_TITLE_FONT = dict(family="Inter, Segoe UI, Helvetica Neue, Arial, sans-serif", size=15, color=TOKENS["ink"])
_BASE_FONT = dict(family="Inter, Segoe UI, Helvetica Neue, Arial, sans-serif", size=12, color=TOKENS["ink_soft"])


def _axis() -> dict:
    return dict(
        showgrid=True,
        gridcolor=TOKENS["hairline_soft"],
        gridwidth=1,
        zeroline=True,
        zerolinecolor=TOKENS["hairline"],
        zerolinewidth=1,
        showline=True,
        linecolor=TOKENS["hairline"],
        linewidth=1,
        ticks="outside",
        ticklen=4,
        tickcolor=TOKENS["hairline"],
        tickfont=dict(size=11, color=TOKENS["muted"]),
        title=dict(font=dict(size=11, color=TOKENS["muted"])),
        automargin=True,
    )


def register_template() -> str:
    """Register the template with Plotly and make it the default."""
    template = go.layout.Template(
        layout=dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=_BASE_FONT,
            title=dict(font=_TITLE_FONT, x=0, xanchor="left", pad=dict(b=12)),
            colorway=list(SERIES),
            xaxis=_axis(),
            yaxis=_axis(),
            margin=dict(l=8, r=12, t=28, b=8),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor=TOKENS["surface"],
                bordercolor=TOKENS["hairline"],
                font=dict(size=12, color=TOKENS["ink"]),
                align="left",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(size=11, color=TOKENS["ink_soft"]),
            ),
            dragmode="pan",
            separators=". ",
        )
    )
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME


GRAPH_CONFIG: dict = {
    # Scroll-to-zoom is a dcc.Graph *config* option, not a layout property.
    # Setting dragmode="pan" alone gives panning but leaves the wheel inert.
    "scrollZoom": True,
    "displayModeBar": "hover",
    "displaylogo": False,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "toggleSpikelines",
    ],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def graph_config(filename: str | None = None) -> dict:
    """Config for ``dcc.Graph``: wheel zooms, drag pans, PNG export named."""
    config = {key: (dict(value) if isinstance(value, dict) else value)
              for key, value in GRAPH_CONFIG.items()}
    if filename:
        config["toImageButtonOptions"]["filename"] = str(filename)
    return config


def apply_theme(
    figure: go.Figure,
    *,
    uirevision: str | None = None,
    height: int | None = None,
    y_title: str | None = None,
    lock_y: bool = False,
) -> go.Figure:
    """Apply the template and the interaction settings the brief calls for.

    ``uirevision`` is the important argument: set it to a value that changes
    only when the underlying run changes (for example ``f"{run_id}:{variable}"``)
    and Plotly will preserve the user's zoom and pan across redraws. Without it,
    toggling a fan band resets the view.
    """
    if TEMPLATE_NAME not in pio.templates:
        register_template()
    figure.update_layout(template=TEMPLATE_NAME)
    if uirevision is not None:
        figure.update_layout(uirevision=uirevision)
    if height is not None:
        figure.update_layout(height=int(height))
    if y_title is not None:
        figure.update_yaxes(title_text=y_title)
    figure.update_xaxes(
        rangeslider=dict(visible=False),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor=TOKENS["hairline"],
        spikethickness=1,
        spikedash="solid",
    )
    # Locking y confines both wheel-zoom and drag-pan to the time axis. The
    # cost is that y no longer rescales when zooming into a narrow window, so
    # a short span can look flat. Off by default.
    figure.update_yaxes(fixedrange=bool(lock_y))
    return figure


def fan_traces(
    dates: Sequence,
    quantiles: Mapping[str, Sequence[float]],
    *,
    bands: Sequence[str] = ("90", "68"),
    color: str | None = None,
    name: str = "Forecast",
    show_median: bool = True,
) -> list[go.Scatter]:
    """Build fan-chart traces from a quantile mapping.

    ``quantiles`` must supply ``q05``/``q95`` for the 90% band, ``q16``/``q84``
    for the 68% band, and ``q50`` for the median. Bands are drawn widest first
    so the narrower one sits on top.
    """
    color = color or TOKENS["cyan"]
    rgb = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    pairs = {"90": ("q05", "q95"), "68": ("q16", "q84")}
    traces: list[go.Scatter] = []

    for band in sorted(bands, key=lambda b: -int(b)):
        if band not in pairs:
            raise ValueError(f"Unknown fan band {band!r}; expected '68' or '90'.")
        low_key, high_key = pairs[band]
        missing = [k for k in (low_key, high_key) if k not in quantiles]
        if missing:
            raise KeyError(f"Fan band {band}% requires {missing} in the quantile mapping.")
        fill = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{FAN_BANDS[band]})"
        traces.append(
            go.Scatter(
                x=list(dates), y=list(quantiles[high_key]), mode="lines",
                line=dict(width=0), hoverinfo="skip", showlegend=False,
                name=f"{name} {band}% upper",
            )
        )
        traces.append(
            go.Scatter(
                x=list(dates), y=list(quantiles[low_key]), mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=fill,
                hoverinfo="skip", name=f"{band}% interval", showlegend=True,
            )
        )

    if show_median:
        if "q50" not in quantiles:
            raise KeyError("show_median=True requires 'q50' in the quantile mapping.")
        traces.append(
            go.Scatter(
                x=list(dates), y=list(quantiles["q50"]), mode="lines",
                line=dict(color=color, width=2), name=f"{name} median",
                hovertemplate="%{y:.2f}<extra></extra>",
            )
        )
    return traces


def forecast_origin_marker(figure: go.Figure, origin, *, label: str = "Forecast origin") -> go.Figure:
    """Draw the vertical rule separating history from the predictive path."""
    figure.add_vline(
        x=origin,
        line=dict(color=TOKENS["muted"], width=1, dash="dot"),
        annotation_text=label,
        annotation_position="top left",
        annotation=dict(font=dict(size=10, color=TOKENS["muted"]), yshift=4),
    )
    return figure


def assert_tokens_match_css(css_path: str | Path) -> dict[str, tuple[str, str]]:
    """Fail loudly if the Python tokens have drifted from the stylesheet.

    Returns the mismatches rather than only raising on the first one, so a
    single run reports every divergence.
    """
    text = Path(css_path).read_text(encoding="utf-8")
    found = {
        name.strip(): value.strip().upper()
        for name, value in re.findall(r"--([a-z0-9-]+):\s*(#[0-9A-Fa-f]{6})\s*;", text)
    }
    mismatches: dict[str, tuple[str, str]] = {}
    for key, value in TOKENS.items():
        css_key = key.replace("_", "-")
        if css_key not in found:
            mismatches[key] = (value.upper(), "absent from CSS")
        elif found[css_key] != value.upper():
            mismatches[key] = (value.upper(), found[css_key])
    if mismatches:
        detail = "; ".join(f"{k}: python={a} css={b}" for k, (a, b) in mismatches.items())
        raise AssertionError(f"Theme tokens have drifted from {css_path}: {detail}")
    return mismatches



# HEADLINE DELIVERY 5 — semantic visual contract
INFLATION_COLORS: dict[str, str] = {
    # Economic series: one stable semantic colour everywhere.
    "headline": TOKENS["ink"],
    "hicp_total": TOKENS["ink"],
    "hicp_energy": TOKENS["cyan"],
    "hicp_food": TOKENS["navy"],
    "hicp_neig": TOKENS["positive"],
    "hicp_services": TOKENS["warning"],
    # Path/statistical roles.
    "observed": TOKENS["ink"],
    "nowcast": TOKENS["warning"],
    "fitted": TOKENS["muted"],
    "baseline": TOKENS["cyan"],
    "conditional": TOKENS["navy"],
    "impact": TOKENS["cyan_deep"],
}


def apply_inflation_figure_style(
    figure: go.Figure,
    *,
    uirevision: str | None = None,
    height: int | None = None,
    y_title: str | None = None,
) -> go.Figure:
    """Headline/Inflation figure geometry layered on the shared SPX theme.

    Visible chart titles live in the surrounding Dash panel, not inside Plotly.
    This prevents title/legend collisions and makes Headline figures use the
    same interaction grammar as Energy while keeping the semantic palette
    explicit.
    """
    apply_theme(
        figure,
        uirevision=uirevision,
        height=height,
        y_title=y_title,
    )
    figure.update_layout(
        title=None,
        margin=dict(l=52, r=22, t=50, b=38),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, color=TOKENS["ink_soft"]),
        ),
    )
    figure.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    )
    return figure


register_template()
