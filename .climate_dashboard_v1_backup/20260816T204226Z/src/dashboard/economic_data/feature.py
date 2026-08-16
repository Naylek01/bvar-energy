"""Optional Economic Data feature for the Inflation Dashboard.

The main dashboard has exactly one integration boundary with this package:
``economic_data_page()`` + ``register_callbacks(...)``.

Set ``DASH_ENABLE_ECONOMIC_DATA=0`` or remove this package to disable the
feature.  No Energy / Headline / Core model imports this module.
"""
from __future__ import annotations

import base64
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import ALL, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

from .heatmap import (
    DEFAULT_END_MONTH,
    DEFAULT_GRAPH_START_DATE,
    DEFAULT_STD_END_DATE,
    DEFAULT_STD_START_DATE,
    EXPORT_SCALE,
    EXPORT_WIDTH_PX,
    MONTH_OPTIONS,
    SERIES_CONFIG,
    SERIES_NAMES,
    build_coverage_table,
    build_empty_figure,
    build_heatmap_figure,
    create_heatmap_snapshot,
    drawable_annotations,
    get_heatmap_snapshot,
    parse_annotation_date,
    raw_table_payload,
    style_figure_for_export,
)
from .monetary_policy import build_monetary_policy_snapshot
from inflation_table_contract import (
    ECONOMIC_CONDITIONS_COLUMNS,
    economic_conditions_records,
    readable_table,
)

TOKENS = {
    "ink": "#111827",
    "muted": "#667085",
    "hairline": "#E4E7EC",
    "panel": "#FFFFFF",
    "background": "#F5F7FA",
    "blue": "#1F4E79",
}


PAGE_STYLE = {
    "maxWidth": "1600px",
    "margin": "0 auto",
    "padding": "4px 0 32px 0",
}
PANEL_STYLE = {
    "backgroundColor": "#FFFFFF",
    "border": "1px solid #E4E7EC",
    "borderRadius": "12px",
    "padding": "18px",
    "marginBottom": "16px",
}
CONTROL_LABEL_STYLE = {
    "display": "block",
    "fontWeight": 650,
    "fontSize": "11px",
    "color": TOKENS["muted"],
    "marginBottom": "7px",
    "textTransform": "uppercase",
    "letterSpacing": ".04em",
}
PRIMARY_BUTTON_STYLE = {
    "backgroundColor": TOKENS["blue"],
    "color": "white",
    "border": "none",
    "borderRadius": "6px",
    "padding": "10px 16px",
    "fontWeight": 700,
    "cursor": "pointer",
}
SECONDARY_BUTTON_STYLE = {
    "backgroundColor": "#FFFFFF",
    "color": TOKENS["blue"],
    "border": f"1px solid {TOKENS['blue']}",
    "borderRadius": "6px",
    "padding": "9px 15px",
    "fontWeight": 700,
    "cursor": "pointer",
}
SMALL_EDIT_BUTTON_STYLE = {
    "backgroundColor": "#FFFFFF",
    "color": TOKENS["blue"],
    "border": "1px solid #AEC4DD",
    "borderRadius": "4px",
    "padding": "3px 10px",
    "fontSize": "11px",
    "cursor": "pointer",
}
SMALL_REMOVE_BUTTON_STYLE = {
    "backgroundColor": "#FFFFFF",
    "color": "#B42318",
    "border": "1px solid #E4A9A2",
    "borderRadius": "4px",
    "padding": "3px 9px",
    "fontSize": "11px",
    "cursor": "pointer",
}

def _policy_date(value) -> str:
    if value in (None, "", "—"):
        return "—"
    try:
        return pd.Timestamp(value).strftime("%d %b %Y")
    except Exception:
        return str(value)

def _policy_value(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.2f}%" if np.isfinite(numeric) else "—"

def _policy_move(value):
    try:
        delta = float(value)
    except (TypeError, ValueError):
        delta = np.nan
    if not np.isfinite(delta):
        return html.Span("—", style={"color": TOKENS["muted"]})
    if delta > 5e-8:
        label, color, bg = f"+{delta:.2f} pp", "#0B7A5C", "rgba(11,122,92,.075)"
    elif delta < -5e-8:
        label, color, bg = f"{delta:.2f} pp", "#B42318", "rgba(180,35,24,.070)"
    else:
        label, color, bg = "0.00 pp", TOKENS["muted"], "rgba(17,24,39,.035)"
    return html.Span(label, style={
        "display": "inline-block", "padding": "3px 7px", "borderRadius": "8px",
        "fontWeight": 750, "fontSize": "10px", "fontVariantNumeric": "tabular-nums",
        "color": color, "background": bg, "whiteSpace": "nowrap",
    })

def _policy_badge(status: str):
    status = str(status or "unavailable").lower()
    if status == "live":
        text, color, bg = "LIVE AT SNAPSHOT", "#0B7A5C", "rgba(11,122,92,.09)"
    elif status == "cached":
        text, color, bg = "CACHED", "#A66A00", "rgba(166,106,0,.09)"
    else:
        text, color, bg = "UNAVAILABLE", TOKENS["muted"], "rgba(17,24,39,.04)"
    return html.Span(text, style={
        "display": "inline-block", "padding": "2px 6px", "borderRadius": "999px",
        "fontSize": "8px", "fontWeight": 750, "letterSpacing": ".035em",
        "color": color, "background": bg, "whiteSpace": "nowrap",
    })

def policy_table(policy: dict | None):
    policy = dict(policy or {})
    ecb, boe = dict(policy.get("ecb") or {}), dict(policy.get("boe") or {})
    header = {
        "textAlign": "left", "fontSize": "10px", "fontWeight": 700,
        "letterSpacing": ".045em", "textTransform": "uppercase",
        "color": TOKENS["muted"], "padding": "8px 10px",
        "borderBottom": f"1px solid {TOKENS['hairline']}", "whiteSpace": "nowrap",
    }
    cell = {
        "padding": "9px 10px", "fontSize": "11px",
        "borderBottom": f"1px solid {TOKENS['hairline']}", "verticalAlign": "middle",
    }
    def current(rec):
        return html.Div(_policy_value(rec.get("value")), style={
            "fontWeight": 750, "fontSize": "13px", "fontVariantNumeric": "tabular-nums"
        })
    def inst(title, subtitle, rec=None, indent=False):
        pad = "16px" if indent else "0"
        kids = [
            html.Div(title, style={"fontWeight": 650 if not indent else 600, "paddingLeft": pad}),
            html.Div(subtitle, style={"fontSize": "9px", "color": TOKENS["muted"], "paddingLeft": pad, "marginTop": "2px"}),
        ]
        if rec is not None:
            kids.append(html.Div(_policy_badge(rec.get("source_status")), style={"paddingLeft": pad, "marginTop": "4px"}))
        return html.Div(kids)
    rows = [html.Tr([
        html.Td(inst("Euro area · ECB", "Single euro-area monetary policy", ecb), style=cell),
        html.Td(ecb.get("rate_name") or "Deposit facility rate", style=cell),
        html.Td(current(ecb), style=cell), html.Td(_policy_move(ecb.get("last_move_pp")), style=cell),
        html.Td(_policy_date(ecb.get("effective_date")), style=cell), html.Td(_policy_date(ecb.get("next_meeting")), style=cell),
    ])]
    shade = {**cell, "background": "rgba(17,24,39,.018)"}
    for country in ("France", "Germany"):
        rows.append(html.Tr([
            html.Td(inst(country, "Eurosystem · ECB policy applies", indent=True), style=shade),
            html.Td("ECB DFR applies", style={**shade, "color": TOKENS["muted"]}),
            html.Td(current(ecb), style=shade),
            html.Td(html.Span("same ECB", style={"fontSize": "9px", "color": TOKENS["muted"]}), style=shade),
            html.Td(_policy_date(ecb.get("effective_date")), style={**shade, "color": TOKENS["muted"]}),
            html.Td(_policy_date(ecb.get("next_meeting")), style={**shade, "color": TOKENS["muted"]}),
        ]))
    rows.append(html.Tr([
        html.Td(inst("United Kingdom · BoE", "Monetary Policy Committee", boe), style=cell),
        html.Td(boe.get("rate_name") or "Bank Rate", style=cell), html.Td(current(boe), style=cell),
        html.Td(_policy_move(boe.get("last_move_pp")), style=cell),
        html.Td(_policy_date(boe.get("effective_date")), style=cell), html.Td(_policy_date(boe.get("next_meeting")), style=cell),
    ]))
    bits = [f"Snapshot status: {str(policy.get('status') or 'unavailable').replace('_', ' ')}."]
    fetched = policy.get("fetched_at_utc")
    if fetched:
        try: bits.append("Frozen " + pd.Timestamp(fetched).strftime("%d %b %Y %H:%M UTC") + ".")
        except Exception: pass
    if policy.get("errors"):
        bits.append("Official endpoint outage detected; cached values are used only where labelled.")
    return html.Div([
        html.Div(html.Table([
            html.Thead(html.Tr([html.Th(x, style=header) for x in (
                "Area / institution", "Policy rate", "Current", "Last move", "Effective", "Next decision"
            )])), html.Tbody(rows)
        ], style={"width": "100%", "borderCollapse": "separate", "borderSpacing": 0}),
        style={"overflowX": "auto", "border": f"1px solid {TOKENS['hairline']}", "borderRadius": "12px", "background": "#FFF"}),
        html.Div(" ".join(bits), style={"marginTop": "7px", "fontSize": "9px", "color": TOKENS["muted"]}),
    ])

def _liquidity_amount(value, currency: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"

    symbol = "€" if str(currency).upper() == "EUR" else "£"
    absolute = abs(numeric)
    if absolute >= 1_000_000:
        return f"{symbol}{numeric / 1_000_000:.2f}tn"
    if absolute >= 1_000:
        return f"{symbol}{numeric / 1_000:.1f}bn"
    return f"{symbol}{numeric:.0f}m"

def _liquidity_delta(value, currency: str):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = np.nan
    if not np.isfinite(numeric):
        return html.Span("—", style={"color": TOKENS["muted"]})

    symbol = "€" if str(currency).upper() == "EUR" else "£"
    absolute = abs(numeric)
    sign = "+" if numeric > 0 else ""
    if absolute >= 1_000_000:
        label = f"{sign}{symbol}{numeric / 1_000_000:.2f}tn"
    elif absolute >= 1_000:
        label = f"{sign}{symbol}{numeric / 1_000:.1f}bn"
    else:
        label = f"{sign}{symbol}{numeric:.0f}m"

    # Liquidity rising/falling is not intrinsically "good" or "bad".
    # Keep the visual treatment directional but economically neutral.
    return html.Span(
        label,
        style={
            "display": "inline-block",
            "padding": "3px 7px",
            "borderRadius": "8px",
            "fontWeight": 700,
            "fontSize": "10px",
            "fontVariantNumeric": "tabular-nums",
            "color": TOKENS["ink"],
            "background": "rgba(17,24,39,.040)",
            "whiteSpace": "nowrap",
        },
    )

def liquidity_table(policy: dict | None):
    policy = dict(policy or {})
    liquidity = dict(policy.get("liquidity") or {})
    ecb_group = dict(liquidity.get("ecb") or {})
    boe_group = dict(liquidity.get("boe") or {})
    ecb_metrics = dict(ecb_group.get("metrics") or {})
    boe_metrics = dict(boe_group.get("metrics") or {})

    header = {
        "textAlign": "left",
        "fontSize": "10px",
        "fontWeight": 700,
        "letterSpacing": ".045em",
        "textTransform": "uppercase",
        "color": TOKENS["muted"],
        "padding": "8px 10px",
        "borderBottom": f"1px solid {TOKENS['hairline']}",
        "whiteSpace": "nowrap",
    }
    cell = {
        "padding": "8px 10px",
        "fontSize": "11px",
        "borderBottom": f"1px solid {TOKENS['hairline']}",
        "verticalAlign": "middle",
    }
    section = {
        **cell,
        "fontWeight": 700,
        "fontSize": "10px",
        "letterSpacing": ".035em",
        "textTransform": "uppercase",
        "background": "rgba(17,24,39,.025)",
        "color": TOKENS["muted"],
    }

    rows = []

    def add_group(title, group, metrics, order):
        rows.append(
            html.Tr(
                [
                    html.Td(
                        [
                            html.Span(title),
                            html.Span(
                                "  ",
                                style={"marginRight": "5px"},
                            ),
                            _policy_badge(group.get("source_status")),
                        ],
                        colSpan=6,
                        style=section,
                    )
                ]
            )
        )
        for key in order:
            metric = dict(metrics.get(key) or {})
            rows.append(
                html.Tr(
                    [
                        html.Td(
                            [
                                html.Div(
                                    metric.get("label") or key.replace("_", " ").title(),
                                    style={"fontWeight": 600},
                                ),
                                html.Div(
                                    metric.get("series") or "—",
                                    style={
                                        "fontSize": "8px",
                                        "color": TOKENS["muted"],
                                        "marginTop": "2px",
                                        "fontFamily": (
                                            "ui-monospace, SFMono-Regular, "
                                            "Menlo, Consolas, monospace"
                                        ),
                                    },
                                ),
                            ],
                            style=cell,
                        ),
                        html.Td(
                            _liquidity_amount(
                                metric.get("value_mn"),
                                metric.get("currency"),
                            ),
                            style={
                                **cell,
                                "fontWeight": 750,
                                "fontVariantNumeric": "tabular-nums",
                            },
                        ),
                        html.Td(
                            _liquidity_delta(
                                metric.get("delta_1m_mn"),
                                metric.get("currency"),
                            ),
                            style=cell,
                        ),
                        html.Td(
                            _liquidity_delta(
                                metric.get("delta_3m_mn"),
                                metric.get("currency"),
                            ),
                            style=cell,
                        ),
                        html.Td(
                            _policy_date(metric.get("date")),
                            style={
                                **cell,
                                "whiteSpace": "nowrap",
                                "color": TOKENS["muted"],
                            },
                        ),
                        html.Td(
                            _policy_badge(metric.get("source_status")),
                            style=cell,
                        ),
                    ]
                )
            )

    add_group(
        "Euro area · ECB",
        ecb_group,
        ecb_metrics,
        (
            "excess_liquidity",
            "current_accounts",
            "minimum_reserves",
            "deposit_facility",
            "marginal_lending",
            "open_market_operations",
        ),
    )
    add_group(
        "United Kingdom · BoE",
        boe_group,
        boe_metrics,
        (
            "reserve_balances",
            "short_term_repo",
            "indexed_long_term_repo",
            "apf_gilts",
        ),
    )

    if not ecb_metrics and not boe_metrics:
        return html.Div(
            "Liquidity data are unavailable in the current snapshot.",
            className="selection-banner",
        )

    status = str(
        policy.get("liquidity_status") or "unavailable"
    ).replace("_", " ")
    notes = [
        f"Liquidity snapshot: {status}.",
        "Δ values are absolute balance-sheet changes, not percentage changes.",
    ]
    if policy.get("errors"):
        notes.append(
            "Where an official endpoint failed, the last valid cached series is used only when explicitly labelled CACHED."
        )

    return html.Div(
        [
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Area / metric", style=header),
                                    html.Th("Level", style=header),
                                    html.Th("Δ 1M", style=header),
                                    html.Th("Δ 3M", style=header),
                                    html.Th("As of", style=header),
                                    html.Th("Source", style=header),
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    style={
                        "width": "100%",
                        "borderCollapse": "separate",
                        "borderSpacing": 0,
                    },
                ),
                style={
                    "overflowX": "auto",
                    "border": f"1px solid {TOKENS['hairline']}",
                    "borderRadius": "12px",
                    "background": "#FFF",
                },
            ),
            html.Div(
                " ".join(notes),
                style={
                    "marginTop": "7px",
                    "fontSize": "9px",
                    "color": TOKENS["muted"],
                },
            ),
        ]
    )


def _panel_heading(eyebrow: str, title: str, subtitle: str):
    return html.Div(
        [
            html.Div(
                eyebrow,
                style={
                    "fontSize": "9px",
                    "fontWeight": 750,
                    "letterSpacing": ".08em",
                    "textTransform": "uppercase",
                    "color": TOKENS["muted"],
                    "marginBottom": "4px",
                },
            ),
            html.H3(
                title,
                style={
                    "margin": "0 0 4px 0",
                    "fontSize": "18px",
                    "fontWeight": 650,
                },
            ),
            html.P(
                subtitle,
                style={
                    "margin": 0,
                    "fontSize": "11px",
                    "color": TOKENS["muted"],
                    "lineHeight": 1.5,
                },
            ),
        ],
        style={"marginBottom": "14px"},
    )


def economic_data_page():
    """Return the fully self-contained Economic Data page layout."""
    return html.Div(
        [
            dcc.Store(id="economic-policy-store", storage_type="memory"),
            dcc.Store(id="economic-heatmap-store", storage_type="memory"),
            dcc.Store(
                id="economic-annotations-store",
                storage_type="memory",
                data=[
                    {"id": 1, "date": "2011-02-01", "label": "Arab Spring"},
                    {"id": 2, "date": "2022-02-01", "label": "Ukraine War"},
                    {"id": 3, "date": "2026-02-01", "label": "Iran War"},
                ],
            ),
            dcc.Store(id="economic-edit-target", storage_type="memory"),
            dcc.Store(id="economic-export-figure-store", storage_type="memory"),
            dcc.Download(id="economic-png-download"),

            html.Div(
                [
                    html.Div("Economic Data", style={
                        "fontSize": "11px", "fontWeight": 750,
                        "letterSpacing": ".08em", "textTransform": "uppercase",
                        "color": TOKENS["muted"],
                    }),
                    html.H1(
                        "Economic Conditions & Central Banks",
                        style={"margin": "4px 0 4px 0", "fontSize": "26px"},
                    ),
                    html.P(
                        "Independent macro-data module: official ECB/BoE policy and liquidity snapshots plus the Haver Economic Conditions Heatmap.",
                        style={"margin": 0, "color": TOKENS["muted"], "fontSize": "12px"},
                    ),
                ],
                style={"marginBottom": "16px"},
            ),

            html.Div(
                [
                    _panel_heading(
                        "Monetary policy",
                        "Official policy rates",
                        "ECB Deposit Facility Rate and Bank of England Bank Rate. France and Germany inherit the ECB rate through the Eurosystem.",
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Refresh policy & liquidity",
                                id="economic-policy-refresh",
                                n_clicks=0,
                                style=SECONDARY_BUTTON_STYLE,
                            ),
                            html.Div(
                                id="economic-policy-status",
                                style={
                                    "fontSize": "10px",
                                    "color": TOKENS["muted"],
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "10px",
                            "marginBottom": "12px",
                            "flexWrap": "wrap",
                        },
                    ),
                    html.Div(id="economic-policy-table"),
                ],
                style=PANEL_STYLE,
            ),

            html.Div(
                [
                    _panel_heading(
                        "Central bank liquidity",
                        "Reserves, facilities & balance-sheet liquidity",
                        "Official ECB and BoE series. Levels are source balances; Δ1M and Δ3M are absolute changes versus the nearest available observations at or before 30/90 calendar days earlier.",
                    ),
                    html.Div(id="economic-liquidity-table"),
                ],
                style=PANEL_STYLE,
            ),

            html.Div(
                [
                    _panel_heading(
                        "Economic conditions heatmap",
                        "Haver macro heatmap",
                        "The economic transformations and sign conventions are preserved from the standalone Economic Heatmap project. Haver loads automatically on first page activation and when extraction parameters change; unchanged revisits are frozen.",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Heatmap date range", style=CONTROL_LABEL_STYLE),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="economic-display-start-month",
                                                options=MONTH_OPTIONS,
                                                value=DEFAULT_GRAPH_START_DATE,
                                                clearable=False,
                                                searchable=True,
                                                style={"width": "130px"},
                                            ),
                                            html.Span("→", style={"color": TOKENS["muted"]}),
                                            dcc.Dropdown(
                                                id="economic-display-end-month",
                                                options=MONTH_OPTIONS,
                                                value=DEFAULT_END_MONTH,
                                                clearable=False,
                                                searchable=True,
                                                style={"width": "130px"},
                                            ),
                                        ],
                                        style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Displayed series", style=CONTROL_LABEL_STYLE),
                                    dcc.Dropdown(
                                        id="economic-series-selector",
                                        options=[{"label": s, "value": s} for s in SERIES_NAMES],
                                        value=list(SERIES_NAMES),
                                        multi=True,
                                        clearable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Colour range", style=CONTROL_LABEL_STYLE),
                                    dcc.Slider(
                                        id="economic-colour-limit",
                                        min=1, max=4, step=0.5, value=3,
                                        marks={1: "±1", 2: "±2", 3: "±3", 4: "±4"},
                                    ),
                                ]
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "minmax(280px,1fr) minmax(360px,1.5fr) minmax(220px,.8fr)",
                            "gap": "20px",
                            "alignItems": "end",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Standardisation sample", style=CONTROL_LABEL_STYLE),
                                    dcc.Dropdown(
                                        id="economic-std-basis",
                                        options=[
                                            {"label": "Fixed range (independent of display)", "value": "fixed"},
                                            {"label": "Match displayed range", "value": "display"},
                                        ],
                                        value="fixed",
                                        clearable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Standardisation range", style=CONTROL_LABEL_STYLE),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="economic-std-start-month",
                                                options=MONTH_OPTIONS,
                                                value=DEFAULT_STD_START_DATE,
                                                clearable=False,
                                                searchable=True,
                                                style={"width": "130px"},
                                            ),
                                            html.Span("→", style={"color": TOKENS["muted"]}),
                                            dcc.Dropdown(
                                                id="economic-std-end-month",
                                                options=MONTH_OPTIONS,
                                                value=DEFAULT_STD_END_DATE,
                                                clearable=False,
                                                searchable=True,
                                                style={"width": "130px"},
                                            ),
                                        ],
                                        style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                    ),
                                ]
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "minmax(280px,1fr) minmax(360px,1.5fr)",
                            "gap": "20px",
                            "marginTop": "18px",
                            "paddingTop": "16px",
                            "borderTop": f"1px solid {TOKENS['hairline']}",
                        },
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Refresh dataset",
                                id="economic-update-data-button",
                                n_clicks=0,
                                style=PRIMARY_BUTTON_STYLE,
                            ),
                            html.Button(
                                "Download report PNG",
                                id="economic-download-png-button",
                                n_clicks=0,
                                style=SECONDARY_BUTTON_STYLE,
                            ),
                            html.Div(
                                "No Haver request has been made in this dashboard session.",
                                id="economic-update-status",
                                style={"fontSize": "11px", "color": TOKENS["muted"]},
                            ),
                        ],
                        style={
                            "display": "flex", "gap": "10px", "alignItems": "center",
                            "flexWrap": "wrap", "marginTop": "18px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div("Standardisation coverage", style={
                                "fontWeight": 650, "fontSize": "11px",
                                "marginBottom": "6px",
                            }),
                            html.Div(id="economic-coverage-report"),
                        ],
                        style={
                            "marginTop": "18px", "paddingTop": "14px",
                            "borderTop": f"1px solid {TOKENS['hairline']}",
                        },
                    ),
                ],
                style=PANEL_STYLE,
            ),

            html.Div(
                [
                    _panel_heading(
                        "Current conditions",
                        "Latest indicator summary",
                        "The same frozen Haver snapshot as the heatmap, expressed numerically for quick reading.",
                    ),
                    readable_table(
                        "economic-conditions-summary-table",
                        ECONOMIC_CONDITIONS_COLUMNS,
                        page_size=12,
                    ),
                ],
                style=PANEL_STYLE,
            ),

            html.Div(
                [
                    _panel_heading(
                        "Interactive view",
                        "Economic Conditions Heatmap",
                        "Pan horizontally, scroll to zoom, and drag the vertical date lines. Display controls reuse the frozen server-side Haver snapshot and never re-query Haver.",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Date line at", style=CONTROL_LABEL_STYLE),
                                    dcc.Input(
                                        id="economic-annotation-date-text",
                                        type="text",
                                        value=DEFAULT_END_MONTH,
                                        debounce=True,
                                        placeholder="YYYY-MM-DD or YYYY",
                                        style={"width": "100%", "marginBottom": "5px"},
                                    ),
                                    dcc.DatePickerSingle(
                                        id="economic-annotation-date",
                                        date=DEFAULT_END_MONTH,
                                        display_format="YYYY-MM-DD",
                                        clearable=True,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Label (optional)", style=CONTROL_LABEL_STYLE),
                                    dcc.Input(
                                        id="economic-annotation-text",
                                        type="text",
                                        value="",
                                        style={"width": "100%"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Add",
                                        id="economic-annotation-add-button",
                                        n_clicks=0,
                                        style=PRIMARY_BUTTON_STYLE,
                                    ),
                                    html.Button(
                                        "Cancel",
                                        id="economic-annotation-cancel-button",
                                        n_clicks=0,
                                        style=SECONDARY_BUTTON_STYLE,
                                    ),
                                ],
                                style={"display": "flex", "gap": "8px"},
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "minmax(220px,1fr) minmax(280px,1.4fr) auto",
                            "gap": "14px",
                            "alignItems": "end",
                        },
                    ),
                    html.Div(
                        id="economic-edit-hint",
                        style={"fontSize": "10px", "color": TOKENS["muted"], "margin": "8px 0"},
                    ),
                    html.Div(
                        [
                            html.Div("Current date lines", style={"fontWeight": 650, "fontSize": "11px"}),
                            html.Button(
                                "Clear all",
                                id="economic-annotation-clear-button",
                                n_clicks=0,
                                style=SECONDARY_BUTTON_STYLE,
                            ),
                        ],
                        style={
                            "display": "flex", "justifyContent": "space-between",
                            "alignItems": "center", "marginBottom": "7px",
                        },
                    ),
                    html.Div(id="economic-annotation-list", style={"marginBottom": "12px"}),
                    dcc.Loading(
                        type="circle",
                        children=dcc.Graph(
                            id="economic-heatmap",
                            figure=build_empty_figure(
                                "Opening Economic Data loads the default Haver snapshot automatically."
                            ),
                            config={
                                "scrollZoom": True,
                                "displayModeBar": True,
                                "displaylogo": False,
                                "responsive": True,
                                "edits": {"shapePosition": True},
                                "modeBarButtonsToRemove": ["zoom2d", "lasso2d", "select2d"],
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "economic_conditions_heatmap",
                                    "width": EXPORT_WIDTH_PX,
                                    "height": 640,
                                    "scale": EXPORT_SCALE,
                                },
                            },
                        ),
                    ),
                ],
                style=PANEL_STYLE,
            ),

            html.Div(
                [
                    _panel_heading(
                        "Source data",
                        "Raw combined Haver dataset",
                        "Latest observations first. The table is generated from the same frozen server-side snapshot used by the heatmap.",
                    ),
                    dash_table.DataTable(export_format="xlsx", export_headers="display", export_columns="all", 
                        id="economic-raw-table",
                        data=[],
                        columns=[],
                        page_size=20,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontSize": "10px",
                            "padding": "6px",
                            "textAlign": "right",
                            "minWidth": "90px",
                            "maxWidth": "160px",
                        },
                        style_header={
                            "fontWeight": 700,
                            "backgroundColor": "#F8FAFC",
                        },
                    ),
                ],
                style=PANEL_STYLE,
            ),
        ],
        style=PAGE_STYLE,
    )


def register_callbacks(
    app,
    *,
    project_root: str | Path,
    registry_store_id: str = "registry-store",
):
    """Register every Economic Data callback on the host Dash app."""
    project_root = Path(project_root)

    @app.callback(
        Output("economic-policy-store", "data"),
        Output("economic-policy-status", "children"),
        Input(registry_store_id, "data"),
        Input("economic-policy-refresh", "n_clicks"),
        Input("url", "pathname"),
        State("economic-policy-store", "data"),
        prevent_initial_call=False,
    )
    def refresh_policy_liquidity(
        registry_snapshot, _manual_refresh, pathname, current_payload
    ):
        if (pathname or "") != "/economic-data":
            raise PreventUpdate
        snapshot_id = str((registry_snapshot or {}).get("snapshot_id") or "")
        manual_refresh = ctx.triggered_id == "economic-policy-refresh"
        if (
            not manual_refresh
            and isinstance(current_payload, dict)
            and current_payload.get("dashboard_registry_snapshot_id") == snapshot_id
        ):
            # Re-entering Economic Data with the same global snapshot is a no-op.
            raise PreventUpdate

        payload = build_monetary_policy_snapshot(
            cache_path=(
                project_root
                / ".dashboard_cache"
                / "economic_data_policy_liquidity_snapshot.json"
            )
        )
        payload["dashboard_registry_snapshot_id"] = snapshot_id
        status = str(payload.get("status") or "unavailable")
        liquidity_status = str(
            payload.get("liquidity_status") or "unavailable"
        )
        fetched = payload.get("fetched_at_utc")
        stamp = "—"
        if fetched:
            try:
                stamp = pd.Timestamp(fetched).strftime("%d %b %Y %H:%M UTC")
            except Exception:
                stamp = str(fetched)
        return payload, (
            f"Policy: {status.replace('_', ' ')} · "
            f"Liquidity: {liquidity_status.replace('_', ' ')} · "
            f"frozen {stamp}"
        )

    @app.callback(
        Output("economic-policy-table", "children"),
        Input("economic-policy-store", "data"),
    )
    def render_policy(payload):
        return policy_table(payload)

    @app.callback(
        Output("economic-liquidity-table", "children"),
        Input("economic-policy-store", "data"),
    )
    def render_liquidity(payload):
        return liquidity_table(payload)

    @app.callback(
        Output("economic-heatmap-store", "data"),
        Output("economic-raw-table", "data"),
        Output("economic-raw-table", "columns"),
        Output("economic-update-status", "children"),
        Output("economic-coverage-report", "children"),
        Input("economic-update-data-button", "n_clicks"),
        Input("url", "pathname"),
        Input("economic-display-start-month", "value"),
        Input("economic-display-end-month", "value"),
        Input("economic-std-basis", "value"),
        Input("economic-std-start-month", "value"),
        Input("economic-std-end-month", "value"),
        State("economic-heatmap-store", "data"),
        prevent_initial_call=False,
    )
    def update_haver_dataset(
        _n_clicks,
        pathname,
        graph_start_date,
        graph_end_date,
        std_basis,
        std_start_date,
        std_end_date,
        current_manifest,
    ):
        if (pathname or "") != "/economic-data":
            raise PreventUpdate
        if not graph_start_date or not graph_end_date:
            return no_update, no_update, no_update, (
                "Select both display dates."
            ), no_update

        if std_basis == "display":
            effective_std_start = graph_start_date
            effective_std_end = graph_end_date
        else:
            if not std_start_date or not std_end_date:
                return no_update, no_update, no_update, (
                    "Select a standardisation range."
                ), no_update
            effective_std_start = std_start_date
            effective_std_end = std_end_date

        request_signature = {
            "graph_start_date": str(graph_start_date),
            "graph_end_date": str(graph_end_date),
            "std_start_date": str(effective_std_start),
            "std_end_date": str(effective_std_end),
        }
        manual_refresh = ctx.triggered_id == "economic-update-data-button"
        current_signature = {
            key: str((current_manifest or {}).get(key) or "")
            for key in request_signature
        }
        if not manual_refresh and current_signature == request_signature:
            # Page revisit or a no-op control update: keep the existing server
            # snapshot and every rendered chart/table exactly as-is.
            raise PreventUpdate

        try:
            manifest = create_heatmap_snapshot(
                graph_start_date=graph_start_date,
                graph_end_date=graph_end_date,
                std_start_date=effective_std_start,
                std_end_date=effective_std_end,
            )
            snapshot = get_heatmap_snapshot(manifest["snapshot_id"])
            if snapshot is None:
                raise RuntimeError("Heatmap snapshot was not retained.")
            rows, columns = raw_table_payload(snapshot["combined"])
            coverage = build_coverage_table(snapshot["coverage"])
            status = (
                f"Frozen Haver snapshot {manifest['snapshot_id'][:10]} · "
                f"updated {manifest['updated_at']} · "
                f"{manifest['rows']:,} monthly rows · "
                f"standardised {effective_std_start} → {effective_std_end}."
            )
            return manifest, rows, columns, status, coverage
        except Exception as exc:
            traceback.print_exc()
            return (
                no_update,
                no_update,
                no_update,
                f"Automatic update failed: {exc}",
                no_update,
            )

    @app.callback(
        Output("economic-annotations-store", "data"),
        Output("economic-edit-target", "data", allow_duplicate=True),
        Input("economic-annotation-add-button", "n_clicks"),
        Input("economic-annotation-clear-button", "n_clicks"),
        Input({"type": "economic-annotation-remove", "index": ALL}, "n_clicks"),
        State("economic-annotation-date", "date"),
        State("economic-annotation-text", "value"),
        State("economic-annotations-store", "data"),
        State("economic-edit-target", "data"),
        prevent_initial_call=True,
    )
    def manage_annotations(
        _add, _clear, _remove, annotation_date, annotation_text,
        annotations, edit_target,
    ):
        annotations = annotations or []
        triggered = ctx.triggered_id
        if triggered is None:
            raise PreventUpdate
        if triggered == "economic-annotation-clear-button":
            return [], None
        if isinstance(triggered, dict) and triggered.get("type") == "economic-annotation-remove":
            if not ctx.triggered or not ctx.triggered[0].get("value"):
                return no_update, no_update
            target = triggered.get("index")
            remaining = [x for x in annotations if x.get("id") != target]
            return remaining, (None if target == edit_target else no_update)
        if triggered == "economic-annotation-add-button":
            if not annotation_date:
                return no_update, no_update
            label = (annotation_text or "").strip()
            if edit_target is not None:
                updated = [
                    (
                        {**item, "date": annotation_date, "label": label}
                        if item.get("id") == edit_target
                        else item
                    )
                    for item in annotations
                ]
                return updated, None
            next_id = max((x.get("id", 0) for x in annotations), default=0) + 1
            return annotations + [{
                "id": next_id,
                "date": annotation_date,
                "label": label,
            }], no_update
        return no_update, no_update

    @app.callback(
        Output("economic-annotation-date", "date", allow_duplicate=True),
        Input("economic-annotation-date-text", "value"),
        State("economic-annotation-date", "date"),
        prevent_initial_call=True,
    )
    def typed_date_to_calendar(value, current):
        if value is None or not str(value).strip():
            return None if current else no_update
        parsed = pd.to_datetime(str(value).strip(), errors="coerce")
        if pd.isna(parsed):
            return no_update
        normalised = parsed.strftime("%Y-%m-%d")
        if current and pd.to_datetime(current).strftime("%Y-%m-%d") == normalised:
            return no_update
        return normalised

    @app.callback(
        Output("economic-annotation-date-text", "value", allow_duplicate=True),
        Input("economic-annotation-date", "date"),
        State("economic-annotation-date-text", "value"),
        prevent_initial_call=True,
    )
    def calendar_to_typed_date(value, current):
        normalised = (
            pd.to_datetime(value).strftime("%Y-%m-%d")
            if value else ""
        )
        return no_update if (current or "").strip() == normalised else normalised

    @app.callback(
        Output("economic-annotations-store", "data", allow_duplicate=True),
        Input("economic-heatmap", "relayoutData"),
        State("economic-annotations-store", "data"),
        State("economic-heatmap-store", "data"),
        prevent_initial_call=True,
    )
    def move_annotations(relayout_data, annotations, manifest):
        if not relayout_data or not manifest:
            raise PreventUpdate
        shape_updates = {}
        for key, value in relayout_data.items():
            if key.startswith("shapes[") and "]." in key:
                try:
                    idx = int(key[len("shapes["):key.index("]")])
                except ValueError:
                    continue
                shape_updates.setdefault(idx, {})[key.split(".", 1)[1]] = value
        if not shape_updates:
            raise PreventUpdate

        snapshot = get_heatmap_snapshot(manifest.get("snapshot_id"))
        if snapshot is None:
            raise PreventUpdate
        zscore = snapshot["zscore"]
        columns = list(pd.to_datetime(zscore.index))
        drawn = drawable_annotations(annotations or [], set(columns))
        updates = {}
        for idx, props in shape_updates.items():
            if idx < 0 or idx >= len(drawn):
                continue
            xvals = []
            for key in ("x0", "x1"):
                if key in props:
                    try:
                        xvals.append(pd.to_datetime(props[key]))
                    except Exception:
                        pass
            if not xvals:
                continue
            centre = pd.Timestamp(
                sum(value.value for value in xvals) // len(xvals)
            )
            snapped = min(
                columns,
                key=lambda month: abs((month - centre).value),
            )
            updates[drawn[idx]["id"]] = snapped.strftime("%Y-%m-%d")
        if not updates:
            raise PreventUpdate
        return [
            (
                {**item, "date": updates[item["id"]]}
                if item.get("id") in updates else item
            )
            for item in (annotations or [])
        ]

    @app.callback(
        Output("economic-edit-target", "data"),
        Output("economic-annotation-date", "date"),
        Output("economic-annotation-text", "value"),
        Input({"type": "economic-annotation-edit", "index": ALL}, "n_clicks"),
        State("economic-annotations-store", "data"),
        prevent_initial_call=True,
    )
    def edit_annotation(_clicks, annotations):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or triggered.get("type") != "economic-annotation-edit":
            raise PreventUpdate
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            raise PreventUpdate
        target = triggered.get("index")
        match = next((x for x in (annotations or []) if x.get("id") == target), None)
        if match is None:
            raise PreventUpdate
        return target, match.get("date"), match.get("label", "")

    @app.callback(
        Output("economic-edit-target", "data", allow_duplicate=True),
        Input("economic-annotation-cancel-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def cancel_edit(_clicks):
        return None

    @app.callback(
        Output("economic-annotation-add-button", "children"),
        Output("economic-edit-hint", "children"),
        Input("economic-edit-target", "data"),
    )
    def reflect_edit_mode(target):
        if target is not None:
            return "Update", (
                f"Editing date line #{target}. Change the date/label and "
                "press Update, or Cancel."
            )
        return "Add", "Use Edit below to change an existing date line."

    @app.callback(
        Output("economic-annotation-list", "children"),
        Input("economic-annotations-store", "data"),
    )
    def render_annotation_list(annotations):
        annotations = annotations or []
        if not annotations:
            return html.Div(
                "No date lines.",
                style={"fontSize": "10px", "color": TOKENS["muted"]},
            )
        rows = []
        for item in annotations:
            description = f"{item['date']} · Date line"
            if item.get("label"):
                description += f" · “{item['label']}”"
            rows.append(
                html.Div(
                    [
                        html.Span(description, style={"fontSize": "10px"}),
                        html.Div(
                            [
                                html.Button(
                                    "Edit",
                                    id={"type": "economic-annotation-edit", "index": item["id"]},
                                    n_clicks=0,
                                    style=SMALL_EDIT_BUTTON_STYLE,
                                ),
                                html.Button(
                                    "Remove",
                                    id={"type": "economic-annotation-remove", "index": item["id"]},
                                    n_clicks=0,
                                    style=SMALL_REMOVE_BUTTON_STYLE,
                                ),
                            ],
                            style={"display": "flex", "gap": "6px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "padding": "6px 8px",
                        "border": f"1px solid {TOKENS['hairline']}",
                        "borderRadius": "5px",
                        "marginBottom": "5px",
                    },
                )
            )
        return rows

    @app.callback(
        Output("economic-conditions-summary-table", "data"),
        Input("economic-heatmap-store", "data"),
        Input("economic-series-selector", "value"),
    )
    def economic_conditions_table(manifest, selected_series):
        if not manifest:
            return []
        snapshot = get_heatmap_snapshot(manifest.get("snapshot_id"))
        if snapshot is None:
            return []
        return economic_conditions_records(
            snapshot, SERIES_CONFIG, selected_series or []
        )

    @app.callback(
        Output("economic-heatmap", "figure"),
        Input("economic-heatmap-store", "data"),
        Input("economic-series-selector", "value"),
        Input("economic-colour-limit", "value"),
        Input("economic-annotations-store", "data"),
    )
    def update_heatmap(manifest, selected_series, colour_limit, annotations):
        if not manifest:
            return build_empty_figure(
                "Opening Economic Data loads the default Haver snapshot automatically."
            )
        snapshot = get_heatmap_snapshot(manifest.get("snapshot_id"))
        if snapshot is None:
            return build_empty_figure(
                "Heatmap snapshot expired. Click Refresh dataset to create a new frozen snapshot."
            )
        revision = (
            f"{manifest.get('snapshot_id')}-"
            f"{manifest.get('graph_start_date')}-"
            f"{manifest.get('graph_end_date')}"
        )
        return build_heatmap_figure(
            combined=snapshot["combined"],
            transformed=snapshot["transformed"],
            zscore=snapshot["zscore"],
            selected_series=selected_series or [],
            colour_limit=float(colour_limit or 3),
            revision_key=revision,
            annotations=annotations or [],
        )

    app.clientside_callback(
        """
        function(nClicks) {
            if (!nClicks) {
                return window.dash_clientside.no_update;
            }
            const container = document.getElementById("economic-heatmap");
            const plot = container ? container.querySelector(".js-plotly-plot") : null;
            if (!plot || !plot.data || !plot.layout) {
                return window.dash_clientside.no_update;
            }
            return {
                figure: {
                    data: JSON.parse(JSON.stringify(plot.data)),
                    layout: JSON.parse(JSON.stringify(plot.layout))
                },
                requested_at: Date.now()
            };
        }
        """,
        Output("economic-export-figure-store", "data"),
        Input("economic-download-png-button", "n_clicks"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("economic-png-download", "data"),
        Input("economic-export-figure-store", "data"),
        prevent_initial_call=True,
    )
    def export_png(payload):
        if not payload or not payload.get("figure"):
            raise PreventUpdate
        figure = style_figure_for_export(go.Figure(payload["figure"]))
        height = int(figure.layout.height or 640)
        png_bytes = pio.to_image(
            figure,
            format="png",
            width=EXPORT_WIDTH_PX,
            height=height,
            scale=EXPORT_SCALE,
        )
        return {
            "content": base64.b64encode(png_bytes).decode("ascii"),
            "filename": (
                "economic_conditions_heatmap_"
                + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                + ".png"
            ),
            "base64": True,
            "type": "image/png",
        }


__all__ = [
    "economic_data_page",
    "register_callbacks",
    "policy_table",
    "liquidity_table",
]
