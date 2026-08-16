from __future__ import annotations

import ast
import sys
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "src" / "dashboard"
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

from dashboard_xlsx_export import build_export_xlsx


def _datatable_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "dash_table"
            and func.attr == "DataTable"
        ):
            out.append(node)
    return out


def test_every_dash_datatable_exports_xlsx():
    calls = []
    for path in DASH.rglob("*.py"):
        calls.extend(
            (path, node)
            for node in _datatable_calls(path)
        )
    assert calls, "No Dash DataTable calls found."
    for path, node in calls:
        kwargs = {
            kw.arg: kw.value
            for kw in node.keywords
            if kw.arg
        }
        value = kwargs.get("export_format")
        assert (
            isinstance(value, ast.Constant)
            and value.value == "xlsx"
        ), (
            f"{path}: DataTable at line {node.lineno} "
            "does not export xlsx"
        )


def test_table_export_embeds_vintage_and_vintage_date():
    content, filename = build_export_xlsx(
        {
            "vintage": "20260816",
            "table_id": "overview-outlook-table",
            "title": "Inflation Outlook",
            "table": {
                "headers": [
                    "Series",
                    "Latest",
                    "Nowcast",
                    "M+1",
                ],
                "rows": [
                    [
                        "Headline HICP",
                        "2.92",
                        "3.08",
                        "3.17",
                    ]
                ],
            },
        }
    )
    assert filename == (
        "Inflation_Outlook_vintage_20260816.xlsx"
    )
    with zipfile.ZipFile(BytesIO(content), "r") as archive:
        metadata = archive.read(
            "xl/worksheets/sheet1.xml"
        ).decode("utf-8")
        data = archive.read(
            "xl/worksheets/sheet2.xml"
        ).decode("utf-8")

    assert "20260816" in metadata
    assert "2026-08-16" in metadata
    assert "Inflation Outlook" in metadata
    assert "Headline HICP" in data
    assert "Nowcast" in data


def test_client_asset_is_table_only_and_covers_outlook():
    text = (
        DASH
        / "assets"
        / "dashboard_xlsx_export.js"
    ).read_text(encoding="utf-8")

    assert "TABLE-ONLY XLSX export contract" in text
    assert (
        'const OUTLOOK_ID = "overview-outlook-table"'
        in text
    )
    assert "extractOutlookMatrix" in text
    assert "attachOutlookTable" in text
    assert "attachLiteralTable" in text
    assert "Download Inflation Outlook as XLSX" in text

    assert "attachGraph(" not in text
    assert "Download the plotted data as XLSX" not in text
    assert "hidePngButtons" not in text
    assert "display:none !important" not in text


def test_plot_png_report_is_not_hidden_by_xlsx_contract():
    text = (
        DASH
        / "economic_data"
        / "feature.py"
    ).read_text(encoding="utf-8")

    marker = 'id="economic-download-png-button"'
    pos = text.find(marker)
    assert pos >= 0
    window = text[
        max(0, pos - 350):
        pos + 500
    ]
    assert '"display": "none"' not in window


def test_main_keeps_vintage_bridge_for_table_workbooks():
    text = (
        DASH / "energy_bvar_dashboard.py"
    ).read_text(encoding="utf-8")

    assert "register_xlsx_export(server)" in text
    assert 'id="xlsx-export-vintage"' in text
    assert (
        'Output("xlsx-export-vintage", "children")'
        in text
    )
    assert "_vintage_display_text" in text
