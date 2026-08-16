from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src/dashboard/energy_bvar_dashboard.py"
COND = ROOT / "src/model/energy_bvar_dashboard_conditional.py"
SCEN = ROOT / "src/model/energy_bvar_dashboard_scenarios.py"
SLICE = ROOT / "src/dashboard/headline_dashboard_slice6.py"
PNG = ROOT / "src/dashboard/assets/dashboard_png_corner.js"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_energy_conditional_editor_is_business_readable():
    source = text(MAIN)
    assert "# ENERGY_SCENARIO_UI_CLARITY_AND_JOINT_WORKSPACE_V1" in source
    assert 'html.Label("Energy component"' in source
    assert 'html.Label("Scenario driver"' in source
    assert 'html.Label("Assumption type"' in source
    assert '"Advanced / source run"' in source
    assert 'id="conditional-question"' in source
    assert '"You are asking: "' in source
    assert '"Open Headline scenarios →"' in source
    assert 'html.Label("Component BVAR"' not in source
    assert 'html.Label("Conditioned variable"' not in source


def test_png_export_has_one_dedicated_top_right_control():
    source = text(PNG)
    assert "GRAPH_PNG_CORNER_CONTRACT_V1" in source
    assert 'button.textContent = "↓ PNG"' in source
    assert 'top: "4px"' in source
    assert 'right: "8px"' in source
    assert 'root.style.paddingTop = "30px"' in source
    assert "window.Plotly.downloadImage" in source
    assert "removeLegacyPngButtons" in source
    assert "hideNativePlotlyDownload" in source


def test_tax_figures_always_name_the_selected_hicp_component():
    source = text(SCEN)
    assert "# TAX_SCENARIO_EXPLICIT_HICP_LABEL_V1" in source
    assert "def _selected_hicp_label(" in source
    assert 'f"{hicp_label} — inflation: baseline vs tax scenario"' in source
    assert 'f"{hicp_label} — price-level impact"' in source
    assert 'f"{hicp_label} — year-on-year inflation impact"' in source
    assert 'f"{hicp_label} — VAT and excise assumptions"' in source
    assert 'title="Tax assumption impact on HICP inflation"' not in source


def test_joint_energy_workspace_builds_before_headline_transfer():
    main = text(MAIN)
    backend = text(COND)
    headline = text(SLICE)

    assert 'id="joint-energy-scenario-store"' in main
    assert 'id="joint-energy-build"' in main
    assert 'id="joint-energy-main-graph"' in main
    assert 'id="joint-energy-impact-graph"' in main
    assert 'id="joint-energy-contribution-graph"' in main
    assert "compute_joint_energy_scenario(" in main

    assert "def compute_joint_energy_scenario(" in backend
    assert '"aggregate_contribution_impact_yoy"' in backend
    assert "def joint_energy_contribution_impact_figure(" in backend
    assert '"marginal_effects_summed": False' in backend

    assert "# HEADLINE_CONSUMES_PREBUILT_JOINT_ENERGY_V1" in headline
    assert 'energy_joint_store_id="joint-energy-scenario-store"' in headline
    assert '"prebuilt_energy_scenarios_workspace"' in headline
    # Headline must no longer rebuild the joint Energy package itself.
    assert "compute_joint_energy_scenario(" not in headline


def test_modified_python_sources_parse():
    for path in (MAIN, COND, SCEN, SLICE):
        ast.parse(text(path))
