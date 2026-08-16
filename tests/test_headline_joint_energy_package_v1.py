from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src/dashboard/energy_bvar_dashboard.py"
BACKEND = ROOT / "src/model/energy_bvar_dashboard_conditional.py"
HEADLINE = ROOT / "src/dashboard/headline_dashboard_slice6.py"


def test_joint_energy_package_is_constructed_in_energy_workspace():
    main = MAIN.read_text(encoding="utf-8")
    backend = BACKEND.read_text(encoding="utf-8")

    assert 'id="joint-energy-scenario-store"' in main
    assert 'id="joint-energy-build"' in main
    assert "compute_joint_energy_scenario(" in main
    assert "def compute_joint_energy_scenario(" in backend
    assert '"marginal_effects_summed": False' in backend
    assert '"aggregate_contribution_impact_yoy"' in backend


def test_headline_consumes_prebuilt_joint_energy_store():
    source = HEADLINE.read_text(encoding="utf-8")
    assert "# HEADLINE_CONSUMES_PREBUILT_JOINT_ENERGY_V1" in source
    assert 'energy_joint_store_id="joint-energy-scenario-store"' in source
    assert '"prebuilt_energy_scenarios_workspace"' in source
    assert "compute_joint_energy_scenario(" not in source


def test_joint_energy_keeps_mean_and_p01_p99_headline_path_contract():
    source = HEADLINE.read_text(encoding="utf-8")
    assert 'id="h6-energy-path-kind"' in source
    assert 'id="h6-energy-percentile"' in source
    assert "_energy_statistic(path_kind, percentile)" in source
