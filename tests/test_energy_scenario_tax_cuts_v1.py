from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))


def _dashboard_text() -> str:
    return (ROOT / "src" / "dashboard" / "energy_bvar_dashboard.py").read_text(
        encoding="utf-8"
    )


def _function_node(text: str, name: str):
    tree = ast.parse(text)
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(nodes) == 1, (name, len(nodes))
    return nodes[0]


def test_joint_single_scenario_contract_and_error_import():
    text = _dashboard_text()
    tree = ast.parse(text)
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "energy_bvar_dashboard_conditional":
            imports.extend(alias.name for alias in node.names)
    assert "ConditionalScenarioError" in imports

    node = _function_node(text, "_joint_energy_dashboard_recipe")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "ConditionalScenarioError": RuntimeError,
        "json": json,
        "conditional_set_components": lambda store: [],
        "conditional_set_payload": lambda store, model_id: None,
        "scenario_set_to_tax_scenarios": lambda store: {
            "gas": {"start_date": "2026-09-01", "vat_percent": 15.0}
        },
        "scenario_set_summary": lambda store: [
            {
                "model_id": "gas",
                "label": "HICP Gas",
                "vat_delta_pp": -5.0,
                "excise_delta": 0.0,
                "excise_unit": "EUR/MWh",
            }
        ],
    }
    exec(compile(module, "<joint-recipe>", "exec"), ns)
    recipe = ns["_joint_energy_dashboard_recipe"](
        conditional_store=None,
        tax_store={"one": True},
        vintage="20260816",
        selected_aggregate_run_id="agg-one",
    )
    assert recipe["scenario_count"] == 1
    assert recipe["aggregate_run_id"] == "agg-one"
    assert len(recipe["labels"]) == 1


def _isolated_function(path: Path, name: str, namespace: dict | None = None):
    text = path.read_text(encoding="utf-8")
    node = _function_node(text, name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = dict(namespace or {})
    exec(compile(module, f"<{name}>", "exec"), ns)
    return ns[name]


def test_vat_validators_accept_full_zero_to_100_range():
    specs = (
        (MODEL / "energy_bvar_gas.py", "_validate_vat_percent"),
        (MODEL / "energy_bvar_electricity.py", "_validate_vat_percent"),
        (MODEL / "energy_bvar_weekly_fuels.py", "_validate_weekly_vat_percent"),
    )
    valid = np.array([0.0, 0.25, 0.99, 1.0, 19.9, 100.0])
    for path, name in specs:
        fn = _isolated_function(path, name, {"np": np})
        fn(valid)
        with pytest.raises(ValueError):
            fn(np.array([-0.01]))
        with pytest.raises(ValueError):
            fn(np.array([100.01]))
        with pytest.raises(ValueError):
            fn(np.array([np.nan]))


def test_component_contract_exposes_selected_start_and_delta_bounds():
    text = (MODEL / "energy_bvar_component_hicp.py").read_text(encoding="utf-8")
    node = _function_node(text, "component_tax_scenario_contract")
    args = [arg.arg for arg in node.args.args + node.args.kwonlyargs]
    assert "start_date" in args
    source = ast.get_source_segment(text, node) or ""
    for key in (
        '"selected_start"',
        '"vat_delta_min"',
        '"vat_delta_max"',
        '"excise_delta_min"',
        '"tax_floor_policy"',
    ):
        assert key in source


def test_dashboard_tax_cut_preview_and_prebuild_guard_contract():
    text = _dashboard_text()
    assert "ENERGY_SCENARIO_TAX_CUTS_V1" in text
    assert "VAT change (pp; − = cut)" in text
    assert "Excise change (" in text and "− = cut" in text
    assert 'Output("scenario-vat-delta", "min")' in text
    assert 'Output("scenario-vat-delta", "max")' in text
    assert 'Output("scenario-excise-delta", "min")' in text
    assert 'Input("scenario-start-date", "date")' in text
    assert "baseline at selected start" in text
    assert "Negative changes are tax cuts" in text

    node = _function_node(text, "_scenario_vat_delta_error")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    exec(compile(module, "<vat-delta-guard>", "exec"), ns)
    guard = ns["_scenario_vat_delta_error"]
    assert guard(baseline_vat=19.9, delta_vat=-5.0) is None
    message = guard(baseline_vat=19.9, delta_vat=-20.0)
    assert "VAT cut too large" in message
    assert "-19.90 pp" in message
    assert "VAT = 0%" in message
