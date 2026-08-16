from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import headline_bvar_conditional as hbc


def _bridge():
    dates = pd.date_range("2026-07-01", periods=7, freq="MS")
    base = np.vstack([
        np.arange(100.0, 107.0),
        np.arange(101.0, 108.0),
        np.arange(102.0, 109.0),
    ])
    scenario = base + np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    bridge = hbc.energy_level_paths_headline_bridge(
        dates=dates,
        baseline_level_paths=base,
        scenario_level_paths=scenario,
        vintage="20260816",
        aggregate_run_id="agg123",
        forecast_name="unconditional",
        scenario_active=True,
        lineage={
            "source_kind": "conditional_observable",
            "source_label": "Conditional · Gas",
            "scenario_components": ["gas"],
        },
    )
    return bridge, scenario


def test_pointwise_percentiles_are_computed_directly_from_energy_draws():
    bridge, scenario = _bridge()
    assert bridge["contract"] == "energy-headline-bridge-v3"
    assert bridge["available_percentiles"] == list(range(1, 100))
    np.testing.assert_allclose(
        bridge["scenario_level"]["p37"],
        np.nanquantile(scenario, 0.37, axis=0),
    )
    np.testing.assert_allclose(
        bridge["scenario_level"]["p84"],
        np.nanquantile(scenario, 0.84, axis=0),
    )
    assert bridge["scenario_level"]["q84"] == bridge["scenario_level"]["p84"]
    assert "level_paths" not in bridge


def test_arbitrary_percentile_is_accepted_by_growth_reanchoring(monkeypatch, tmp_path):
    bridge, _ = _bridge()
    history_dates = pd.date_range("2025-01-01", "2026-12-01", freq="MS")
    history = pd.Series(np.linspace(95.0, 105.0, len(history_dates)), index=history_dates)
    posterior = SimpleNamespace(
        vintage="20260816",
        run_directory=tmp_path,
        inputs=SimpleNamespace(native_levels={"hicp_energy": history}),
    )
    monkeypatch.setattr(
        hbc,
        "energy_headline_ratio_diagnostic",
        lambda **kwargs: {"pathological_flag": False},
    )
    targets = pd.date_range("2026-08-01", periods=3, freq="MS")
    values, lineage = hbc.energy_bridge_condition(
        {"headline_bridge": bridge},
        posterior=posterior,
        target_dates=targets,
        statistic="p37",
        project_root=tmp_path,
    )
    p37 = pd.Series(
        bridge["scenario_level"]["p37"],
        index=pd.DatetimeIndex(pd.to_datetime(bridge["dates"])),
    )
    gross = np.array([
        p37.loc[d] / p37.loc[d - pd.DateOffset(months=1)] for d in targets
    ])
    anchor = float(history.loc[pd.Timestamp("2026-07-01")])
    np.testing.assert_allclose(values, anchor * np.cumprod(gross))
    assert lineage["condition_statistic"] == "p37"
    assert lineage["condition_statistic_label"] == "Pointwise percentile P37"


def test_dashboard_source_contract_for_scenario_selection_and_axes():
    root = Path(__file__).resolve().parents[1]
    slice6 = (root / "src/dashboard/headline_dashboard_slice6.py").read_text(encoding="utf-8")
    energy_conditional = (root / "src/model/energy_bvar_dashboard_conditional.py").read_text(encoding="utf-8")
    main = (root / "src/dashboard/energy_bvar_dashboard.py").read_text(encoding="utf-8")

    for token in (
        'id="h6-energy-scenario-select"',
        'id="h6-condition-horizon"',
        'id="h6-display-horizon"',
        'id="h6-energy-path-kind"',
        'id="h6-energy-percentile"',
        "def _energy_scenario_candidates(",
        "short_horizon_labels(dates)",
        "_symmetric_zero_axis(fig",
        'line_x = ["Actual"] + line_x',
    ):
        assert token in slice6

    assert 'CONDITIONAL_CONTRACT_VERSION = "energy-conditional-observable-v3"' in energy_conditional
    assert '"headline_bridge": headline_bridge' in energy_conditional
    assert 'energy_conditional_store_id="conditional-store"' in main
