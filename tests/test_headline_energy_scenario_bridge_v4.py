from __future__ import annotations

from pathlib import Path

import pytest

import headline_dashboard_slice6 as h6


def _headline_store(vintage="20260816"):
    return {
        "meta": {
            "model_id": "headline_joint",
            "vintage": vintage,
            "run_id": "headline-run",
        }
    }


def _legacy_gas(vintage="20260816"):
    return {
        "ok": True,
        "meta": {
            "vintage": vintage,
            "aggregate_run_id": "agg123",
            "model_id": "gas",
            "model_label": "Natural gas",
            "condition_variable": (
                "natural_gas_wholesale"
            ),
            "condition_description": (
                "+10.00% vs last observed"
            ),
            "path_mode": "percent",
            "path_value": 10.0,
        },
    }


def _conditional_store(payload):
    return {
        "items": {
            "gas": {
                "payload": payload,
            }
        }
    }


def test_legacy_conditional_is_selectable_not_blocked():
    rows = h6._energy_scenario_candidates(
        _conditional_store(_legacy_gas()),
        None,
        _headline_store(),
    )
    assert len(rows) == 1
    row = rows[0]

    assert row["enabled"] is True
    assert row["refresh_on_apply"] is True
    assert row["refresh_spec"]["model_id"] == "gas"
    assert "BLOCKED" not in row["label"]
    assert (
        "refreshes automatically on Apply"
        in row["label"]
    )

    state = h6._bridge_readiness_for_record(
        _headline_store(),
        row,
        source="energy",
        horizon=3,
        statistic="p84",
    )
    assert state["ready"] is True
    assert "rebuilt automatically" in state["detail"]
    assert "Gibbs is not re-estimated" in state["detail"]


def test_legacy_conditional_keeps_same_vintage_guard():
    rows = h6._energy_scenario_candidates(
        _conditional_store(
            _legacy_gas("20260812")
        ),
        None,
        _headline_store("20260816"),
    )
    assert len(rows) == 1
    assert rows[0]["enabled"] is False
    assert "BLOCKED" in rows[0]["label"]
    assert "20260812" in rows[0]["reason"]


def test_apply_rebuilds_from_saved_energy_posterior(
    tmp_path,
    monkeypatch,
):
    results = tmp_path / "results"
    aggregate = (
        results
        / "hicp_energy_aggregate"
        / "20260816"
        / "agg123"
    )
    aggregate.mkdir(parents=True)

    row = h6._energy_scenario_candidates(
        _conditional_store(_legacy_gas()),
        None,
        _headline_store(),
    )[0]

    calls = {}

    def fake_compute(directory, **kwargs):
        calls["directory"] = Path(directory)
        calls.update(kwargs)
        return {
            "ok": True,
            "headline_bridge": {
                "contract": (
                    h6.ENERGY_BRIDGE_CONTRACT_VERSION
                ),
                "scenario_active": True,
                "vintage": "20260816",
            },
        }

    monkeypatch.setattr(
        h6,
        "compute_conditional_scenario",
        fake_compute,
    )

    payload, rebuilt = h6._ensure_current_energy_bridge(
        row,
        results_root=results,
        project_root=tmp_path,
    )

    assert rebuilt is True
    assert calls["directory"] == aggregate
    assert calls["model_id"] == "gas"
    assert calls["condition_variable"] == (
        "natural_gas_wholesale"
    )
    assert calls["path_mode"] == "percent"
    assert calls["path_value"] == pytest.approx(10.0)
    assert payload["headline_bridge"]["contract"] == (
        h6.ENERGY_BRIDGE_CONTRACT_VERSION
    )


def test_current_bridge_is_not_recomputed(monkeypatch):
    payload = _legacy_gas()
    payload["headline_bridge"] = {
        "contract": h6.ENERGY_BRIDGE_CONTRACT_VERSION,
        "scenario_active": True,
        "vintage": "20260816",
    }

    row = h6._energy_scenario_candidates(
        _conditional_store(payload),
        None,
        _headline_store(),
    )[0]
    assert row["refresh_on_apply"] is False

    def fail(*args, **kwargs):
        raise AssertionError(
            "current bridge must not be recomputed"
        )

    monkeypatch.setattr(
        h6,
        "compute_conditional_scenario",
        fail,
    )

    current, rebuilt = h6._ensure_current_energy_bridge(
        row,
        results_root=Path("results"),
        project_root=Path("."),
    )
    assert rebuilt is False
    assert current["headline_bridge"]["contract"] == (
        h6.ENERGY_BRIDGE_CONTRACT_VERSION
    )
