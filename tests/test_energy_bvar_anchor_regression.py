from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from energy_bvar_anchor_regression import (  # noqa: E402
    AnchorRegressionConfig,
    assert_corrected_anchor,
    run_gas_anchor_regression,
)


def _synthetic_result() -> dict:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2020-01-01", periods=36, freq="MS", name="date")
    changes = np.column_stack(
        [
            0.30 + 0.15 * rng.standard_normal(len(dates)),
            0.20 + 0.08 * rng.standard_normal(len(dates)),
        ]
    )
    levels_array = np.array([40.0, 75.0])[None, :] + np.cumsum(changes, axis=0)
    levels = pd.DataFrame(
        levels_array,
        index=dates,
        columns=["natural_gas_wholesale", "gas_pre_tax"],
    )

    balanced_end = dates[-4]
    levels.loc[dates[-2]:, "gas_pre_tax"] = np.nan
    complete_differences = pd.DataFrame(
        levels_array,
        index=dates,
        columns=levels.columns,
    ).diff()
    p = 2
    last_rows = complete_differences.loc[:balanced_end].iloc[-p:].to_numpy()[::-1]
    last_companion_state = last_rows.reshape(-1)
    base_level = pd.DataFrame(
        levels_array,
        index=dates,
        columns=levels.columns,
    ).loc[balanced_end].to_numpy(dtype=float)

    draws = 8
    n = 2
    k = 1 + n * p
    B = np.zeros((draws, k, n), dtype=float)
    B[:, 0] = np.array([0.05, 0.03])
    B[:, 1:5] = np.array(
        [
            [0.20, 0.04],
            [0.02, 0.15],
            [0.05, 0.01],
            [0.01, 0.04],
        ]
    )[None, :, :]
    B += 0.002 * rng.standard_normal(B.shape)

    A = np.repeat(np.eye(n)[None, :, :], draws, axis=0)
    A[:, 1, 0] = -0.08
    phi = np.full((draws, n), 0.01)
    outlier_probabilities = np.zeros((draws, n))
    log_variance = np.zeros((draws, 6, n))
    log_variance[:, :, 0] = np.log(0.25)
    log_variance[:, :, 1] = np.log(0.10)

    return {
        "B": B,
        "A": A,
        "phi": phi,
        "outlier_probabilities": outlier_probabilities,
        "log_variance": log_variance,
        "prior": {"outlier_grid": np.arange(2.0, 21.0)},
        "variables": list(levels.columns),
        "n_draws": draws,
        "metadata": {"frequency": "monthly", "run_id": "synthetic"},
        "prep": {
            "levels": levels,
            "balanced_end": balanced_end,
            "last_calendar_date": dates[-1],
            "level_at_balanced_end": base_level,
            "last_companion_state": last_companion_state,
            "p": p,
        },
    }


def _make_broken_before(current_path: Path, target: Path) -> Path:
    source = current_path.read_text(encoding="utf-8")
    needle = "    level_endog = np.asarray(level_endog, dtype=float)"
    replacement = (
        "    if p > 1:\n"
        "        raise ValueError(\"historical p>1 anchor failure fixture\")\n"
        + needle
    )
    broken, count = source.replace(needle, replacement, 1), source.count(needle)
    if count < 1:
        raise AssertionError("Could not create the historical broken anchor fixture.")
    target.write_text(broken, encoding="utf-8")
    return target


def test_corrected_anchor_invariants(tmp_path: Path):
    current = Path.cwd() / "src" / "model" / "energy_bvar_model.py"
    if not current.exists():
        raise FileNotFoundError(current)
    result = _synthetic_result()
    condition = {
        "natural_gas_wholesale": np.full(
            3,
            1.10 * result["prep"]["levels"]["natural_gas_wholesale"].dropna().iloc[-1],
        )
    }
    report = run_gas_anchor_regression(
        result,
        old_model_path=current,
        new_model_path=current,
        level_conditions=condition,
        config=AnchorRegressionConfig(horizon=3, n_draws=6, seed=44),
    )
    assert_corrected_anchor(report)
    assert report["same_model_file"]
    assert report["comparison"].loc[
        "max_abs_difference_level_paths", "value"
    ] == 0.0


def test_historical_dimension_failure_is_reported(tmp_path: Path):
    current = Path.cwd() / "src" / "model" / "energy_bvar_model.py"
    if not current.exists():
        raise FileNotFoundError(current)
    broken = _make_broken_before(current, tmp_path / "energy_bvar_model_before.py")
    result = _synthetic_result()
    report = run_gas_anchor_regression(
        result,
        old_model_path=broken,
        new_model_path=current,
        config=AnchorRegressionConfig(horizon=3, n_draws=4, seed=55),
    )
    assert report["old"]["status"] == "error"
    assert report["new"]["status"] == "ok"
    assert_corrected_anchor(report)
