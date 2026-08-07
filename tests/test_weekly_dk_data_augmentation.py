from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'src' / 'model'
sys.path.insert(0, str(MODEL_DIR))

import energy_bvar_model as model
import energy_bvar_weekly_fuels as weekly


def make_weekly_levels(nobs=140, seed=123):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2020-01-06', periods=nobs, freq='W-MON', name='date')
    diffs = rng.normal(scale=[1.0, 0.6, 0.8], size=(nobs, 3))
    # Mild VAR-like persistence in differences.
    for t in range(1, nobs):
        diffs[t] += 0.25 * diffs[t-1]
    levels = np.array([100.0, 60.0, 90.0]) + np.cumsum(diffs, axis=0)
    return pd.DataFrame(levels, index=idx, columns=['consumer', 'crude', 'refined'])


def test_interior_missing_levels_are_augmented_and_forecastable():
    levels = make_weekly_levels(150)
    # Missing WOB-like interior weeks only in the consumer price.
    missing_dates = [levels.index[55], levels.index[83], levels.index[110]]
    levels.loc[missing_dates, 'consumer'] = np.nan

    prep = model.prepare_bvar_panel(levels, p=6, frequency='weekly')
    assert prep['requires_data_augmentation'] is True
    assert prep['n_interior_missing_dates'] >= 3
    assert prep['n_regression_observations'] > 100

    cfg = model.SamplerConfig(reps=12, burn=6, thin=1, seed=7, progress_every=0)
    prior = model.BVARSVOPriorConfig()
    result = model.gibbs_bvar_sv_outlier(
        levels,
        p=6,
        frequency='weekly',
        prior_config=prior,
        sampler_config=cfg,
    )

    assert result['data_augmentation'] == 'durbin_koopman_interior_levels'
    assert result['completed_differences_draws'].shape[0] == 6
    assert result['last_companion_state_draws'].shape == (6, 3 * 6)
    assert result['missing_level_draws'].shape[1] >= 3
    assert np.isfinite(result['missing_level_draws']).all()

    # Released levels inside the augmentation window must be reproduced exactly.
    median_diffs = model.posterior_completed_differences(result)
    assert not median_diffs.isna().any().any()

    fc = model.forecast_bvar_sv_outlier(result, H=4, n_draws=4, seed=99)
    base = np.asarray(result['prep']['level_at_balanced_end'], dtype=float)
    reconstructed = base[None, None, :] + np.cumsum(fc['diff_paths'], axis=1)
    assert np.max(np.abs(reconstructed - fc['level_paths'])) < 1e-8
    assert (fc['future_dates'].dayofweek == 0).all()


def test_p24_augmented_state_dimensions():
    levels = make_weekly_levels(90, seed=321)
    levels.loc[levels.index[40], 'consumer'] = np.nan
    prep = model.prepare_bvar_panel(levels, p=24, frequency='weekly')
    assert prep['requires_data_augmentation'] is True
    cfg = model.SamplerConfig(reps=4, burn=2, thin=1, seed=5, progress_every=0)
    result = model.gibbs_bvar_sv_outlier(
        levels,
        p=24,
        frequency='weekly',
        sampler_config=cfg,
    )
    assert result['last_companion_state_draws'].shape[1] == 72


def test_partial_coverage_mask_is_adapter_only(tmp_path):
    vintage = tmp_path / '20260807'
    vintage.mkdir()
    idx = pd.date_range('2026-07-13', periods=5, freq='W-MON', name='date')
    panel = pd.DataFrame({
        'date': idx,
        'wob_diesel_pre_tax': [1000, 1001, 1002, 1003, np.nan],
        'wob_petrol_pre_tax': [1100, 1101, 1102, 1103, np.nan],
        'crude_oil_eur': [70, 71, 72, 73, 74],
        'refined_petroleum_eur': [80, 81, 82, 83, 84],
        'refined_diesel_eur': [90, 91, 92, 93, 94],
    })
    dataset = vintage / 'car_fuels_weekly.csv'
    panel.to_csv(dataset, index=False)
    coverage = pd.DataFrame([
        {
            'series': 'crude_oil_eur', 'panel_column': 'crude_oil_eur',
            'panel_label_date': idx[-1], 'affected_files': 'car_fuels_weekly.csv|liquid_fuels_weekly.csv',
            'final_period_complete': False, 'coverage_ratio': 0.2,
            'final_period_input_observations': 1, 'expected_input_observations': 5,
        },
        {
            'series': 'refined_diesel_eur', 'panel_column': 'refined_diesel_eur',
            'panel_label_date': idx[-1], 'affected_files': 'car_fuels_weekly.csv|liquid_fuels_weekly.csv',
            'final_period_complete': False, 'coverage_ratio': 0.2,
            'final_period_input_observations': 1, 'expected_input_observations': 5,
        },
    ])
    coverage.to_csv(vintage / 'aggregation_coverage.csv', index=False)

    loaded = weekly.load_weekly_fuel_panel(dataset, model='diesel')
    assert np.isnan(loaded.loc[idx[-1], 'crude_oil_eur'])
    assert np.isnan(loaded.loc[idx[-1], 'refined_diesel_eur'])
    assert len(loaded.attrs['partial_period_mask']) == 2


def _synthetic_augmented_state_from_levels(prep, full_levels, perturb=0.0):
    n = prep['n']
    p = prep['p']
    companion_size = n * p
    dates = prep['dates']
    anchor = np.asarray(prep['augmentation_anchor_level'], dtype=float)
    path_levels = full_levels.reindex(dates).to_numpy(dtype=float)
    relative = path_levels - anchor[None, :]
    current_diffs = np.diff(
        np.vstack([np.zeros((1, n)), relative]), axis=0
    )
    state = np.zeros((len(dates), companion_size + n), dtype=float)
    state[:, :n] = current_diffs
    state[:, companion_size:] = relative
    if perturb:
        observed = prep['estimation_levels'].to_numpy(dtype=float)
        mask = np.isfinite(observed)
        # perturb one published cumulative-level state by a tiny amount
        row, col = np.argwhere(mask)[len(np.argwhere(mask)) // 2]
        state[row, companion_size + col] += perturb
    return state


def test_observed_level_roundoff_is_projected_not_rejected():
    full = make_weekly_levels(180, seed=777)
    levels = full.copy()
    levels.loc[levels.index[[60, 95, 130]], 'consumer'] = np.nan
    prep = model.prepare_bvar_panel(levels, p=24, frequency='weekly')
    state = _synthetic_augmented_state_from_levels(prep, full, perturb=2.58e-8)

    completed_differences, _, _, _, completed_levels, last_state = (
        model._completed_regression_from_augmented_state(prep, state)
    )

    observed = prep['estimation_levels'].to_numpy(dtype=float)
    mask = np.isfinite(observed)
    assert np.array_equal(completed_levels[mask], observed[mask])

    anchor = np.asarray(prep['augmentation_anchor_level'], dtype=float)
    rebuilt = anchor[None, :] + np.cumsum(
        completed_differences.loc[prep['dates']].to_numpy(dtype=float), axis=0
    )
    assert np.max(np.abs(rebuilt - completed_levels)) < 1e-10

    expected_last = (
        completed_differences.loc[prep['dates']].to_numpy(dtype=float)[-prep['p']:]
        [::-1]
        .reshape(-1)
    )
    assert np.array_equal(last_state, expected_last)


def test_material_observed_level_mismatch_still_fails():
    import pytest

    full = make_weekly_levels(180, seed=888)
    levels = full.copy()
    levels.loc[levels.index[[60, 95, 130]], 'consumer'] = np.nan
    prep = model.prepare_bvar_panel(levels, p=24, frequency='weekly')
    state = _synthetic_augmented_state_from_levels(prep, full, perturb=1e-3)

    with pytest.raises(RuntimeError, match='beyond the configured projection gate'):
        model._completed_regression_from_augmented_state(prep, state)


def test_record_only_mode_logs_projection_diagnostics_without_machine_precision_gate():
    full = make_weekly_levels(180, seed=999)
    levels = full.copy()
    levels.loc[levels.index[[60, 95, 130]], 'consumer'] = np.nan
    prep = model.prepare_bvar_panel(levels, p=24, frequency='weekly')
    state = _synthetic_augmented_state_from_levels(prep, full, perturb=3.5e-7)

    out = model._completed_regression_from_augmented_state(
        prep,
        state,
        projection_mode='record_only',
        return_diagnostics=True,
    )
    diagnostics = out[-1]
    assert diagnostics['max_relative_level_error'] > 0
    assert 0 <= diagnostics['level_argmax_time_fraction'] <= 1
    assert diagnostics['level_argmax_variable'] in prep['variables']
    assert diagnostics['max_relative_difference_adjustment'] >= 0


def test_short_real_initialisation_calibration_records_every_sweep():
    levels = make_weekly_levels(150, seed=1001)
    levels.loc[levels.index[[55, 83, 110]], 'consumer'] = np.nan
    cfg = model.SamplerConfig(
        reps=8,
        burn=4,
        thin=1,
        seed=11,
        progress_every=0,
        dk_projection_mode='record_only',
    )
    result = model.gibbs_bvar_sv_outlier(
        levels,
        p=6,
        frequency='weekly',
        sampler_config=cfg,
    )
    frame = model.dk_projection_diagnostics_frame(result)
    assert len(frame) == 8
    summary = result['dk_projection_diagnostics']
    assert summary['n_sweeps'] == 8
    assert summary['level_error']['max'] >= summary['level_error']['median']
    assert summary['difference_adjustment']['max'] >= 0
    assert 0 <= summary['level_argmax']['top_position_share'] <= 1


def test_strict_mode_keeps_only_compact_projection_summary():
    levels = make_weekly_levels(140, seed=1201)
    levels.loc[levels.index[[55, 83, 110]], 'consumer'] = np.nan
    cfg = model.SamplerConfig(
        reps=8, burn=4, thin=1, seed=13, progress_every=0,
        dk_projection_mode='strict',
        dk_level_relative_gate=1e-6,
        dk_difference_relative_gate=1e-6,
    )
    result = model.gibbs_bvar_sv_outlier(
        levels, p=6, frequency='weekly', sampler_config=cfg,
    )
    assert result['dk_projection_records'] == []
    summary = result['dk_projection_diagnostics']
    assert summary['mode'] == 'strict'
    assert summary['n_sweeps'] == 8
    assert summary['level_error']['max'] >= 0
    assert summary['difference_adjustment']['max'] >= 0
