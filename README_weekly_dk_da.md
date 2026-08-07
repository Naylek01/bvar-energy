# Weekly diesel + liquid-fuels DK data-augmentation update

## Replace

Copy into `src/model/`:

- `energy_bvar_model.py`
- `energy_bvar_weekly_fuels.py`
- `energy_bvar_io.py`

Add into `notebooks/`:

- `08_car_fuels_diesel_weekly_bvar_sv_outlier.ipynb`
- `09_liquid_fuels_weekly_bvar_sv_outlier.ipynb`

Optional test:

- `tests/test_weekly_dk_data_augmentation.py`

`energy_bvar_plot.py` is not changed by this delivery.

## Statistical change

The validated balanced monthly path is preserved. Interior missing weekly WOB levels are no longer rejected or deterministically interpolated. When interior gaps are detected, each Gibbs sweep begins by drawing the complete latent level/difference path from the conditional Gaussian state-space distribution using the existing augmented `_durbin_koopman_level_draw` smoother.

The Gibbs ordering is:

1. draw interior missing levels / differences by DK conditional on current `B, A, h, O`;
2. rebuild `Y, X` from that completed draw;
3. draw `B`;
4. draw constant Cholesky `A`;
5. draw stochastic-volatility paths and vol-of-vol parameters;
6. draw outlier scales / indicators and outlier probabilities.

Released levels are exact observations (`obs_cov = 0`); missing levels are `NaN` observations and remain stochastic. No special down-weighting is applied to imputed weeks.

## Partial final Bloomberg aggregates

`energy_bvar_weekly_fuels.load_weekly_fuel_panel()` now reads `aggregation_coverage.csv`. Builder-flagged incomplete final aggregates linked to the selected panel are masked to `NaN` only at the model-adapter boundary. The processed dataset and `partial_period_inputs.csv` are not altered. This is the temporary policy until the positive-`obs_cov` partial-period measurement model is implemented.

## Posterior state saved for coherent forecasts

Data-augmented runs additionally retain:

- `completed_differences_draws`
- `missing_level_draws`
- `missing_level_positions`
- `last_companion_state_draws`

Forecasts use the terminal companion state from the *same posterior draw* rather than one fixed state.

## Tests run

- Python syntax compilation of all changed modules and all notebook code cells.
- 3/3 unit tests:
  - interior missing levels are stochastically augmented and forecastable;
  - `p=24` companion dimensions are correct;
  - partial-period masking occurs in the adapter only.
- Balanced-path non-regression check against the previously validated engine: retained `B`, `A`, SV, outlier, spectral-radius and log-likelihood arrays were exactly equal for the same seed on a no-missing sample.
- End-to-end execution of both notebooks in `ENERGY_BVAR_FAST_TEST=1` on a synthetic W-MON vintage with `p=24`, WOB interior gaps, ragged edge, tax re-attribution, scenarios, IRF, FEVD, historical decomposition and final assertions. Both notebooks ended with:

  `All weekly calendar, interior-DK, ragged-edge and conditioning checks passed.`

## Runtime

Exact data augmentation invokes a simulation smoother every Gibbs sweep. It is therefore materially slower than the balanced model. A synthetic 1,128-week, 3-variable, VAR(24) benchmark in the test container took roughly 1.6 seconds per Gibbs sweep. A full 6,000-sweep run can consequently take hours. The notebooks keep the research configuration at 6,000 / 3,000 and only reduce it when `ENERGY_BVAR_FAST_TEST=1`.
