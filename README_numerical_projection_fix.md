# DK data-augmentation numerical projection fix

This patch changes only the interior-missing-data branch of `energy_bvar_model.py`.

## Why

For long weekly VARs (notably n=3, p=24), the Durbin-Koopman simulation smoother can reproduce exact published level observations with floating-point discrepancies around 1e-8 even though `obs_cov=0`. The former absolute `1e-8` assertion could therefore reject an otherwise valid draw.

## Fix

- Keep `obs_cov=0`; no nugget / measurement noise is introduced.
- Use a scale-aware numerical guard (`atol=1e-7`, column-wise `rtol=1e-10` equivalent).
- Project published cells back onto the exact observed levels after the smoother draw.
- Re-derive current differences from the projected level path.
- Rebuild the terminal companion state from those corrected differences so forecasts start from the same path used in the VAR regression.
- Keep an additional guard that rejects any material change between the DK difference state and the re-derived difference path.

The balanced/no-data-augmentation Gibbs path is untouched and was checked seed-for-seed against the pre-patch file: posterior arrays are byte-identical in the smoke test.

## Tests run

`python -m pytest tests/test_weekly_dk_data_augmentation.py -q`

Result: `5 passed`.

A long weekly p=24 smoke test with 650 weeks and 12 WOB-like interior gaps also completed successfully.
