# Energy BVAR dashboard patch v3

Replace these project files:

- `src/model/energy_bvar_pipeline.py`
- `src/model/energy_bvar_display.py`
- `src/model/energy_bvar_io.py`
- `src/dashboard/energy_bvar_dashboard.py`
- `src/dashboard/assets/dashboard.css`

## What this patch fixes

1. Weekly panel dispatch now calls `load_weekly_fuel_panel(dataset, model)` using the current adapter signature. The dashboard error `unexpected keyword argument 'variables'` is removed.
2. Petrol, diesel and liquid-fuels pipeline specs use `missing_data_method='dk'`, matching the current weekly notebooks.
3. Gas/electricity constructed-price units are resolved from the processed vintage `manifest.json` instead of being hard-coded.
4. New saved runs persist a compact `history.parquet`/`history.csv` beside the run. `display_v1` therefore no longer needs `data/processed` or a model adapter for new runs.
5. Legacy runs remain supported once through a processed-data fallback. After `display_v1.parquet` is materialised, normal dashboard interactions are filesystem-light again.
6. The dashboard styling is replaced by a modern light product UI with a horizontal product header, compact context toolbar, rounded cards, softer surfaces and a clearer empty state.

## Run

From the project root:

```powershell
python src/dashboard/energy_bvar_dashboard.py
```

Then hard refresh the browser (`Ctrl+F5`).

Existing forecast stores do not need to be re-estimated. The first load of an old run can use the legacy processed-data fallback to create `display_v1.parquet`.
