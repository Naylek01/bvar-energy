# Tax-assumption revision

Replace the corresponding files under `src/model/`, put `run_review_tests.py`
at the project root, and replace the seven notebooks under `notebooks/`.

The tax-enabled notebooks now contain a separate **Tax assumptions and inflation
impact** section after the usual forecasts. The section shows:

- historical VAT/excise from the source file;
- the tax path actually applied by the model;
- future baseline and scenario paths;
- price-level scenario effects;
- YoY/52-week inflation effects with posterior uncertainty;
- draw-wise timing/architecture assertions.

For gas/electricity, raw semiannual Eurostat markers are overlaid on the monthly
expanded path. For weekly fuels, WOB tax history is already at model frequency.
Heat energy and solid fuels keep the same section heading but explicitly state
that no separate VAT/excise re-attribution exists because those models forecast
the HICP target directly.

The monthly single-workbook v9 source-context compatibility is retained via
`energy_bvar_source_context.py`.

Run from the project root:

```powershell
python run_review_tests.py
```

Optional full DK calibration:

```powershell
python run_review_tests.py --dk-calibration
```

Restart the notebook kernel after replacing Python modules.
