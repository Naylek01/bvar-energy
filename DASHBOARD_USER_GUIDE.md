# Dashboard User Guide

## Euro Area Inflation BVAR Dashboard

This guide is the operational companion to `README.md` and `EXCEL_DATA_PIPELINE.md`.

- `README.md` explains the modelling architecture and econometric contracts.
- `EXCEL_DATA_PIPELINE.md` explains the Excel / Power Query source layer.
- **This file explains how to operate the dashboard, refresh the data, refit the production models, validate a new vintage and use the analytical pages.**

The production workflow is deliberately separated into two layers:

```text
Excel / external-source refresh
        ↓
processed vintage
        ↓
BVAR estimation and persisted model artefacts
        ↓
HICP Energy aggregation
        ↓
Headline estimation
        ↓
registry / promoted production state
        ↓
dashboard display, scenarios and structural analysis
```

Ordinary dashboard navigation, display-horizon changes, scenario display and structural visualisation **do not re-estimate the BVARs**.

---

## 1. Starting the dashboard

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\src\dashboard\energy_bvar_dashboard.py
```

The default local address is:

```text
http://127.0.0.1:8050
```

The application is organised around:

- **Overview**
- **Data**
- **Energy**
- **Headline HICP**
- **Core**

and the analytical sections:

- **Forecast**
- **Scenarios**
- **Structural**
- **Estimation**

Core does not currently expose a Structural page.

---

## 2. Two different operating modes

It is important to distinguish normal dashboard use from a production refresh.

### Normal use

Normal use includes:

- opening the dashboard;
- navigating between Energy, Headline and Core;
- changing chart controls;
- changing a Structural display horizon;
- running conditional or tax scenarios from already-saved posterior draws;
- exporting charts or tables.

These actions use persisted artefacts and do **not** require model estimation.

### Production refresh

A production refresh is required when the source data have been updated and a new modelling vintage must be created.

That workflow is:

```text
1. Refresh Excel
2. Build a new processed vintage
3. Validate the processed vintage
4. Fit the seven Energy models
5. Build HICP Energy
6. Fit the Headline model
7. Promote / register the selected runs
8. Refresh the dashboard snapshot
9. Perform a short smoke test
```

Do not mix vintages across these steps.

---

# 3. Refreshing the data and refitting the models

## 3.1 Refresh the production Excel workbook

The Excel workbook is the upstream interface between the external providers and Python.

The production workbook is normally:

```text
bvar_energy_raw_data.xlsm
```

with `.xlsx` also supported by the Python builder.

Follow the detailed source-specific procedure in:

```text
EXCEL_DATA_PIPELINE.md
```

At a minimum, the workbook must be fully refreshed and saved before Python creates a new processed vintage.

The production builder itself does **not** execute:

- Excel VBA;
- Power Query;
- Haver;
- Bloomberg formulas.

Therefore complete the workbook refresh in Excel first.

The current source contract includes, where applicable:

- Haver refresh;
- Bloomberg refresh;
- the `Bloomberg_Live` block;
- copying the final Bloomberg values into the persisted `Bloomberg_copy` output;
- Power Query refreshes;
- Eurostat / HICP / VAT-weight outputs;
- the European Commission / Weekly Oil Bulletin outputs.

A complete workbook refresh can take approximately one hour. Do not start the Python dataset build while Excel is still refreshing or saving the production workbook.

### Operator rule

**Excel first, Python second.**

Never treat a partially refreshed workbook as a new modelling vintage.

---

## 3.2 Create the processed vintage

After the refreshed workbook has been saved, use the dashboard **Data** workflow to materialise the shared processed vintage.

The current production builder is:

```text
src/data_pipeline/build_dataset_headline_joint_v14_FINAL.py
```

The builder writes directly to:

```text
data/processed/<vintage>/
```

It does not require the old multi-stage `data/interim` pipeline.

The dashboard Data page is the preferred operator interface because it:

- resolves the production workbook;
- creates the shared processed vintage;
- exposes the resulting data-readiness state;
- shows the validation artefacts already written for that vintage;
- does not modify saved BVAR results when the dataset itself is being built.

If the workbook is kept in the canonical `data/raw` location, the production code can auto-detect the supported workbook names.

### Important

A **processed vintage** is the immutable modelling input.

The living Excel workbook is **not** the modelling vintage.

Once a processed vintage has been created, all Energy and Headline estimations for that production round should use that same vintage.

---

## 3.3 Validate the processed vintage before fitting anything

Do not start a multi-hour estimation immediately after building the data.

First use the **Data** page to confirm that the new vintage is complete and internally coherent.

The dashboard exposes read-only summaries of the validation artefacts under:

```text
data/processed/<vintage>/
```

The checks include the model-dataset and HICP validation layers used by the production code.

Before estimation, verify at least:

- the intended new vintage is selected;
- the expected Energy model datasets exist;
- the Headline inputs exist;
- the HICP indices and aggregation sidecars exist;
- there is no reported HICP validation failure requiring investigation;
- annual weight / identity diagnostics are acceptable;
- dates and source coverage correspond to the intended data refresh.

If the new processed vintage is not complete, **stop here**. Do not fit models against a partially built vintage.

---

## 3.4 Fit the seven Energy models

Go to:

```text
Energy → Estimation
```

The production Energy suite consists of:

1. `gas`
2. `electricity`
3. `heat_energy`
4. `solid_fuels`
5. `liquid_fuels`
6. `car_fuels_petrol`
7. `car_fuels_diesel`

The production orchestration resolves one **common vintage** across the component models. This is intentional: separate components must not silently use different latest dates.

### Recommended production procedure

1. Select the newly created processed vintage.
2. Load the intended saved estimation profile, or confirm the effective production configuration shown on the Estimation page.
3. Confirm the missing-data method.
4. Confirm the sampler settings, including repetitions, burn-in, thinning and seed.
5. Run the **full Energy suite** for the selected vintage.
6. Promote/register the accepted runs when the production configuration is intended to become active.
7. Wait until all seven models have completed successfully.

Saved estimation profiles live under:

```text
configs/estimation/
```

The complete effective configuration is also persisted in the run metadata. A saved profile is therefore a convenience for reproducible operator input; the persisted run metadata remains the authoritative record of what was actually estimated.

### Linear vs Durbin–Koopman

The dashboard supports both missing-data approaches.

- **Linear**: faster operational approximation.
- **Durbin–Koopman (DK)**: more exact missing-data treatment, materially slower.

Use the method required by the production run specification. Do not switch methods casually between vintages, because it changes the effective estimation configuration and therefore the run identity.

### Runtime

Production estimation is computationally intensive. A long production run can take several hours depending on the machine and the selected sampler configuration.

Do not navigate away assuming that a long calculation has failed merely because it is still running.

---

## 3.5 Build HICP Energy

Once the seven Energy models have valid saved forecasts for the same vintage, build the production Energy aggregate from:

```text
Energy → Estimation
```

using the **Build Energy aggregate** action.

This step:

- uses the seven saved component forecast stores;
- applies the component-specific HICP transformations;
- applies the production admissibility rules;
- pairs the independently estimated component posterior draws according to the production pairing contract;
- reconstructs HICP Energy draw by draw;
- writes the aggregate result under `results/hicp_energy_aggregate/...`;
- materialises the compact display artefact used by the dashboard;
- can promote the aggregate in the registry.

The former HICP Energy aggregate notebook is now a **reference / audit notebook**, not the preferred production operator path.

### Optional CLI aggregate entry point

The repository also contains a dedicated aggregate runner:

```powershell
.\.venv\Scripts\python.exe .\src\model\run_hicp_energy_aggregate.py --vintage YYYYMMDD
```

If `--vintage` is omitted, that script resolves the latest complete common vintage.

For routine production operation, prefer the dashboard Estimation workflow unless there is a specific reason to run the aggregate from the terminal.

---

## 3.6 Fit the Headline model

After the Energy production state is available, go to:

```text
Headline HICP → Estimation
```

Select the **same processed vintage** used for the Energy production round.

The current Headline production model is the locked joint monthly BVAR used by the Headline pipeline.

The Estimation page exposes the effective prior and sampler controls and calls the production Headline pipeline with persisted results.

The production callback also supports registration / promotion of the completed Headline run.

### Recommended procedure

1. Select the same vintage used for the Energy suite.
2. Confirm the production prior / sampler settings.
3. Confirm the missing-data method.
4. Run the Headline estimation.
5. Wait for the status to report completion.
6. Promote/register the accepted Headline run if it is the production run.
7. Confirm that the persisted unconditional Headline forecast exists.

Do not manually replace the production Headline workflow with one of the historical research notebooks.

The notebooks remain useful for research, diagnostics and replication, but the dashboard Estimation path is the production operator interface.

---

## 3.7 Core does not require a separate fit

Core is constructed from the saved Headline component paths:

```text
NEIG + Services
```

draw by draw.

There is therefore no separate Core BVAR estimation step in the production refresh.

Core reads the persisted Headline outputs and aggregates the relevant components.

---

## 3.8 Refresh the dashboard production snapshot

The dashboard intentionally maintains a stable registry/result snapshot during normal navigation.

After creating and promoting a new production vintage, explicitly refresh the dashboard data / registry snapshot so that the application sees the new production state.

Do not expect ordinary page navigation by itself to invalidate and rebuild the production snapshot.

---

# 4. Production release gate

A new vintage should not be considered production-ready merely because the samplers finished.

Before using it operationally, verify the complete chain.

## Energy

Confirm:

- all seven Energy models exist for the intended vintage;
- the selected production runs are valid;
- forecast stores exist;
- the component HICP products exist where required;
- the HICP Energy aggregate exists;
- the aggregate is built from the intended component runs;
- the aggregate validation reports no blocking error.

Structural views require the persisted full posterior draw store (`draws.npz`).

## Headline

Confirm:

- the Headline run is for the same processed vintage;
- its unconditional forecast is present;
- `headline_draws.npz` and Headline metadata are available;
- the run diagnostics are acceptable;
- the promoted run is the intended one.

## Cross-domain

Confirm:

- Energy and Headline use the same processed vintage;
- the Overview resolves the intended production state;
- no page reports a run/vintage mismatch.

Only after this gate should the new vintage be treated as the production dashboard state.

---

# 5. Short post-refit smoke test

After a production refresh, perform a short operational smoke test rather than re-running the models.

Recommended sequence:

1. **Overview**
   - production vintage is correct;
   - headline indicators render;
   - Energy contribution / forecast content renders.

2. **Data**
   - expected vintage is selected;
   - Energy readiness is complete;
   - Energy aggregate is available;
   - Headline is available.

3. **Energy → Forecast**
   - open at least one component forecast;
   - open HICP Energy.

4. **Energy → Scenarios**
   - run one simple conditional scenario;
   - confirm Paths and Impact render.

5. **Energy → Structural**
   - load one IRF or FEVD;
   - confirm no re-estimation is launched.

6. **Headline → Forecast**
   - total and component forecast views render.

7. **Headline → Scenarios**
   - run one Energy-to-Headline propagation if required.

8. **Core**
   - forecast and scenario pages render from the persisted Headline paths.

A display problem should be diagnosed as a display/runtime issue first. Do not re-estimate the BVAR simply because a chart is stale or blank.

---

# 6. Forecast pages

Forecast pages read persisted forecast artefacts.

Typical uses:

- select a model / aggregate;
- inspect levels or inflation;
- change the display horizon;
- inspect credible intervals;
- inspect contributions and model history;
- export the displayed result.

The dashboard distinguishes observed history from predictive paths.

Changing a display horizon does not change the stored model horizon and does not refit the model.

---

# 7. Scenario pages

## Energy scenarios

Energy supports:

- conditional observable / commodity paths;
- tax scenarios where the model contract exposes explicit VAT / excise re-attribution;
- joint conditional-plus-tax scenarios.

### Conditional paths

A hard condition fixes the chosen driver over the selected future window.

A value of:

```text
0% vs last observed
```

means that the future driver is held at its **last observed level** over the selected conditioning window.

It does **not** mean "equal to the unconditional forecast".

Therefore a zero-percent condition can still produce a non-zero HICP Energy effect.

## Tax scenarios

Tax scenarios are applied downstream of the pre-tax BVAR for the components whose model architecture supports explicit tax re-attribution.

The tax scenario does not imply that the pre-tax BVAR target itself changes.

## Joint Energy scenario

When conditional and tax assumptions are both active, they are combined before the final HICP Energy aggregation.

Their marginal effects should not be added manually.

## Visual convention

In path figures:

- **Baseline / Unconditional** = solid line
- **Scenario / Conditional** = dashed line

---

# 8. Energy-to-Headline propagation

Energy and Headline scenario calculations are separate stages.

The Energy scenario result is materialised first.

Headline propagation is then calculated from the paired Energy baseline / scenario paths.

The Headline calculation should not be required merely to create the HICP Energy scenario result.

For cross-domain propagation, the Energy and Headline vintages must match.

---

# 9. Structural analysis

Structural pages consume persisted posterior draws. They do not re-estimate the BVAR.

The dashboard exposes:

- IRF;
- FEVD;
- historical decomposition where supported.

## Display horizon versus computation horizon

The **display horizon** is a presentation control.

The final default Structural display horizon is:

```text
6 months
```

Changing it only slices an already-computed object for display.

It does not change the stored computation horizon and does not trigger BVAR estimation.

## FEVD

Energy and the regular Headline joint-BVAR FEVD charts show the **complete shock decomposition**, including:

- the selected response variable's own shock;
- cross-shocks.

## Headline Total structural object

Headline Total is a nonlinear draw-by-draw aggregation object.

The current production contract exposes the nonlinear aggregate IRF.

An aggregated Headline Total FEVD or HD is **not** currently defined as a validated production object.

---

# 10. Estimation status and reproducibility

Each estimation is tied to:

- model ID;
- processed vintage;
- effective prior configuration;
- effective sampler configuration;
- missing-data method;
- seed;
- run ID.

Changing the effective configuration changes the run identity.

For reproducible production operation:

- prefer saved estimation profiles;
- record the processed vintage;
- retain the persisted metadata;
- promote only the accepted run;
- do not overwrite an older production run merely to make the UI point to a different result.

The registry and persisted run directories are part of the reproducibility contract.

---

# 11. Reading CURRENT / STALE / COMPUTING / INVALID

Where these states are displayed:

- **CURRENT** — the displayed result matches the current effective inputs.
- **STALE** — an input changed after the result was computed; recomputation may be required.
- **COMPUTING** — the relevant operation is running.
- **INVALID** — the current input combination cannot produce a valid result.

A STALE display result is not evidence that the underlying BVAR must be re-estimated.

Check whether the stale object is a scenario, structural result or other downstream calculation before launching an expensive estimation.

---

# 12. Exports

The dashboard supports table / result export, including `.xlsx` for supported views.

Exports should be treated as representations of the selected persisted vintage and run.

Where available, exported files include the vintage context in the filename or metadata.

---

# 13. What not to do

For production operation:

- do not run individual historical notebooks as a substitute for the production Estimation workflow;
- do not mix Energy models from different processed vintages;
- do not build HICP Energy while the processed dataset is still being rebuilt;
- do not assume a saved posterior alone is enough for production if required forecast / HICP / display artefacts are missing;
- do not re-estimate models to fix a presentation-only issue;
- do not manually add marginal scenario effects when the production pipeline provides a joint draw-wise calculation;
- do not promote a run before checking its diagnostics and vintage.

---

# 14. Recommended monthly / data-release runbook

For each new production data vintage:

```text
[ ] 1. Open and refresh the canonical Excel workbook
[ ] 2. Wait for Haver / Bloomberg / Power Query refreshes to finish
[ ] 3. Save the workbook
[ ] 4. Build the new processed vintage from the Data page
[ ] 5. Review processed-vintage validation / readiness
[ ] 6. Select the exact new vintage in Energy → Estimation
[ ] 7. Run the seven Energy models
[ ] 8. Review Energy completion / diagnostics
[ ] 9. Build HICP Energy
[ ] 10. Promote/register accepted Energy runs + aggregate
[ ] 11. Select the same vintage in Headline → Estimation
[ ] 12. Run the Headline production model
[ ] 13. Review Headline completion / diagnostics
[ ] 14. Promote/register the accepted Headline run
[ ] 15. Explicitly refresh the dashboard data/registry snapshot
[ ] 16. Verify Overview and Data readiness
[ ] 17. Run the short Forecast / Scenario / Structural smoke test
[ ] 18. Treat the new vintage as production only after all checks pass
```

---

# 15. Troubleshooting

## The new vintage does not appear

Check:

1. the workbook was saved;
2. the processed vintage was actually materialised;
3. the expected files exist under `data/processed/<vintage>/`;
4. the dashboard snapshot was explicitly refreshed.

## Energy aggregate is unavailable

Check:

1. all seven Energy component forecasts exist for the same vintage;
2. the intended component runs are registered / promoted;
3. build the aggregate from the Estimation page.

Do not use the old aggregate notebook as the primary production repair path.

## Structural page cannot load

Structural analysis requires persisted posterior draws.

If the selected run was saved without the required `draws.npz`, select or re-run an estimation configuration that retains the full posterior required by Structural.

## Headline scenario reports a vintage mismatch

Energy and Headline scenario propagation is same-vintage only.

Select or re-run both domains on the same processed vintage.

## Dashboard chart looks stale after a new production run

Refresh the dashboard snapshot / registry state first.

Do not immediately re-estimate the BVAR.

---

# 16. Related documentation

- `README.md` — model architecture, econometric definitions and installation.
- `EXCEL_DATA_PIPELINE.md` — source refresh, workbook configuration and Power Query workflow.
- `requirements.txt` — Python environment.
- `configs/estimation/` — saved estimation profiles.
- `configs/production_manifest.json` — repository-versioned production-state contract where applicable.

