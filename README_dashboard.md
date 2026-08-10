# Energy BVAR dashboard — first wired slice

This directory contains the first production dashboard slice: shell/navigation, the
shared selection context, registry-backed selectors and the Forecast screen.

## Placement

Recommended project layout:

```text
bvar-energy/
├─ data/
├─ results/
└─ src/
   ├─ model/
   │  ├─ energy_bvar_pipeline.py
   │  ├─ energy_bvar_aggregate_pipeline.py
   │  ├─ energy_bvar_registry.py
   │  ├─ energy_bvar_display.py
   │  └─ ...
   └─ dashboard/
      ├─ energy_bvar_dashboard.py
      └─ assets/dashboard.css
```

Copy `requirements-dashboard.txt` to the project root or merge it into the existing
requirements file.

## Install

From the project virtual environment:

```powershell
pip install -r requirements-dashboard.txt
```

`pyarrow` is required because the dashboard contract is `display_v1.parquet`.
`dash[diskcache]` initialises the background callback manager that the Estimation and
Structural pages will use.

## Run for development

```powershell
python src/dashboard/energy_bvar_dashboard.py
```

Default address: `http://127.0.0.1:8050`.

## Run as a single-worker service

From `src/dashboard`:

```powershell
waitress-serve --threads=4 --listen=127.0.0.1:8050 energy_bvar_dashboard:server
```

Do not add multiple process workers: the dashboard design deliberately uses one server
worker plus Diskcache child processes for background jobs.

## First-use display materialisation

The registry is scanned at startup. If a selected valid forecast does not yet contain
`display_v1.parquet`, the dashboard materialises it once from the saved forecast store,
then reads the Parquet file. Set:

```powershell
$env:ENERGY_BVAR_AUTO_BUILD_DISPLAY="0"
```

to forbid this and require display artefacts to be built ahead of time.

After first-use materialisation, click **Refresh** so the SQLite registry records
`has_display=1`.

## Optional environment variables

```text
ENERGY_BVAR_PROJECT_ROOT
ENERGY_BVAR_RESULTS_ROOT
ENERGY_BVAR_REGISTRY
ENERGY_BVAR_DASH_CACHE
ENERGY_BVAR_DASH_HOST
ENERGY_BVAR_DASH_PORT
ENERGY_BVAR_DASH_DEBUG
ENERGY_BVAR_AUTO_BUILD_DISPLAY
```

## Callback contract

```text
SQLite registry
   ↓
Vintage → Model → Forecast → Run
   ↓
ctx-store
   ↓
ONE display_v1.parquet read
   ↓
data-store
   ├─ metric / series controls
   ├─ forecast figure
   └─ values table + headline cards
```

Changing 68% / 90% / both, the selected metric, or the selected series does **not**
reopen any `.npz` file or Parquet file. Those callbacks operate on `data-store` only.
`uirevision` is keyed to model/series/metric so switching fan widths preserves the
user's zoom and pan.

The Aggregate, Scenarios, Structural and Estimation routes are already reserved in the
shell and share the same context. They are intentionally placeholders in this first
slice; the next slices can be wired without changing the selection contract.
