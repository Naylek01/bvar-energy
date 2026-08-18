# Excel / Power Query Data Pipeline

This document describes the **operational data-ingestion workbook** used by the Euro Area Inflation BVAR project.

The workbook is the interface between external data providers and the Python modelling stack. It is responsible for refreshing, normalising and snapshotting raw source data. Econometric transformations, vintage construction, BVAR estimation, HICP aggregation and dashboard analysis are performed in Python.

> **Operational rule:** update the workbook first, save it, then run the Python data builder.

> **Refresh duration:** a complete refresh of the workbook queries takes approximately **1 hour** under normal conditions. The exact duration depends on network access, source response times and the machine running Excel. Do not interrupt Excel or launch the Python builder before all required queries have finished and the workbook has been saved.

---

## 1. Role of the workbook

The workbook combines several source types:

- Bloomberg market data;
- Haver data;
- European Commission Weekly Oil Bulletin data;
- Eurostat Energy data;
- Eurostat HICP and HICP weights;
- Eurostat VAT / country-weight data;
- World Bank data;
- workbook metadata and configuration.

The general architecture is:

```text
External sources
      ↓
Bloomberg / Haver / Power Query
      ↓
RAW queries / live sheets
      ↓
VIEW queries / values-only snapshots
      ↓
Excel output sheets
      ↓
Save workbook
      ↓
Python dataset builder
      ↓
data/processed/YYYYMMDD/
      ↓
BVAR estimation / aggregation / dashboard
```

The workbook is therefore an **ingestion and source-control layer**, not a modelling layer.

---

## 2. Main workbook sheets

The final workbook contains the following operational sheets.

| Sheet | Purpose | Update mechanism | Used by Python |
|---|---|---|---|
| `Config` | Selects Web or Local mode and local-backup location | Manual + VBA button | Indirectly |
| `Haver` | Haver source data / snapshot | Haver refresh shortcut | Yes |
| `Bloomberg_live` | Live Bloomberg formulas | Bloomberg terminal/add-in | No |
| `Bloomberg_copy` | Values-only Bloomberg snapshot | VBA copy button | Yes |
| `European_Commission` | Clean Weekly Oil Bulletin output | Power Query | Yes |
| `World_Bank` | World Bank source / snapshot | Query or saved source contract | Yes |
| `Eurostat` | Eurostat Energy output | Power Query | Yes |
| `Eurostat_VAT_Weights` | VAT / country-weight data | Power Query | Yes |
| `EUROSTAT_HEADLINE_HICP_VIEW` | Headline HICP source view | Power Query | Yes |
| `Eurostat_HICP` | HICP indices / annual weights used downstream | Power Query | Yes |
| `Metadata` | Source and contract metadata | Workbook / pipeline | Audit |

The exact sheet names are part of the Python ingestion contract and should not be renamed casually.

---

## 3. Power Query design

The workbook follows a RAW → VIEW pattern whenever appropriate:

```text
External source
      ↓
*_RAW query
      ↓
validation / filtering / normalisation
      ↓
*_VIEW query
      ↓
Excel output table
      ↓
Python
```

Important queries used during development include:

```text
EC_WOB_RAW_FULL
EC_WOB_VIEW

WORLD_BANK_GAS_RAW

EUROSTAT_ENERGY_RAW
EUROSTAT_ENERGY_VIEW

EUROSTAT_HICP_RAW

EUROSTAT_VAT_COUNTRY_WEIGHTS_RAW

EUROSTAT_HEADLINE_HICP_VIEW
```

RAW queries should normally remain **connection-only** when their intermediate output is not required in a worksheet.

The Python layer should read the final workbook tables / views, not internal Power Query staging tables.

---

# 4. Normal workbook update procedure

This is the standard update procedure before creating a new Python vintage.

## Step 1 — Update Bloomberg on a Bloomberg-enabled PC

Move or open the workbook on a PC with a working Bloomberg installation and Bloomberg Excel Add-In.

Open:

```text
Bloomberg_live
```

Allow the Bloomberg formulas to refresh and verify that the latest observations have populated.

Temporary formula errors such as:

```text
#NAME?
#N/A
```

can appear when Bloomberg is unavailable or formulas are still resolving. Do **not** use `Bloomberg_live` directly as the Python source.

Once the live data are current, press the workbook VBA button:

```text
Update "Bloomberg_copy"
```

The button copies the current Bloomberg data into the values-only sheet:

```text
Bloomberg_live
      ↓
VBA button
      ↓
Bloomberg_copy
```

`Bloomberg_copy` is the stable Bloomberg interface consumed downstream.

### Check before continuing

Verify that:

- the latest expected date is present;
- the main Bloomberg series contain numeric values;
- `Bloomberg_copy` has been updated;
- the workbook has not been left with stale values from an earlier date.

---

## Step 2 — Update Haver

Open the:

```text
Haver
```

sheet.

Use:

```text
Ctrl + D
```

to update the Haver data according to the workbook's Haver workflow.

After the refresh, verify that the expected series and recent dates are populated.

Do not rename or reorganise the Haver columns unless the corresponding Python ingestion contract is deliberately updated.

---

## Step 3 — Configure the source mode

Open:

```text
Config
```

The operational parameters are:

| Parameter | Meaning |
|---|---|
| `Mode` | `web` = live APIs / online sources; `local` = saved local source files |
| `BackupFolder` | Root folder containing saved local source snapshots |
| `BackupVintage` | Snapshot/vintage to use in Local mode |

The workbook displays the active mode at the top of the sheet.

### Normal case: Web mode

Set:

```text
Mode = web
```

In Web mode, the Power Queries use the live external sources configured in the workbook.

Use this mode for the normal production refresh when the external sources are available.

### Fallback case: Local mode

If a live source or API is unavailable, set:

```text
Mode = local
```

Then set:

```text
BackupFolder
```

to the **correct folder containing the saved local source files**, and set:

```text
BackupVintage
```

to the backup snapshot that should be used.

Example conceptually:

```text
Mode          local
BackupFolder  C:\...\local_source_backups
BackupVintage 13/08/2026
```

The exact path is machine-specific. The important point is that `BackupFolder` must point to the root that actually contains the saved source snapshot corresponding to `BackupVintage`.

Local mode is a recovery / reproducibility mechanism. It should not silently point to an arbitrary or stale backup.

---

## Step 4 — Apply the configuration and refresh the queries

Once `Mode`, `BackupFolder` and `BackupVintage` are correct, press the VBA button:

```text
ApplyConfigandRefreshQuery
```

The button applies the selected configuration and refreshes the workbook queries.

Conceptually:

```text
Config
  ├─ Mode
  ├─ BackupFolder
  └─ BackupVintage
        ↓
ApplyConfigandRefreshQuery
        ↓
Power Query refresh
        ↓
final workbook source sheets updated
```

Wait until the refresh has fully completed before saving the workbook or launching Python. A full query refresh is operationally expected to take approximately **1 hour**.

---

# 5. Web mode vs Local mode

## Web mode

Use:

```text
Mode = web
```

when the external sources are available.

The workbook retrieves current data through its configured web/API connections.

Advantages:

- latest available observations;
- normal production workflow;
- no dependence on manually selected historical snapshots.

Risks:

- provider outage;
- changed upstream workbook/API structure;
- changed Eurostat contract;
- network/privacy/Power Query errors.

---

## Local mode

Use:

```text
Mode = local
```

when Web mode fails or when an exact saved source state must be reproduced.

Local mode reads previously saved source files from:

```text
BackupFolder + BackupVintage
```

Advantages:

- reproducible source state;
- recovery when an external API is unavailable;
- protection against temporary upstream changes.

Risks:

- selecting the wrong backup path;
- selecting the wrong vintage;
- using stale source files unintentionally.

Always verify both fields before pressing:

```text
ApplyConfigandRefreshQuery
```

---

# 6. Bloomberg contract

`Bloomberg_live` and `Bloomberg_copy` have deliberately different roles.

```text
Bloomberg_live
    = live formulas
    = Bloomberg-dependent
    = NOT the stable Python interface

Bloomberg_copy
    = values-only snapshot
    = stable after workbook save
    = consumed downstream
```

The update sequence must therefore remain:

```text
Bloomberg-enabled PC
      ↓
refresh Bloomberg_live
      ↓
check latest values
      ↓
press Update "Bloomberg_copy"
      ↓
Bloomberg_copy refreshed
```

Do not make Python depend directly on unresolved Bloomberg formulas.

---

# 7. European Commission Weekly Oil Bulletin

The European Commission source is used for weekly petroleum-product information and tax-related inputs.

The intended pattern is:

```text
EC_WOB_RAW_FULL
      ↓
EC_WOB_VIEW
      ↓
European_Commission
```

The robust design uses the historical workbook source and validates the expected sheet structure.

Important operational rules:

- do not reintroduce dynamic URL discovery casually;
- do not rely on fixed Excel row offsets when the query already detects structure;
- preserve the distinction between prices with and without taxes;
- preserve the ragged edge instead of deleting otherwise valid observations;
- investigate source-layout changes rather than weakening guards.

If the WOB endpoint changes, verify the official source and workbook structure before editing the query.

---

# 8. Eurostat contract

Eurostat is used for Energy data, HICP indices, HICP weights and VAT/country-weight information.

The most important rule is:

> **Do not guess an SDMX key, dimension order, dimension name, code or unit.**

The robust ingestion design downloads the relevant dataset and filters locally rather than constructing unverified filtered SDMX keys.

In particular:

- do not hard-code an unverified dimension order;
- preserve codes that have already been validated in the workbook;
- read the dataset structure from the returned data;
- treat a changed contract as a source change to investigate, not something to bypass automatically.

This rule exists because a syntactically plausible but incorrect SDMX key can either fail with a `400 Bad Request` or, worse, retrieve the wrong economic series.

---

# 9. World Bank

The World Bank source provides the historical natural-gas proxy used by the data pipeline.

Its role is primarily historical / backcast support rather than the principal current market-price source.

Whether the workbook accesses it through the active query path or through a saved source snapshot, preserve:

- the expected series definition;
- the expected units;
- the date column;
- the output column names consumed by Python.

Do not silently substitute another World Bank commodity series merely to make a refresh succeed.

---

# 10. Metadata and source lineage

The `Metadata` sheet exists to make the workbook source state auditable.

Useful information includes:

- active source mode;
- source update dates;
- backup vintage when Local mode is used;
- source contracts;
- workbook/source version information;
- Bloomberg snapshot update time;
- Haver update state;
- Eurostat contract information.

Metadata are audit information. They are not BVAR explanatory variables.

---

# 11. Final checks before saving

Before saving the workbook, verify:

- `Bloomberg_copy` reflects the newly refreshed `Bloomberg_live`;
- Haver has been updated;
- `Config` contains the intended `web` or `local` mode;
- if `local`, `BackupFolder` and `BackupVintage` are correct;
- `ApplyConfigandRefreshQuery` completed successfully;
- no required Power Query is in an error state;
- final source sheets contain plausible recent dates;
- no required sheet name or table contract has been changed.

Then:

```text
SAVE THE WORKBOOK
```

Python should only be launched after the workbook has been saved.

---

# 12. Python handoff

The workbook ends its responsibility at the saved source tables.

The next stage is Python:

```text
saved workbook
      ↓
Python dataset builder
      ↓
data/processed/YYYYMMDD/
      ↓
Energy / Headline model estimation
      ↓
aggregation
      ↓
dashboard
```

The exact builder command may evolve with the repository. Use the current production builder under:

```text
src/data_pipeline/
```

rather than an archived command copied from an old README.

A processed vintage is the reproducible modelling input. The workbook itself is a living source workbook and should not be confused with the dated processed vintages.

---

# 13. If the refresh fails

Use the following decision sequence.

```text
REFRESH FAILS
│
├─ Bloomberg?
│   ├─ confirm Bloomberg Terminal / Add-In is available
│   ├─ wait for Bloomberg_live formulas
│   └─ rerun Update "Bloomberg_copy"
│
├─ Haver?
│   ├─ confirm Haver access
│   └─ rerun Ctrl + D
│
├─ Power Query in Web mode?
│   ├─ identify the failing source/query
│   ├─ do NOT weaken source-contract guards blindly
│   └─ if the failure is temporary, use validated Local mode
│
└─ Local mode?
    ├─ check BackupFolder
    ├─ check BackupVintage
    ├─ confirm the snapshot exists
    └─ rerun ApplyConfigandRefreshQuery
```

If Web mode fails because an upstream provider has genuinely changed its contract, Local mode can keep the project operational while the live query is investigated.

---

# 14. Rules that must not be broken

1. **Update Bloomberg on a Bloomberg-enabled PC before copying the snapshot.**
2. **Python must use `Bloomberg_copy`, not unresolved `Bloomberg_live` formulas.**
3. **Update Haver before the Power Query refresh is considered complete.**
4. **Choose `web` for live sources and `local` only with the correct backup path and vintage.**
5. **Press `ApplyConfigandRefreshQuery` after changing the Config sheet.**
6. **Wait for all required queries to finish before saving.**
7. **Save the workbook before running Python.**
8. **Do not rename workbook sheets or output columns without updating the Python contract.**
9. **Do not invent Eurostat SDMX dimension orders, names, codes or units.**
10. **Do not weaken a source-contract guard merely to make a refresh pass.**
11. **Use Local mode as a controlled fallback, not as an unnoticed stale-data mode.**
12. **Keep econometric logic out of Excel; modelling belongs in Python.**

---

# 15. Quick production checklist

For routine use, the complete workflow is:

```text
1. Open / send workbook to Bloomberg-enabled PC
2. Refresh Bloomberg_live
3. Press: Update "Bloomberg_copy"
4. Open Haver sheet
5. Press: Ctrl + D
6. Open Config
7. Set Mode:
      web
   or local + correct BackupFolder + BackupVintage
8. Press: ApplyConfigandRefreshQuery
9. Wait for all queries to finish
10. Check latest dates / errors
11. Save workbook
12. Run the current Python dataset builder
13. Validate the new processed vintage
```

That is the canonical workbook update procedure for the final project.
