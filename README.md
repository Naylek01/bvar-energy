# Euro Area Inflation BVAR Dashboard

A production-oriented Bayesian VAR framework for **euro-area inflation forecasting, scenario analysis and structural analysis**, with a unified Dash interface for:

- **HICP Energy**
- **Headline HICP**
- **Core HICP**
- component-level forecasts and diagnostics
- conditional paths and tax scenarios
- IRFs, FEVDs and historical decompositions
- persisted, vintage-aware model results

The project combines seven Energy BVARs with a separate joint monthly Headline BVAR and exact draw-by-draw HICP aggregation.

> **Project status:** final research / production snapshot, October 2026.

---

## Final dashboard contract

The final dashboard snapshot follows these presentation and runtime contracts:

- **Energy, Headline HICP and Core HICP** share the same section-first navigation for Forecast, Scenarios and Estimation; Structural analysis is available for Energy and Headline, while **Core Structural is intentionally unavailable**.
- Structural **display horizons are presentation-only**. The default Structural display horizon is **6 months**; changing it does not change the stored computational horizon or re-estimate a BVAR.
- Energy and regular Headline joint-BVAR FEVD charts show the **complete shock decomposition**, including the response variable's own shock as well as cross-shocks.
- In scenario path charts, **Baseline / Unconditional paths are solid** and **Scenario / Conditional paths are dashed**.
- Joint conditional-plus-tax Energy scenarios materialise the current **HICP Energy** result independently; Headline propagation is an additional downstream calculation rather than a prerequisite for the Energy result.
- A condition expressed as **0% vs last observed** means holding the conditioned driver at its last observed level over the selected conditioning window. It is therefore not, in general, identical to the unconditional forecast path.

---

## 1. Project architecture

### Energy

Seven component models are estimated separately:

| Model ID | Component |
|---|---|
| `gas` | Natural gas |
| `electricity` | Electricity |
| `heat_energy` | Heat energy |
| `solid_fuels` | Solid fuels |
| `liquid_fuels` | Liquid fuels |
| `car_fuels_petrol` | Petrol |
| `car_fuels_diesel` | Diesel |

Monthly components are modelled at monthly frequency. Petrol, diesel and liquid fuels use weekly models and are converted to the monthly HICP aggregation frequency downstream.

The seven component paths are aggregated draw by draw into **HICP Energy**.

### Headline HICP

Headline inflation is built from four major blocks:

- Energy
- Food
- NEIG
- Services

The monthly joint BVAR models the non-Energy Headline system, while HICP Energy is supplied by the Energy suite.

### Core HICP

Core corresponds to the non-Energy, non-Food part of the basket and is constructed from:

- NEIG
- Services

---

## 2. BVAR framework

A generic VAR representation is

```math
y_t
=
c
+
A_1 y_{t-1}
+
A_2 y_{t-2}
+
\cdots
+
A_p y_{t-p}
+
B x_t
+
\varepsilon_t,
```

with

```math
\varepsilon_t \sim \mathcal{N}(0,\Sigma_t).
```

The production framework supports Minnesota-style shrinkage together with stochastic-volatility and outlier-robust specifications used by the component models.

The lag order is model-specific. In the current production design:

- monthly Energy models typically use $p=12$;
- weekly fuel models typically use $p=24$;
- the joint Headline BVAR uses $p=12$.

The joint Headline system also uses monthly seasonal dummies, with December as the reference month.

Missing observations can be handled with either:

- **Linear interpolation** — the faster operational option. It fills missing observations before estimation and is therefore an approximation to the full missing-data treatment.
- **Durbin-Koopman (DK)** state-space augmentation / smoothing — the more exact missing-data treatment used inside the Bayesian estimation, but materially more computationally expensive.

The method is configurable by model. Linear is useful when turnaround time matters; DK should be preferred when the more exact treatment of missing observations is required.

---

## 3. HICP transformation and inflation

For an HICP index $I_t$, year-on-year inflation is

```math
\pi_t^{YoY}
=
100
\left(
\frac{I_t}{I_{t-12}} - 1
\right).
```

For weekly series the analogous 52-week transformation is

```math
\pi_t^{52w}
=
100
\left(
\frac{I_t}{I_{t-52}} - 1
\right).
```

Model targets and source variables may be transformed internally, including log differences. Inflation, levels and contributions shown in the dashboard are reconstructed downstream from the saved posterior paths.

---

## 4. Draw-by-draw HICP Energy aggregation

Let $i=1,\ldots,N_E$ index the Energy components and let $w_{i,y}$ denote their annual HICP weights.

Weights are normalised inside the Energy basket:

```math
\widetilde{w}_{i,y}
=
\frac{w_{i,y}}
{\sum_{j=1}^{N_E} w_{j,y}}.
```

For a month $t$ in calendar year $y$, the chain-linked Energy index is reconstructed relative to December of the previous year:

```math
I^{E}_t
=
I^{E}_{Dec(y-1)}
\sum_{i=1}^{N_E}
\widetilde{w}_{i,y}
\frac{I_{i,t}}
{I_{i,Dec(y-1)}}.
```

The recursion is performed **for every posterior draw**. This preserves nonlinear aggregation and posterior dependence across components.

The December re-linking means that January dynamics can reflect both the new within-year link and year-on-year base effects.

---

## 5. Headline HICP aggregation

Let the four Headline blocks be

```math
j \in
\{
Energy,\ Food,\ NEIG,\ Services
\}.
```

With annual normalised weights $\widetilde{w}_{j,y}$, Headline is reconstructed using the same December-based chain-linking principle:

```math
I^{H}_t
=
I^{H}_{Dec(y-1)}
\sum_j
\widetilde{w}_{j,y}
\frac{I_{j,t}}
{I_{j,Dec(y-1)}}.
```

All aggregation is performed **draw by draw**, before posterior means, medians or credible intervals are calculated.

This matters because, in general,

```math
f\!\left(E[X]\right)
\neq
E[f(X)].
```

The dashboard therefore does not aggregate already-summarised component forecasts.

---

## 6. Exact contribution accounting

Headline YoY contribution decomposition is constructed so that the component contributions add exactly to total Headline inflation, up to numerical precision:

```math
\pi^{H}_t
=
\sum_j c_{j,t}.
```

For a scenario comparison between baseline $A$ and scenario $B$, the component contribution effect is

```math
\Delta c_{j,t}
=
c^{B}_{j,t}
-
c^{A}_{j,t}.
```

The exact Headline scenario effect therefore satisfies

```math
\Delta \pi^{H}_t
=
\pi^{H,B}_t
-
\pi^{H,A}_t
=
\sum_j \Delta c_{j,t}.
```

This is different from simply adding component IRFs or marginal scenario statistics.

---

## 7. Forecasts and uncertainty

Forecast distributions are generated from posterior draws and future shocks.

For a VAR moving-average representation

```math
y_{t+h}
=
\mu_{t+h}
+
\sum_{s=0}^{h-1}
\Psi_s \varepsilon_{t+h-s},
```

the conditional forecast variance contains accumulated future-shock uncertainty:

```math
Var(y_{t+h}\mid\theta)
=
\sum_{s=0}^{h-1}
\Psi_s
\Sigma
\Psi_s'.
```

This is why forecast fan charts generally widen with horizon.

The dashboard reports posterior summaries such as:

- mean;
- median;
- 68% interval;
- 90% interval.

Observed, nowcast and forecast segments are kept distinct in exported results.

---

## 8. Conditional forecasts

The framework supports hard conditioning over arbitrary future windows.

If a variable is constrained only over horizons

```math
h \in [h_s,h_e],
```

then observations before $h_s$ and after $h_e$ are **not hard constrained**, but they may still move because the conditional forecast is solved jointly.

For example, conditioning only from $M+3$ to $M+5$ means:

- $M+1$ and $M+2$: unconstrained, but may move;
- $M+3$ to $M+5$: hard-conditioned;
- $M+6$ onward: unconstrained, but may move.

For delayed percentage scenarios relative to the last observed value $x_T$,

```math
x_{T+h}^{cond}
=
x_T(1+\delta),
\qquad
h_s \le h \le h_e.
```

---

## 9. Tax scenarios

VAT and excise scenarios are handled downstream of the BVAR where the model architecture supports explicit tax re-attribution.

For a pure tax scenario, the underlying pre-tax BVAR target is unchanged by construction:

```math
\Delta y_t^{BVAR}=0.
```

The consumer-price effect is produced by tax re-attribution and then propagated through the same draw-by-draw HICP Energy aggregation.

Conditional paths and tax assumptions can also be combined in a **Joint Energy scenario**.

The joint distribution is calculated before HICP Energy aggregation; marginal effects are **not** simply added.

---

## 10. Energy-to-Headline scenario propagation

Energy scenarios are propagated into Headline using paired conditional forecasts.

Define:

- $A$: Headline conditioned on the **Energy baseline** path;
- $B$: Headline conditioned on the **Energy scenario** path.

The Headline scenario effect is calculated draw by draw:

```math
\Delta \pi^{H,(d)}_{t}
=
\pi^{H,B,(d)}_{t}
-
\pi^{H,A,(d)}_{t}.
```

Posterior summaries are computed only after the paired difference has been formed.

For Energy-only shocks, the Core impact is zero by construction because the Core definition excludes Energy.

---

## 11. Structural analysis

The dashboard supports structural analysis from persisted posterior draws without re-estimating the BVAR.

Available objects include:

- impulse response functions (IRFs);
- forecast error variance decompositions (FEVDs);
- historical decompositions (HDs).

The validated baseline identification is recursive / Cholesky.

If

```math
\Sigma = PP',
```

and $\Psi_h$ is the reduced-form moving-average coefficient at horizon $h$, the structural IRF is

```math
\Theta_h
=
\Psi_h P.
```

For response variable $i$ to structural shock $j$,

```math
IRF_{i,j}(h)
=
e_i'\Theta_h e_j.
```

Unlike a forecast fan chart, an IRF does **not** accumulate future random innovations. Its posterior band reflects uncertainty in estimated parameters and, where relevant, identification.

A standard recursive FEVD share for variable $i$, shock $j$ and horizon $H$ is

```math
FEVD_{i \leftarrow j}(H)
=
\frac{
\sum_{h=0}^{H-1}
\left(
e_i'\Psi_h P e_j
\right)^2
}{
\sum_{h=0}^{H-1}
e_i'\Psi_h
\Sigma
\Psi_h'
e_i
}.
```

---

## 12. Nonlinear Headline Total structural response

Headline Total structural responses are **not** obtained by summing component IRFs.

For component $i$, horizon $h$ and posterior draw $d$, the shocked component index is reconstructed from the deterministic baseline path and cumulative log IRF:

```math
I^{shock}_{i,h,d}
=
I^{base}_{i,h,d}
\exp
\left(
\sum_{s=0}^{h}
IRF_{i,s,d}
\right).
```

The baseline and shocked component paths are then both passed through the production Headline aggregation.

The Headline Total level response is

```math
R^{level}_{h,d}
=
100
\left(
\frac{
I^{H,shock}_{h,d}
}{
I^{H,base}_{h,d}
}
-1
\right).
```

The Headline Total YoY response is

```math
R^{YoY}_{h,d}
=
\pi^{H,shock}_{h,d}
-
\pi^{H,base}_{h,d}.
```

This is an exact nonlinear draw-by-draw re-aggregation.

### Important structural contract

Headline Total currently exposes the nonlinear IRF object, but **not** an aggregated FEVD or HD.

The reason is that Headline is additive in index levels while the BVAR is estimated in transformed component space. Effective shares are state-dependent and covariance terms matter. Therefore, in general,

```math
FEVD^{Headline}
\neq
\sum_j FEVD_j.
```

An aggregated Headline FEVD or HD would be a separate econometric object requiring its own definition and validation.

---

## 13. Dashboard

The main application is:

```text
src/dashboard/energy_bvar_dashboard.py
```

Run it from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe .\src\dashboard\energy_bvar_dashboard.py
```

The default development server is:

```text
http://127.0.0.1:8050
```

The product is organised around three domains:

- **Energy**
- **Headline HICP**
- **Core**

and the main analytical sections:

- **Overview**
- **Data**
- **Forecast**
- **Scenarios**
- **Structural**
- **Estimation**

The dashboard uses persisted model artefacts and a registry to avoid unnecessary re-estimation.

Structural display controls are separated from stored computational horizons. In the final UI the default Structural display horizon is **6 months**. Core currently exposes Forecast, Scenarios and Estimation; **Core Structural is intentionally unavailable**.

---

## 14. Result persistence

Runs are vintage-aware and identified by deterministic run IDs.

A typical saved run contains some or all of:

```text
metadata.json
summary.parquet
diagnostics.parquet
draws.npz
history.parquet
display_v1.parquet
```

The dashboard is designed to read compact persisted display artefacts for ordinary navigation and plotting.

Large model outputs under `results/` are local runtime artefacts and should normally not be committed to Git.

---

## 15. Repository structure

```text
bvar-energy/
├─ src/
│  ├─ model/          # BVAR, aggregation, scenarios, structural analysis
│  └─ dashboard/      # Dash application and UI assets
├─ notebooks/         # Research / replication notebooks
├─ tests/             # Regression and contract tests
├─ tools/             # Audits, validation and maintenance utilities
├─ scripts/           # Setup / environment helpers
├─ configs/           # Saved estimation profiles / configuration
├─ config/            # Project configuration
├─ data/              # Local data, mostly ignored by Git
├─ results/           # Persisted local runs, ignored by Git
├─ requirements.txt
└─ README.md
```

---

## 16. Estimation runtime

Production estimation is computationally intensive.

As an operational order of magnitude, an estimation configured with **6,000 MCMC repetitions** can take approximately **3–4 hours**. Runtime depends on the selected model, machine, missing-data method and effective configuration.

The missing-data choice has an important runtime trade-off:

| Method | Runtime | Missing-data treatment |
|---|---|---|
| **Linear** | Faster | Approximate: missing values are interpolated before estimation |
| **Durbin-Koopman (DK)** | Slower | More exact: missing observations are handled through state-space augmentation / smoothing during estimation |

Therefore:

- use **Linear** when a faster production run is needed and the approximation is acceptable;
- use **DK** when the more exact missing-data treatment is preferred and the additional computation time is acceptable.

The dashboard exposes this choice in the Estimation configuration. Changing presentation controls, forecasts already persisted to disk, scenario displays or Structural views does **not** imply re-estimating the BVAR.

## 17. Installation

### Python

### Dependency audit

The final repository-wide static import audit was run against `src/`, `tests/`, `tools/`, `scripts/` and `notebooks/` without importing project code or executing notebooks.

Two direct dependencies that were previously only available transitively are now declared explicitly:

- `Flask` — directly imported by the dashboard XLSX export layer;
- `ipython` — directly imported by an interactive plotting utility and used by notebooks.

One legacy diagnostics notebook imports `pmdarima`. It is **not required by the production dashboard/runtime** and remains an optional notebook-only installation rather than a mandatory production dependency.

Python **3.12** is the final audited environment; the final dependency audit was run on Python **3.12.6**.

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python -c "import numpy, pandas, scipy, dash, flask, plotly, pyarrow, diskcache, IPython; print('environment: OK')"
```

---

## 18. Optional Haver dependency

Haver is proprietary and desk-specific, so it is intentionally **not** included in `requirements.txt`.

Install/configure it separately in the same `.venv` only when Haver extraction is required.

The model and dashboard layers can operate from already-built processed datasets and persisted model artefacts without requiring Haver at ordinary runtime.

---

## 19. Excel / Power Query ingestion

The Excel / Power Query ingestion layer is intentionally **not documented in this README**.

It includes external-source ingestion, workbook configuration, Power Query/M logic and source-specific contracts.

That layer is documented separately in a dedicated file:

```text
EXCEL_DATA_PIPELINE.md
```

---

## 20. Reproducibility principles

The project follows several strict engineering rules:

1. **Vintage-aware data and results.**
2. **Deterministic run identity from effective configuration.**
3. **Posterior draws persisted for downstream analysis.**
4. **Draw-by-draw aggregation before posterior summaries.**
5. **No silent addition of marginal scenario effects.**
6. **No model re-estimation for ordinary dashboard presentation changes.**
7. **Structural and conditional results preserve explicit source lineage.**
8. **Observed / nowcast / forecast segments remain distinct.**
9. **UI display horizons are separate from stored computational horizons.**
10. **Large local data and result artefacts are excluded from Git.**

---
## License / data access

No licence is asserted here for third-party datasets.

Users are responsible for complying with the terms of the underlying data providers and any proprietary data systems used during ingestion.
