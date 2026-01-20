# SDID Causal + Robustness Suite (Prop 99 / Cigarette Sales)

A fully reproducible panel causal inference repo comparing **Synthetic Control (SCM)** vs **Synthetic Difference-in-Differences (SDID)** on a canonical policy intervention (Prop 99), with a complete **placebo + robustness** suite.

**Thesis:** Modern SCM-family estimators can produce materially different magnitudes under plausible pre-trend drift. A credible analysis must (i) define the estimand, (ii) report diagnostics, (iii) run falsification tests, and (iv) stress-test analyst choices.

---

## What this repo answers

### Research questions
1) Does SDID change the estimated effect relative to classic SCM? If so, why?  
2) Are conclusions stable to donor pool choice, pre-window definition, and SDID regularization?  
3) Do falsification tests (placebos in space/time) behave as expected?

### Primary estimand
Let unit 1 be treated (California), donors be the other states, and let treatment start in year \(T_0+1\).

**Estimand:** Average post-treatment treatment effect on the treated (ATT):
\[
\text{ATT} = \frac{1}{T - T_0}\sum_{t>T_0}\left(Y_{1t}(1) - Y_{1t}(0)\right)
\]
Operationally, \(Y_{1t}(0)\) is estimated via SCM or SDID.

---

## Data

- **Dataset:** canonical “smoking” panel (packs per capita) used in many SCM tutorials.
- **Unit:** state
- **Outcome:** per-capita cigarette sales (`cigsale`, packs per person)
- **Treated unit:** California (`treated_unit_id = 3`)
- **Intervention timing:** Prop 99 passed in 1988; we treat **1989 as the first post year** (conservative “ramp” convention)
- **Pre-period:** 1970–1988
- **Post-period:** 1989–2000

A deterministic build step produces:
- `data/processed/Y.parquet` (balanced outcome matrix: units × years)
- `data/processed/meta.json` (hashes + panel dimensions + treatment split)

---

## Methods (high-level)

### 1) Synthetic Control (SCM)
Choose donor weights \(w\) (nonnegative, sum to 1) to match the treated unit’s **pre-treatment** trajectory. The estimated counterfactual is the donor-weighted series.

### 2) Synthetic Difference-in-Differences (SDID)
SDID augments SCM with **time weights** over the pre-period and estimates a **weighted DID** effect. Intuition: if the treated unit already diverges in late pre-years, SDID can subtract that baseline gap rather than attributing it to treatment.

A key diagnostic in this run: SDID’s learned pre-period time weights \(\lambda\) concentrate heavily on the **late pre years (1986–1988)**, indicating those years are most representative of the “baseline” relevant for post-period comparison.

---

## Results (baseline)

| Estimator | Effect (packs per capita) | Interpretation |
|---|---:|---|
| SCM (ATT) | **-19.51** | Large post-treatment reduction vs synthetic control |
| SDID (τ)  | **-10.90** | More conservative: subtracts a non-zero late-pre gap |
| Difference (SCM − SDID) | **8.61** | SDID shrinks effect magnitude vs SCM |

---

## Inference via placebo tests (randomization-style)

### In-space placebo (inference)
Treat each control state as if it were treated; compare California’s effect to the placebo distribution.

**SCM**
- Two-sided placebo p-value: **0.0769** (rank 3 / 39)
- One-sided (negative direction) p-value: **0.0513**

**SDID**
- Two-sided placebo p-value: **0.2564** (rank 10 / 39)
- One-sided (negative direction) p-value: **0.1538**

**Interpretation:** SDID is **less extreme** in the placebo distribution than SCM (consistent with SDID being more conservative under late-pre divergence).

### In-time placebo (falsification / no-anticipation)
Fake treatment years entirely in the pre-period (strictly pre-1988) and measure pseudo-effects.

- SCM time-placebo mean: **-4.44** (median **-4.00**)
- SDID time-placebo mean: **-3.02** (median **-2.13**)

**Interpretation:** time placebos are not centered exactly at 0, suggesting **some pre-period drift or imperfect comparability**. SDID reduces (but does not eliminate) this pseudo-effect.

---

## Robustness suite (Script 04)

A single robustness runner generates a unified grid of perturbations across SCM and SDID.

### What we stress-test
1) **Donor pool sensitivity**
   - Leave-one-out donor drops (all donors)
   - Drop top-K donors (based on baseline weights)

2) **Window sensitivity**
   - Pre-window start year: 1970 (baseline), 1975, 1980
   - Post aggregation: full post (1989–2000) vs late-post (1995–2000)

3) **SDID regularization sensitivity**
   - Grid sweep over \((\zeta, \eta)\) (unit-weight and time-weight regularization)

4) **Falsification recap**
   - In-space placebo ranks/p-values
   - Strict-pre time-placebo summary stats

### Robustness summary (headline)
**Sign stability**
- Leave-one-out donor sign stability:
  - SCM: **100%**
  - SDID: **100%**
- No sign flips across window variants (both SCM and SDID)

**Magnitude sensitivity**
- SCM window range: **[-26.60, -19.51]**
- SDID window range: **[-24.38, -10.90]**
- SDID hyperparam range (τ): **[-15.61, -10.67]**

**Weight concentration (baseline)**
- SCM max donor weight: **0.394** (effective donors ≈ **3.77**)
- SDID max donor weight: **0.259** (effective donors ≈ **5.86**)
- SDID max time weight: **0.426** (effective pre-years ≈ **2.79**)

**Takeaway:** Conclusions are **sign-robust** across donors/windows/tuning. Magnitudes vary meaningfully with window definitions and SDID’s unit-regularization \(\zeta\), so the repo reports **ranges** (not just point estimates) and documents the hyperparameter selection policy.

---

## How to reproduce (one command)

This repo is designed to be reproducible end-to-end: build data → estimate SCM/SDID → run placebo inference → run robustness suite → write tables/figures.

Typical workflow:
1) Create environment and install dependencies
2) Run the pipeline script(s) that generate:
   - `reports/tables/main_results.json`
   - `reports/tables/robustness_grid.csv`
   - `reports/tables/robustness_summary.json`
   - figures under `reports/figures/`

---

## Threats to validity (what a reviewer will ask)

1) **Pre-period drift / imperfect comparability:**  
   Strict-pre time placebos show mild negative pseudo-effects (especially for SCM), suggesting the treated unit may be drifting relative to donors in the 1980s. SDID partially addresses this by reweighting late pre-years.

2) **Window sensitivity:**  
   Effects strengthen when focusing on late post years (1995–2000). This could reflect accumulating policy impact, but it also means magnitude claims should be stated with window context.

3) **Hyperparameter dependence (SDID \(\zeta\)):**  
   SDID is stable across \(\eta\) but more sensitive to \(\zeta\). This repo reports ranges and documents a selection policy.

---

## Design decisions (high signal)
- Treatment begins in 1989 (post), not 1988 (policy enactment year), to avoid contamination from ramp/partial exposure.
- Placebo-in-time windows are bounded strictly before the enactment year to prevent leakage.
- Inference uses in-space placebo ranks, reported both unfiltered and with pre-fit filtering.
- Robustness grid is the source of truth; the README reports both point estimates and sensitivity ranges.

---

## References (conceptual)
- Synthetic Control Methods (Abadie et al.)
- Synthetic Difference-in-Differences (Arkhangelsky et al.)

### Tables
- `main_results.json`  
  Baseline SCM vs SDID effects + placebo p-values + time-placebo recap

- `robustness_grid.csv`  
  Every perturbation run (donor drops, windows, SDID hyperparams, placebos) in one flat table suitable for audit/review

- `robustness_summary.json`  
  Human-readable robustness headlines + pass/fail-style flags
