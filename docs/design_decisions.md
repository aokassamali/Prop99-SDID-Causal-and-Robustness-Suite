- DD-01 Dataset: Use canonical Prop99 smoking panel first; Prop47 follow-on
    - Rationale: clean benchmark + replicability against established example
    - Alternative: start Prop47; rejected for higher preprocessing noise.

- DD-02 Treatment design: single treated unit (CA), treated from 1989
    - Rationale: canonical + best-understood SCM/SDID setting

- DD-03 Outcome scale: per-capita packs (primary), log(packs) (secondary robustness)
    - Rationale: comparability + interpretability; log as sensitivity.

- DD-04 Timing convention: intervention_year=1988, treat_start_year=1989, T0=1988
    - Alternative: start post in 1990 (ramp-up) → include as sensitivity, not baseline.

- DD-05 Meta schema: store treated_unit_id, timing (treat_start_year, T0_year), year range, and hashes
    - Why: makes results auditable and prevents silent spec drift
    - Alternative: keep in code only → rejected (non-auditable)

- DD-006 SCM solver choice: use SciPy constrained optimization (SLSQP) for classic SCM weights
    - Why: standard, lightweight in Python, easy to sanity-check
    - Alternative: CVXPY QP (more robust numerics, heavier dependency)

- DD-007 Placebo statistic: primary = RMSPE ratio; secondary = ATT magnitude
    - Rationale: penalizes designs with poor pre-fit, common in SCM literature.

- DD-08 Time placebo definition: time-placebo evaluation window must end at 1987 (strict pre) to avoid contamination by Prop 99 start.

- DD-09 Time placebo eligibility: require min_pre_periods >= 10 and min_post_periods >= 1 for stability + comparability.

- DD-10 Hyperparameter selection: Chose zeta by a pre-specified criterion 
    - Minimize pre-RMSPE subject to an “interpretability/ stability” constraint like max_omega ≤ 0.4 and eff_donors ≥ 4.
    - Rationale: eta doesn’t matter much; zeta does, so we need a principled pick. (Based on stability sweep)