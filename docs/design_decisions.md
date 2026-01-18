- DD-001 Dataset: Use canonical Prop99 smoking panel first; Prop47 follow-on
    - Rationale: clean benchmark + replicability against established example
    - Alternative: start Prop47; rejected for higher preprocessing noise.

- DD-002 Treatment design: single treated unit (CA), treated from 1989
    - Rationale: canonical + best-understood SCM/SDID setting

- DD-003 Outcome scale: per-capita packs (primary), log(packs) (secondary robustness)
    - Rationale: comparability + interpretability; log as sensitivity.

- DD-004 Timing convention: intervention_year=1988, treat_start_year=1989, T0=1988
    - Alternative: start post in 1990 (ramp-up) → include as sensitivity, not baseline.

- DD-005 Meta schema: store treated_unit_id, timing (treat_start_year, T0_year), year range, and hashes
    - Why: makes results auditable and prevents silent spec drift
    - Alternative: keep in code only → rejected (non-auditable)

- DD-006 SCM solver choice: use SciPy constrained optimization (SLSQP) for classic SCM weights
    - Why: standard, lightweight in Python, easy to sanity-check
    - Alternative: CVXPY QP (more robust numerics, heavier dependency)