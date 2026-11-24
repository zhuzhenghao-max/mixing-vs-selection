# Audit checklist

Eight reporting checks for a leave-one-line-out perturbation benchmark.
Each check maps to code in this release. A tick means the output exists and
is reported. It does not certify the choice. The same checklist appears as
Supplementary Note 2 in the manuscript.

- [ ] **Equal-mix baseline.** Report the training-free matched-source mix and
  the target-agnostic mean on the same targets.
  Scripts: `scripts/bench_transfer_nadig.py`, `scripts/bench_transfer.py`,
  `scripts/audit_numbers.py`.

- [ ] **Composition versus selection.** Report the oracle, the best fixed rule,
  the split-half estimate, and the draw count. Score gated and equal mixing on
  shared targets.
  Scripts: `scripts/selection_audit.py`, `scripts/selection_figure.py`.

- [ ] **Centering.** Validate it per dataset and line pair. Do not apply it by
  default.
  Scripts: `scripts/selection_audit.py`, `scripts/make_tables.py`.

- [ ] **Scoring policy.** Name target-coordinate exclusion and tie handling.
  Report paired intervals and one alternative policy.
  Scripts: `scripts/bench_scope_sensitivity.py`,
  `scripts/secondary_metrics.py`, `scripts/bench_transfer_analysis.py`.

- [ ] **Retrieval baseline.** Give any learned model the same targets, pool,
  and scoring policy.
  Scripts: `scripts/bench_transfer.py`, `scripts/bench_transfer_nadig.py`,
  `scripts/bench_scgpt_baseline.py`, `scripts/gears_baseline.py`.

- [ ] **Small calibration set.** Compare selected labels with random labels
  and with the uncalibrated mix. Report both exchange routes.
  Scripts: `scripts/phase1_budget_curve.py`,
  `scripts/phase1_selection_pilot.py`, `scripts/phase2_source_budget.py`,
  `scripts/phase2_exchange_summary.py`.

- [ ] **Pool-specific level.** Resample the realized pool. State any stopping
  threshold as post hoc.
  Scripts: `scripts/regime_summary.py`,
  `scripts/tahoe_cal_ci.py --sensitivity`,
  `scripts/pool_composition_control.py`.

- [ ] **Coherence.** Report within-panel rank summaries with partial controls.
  Do not treat them as a measurement rule.
  Scripts: `scripts/tahoe_coherence.py`, `scripts/tahoe_coherence_fast.py`,
  `scripts/regime_diagnostic.py`, `scripts/tahoe_partial_check.py`.
