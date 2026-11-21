# Key points

- A leave-one-line-out benchmark holds out a whole cell line and pairs every
  prediction with the same targets, reference pool, and scoring rule. The
  retained panels contain 5,486 queries over four harmonized lines and 18,678
  over five heterogeneous lines.

- A training-free equal mix of matched source responses beat the
  target-agnostic mean in all nine holdouts, by 0.087 to 0.329 PDS, and beat a
  GEARS adaptation by 0.16 to 0.19 PDS in every Tier A fold.

- Per-target oracle gains overstate what a single rule delivers. Split-half
  evaluation retained 57% of the oracle gain in the harmonized tier and 4% in
  the heterogeneous tier, so oracle and out-of-sample estimates belong in
  separate columns.

- Pool composition moved the equal-mix level from 0.438 to 0.996 PDS, and
  scoring policy could reverse secondary effects. Target composition, pool
  size, and exclusion rules need to be reported next to the score.

- Source-response coherence and pool geometry are reported as descriptive
  within-panel summaries. Coherence correlated with retrieval at Spearman
  +0.34 across genetic holdouts and +0.30 across the drug panel, but the
  diagnostic does not establish a mechanism.

Corresponding figure: figures/graphical_abstract.pdf and .png. Panels show
equal-mix gains over every baseline (A) and the split-half decomposition of
per-target oracle gains by tier (B).
