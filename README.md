# Separating mixing from selection

A leave-one-line-out benchmark and an audit protocol for cross-cell-line perturbation evaluation.

Zhenghao Zhu and Di Cui, Heze University, Heze, Shandong, China

Release 1.3.2 (24 November 2026). Concept DOI:
https://doi.org/10.5281/zenodo.24223291.

## What this project is

Predicting how a cell line responds to a genetic or chemical perturbation
(for example, knocking down a gene or applying a drug) is a central problem
in computational biology. A common shortcut is to predict a new line's
response by averaging the same perturbations measured in other lines. This
project asks two questions: how far does that shortcut actually go, and when
an evaluation reports that a method beats it, is the gain real?

The benchmark holds out one cell line at a time. A query is a perturbation
measured both in the held-out line and in at least one source line (another
line in the panel). Predictions are built from the source lines only and
scored against the held-out line with the perturbation discrimination score
(PDS): the normalized rank of the true response among all candidates after
removing the target gene's own coordinate. PDS 1.0 means the true response
ranks first; 0.5 is chance. A separate experiment asks what a small
target-line label budget adds to the training-free mix.

The design separates two things that are easy to confuse. *Mixing* is how
source responses are combined: equal weights, learned weights, or a gate.
*Selection* is which perturbations or targets enter training or scoring. The
benchmark uses leave-one-line-out prediction, cross-fitted evaluation, and a
per-target best-of audit so that selection effects are measured rather than
mistaken for prediction quality.

| Panel | Design | Size |
|---|---|---|
| Tier A | four harmonized lines (K562, RPE1, HepG2, Jurkat), leave one out | 5,486 queries |
| Tier B | five heterogeneous lines (H1, K562, RPE1, HCT116, HEK293T), leave one out | 18,678 queries |
| Drug panel | seven and twelve Tahoe-100M cancer lines | 648 / 646 shared conditions |

Tier A lines were processed with a common pipeline. Tier B lines come from
different studies and measurement platforms, which makes the transfer harder.

## Key findings

- **Equal mixing is hard to beat.** A training-free equal mix of matched
  source responses beat a target-agnostic mean (the average response over all
  perturbations, which ignores the query) by 0.087 to 0.329 PDS in all nine
  holdouts. Twelve reweighting and aggregation variants improved on it by at
  most 0.008.
- **A gate can look better without mixing better.** Gating by self-knockdown
  raised PDS, but the gain came from which targets it selected. On shared
  targets the gate-versus-equal difference was at most 0.014, while
  per-target best-of reporting raised PDS by 0.10 to 0.13 in the harmonized
  tier and 0.06 to 0.11 in the heterogeneous tier; selection accounted for
  0.03 to 0.08 of the harmonized gain and almost all of the heterogeneous gain.
- **Learned baselines do not close the gap.** GEARS, trained
  leave-one-line-out, scored 0.163 to 0.193 below equal mixing in the four
  Tier A folds, a result reproduced with three random seeds. A stored scGPT
  run scored 0.190 below equal mixing on RPE1 and is reported in
  Supplementary Fig. S1 and Table S4.
- **The metric is coarse and pool-dependent.** Gene-shuffled truth still
  scored 0.655 in both tiers, and truth plus noise at four times the response
  norm scored 1.000. Across seven reference-pool compositions, equal-mix PDS
  ranged from 0.438 to 0.996 on a balanced 638-query panel, and method ranks
  flipped with composition. A PDS level is a property of the method and the
  pool together.
- **An independent H1 truth table supports the tier level.** Replacing the
  decoded H1 cache with a DESeq2 pseudobulk table for the 150 official
  training targets moved the pooled equal-mix score by 0.007 or less. Paired
  per-target agreement was weak (Spearman +0.18), and a Hallmark profile
  analysis kept a positive signal over mismatched targets (+0.080, 95% CI
  +0.004 to +0.157).
- **The drug-panel ceiling survives panel expansion.** On twelve
  Tahoe-100M lines (646 shared conditions), equal-mix PDS was 0.964 to 0.997
  (pooled 0.990). Adding five lines moved the original seven by -0.006 to
  +0.010. Calibration stayed below equal mixing at every budget under both
  calibration rules (k-means: -0.048 to -0.0045; random: -0.046 to -0.0040;
  all eight panel intervals below zero), and a 50-condition pilot flagged
  the near-ceiling regime in 99.87% of draws.
- **Coherence is a within-panel diagnostic.** Source-response coherence (how
  well source lines agree with each other) ordered perturbations within a
  panel, but did not rank panels. Pooled rank correlation with retrieval was
  +0.34 across the genetic holdouts and +0.30 across the drug panel. In the
  drug panel, equal mixing was already near the ceiling because the
  candidate pool was weak, not because the mix was unusually accurate; a
  50-condition pilot separated the two regimes.

## What is included

The repository contains the analysis code, the derived summary tables and
the per-target scores behind them, the manuscript source, the figure files,
and the audit scripts. Raw data are not redistributed.

```
scripts/              analysis code (Python; run from the repository root)
src/                  shared modules imported by some scripts
configs/              evaluation configs used during development
data/                 gene sets for the GEARS graph and gene-name axes
outputs/              derived summary tables and per-target arrays
outputs/tables/       main and supplementary table source data
outputs/seed_runs/    three-seed GEARS summary and per-target scores
figures/              manuscript figures (PNG and PDF), including the graphical abstract
manuscript/draft.tex  manuscript source (compile with pdfLaTeX)
manuscript/key_points.md  five key points for the submission
MANIFEST.md           file inventory, data provenance, Zenodo file list
REPRODUCE.md          step-by-step reproduction guide
AUDIT_CHECKLIST.md    eight reporting checks mapped to scripts
LICENSE, LICENSE_DATA code (MIT) and derived data/figures (CC BY 4.0)
```

## Quick start

Python 3.12 or newer is recommended. Install the core dependencies with

```
pip install -r requirements.txt
```

The lightweight checks run from the summary data included here:

```
python scripts/audit_numbers.py
python scripts/check_refs.py
python scripts/check_tables.py
python scripts/check_tex.py
```

`audit_numbers.py` recomputes every headline number in the abstract and
supplements from the stored outputs (14 checks). The other three scripts
check the manuscript's citations, tables, and TeX structure. The Tier B
arrays are included, so the audit runs offline.

## Full reproduction

Starting from the public raw data, follow `REPRODUCE.md`. The heavy per-target
arrays are listed in `MANIFEST.md`; download them into `outputs/` with the
same file names before running `secondary_metrics.py` or any GEARS per-target
rescoring. `CHECKSUMS.sha256` covers every file in this release.

## Data and licenses

Raw data come from the original studies: the harmonized four-line atlas of
Nadig et al. (2025), the genome-wide Perturb-seq screens of Replogle et al.
(2022), X-Atlas/Orion, the 2025 Virtual Cell Challenge H1 training split, and
Tahoe-100M. Dataset identifiers and citations are in `MANIFEST.md`. No
challenge test data were used; the H1 responses are the released training
split.

Derived data and per-target scores are released here and at the Zenodo
archive. Code is released under the MIT license; derived data and figures are
released under CC BY 4.0. Raw third-party data remain under their original
licenses.

## Scope and caveats

- The learned baselines are single configurations (scGPT fine-tuned on K562;
  GEARS with one training setting), not exhaustive hyperparameter searches.
- The drug panel measures chemical perturbations. It has no single target
  coordinate per condition, so those scores use plain cosine PDS without
  target-gene exclusion.
- The twelve-line panel is a robustness extension, not a survey of cell
  types or tissues.
- H1 enters Tier B as an scVI-decoded response cache, and the original model
  and production records are unavailable. An independent DESeq2 table for
  150 H1 targets is included as a truth check and moved the pooled equal-mix
  score by less than 0.01. The scGPT comparison depends on the cache and is
  reported in Supplementary Fig. S1.
- Tahoe coherence and drug-exclusion summaries use a corrected scorer. The
  true score is read from the matching reference-matrix entry and ties use
  the midrank, so PDS is bounded at 1. Rebuilding the underlying per-line
  matrices still requires the public Tahoe shards.

## Citation

If you use the benchmark, please cite the manuscript and the Zenodo archive.
The concept DOI https://doi.org/10.5281/zenodo.24223291 always resolves to the
latest release. Version 1.3.0 is archived at
https://doi.org/10.5281/zenodo.24410000 and version 1.2.1 at
https://doi.org/10.5281/zenodo.24356413; version 1.3.2 shares the concept
DOI. `CITATION.cff` contains the machine-readable record, including the
repository URL and DOI.

## Contact

Zhenghao Zhu, 2727565120@qq.com.
