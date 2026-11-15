# Separating mixing from selection

A leave-one-line-out benchmark for cross-cell-line perturbation prediction.

Zhenghao Zhu, Heze University, Heze, Shandong, China

Release 1.1.0 (15 November 2026). Concept DOI:
https://doi.org/10.5281/zenodo.24223291.

## What this project is

Predicting how a cell line responds to a genetic or chemical perturbation
(for example, knocking down a gene or applying a drug) is a central problem
in computational biology. A common shortcut is to predict a new line's
response by averaging the same perturbations measured in other lines. This project asks two questions: how far does that shortcut actually
go, and when an evaluation reports that a method beats it, is the gain real?

The benchmark holds out one cell line at a time. A query is a perturbation
measured both in the held-out line and in at least one source line (another
line in the panel). Predictions are built from the source lines only and
scored against the
held-out line with the perturbation discrimination score (PDS), the
normalized rank of the true response among all candidates after removing the
target gene's own coordinate. PDS 1.0 means the true response ranks first;
0.5 is chance. A separate experiment asks what a small target-line label
budget adds to the training-free mix.

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
  per-target best-of reporting inflated PDS by 0.03 to 0.12.
- **Learned baselines do not close the gap.** scGPT, fine-tuned on K562 and
  tested on RPE1, scored 0.190 mean PDS below equal mixing. GEARS, trained
  leave-one-line-out, scored 0.163 to 0.193 below it in the four Tier A
  folds, a result reproduced with three random seeds.
- **The drug-panel ceiling survives panel expansion.** On twelve
  Tahoe-100M lines (646 shared conditions), equal-mix PDS was 0.964 to 0.997
  (pooled 0.990). Adding five lines moved the original seven by -0.006 to
  +0.010. Calibration stayed below equal mixing at every budget under both
  selection rules (k-means: -0.048 at k=20 to -0.0045 at k=200; all eight
  panel intervals below zero), and a 50-condition pilot flagged the regime
  in 0.13% of draws.
- **Coherence is a within-panel diagnostic.** Source-response coherence (how
  well source lines agree with each other) ordered conditions within a
  panel, but did not rank panels. In the drug panel, equal mixing was
  already near the ceiling because the candidate pool was weak, not because
  the mix was unusually accurate; a 50-condition pilot separated the two
  regimes.

## What is included

The repository contains the analysis code, the derived summary tables and
per-target scores that the tables are built from, the manuscript source, the
figure files, and the audit scripts. Raw data are not redistributed.

```
scripts/              analysis code (Python; run from the repository root)
src/                  shared modules imported by some scripts
configs/              evaluation configs used during development
data/                 gene sets for the GEARS graph and gene-name axes
outputs/              derived summary tables and per-target arrays
outputs/tables/       main and supplementary table source data
outputs/seed_runs/    three-seed GEARS summary and per-target scores
figures/              manuscript figures (PNG and PDF)
manuscript/draft.tex  manuscript source (compile with pdfLaTeX)
MANIFEST.md           file inventory, data provenance, Zenodo file list
REPRODUCE.md          step-by-step reproduction guide
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

`audit_numbers.py` recomputes every headline number in the abstract from the
stored outputs (11 checks). The other three scripts check the manuscript's
citations, tables, and TeX structure. The Tier B arrays are included, so the
audit runs offline.

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

## Citation

If you use the benchmark, please cite the manuscript and the Zenodo archive.
The concept DOI https://doi.org/10.5281/zenodo.24223291 always resolves to the
latest release; release 1.1.0 is archived at
https://doi.org/10.5281/zenodo.24247539. `CITATION.cff` contains the
machine-readable record, including the repository URL and DOI.

## Contact

Zhenghao Zhu, 2727565120@qq.com.
