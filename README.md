# Separating mixing from selection

A leave-one-line-out benchmark for cross-cell-line perturbation prediction.

Zhenghao Zhu, Heze University, Heze, Shandong, China.

This repository accompanies the manuscript *Separating mixing from selection:
a leave-one-line-out benchmark reveals systematic bias in cross-cell-line
perturbation evaluation*. It contains the analysis code, the derived summary
tables, the per-target scores that the tables are built from, and the figure
files. Raw public datasets are not redistributed; they are available from the
original publications listed in `MANIFEST.md`.

## The benchmark in one paragraph

For each held-out cell line, a query is a perturbation measured both in that
line and in at least one other line. The prediction is built from the source
lines only, and it is scored against the held-out line's measured responses
with the perturbation discrimination score (PDS): the normalized rank of the
true response in a retrieval pool, computed after removing the target
coordinate. Tier A holds out each of four harmonized lines (K562, RPE1, HepG2,
Jurkat; 5,486 queries). Tier B holds out each of five heterogeneous lines (H1,
K562, RPE1, HCT116, HEK293T; 18,678 queries). The benchmark supports two
questions: how far training-free mixing goes, and what a small target-line
label budget adds.

## Key results

- A training-free equal mix beat a target-agnostic mean by 0.087 to 0.329 PDS
  in all nine holdouts. A dozen reweighting and aggregation variants improved
  on it by at most 0.008.
- Gating by self-knockdown appeared to help, but the gain came from the
  selected targets. Per-target best-of reporting inflated PDS by 0.03 to 0.12;
  cross-fitting kept only mean centering in the harmonized tier.
- scGPT, fine-tuned on K562 and tested on RPE1, scored 0.190 mean PDS below
  equal mixing. GEARS, trained leave-one-line-out, scored 0.163 to 0.193 below
  it in the four Tier A folds, a result reproduced under three random seeds.
- Source-response coherence ordered targets within a panel, but did not rank
  panels. In a seven-line drug panel, equal mixing was already near the
  ceiling, and a 50-condition pilot separated the two regimes.

## Repository layout

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

The largest per-target arrays (the GEARS predictions and the raw row caches)
are archived on Zenodo. The Tier B arrays are included here so the headline
audit runs offline. See `MANIFEST.md` for the file list and checksums.

## Quick start

Python 3.12 or newer is recommended. Install the core dependencies with

```
pip install -r requirements.txt
```

The lightweight checks run with the summary data included here:

```
python scripts/audit_numbers.py
python scripts/check_refs.py
python scripts/check_tables.py
python scripts/check_tex.py
```

`audit_numbers.py` recomputes every headline number in the abstract from the
stored outputs. The other three scripts check the manuscript's citations,
tables, and TeX structure.

Full reproduction, starting from the public raw data, is described in
`REPRODUCE.md`. The heavy arrays are listed in `MANIFEST.md`; download them
into `outputs/` with the same file names before running `secondary_metrics.py`
or any GEARS per-target rescoring.

## Data

Raw data come from the original studies: the harmonized four-line atlas of
Nadig et al. (2025), the genome-wide Perturb-seq screens of Replogle et al.
(2022), X-Atlas/Orion, the 2025 Virtual Cell Challenge H1 training split, and
Tahoe-100M. Dataset identifiers and citations are in `MANIFEST.md`. No
challenge test data were used. The H1 responses are the released training
split.

Derived data and per-target scores are released with this repository and at
the Zenodo archive (DOI to be inserted after deposit). Code is released under
the MIT license; derived data and figures are released under CC BY 4.0. Raw
third-party data remain under their original licenses.

## Citation

If you use the benchmark, please cite the manuscript and the Zenodo archive.
`CITATION.cff` contains the machine-readable record; add the repository URL
and DOI after they are assigned.

## Contact

Zhenghao Zhu, 2727565120@qq.com.
