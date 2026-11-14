# Reproduction guide

All commands are run from the repository root. Paths in the scripts are
relative to that root, and outputs are written under `outputs/`. The benchmark
and the audits are CPU-only; the scGPT fine-tuning and GEARS training used a
single GPU, and GEARS also runs on CPU with a small batch size.

## 1. Environment

Python 3.12 or newer. The core dependencies are pinned in
`requirements.txt`:

```
pip install -r requirements.txt
```

The released scGPT and GEARS code are external. Use the versions referenced in
the manuscript: scGPT (`scGPT_human` checkpoint) and GEARS as released by
Roohani et al. Put the external repositories on `PYTHONPATH` or install them
in the environment. `gears_baseline.py` looks for the GEARS checkout at
`GEARS_PATH` (default `external/GEARS`), and the scGPT scripts read the
checkpoint from `models/scgpt/scGPT_human`. The H1 evaluation uses cell-eval2
version 0.18.0
(commit 0ce25cf) only as a convention reference; the benchmark computes PDS
directly and does not call cell-eval2.

## 2. Raw data

Raw data are not included. Download them from the original sources listed in
`MANIFEST.md` and place them under `data/`. The helper scripts
`scripts/gwps_download.sh`, `scripts/rpe1_download.sh`,
`scripts/perturbatlas_dl_panel.py`, and `scripts/parallel_download.py`
automate the public downloads where direct access is available. H1 responses
come from the 2025 Virtual Cell Challenge training split; no challenge test
data are used.

## 3. Build response tables

```
python scripts/build_nadig_rows.py          # Tier A rows, outputs/nadig_rows.npz
python scripts/rpe1_pseudobulk.py           # RPE1 pseudobulk rows
python scripts/gwps_pseudobulk.py           # K562/RPE1 GWPS pseudobulk rows
python scripts/tahoe_build_line.py          # Tahoe-100M per-line responses
```

The Tier B gene axis and evaluation mask are built by the corresponding
scripts under `scripts/`; check each script's defaults before rerunning on a
different data release. The 18,080-gene axis cache
(`outputs/scvi_map_cache.npz`) and the evaluation mask
(`outputs/eval_pos_mask.npz`) are included, so the row builders and
benchmarks use the same axis as the reported runs.

## 4. Main benchmark

```
python scripts/bench_transfer_nadig.py      # Tier A equal/mean/gated/centered
python scripts/bench_transfer.py            # Tier B
python scripts/bench_variants.py            # similarity weighting and variants
python scripts/bench_scope_sensitivity.py   # target-coordinate sensitivity
python scripts/transfer_determinants_nadig.py
python scripts/bench_self_lfc_corr.py
```

These produce `outputs/bench_transfer_nadig_v2.{npz,csv}`,
`outputs/bench_transfer_v2.{npz,csv}`, and the variant and determinant
summaries used by Tables 1, 2, and S1--S10.

## 5. Selection audit

```
python scripts/selection_audit.py
python scripts/selection_figure.py
```

This writes `outputs/tables/tableS11_selection_audit.csv` and Figure 10. The
audit reuses the per-target PDS values from the main runs; no predictor is
refit.

## 6. Calibration and source budgets

```
python scripts/phase1_selection_pilot.py --tier A --holdout k562,rpe1 --k 20
python scripts/phase1_budget_curve.py
python scripts/phase2_source_budget.py
python scripts/phase2_exchange_summary.py
python scripts/phase2_exante_allocator.py
```

These produce the calibration curves (Figure 7), the source-budget exchange
rates (Figure 8), and the budget table sources. The pilot uses the same
selection rules at the reported budgets; pass `--help` to override the
holdout, k, split, and repeat options.

## 7. Learned baselines

### scGPT

```
python scripts/scgpt_ft_gwps.py
python scripts/extract_scgpt_all.py
python scripts/bench_scgpt_baseline.py
```

The fine-tuning uses the K562 GWPS essential screen and the released
`scGPT_human` weights. RPE1 is held out from fine-tuning. The comparison table
is `outputs/tables/tableS4_scgpt_baseline.csv`.

### GEARS

The reported run used the following configuration:

```
python scripts/gears_baseline.py --tier A --fold k562,rpe1,hepg2,jurkat \
  --agg mean --val-mode rows --epochs 40 --patience 8 --batch 128 \
  --bn-recalib --diag-train --seed 0
```

Repeat with `--seed 1` and `--seed 2` and merge the per-fold CSV/npz outputs
into `outputs/seed_runs/`. The multi-seed summary is produced by

```
python scripts/seed_summary.py
python scripts/gears_summary.py
python scripts/gears_ctrl_equiv.py
```

`seed_summary.py` writes `outputs/seed_runs/multiseed_summary.csv` (Table S15).
`gears_ctrl_equiv.py` checks the zero cell-input reparameterization described
in Methods.

## 8. Drug panel

```
python scripts/tahoe_external.py --lines A498 A549 HEC-1-A HS-578T J82 MIA_PaCa-2 SK-MEL-2 ...
python scripts/tahoe_drug_excl.py
python scripts/tahoe_coherence.py
python scripts/regime_diagnostic.py
python scripts/regime_pool_geometry.py
python scripts/mix_other_cos.py
python scripts/control_similarity_check.py
python scripts/make_figures_regime.py
python scripts/make_table_regime.py
```

The per-line Tahoe matrices are built from the public pseudobulk differential
expression shards by

```
python scripts/tahoe_build_line.py <line> <first-shard> <last-shard>
```

which downloads and caches each shard. Set `TAHOE_ROOT` to change the working
directory (default `data/tahoe`; the gene axis is read from
`data/tahoe/meta/axis_genes.txt`). `tahoe_coherence.py` reads the matrices
from `data/tahoe/lines` and writes `outputs/tahoe_coherence.csv` by default.
The `tahoe_external.py` subcommands and their required `--out` paths are
documented by `python scripts/tahoe_external.py --help`. The drug-panel
coherence table is `outputs/tables/tableS13` source data; the regime summary is
`outputs/tables/table5_regime.csv`.

## 9. Secondary metrics

```
python scripts/secondary_metrics.py
```

This rebuilds the equal, centered, gated, and mean predictions on the
evaluation axis, aligns them with the stored GEARS predictions, verifies PDS
exactly (maximum absolute difference 0 in every fold), and writes
`outputs/secondary_metrics_A.csv`, `outputs/secondary_metrics_A_pairs.csv`,
and `outputs/secondary_metrics_A.npz` (Table S16). It requires
`outputs/nadig_rows.npz`, `outputs/eval_pos_mask.npz`, and
`outputs/gears_baseline_A.npz`.

## 10. Figures and tables

```
python scripts/make_figures.py              # Figures 1--6
python scripts/phase1_budget_curve.py       # Figure 7
python scripts/phase2_exchange_summary.py   # Figure 8
python scripts/make_figures_regime.py       # Figure 9
python scripts/selection_figure.py          # Figure 10
python scripts/make_figure_coherence.py     # Figure 11
python scripts/make_tables.py               # main and supplementary tables
python scripts/make_tables_budget.py
python scripts/make_table_regime.py
```

## 11. Audits

```
python scripts/audit_numbers.py             # abstract headline numbers
python scripts/check_tex.py                 # braces and environments
python scripts/check_refs.py                # citation and float coverage
python scripts/check_tables.py              # tabular column counts
python scripts/check_draft.py               # abstract length and S15/S16
```

`audit_numbers.py` prints one line per claim and reports 11/11 passes when the
released summary data are complete. The other four scripts are structural
checks, not statistical tests.

## 12. Compute notes

- Tier A/B benchmark runs, variant scans, calibration budgets, and all audits
  are CPU-only.
- GEARS was trained on one GPU with batch size 128 and the configuration
  above; the default batch size in the script is 16, so pass `--batch 128`
  explicitly if you want to reproduce the reported run exactly.
- scGPT fine-tuning used a single GPU; the exact data sampling seed and
  checkpoint are recorded in the manuscript's Supplementary Note 1.
- The full per-seed GEARS arrays are large (about 130 MB per run). The
  released `outputs/seed_runs/` contains the merged summaries and lite per-
  target arrays; the full arrays are in the Zenodo archive.

## 13. Where the large arrays go

The GitHub repository contains every summary table and the arrays needed to
rerun `audit_numbers.py` and the secondary-metric headline checks except for
the GEARS prediction array and the per-target row caches. Download the Zenodo files
into `outputs/` with their original names before running
`secondary_metrics.py` or any GEARS per-target rescoring. The file list and
sizes are in `MANIFEST.md`.
