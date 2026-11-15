# MANIFEST

Release 1.1.0, 2026-11-15. This file lists what is in the release, where the
raw data come from, which files are archived separately, and how to deposit
the release. `CHECKSUMS.sha256` covers every file in this repository.
Release 1.1.0 adds the twelve-line Tahoe-100M robustness panel, its source
data (Tables S17 and S18), and the two scripts that produce it.

## 1. Contents

| Path | Files | Size | Contents |
|---|---:|---:|---|
| `scripts/` | 143 | 1.1 MB | Python and shell analysis code; run from the repository root |
| `src/` | 3 | 36 KB | Shared modules imported by some scripts |
| `configs/` | 9 | 13 KB | Evaluation configs used during development |
| `data/` | 7 | 1.9 MB | Gene sets for the GEARS graph, gene-name axes, Tahoe axis |
| `outputs/` | 245 | 163 MB | Derived scores, summary tables, run logs |
| `outputs/tables/` | 29 | 151 KB | Main and supplementary table source data (CSV and MD) |
| `outputs/seed_runs/` | 7 | 540 KB | Three-seed GEARS summaries, lite arrays, run logs |
| `figures/` | 22 | 3.1 MB | Figures 1--11, PDF and PNG |
| `manuscript/` | 1 | 164 KB | `draft.tex`, the manuscript source |
| root | 10 | 97 KB | `README.md`, `REPRODUCE.md`, `MANIFEST.md`, licenses, `CITATION.cff`, `.zenodo.json`, `requirements.txt`, `.gitignore`, checksums |

Total: 440 files, about 169 MB. Of these, `outputs/scvi_map_cache.npz`
(54 MB), `outputs/bench_transfer_v2.npz` (33 MB), and
`outputs/scgpt_delta_check.npz` (14 MB) are the only files above 10 MB.

## 2. Primary result files

| File | Produced by | Used for |
|---|---|---|
| `outputs/bench_transfer_nadig_v2.{npz,csv}` | `scripts/bench_transfer_nadig.py` | Tier A per-target PDS; Tables 1, 2, S1, S3, S5, S6, S11; Figures 1--5, 10 |
| `outputs/bench_transfer_nadig_v2_pairs.csv` | `scripts/bench_transfer_analysis.py` | Paired effect sizes for Tier A; Table S1 |
| `outputs/bench_transfer_nadig_v2_triples.csv` | `scripts/transfer_determinants_nadig.py` | Per source--holdout determinants; Table S2 |
| `outputs/bench_transfer_v2.{npz,csv}` | `scripts/bench_transfer.py` | Tier B per-target PDS; Tables 1, 2, S1, S3, S5, S6, S11 |
| `outputs/bench_transfer_v2_pairs.csv` | `scripts/bench_transfer_analysis.py` | Paired effect sizes for Tier B; Table S1 |
| `outputs/bench_transfer_v2_triples.csv` | `scripts/transfer_determinants.py` | Per source--holdout determinants; Table S2 |
| `outputs/bench_scope_sensitivity.csv`, `outputs/bench_scope_sensitivity.log` | `scripts/bench_scope_sensitivity.py` | Panel-wide exclusion sensitivity; Table S7 |
| `outputs/bench_self_lfc_corr*.csv` | `scripts/bench_self_lfc_corr.py` | Self-log-fold-change correlation; Table S5b |
| `outputs/bench_supervised_*` (CSV, NPZ, logs) | `scripts/bench_supervised*.py` | Supervised calibration; Table S8 |
| `outputs/bench_scgpt_baseline.npz` | `scripts/bench_scgpt_baseline.py` | scGPT comparison; Table S4, Figure 6 |
| `outputs/gears_baseline_A.csv`, `outputs/gears_summary_A.csv` | `scripts/gears_baseline.py`, `scripts/gears_summary.py` | GEARS baseline; Table S14 |
| `outputs/seed_runs/multiseed_summary.csv` | `scripts/seed_summary.py` | Three-seed GEARS reproducibility; Table S15 |
| `outputs/secondary_metrics_A.{csv,pairs.csv,npz}` | `scripts/secondary_metrics.py` | Pool-free metrics; Table S16 |
| `outputs/selection_audit_kcurve.csv`, `outputs/tables/tableS11_selection_audit.csv` | `scripts/selection_audit.py` | Selection audit; Table S11, Figure 10 |
| `outputs/phase1_*.csv` | `scripts/phase1_budget_curve.py`, `scripts/phase1_selection_pilot.py` | Calibration curves; Figure 7, Table 3 |
| `outputs/phase2_*.csv` | `scripts/phase2_source_budget.py`, `scripts/phase2_exchange_summary.py` | Source-budget exchange; Figure 8, Tables 3, 4 |
| `outputs/phase2_exante_allocator.csv` | `scripts/phase2_exante_allocator.py` | Source-side allocation; Table S9 |
| `outputs/phase2_fullmix_noexcl.csv` | `scripts/phase2_fullmix_noexcl.py` | Calibration under no-exclusion scoring; Table S10 |
| `outputs/regime_diagnostic_holdouts.csv`, `outputs/regime_diagnostic_targets.csv` | `scripts/regime_diagnostic.py` | Coherence diagnostic; Table S12, Figure 11 |
| `outputs/tahoe_coherence.csv` | `scripts/tahoe_coherence.py` | Drug-panel coherence; Table S13 |
| `outputs/tahoe12_full.csv`, `outputs/tahoe12_delta.csv`, `outputs/tahoe12_sat.csv`, `outputs/tahoe12_pilot.csv`, `outputs/p7_646_full.csv` | `scripts/tahoe_panel12.py` | Twelve-line panel retrieval, leave-one-source-out deltas, saturation, and pilot; Table S17 |
| `outputs/tahoe12_panel646.csv` | Intersection of the twelve per-line condition lists (646 conditions), saved as the panel file for `scripts/tahoe_panel12.py` | Twelve-line panel and the seven-line comparison on the same panel; Table S17 |
| `outputs/tahoe12_coherence.csv` | `scripts/tahoe_coherence_fast.py` | Twelve-line per-condition coherence; Table S17 |
| `outputs/tahoe12_select.csv`, `outputs/tahoe12_src.csv`, `outputs/tahoe12_deltas.csv`, `outputs/tahoe12_ci.csv` | `scripts/tahoe_external.py` (`select`), `scripts/tahoe_cal_ci.py` | Twelve-line calibration selection; Table S18 |
| `outputs/tahoe12_exch.csv` | `scripts/tahoe_external.py` (`exchange`) | Twelve-line source-exchange rates |
| `outputs/selectX.csv`, `outputs/srcX.csv`, `outputs/exchX.csv` | `scripts/tahoe_external.py` (`select`, `exchange`) | Drug-panel calibration, saturation, exchange; Figure 9C--E, Table 5 |
| `outputs/regime_pool_geometry*.csv`, `outputs/mix_other_cos.csv`, `outputs/control_similarity_tierB.csv`, `outputs/pilot_rates.csv` | `scripts/regime_pool_geometry.py`, `scripts/mix_other_cos.py`, `scripts/control_similarity_check.py`, `scripts/regime_summary.py` | Regime analysis; Figure 9, Table 5 |
| `outputs/pilot_x.npz`, `outputs/pilot_pds_nadig_noexcl.npz` | Pilot draw caches (`scripts/tahoe_drug_excl.py` and the matching Nadig pilot run) | 50-condition pilot; Figure 9F, Table 5 |
| `outputs/*.log` | the corresponding benchmark runs | Pool sizes in Figure 1 and the variant scans in Tables S3 and S7 |

## 3. Script to manuscript map

| Manuscript object | Script |
|---|---|
| Tables 1, 2, S1--S8 | `scripts/make_tables.py` |
| Tables 3, 4, S9, S10 | `scripts/make_tables_budget.py` |
| Table 5 | `scripts/make_table_regime.py` |
| Table S11 | `scripts/selection_audit.py` |
| Table S12 | `scripts/regime_diagnostic.py` |
| Table S13 | `scripts/tahoe_coherence.py` |
| Table S14 | `scripts/gears_baseline.py`, `scripts/gears_summary.py` |
| Table S15 | `scripts/seed_summary.py` |
| Table S16 | `scripts/secondary_metrics.py` |
| Table S17 | `scripts/tahoe_panel12.py`, `scripts/tahoe_coherence_fast.py` |
| Table S18 | `scripts/tahoe_external.py` (`select`, `exchange`), `scripts/tahoe_cal_ci.py` |
| Figures 1--6 | `scripts/make_figures.py` |
| Figure 7 | `scripts/phase1_budget_curve.py` |
| Figure 8 | `scripts/phase2_exchange_summary.py` |
| Figure 9 | `scripts/make_figures_regime.py` |
| Figure 10 | `scripts/selection_figure.py` |
| Figure 11 | `scripts/make_figure_coherence.py` |
| Headline-number audit | `scripts/audit_numbers.py` |
| Structural checks | `scripts/check_tex.py`, `scripts/check_refs.py`, `scripts/check_tables.py`, `scripts/check_draft.py` |

## 4. Raw data provenance

Raw data are not redistributed. They remain under the licenses of the
original publications. Dataset identifiers and references match the
manuscript bibliography.

| Dataset | Used for | Access | Reference |
|---|---|---|---|
| Nadig et al. harmonized atlas (K562, RPE1, HepG2, Jurkat) | Tier A | Processed table placed at `data/external/all_delta.parquet` (about 6.5 GB); `scripts/build_nadig_rows.py` streams it | Nadig et al., *Nature Genetics* 2025, doi:10.1038/s41588-025-02169-3 [7] |
| Replogle et al. K562 genome-wide Perturb-seq | Tier B source; scGPT fine-tuning | `scripts/gwps_download.sh` fetches `replogle_2022_k562_gwps.h5ad` (about 8.8 GB) from `https://exampledata.scverse.org/pertpy/` | Replogle et al., *Cell* 2022, doi:10.1016/j.cell.2022.05.013 [1] |
| Replogle et al. RPE1 Perturb-seq | Tier B source | `scripts/rpe1_download.sh` fetches `replogle_2022_rpe1.h5ad` (about 1.2 GB) from the same host | Replogle et al., *Cell* 2022 [1] |
| X-Atlas/Orion (HCT116, HEK293T) | Tier B sources | `data/external/orion_lfc_true.npy`, `orion_pert_ids.npy`, `orion_ctx_ids.npy`, `orion_meta.json` from the X-Atlas/Orion release | Huang et al., bioRxiv 2025, doi:10.1101/2025.06.11.659105 [6] |
| Virtual Cell Challenge H1 responses | Tier B holdout H1 | 2025 challenge training split, embedded on the Tier B axis in `outputs/bench_transfer_v2.npz`; no challenge test data are used | Roohani et al., *Cell* 2025, doi:10.1016/j.cell.2025.06.008 [10] |
| Tahoe-100M pseudobulk differential expression | Drug panel | `scripts/tahoe_build_line.py` downloads the public shards (`train-00000-of-01026.parquet`) from the Tahoe-100M ModelScope/Hugging Face mirrors | Zhang et al., bioRxiv 2025, doi:10.1101/2025.02.20.639398 [12] |
| Gene sets (MSigDB Hallmark 2020, KEGG 2021 Human, Reactome 2022, ENCODE/ChEA consensus TFs) | Perturbation graph for GEARS, as GO stand-ins | `data/gs_*.txt` | Gene Ontology Consortium 2021 [22]; GEARS [11] |

The 18,080-gene analysis axis is stored in `outputs/scvi_map_cache.npz`, and
the evaluation mask in `outputs/eval_pos_mask.npz`. `data/gene_names.csv` and
`data/gene_names_2025.csv` record the gene-name axes used during development;
`data/tahoe/meta/axis_genes.txt` is the 10,778-gene panel axis for the drug
analysis.

## 5. Files archived on Zenodo

These are too large for the GitHub repository or are easier to deposit as
separate files. Download them into `outputs/` with the names below before
rerunning the benchmarks or `secondary_metrics.py`.

| File | Bytes | Needed for |
|---|---:|---|
| `outputs/nadig_rows.npz` | 131,220,703 | Tier A per-target rows; rebuilding Tier A and `secondary_metrics.py` |
| `outputs/gears_baseline_A.npz` | 130,866,724 | Seed-0 GEARS predictions and per-target PDS; Tables S14, S16 |
| `outputs/rpe1_rows.npz` | 173,186,946 | RPE1 rows; rebuilding Tier B |
| `outputs/gwps_rows.npz` | 714,141,294 | K562 GWPS rows; rebuilding Tier B and the scGPT baseline |
| `outputs/seed_runs/gears_baseline_A_seed1_full.npz` | about 130,000,000 | Seed-1 full predictions; per-target seed agreement |
| `outputs/seed_runs/gears_baseline_A_seed2_full.npz` | about 130,000,000 | Seed-2 full predictions; per-target seed agreement |

The seed 1/2 full arrays are kept on the compute instance and are not in this
repository. The released `seed_runs/*_full_lite.npz` files contain the lite
per-target arrays that `scripts/seed_summary.py` reads, so Table S15 can be
rebuilt without them.

The following intermediates are regenerable from the files above and are not
deposited: `outputs/bench_scope_sensitivity.npz` (17 MB),
`outputs/replogle_pooled.npz` (6.4 MB), and the larger `scgpt_delta_check*.npz`
files used during model selection.

## 6. Verification

From the repository root:

```
sha256sum -c CHECKSUMS.sha256
python scripts/audit_numbers.py          # 11/11 checks
python scripts/check_refs.py
python scripts/check_tables.py
python scripts/check_tex.py
python scripts/check_draft.py
```

The table and figure builders run from this release without the Zenodo files:

```
python scripts/make_tables.py
python scripts/make_figures.py
```

## 7. Depositing the release

The release is available at https://github.com/zhuzhenghao-max/mixing-vs-selection
and archived at https://doi.org/10.5281/zenodo.24223291 (concept DOI; version
DOIs resolve to each release). The GitHub tag is `v1.1.0`, matching the
version in `CITATION.cff` and `.zenodo.json`.

The archive license is set to CC BY 4.0 in `.zenodo.json` because the deposit
is data-heavy; the code itself is MIT-licensed (see `LICENSE`).
