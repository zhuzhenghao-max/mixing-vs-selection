"""Paired statistics for the H1 truth-pipeline confirmation.

Reads outputs/vccp_confirm_targets.csv and restricts to targets with at
least one source response. Reports the paired DESeq2-minus-cache PDS
difference and the pathway-profile difference against the derangement
control, both with 4,000-draw bootstrap intervals and two-sided Wilcoxon
tests. Writes outputs/vccp_paired_stats.csv.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, "scripts")
from bench_common import boot_ci

IN = "outputs/vccp_confirm_targets.csv"
OUT = "outputs/vccp_paired_stats.csv"


def summarize(name, d):
    d = np.asarray(d, dtype=float)
    lo, hi, mu = boot_ci(d, n=4000, seed=0)
    return dict(contrast=name, n=len(d), mean=mu, lo=lo, hi=hi,
                median=float(np.median(d)),
                p_wilcoxon=float(wilcoxon(d).pvalue))


def main():
    d = pd.read_csv(IN)
    ok = (d.n_src > 0) & np.isfinite(d.pds_cache) & np.isfinite(d.pds_vccp)
    g = d.loc[ok]
    rows = [summarize("deseq2_minus_cache_pds",
                      g.pds_vccp - g.pds_cache),
            summarize("pathway_mix_minus_mismatch",
                      g.path_corr_mix - g.path_corr_mismatch)]
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, float_format="%.6f")
    rho = spearmanr(g.pds_cache, g.pds_vccp)
    print(out.to_string(index=False))
    print(f"target-level Spearman {rho.statistic:+.4f} (p = {rho.pvalue:.4g}, "
          f"n = {len(g)})")
    print(f"pathway means mix {g.path_corr_mix.mean():+.4f}, "
          f"mismatch {g.path_corr_mismatch.mean():+.4f}, "
          f"cache {g.path_corr_cache.mean():+.4f}")
    print(f"PDS means cache {g.pds_cache.mean():.4f}, "
          f"DESeq2 {g.pds_vccp.mean():.4f}, "
          f"stored benchmark {g.pds_bench.mean():.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
