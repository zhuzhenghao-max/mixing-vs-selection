"""Recompute the Tahoe coherence partial correlations with the standard
rank-residual estimator.

The stored per-condition table already contains source coherence, mix-truth
alignment, and full-pool PDS, so the estimator can be corrected without the
raw response matrices. The stored PDS values remain affected by the
separate-truth-score defect noted in the manuscript; these partial values are
descriptive and inherit that caveat.

Writes outputs/tahoe_partial_corrected.csv.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

IN = "outputs/tahoe_coherence.csv"
OUT = "outputs/tahoe_partial_corrected.csv"


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return pearsonr(ex, ey).statistic


def fisher_mean(v):
    v = np.clip(np.asarray(v, dtype=float), -0.999, 0.999)
    return float(np.tanh(np.mean(np.arctanh(v))))


def main():
    d = pd.read_csv(IN)
    rows = []
    for h, g in d.groupby("holdout"):
        rows.append(dict(
            holdout=h, n=len(g),
            coherence=g.pair_cos.mean(),
            alignment=g.mix_truth_cos.mean(),
            rho_raw=spearmanr(g.pair_cos, g.pds_full).statistic,
            rho_adj_alignment=partial_spearman(
                g.pair_cos.values, g.pds_full.values,
                g.mix_truth_cos.values),
            rho_alignment_raw=spearmanr(g.mix_truth_cos, g.pds_full).statistic,
            rho_alignment_adj_coherence=partial_spearman(
                g.mix_truth_cos.values, g.pds_full.values,
                g.pair_cos.values)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, float_format="%.6f")
    print(out.to_string(index=False))
    print(f"pooled coherence|alignment {fisher_mean(out.rho_adj_alignment):+.3f}")
    print(f"pooled alignment|coherence "
          f"{fisher_mean(out.rho_alignment_adj_coherence):+.3f}")
    print(f"coherence mean range {out.coherence.min():.3f}-"
          f"{out.coherence.max():.3f}")
    print(f"stored PDS above 1: {100 * (d.pds_full > 1).mean():.1f}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
