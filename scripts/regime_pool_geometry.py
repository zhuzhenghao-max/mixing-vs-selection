"""Candidate-pool geometry for the regime diagnostic.

PDS is a rank against the held-out line's measured pool, so what matters is
not the absolute coherence of the source responses but their coherence
relative to how similar the candidate responses are to one another.  For each
query this script computes (a) the mean cosine between the held-out truth and
the other pool rows, and (b) for every source, the mean pairwise cosine among
that source's rows for the query set, both from unit-normalized rows.
Joins the per-target table written by regime_diagnostic.py.

Output: outputs/regime_pool_geometry.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from bench_common import load_axis, load_lines, load_nadig_lines, unit_rows

OUT = "outputs"


def src_pool_cos(rows_n, sel):
    V = rows_n[sel]
    s = V.sum(0)
    n = len(V)
    if n < 2:
        return np.nan
    return float((s @ s - n) / (n * (n - 1)))


def main():
    genes, pos, gidx, m = load_axis()
    base = pd.read_csv(f"{OUT}/regime_diagnostic_targets.csv")

    def tier_stats(tier, lines, holds):
        d = base[base.tier == tier]
        rows = []
        for h in holds:
            if h not in lines:
                continue
            hl = lines[h]
            R = hl["rows"][:, pos].astype(np.float64)
            Rn = unit_rows(R)
            srcs = [o for o in holds if o != h and o in lines]
            sub = d[d.holdout == h]
            pool, mix_sum = [], []
            for t in sub.target:
                i = hl["index"].get(t)
                if i is None:
                    continue
                c = Rn @ Rn[i]
                pool.append(float((c.sum() - 1.0) / (len(c) - 1)))
            src_sel = {}
            for o in srcs:
                ol = lines[o]
                sel = [ol["index"][t] for t in sub.target
                       if t in ol["index"]]
                if len(sel) < 2:
                    src_sel[o] = np.nan
                    continue
                Ro = unit_rows(ol["rows"][:, pos].astype(np.float64))
                src_sel[o] = src_pool_cos(Ro, sel)
            for t, pc in zip(sub.target, pool):
                rows.append(dict(tier=tier, holdout=h, target=t,
                                 pool_cos=pc,
                                 src_pool_cos=float(np.nanmean(
                                     list(src_sel.values())))))
        return rows

    rows = []
    lines_a = load_nadig_lines(genes, gidx)
    rows += tier_stats("A", lines_a, ["k562", "rpe1", "hepg2", "jurkat"])
    del lines_a
    lines_b = load_lines(genes, gidx, m)
    lines_b.pop("_gidx", None)
    rows += tier_stats("B", lines_b,
                       ["h1", "k562", "rpe1", "hct116", "hek293t"])
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/regime_pool_geometry.csv", index=False,
             float_format="%.6f")
    s = (d.groupby(["tier", "holdout"])
         .agg(n=("target", "size"), pool_cos=("pool_cos", "mean"),
              src_pool_cos=("src_pool_cos", "mean"))
         .reset_index())
    b = (base.groupby(["tier", "holdout"])
         .agg(pds=("pds", "mean"), pds_noexcl=("pds_noexcl", "mean"),
              pair_cos=("pair_cos", "mean"),
              mix_truth_cos=("mix_truth_cos", "mean"),
              resultant=("resultant", "mean"))
         .reset_index())
    s = s.merge(b, on=["tier", "holdout"])
    s["pair_rel"] = s.pair_cos / s.src_pool_cos
    s["mix_rel"] = s.mix_truth_cos / s.pool_cos
    s["margin"] = s.mix_truth_cos - s.pool_cos
    print(s.to_string(index=False))
    s.to_csv(f"{OUT}/regime_pool_geometry_holdouts.csv", index=False,
             float_format="%.6f")
    print("\nacross 9 genetic holdouts, Spearman of means with mean PDS:")
    for c in ("pair_cos", "mix_truth_cos", "pool_cos", "src_pool_cos",
              "pair_rel", "mix_rel", "margin"):
        r_ex = spearmanr(s[c], s.pds).statistic
        r_nx = spearmanr(s[c], s.pds_noexcl).statistic
        print(f"  {c:12s} excl {r_ex:+.3f}  noexcl {r_nx:+.3f}")
    print("\nwithin-holdout Spearman pool_cos vs pds:")
    for (tier, h), g in d.groupby(["tier", "holdout"]):
        b2 = base[(base.tier == tier) & (base.holdout == h)]
        g = g.merge(b2[["target", "pds", "pds_noexcl", "pair_cos"]],
                    on="target")
        print(f"  {tier} {h:9s} n={len(g):5d} pool_cos {spearmanr(g.pool_cos, g.pds).statistic:+.3f} "
              f"pair {spearmanr(g.pair_cos, g.pds).statistic:+.3f} "
              f"srcpool {spearmanr(g.src_pool_cos, g.pds).statistic:+.3f}")


if __name__ == "__main__":
    main()
