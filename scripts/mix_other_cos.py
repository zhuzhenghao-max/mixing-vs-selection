"""Cosine of the equal mix to other candidate truths (pool contamination).

PDS depends on where the correct truth sits in the distribution of cosines
between the prediction and the other pool rows.  This script computes that
mean cosine for the genetic panels so it can be compared with the drug panel.
Output: outputs/mix_other_cos.csv
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bench_common import load_axis, load_lines, load_nadig_lines, unit_rows

OUT = "outputs"


def run_tier(tier, lines, holds, pds_csv):
    rows = []
    for h in holds:
        if h not in lines:
            continue
        hl = lines[h]
        srcs = [o for o in holds if o != h and o in lines]
        rows_h = []
        for t in hl["names"]:
            if t in ("",):
                continue
            vecs = [lines[o]["rows"][lines[o]["index"][t]]
                    for o in srcs if t in lines[o]["index"]]
            if not vecs:
                continue
            mix = unit_rows(np.stack(vecs)).mean(0)
            rows_h.append((t, mix))
        base = pd.read_csv(pds_csv)
        base = base[base.holdout == h]
        qset = set(base.target)
        names = [t for t, _ in rows_h if t in qset]
        M = np.stack([m for t, m in rows_h if t in qset])
        pos = np.load("outputs/eval_pos_mask.npz")["mask"]
        # truth pool on the scored axis
        T = hl["rows"][:, pos].astype(np.float64)
        Tn = unit_rows(T)
        Mn = unit_rows(M[:, pos])
        S = Tn @ Mn.T
        for i, t in enumerate(names):
            j = hl["index"][t]
            col = S[:, i]
            other = (col.sum() - col[j]) / (len(col) - 1)
            rest = np.delete(col, j)
            rows.append(dict(tier=tier, holdout=h, target=t,
                             other_cos=float(other),
                             other_q95=float(np.quantile(rest, 0.95)),
                             other_q99=float(np.quantile(rest, 0.99))))
    return rows


def main():
    genes, pos, gidx, m = load_axis()
    rows = []
    la = load_nadig_lines(genes, gidx)
    rows += run_tier("A", la, ["k562", "rpe1", "hepg2", "jurkat"],
                     f"{OUT}/regime_diagnostic_targets.csv")
    del la
    lb = load_lines(genes, gidx, m)
    lb.pop("_gidx", None)
    rows += run_tier("B", lb, ["h1", "k562", "rpe1", "hct116", "hek293t"],
                     f"{OUT}/regime_diagnostic_targets.csv")
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/mix_other_cos.csv", index=False, float_format="%.6f")
    b = pd.read_csv(f"{OUT}/regime_diagnostic_targets.csv")
    s = (d.groupby(["tier", "holdout"])[["other_cos", "other_q95",
                                          "other_q99"]].mean()
         .reset_index()
         .merge(b.groupby(["tier", "holdout"])
                .agg(pds=("pds", "mean"), pair=("pair_cos", "mean"),
                     mix_truth=("mix_truth_cos", "mean"))
                .reset_index(), on=["tier", "holdout"]))
    s["margin"] = s.mix_truth - s.other_cos
    s["margin95"] = s.mix_truth - s.other_q95
    s["margin99"] = s.mix_truth - s.other_q99
    print(s.to_string(index=False))
    s.to_csv(f"{OUT}/mix_other_cos_holdouts.csv", index=False,
             float_format="%.6f")


if __name__ == "__main__":
    main()
