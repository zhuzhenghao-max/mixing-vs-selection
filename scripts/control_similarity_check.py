"""Control-profile similarity vs equal-mix PDS, Tier B (and available Nadig)."""

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from bench_common import load_axis, load_lines, load_nadig_lines, profile_sim

OUT = "outputs"


def equal_pds(path):
    z = np.load(path, allow_pickle=True)
    ho = z["holdout"].astype(str)
    me = z["method"].astype(str)
    keep = me == "equal"
    s = pd.DataFrame({"holdout": ho[keep],
                      "pds": z["pds_raw"][keep].astype(float)})
    return s.groupby("holdout").pds.mean()


def main():
    genes, pos, gidx, m = load_axis()
    lines = load_lines(genes, gidx, m)
    lines.pop("_gidx", None)
    pdsb = equal_pds(f"{OUT}/bench_transfer_v2.npz")
    holds = ["h1", "k562", "rpe1", "hct116", "hek293t"]
    prof = {h: lines[h]["profile"] for h in holds}
    rows = []
    for h in holds:
        srcs = [o for o in holds if o != h]
        ss = [profile_sim(prof[a], prof[b])
              for i, a in enumerate(srcs) for b in srcs[i + 1:]]
        hs = [profile_sim(prof[h], prof[o]) for o in srcs]
        rows.append(dict(tier="B", holdout=h, pds=pdsb.get(h),
                         src_src=np.mean(ss), src_hold=np.mean(hs)))
    b = pd.DataFrame(rows)
    print(b.to_string(index=False))
    for c in ("src_src", "src_hold"):
        r = spearmanr(b[c], b.pds)
        print(f"Tier B n=5 {c} vs PDS: rho {r.statistic:+.3f} p {r.pvalue:.3f}")
    b.to_csv(f"{OUT}/control_similarity_tierB.csv", index=False,
             float_format="%.6f")

    na = load_nadig_lines(genes, gidx)
    pdsa = equal_pds(f"{OUT}/bench_transfer_nadig_v2.npz")
    rows = []
    for h in ("k562", "rpe1"):
        if na[h]["profile"] is None:
            continue
        srcs = [o for o in ("k562", "rpe1", "hepg2", "jurkat")
                if o != h and na[o]["profile"] is not None]
        if len(srcs) < 1:
            continue
        hs = [profile_sim(na[h]["profile"], na[o]["profile"]) for o in srcs]
        rows.append(dict(tier="A", holdout=h, pds=pdsa.get(h),
                         src_src=np.nan, src_hold=np.mean(hs)))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
