"""Tier A (Nadig) 迁移决定因素: 单源 PDS vs self-LFC / 响应幅度.

评测位点 = 18080 轴 eval mask ∩ Nadig 6118 被测基因 (同 bench_transfer_nadig).
"""

import csv
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from bench_common import load_axis, load_nadig_lines


def main():
    genes, pos, gidx, m = load_axis()
    lines = load_nadig_lines(genes, gidx)
    z0 = np.load("outputs/nadig_rows.npz", allow_pickle=True)
    in_g = np.zeros(len(genes), dtype=bool)
    in_g[z0["gpos"]] = True
    posn = pos[in_g[pos]]

    z = np.load("outputs/bench_transfer_nadig_v2.npz", allow_pickle=True)
    ho = z["holdout"].astype(str)
    me = z["method"].astype(str)
    tg = z["target"].astype(str)
    raw = z["pds_raw"].astype(np.float64)

    triples = []
    for h in sorted(set(ho)):
        singles = sorted({mth for i, mth in enumerate(me)
                          if ho[i] == h and mth.startswith("single_")})
        for meth in singles:
            s = meth[len("single_"):]
            if s not in lines or h not in lines:
                continue
            nrm = np.linalg.norm(lines[s]["rows"][:, posn], axis=1)
            for i in np.where((ho == h) & (me == meth))[0]:
                t = tg[i]
                j = lines[s]["index"].get(t)
                if j is None:
                    continue
                triples.append(dict(holdout=h, source=s, target=t,
                                    pds_raw=float(raw[i]),
                                    self_lfc=float(lines[s]["self_lfc"][j]),
                                    row_norm=float(nrm[j])))
    print(f"triples {len(triples)}")

    csv_out = "outputs/bench_transfer_nadig_v2_triples.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(triples[0]))
        w.writeheader()
        w.writerows(triples)

    print("\nSpearman(PDS, x) per (holdout<-source)")
    pairs = sorted({(t["holdout"], t["source"]) for t in triples})
    for h, s in pairs:
        rows = [t for t in triples if t["holdout"] == h and t["source"] == s]
        if len(rows) < 20:
            continue
        p = np.array([r["pds_raw"] for r in rows])
        desc = []
        for k in ("self_lfc", "row_norm"):
            x = np.array([r[k] for r in rows])
            ok = np.isfinite(x)
            rho = spearmanr(x[ok], p[ok]).statistic
            desc.append(f"{k} {rho:+.2f}")
        print(f"  {h:8s}<-{s:8s} n={len(rows):5d}  " + "  ".join(desc))

    print("\nself-LFC bin -> median PDS (within-pair deciles, averaged)")
    bins = [[] for _ in range(10)]
    for h, s in pairs:
        rows = [t for t in triples if t["holdout"] == h and t["source"] == s
                and np.isfinite(t["self_lfc"])]
        if len(rows) < 50:
            continue
        v = np.array([r["self_lfc"] for r in rows])
        p = np.array([r["pds_raw"] for r in rows])
        edges = np.quantile(v, np.linspace(0, 1, 11))
        for b in range(10):
            msk = (v >= edges[b]) & (v <= edges[b + 1] if b == 9
                                     else v < edges[b + 1])
            if msk.sum():
                bins[b].append((float(np.median(v[msk])),
                                float(np.median(p[msk]))))
    for b, acc in enumerate(bins):
        if not acc:
            continue
        vv = np.mean([a[0] for a in acc])
        pp = np.mean([a[1] for a in acc])
        print(f"  bin {b}: self_lfc ~ {vv:+.2f}  pds {pp:.4f}  "
              f"(pairs {len(acc)})")

    print(f"\nsaved {csv_out}")


if __name__ == "__main__":
    main()
