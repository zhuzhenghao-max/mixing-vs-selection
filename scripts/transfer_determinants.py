"""迁移决定因素: 单源 PDS 与源系 self-LFC / 响应幅度 / 靶表达 / 系间相似度.

对每个 (holdout, source) 对, 在单源方法覆盖的靶上算 Spearman 相关;
再按 self-LFC 分位分箱给出 PDS 曲线 (跨对平均).
逐三元组数据存 outputs/transfer_triples.csv 供画图.
"""

import csv
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from bench_common import load_axis, load_lines, profile_sim


def main(path="outputs/bench_transfer_v2.npz"):
    genes, pos, gidx, m = load_axis()
    lines = load_lines(genes, gidx, m)
    lines.pop("_gidx", None)

    z = np.load(path, allow_pickle=True)
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
            pair_sim = profile_sim(lines[h]["profile"], lines[s]["profile"])
            nrm = np.linalg.norm(lines[s]["rows"][:, pos], axis=1)
            for i in np.where((ho == h) & (me == meth))[0]:
                t = tg[i]
                j = lines[s]["index"].get(t)
                if j is None:
                    continue
                g = gidx.get(t)
                triples.append(dict(
                    holdout=h, source=s, target=t,
                    pds_raw=float(raw[i]),
                    self_lfc=float(lines[s]["self_lfc"][j]),
                    row_norm=float(nrm[j]),
                    expr=float(lines[s]["profile"][g]) if g is not None else np.nan,
                    pair_sim=float(pair_sim)))
    print(f"triples {len(triples)}")

    csv_out = path.replace(".npz", "_triples.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(triples[0]))
        w.writeheader()
        w.writerows(triples)

    print("\nSpearman(PDS, x) per (holdout<-source), x = self_lfc / row_norm / expr")
    pairs = sorted({(t["holdout"], t["source"]) for t in triples})
    for h, s in pairs:
        rows = [t for t in triples if t["holdout"] == h and t["source"] == s]
        n = len(rows)
        if n < 20:
            continue
        p = np.array([r["pds_raw"] for r in rows])
        out = []
        for k, xk in (("self_lfc", "self_lfc"), ("row_norm", "row_norm"),
                      ("expr", "expr")):
            x = np.array([r[xk] for r in rows])
            ok = np.isfinite(x)
            rho = spearmanr(x[ok], p[ok]).statistic
            out.append(f"{k} {rho:+.2f}")
        print(f"  {h:8s}<-{s:8s} n={n:5d}  " + "  ".join(out))

    # self-LFC 分位分箱, 跨对平均
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
                bins[b].append((float(np.median(v[msk])), float(np.median(p[msk]))))
    for b, acc in enumerate(bins):
        if not acc:
            continue
        vv = np.mean([a[0] for a in acc])
        pp = np.mean([a[1] for a in acc])
        print(f"  bin {b}: self_lfc ~ {vv:+.2f}  pds {pp:.4f}  (pairs {len(acc)})")

    print(f"\nsaved {csv_out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/bench_transfer_v2.npz")
