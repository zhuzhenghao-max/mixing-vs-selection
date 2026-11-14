"""对 bench_transfer npz 做配对统计: 同靶交集上比较方法, bootstrap 中位数差 CI."""

import sys

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

PAIRS = [
    ("gated_equal", "equal"),
    ("gated_selfw", "gated_equal"),
    ("equal", "mean"),
    ("equal_center", "equal"),
    ("equal_raw", "equal"),
    ("top2_selfw", "equal"),
    ("simT0.2", "equal"),
    ("gated_equal", "simT0.2"),
]


def load(path):
    z = np.load(path, allow_pickle=True)
    ho = z["holdout"].astype(str)
    me = z["method"].astype(str)
    tg = z["target"].astype(str)
    raw = z["pds_raw"]
    res = z["pds_res"]
    out = {}
    for i in range(len(ho)):
        out.setdefault((ho[i], me[i]), {})[tg[i]] = (raw[i], res[i])
    return out


def paired_stats(d, n=4000, seed=0):
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return None
    rng = np.random.default_rng(seed)
    bs = np.median(rng.choice(d, (n, len(d))), axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = np.nan
    if wilcoxon is not None and np.any(d != 0):
        p = float(wilcoxon(d).pvalue)
    nz = d[d != 0]
    return dict(n=len(d), diff_med=float(np.median(d)),
                diff_mean=float(np.mean(d)),
                frac_nonzero=float(np.mean(d != 0)),
                win_all=float(np.mean(d > 0)),
                win_nonzero=float(np.mean(nz > 0)) if len(nz) else np.nan,
                ci_lo=float(lo), ci_hi=float(hi), p=p)


def main(path="outputs/bench_transfer_v2.npz"):
    D = load(path)
    holds = sorted({h for h, _ in D})
    methods = sorted({m for _, m in D})
    print("holdout   method          n     med_raw  med_res")
    for h in holds:
        for m in methods:
            v = D.get((h, m))
            if not v:
                continue
            raw = np.array([x[0] for x in v.values()])
            res = np.array([x[1] for x in v.values()])
            print(f"{h:9s} {m:15s} {len(v):5d}  {np.nanmedian(raw):.4f}  "
                  f"{np.nanmedian(res):.4f}")

    print("\n=== paired (raw PDS, same targets) ===")
    rows = []
    for h in holds:
        pairs = list(PAIRS)
        for m in methods:
            if m.startswith("single_"):
                pairs.append(("gated_equal", m))
                pairs.append(("equal", m))
        for a, b in pairs:
            da, db = D.get((h, a)), D.get((h, b))
            if not da or not db:
                continue
            common = sorted(set(da) & set(db))
            if len(common) < 5:
                continue
            d = np.array([da[t][0] - db[t][0] for t in common])
            st = paired_stats(d)
            if st is None:
                continue
            med_a = float(np.nanmedian([da[t][0] for t in common]))
            med_b = float(np.nanmedian([db[t][0] for t in common]))
            rows.append(dict(holdout=h, a=a, b=b, **st,
                             med_a=med_a, med_b=med_b))
            print(f"{h:9s} {a:13s} vs {b:13s} n={st['n']:5d} "
                  f"med {med_a:.4f} vs {med_b:.4f} "
                  f"d_med={st['diff_med']:+.4f} d_mean={st['diff_mean']:+.4f} "
                  f"nz={st['frac_nonzero']:.2f} win_nz={st['win_nonzero']:.2f} "
                  f"p={st['p']:.1e}")

    import csv
    out = path.replace(".npz", "_pairs.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/bench_transfer_v1.npz")
