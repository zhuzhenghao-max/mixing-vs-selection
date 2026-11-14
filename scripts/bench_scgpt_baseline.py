"""scGPT 零样本 vs 跨系混合基线, 同 93 靶同池同口径 (self 剔除, 分母 n).

scGPT 预测为 outputs/scgpt_delta_check*.npz 的 delta_rpe1 / delta_k562gwps;
混合基线来自 outputs/bench_transfer_v2.npz.
"""

import sys

import numpy as np

sys.path.insert(0, "scripts")
from bench_common import Retrieval, load_axis, load_lines

HOLDS = {"rpe1": "delta_rpe1", "k562": "delta_k562gwps"}


def main():
    genes, pos, gidx, m = load_axis()
    lines = load_lines(genes, gidx, m)
    lines.pop("_gidx", None)

    sc = np.load("outputs/scgpt_delta_check.npz", allow_pickle=True)
    sc_t = [str(x) for x in sc["targets"].astype(str)]
    sc_g = [str(x) for x in sc["genes"].astype(str)]
    assert sc_g == [str(x) for x in genes], "gene axis mismatch"
    sc10 = np.load("outputs/scgpt_delta_check_r10_pbadd.npz",
                   allow_pickle=True)

    z = np.load("outputs/bench_transfer_v2.npz", allow_pickle=True)
    ho, me, tg = (z[k].astype(str) for k in ("holdout", "method", "target"))
    raw = z["pds_raw"].astype(np.float64)
    D = {}
    for i in range(len(ho)):
        D.setdefault((ho[i], me[i]), {})[tg[i]] = raw[i]

    store = {}
    for L, key in HOLDS.items():
        ref = lines[L]
        names = [t for t in sc_t if t in ref["index"]]
        preds = {
            "scgpt_check": (names,
                            sc[key][[sc_t.index(t) for t in names]]),
        }
        if L == "rpe1" and "delta_rpe1" in sc10.files:
            names10 = [str(x) for x in sc10["targets"].astype(str)]
            keep = [i for i, t in enumerate(names10) if t in ref["index"]]
            preds["scgpt_r10"] = ([names10[i] for i in keep],
                                  sc10["delta_rpe1"][keep])
        R = Retrieval(ref["rows"], ref["names"], pos, gidx=gidx)
        for tag, (ns, P) in preds.items():
            pds = R.score(P, ns, exclude="self", denominator="n")
            store[f"{L}|{tag}|targets"] = np.array(ns)
            store[f"{L}|{tag}|pds"] = pds
            print(f"\n=== {L} vs {tag}: n={len(ns)} ===")
            print(f"  {tag:14s} med {np.nanmedian(pds):.4f} "
                  f"mean {np.nanmean(pds):.4f}")
            for meth in ("equal", "equal_center", "gated_equal",
                         "single_k562", "single_rpe1", "mean"):
                if (L, meth) not in D:
                    continue
                common = [t for t in ns if t in D[(L, meth)]]
                if len(common) < 20:
                    continue
                p = R.score(P[[ns.index(t) for t in common]], common,
                            exclude="self", denominator="n")
                q = np.array([D[(L, meth)][t] for t in common])
                ok = np.isfinite(p) & np.isfinite(q)
                d = q[ok] - p[ok]
                store[f"{L}|{tag}|{meth}|targets"] = np.array(common)
                store[f"{L}|{tag}|{meth}|pds"] = q
                print(f"  {meth:14s} n={len(common):3d} "
                      f"med {np.nanmedian(q):.4f} mean {np.nanmean(q):.4f} "
                      f"vs {tag}: d_med={np.nanmedian(d):+.4f} "
                      f"d_mean={np.nanmean(d):+.4f} "
                      f"win={np.mean(d > 0):.2f}")
    np.savez("outputs/bench_scgpt_baseline.npz", **store)
    print("\nsaved outputs/bench_scgpt_baseline.npz")


if __name__ == "__main__":
    main()
