"""Per-query self-coordinate drop versus target knockdown strength.

Reproduces the equal-mixing sample of bench_variants.py (same seed and
holdout order) and correlates the PDS drop from including the target
coordinate with the query's own target-gene log fold-change.

Outputs: outputs/bench_self_lfc_corr.csv         (per query)
         outputs/bench_self_lfc_corr_summary.csv (per holdout)
"""

import argparse
import csv
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from bench_common import Retrieval, load_axis, load_lines, unit_rows

HOLDS = ["k562", "rpe1", "hct116"]
ORDER = ["h1", "k562", "rpe1", "hct116", "hek293t"]


def bootstrap_ci(x, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        rho = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(rho):
            vals.append(rho)
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    genes, pos, gidx, m = load_axis()
    lines = load_lines(genes, gidx, m)
    lines.pop("_gidx", None)
    posset = set(int(p) for p in pos)

    per_query = []
    summary = []
    for L in HOLDS:
        ref = lines[L]
        srcs = [o for o in ORDER if o in lines and o != L]
        q = [t for t in ref["names"] if any(t in lines[o]["index"] for o in srcs)]
        q = list(rng.permutation(q))[:args.limit]

        P = np.zeros((len(q), len(genes)))
        q_self = np.full(len(q), np.nan)
        src_self = np.full(len(q), np.nan)
        pred_t = np.full(len(q), np.nan)
        excludable = np.zeros(len(q), bool)
        for i, t in enumerate(q):
            os_ = [o for o in srcs if t in lines[o]["index"]]
            rows = {o: lines[o]["rows"][lines[o]["index"][t]] for o in os_}
            P[i] = np.mean([unit_rows(rows[o][None, :])[0] for o in os_], axis=0)
            j = gidx.get(t)
            if t in ref["index"]:
                q_self[i] = ref["self_lfc"][ref["index"][t]]
            if j is not None:
                pred_t[i] = P[i, j]
                excludable[i] = int(j) in posset
            sl = [lines[o]["self_lfc"][lines[o]["index"][t]] for o in os_]
            sl = [v for v in sl if np.isfinite(v)]
            if sl:
                src_self[i] = float(np.mean(sl))

        R0 = Retrieval(ref["rows"], ref["names"], pos)
        R1 = Retrieval(ref["rows"], ref["names"], pos, gidx=gidx)
        p0 = R0.score(P, q, exclude="none")
        p1 = R1.score(P, q, exclude="self")
        drop = p0 - p1

        ok = excludable & np.isfinite(q_self) & np.isfinite(drop)
        rho_all = spearmanr(drop[np.isfinite(q_self)], q_self[np.isfinite(q_self)])
        rho = spearmanr(drop[ok], q_self[ok])
        lo, hi = bootstrap_ci(drop[ok], q_self[ok], args.boot, args.seed + 1)

        print(f"\n=== {L}: sampled {len(q)}, excludable {int(ok.sum())} ===")
        print(f"  PDS none med {np.nanmedian(p0):.4f} mean {np.nanmean(p0):.4f}; "
              f"self med {np.nanmedian(p1):.4f} mean {np.nanmean(p1):.4f}")
        print(f"  per-query drop: all med {np.nanmedian(drop):+.4f} "
              f"mean {np.nanmean(drop):+.4f}; excludable med "
              f"{np.nanmedian(drop[ok]):+.4f} mean {np.nanmean(drop[ok]):+.4f}")
        print(f"  self-LFC excludable med {np.nanmedian(q_self[ok]):+.3f} "
              f"(frac < -1 {np.nanmean(q_self[ok] < -1):.3f})")
        print(f"  Spearman(drop, self-LFC): all {rho_all.statistic:+.3f} "
              f"(p {rho_all.pvalue:.1e}, n {len(drop[np.isfinite(q_self)])}); "
              f"excludable {rho.statistic:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}] p {rho.pvalue:.1e} n {int(ok.sum())}")

        summary.append([
            L, len(q), int(ok.sum()),
            f"{np.nanmedian(p0):.4f}", f"{np.nanmean(p0):.4f}",
            f"{np.nanmedian(p1):.4f}", f"{np.nanmean(p1):.4f}",
            f"{np.nanmedian(drop[ok]):+.4f}", f"{np.nanmean(drop[ok]):+.4f}",
            f"{np.nanmedian(q_self[ok]):+.4f}",
            f"{rho_all.statistic:+.4f}", f"{rho_all.pvalue:.3e}",
            f"{rho.statistic:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
            f"{rho.pvalue:.3e}",
        ])
        for i, t in enumerate(q):
            per_query.append([
                L, t, f"{p0[i]:.6f}", f"{p1[i]:.6f}", f"{drop[i]:+.6f}",
                "" if not np.isfinite(q_self[i]) else f"{q_self[i]:+.6f}",
                "" if not np.isfinite(src_self[i]) else f"{src_self[i]:+.6f}",
                "" if not np.isfinite(pred_t[i]) else f"{pred_t[i]:+.6f}",
                int(excludable[i]),
            ])

    with open("outputs/bench_self_lfc_corr.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["holdout", "target", "pds_none", "pds_self", "drop",
                    "query_self_lfc", "source_mean_self_lfc", "pred_target_coord",
                    "excludable"])
        w.writerows(per_query)

    with open("outputs/bench_self_lfc_corr_summary.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["holdout", "n_sampled", "n_excludable",
                    "pds_none_median", "pds_none_mean",
                    "pds_self_median", "pds_self_mean",
                    "drop_median_excludable", "drop_mean_excludable",
                    "self_lfc_median_excludable",
                    "spearman_all", "p_all",
                    "spearman_excludable", "ci_lo", "ci_hi", "p_excludable"])
        w.writerows(summary)

    print("\nsaved outputs/bench_self_lfc_corr.csv / _summary.csv")


if __name__ == "__main__":
    main()
