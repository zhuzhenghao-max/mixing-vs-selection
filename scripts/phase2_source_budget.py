"""Phase 2: source-line count versus calibration budget.

For each holdout, fix the query set to targets covered by every remaining
line and build the equal mix of every non-empty source subset. Score each
equal mix with the canonical PDS, then calibrate at every subset with k
kmeans-selected or random calibration perturbations. The exchange rate is
the k at which adding k target-line calibration perturbations matches the
PDS gain of adding the missing source line.

A single PCA per split, fitted on the full-source features, is reused for
all subsets so that subset comparisons share one feature basis.
"""

import argparse
import csv
import itertools
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "scripts")
from bench_common import boot_ci, load_axis, load_lines, load_nadig_lines
from bench_supervised import TIERS, build_features, build_targets
from phase1_selection_pilot import FastRetrieval, _unit


def kmeans_sel(Xn, k, rng):
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(Xn)
    sel = []
    for c in km.cluster_centers_:
        d = np.linalg.norm(Xn - c, axis=1)
        d[sel] = np.inf
        sel.append(int(np.argmin(d)))
    if len(set(sel)) < k:
        rest = [i for i in range(len(Xn)) if i not in set(sel)]
        rng.shuffle(rest)
        sel += rest[:k - len(set(sel))]
    return np.array(sel[:k])


def subset_key(S):
    return "-".join(str(i) for i in S)


def exchange_k(grid, vals, g):
    if not np.isfinite(g) or g <= 0:
        return 0.0
    for i in range(1, len(grid)):
        a, b = vals[i - 1], vals[i]
        if np.isfinite(a) and np.isfinite(b) and a < g <= b:
            return float(grid[i - 1] + (g - a) * (grid[i] - grid[i - 1])
                         / (b - a))
    return np.nan


def mean_ci(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    lo, hi, mu = boot_ci(a, n=4000, seed=0)
    return mu, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B"], default="A")
    ap.add_argument("--holdout", default="")
    ap.add_argument("--ks", default="20,50,100,200")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--eval", type=int, default=150)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--calib-min-src", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    genes, pos, gidx, m = load_axis()
    lines = (load_lines(genes, gidx, m) if args.tier == "B"
             else load_nadig_lines(genes, gidx))
    lines.pop("_gidx", None)
    all_holdouts = TIERS[args.tier][0]
    holdouts = (args.holdout.split(",") if args.holdout
                else list(all_holdouts))

    summary = []
    for L in holdouts:
        if L not in lines or len(lines[L]["names"]) == 0:
            continue
        ref = lines[L]
        srcs = [o for o in all_holdouts if o != L and o in lines
                and len(lines[o]["names"]) > 0]
        q = [t for t in ref["names"]
             if all(t in lines[o]["index"] for o in srcs)]
        print(f"\n=== {args.tier}:{L} | sources {srcs} | "
              f"covered by all {len(q)} ===", flush=True)
        if len(q) < 60 or len(srcs) < 2:
            print("  skipped: too few queries or sources", flush=True)
            continue
        n_eval = min(args.eval, max(20, len(q) // 3))

        Y = build_targets(ref, q, pos)
        R = FastRetrieval(ref["rows"], ref["names"], pos, gidx)
        rng = np.random.default_rng(args.seed)
        subsets = [tuple(c) for r in range(1, len(srcs) + 1)
                   for c in itertools.combinations(range(len(srcs)), r)]
        full = tuple(range(len(srcs)))

        Xs, eq_pds = {}, {}
        for S in subsets:
            Xs[S] = build_features(lines, [srcs[i] for i in S], q, pos, "mix")
            eq_pds[S] = R.score_pos(Xs[S], q)
        print("  equal-mix PDS: " + "  ".join(
            f"[{subset_key(S)}] {np.nanmean(eq_pds[S]):+.4f}"
            for S in subsets), flush=True)
        for S in subsets:
            summary.append(dict(
                tier=args.tier, holdout=L, subset=subset_key(S),
                srcs=",".join(srcs[i] for i in S), n_src=len(S),
                strategy="equal", k=0, delta=0.0, lo=np.nan, hi=np.nan,
                pds=float(np.nanmean(eq_pds[S])),
                eq=float(np.nanmean(eq_pds[S])), added=""))

        splits = []
        for split in range(args.splits):
            perm = rng.permutation(len(q))
            E = perm[:n_eval]
            C = perm[n_eval:]
            ncomp = min(1024, len(C) - 1, Xs[full].shape[1])
            pca = PCA(n_components=ncomp, svd_solver="randomized",
                      random_state=args.seed).fit(Xs[full][C])
            ZC = {S: pca.transform(Xs[S][C]) for S in subsets}
            ZE = {S: pca.transform(Xs[S][E]) for S in subsets}
            Xn = {S: _unit(Xs[S][C]) for S in subsets}
            splits.append((E, ZC, ZE, Xn, Y[C]))
            print(f"  split {split}: eval {len(E)} calib {len(C)} "
                  f"comp {ncomp}", flush=True)

        calib_subsets = [S for S in subsets
                         if len(S) >= args.calib_min_src]
        for S in calib_subsets:
            for k in ks:
                for strat in ("kmeans", "random"):
                    reps = args.repeats if strat == "random" else 1
                    pq, pds_pq = {}, {}
                    for (E, ZC, ZE, Xn, YC) in splits:
                        pds_reps = []
                        for _ in range(reps):
                            if strat == "kmeans":
                                sel = kmeans_sel(Xn[S], k, rng)
                            else:
                                sel = rng.choice(len(Xn[S]), size=k,
                                                 replace=False)
                            model = RidgeCV(
                                alphas=np.logspace(-2, 4, 13),
                                fit_intercept=False)
                            model.fit(ZC[S][sel], YC[sel])
                            P = model.predict(ZE[S])
                            pds_reps.append(
                                R.score_pos(P, [q[i] for i in E]))
                        pds = np.nanmean(np.stack(pds_reps), axis=0)
                        for i, v, pv in zip(E, pds - eq_pds[S][E], pds):
                            pq.setdefault(int(i), []).append(float(v))
                            pds_pq.setdefault(int(i), []).append(float(pv))
                    mu, lo, hi = mean_ci([np.mean(v) for v in pq.values()])
                    summary.append(dict(
                        tier=args.tier, holdout=L, subset=subset_key(S),
                        srcs=",".join(srcs[i] for i in S), n_src=len(S),
                        strategy=strat, k=k, delta=mu, lo=lo, hi=hi,
                        pds=float(np.nanmean([np.mean(v)
                                              for v in pds_pq.values()])),
                        eq=float(np.nanmean(eq_pds[S])), added=""))
                    print(f"  [{subset_key(S):5s}] k={k:3d} {strat:6s} "
                          f"delta {mu:+.4f} [{lo:+.4f},{hi:+.4f}] "
                          f"pds {summary[-1]['pds']:+.4f}", flush=True)

        for S in subsets:
            if len(S) >= len(srcs):
                continue
            for i in range(len(srcs)):
                if i in S:
                    continue
                Sp = tuple(sorted(S + (i,)))
                mu, lo, hi = mean_ci(eq_pds[Sp] - eq_pds[S])
                summary.append(dict(
                    tier=args.tier, holdout=L, subset=subset_key(S),
                    srcs=",".join(srcs[j] for j in S), n_src=len(S),
                    strategy="equal", k=-1, delta=mu, lo=lo, hi=hi,
                    pds=float(np.nanmean(eq_pds[Sp])),
                    eq=float(np.nanmean(eq_pds[S])), added=srcs[i]))
                print(f"  add {srcs[i]:6s} to [{subset_key(S):5s}] "
                      f"source gain {mu:+.4f} [{lo:+.4f},{hi:+.4f}]",
                      flush=True)

    prefix = f"outputs/phase2_source_budget_{args.tier}"
    keys = ["tier", "holdout", "subset", "srcs", "n_src", "strategy", "k",
            "delta", "lo", "hi", "pds", "eq", "added"]
    with open(f"{prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {prefix}.csv")

    grid = [0] + ks
    print("\nExchange rate: k* calibration perturbations ~ one source line")
    for L in holdouts:
        rows = [r for r in summary if r["holdout"] == L]
        if not rows:
            continue
        n_src_tot = max(r["n_src"] for r in rows)
        for S in sorted({r["subset"] for r in rows},
                        key=lambda s: (s.count("-"), s)):
            if S.count("-") + 1 != n_src_tot - 1:
                continue
            state = [r for r in rows if r["subset"] == S]
            src_gains = [r for r in state
                         if r["strategy"] == "equal" and r["k"] == -1]
            if not src_gains:
                continue
            best = max(src_gains, key=lambda r: r["delta"])
            for strat in ("kmeans", "random"):
                curve = {r["k"]: r["delta"] for r in state
                         if r["strategy"] == strat}
                vals = [0.0] + [curve.get(k, np.nan) for k in ks]
                ks_best = exchange_k(grid, vals, best["delta"])
                ks_all = [exchange_k(grid, vals, sg["delta"])
                          for sg in src_gains]
                fmt = (f"{ks_best:.0f}" if np.isfinite(ks_best)
                       else f">{max(ks)}")
                print(f"  {L:8s} [{S:5s}] best +{best['added']:6s} "
                      f"gain {best['delta']:+.4f} | {strat:6s} k* {fmt} "
                      f"| all {ks_all}")
    print("\ndone")


if __name__ == "__main__":
    main()
