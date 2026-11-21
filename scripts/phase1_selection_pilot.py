"""Phase 1 pilot: does the choice of calibration perturbations matter?

For each held-out line, split its queries into a calibration pool C and an
evaluation set E. Select k perturbations from C using source-only features,
fit a PCA + ridge calibration map on those k, and score the remaining
queries with the canonical PDS. Compare eight selection rules against
random selection at the same budget.

All selection rules use source-line responses only; the target line's
responses are used only as labels for the k selected calibration points.
"""

import argparse
import csv
import sys

import numpy as np
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "scripts")
from bench_common import boot_ci, load_axis, load_lines, load_nadig_lines
from bench_supervised import TIERS, build_features, build_targets, load_equal_pds


def source_stats(lines, srcs, q, pos):
    n = len(q)
    n_src = np.zeros(n)
    raw_norm = np.zeros(n)
    agreement = np.ones(n)
    for i, t in enumerate(q):
        vs = []
        for o in srcs:
            j = lines[o]["index"].get(t)
            if j is None:
                continue
            v = lines[o]["rows"][j, pos].astype(np.float64)
            vn = np.linalg.norm(v)
            if vn > 1e-9:
                vs.append((v, vn))
        n_src[i] = len(vs)
        if vs:
            raw_norm[i] = np.mean([vn for _, vn in vs])
            if len(vs) > 1:
                U = np.stack([v / vn for v, vn in vs])
                C = U @ U.T
                iu = np.triu_indices(len(vs), 1)
                agreement[i] = float(np.mean(C[iu]))
    return dict(n_src=n_src, raw_norm=raw_norm,
                agreement=agreement, disagreement=1.0 - agreement)


class FastRetrieval:
    """PDS with exact self-coordinate exclusion, float32 matmul."""

    def __init__(self, ref_rows, ref_names, pos, gidx):
        self.ridx = {t: i for i, t in enumerate(ref_names)}
        self.pos = pos
        self.gidx = gidx
        self.local = {int(g): i for i, g in enumerate(pos)}
        R = ref_rows[:, pos].astype(np.float32)
        self.Rn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
        self.n = len(ref_names)

    def score(self, qfull, qnames):
        return self.score_pos(qfull[:, self.pos], qnames)

    def score_pos(self, qpos, qnames):
        Q = np.asarray(qpos, dtype=np.float32).copy()
        cols = []
        for k, t in enumerate(qnames):
            j = self.gidx.get(t)
            li = self.local.get(j) if j is not None else None
            cols.append(li)
            if li is not None:
                Q[k, li] = 0.0
        Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
        S = self.Rn @ Qn.T
        for k, li in enumerate(cols):
            if li is not None:
                S[:, k] /= np.sqrt(np.maximum(
                    1.0 - self.Rn[:, li] ** 2, 1e-12))
        out = np.full(len(qnames), np.nan, dtype=np.float64)
        for k, t in enumerate(qnames):
            ri = self.ridx.get(t)
            if ri is None:
                continue
            col = S[:, k]
            sv = col[ri]
            rank0 = np.sum(col > sv) + 0.5 * (np.sum(col == sv) - 1)
            out[k] = 1.0 - rank0 / self.n
        return out


def _unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def diversity_select(X, idx, k, weight=None):
    Xn = _unit(X[idx])
    m = Xn.mean(0)
    m = m / max(np.linalg.norm(m), 1e-9)
    first = int(np.argmin(Xn @ m))
    sel = [first]
    mind = 1.0 - Xn @ Xn[first]
    for _ in range(k - 1):
        s = mind if weight is None else mind * (1.0 + weight[idx])
        s[np.array(sel)] = -np.inf
        nxt = int(np.argmax(s))
        sel.append(nxt)
        mind = np.minimum(mind, 1.0 - Xn @ Xn[nxt])
    return idx[np.array(sel)]


def dopt_select(X, idx, k, d=128, lam=1.0):
    """Greedy D-optimal subset on a low-rank source-feature design.

    Maximizes log det(Z_S^T Z_S + lam I) in a standardized PCA space of the
    calibration pool, using Sherman-Morrison rank-one updates. The criterion
    targets the linear calibration map's coefficient variance rather than
    geometric coverage.
    """
    ncomp = min(d, len(idx) - 1, X.shape[1])
    Z = PCA(n_components=ncomp, svd_solver="randomized",
            random_state=0).fit_transform(X[idx])
    Z = Z / np.maximum(Z.std(0, keepdims=True), 1e-9)
    M = np.eye(Z.shape[1]) / lam
    sel = []
    for _ in range(k):
        lev = np.einsum("ij,ij->i", Z @ M, Z)
        lev[sel] = -np.inf
        j = int(np.argmax(lev))
        sel.append(j)
        z = Z[j]
        Mz = M @ z
        M -= np.outer(Mz, Mz) / (1.0 + z @ Mz)
    return idx[np.array(sel)]


def strat_select(X, idx, k, rng, n_init=10, m_max=50):
    """Proportional stratified representative sampling.

    At k <= m, one representative per cluster (same as kmeans). Above that,
    allocate k points across a fixed number of clusters proportional to
    cluster size and spread them within each cluster by distance quantiles.
    This keeps coverage at small k while converging toward random sampling
    as k grows, avoiding the equal-allocation bias of one-point-per-cluster.
    """
    n = len(idx)
    k = min(k, n)
    if k >= n:
        return idx
    Xn = _unit(X[idx])
    m = min(m_max, k)
    if k <= m:
        return kmeans_select(X, idx, k, rng, n_init=n_init)
    km = KMeans(n_clusters=m, n_init=n_init, random_state=0).fit(Xn)
    lab = km.labels_
    sizes = np.bincount(lab, minlength=m)
    shares = k * sizes / float(n)
    alloc = np.minimum(np.floor(shares).astype(int), sizes)
    rem = shares - np.floor(shares)
    while alloc.sum() < k:
        cand = np.where(alloc < sizes)[0]
        if len(cand) == 0:
            break
        j = cand[np.argmax(rem[cand])]
        alloc[j] += 1
        rem[j] = -1.0
    for j in np.where((alloc == 0) & (sizes > 0))[0]:
        donors = np.where(alloc > 1)[0]
        if len(donors) == 0:
            break
        d = donors[np.argmax(alloc[donors])]
        alloc[d] -= 1
        alloc[j] = 1
    dcenter = np.linalg.norm(Xn - km.cluster_centers_[lab], axis=1)
    sel = []
    for j in range(m):
        members = np.where(lab == j)[0]
        if len(members) == 0:
            continue
        a = min(alloc[j], len(members))
        if a == len(members):
            take = members
        else:
            order = members[np.argsort(dcenter[members])]
            pos = np.linspace(0, len(order) - 1, a).round().astype(int)
            take = order[np.unique(pos)]
            if len(take) < a:
                rest = np.setdiff1d(order, take)
                take = np.concatenate([take, rest[:a - len(take)]])
        sel.append(take)
    sel = np.concatenate(sel) if sel else np.array([], dtype=int)
    if len(sel) < k:
        rest = np.setdiff1d(np.arange(n), sel)
        rng.shuffle(rest)
        sel = np.concatenate([sel, rest[:k - len(sel)]])
    return idx[sel[:k]]


def kmeans_select(X, idx, k, rng, n_init=10):
    Xn = _unit(X[idx])
    km = KMeans(n_clusters=k, n_init=n_init, random_state=0).fit(Xn)
    sel = []
    for c in km.cluster_centers_:
        d = np.linalg.norm(Xn - c, axis=1)
        d[sel] = np.inf
        sel.append(int(np.argmin(d)))
    if len(set(sel)) < k:
        rest = [i for i in range(len(idx)) if i not in set(sel)]
        rng.shuffle(rest)
        sel += rest[:k - len(set(sel))]
    return idx[np.array(sel[:k])]


def select(strategy, C, X, stats, k, rng, ninit=10):
    if strategy == "random":
        return rng.choice(C, size=k, replace=False)
    if strategy == "shared":
        order = np.lexsort((-stats["raw_norm"][C], -stats["n_src"][C]))
    elif strategy == "magnitude":
        order = np.argsort(-stats["raw_norm"][C])
    elif strategy == "agreement":
        order = np.argsort(-stats["agreement"][C])
    elif strategy == "disagreement":
        order = np.argsort(-stats["disagreement"][C])
    elif strategy == "diversity":
        return diversity_select(X, C, k)
    elif strategy == "hybrid":
        return diversity_select(X, C, k, weight=stats["disagreement"])
    elif strategy == "kmeans":
        return kmeans_select(X, C, k, rng, n_init=ninit)
    elif strategy == "dopt":
        return dopt_select(X, C, k)
    elif strategy == "strat":
        return strat_select(X, C, k, rng, n_init=ninit)
    else:
        raise ValueError(strategy)
    return C[order[:k]]


def paired_delta(p, q):
    d = np.asarray(p) - np.asarray(q)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return dict(n=0, delta=np.nan, lo=np.nan, hi=np.nan, p=np.nan, win=np.nan)
    lo, hi, mu = boot_ci(d, n=4000, seed=0)
    pval = (float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
    return dict(n=len(d), delta=mu, lo=lo, hi=hi, p=pval, win=float(np.mean(d > 0)))


def run_holdout(tier, L, n_splits, n_eval, k, repeats, seed, strategies,
                residual=False, ninit=10):
    genes, pos, gidx, m = load_axis()
    if tier == "B":
        lines = load_lines(genes, gidx, m)
    else:
        lines = load_nadig_lines(genes, gidx)
    lines.pop("_gidx", None)
    ref = lines[L]
    all_h, eq_path = TIERS[tier]
    srcs = [o for o in all_h if o != L and o in lines
            and len(lines[o]["names"]) > 0]
    q = [t for t in ref["names"]
         if any(t in lines[o]["index"] for o in srcs)]
    eq = load_equal_pds(eq_path)
    keep = [i for i, t in enumerate(q)
            if (L, t) in eq and np.isfinite(eq[(L, t)])]
    q = [q[i] for i in keep]
    n_eval = min(n_eval, max(20, len(q) // 2))
    print(f"\n=== {tier}:{L} | pool {len(ref['names'])} queries {len(q)} "
          f"| k={k} splits={n_splits} eval={n_eval} ===", flush=True)

    X = build_features(lines, srcs, q, pos, "mix")
    Y = build_targets(ref, q, pos)
    stats = source_stats(lines, srcs, q, pos)
    for key in stats:
        stats[key] = stats[key][keep]
    R = FastRetrieval(ref["rows"], ref["names"], pos, gidx)
    rng = np.random.default_rng(seed)

    per_query = {s: {} for s in strategies}
    per_split = {s: [] for s in strategies}
    for split in range(n_splits):
        perm = rng.permutation(len(q))
        E = perm[:n_eval]
        C = perm[n_eval:]
        ncomp = min(1024, len(C) - 1, X.shape[1])
        pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=seed)
        ZC = pca.fit_transform(X[C])
        ZE = pca.transform(X[E])
        YC = Y[C]
        eq_e = np.array([eq[(L, q[i])] for i in E])
        pos_in_C = {int(v): j for j, v in enumerate(C)}
        for strat in strategies:
            reps = repeats if strat == "random" else 1
            pds_all = []
            for _ in range(reps):
                sel = select(strat, C, X, stats, k, rng, ninit=ninit)
                sel_pos = np.array([pos_in_C[int(v)] for v in sel])
                model = RidgeCV(alphas=np.logspace(-2, 4, 13),
                                fit_intercept=False)
                if residual:
                    model.fit(ZC[sel_pos], YC[sel_pos] - X[C][sel_pos])
                else:
                    model.fit(ZC[sel_pos], YC[sel_pos])
                P = model.predict(ZE)
                if residual:
                    P = P + X[E]
                P_full = np.zeros((len(E), len(genes)), dtype=np.float32)
                P_full[:, pos] = P
                pds_all.append(R.score(P_full, [q[i] for i in E]))
            pds = np.nanmean(np.stack(pds_all), axis=0)
            d = pds - eq_e
            per_split[strat].append(float(np.nanmean(d)))
            for i, v in zip(E, d):
                per_query[strat].setdefault(int(i), []).append(float(v))
        print(f"  split {split}: " + "  ".join(
            f"{s} {per_split[s][-1]:+.4f}" for s in strategies), flush=True)

    rows = []
    random_pq = per_query["random"]
    for strat in strategies:
        split_vals = np.array(per_split[strat])
        pq = np.array([np.mean(v) for v in per_query[strat].values()])
        # per-query paired comparison against random, over common queries
        common = sorted(set(per_query[strat]) & set(random_pq))
        a = np.array([np.mean(per_query[strat][i]) for i in common])
        b = np.array([np.mean(random_pq[i]) for i in common])
        st = paired_delta(a, b)
        rows.append(dict(
            tier=tier, holdout=L, k=k, strategy=strat,
            delta_split_mean=float(np.mean(split_vals)),
            delta_split_sd=float(np.std(split_vals)),
            n_common=st["n"], vs_random=st["delta"],
            vs_random_lo=st["lo"], vs_random_hi=st["hi"],
            vs_random_p=st["p"], win_vs_random=st["win"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B"], default="A")
    ap.add_argument("--holdout", default="k562,rpe1")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--eval", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategies", default="random,shared,magnitude,agreement,"
                                           "disagreement,diversity,hybrid,kmeans")
    ap.add_argument("--tag", default="")
    ap.add_argument("--residual", type=int, choices=[0, 1], default=0)
    ap.add_argument("--ninit", type=int, default=10)
    args = ap.parse_args()
    strategies = args.strategies.split(",")
    all_rows = []
    for L in args.holdout.split(","):
        all_rows += run_holdout(args.tier, L, args.splits, args.eval,
                                args.k, args.repeats, args.seed, strategies,
                                residual=bool(args.residual),
                                ninit=args.ninit)

    print("\nSummary (delta PDS vs equal mixing)")
    print(f"{'tier':4s} {'line':8s} {'strategy':12s} {'delta':>8s} "
          f"{'sd':>7s} {'vs_rnd':>8s} {'CI':>18s} {'p':>9s} {'win':>5s}")
    for r in all_rows:
        print(f"{r['tier']:4s} {r['holdout']:8s} {r['strategy']:12s} "
              f"{r['delta_split_mean']:+8.4f} {r['delta_split_sd']:7.4f} "
              f"{r['vs_random']:+8.4f} "
              f"[{r['vs_random_lo']:+.4f},{r['vs_random_hi']:+.4f}] "
              f"{r['vs_random_p']:9.2g} {r['win_vs_random']:5.2f}")

    for L in args.holdout.split(","):
        sub = [r for r in all_rows if r["holdout"] == L]
        vals = [r["delta_split_mean"] for r in sub]
        best = max(sub, key=lambda r: r["delta_split_mean"])
        rnd = [r for r in sub if r["strategy"] == "random"][0]
        print(f"\n{L}: spread {max(vals) - min(vals):+.4f}; best "
              f"{best['strategy']} {best['delta_split_mean']:+.4f}; "
              f"best-random {best['delta_split_mean'] - rnd['delta_split_mean']:+.4f}")

    suffix = f"_{args.tag}" if args.tag else ""
    if args.residual:
        suffix += "_resid"
    out = f"outputs/phase1_selection_pilot_{args.tier}_k{args.k}{suffix}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
