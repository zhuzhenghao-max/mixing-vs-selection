"""External validation on Tahoe-100M pseudobulk drug responses.

Loads per-line npz files built by tahoe_build_line.py. Conditions are aligned
over all lines. Source-subset PDS is cached for every subset so saturation,
exchange-rate, and source-side cross-validation read from one table.

Subcommands:
  cache   precompute PDS for every line x source subset
  select  calibration selection experiment (kmeans/strat/random)
  report  saturation, exchange-rate, and SSCV tables from the cache
"""

import argparse
import csv
import itertools
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "scripts")


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def load_lines(paths, metric="S"):
    lines = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        name = os.path.basename(p)[:-4]
        X = d[metric].astype(np.float32)
        cond = [str(c) for c in d["cond"]]
        lines[name] = dict(
            X=X, cond=cond, index={c: i for i, c in enumerate(cond)},
            n_trt=np.asarray(d["n_trt"]),
            active=(np.abs(X) > 2).mean(axis=1))
        print(f"  {name:12s} {X.shape}")
    return lines


def common_conditions(lines, min_active=0.005):
    names = list(lines)
    common = set(lines[names[0]]["cond"])
    for n in names[1:]:
        common &= set(lines[n]["index"])
    common = sorted(common)
    act = np.mean([np.array([lines[n]["active"][lines[n]["index"][c]]
                             for c in common]) for n in names], axis=0)
    keep = [c for c, a in zip(common, act) if a >= min_active]
    print(f"common {len(common)} -> active {len(keep)}")
    return keep


def pds_fast(P, Y, qidx):
    Yn = unit(Y.astype(np.float32))
    Pn = unit(np.asarray(P, dtype=np.float32))
    S = Yn @ Pn.T
    st = S[qidx, np.arange(len(qidx))]
    rank = (S > st[None, :]).sum(axis=0) + 0.5 * (
        (S == st[None, :]).sum(axis=0) - 1)
    return 1.0 - rank / len(Y)


def exchange_k(grid, vals, g):
    if not np.isfinite(g) or g <= 0:
        return 0.0
    for i in range(1, len(grid)):
        a, b = vals[i - 1], vals[i]
        if np.isfinite(a) and np.isfinite(b) and a < g <= b:
            return float(grid[i - 1] + (g - a) * (grid[i] - grid[i - 1])
                         / (b - a))
    return np.nan


def mix_rows(lines, srcs, conds, metric_rows=None):
    mats = []
    for o in srcs:
        idx = lines[o]["index"]
        rows = np.array([idx[c] for c in conds])
        mats.append(unit(lines[o]["X"][rows]))
    return np.mean(mats, axis=0)


def cache_subsets(lines, common, out):
    names = list(lines)
    rows = {n: np.array([lines[n]["index"][c] for c in common]) for n in names}
    subsets = []
    for r in range(0, len(names) + 1):
        subsets += [frozenset(s) for s in itertools.combinations(names, r)]
    fields = ["line", "subset", "size", "pds"]
    done = set()
    if os.path.exists(out) and os.path.getsize(out) > 0:
        with open(out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["line"], r["subset"]))
        f = open(out, "a", newline="")
        w = csv.DictWriter(f, fieldnames=fields)
        print(f"resume {out}: {len(done)} rows already done")
    else:
        f = open(out, "w", newline="")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
    for L in names:
        srcs = [o for o in names if o != L]
        Y = lines[L]["X"][rows[L]]
        qidx = np.arange(len(common))
        for S in subsets:
            if not S <= set(srcs):
                continue
            if (L, "+".join(sorted(S))) in done:
                continue
            if not S:
                val = 0.0
            else:
                P = mix_rows(lines, sorted(S), common)
                val = float(np.mean(pds_fast(P, Y, qidx)))
            w.writerow(dict(line=L, subset="+".join(sorted(S)),
                            size=len(S), pds=val))
            f.flush()
        print(f"  {L} done", flush=True)
    f.close()
    print(f"wrote {out}")


def load_cache(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["line"], frozenset(r["subset"].split("+"))
                   if r["subset"] else frozenset())
            out[key] = float(r["pds"])
    return out


def kmeans_pick(X, k, rng):
    Xn = unit(X)
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(Xn)
    sel = []
    for c in km.cluster_centers_:
        d = np.linalg.norm(Xn - c, axis=1)
        d[sel] = np.inf
        sel.append(int(np.argmin(d)))
    if len(set(sel)) < k:
        rest = [i for i in range(len(X)) if i not in set(sel)]
        rng.shuffle(rest)
        sel += rest[:k - len(set(sel))]
    return np.array(sel[:k])


def strat_pick(X, k, rng, m_max=50):
    n = len(X)
    k = min(k, n)
    if k <= m_max:
        return kmeans_pick(X, k, rng)
    Xn = unit(X)
    m = min(m_max, k)
    km = KMeans(n_clusters=m, n_init=10, random_state=0).fit(Xn)
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
    dcenter = np.linalg.norm(Xn - km.cluster_centers_[lab], axis=1)
    sel = []
    for j in range(m):
        members = np.where(lab == j)[0]
        if len(members) == 0:
            continue
        a = min(alloc[j], len(members))
        if a == len(members):
            sel.append(members)
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
    return sel[:k]


def source_stats(lines, srcs, rows):
    n = len(rows)
    raw_norm = np.zeros(n)
    agreement = np.ones(n)
    for i, c in enumerate(rows):
        vs = []
        for o in srcs:
            v = lines[o]["X"][lines[o]["index"][c]].astype(np.float64)
            vn = np.linalg.norm(v)
            if vn > 1e-9:
                vs.append((v, vn))
        if vs:
            raw_norm[i] = np.mean([vn for _, vn in vs])
            if len(vs) > 1:
                U = np.stack([v / vn for v, vn in vs])
                C = U @ U.T
                iu = np.triu_indices(len(vs), 1)
                agreement[i] = float(np.mean(C[iu]))
    return dict(raw_norm=raw_norm, agreement=agreement,
                disagreement=1.0 - agreement)


def select(strategy, C, X, stats, k, rng):
    if strategy == "random":
        return rng.choice(len(C), size=k, replace=False)
    sub = {key: v[C] for key, v in stats.items()}
    if strategy == "magnitude":
        order = np.argsort(-sub["raw_norm"])
    elif strategy == "agreement":
        order = np.argsort(-sub["agreement"])
    elif strategy == "disagreement":
        order = np.argsort(-sub["disagreement"])
    elif strategy == "kmeans":
        return kmeans_pick(X[C], k, rng)
    elif strategy == "strat":
        return strat_pick(X[C], k, rng)
    else:
        raise ValueError(strategy)
    return order[:k]


def run_select(args):
    lines = load_lines(args.lines, args.metric)
    common = common_conditions(lines)
    names = list(lines)
    rows = {n: np.array([lines[n]["index"][c] for c in common]) for n in names}
    rng = np.random.default_rng(args.seed)
    n_eval = max(80, len(common) // 4)
    out_fields = ["holdout", "k", "strategy", "split", "delta"]
    src_fields = ["holdout", "added", "split", "delta"]
    delta_fields = ["holdout", "k", "strategy", "split", "cond", "delta"]
    f_out = open(args.out, "w", newline="")
    w_out = csv.DictWriter(f_out, fieldnames=out_fields)
    w_out.writeheader()
    f_src = open(args.out_src, "w", newline="") if args.out_src else None
    w_src = csv.DictWriter(f_src, fieldnames=src_fields) if f_src else None
    if w_src is not None:
        w_src.writeheader()
    f_d = open(args.out_deltas, "w", newline="") if args.out_deltas else None
    w_d = csv.DictWriter(f_d, fieldnames=delta_fields) if f_d else None
    if w_d is not None:
        w_d.writeheader()
    for L in names:
        srcs = [o for o in names if o != L]
        Q = mix_rows(lines, srcs, common)
        Y = lines[L]["X"][rows[L]]
        qidx = np.arange(len(common))
        base = pds_fast(Q, Y, qidx)
        stats = source_stats(lines, srcs, common)
        for split in range(args.splits):
            perm = rng.permutation(len(common))
            E, C = perm[:n_eval], perm[n_eval:]
            for i in srcs:
                rest = [o for o in srcs if o != i]
                Pi = mix_rows(lines, rest, common)
                d = base[E] - pds_fast(Pi[E], Y, qidx[E])
                w_src.writerow(dict(holdout=L, added=i, split=split,
                                    delta=float(np.mean(d))))
                f_src.flush()
            ncomp = min(512, len(C) - 1, Q.shape[1])
            pca = PCA(n_components=ncomp, svd_solver="randomized",
                      random_state=split)
            ZC = pca.fit_transform(Q[C])
            ZE = pca.transform(Q[E])
            for k in args.ks:
                if k >= len(C):
                    continue
                for strat in args.strategies:
                    reps = args.repeats if strat == "random" else 1
                    vals = []
                    acc = np.zeros(len(E))
                    for _ in range(reps):
                        sel = select(strat, C, Q, stats, k, rng)
                        model = RidgeCV(alphas=np.logspace(-2, 4, 13),
                                        fit_intercept=False)
                        model.fit(ZC[sel], Y[qidx[C[sel]]])
                        P = model.predict(ZE)
                        d = pds_fast(P, Y, qidx[E]) - base[E]
                        vals.append(float(np.mean(d)))
                        acc += d
                    acc /= reps
                    w_out.writerow(dict(holdout=L, k=k, strategy=strat,
                                        split=split, delta=np.mean(vals)))
                    f_out.flush()
                    if w_d is not None:
                        for ci, dv in zip(E, acc):
                            w_d.writerow(dict(holdout=L, k=k, strategy=strat,
                                              split=split, cond=int(ci),
                                              delta=float(dv)))
                        f_d.flush()
                    print(f"  {L} k={k:3d} {strat:11s} "
                          f"split{split} {np.mean(vals):+.4f}", flush=True)
        print(f"  {L} done", flush=True)
    f_out.close()
    print(f"wrote {args.out}")
    if f_src is not None:
        f_src.close()
        print(f"wrote {args.out_src}")
    if f_d is not None:
        f_d.close()
        print(f"wrote {args.out_deltas}")


def exchange(args):
    def read(path, fields):
        out = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[tuple(r[k] for k in fields)] = float(r["delta"])
        return out
    cal = read(args.select, ("holdout", "k", "strategy", "split"))
    src = read(args.src, ("holdout", "added", "split"))
    names = sorted({k[0] for k in cal})
    ks = sorted({int(k[1]) for k in cal})
    splits = sorted({int(k[3]) for k in cal})
    strategies = sorted({k[2] for k in cal})
    rows = []
    for L in names:
        added = sorted({k[1] for k in src if k[0] == L})
        for i in added:
            g = np.mean([src[(L, i, str(s))] for s in splits])
            rec = dict(holdout=L, added=i, source_delta=g)
            for strat in strategies:
                vals = [np.mean([cal.get((L, str(k), strat, str(s)),
                                         np.nan) for s in splits])
                        for k in ks]
                rec[f"{strat}_kstar"] = float(exchange_k(
                    [0] + ks, [0.0] + vals, g))
            rows.append(rec)
            print(f"  {L:12s} -{i:12s} dsrc {g:+.4f}  " +
                  "  ".join(f"{s} k*={rec[s + '_kstar']:.0f}"
                            for s in strategies), flush=True)
    print("\n=== per-line medians ===")
    for L in names:
        sub = [r for r in rows if r["holdout"] == L]
        line = f"{L:12s} dsrc {np.median([r['source_delta'] for r in sub]):+.4f}"
        for strat in strategies:
            v = [r[f"{strat}_kstar"] for r in sub]
            line += f"  {strat} k* med {np.median(v):.0f}"
        print(line)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


def report(args):
    cache = load_cache(args.cache)
    names = sorted({k[0] for k in cache})
    # saturation by subset size
    print("\n=== saturation ===")
    for L in names:
        vals = {}
        for (line, S), v in cache.items():
            if line == L:
                vals.setdefault(len(S), []).append(v)
        line = f"{L:12s}"
        for r in sorted(vals):
            if r == 0:
                continue
            line += f"  |S|={r} {np.mean(vals[r]):+.4f}"
        print(line)
    # SSCV: gain of adding i to S (|S|=2), actual vs pseudo
    print("\n=== SSCV ===")
    rows = []
    for L in names:
        srcs = [o for o in names if o != L]
        for S in itertools.combinations(srcs, 2):
            S = frozenset(S)
            base = cache[(L, S)]
            for i in srcs:
                if i in S:
                    continue
                act = cache[(L, S | {i})] - base
                pseudo = []
                for o in S:
                    So = S - {o}
                    if (o, So) not in cache or (o, So | {i}) not in cache:
                        continue
                    pseudo.append(cache[(o, So | {i})] - cache[(o, So)])
                rows.append(dict(holdout=L, state="+".join(sorted(S)),
                                 added=i, actual=act,
                                 pseudo=float(np.mean(pseudo))))
    a = np.array([r["actual"] for r in rows])
    p = np.array([r["pseudo"] for r in rows])
    rho, pv = spearmanr(p, a)
    r, rp = pearsonr(p, a)
    print(f"n={len(rows)} Spearman {rho:+.3f} (p={pv:.2g}) "
          f"Pearson {r:+.3f} (p={rp:.2g}) "
          f"sign {np.mean((p > 0) == (a > 0)):.2f}")
    for L in names:
        sub = [r for r in rows if r["holdout"] == L]
        aa = np.array([r["actual"] for r in sub])
        pp = np.array([r["pseudo"] for r in sub])
        rr = spearmanr(pp, aa).statistic if len(sub) > 2 else np.nan
        print(f"  {L:12s} n={len(sub):3d} rho {rr:+.3f}")
    if args.out_sscv:
        with open(args.out_sscv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out_sscv}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--lines", nargs="+", required=True)
    c.add_argument("--metric", default="S")
    c.add_argument("--out", required=True)
    s = sub.add_parser("select")
    s.add_argument("--lines", nargs="+", required=True)
    s.add_argument("--metric", default="S")
    s.add_argument("--metric-name", default="S")
    s.add_argument("--ks", nargs="+", type=int, default=[20, 50, 100])
    s.add_argument("--splits", type=int, default=3)
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--out-src", default="")
    s.add_argument("--out-deltas", default="")
    s.add_argument("--strategies", nargs="+",
                   default=["random", "kmeans", "strat", "magnitude",
                            "agreement", "disagreement"])
    r = sub.add_parser("report")
    r.add_argument("--cache", required=True)
    r.add_argument("--out-sscv", default="")
    e = sub.add_parser("exchange")
    e.add_argument("--select", required=True)
    e.add_argument("--src", required=True)
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "cache":
        lines = load_lines(args.lines, args.metric)
        common = common_conditions(lines)
        cache_subsets(lines, common, args.out)
    elif args.cmd == "select":
        run_select(args)
    elif args.cmd == "exchange":
        exchange(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
