"""监督式跨系迁移的 kNN / 学习源权重基线.

kNN: 对每个查询, 在训练折中找源响应最相似的靶, 预测其留出系响应
(knn) 或等权混合加上近邻残差 (knn_resid).
weights: 学习一组全局源权重 (softmax), 最大化训练折上的平均 cosine.

五折交叉验证与 bench_supervised.py 完全一致, 用 canonical PDS 评分.
"""

import argparse
import csv
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import wilcoxon

sys.path.insert(0, "scripts")
from bench_common import Retrieval, boot_ci, load_axis, load_lines, load_nadig_lines
from bench_supervised import (TIERS, build_features, build_targets,
                              load_equal_pds, paired_stats)


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def folds_of(n, n_folds, seed):
    return np.array_split(np.random.default_rng(seed).permutation(n), n_folds)


def knn_predict(Xn, Yn, k, resid, folds, temp=0.1, chunk=128):
    P = np.zeros_like(Yn, dtype=np.float32)
    for val in folds:
        tr = np.setdiff1d(np.arange(len(Xn)), val, assume_unique=False)
        tr = tr[np.linalg.norm(Yn[tr], axis=1) > 1e-6]
        kk = min(k, len(tr))
        Ytr = Yn[tr]
        Xtr = Xn[tr]
        for start in range(0, len(val), chunk):
            vv = val[start:start + chunk]
            S = Xn[vv] @ Xtr.T
            idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            sim = np.take_along_axis(S, idx, axis=1)
            w = np.exp((sim - sim.max(1, keepdims=True)) / temp)
            w = w / w.sum(1, keepdims=True)
            if resid:
                corr = Ytr[idx] - Xtr[idx]
                P[vv] = Xn[vv] + np.einsum("vk,vkd->vd", w, corr)
            else:
                P[vv] = np.einsum("vk,vkd->vd", w, Ytr[idx])
    return P


def weights_predict(lines, srcs, q, pos, Y, folds):
    S = len(srcs)
    d = len(pos)
    Xs = np.zeros((S, len(q), d), dtype=np.float32)
    avail = np.zeros((len(q), S), dtype=bool)
    for s, o in enumerate(srcs):
        for i, t in enumerate(q):
            j = lines[o]["index"].get(t)
            if j is None:
                continue
            v = lines[o]["rows"][j, pos].astype(np.float64)
            vn = np.linalg.norm(v)
            if vn > 1e-9:
                Xs[s, i] = (v / vn).astype(np.float32)
                avail[i, s] = True
    P = np.zeros_like(Y, dtype=np.float32)
    w_folds = []
    for val in folds:
        tr = np.setdiff1d(np.arange(len(q)), val, assume_unique=False)
        tr = tr[np.linalg.norm(Y[tr], axis=1) > 1e-6]
        Ytr = unit(Y[tr].astype(np.float64))
        A = avail[tr].astype(np.float64)
        Xtr = Xs[:, tr].astype(np.float64)
        Xv = Xs[:, val].astype(np.float64)

        def obj(z):
            w = softmax(z)
            num = np.zeros((len(tr), d), dtype=np.float64)
            den = np.zeros(len(tr), dtype=np.float64)
            for s in range(S):
                num += (w[s] * A[:, s])[:, None] * Xtr[s]
                den += w[s] * A[:, s]
            pred = num / np.maximum(den, 1e-9)[:, None]
            cos = (np.sum(pred * Ytr, axis=1) /
                   np.maximum(np.linalg.norm(pred, axis=1), 1e-9))
            return -float(np.mean(cos))

        res = minimize(obj, np.zeros(S), method="Powell",
                       options={"maxiter": 300, "xtol": 1e-3, "ftol": 1e-4})
        w = softmax(res.x)
        w_folds.append(w)
        num = np.zeros((len(val), d), dtype=np.float64)
        den = np.zeros(len(val), dtype=np.float64)
        Av = avail[val].astype(np.float64)
        for s in range(S):
            num += (w[s] * Av[:, s])[:, None] * Xv[s]
            den += w[s] * Av[:, s]
        P[val] = (num / np.maximum(den, 1e-9)[:, None]).astype(np.float32)
        print(f"  weights {np.round(w, 4)} obj {res.fun:.4f}", flush=True)
    return P, w_folds


def run(tier, model, ks, n_folds, seed, only):
    genes, pos, gidx, m = load_axis()
    lines = load_lines(genes, gidx, m) if tier == "B" else load_nadig_lines(genes, gidx)
    lines.pop("_gidx", None)
    all_holdouts, eq_path = TIERS[tier]
    holdouts = [h for h in all_holdouts if h in only] if only else all_holdouts
    eq = load_equal_pds(eq_path)
    summary, flat = [], []

    print(f"\n######## Tier {tier} | {model} | ks {ks} ########", flush=True)
    for L in holdouts:
        if L not in lines or len(lines[L]["names"]) == 0:
            continue
        ref = lines[L]
        srcs = [o for o in all_holdouts if o != L and o in lines
                and len(lines[o]["names"]) > 0]
        q = [t for t in ref["names"]
             if any(t in lines[o]["index"] for o in srcs)]
        print(f"\n=== hold-out {L}: pool {len(ref['names'])}, "
              f"queries {len(q)}, sources {srcs} ===", flush=True)
        if len(q) < 10:
            continue

        X = build_features(lines, srcs, q, pos, "mix")
        Y = build_targets(ref, q, pos)
        folds = folds_of(len(q), n_folds, seed)
        R = Retrieval(ref["rows"], ref["names"], pos, gidx=gidx)

        def score(P, tag):
            P_full = np.zeros((len(q), len(genes)), dtype=np.float32)
            P_full[:, pos] = P
            pds = R.score(P_full, q, exclude="self", denominator="n")
            common = [t for t in q if (L, t) in eq and np.isfinite(eq[(L, t)])]
            ps = np.array([pds[q.index(t)] for t in common])
            pe = np.array([eq[(L, t)] for t in common])
            st = paired_stats(ps, pe)
            st.update(tier=tier, holdout=L, method=tag, n=len(common),
                      mean=float(np.nanmean(ps)), median=float(np.nanmedian(ps)),
                      equal_mean=float(np.nanmean(pe)))
            summary.append(st)
            print(f"  {tag:18s} mean {st['mean']:.4f} (equal {st['equal_mean']:.4f}) "
                  f"delta {st['delta_mean']:+.4f} "
                  f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}] "
                  f"p={st['p']:.2g} win={st['win']:.2f}", flush=True)
            for t, p, e in zip(common, ps, pe):
                flat.append((L, tag, t, p, e))

        if model in ("knn", "knn_resid"):
            Xn = unit(X.astype(np.float64)).astype(np.float32)
            for k in ks:
                P = knn_predict(Xn, Y, k, model == "knn_resid", folds)
                score(P, f"{model}_k{k}")
        elif model == "weights":
            P, _ = weights_predict(lines, srcs, q, pos, Y, folds)
            score(P, "learned_weights")
        del X, Y

    prefix = ("outputs/bench_supervised_v2" if tier == "B"
              else "outputs/bench_supervised_nadig_v2")
    prefix += f"_{model}"
    np.savez(f"{prefix}.npz",
             holdout=np.array([f[0] for f in flat]),
             method=np.array([f[1] for f in flat]),
             target=np.array([f[2] for f in flat]),
             pds_raw=np.array([f[3] for f in flat], dtype=np.float64),
             pds_equal=np.array([f[4] for f in flat], dtype=np.float64))
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nsaved {prefix}.csv / .npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B"], default="B")
    ap.add_argument("--model", choices=["knn", "knn_resid", "weights"],
                    default="knn_resid")
    ap.add_argument("--ks", default="25")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="")
    args = ap.parse_args()
    only = set(args.holdout.split(",")) if args.holdout else None
    ks = [int(k) for k in args.ks.split(",")]
    run(args.tier, args.model, ks, args.folds, args.seed, only)


if __name__ == "__main__":
    main()
