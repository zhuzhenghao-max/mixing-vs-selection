"""监督式跨系迁移基线: 用留出系参考池中除查询靶外的配对数据学习映射.

对每个留出系, 以查询靶的源响应特征 (等权混合或按源拼接) 为输入,
以留出系的单位化响应为输出, 在其余靶上拟合 PCA + RidgeCV, 再预测
当前折的靶. 五折交叉验证保证每个查询靶的响应都不出现在自己的训练集.
预测用 canonical PDS 评分 (剔除靶基因自身坐标, 分母 n).

用法:
  python scripts/bench_supervised.py --tier B --mode mix
  python scripts/bench_supervised.py --tier A --mode concat
"""

import argparse
import csv
import sys

import numpy as np
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "scripts")
from bench_common import (Retrieval, boot_ci, load_axis, load_lines,
                          load_nadig_lines)

TIERS = {
    "B": (["h1", "k562", "rpe1", "hct116", "hek293t"],
          "outputs/bench_transfer_v2.npz"),
    "A": (["k562", "rpe1", "hepg2", "jurkat"],
          "outputs/bench_transfer_nadig_v2.npz"),
}


def build_features(lines, srcs, q, pos, mode):
    d = len(pos)
    if mode == "mix":
        X = np.zeros((len(q), d), dtype=np.float32)
        for i, t in enumerate(q):
            acc = np.zeros(d, dtype=np.float64)
            cnt = 0
            for o in srcs:
                j = lines[o]["index"].get(t)
                if j is None:
                    continue
                v = lines[o]["rows"][j, pos].astype(np.float64)
                vn = np.linalg.norm(v)
                if vn > 1e-9:
                    acc += v / vn
                    cnt += 1
            if cnt:
                X[i] = (acc / cnt).astype(np.float32)
        return X

    X = np.zeros((len(q), len(srcs) * d), dtype=np.float32)
    for i, t in enumerate(q):
        for k, o in enumerate(srcs):
            j = lines[o]["index"].get(t)
            if j is None:
                continue
            v = lines[o]["rows"][j, pos].astype(np.float64)
            vn = np.linalg.norm(v)
            if vn > 1e-9:
                X[i, k * d:(k + 1) * d] = (v / vn).astype(np.float32)
    return X


def build_targets(ref, q, pos):
    Y = np.zeros((len(q), len(pos)), dtype=np.float32)
    for i, t in enumerate(q):
        v = ref["rows"][ref["index"][t], pos].astype(np.float64)
        vn = np.linalg.norm(v)
        if vn > 1e-9:
            Y[i] = (v / vn).astype(np.float32)
    return Y


def zero_self(X, Y, q, gidx, pos):
    """置零每行靶基因自身坐标并重归一化, 用于检验监督映射是否靠自坐标泄漏."""
    local = {int(g): i for i, g in enumerate(pos)}
    for i, t in enumerate(q):
        li = local.get(gidx.get(t))
        if li is None:
            continue
        X[i, li] = 0.0
        Y[i, li] = 0.0
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9)
    return X, Y


def cv_predict(X, Y, n_folds=5, n_comp=256, seed=0, intercept=True,
               shuffle=False):
    n = X.shape[0]
    folds = np.array_split(np.random.default_rng(seed).permutation(n), n_folds)
    P = np.zeros_like(Y, dtype=np.float32)
    for fi, val in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), val, assume_unique=False)
        tr = tr[np.linalg.norm(Y[tr], axis=1) > 1e-6]
        Ytr = Y[tr]
        if shuffle:
            Ytr = Ytr[np.random.default_rng(seed + 101 * fi)
                      .permutation(len(tr))]
        k = min(n_comp, len(tr) - 1, X.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
        Ztr = pca.fit_transform(X[tr])
        Zv = pca.transform(X[val])
        model = RidgeCV(alphas=np.logspace(-2, 4, 13),
                        fit_intercept=intercept)
        model.fit(Ztr, Ytr)
        P[val] = model.predict(Zv).astype(np.float32)
        print(f"  fold {fi}: train {len(tr)} val {len(val)} "
              f"comp {k} alpha {model.alpha_:.3g}", flush=True)
    return P


def load_equal_pds(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    for h, m, t, p in zip(z["holdout"].astype(str), z["method"].astype(str),
                          z["target"].astype(str), z["pds_raw"]):
        if m == "equal":
            out[(h, t)] = float(p)
    return out


def paired_stats(p, q):
    d = np.asarray(p) - np.asarray(q)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return dict(n=0, delta_mean=np.nan, delta_median=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, p=np.nan, win=np.nan)
    lo, hi, mu = boot_ci(d, n=4000, seed=0)
    pval = (float(wilcoxon(d, alternative="two-sided").pvalue)
            if np.any(d != 0) else 1.0)
    return dict(n=len(d), delta_mean=mu, delta_median=float(np.median(d)),
                ci_lo=lo, ci_hi=hi, p=pval, win=float(np.mean(d > 0)))


def run_tier(tier, mode, n_folds, n_comp, seed, only=None, intercept=True,
             selfzero=False, shuffle=False):
    genes, pos, gidx, m = load_axis()
    if tier == "B":
        lines = load_lines(genes, gidx, m)
    else:
        lines = load_nadig_lines(genes, gidx)
    lines.pop("_gidx", None)
    all_holdouts, eq_path = TIERS[tier]
    holdouts = ([h for h in all_holdouts if h in only] if only
                else list(all_holdouts))
    eq = load_equal_pds(eq_path)

    summary, flat_h, flat_m, flat_t, flat_p, flat_e = [], [], [], [], [], []
    print(f"\n######## Tier {tier} | mode {mode} | "
          f"{n_folds}-fold | {n_comp} PCs ########", flush=True)

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
            print("  skipped: too few queries", flush=True)
            continue

        X = build_features(lines, srcs, q, pos, mode)
        Y = build_targets(ref, q, pos)
        if selfzero:
            X, Y = zero_self(X, Y, q, gidx, pos)
        print(f"  features {X.shape}, targets {Y.shape}", flush=True)

        P = cv_predict(X, Y, n_folds=n_folds, n_comp=n_comp, seed=seed,
                       intercept=intercept, shuffle=shuffle)
        del X, Y

        R = Retrieval(ref["rows"], ref["names"], pos, gidx=gidx)
        P_full = np.zeros((len(q), len(genes)), dtype=np.float32)
        P_full[:, pos] = P
        pds = R.score(P_full, q, exclude="self", denominator="n")
        del P, P_full

        method = f"ridge_{mode}"
        common = [t for t in q if (L, t) in eq and np.isfinite(eq[(L, t)])]
        ps = np.array([pds[q.index(t)] for t in common])
        pe = np.array([eq[(L, t)] for t in common])
        st = paired_stats(ps, pe)
        st.update(tier=tier, holdout=L, method=method, n=len(common),
                  mean=float(np.nanmean(ps)), median=float(np.nanmedian(ps)),
                  equal_mean=float(np.nanmean(pe)),
                  equal_median=float(np.nanmedian(pe)))
        summary.append(st)
        print(f"  {method:12s} n={len(common):4d} "
              f"mean {st['mean']:.4f} (equal {st['equal_mean']:.4f}) "
              f"median {st['median']:.4f} (equal {st['equal_median']:.4f})",
              flush=True)
        print(f"  delta vs equal {st['delta_mean']:+.4f} "
              f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}] "
              f"p={st['p']:.2g} win={st['win']:.2f}", flush=True)

        for t, p, e in zip(common, ps, pe):
            flat_h.append(L)
            flat_m.append(method)
            flat_t.append(t)
            flat_p.append(p)
            flat_e.append(e)

    prefix = ("outputs/bench_supervised_v2" if tier == "B"
              else "outputs/bench_supervised_nadig_v2")
    prefix += f"_{mode}"
    if not intercept:
        prefix += "_noint"
    if n_comp != 256:
        prefix += f"_pc{n_comp}"
    if selfzero:
        prefix += "_selfzero"
    if shuffle:
        prefix += "_yshuf"
    np.savez(f"{prefix}.npz",
             holdout=np.array(flat_h), method=np.array(flat_m),
             target=np.array(flat_t),
             pds_raw=np.array(flat_p, dtype=np.float64),
             pds_equal=np.array(flat_e, dtype=np.float64))
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nsaved {prefix}.csv / .npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B", "all"], default="B")
    ap.add_argument("--mode", choices=["mix", "concat"], default="mix")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pcs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="",
                    help="comma-separated holdouts, default all")
    ap.add_argument("--intercept", type=int, choices=[0, 1], default=1)
    ap.add_argument("--selfzero", type=int, choices=[0, 1], default=0)
    ap.add_argument("--shuffle", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()
    only = set(args.holdout.split(",")) if args.holdout else None
    tiers = ["B", "A"] if args.tier == "all" else [args.tier]
    for t in tiers:
        run_tier(t, args.mode, args.folds, args.pcs, args.seed, only=only,
                 intercept=bool(args.intercept),
                 selfzero=bool(args.selfzero),
                 shuffle=bool(args.shuffle))


if __name__ == "__main__":
    main()
