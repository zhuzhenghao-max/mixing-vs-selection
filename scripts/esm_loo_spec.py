"""LOO 特异性信号诊断: 去掉 fold-mean 响应分量后的 per-pert 相关.

plain corr 被共享响应分量主导 (fold-mean 基线 0.315 ≈ ridge 0.322), 无法区分真特异性.
给定 fold 训练集均值响应 m, 把 pred 和 y 都投影掉 m 再算相关 = 特异性分量相关.
配置: F454 / ESM / F+ESM / ESM 空间 kNN.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from linear_baseline import fit_ridge  # noqa: E402


def resid(x, m):
    c = (x @ m) / (m @ m)
    return x - c * m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--ks", default="5,10,20,50")
    args = p.parse_args()

    m = np.load("outputs/scvi_map_cache.npz")
    F_all, genes = m["F"], m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    posg = {g: i for i, g in enumerate(genes)}
    F_tr = np.stack([F_all[posg[g]] for g in train_names])
    emb = np.load(args.emb)
    enames, E = emb["names"].astype(str), emb["E"]
    epos = {g: i for i, g in enumerate(enames)}
    E_tr = np.stack([E[epos[g]] for g in train_names])
    En = E_tr / np.maximum(np.linalg.norm(E_tr, axis=1, keepdims=True), 1e-9)

    n = len(train_names)
    folds = np.random.default_rng(0).permutation(n) % 5

    def spec_stats(pred_rows, folds, tag):
        vals, dmeans = [], []
        for k in range(5):
            te = folds == k
            tr = ~te
            mvec = LFC[tr].mean(0)
            for j in np.where(te)[0]:
                y = LFC[j]
                f = filt & np.isfinite(pred_rows[j]) & np.isfinite(y)
                x, yv = pred_rows[j][f], y[f]
                if np.std(yv) < 1e-9 or np.std(x) < 1e-9:
                    continue
                c_full = np.corrcoef(x, yv)[0, 1]
                c_mean = np.corrcoef(mvec[f], yv)[0, 1]
                xr, yr, mr = resid(x, mvec[f]), resid(yv, mvec[f]), mvec[f]
                if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
                    continue
                vals.append(np.corrcoef(xr, yr)[0, 1])
                dmeans.append(c_full - c_mean)
        vals, dmeans = np.array(vals), np.array(dmeans)
        print(f"{tag}: spec-corr median {np.median(vals):+.3f} mean {np.mean(vals):+.3f} "
              f"| corr-meancorr median {np.median(dmeans):+.3f} "
              f"frac>0 {np.mean(dmeans > 0):.2f} (n={len(vals)})", flush=True)

    pred = np.zeros_like(LFC)
    for k in range(5):
        te = folds == k
        W, b = fit_ridge(F_tr[~te], LFC[~te], 100.0)
        pred[te] = F_tr[te] @ W + b
    spec_stats(pred, folds, "F454 lam100 ")

    pred = np.zeros_like(LFC)
    for k in range(5):
        te = folds == k
        W, b = fit_ridge(E_tr[~te], LFC[~te], 100.0)
        pred[te] = E_tr[te] @ W + b
    spec_stats(pred, folds, "ESM  lam100 ")

    pred = np.zeros_like(LFC)
    Fc = np.hstack([F_tr, E_tr])
    for k in range(5):
        te = folds == k
        W, b = fit_ridge(Fc[~te], LFC[~te], 100.0)
        pred[te] = Fc[te] @ W + b
    spec_stats(pred, folds, "F+ESM lam100")

    for kk in [int(x) for x in args.ks.split(",")]:
        pred = np.zeros_like(LFC)
        for k in range(5):
            te = folds == k
            tr_rows = np.where(~te)[0]
            S = En[te] @ En[tr_rows].T  # cosine
            for j, i in enumerate(np.where(te)[0]):
                sim = S[j]
                top = np.argsort(-sim)[:kk]
                wv = np.maximum(sim[top], 0)
                if wv.sum() < 1e-9:
                    wv = np.ones_like(wv)
                pred[i] = (LFC[tr_rows[top]] * wv[:, None]).sum(0) / wv.sum()
        spec_stats(pred, folds, f"kNN-ESM k={kk:<3d}")


if __name__ == "__main__":
    main()
