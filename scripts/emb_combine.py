"""嵌入组合特异性诊断: concat 岭回归 vs kNN vs 特异性方向集成.

所有 pred 在同一 5-fold 口径下生成; spec 相关 = 投影掉 fold-mean 后的 per-pert 残差相关.
"""

import sys

import numpy as np

sys.path.insert(0, "src")
from linear_baseline import fit_ridge  # noqa: E402


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def main():
    m = np.load("outputs/scvi_map_cache.npz")
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    n = len(train_names)

    def emb(path):
        e = np.load(path)
        p = {g: i for i, g in enumerate(e["names"].astype(str))}
        return np.stack([e["E"][p[g]] for g in train_names])

    E650 = emb("outputs/esm650_emb.npz")
    ESC = emb("outputs/scgpt_emb.npz")
    E150 = emb("outputs/esm150_emb.npz")

    folds = np.random.default_rng(0).permutation(n) % 5

    def ridge_pred(Feat, lam):
        pred = np.zeros_like(LFC)
        for k in range(5):
            te = folds == k
            W, b = fit_ridge(Feat[~te], LFC[~te], lam)
            pred[te] = Feat[te] @ W + b
        return pred

    def knn_pred(Feat, kk):
        En = Feat / np.maximum(np.linalg.norm(Feat, axis=1, keepdims=True), 1e-9)
        pred = np.zeros_like(LFC)
        for k in range(5):
            te = folds == k
            tr = np.where(~te)[0]
            S = En[te] @ En[tr].T
            for j, i in enumerate(np.where(te)[0]):
                top = np.argsort(-S[j])[:kk]
                w = np.maximum(S[j][top], 0)
                if w.sum() < 1e-9:
                    w = np.ones_like(w)
                pred[i] = (LFC[tr[top]] * w[:, None]).sum(0) / w.sum()
        return pred

    def spec(pred, tag, orth_cache=None):
        if orth_cache is not None:
            pred = resid(pred, orth_cache)
        vals = []
        for k in range(5):
            te = folds == k
            mvec = LFC[~te].mean(0)
            for j in np.where(te)[0]:
                y = LFC[j]
                f = filt & np.isfinite(pred[j]) & np.isfinite(y)
                x, yv = pred[j][f], y[f]
                xr, yr = resid(x, mvec[f]), resid(yv, mvec[f])
                if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
                    continue
                vals.append(np.corrcoef(xr, yr)[0, 1])
        vals = np.array(vals)
        print(f"{tag}: spec median {np.median(vals):+.3f} mean {np.mean(vals):+.3f}", flush=True)
        return pred

    # 先把每个 pred 对 fold-mean 正交化, 再平均方向 (各 pert 一行)
    def orth_pred(pred):
        out = pred.copy()
        for k in range(5):
            te = folds == k
            mvec = LFC[~te].mean(0)
            for j in np.where(te)[0]:
                f = np.isfinite(out[j]) & np.isfinite(mvec)
                out[j][f] = resid(out[j][f], mvec[f])
        return out

    p_cat = ridge_pred(np.hstack([E650, ESC]), 100.0)
    spec(p_cat, "C1 ridge[E650+SC] lam100")

    p_catk = knn_pred(np.hstack([E650, ESC]), 10)
    spec(p_catk, "C2 kNN[E650+SC] k10   ")

    p1 = orth_pred(ridge_pred(E650, 100.0))
    p2 = orth_pred(knn_pred(ESC, 10))
    p3 = orth_pred(knn_pred(E150, 20))
    p4 = orth_pred(knn_pred(E650, 5))
    p_ens = (p1 + p2 + p3 + p4) / 4.0
    spec(p_ens, "C3 特异性方向集成(4)    ")

    # 两个最强单模型直接平均
    spec((p1 + p2) / 2.0, "C4 E650-ridge + SC-kNN10")


if __name__ == "__main__":
    main()
