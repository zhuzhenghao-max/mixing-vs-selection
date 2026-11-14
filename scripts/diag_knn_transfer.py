"""kNN 效应迁移 LOO 诊断: 目标基因特征邻近的 H1 扰动效应加权平均, 能否胜过岭回归.

TransPert 思路的单系移植 (无多系参考数据): 在基因特征空间 (F) 找邻居,
对邻居的真实 LFC 行做相似度软加权平均. 与 fit_lfc_map 同口径 5-fold LOO
(per-pert Pearson corr over filt), 岭 λ=100 的 CV 中位 0.322 作对比.
"""

import sys

import numpy as np

sys.path.insert(0, "src")

cache = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
F_all = cache["F"]
genes = cache["genes"].astype(str)
train_names = cache["train_names"].astype(str)
LFC = cache["LFC"]
filt = cache["filt"].astype(bool)
pos = {g: i for i, g in enumerate(genes)}
names = sorted(train_names)

F_tr = np.stack([F_all[pos[g]] for g in names])
LFC_tr = LFC[[np.where(train_names == g)[0][0] for g in names]]

mu, sd = F_tr.mean(0), F_tr.std(0)
sd = np.where(sd > 1e-9, sd, 1.0)
Z = (F_tr - mu) / sd
Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)

n = len(names)
folds = np.random.default_rng(0).permutation(n) % 5

def corr_med(pred_rows, true_rows, both_filt):
    cs = []
    for x, y in zip(pred_rows, true_rows):
        f = both_filt
        if np.std(y[f]) > 1e-9 and np.std(x[f]) > 1e-9:
            cs.append(np.corrcoef(x[f], y[f])[0, 1])
    return np.median(cs), np.mean(cs)

def run_knn(k, temp_q):
    pred = np.zeros_like(LFC_tr)
    for fold in range(5):
        te = folds == fold
        tr = ~te
        S = Zn[te] @ Zn[tr].T  # cosine sim in standardized feature space
        for i, j in enumerate(np.flatnonzero(te)):
            s = S[i]
            order = np.argsort(-s)[:k]
            sw = s[order]
            w = np.exp((sw - sw[0]) / (temp_q * np.std(s) + 1e-9))
            w = w / w.sum()
            pred[j] = w @ LFC_tr[tr][order]
    return corr_med(pred, LFC_tr, filt & np.isfinite(LFC_tr).all(0))

# 平均响应基线 (per-fold train mean)
pred_avg = np.zeros_like(LFC_tr)
for fold in range(5):
    te = folds == fold
    tr = ~te
    pred_avg[te] = LFC_tr[tr].mean(0)
m, a = corr_med(pred_avg, LFC_tr, filt & np.isfinite(LFC_tr).all(0))
print(f"fold-mean baseline: corr median {m:.3f} mean {a:.3f}")

for k in (3, 5, 10, 20, 40):
    for tq in (0.25, 0.5, 1.0):
        m, a = run_knn(k, tq)
        print(f"knn k={k:3d} T={tq:.2f}: corr median {m:.3f} mean {a:.3f}")
