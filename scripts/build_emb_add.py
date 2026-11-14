"""构建嵌入特异性分量的加性注入表 (275 无数据扰动用).

方法 (与 emb_combine.py C3 同构, LOO spec-corr +0.090):
  4 个弱模型 (E650-ridge100 / SC-kNN10 / E150-kNN20 / E650-kNN5) 各自预测 lfc,
  对 285 训练均值响应 mvec 正交化, 等权平均 = 纯特异分量.
归一: gamma = median(||F454-ridge 特异残差||) / median(||ens||) (filt 基因),
  使加性分量与现有 F454 特异分量同量级 — 方向升级、几何不变.
输出 npz: names(300 面板), genes(18080 cache 轴), lfc = gamma * ens.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from linear_baseline import fit_ridge  # noqa: E402


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def orth_rows(P, mvec):
    out = P.copy()
    for j in range(P.shape[0]):
        f = np.isfinite(out[j]) & np.isfinite(mvec)
        out[j][f] = resid(out[j][f], mvec[f])
    return out


def knn_pred(E_all, train_rows, LFC, k):
    En = E_all / np.maximum(np.linalg.norm(E_all, axis=1, keepdims=True), 1e-9)
    Tn = train_rows / np.maximum(np.linalg.norm(train_rows, axis=1, keepdims=True), 1e-9)
    S = En @ Tn.T
    P = np.zeros((E_all.shape[0], LFC.shape[1]), dtype=np.float32)
    for j in range(E_all.shape[0]):
        top = np.argsort(-S[j])[:k]
        w = np.maximum(S[j][top], 0)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        P[j] = (LFC[top] * w[:, None]).sum(0) / w.sum()
    return P


def main():
    m = np.load("outputs/scvi_map_cache.npz")
    F_all, genes = m["F"], m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    posg = {g: i for i, g in enumerate(genes)}
    F_tr = np.stack([F_all[posg[g]] for g in train_names])

    panel = pd.read_csv("data/pert_counts.csv")["target_gene"].astype(str).tolist()
    v1 = np.load("outputs/lfc_override_2026.npz", allow_pickle=True)
    heldout = set(pd.read_csv("data/internal/holdout_genes.csv")["target_gene"].astype(str))
    ov25 = set(v1["names"].astype(str))
    non25 = [g for g in panel if g not in ov25]
    print(f"panel {len(panel)}, injected {len(set(panel) & ov25)}, additive targets {len(non25)}")
    assert set(non25) == set(panel) - ov25

    def emb(path):
        e = np.load(path)
        p = {g: i for i, g in enumerate(e["names"].astype(str))}
        return np.stack([e["E"][p[g]] for g in panel]), \
               np.stack([e["E"][p[g]] for g in train_names])

    E650_p, E650_t = emb("outputs/esm650_emb.npz")
    E150_p, E150_t = emb("outputs/esm150_emb.npz")
    SC_p, SC_t = emb("outputs/scgpt_emb.npz")

    mvec = LFC.mean(0)

    models = []
    W, b = fit_ridge(E650_t, LFC, 100.0)
    models.append(("E650-ridge100", E650_p @ W + b))
    models.append(("SC-kNN10 ", knn_pred(SC_p, SC_t, LFC, 10)))
    models.append(("E150-kNN20", knn_pred(E150_p, E150_t, LFC, 20)))
    models.append(("E650-kNN5 ", knn_pred(E650_p, E650_t, LFC, 5)))

    orths = [orth_rows(P, mvec) for _, P in models]
    ens = np.mean(orths, 0)
    for (name, P), O in zip(models, orths):
        print(f"  {name}: pred norm {np.linalg.norm(P[:, filt], axis=1).mean():.1f}, "
              f"spec norm {np.linalg.norm(O[:, filt], axis=1).mean():.1f}")

    # gamma: 加性分量与 F454 特异残差同量级
    W4, b4 = fit_ridge(F_tr, LFC, 100.0)
    R4 = F_tr @ W4 + b4
    O4 = orth_rows(R4, mvec)
    n4 = np.median(np.linalg.norm(O4[:, filt], axis=1))
    ne = np.median(np.linalg.norm(ens[:, filt], axis=1))
    gamma = n4 / max(ne, 1e-9)
    print(f"F454 spec norm median {n4:.2f}, ens norm median {ne:.2f} -> gamma {gamma:.3f}")

    add = (gamma * ens).astype(np.float32)
    print(f"add rows {add.shape}, norm median {np.median(np.linalg.norm(add[:, filt], axis=1)):.2f}")

    np.savez("outputs/lfc_add_emb_ens.npz", names=np.array(panel), genes=genes, lfc=add,
             mvec=mvec.astype(np.float32), gamma=np.float32(gamma))
    print(f"saved outputs/lfc_add_emb_ens.npz ({len(panel)} rows incl. 25 injected (will be overridden))")


if __name__ == "__main__":
    main()
