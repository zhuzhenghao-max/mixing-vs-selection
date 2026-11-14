"""组装 v5 override 表: 与 v4 同构, 275 行换成新集成 ens5 (跨种子稳定性筛选后).

成员在全部 285 train pert 上训练, 预测 300 面板; 各自对全局 mvec=LFC.mean(0) 正交化,
等权平均; filt 外置 0; c5 = raw17 median ||lfc||_filt / median ||ens5||_filt 保持注入
强度与已验证配置同量级; clip ±4. 25 注入行从 v3 表原样复制 (断言 bit-identical).
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from linear_baseline import fit_ridge  # noqa: E402

MEMBERS = [
    ("E650r30", "ridge", ("s", "E650"), 30.0),
    ("GF316r100", "ridge", ("s", "GF316"), 100.0),
    ("GF316r30", "ridge", ("s", "GF316"), 30.0),
    ("GF316r300", "ridge", ("s", "GF316"), 300.0),
    ("GF316k20", "knn", ("s", "GF316"), 20),
    ("SCr100", "ridge", ("s", "SC"), 100.0),
    ("E150k20", "knn", ("s", "E150"), 20),
    ("CatGFSCr100", "ridge", ("cat", "GF316", "SC"), 100.0),
    ("CatESr100", "ridge", ("cat", "E650", "SC"), 100.0),
]


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
    P = np.zeros((E_all.shape[0], LFC.shape[1]), dtype=np.float64)
    for j in range(E_all.shape[0]):
        top = np.argsort(-S[j])[:k]
        w = np.maximum(S[j][top], 0)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        P[j] = (LFC[top] * w[:, None]).sum(0) / w.sum()
    return P


def main():
    m = np.load("outputs/scvi_map_cache.npz")
    genes = m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]

    panel = pd.read_csv("data/pert_counts.csv")["target_gene"].astype(str).tolist()
    v3 = np.load("outputs/lfc_override_2026_v3.npz", allow_pickle=True)
    v3_names = list(v3["names"].astype(str))
    v3_tab = {g: v3["lfc"][i] for i, g in enumerate(v3_names)}

    def emb(path):
        e = np.load(path)
        p = {g: i for i, g in enumerate(e["names"].astype(str))}
        return np.stack([e["E"][p[g]] for g in panel]), \
               np.stack([e["E"][p[g]] for g in train_names])

    spaces_p = {k: emb(f"outputs/{f}_emb.npz")[0] for k, f in
                [("E650", "esm650"), ("E150", "esm150"), ("SC", "scgpt"), ("GF316", "gf316")]}
    spaces_t = {k: emb(f"outputs/{f}_emb.npz")[1] for k, f in
                [("E650", "esm650"), ("E150", "esm150"), ("SC", "scgpt"), ("GF316", "gf316")]}

    def feat(key):
        if key[0] == "cat":
            return (np.hstack([spaces_p[k] for k in key[1:]]),
                    np.hstack([spaces_t[k] for k in key[1:]]))
        return spaces_p[key[1]], spaces_t[key[1]]

    mvec = LFC.mean(0)
    preds = []
    for name, kind, key, param in MEMBERS:
        Fp, Ft = feat(key)
        if kind == "ridge":
            W, b = fit_ridge(Ft, LFC, param)
            P = Fp @ W + b
        else:
            P = knn_pred(Fp, Ft, LFC, param)
        O = orth_rows(P, mvec)
        preds.append(O)
        print(f"  {name:12s}: pred med norm {np.median(np.linalg.norm(P[:, filt], axis=1)):.2f}, "
              f"spec med norm {np.median(np.linalg.norm(O[:, filt], axis=1)):.3f}", flush=True)
    ens = np.mean(preds, 0)
    print(f"ens5 {ens.shape} from {len(MEMBERS)} members")

    heldout = set(pd.read_csv("data/internal/holdout_genes.csv")["target_gene"].astype(str))
    train_set = set(map(str, pd.read_parquet("data/vccp_de_log2fc.parquet").index))
    target_set = set(panel)
    train13 = sorted((target_set & train_set) & set(v3_tab))
    val4 = sorted(target_set & heldout)
    raw17 = train13 + val4
    assert len(raw17) == 17

    n_raw = np.median([np.linalg.norm(v3_tab[g][filt]) for g in raw17])
    ens275_idx = [i for i, g in enumerate(panel) if g not in v3_tab]
    assert len(ens275_idx) == 275
    n_ens = np.median([np.linalg.norm(ens[i][filt]) for i in ens275_idx])
    c5 = n_raw / n_ens
    print(f"raw17 norm med {n_raw:.2f}, ens5 norm med {n_ens:.4f} -> c5 {c5:.1f}")

    rows = []
    for i, g in enumerate(panel):
        if g in v3_tab:
            rows.append(v3_tab[g])
        else:
            r = ens[i].astype(np.float64).copy()
            r[~filt] = 0.0
            rows.append(np.clip(c5 * r, -4, 4))
    ov = np.stack(rows).astype(np.float32)
    v3_idx = [i for i, g in enumerate(panel) if g in v3_tab]
    assert np.array_equal(ov[v3_idx], np.stack([v3_tab[g] for g in panel if g in v3_tab]))
    v4 = np.load("outputs/lfc_override_2026_v4.npz", allow_pickle=True)
    v4_rows = {g: v4["lfc"][i] for i, g in enumerate(v4["names"].astype(str))}
    assert np.array_equal(ov[v3_idx], np.stack([v4_rows[g] for g in panel if g in v3_tab]))
    d = np.array([np.linalg.norm(ov[i][filt] - v4_rows[g][filt]) for i, g in enumerate(panel) if g not in v3_tab])
    print(f"v5 table {ov.shape}, max|.| {np.abs(ov).max():.2f}, finite {np.isfinite(ov).all()}, "
          f"275-row ||v5-v4|| med {np.median(d):.2f} (v4 行 norm med {n_raw:.2f})")
    np.savez("outputs/lfc_override_2026_v5.npz", names=np.array(panel), genes=genes, lfc=ov)
    print("saved outputs/lfc_override_2026_v5.npz")


if __name__ == "__main__":
    main()
