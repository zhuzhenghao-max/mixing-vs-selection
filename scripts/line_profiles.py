"""提取各细胞系对照表达谱 (供相似度加权 / 细胞系条件化使用).

输出 outputs/line_profiles.npz: names, genes (18080 面板轴), prof (n_lines x 18080)
- A/B/C: context h5ad 全细胞 mean CP1M
- k562: gwps h5ad 对照细胞 mean CP1M
- hct116/hek293t: orion ctrl_means (22182 轴, 原样)
- rpe1: rpe1 h5ad 对照细胞 mean CP1M (存在则提取)
缺基因 = 0; 相似度在调用侧用交集基因算.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")

from linear_baseline import mean_cpm  # noqa: E402

CHUNK = 25000


def h5ad_ctrl_prof(path, label_col="perturbation", ctrl_names=("control",)):
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    genes = a.var_names.to_numpy().astype(str)
    pert = a.obs[label_col].to_numpy().astype(str)
    cmask = np.isin(pert, ctrl_names)
    n_ctrl = int(cmask.sum())
    acc = np.zeros(len(genes), dtype=np.float64)
    for s in range(0, len(pert), CHUNK):
        e = min(s + CHUNK, len(pert))
        blk = np.asarray(a.X[s:e], dtype=np.float32)
        tot = blk.sum(1)
        cpm = blk / np.maximum(tot, 1)[:, None] * 1e6
        acc += cpm[cmask[s:e]].sum(0)
    prof = acc / max(n_ctrl, 1)
    a.file.close()
    return genes, prof, n_ctrl


def main():
    v10 = np.load("outputs/lfc_override_2026_v10.npz", allow_pickle=True)
    axis = [str(x) for x in v10["genes"].astype(str)]
    axpos = {g: i for i, g in enumerate(axis)}

    names, profs = [], []

    for nm, path in (("A", "data/context_A.h5ad"), ("B", "data/context_B.h5ad"),
                     ("C", "data/context_C.h5ad")):
        import anndata as ad
        a = ad.read_h5ad(path)
        m = mean_cpm(a)
        names.append(nm)
        profs.append((a.var_names.to_numpy().astype(str), m))
        print(f"{nm}: {len(m)} genes, {a.n_obs} cells")

    g, p, n = h5ad_ctrl_prof("data/external/gwps/replogle_2022_k562_gwps.h5ad")
    names.append("k562")
    profs.append((g, p))
    print(f"k562: {len(g)} genes, {n} control cells")

    c = np.load("data/external/orion_ctrl_means.npy")
    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    og = meta["genes"]
    for nm, i in (("hct116", 0), ("hek293t", 1)):
        names.append(nm)
        profs.append((og, np.asarray(c[i], dtype=np.float64)))
        print(f"{nm}: {len(og)} genes from ctrl_means")

    rp = "data/external/rpe1/replogle_2022_rpe1.h5ad"
    if os.path.exists(rp) and os.path.getsize(rp) > 1e9:
        g, p, n = h5ad_ctrl_prof(rp)
        names.append("rpe1")
        profs.append((g, p))
        print(f"rpe1: {len(g)} genes, {n} control cells")
    else:
        print("rpe1 h5ad not ready, skip (rerun later)")

    P = np.zeros((len(names), len(axis)), dtype=np.float32)
    for i, (g, p) in enumerate(profs):
        for j, gene in enumerate(g):
            if gene in axpos:
                P[i, axpos[gene]] = p[j]
        print(f"{names[i]:8s} mapped {(P[i] > 0).sum()} genes onto 18080 axis")

    np.savez("outputs/line_profiles.npz", names=np.array(names), genes=np.array(axis), prof=P)
    print("saved outputs/line_profiles.npz", P.shape)


if __name__ == "__main__":
    main()
