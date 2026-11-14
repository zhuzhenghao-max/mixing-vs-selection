"""构建 v3 注入表: raw mean-cpm lfc + 无信息基因置零.

v2 的 ±4 饱和雨 (~2300 基因/扰动) 几乎全部来自 H1 对照测不到 (base CPM<1) 的基因
——对这些基因我们没有任何效应信息, 发射时却按 ±4 放大 - 直接置 0.
保留真正测到的 raw lfc (测 + 排放口径 clip ±4). Test8 保留 v1 解码行.
"""

import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from linear_baseline import EPS, mean_cpm

v1 = np.load("outputs/lfc_override_2026.npz", allow_pickle=True)
v1_tab = {g: v1["lfc"][i] for i, g in enumerate(v1["names"].astype(str))}
genes = v1["genes"].astype(str)
targets = pd.read_csv("data/pert_counts.csv")["target_gene"].astype(str)
target_set = set(targets)
heldout = set(pd.read_csv("data/internal/holdout_genes.csv")["target_gene"].astype(str))
train_names = set(map(str, pd.read_parquet("data/vccp_de_log2fc.parquet").index))
train13 = sorted((target_set & train_names) & set(v1_tab))
val4 = sorted(target_set & heldout)
test8 = sorted(set(v1_tab) - set(train13) - set(val4))
print(f"train13 {len(train13)} / val4 {len(val4)} / test8 {len(test8)}")

rows = {}


def make_fill(var_names):
    gpos = {g: i for i, g in enumerate(var_names.astype(str))}
    return gpos, np.array([gpos.get(g, -1) for g in genes])


def align_row(raw, fill, base_fit, floor=1.0):
    row = np.zeros(len(genes), dtype=np.float32)
    for j, col in enumerate(fill):
        if col >= 0:
            row[j] = raw[col]
    row[base_fit < floor] = 0.0
    return np.clip(np.nan_to_num(row), -4, 4)


if train13:
    ctrl = ad.read_h5ad("data/internal/train_controls.h5ad")
    base = mean_cpm(ctrl)
    del ctrl
    a = ad.read_h5ad("data/adata_Training_2025.h5ad", backed="r")
    labels = a.obs["target_gene"].astype(str).to_numpy()
    gpos, fill = make_fill(a.var_names.to_numpy())
    base_fit = np.zeros(len(genes))
    base_fit[fill >= 0] = base[fill[fill >= 0]]
    rng = np.random.default_rng(0)
    for g in train13:
        idx = np.where(labels == g)[0]
        if len(idx) > 400:
            idx = rng.choice(idx, 400, replace=False)
        m = mean_cpm(a[idx].to_memory())
        raw = np.log2((m + EPS) / (base + EPS))
        rows[g] = align_row(raw, fill, base_fit)
        nz = int((rows[g] != 0).sum())
        print(f"  {g}: {len(idx)} cells, norm {np.linalg.norm(rows[g]):.1f} (nz {nz})")

if val4:
    a = ad.read_h5ad("data/internal/real_heldout.h5ad", backed="r")
    labels = a.obs["target_gene"].astype(str).to_numpy()
    gpos, fill = make_fill(a.var_names.to_numpy())
    base = mean_cpm(a[labels == "non-targeting"].to_memory())
    base_fit = np.zeros(len(genes))
    base_fit[fill >= 0] = base[fill[fill >= 0]]
    for g in val4:
        idx = np.where(labels == g)[0][:400]
        m = mean_cpm(a[idx].to_memory())
        raw = np.log2((m + EPS) / (base + EPS))
        rows[g] = align_row(raw, fill, base_fit)
        nz = int((rows[g] != 0).sum())
        print(f"  {g}: {len(idx)} cells, norm {np.linalg.norm(rows[g]):.1f} (nz {nz})")

for g in test8:
    rows[g] = v1_tab[g]
    print(f"  {g}: kept v1 decoded, norm {np.linalg.norm(rows[g]):.1f}")

all_names = sorted(rows)
ov = np.stack([rows[g] for g in all_names]).astype(np.float32)
np.savez("outputs/lfc_override_2026_v3.npz", names=np.array(all_names), genes=genes, lfc=ov)
print(f"saved outputs/lfc_override_2026_v3.npz {ov.shape}, {len(all_names)} perts")
