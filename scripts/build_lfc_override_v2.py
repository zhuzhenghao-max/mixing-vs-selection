"""构建 v2 注入表: Train13 + Val4 换 raw mean-cpm lfc (pds real 侧原生空间).

v1 (解码 lfc) 已验证跨系迁移 (+0.0047); v2 测估计量保真度:
- Train13: 原始细胞在 data/adata_Training_2025.h5ad (backed 抽), baseline=H1 train controls
- Val4 (ACLY/HDAC8/NFE2L1/SLIRP): real_heldout 原始细胞 + 自带对照
- Test8: 原始细胞已删, 保留 v1 解码行
全部 clip ±4 (与 gen_cells 发射口径一致). 输出 lfc_override_2026_v2.npz.
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

parquet = pd.read_parquet("data/vccp_de_log2fc.parquet")
print("parquet shape", parquet.shape, "index head", list(parquet.index[:3]))
train_names = set(map(str, parquet.index))
train13 = sorted((target_set & train_names) & set(v1_tab))
val4 = sorted(target_set & heldout)
test8 = sorted(set(v1_tab) - set(train13) - set(val4))
print(f"train13 {len(train13)}: {train13}")
print(f"val4 {len(val4)}: {val4}")
print(f"test8 {len(test8)}: {test8}")

rows = {}


def align_row(raw, gpos, fill):
    row = np.zeros(len(genes), dtype=np.float32)
    for j, col in enumerate(fill):
        if col >= 0:
            row[j] = raw[col]
    return np.clip(np.nan_to_num(row), -4, 4)


if train13:
    ctrl = ad.read_h5ad("data/internal/train_controls.h5ad")
    base = mean_cpm(ctrl)
    del ctrl
    a = ad.read_h5ad("data/adata_Training_2025.h5ad", backed="r")
    labels = a.obs["target_gene"].astype(str).to_numpy()
    gpos = {g: i for i, g in enumerate(a.var_names.to_numpy().astype(str))}
    fill = np.array([gpos.get(g, -1) for g in genes])
    rng = np.random.default_rng(0)
    for g in train13:
        idx = np.where(labels == g)[0]
        if len(idx) > 400:
            idx = rng.choice(idx, 400, replace=False)
        m = mean_cpm(a[idx].to_memory())
        raw = np.log2((m + EPS) / (base + EPS))
        rows[g] = align_row(raw, gpos, fill)
        print(f"  {g}: {len(idx)} cells, norm {np.linalg.norm(rows[g]):.1f}")

if val4:
    a = ad.read_h5ad("data/internal/real_heldout.h5ad", backed="r")
    labels = a.obs["target_gene"].astype(str).to_numpy()
    gpos = {g: i for i, g in enumerate(a.var_names.to_numpy().astype(str))}
    fill = np.array([gpos.get(g, -1) for g in genes])
    base = mean_cpm(a[labels == "non-targeting"].to_memory())
    for g in val4:
        idx = np.where(labels == g)[0][:400]
        m = mean_cpm(a[idx].to_memory())
        raw = np.log2((m + EPS) / (base + EPS))
        rows[g] = align_row(raw, gpos, fill)
        print(f"  {g}: {len(idx)} cells, norm {np.linalg.norm(rows[g]):.1f}")

for g in test8:
    rows[g] = v1_tab[g]
    print(f"  {g}: kept v1 decoded, norm {np.linalg.norm(rows[g]):.1f}")

all_names = sorted(rows)
ov = np.stack([rows[g] for g in all_names]).astype(np.float32)
np.savez("outputs/lfc_override_2026_v2.npz", names=np.array(all_names), genes=genes, lfc=ov)
print(f"saved outputs/lfc_override_2026_v2.npz {ov.shape}, {len(all_names)} perts")
