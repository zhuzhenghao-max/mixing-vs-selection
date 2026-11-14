"""构建 25 个 H1∩2026 重叠扰动的真实 lfc 覆盖表 (统一解码空间).

21 个来自 scvi_map_cache.npz; 4 个落在 H1 val holdout 的
(ACLY/HDAC8/NFE2L1/SLIRP) 现场用 scvi_h1 编码->dz->解码, 与 cache LFC 同估计量.
用途: linear_baseline.py predict --lfc-override 面值注入.
"""

import torch  # 必须最先 import, 否则 c10.dll 崩

import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import anndata as ad
import numpy as np
import pandas as pd
import scvi

from cache_map_data import decode_mean
from linear_baseline import EPS

cache = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
genes = cache["genes"].astype(str)
names = cache["train_names"].astype(str)
lfc = cache["LFC"]
pos = {g: i for i, g in enumerate(names)}

targets = pd.read_csv("data/pert_counts.csv")["target_gene"].astype(str)
target_set = set(targets)
rows = {g: lfc[pos[g]] for g in sorted(target_set & set(names))}

heldout = pd.read_csv("data/internal/holdout_genes.csv")["target_gene"].astype(str)
hits4 = sorted(target_set & set(heldout))
print(f"cache hit {len(rows)}, holdout hit {len(hits4)}: {hits4}")

model = scvi.model.SCVI.load("outputs/scvi_h1", adata=None)
a = ad.read_h5ad("data/internal/real_heldout.h5ad", backed="r")
labs = a.obs["target_gene"].astype(str).to_numpy()
cidx = np.where(labs == "non-targeting")[0]
rng = np.random.default_rng(0)
if len(cidx) > 3000:
    cidx = rng.choice(cidx, 3000, replace=False)
ctrl = a[cidx].to_memory()
scvi.model.SCVI.setup_anndata(ctrl)
z_ctrl = model.get_latent_representation(ctrl, give_mean=True)
z_base = z_ctrl.mean(0)
libs = np.asarray(ctrl.X.sum(1)).ravel()
rates_ctrl = decode_mean(model, z_ctrl, libs)
gpos = {g: i for i, g in enumerate(ctrl.var_names.astype(str))}
fill = np.array([gpos.get(g, -1) for g in genes])

for g in hits4:
    jdx = np.where(labs == g)[0][:200]
    s = a[jdx].to_memory()
    scvi.model.SCVI.setup_anndata(s)
    z_p = model.get_latent_representation(s, give_mean=True)
    dz = z_p.mean(0) - z_base
    rates_p = decode_mean(model, z_ctrl + dz, libs)
    lfc_dec = np.log2((rates_p + EPS) / (rates_ctrl + EPS))
    row = np.zeros(len(genes), dtype=np.float32)
    for j, col in enumerate(fill):
        if col >= 0:
            row[j] = lfc_dec[col]
    rows[g] = row
    print(f"  {g}: {len(jdx)} cells, norm {np.linalg.norm(row):.1f}")

all_names = sorted(rows)
ov = np.stack([rows[g] for g in all_names]).astype(np.float32)
np.savez("outputs/lfc_override_2026.npz", names=np.array(all_names), genes=genes, lfc=ov)
print(f"saved outputs/lfc_override_2026.npz {ov.shape}, {len(all_names)} perts, "
      f"max |lfc| {np.abs(ov).max():.2f}")
