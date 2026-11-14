"""组装 v4 override 表: v3 的 25 行原样 + 275 行 = c * ens (纯特异方向).

背景: 官方 pds 对共享 profile 锁死 0.500 (全并列); LOO 代理 (285 pert, 同口径余弦检索)
显示纯特异方向 raw-pds 0.61-0.63, 共享主导仅 0.525. DE 四成员跨度小且饱和, mse 被
clamp [0,1] (我们已在 0) — pds 是唯一大杠杆, 故 275 注入纯特异分量.
c 校准: raw17 (train13+val4, 已验证注入) median ||lfc||_filt / median ||ens||_filt,
保持注入强度与 v3 已验证配置同量级; filt 外基因置 0 (与代理评估空间一致).
"""

import numpy as np
import pandas as pd

m = np.load("outputs/scvi_map_cache.npz")
genes = m["genes"].astype(str)
filt = m["filt"]
panel = pd.read_csv("data/pert_counts.csv")["target_gene"].astype(str).tolist()

v3 = np.load("outputs/lfc_override_2026_v3.npz", allow_pickle=True)
v3_names = list(v3["names"].astype(str))
v3_tab = {g: v3["lfc"][i] for i, g in enumerate(v3_names)}

add = np.load("outputs/lfc_add_emb_ens.npz", allow_pickle=True)
add_names = list(add["names"].astype(str))
assert list(add["genes"].astype(str)) == list(genes)
add_tab = {g: add["lfc"][i] for i, g in enumerate(add_names)}

target_set = set(panel)
heldout = set(pd.read_csv("data/internal/holdout_genes.csv")["target_gene"].astype(str))
train_names = set(map(str, pd.read_parquet("data/vccp_de_log2fc.parquet").index))
train13 = sorted((target_set & train_names) & set(v3_tab))
val4 = sorted(target_set & heldout)
test8 = sorted(set(v3_tab) - set(train13) - set(val4))
raw17 = train13 + val4
print(f"raw17 {len(raw17)} (train13 {len(train13)} + val4 {len(val4)}), decoded {len(test8)}")

n_raw = np.median([np.linalg.norm(v3_tab[g][filt]) for g in raw17])
n_dec = np.median([np.linalg.norm(v3_tab[g][filt]) for g in test8])
ens275 = [g for g in panel if g not in v3_tab]
assert len(ens275) == 275
n_ens = np.median([np.linalg.norm(add_tab[g][filt]) for g in ens275])
c = n_raw / n_ens
print(f"raw17 norm med {n_raw:.2f}, decoded med {n_dec:.2f}, ens med {n_ens:.4f} -> c {c:.1f}")

rows = []
n_inj = []
for g in panel:
    if g in v3_tab:
        rows.append(v3_tab[g])
    else:
        r = add_tab[g].astype(np.float64).copy()
        r[~filt] = 0.0
        r = np.clip(c * r, -4, 4)
        rows.append(r)
        n_inj.append(np.linalg.norm(r[filt]))
ov = np.stack(rows).astype(np.float32)
inj_idx = [i for i, g in enumerate(panel) if g not in v3_tab]
v3_idx = [i for i, g in enumerate(panel) if g in v3_tab]
assert np.array_equal(ov[v3_idx], np.stack([v3_tab[g] for g in panel if g in v3_tab]))
print(f"v4 table {ov.shape}, injected norm med {np.median(n_inj):.2f}, "
      f"max|.| {np.abs(ov).max():.2f}, finite {np.isfinite(ov).all()}")
np.savez("outputs/lfc_override_2026_v4.npz", names=np.array(panel), genes=genes, lfc=ov)
print("saved outputs/lfc_override_2026_v4.npz")
