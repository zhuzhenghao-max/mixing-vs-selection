"""诊断: scVI 潜空间预测 lfc vs 真值 lfc, 校准 dz 缩放

预测 lfc: 同一批对照细胞的 z 加上 s*dz 后解码, 与对照解码均值做 log2 比值
真值 lfc 同 cell-eval2 定义, 只看过滤基因 (对照均值 > 5 CPM)
"""

import argparse
import sys

sys.path.insert(0, "src")

import torch  # 必须最先 import, 否则 c10.dll 崩
import anndata as ad
import numpy as np
import scanpy as sc
import scvi

from linear_baseline import mean_cpm, lognorm_cpm, gene_features, EPS
from prior_features import build_priors

CONTROL = "non-targeting"


def decode_mean(model, z, lib):
    z = torch.as_tensor(np.clip(z, -6, 6), dtype=torch.float32)
    # generative 收 log library, 传原始 UMI 会 exp 溢出成 inf
    lib_t = torch.as_tensor(np.log(np.maximum(lib, 1.0)), dtype=torch.float32)[:, None]
    batch = torch.zeros(z.shape[0], dtype=torch.long)
    with torch.no_grad():
        out = model.module.generative(z, lib_t, batch)
        px = out["px"].mean
    p = px.detach().numpy()
    return np.asarray(p.mean(0)).ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--ctrl", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--pert-col", default="target_gene")
    p.add_argument("--scales", default="0.1,0.25,0.5,1.0,2.0")
    p.add_argument("--max-cells-ctrl", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    model = scvi.model.SCVI.load(args.model, adata=None)
    m = np.load(args.map + "/map.npz")
    W, b = m["W"], m["b"]
    n_pcs = int(m["n_pcs"]) if "n_pcs" in m else 40
    prior_tf = [x for x in str(m["prior_tf"]).split(";") if x] if "prior_tf" in m else []
    prior_path = [x for x in str(m["prior_path"]).split(";") if x] if "prior_path" in m else []

    ctrl = ad.read_h5ad(args.ctrl, backed="r")
    if ctrl.n_obs > args.max_cells_ctrl:
        idx = rng.choice(ctrl.n_obs, args.max_cells_ctrl, replace=False)
        ctrl = ctrl[idx].to_memory()
    else:
        ctrl = ctrl.to_memory()
    genes = ctrl.var_names.to_numpy().astype(str)
    scvi.model.SCVI.setup_anndata(ctrl)
    z_ctrl = model.get_latent_representation(ctrl, give_mean=True)
    libs = np.asarray(ctrl.X.sum(1)).ravel()
    rates_ctrl = decode_mean(model, z_ctrl, libs)

    ctrl_ln = lognorm_cpm(ctrl)
    F = gene_features(ctrl_ln, n_pcs)
    if prior_tf or prior_path:
        pc = np.asarray(ctrl_ln.varm["PCs"])
        F = np.hstack([F, build_priors(ctrl_ln, pc, genes, prior_tf, prior_path)])
    pos = {g: i for i, g in enumerate(genes)}

    real = ad.read_h5ad(args.real, backed="r")
    labels = real.obs[args.pert_col].astype(str).to_numpy()
    cidx = np.where(labels == CONTROL)[0]
    base = mean_cpm(real[cidx].to_memory())
    filt = base > 5

    scales = [float(x) for x in args.scales.split(",")]
    perts = sorted(set(labels) - {CONTROL})
    print(f"{'pert':12s} " + " ".join(f"nmae@{s:g}" for s in scales) + "   corr  sign>0.5")
    for g in perts:
        dz = F[pos[g]] @ W + b
        m2 = mean_cpm(real[np.where(labels == g)[0]].to_memory())
        lfc_true = np.log2((m2 + EPS) / (base + EPS))
        row = []
        corr = np.nan
        sgn = np.nan
        for s in scales:
            rates_pert = decode_mean(model, z_ctrl + s * dz, libs)
            lfc_pred = np.log2((rates_pert + EPS) / (rates_ctrl + EPS))
            f = filt & np.isfinite(lfc_true) & np.isfinite(lfc_pred)
            x, y = lfc_pred[f], lfc_true[f]
            nmae = np.mean(np.abs(x - y)) / (np.mean(np.abs(y)) + EPS)
            row.append(f"{nmae:7.3f}")
            corr = np.corrcoef(x, y)[0, 1]
            sgn = (x[np.abs(y) > 0.5] * y[np.abs(y) > 0.5] > 0).mean()
        print(f"{g:12s} " + " ".join(row) + f"  {corr:6.3f}  {sgn:6.3f}")


if __name__ == "__main__":
    main()
