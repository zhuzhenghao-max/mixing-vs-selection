"""诊断: scVI 潜空间映射质量 + 解码天花板

1. map 质量: 285 训练扰动上, 预测 dz vs 真实 dz 的 R2/cos/范数比, 跨扰动逐维相关
2. 天花板: 用真实 dz 解码的 lfc vs 真实 lfc 的 corr/nmae (解码器能力上限)
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
    p.add_argument("--perturbed", required=True, help="逗号分隔 h5ad, 与 fit-map 一致")
    p.add_argument("--pert-col", default="target_gene")
    p.add_argument("--max-cells-ctrl", type=int, default=1500)
    p.add_argument("--max-cells-per-pert", type=int, default=200)
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
    z_base = z_ctrl.mean(0)
    libs = np.asarray(ctrl.X.sum(1)).ravel()
    rates_ctrl = decode_mean(model, z_ctrl, libs)
    base = mean_cpm(ctrl)
    filt = base > 5

    ctrl_ln = lognorm_cpm(ctrl)
    F = gene_features(ctrl_ln, n_pcs)
    if prior_tf or prior_path:
        pc = np.asarray(ctrl_ln.varm["PCs"])
        F = np.hstack([F, build_priors(ctrl_ln, pc, genes, prior_tf, prior_path)])
    pos = {g: i for i, g in enumerate(genes)}

    DZ_pred, DZ_true = [], []
    r2s, coss, nrs, ceil_corrs, ceil_nmaes = [], [], [], [], []
    print(f"{'pert':12s} {'mapR2':>7s} {'cos':>6s} {'nr':>6s} {'ceilcorr':>9s} {'ceilnmae':>9s}")
    for path in args.perturbed.split(","):
        pert = ad.read_h5ad(path, backed="r")
        labs = pert.obs[args.pert_col].astype(str).to_numpy()
        for g in sorted(set(labs) - {CONTROL}):
            idx = np.where(labs == g)[0]
            if len(idx) > args.max_cells_per_pert:
                idx = rng.choice(idx, args.max_cells_per_pert, replace=False)
            sub = pert[idx].to_memory()
            scvi.model.SCVI.setup_anndata(sub)
            z_pert = model.get_latent_representation(sub, give_mean=True)
            dz_true = z_pert.mean(0) - z_base
            dz_pred = F[pos[g]] @ W + b
            r2 = 1 - np.sum((dz_pred - dz_true) ** 2) / (np.sum(dz_true ** 2) + EPS)
            cos = np.dot(dz_pred, dz_true) / (np.linalg.norm(dz_pred) * np.linalg.norm(dz_true) + EPS)
            nr = np.linalg.norm(dz_pred) / (np.linalg.norm(dz_true) + EPS)

            m2 = mean_cpm(sub)
            lfc_true = np.log2((m2 + EPS) / (base + EPS))
            rates_pert = decode_mean(model, z_ctrl + dz_true, libs)
            lfc_pred = np.log2((rates_pert + EPS) / (rates_ctrl + EPS))
            f = filt & np.isfinite(lfc_true) & np.isfinite(lfc_pred)
            cc = np.corrcoef(lfc_pred[f], lfc_true[f])[0, 1]
            nmae = np.mean(np.abs(lfc_pred[f] - lfc_true[f])) / (np.mean(np.abs(lfc_true[f])) + EPS)
            print(f"{g:12s} {r2:7.3f} {cos:6.3f} {nr:6.2f} {cc:9.3f} {nmae:9.3f}")
            DZ_pred.append(dz_pred)
            DZ_true.append(dz_true)
            r2s.append(r2)
            coss.append(cos)
            nrs.append(nr)
            ceil_corrs.append(cc)
            ceil_nmaes.append(nmae)
        print(f"{path}: done")

    DZ_pred = np.stack(DZ_pred)
    DZ_true = np.stack(DZ_true)
    dim_corr = [np.corrcoef(DZ_pred[:, d], DZ_true[:, d])[0, 1] for d in range(DZ_true.shape[1])]
    print("median map R2:", np.median(r2s))
    print("median cos:", np.median(coss))
    print("median norm ratio (pred/true):", np.median(nrs))
    print("per-dim corr: min %.3f med %.3f max %.3f" % (np.min(dim_corr), np.median(dim_corr), np.max(dim_corr)))
    print("ceiling: median corr %.3f, median nmae %.3f" % (np.median(ceil_corrs), np.median(ceil_nmaes)))


if __name__ == "__main__":
    main()
