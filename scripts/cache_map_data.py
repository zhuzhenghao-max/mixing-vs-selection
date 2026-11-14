"""缓存 map 拟合数据: 285 训练扰动的特征 + 真实 dz + 解码 lfc

解码 lfc 用真实 dz 平移 ctrl z 得到, 是解码器能表达的最干净扰动信号
用途: 岭回归 F -> 解码 lfc (混合模型), 或换 regressor 拟合 F -> dz
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
    p.add_argument("--ctrl", required=True)
    p.add_argument("--perturbed", required=True)
    p.add_argument("--pert-col", default="target_gene")
    p.add_argument("--max-cells-ctrl", type=int, default=3000)
    p.add_argument("--max-cells-per-pert", type=int, default=200)
    p.add_argument("--n-pcs", type=int, default=40)
    p.add_argument("--prior-tf", nargs="*", default=[])
    p.add_argument("--prior-path", nargs="*", default=[])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    model = scvi.model.SCVI.load(args.model, adata=None)

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
    F = gene_features(ctrl_ln, args.n_pcs)
    if args.prior_tf or args.prior_path:
        pc = np.asarray(ctrl_ln.varm["PCs"])
        F = np.hstack([F, build_priors(ctrl_ln, pc, genes, args.prior_tf, args.prior_path)])
    pos = {g: i for i, g in enumerate(genes)}

    LFC, DZ, names = [], [], []
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
            dz = z_pert.mean(0) - z_base
            rates_pert = decode_mean(model, z_ctrl + dz, libs)
            lfc_dec = np.log2((rates_pert + EPS) / (rates_ctrl + EPS))
            LFC.append(lfc_dec.astype(np.float32))
            DZ.append(dz.astype(np.float32))
            names.append(g)
        print(f"{path}: {len(set(labs)) - 1} perts")

    missing = [g for g in names if g not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} train targets not in panel: {missing[:10]}")
    LFC = np.stack(LFC)
    DZ = np.stack(DZ)
    np.savez(args.out, F=F.astype(np.float32), genes=genes, train_names=np.array(names),
             LFC=LFC, DZ=DZ, base=base.astype(np.float32), filt=filt,
             n_pcs=args.n_pcs,
             prior_tf=";".join(args.prior_tf), prior_path=";".join(args.prior_path))
    print(f"saved {args.out}: F {F.shape}, LFC {LFC.shape}, DZ {DZ.shape}")


if __name__ == "__main__":
    main()
