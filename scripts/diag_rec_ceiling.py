"""诊断: scVI 重构天花板 vs z 平移天花板 (留出集)

rec: 解码真实扰动细胞自己的 z/lib, 对比真实 lfc
shift: 解码 ctrl z + 真实 dz, 对比真实 lfc
"""

import argparse
import sys

sys.path.insert(0, "src")

import torch  # 必须最先 import, 否则 c10.dll 崩
import anndata as ad
import numpy as np
import scvi

from linear_baseline import mean_cpm, EPS

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
    p.add_argument("--real", required=True)
    p.add_argument("--pert-col", default="target_gene")
    p.add_argument("--max-cells", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    model = scvi.model.SCVI.load(args.model, adata=None)

    ctrl = ad.read_h5ad(args.ctrl, backed="r")
    if ctrl.n_obs > 4000:
        idx = rng.choice(ctrl.n_obs, 4000, replace=False)
        ctrl = ctrl[idx].to_memory()
    else:
        ctrl = ctrl.to_memory()
    scvi.model.SCVI.setup_anndata(ctrl)
    z_ctrl = model.get_latent_representation(ctrl, give_mean=True)
    z_base = z_ctrl.mean(0)
    libs_ctrl = np.asarray(ctrl.X.sum(1)).ravel()
    rates_ctrl = decode_mean(model, z_ctrl, libs_ctrl)
    base = mean_cpm(ctrl)
    filt = base > 5

    real = ad.read_h5ad(args.real, backed="r")
    labels = real.obs[args.pert_col].astype(str).to_numpy()
    cidx = np.where(labels == CONTROL)[0]
    if len(cidx) > args.max_cells:
        cidx = rng.choice(cidx, args.max_cells, replace=False)
    rctrl = real[cidx].to_memory()
    scvi.model.SCVI.setup_anndata(rctrl)
    z_rctrl = model.get_latent_representation(rctrl, give_mean=True)
    libs_rctrl = np.asarray(rctrl.X.sum(1)).ravel()
    rates_rctrl = decode_mean(model, z_rctrl, libs_rctrl)
    rbase = mean_cpm(rctrl)
    rfilt = rbase > 5

    print(f"{'pert':12s} {'rec_corr':>9s} {'rec_nmae':>9s} {'shift_corr':>10s} {'shift_nmae':>10s}")
    for g in sorted(set(labels) - {CONTROL}):
        idx = np.where(labels == g)[0]
        if len(idx) > args.max_cells:
            idx = rng.choice(idx, args.max_cells, replace=False)
        sub = real[idx].to_memory()
        scvi.model.SCVI.setup_anndata(sub)
        z_p = model.get_latent_representation(sub, give_mean=True)
        libs_p = np.asarray(sub.X.sum(1)).ravel()
        rates_rec = decode_mean(model, z_p, libs_p)
        rates_shift = decode_mean(model, z_ctrl + (z_p.mean(0) - z_base), libs_ctrl)

        m2 = mean_cpm(sub)
        lfc_true = np.log2((m2 + EPS) / (rbase + EPS))
        lfc_rec = np.log2((rates_rec + EPS) / (rates_rctrl + EPS))
        lfc_shift = np.log2((rates_shift + EPS) / (rates_ctrl + EPS))
        f = rfilt & np.isfinite(lfc_true) & np.isfinite(lfc_rec)
        cr = np.corrcoef(lfc_rec[f], lfc_true[f])[0, 1]
        nr = np.mean(np.abs(lfc_rec[f] - lfc_true[f])) / (np.mean(np.abs(lfc_true[f])) + EPS)
        f = rfilt & np.isfinite(lfc_true) & np.isfinite(lfc_shift)
        cs = np.corrcoef(lfc_shift[f], lfc_true[f])[0, 1]
        ns = np.mean(np.abs(lfc_shift[f] - lfc_true[f])) / (np.mean(np.abs(lfc_true[f])) + EPS)
        print(f"{g:12s} {cr:9.3f} {nr:9.3f} {cs:10.3f} {ns:10.3f}")


if __name__ == "__main__":
    main()
