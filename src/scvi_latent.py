"""scVI 潜空间零样本: scVI 编码 -> 线性 F->dz 映射 -> 解码生成

train: scVI 在 H1 对照+扰动细胞上训练 (CPU)
fit_map: 每扰动 dz = mean(z_pert) - mean(z_ctrl), 岭回归 F -> dz
predict: 目标上下文 z_i + dz_pred 解码采样
"""

import argparse
import os
import shutil

# torch 必须最先导入, numpy/scanpy 先加载会让 c10.dll 初始化失败 (WinError 1114)
import torch
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy import sparse

from linear_baseline import gene_features, lognorm_cpm
from prior_features import build_priors

CONTROL = "non-targeting"
EPS = 1e-9
CLIP = 4.0


def load_cells(paths, pert_col, max_ctrl=12000, max_per_pert=400, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for path in paths.split(","):
        a = ad.read_h5ad(path, backed="r")
        labs = a.obs[pert_col].astype(str).to_numpy()
        keep = []
        for p in sorted(set(labs)):
            idx = np.where(labs == p)[0]
            if p == CONTROL and len(idx) > max_ctrl:
                idx = rng.choice(idx, max_ctrl, replace=False)
            elif p != CONTROL and len(idx) > max_per_pert:
                idx = rng.choice(idx, max_per_pert, replace=False)
            keep.append(idx)
        parts.append(a[np.concatenate(keep)].to_memory())
    out = ad.concat(parts, axis=0, join="outer", merge="same")
    del parts
    return out


def train(args):
    adata = load_cells(args.cells, args.pert_col,
                       max_ctrl=args.max_cells_ctrl, max_per_pert=args.max_cells_per_pert)
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=args.n_latent, n_layers=args.n_layers,
                            n_hidden=args.n_hidden)
    model.train(max_epochs=args.epochs, batch_size=args.batch, train_size=0.9,
                early_stopping=True, early_stopping_patience=args.patience,
                check_val_every_n_epoch=5, accelerator="cpu")
    model.save(args.outdir, save_anndata=True)
    print(f"saved scvi to {args.outdir}, {adata.n_obs} cells")


def get_z(model, adata, max_cells=0, seed=0):
    if max_cells and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        adata = adata[rng.choice(adata.n_obs, max_cells, replace=False)].copy()
    return model.get_latent_representation(adata, give_mean=True)


def fit_map(args):
    model = scvi.model.SCVI.load(args.model, adata=None)
    rng = np.random.default_rng(args.seed)
    ctrl = ad.read_h5ad(args.ctrl, backed="r")
    genes = ctrl.var_names.to_numpy().astype(str)
    if args.max_cells_ctrl and ctrl.n_obs > args.max_cells_ctrl:
        idx = rng.choice(ctrl.n_obs, args.max_cells_ctrl, replace=False)
        ctrl = ctrl[idx].to_memory()
    else:
        ctrl = ctrl.to_memory()
    scvi.model.SCVI.setup_anndata(ctrl)
    z_ctrl = get_z(model, ctrl)
    z_base = z_ctrl.mean(0)

    ctrl_ln = lognorm_cpm(ctrl)
    F = gene_features(ctrl_ln, args.n_pcs)
    if args.prior_tf or args.prior_path:
        pc = np.asarray(ctrl_ln.varm["PCs"])
        F = np.hstack([F, build_priors(ctrl_ln, pc, genes, args.prior_tf, args.prior_path)])
    pos = {g: i for i, g in enumerate(genes)}

    DZ, names = [], []
    for path in args.perturbed.split(","):
        pert = ad.read_h5ad(path, backed="r")
        labs = pert.obs[args.pert_col].astype(str).to_numpy()
        for p in sorted(set(labs) - {CONTROL}):
            idx = np.where(labs == p)[0]
            if args.max_cells_per_pert and len(idx) > args.max_cells_per_pert:
                idx = rng.choice(idx, args.max_cells_per_pert, replace=False)
            sub = pert[idx].to_memory()
            scvi.model.SCVI.setup_anndata(sub)
            z = get_z(model, sub)
            DZ.append(z.mean(0) - z_base)
            names.append(p)
        print(f"{path}: {len(set(labs)) - 1} perts")

    missing = [g for g in names if g not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} train targets not in panel: {missing[:10]}")
    DZ = np.stack(DZ)
    F_tr = np.stack([F[pos[g]] for g in names])
    W, b = fit_ridge(F_tr, DZ, args.lam)

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "map.npz"), W=W, b=b, genes=genes,
             n_pcs=args.n_pcs,
             prior_tf=";".join(args.prior_tf), prior_path=";".join(args.prior_path),
             z_base=z_base)
    print(f"fit dz map on {len(names)} perts, W {W.shape}")


def fit_ridge(F, D, lam):
    n = F.shape[0]
    F1 = np.hstack([F, np.ones((n, 1))])
    A = F1.T @ F1 + lam * np.eye(F1.shape[1])
    A[-1, -1] -= lam
    W = np.linalg.solve(A, F1.T @ D)
    return W[:-1], W[-1]


def gen_cells_scvi(model, z_shift, libs, rng):
    z = torch.as_tensor(np.clip(z_shift, -6, 6), dtype=torch.float32)
    # generative 收 log library, 传原始 UMI 会 exp 溢出成 inf
    lib = torch.as_tensor(np.log(np.maximum(libs, 1.0)), dtype=torch.float32)[:, None]
    batch = torch.zeros(z.shape[0], dtype=torch.long)
    with torch.no_grad():
        out = model.module.generative(z, lib, batch)
        p = out["px"].mean.detach().numpy()
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0, None)
    return sample_cells(p, libs, rng)


def sample_cells(p, libs, rng):
    # 逐细胞 cdf + searchsorted (基准 1.35s/400 细胞, 比 scipy 全局方案快 ~2x);
    # float64 归一化且强制末列=1, 避免 float32 误差致 multinomial 校验失败
    n, g = p.shape
    cdf = np.cumsum(np.asarray(p, dtype=np.float64), axis=1)
    cdf /= np.where(cdf[:, -1:] > 0, cdf[:, -1:], 1.0)
    cdf[:, -1] = 1.0
    rows, cols, vals = [], [], []
    for i in range(n):
        ni = int(libs[i])
        if ni <= 0:
            continue
        idx = np.minimum(np.searchsorted(cdf[i], rng.random(ni), side="right"), g - 1)
        ri = np.bincount(idx, minlength=g).astype(np.float32)
        nz = np.nonzero(ri)[0]
        rows.append(np.full(nz.size, i))
        cols.append(nz)
        vals.append(ri[nz])
    if not rows:
        return sparse.csr_matrix((n, g), dtype=np.float32)
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, g))


def expand_to_panel(X, ctrl_rows, col_src, present, miss_full, n_full):
    # 模型空间采样 X (n, n_model) 映射回面板列序; 面板独有列用配对对照细胞原始 counts 填
    Xm = X[:, present].tocoo()
    mc = ctrl_rows[:, miss_full].tocoo()
    rows = np.concatenate([Xm.row, mc.row])
    cols = np.concatenate([col_src[present][Xm.col], miss_full[mc.col]])
    vals = np.concatenate([Xm.data, mc.data])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(X.shape[0], n_full))


def decode_px(model, z, lib, chunk=4096):
    z = torch.as_tensor(np.clip(z, -6, 6), dtype=torch.float32)
    lib_t = torch.as_tensor(np.log(np.maximum(lib, 1.0)), dtype=torch.float32)
    batch = torch.zeros(z.shape[0], dtype=torch.long)
    parts = []
    for i in range(0, z.shape[0], chunk):
        with torch.no_grad():
            o = model.module.generative(z[i:i + chunk], lib_t[i:i + chunk, None], batch[i:i + chunk])
            p = o["px"].mean.detach().numpy()
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(np.clip(p, 0, None))
    return np.vstack(parts)


def predict(args):
    model = scvi.model.SCVI.load(args.model, adata=None)
    m = None
    W = b = None
    n_pcs = 40
    prior_tf = prior_path = []
    if args.map:
        m = np.load(os.path.join(args.map, "map.npz"))
        W, b = m["W"], m["b"]
        n_pcs = int(m["n_pcs"]) if "n_pcs" in m else 40
        prior_tf = [x for x in str(m["prior_tf"]).split(";") if x] if "prior_tf" in m else []
        prior_path = [x for x in str(m["prior_path"]).split(";") if x] if "prior_path" in m else []
    lfc_map = None
    if args.lfc_map:
        lm = np.load(args.lfc_map)
        LW, Lb = lm["W"], lm["b"]
        lg = lm["genes"].astype(str)
        if "n_pcs" in lm:
            n_pcs = int(lm["n_pcs"])
        if "prior_tf" in lm:
            prior_tf = [x for x in str(lm["prior_tf"]).split(";") if x]
        if "prior_path" in lm:
            prior_path = [x for x in str(lm["prior_path"]).split(";") if x]
        lfc_map = (LW, Lb, {g: i for i, g in enumerate(lg)})
    if lfc_map is None and W is None:
        raise SystemExit("需要 --map 或 --lfc-map 之一")

    ctx = pd.read_csv(args.contexts)
    targets = pd.read_csv(args.targets)["target_gene"].tolist()
    rng = np.random.default_rng(args.seed)
    model_genes = model.adata.var_names.to_numpy().astype(str)

    part_dir = args.out + ".parts"
    shutil.rmtree(part_dir, ignore_errors=True)
    os.makedirs(part_dir)
    files = []
    for _, row in ctx.iterrows():
        # 面板(如 18533) -> 模型基因集(18080) 对齐: 模型缺的 3 个基因补零列,
        # 面板独有的基因采样后用配对对照细胞 counts 填回
        a_full = ad.read_h5ad(row["path"])
        full_genes = a_full.var_names.to_numpy().astype(str)
        fpos = {g: i for i, g in enumerate(full_genes)}
        col_src = np.array([fpos.get(g, -1) for g in model_genes])
        present = col_src >= 0
        if present.all():
            if np.array_equal(col_src, np.arange(len(model_genes))):
                a = a_full  # 基因序已与模型完全一致, 跳拷贝省内存 (train_controls 这类大文件)
            else:
                a = a_full[:, col_src].copy()
        else:
            Xs = a_full.X.tocsr()[:, col_src[present]].tocoo()
            mcols = np.flatnonzero(present)
            a = ad.AnnData(
                X=sparse.csr_matrix((Xs.data, (Xs.row, mcols[Xs.col])),
                                    shape=(a_full.n_obs, len(model_genes))),
                obs=a_full.obs.copy(), var=pd.DataFrame(index=model_genes))
        scvi.model.SCVI.setup_anndata(a)
        z = get_z(model, a)
        libs = np.asarray(a.X.sum(1)).ravel()
        ctrl_ln = lognorm_cpm(a)
        F = gene_features(ctrl_ln, n_pcs)
        genes = model_genes
        if prior_tf or prior_path:
            pc = np.asarray(ctrl_ln.varm["PCs"])
            F = np.hstack([F, build_priors(ctrl_ln, pc, genes, prior_tf, prior_path)])
        pos = {g: i for i, g in enumerate(genes)}
        missing = [g for g in targets if g not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} targets not in model: {missing[:10]}")
        miss_full = np.flatnonzero(np.isin(full_genes, model_genes, invert=True))
        px_ctrl = None
        lsel = None
        if lfc_map is not None:
            LW, Lb, lpos = lfc_map
            lsel = np.array([lpos[g2] for g2 in genes])
            px_ctrl = decode_px(model, z, libs)
        for g in targets:
            cell_idx = rng.choice(z.shape[0], args.n_cells, replace=False)
            if lfc_map is not None:
                lfc = args.lfc_scale * (F[pos[g]] @ LW + Lb)[lsel]
                p = px_ctrl[cell_idx] * np.exp2(lfc[None, :])
                X = sample_cells(p, libs[cell_idx], rng)
            else:
                dz = args.dz_scale * (F[pos[g]] @ W + b)
                X = gen_cells_scvi(model, z[cell_idx] + dz, libs[cell_idx], rng)
            X = expand_to_panel(X, a_full.X[cell_idx], col_src, present, miss_full,
                                len(full_genes))
            obs = pd.DataFrame({"target_gene": g, "context": row["context"]},
                               index=[f"{row['context']}_{g}_{k}" for k in range(args.n_cells)])
            part = ad.AnnData(X=X, obs=obs)
            part.var_names = full_genes
            ppath = os.path.join(part_dir, f"{row['context']}_{g}.h5ad")
            part.write_h5ad(ppath)
            files.append(ppath)
        print(f"{row['context']}: {len(targets)} perts x {args.n_cells} cells "
              f"(panel {len(full_genes)}, model {int(present.sum())}, ctrl-fill {miss_full.size})")

    ad.experimental.concat_on_disk(files, args.out, axis=0, join="outer")
    shutil.rmtree(part_dir)
    print(f"wrote {args.out} from {len(files)} parts")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--cells", required=True, help="h5ad 逗号分隔, 对照+扰动一起")
    t.add_argument("--pert-col", default="target_gene")
    t.add_argument("--max-cells-ctrl", type=int, default=12000)
    t.add_argument("--max-cells-per-pert", type=int, default=200)
    t.add_argument("--n-latent", type=int, default=32)
    t.add_argument("--n-layers", type=int, default=2)
    t.add_argument("--n-hidden", type=int, default=128)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--outdir", required=True)
    t.set_defaults(fn=train)

    f = sub.add_parser("fit-map")
    f.add_argument("--model", required=True, help="scVI 模型目录")
    f.add_argument("--ctrl", required=True)
    f.add_argument("--perturbed", required=True)
    f.add_argument("--pert-col", default="target_gene")
    f.add_argument("--lam", type=float, default=1.0)
    f.add_argument("--max-cells-ctrl", type=int, default=12000)
    f.add_argument("--max-cells-per-pert", type=int, default=400)
    f.add_argument("--n-pcs", type=int, default=40)
    f.add_argument("--prior-tf", nargs="*", default=[])
    f.add_argument("--prior-path", nargs="*", default=[])
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--outdir", required=True)
    f.set_defaults(fn=fit_map)

    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True, help="scVI 模型目录")
    pr.add_argument("--map", default="", help="fit-map 输出目录 (map.npz)")
    pr.add_argument("--lfc-map", default="", help="混合模式: F->解码lfc 的 npz")
    pr.add_argument("--contexts", required=True)
    pr.add_argument("--targets", required=True)
    pr.add_argument("--n-cells", type=int, default=400)
    pr.add_argument("--dz-scale", type=float, default=1.0,
                    help="dz 整体缩放, 校准时用")
    pr.add_argument("--lfc-scale", type=float, default=1.0,
                    help="混合模式 lfc 缩放, 校准时用")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", required=True)
    pr.set_defaults(fn=predict)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
