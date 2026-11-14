"""诊断: 模型预测 lfc vs 真值 lfc 的尺度关系, 决定预测该放大还是缩小

真值 lfc 用 cell-eval2 同款定义: 每细胞归一到 1e6 后取均值, log2 比值
只看过滤基因 (对照均值 > 5 CPM, 即被打分的基因集)
"""

import argparse
import sys

sys.path.insert(0, "src")

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

CONTROL = "non-targeting"
EPS = 1e-9


def lognorm_cpm(a):
    b = a.copy()
    sc.pp.normalize_total(b, target_sum=1e6)
    sc.pp.log1p(b)
    return b


def gene_features(ctrl_lognorm, n_pcs=40):
    sc.pp.pca(ctrl_lognorm, n_comps=n_pcs, svd_solver="arpack")
    pc = np.asarray(ctrl_lognorm.varm["PCs"])
    X = ctrl_lognorm.X
    X2 = X.power(2) if hasattr(X, "power") else np.square(X)
    mean = np.asarray(X.mean(0)).ravel()
    var = np.asarray(X2.mean(0)).ravel() - mean**2
    det = np.asarray((X > 0).mean(0)).ravel()
    stats = np.stack([np.log1p(mean), np.log1p(np.clip(var, 0, None)), det], 1)
    return np.hstack([pc, stats])


def mean_cpm(a):
    X = a.X
    tot = np.asarray(X.sum(1)).ravel()
    tot = np.where(tot > 0, tot, 1.0)
    Xn = X.multiply(1e6 / tot[:, None]) if sparse.issparse(X) else X * (1e6 / tot)[:, None]
    return np.asarray(Xn.mean(0)).ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ctrl", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--pert-col", default="target_gene")
    p.add_argument("--max-cells-ctrl", type=int, default=10000)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    m = np.load(args.model + "/model.npz") if not args.model.endswith(".npz") else np.load(args.model)
    W, b = m["W"], m["b"]
    fit_genes = m["genes"].astype(str)
    n_pcs = int(m["n_pcs"]) if "n_pcs" in m else 40
    prior_tf = [x for x in str(m["prior_tf"]).split(";") if x] if "prior_tf" in m else []
    prior_path = [x for x in str(m["prior_path"]).split(";") if x] if "prior_path" in m else []

    ctrl_full = ad.read_h5ad(args.ctrl, backed="r")
    if ctrl_full.n_obs > args.max_cells_ctrl:
        idx = rng.choice(ctrl_full.n_obs, args.max_cells_ctrl, replace=False)
        ctrl = ctrl_full[idx].to_memory()
    else:
        ctrl = ctrl_full.to_memory()
    ctrl_ln = lognorm_cpm(ctrl)
    F = gene_features(ctrl_ln, n_pcs)
    genes = ctrl.var_names.to_numpy().astype(str)
    if prior_tf or prior_path:
        from prior_features import build_priors
        pc = np.asarray(ctrl_ln.varm["PCs"])
        F = np.hstack([F, build_priors(ctrl_ln, pc, genes, prior_tf, prior_path)])
    if "fmean" in m and len(m["fmean"]) > 0:
        F = (F - m["fmean"]) / m["fstd"]
    pos = {g: i for i, g in enumerate(genes)}

    real = ad.read_h5ad(args.real, backed="r")
    labels = real.obs[args.pert_col].astype(str).to_numpy()
    cidx = np.where(labels == CONTROL)[0]
    base = mean_cpm(real[cidx].to_memory())
    filt = base > 5
    # predict 里用上下文对照 (这里就是 train_controls) 的 base 定义过滤子集
    base_ctx = mean_cpm(ctrl)
    filt_ctx = base_ctx > 5
    keep = np.array([pos.get(g, -1) for g in fit_genes])
    kept = keep[keep >= 0]
    fit_ok = np.where(keep >= 0)[0]

    perts = sorted(set(labels) - {CONTROL})
    print(f"{'pert':12s} {'corr':>6s} {'slope':>6s} {'nmae@1':>7s} {'best_s':>6s} {'sign>0.5':>8s} {'sign>1':>6s}")
    slopes = []
    nmaes = []
    for g in perts:
        lfc_pred = F[pos[g]] @ W + b
        lfc_full = np.zeros(len(genes))
        lfc_full[kept] = lfc_pred[fit_ok]
        if "lfc_norm" in m:
            n = np.linalg.norm(lfc_full[filt_ctx])
            if n > 1e-9:
                lfc_full = lfc_full * min(float(m["lfc_norm"]) / n, 20.0)
        m2 = mean_cpm(real[np.where(labels == g)[0]].to_memory())
        lfc_true = np.log2((m2 + EPS) / (base + EPS))
        f = filt & np.isfinite(lfc_true) & np.isfinite(lfc_full)
        x, y = lfc_full[f], lfc_true[f]
        corr = np.corrcoef(x, y)[0, 1]
        slope = np.sum(x * y) / np.sum(x * x)
        nmae1 = np.mean(np.abs(x - y)) / (np.mean(np.abs(y)) + EPS)
        best_s = np.sum(x * y) / np.sum(x * x)
        sgn = lambda t: (x[np.abs(y) > t] * y[np.abs(y) > t] > 0).mean()
        print(f"{g:12s} {corr:6.3f} {slope:6.3f} {nmae1:7.3f} {best_s:6.3f} {sgn(0.5):8.3f} {sgn(1.0):6.3f}")
        slopes.append(best_s)
        nmaes.append(nmae1)
    print("median best scale:", np.median(slopes))
    print("median nmae (post-norm):", np.median(nmaes))


if __name__ == "__main__":
    main()
