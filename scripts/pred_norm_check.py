"""检查 prior 模型裸预测 lfc 范数 与 真实范数 的关系 (15 holdout, 出样本)

用途: 判断能否用 预测范数 -> 真实范数 的映射做 per-gene 幅度校准 (对官方 300 全覆盖)
"""

import sys

sys.path.insert(0, "src")

import anndata as ad
import numpy as np
import pandas as pd

from linear_baseline import CONTROL, EPS, gene_features, lognorm_cpm, mean_cpm
from prior_features import build_priors


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/linear_h1_285_prior"
    m = np.load(model_dir + "/model.npz")
    W, b = m["W"], m["b"]
    fit_genes = m["genes"].astype(str)
    n_pcs = int(m["n_pcs"])
    prior_tf = [x for x in str(m["prior_tf"]).split(";") if x] if "prior_tf" in m else []
    prior_path = [x for x in str(m["prior_path"]).split(";") if x] if "prior_path" in m else []

    ctrl = ad.read_h5ad("data/internal/train_controls.h5ad")
    ctrl_ln = lognorm_cpm(ctrl)
    F = gene_features(ctrl_ln, n_pcs)
    genes = ctrl.var_names.to_numpy().astype(str)
    if prior_tf or prior_path:
        pc = np.asarray(ctrl_ln.varm["PCs"])
        F = np.hstack([F, build_priors(ctrl_ln, pc, genes, prior_tf, prior_path)])
    if "fmean" in m and len(m["fmean"]) > 0:
        F = (F - m["fmean"]) / m["fstd"]
    pos = {g: i for i, g in enumerate(genes)}
    base = mean_cpm(ctrl)
    filt = base > 5
    keep = np.array([pos.get(g, -1) for g in fit_genes])
    kept = keep[keep >= 0]
    fit_ok = np.where(keep >= 0)[0]

    rng = np.random.default_rng(0)
    a = ad.read_h5ad("data/internal/real_heldout.h5ad", backed="r")
    gpos = {g: i for i, g in enumerate(a.var_names.to_numpy().astype(str))}
    labs = a.obs["target_gene"].astype(str).to_numpy()
    rows = []
    for p in sorted(set(labs) - {CONTROL}):
        idx = np.where(labs == p)[0]
        if len(idx) > 400:
            idx = rng.choice(idx, 400, replace=False)
        mm = mean_cpm(a[idx].to_memory())
        lt = np.log2((mm + EPS) / (base + EPS))
        lt[gpos[p]] = 0
        lt = np.clip(np.nan_to_num(lt), -4, 4)
        true_n = float(np.linalg.norm(lt[filt]))
        lfc = np.zeros(len(genes))
        lfc[kept] = (F[pos[p]] @ W + b)[fit_ok]
        pred_n = float(np.linalg.norm(lfc[filt]))
        corr = float(np.corrcoef(lfc[filt], lt[filt])[0, 1])
        rows.append((p, pred_n, true_n, corr))
    df = pd.DataFrame(rows, columns=["gene", "pred_norm", "true_norm", "corr"])
    print(df.to_string(index=False))
    print("corr(pred_norm, true_norm) all 15:", round(df["pred_norm"].corr(df["true_norm"]), 3))
    sig = df[df["corr"] > 0.2]
    if len(sig) > 2:
        print(f"corr(pred_norm, true_norm) | within-gene corr>0.2 (n={len(sig)}):",
              round(sig["pred_norm"].corr(sig["true_norm"]), 3))
    np.savez("outputs/pred_norm_check.npz", genes=df["gene"].to_numpy(),
             pred_norm=df["pred_norm"].to_numpy(), true_norm=df["true_norm"].to_numpy())


if __name__ == "__main__":
    main()
