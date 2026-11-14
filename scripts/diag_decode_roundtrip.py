"""诊断 scVI 编解码往返在目标上下文上的保真度

解码出的对照细胞 profile 与真实对照偏差大 → 任何经 scVI 解码的输出
(dz-map / hybrid) 在目标细胞系上都会失真 (表达基线漂移会假装成扰动效应)
"""

import argparse
import sys

sys.path.insert(0, "src")

# torch 必须最先导入, numpy/scanpy 先加载会让 c10.dll 初始化失败 (WinError 1114)
import torch  # noqa: F401

import anndata as ad
import numpy as np
import pandas as pd
import scvi
from scipy import sparse

from linear_baseline import mean_cpm
from scvi_latent import decode_px, get_z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--contexts", required=True)
    p.add_argument("--max-cells", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = scvi.model.SCVI.load(args.model, adata=None)
    model_genes = model.adata.var_names.to_numpy().astype(str)
    ctx = pd.read_csv(args.contexts)

    for _, row in ctx.iterrows():
        a_full = ad.read_h5ad(row["path"])
        full_genes = a_full.var_names.to_numpy().astype(str)
        fpos = {g: i for i, g in enumerate(full_genes)}
        col_src = np.array([fpos.get(g, -1) for g in model_genes])
        present = col_src >= 0
        if not present.all():
            print(f"{row['context']}: {int((~present).sum())} model genes missing from panel, skipped cols")
        Xs = a_full.X.tocsr()[:, col_src[present]].tocoo()
        mcols = np.flatnonzero(present)
        a = ad.AnnData(
            X=sparse.csr_matrix((Xs.data, (Xs.row, mcols[Xs.col])),
                                shape=(a_full.n_obs, len(model_genes))),
            obs=a_full.obs.copy(), var=pd.DataFrame(index=model_genes))
        scvi.model.SCVI.setup_anndata(a)
        z = get_z(model, a, max_cells=args.max_cells, seed=args.seed)
        if z.shape[0] < a.n_obs:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(a.n_obs, z.shape[0], replace=False)
        else:
            idx = np.arange(a.n_obs)
        libs = np.asarray(a.X[idx].sum(1)).ravel()
        px = decode_px(model, z, libs)
        real_prof = mean_cpm(a[idx])
        # px 行按自身行和归一化到比例, 避免对 generative 的 library 校准做假设
        dec_prof = (px / px.sum(1, keepdims=True)).mean(0) * 1e6
        filt = real_prof > 5
        c = np.corrcoef(dec_prof[filt], real_prof[filt])[0, 1]
        logratio = np.log2((dec_prof[filt] + 1) / (real_prof[filt] + 1))
        print(f"{row['context']}: cells {z.shape[0]}, profile corr {c:.3f}, "
              f"log2 ratio med {np.median(logratio):+.2f}, "
              f"|log2 ratio| med {np.median(np.abs(logratio)):.2f}")


if __name__ == "__main__":
    main()
