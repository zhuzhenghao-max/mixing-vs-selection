import json
import os
import sys

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

DATA = "data"
OUT = "outputs/eda"


def load_context(ctx, max_cells=None):
    a = ad.read_h5ad(os.path.join(DATA, f"context_{ctx}.h5ad"))
    if max_cells and a.n_obs > max_cells:
        rng = np.random.default_rng(0)
        a = a[rng.choice(a.n_obs, max_cells, replace=False)].copy()
    return a


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = {}

    ads = []
    for ctx in "ABC":
        a = load_context(ctx)
        sums = np.asarray(a.X.sum(1)).ravel()
        nz = np.asarray((a.X > 0).sum(1)).ravel()
        stats[ctx] = {
            "cells": int(a.n_obs),
            "median_umi": float(np.median(sums)),
            "mean_umi": float(sums.mean()),
            "median_genes": float(np.median(nz)),
        }
        a.obs["depth"] = sums
        a.obs["n_genes"] = nz
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.obs["context"] = ctx
        ads.append(a)

    for ctx, s in stats.items():
        print(ctx, s)
    json.dump(stats, open(os.path.join(OUT, "depth_stats.json"), "w"), indent=2)

    # depth comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ctx, a in zip("ABC", ads):
        axes[0].hist(np.log10(a.obs["depth"]), bins=60, alpha=0.5, label=ctx)
    axes[0].set_xlabel("log10 UMIs")
    axes[0].legend()
    for ctx, a in zip("ABC", ads):
        axes[1].hist(a.obs["n_genes"], bins=60, alpha=0.5, label=ctx)
    axes[1].set_xlabel("genes detected")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "depth_comparison.png"), dpi=100)

    # PCA + UMAP on merged lognorm data
    merged = ad.concat(ads, join="outer", label="batch", index_unique="-")
    merged.obs["context"] = np.array(["A", "B", "C"])[merged.obs["batch"].astype(int)]
    sc.pp.highly_variable_genes(merged, n_top_genes=3000, flavor="seurat_v3")
    sc.pp.pca(merged, n_comps=30, svd_solver="arpack")
    sc.pp.neighbors(merged, n_neighbors=15)
    sc.tl.umap(merged)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.pca(merged, color="context", ax=axes[0], show=False, legend_loc="on data")
    sc.pl.umap(merged, color="context", ax=axes[1], show=False, legend_loc="on data")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "context_pca_umap.png"), dpi=100)
    merged.write_h5ad(os.path.join(OUT, "merged_lognorm.h5ad"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
