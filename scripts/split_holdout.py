"""从 H1 h5ad 切出内部留出集: 控制 + 部分扰动作为真值, 其余扰动用于训练"""

import argparse
import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

CONTROL = "non-targeting"
CHUNK = 5000


def write_sparse(a, idx, path):
    # 分块转稀疏再 concat_on_disk, 防稠密切片把内存顶爆
    tmp = path + ".parts"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    parts = []
    for s in range(0, len(idx), CHUNK):
        sub = a[idx[s:s + CHUNK]].to_memory()
        sub.X = sparse.csr_matrix(sub.X)
        p = Path(tmp) / f"part_{s // CHUNK:03d}.h5ad"
        sub.write_h5ad(p)
        parts.append(p)
    # concat_on_disk 单文件分支对 str 调 is_dir 会炸, 传 Path
    ad.experimental.concat_on_disk(parts, path, axis=0, join="outer")
    shutil.rmtree(tmp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True)
    p.add_argument("--n-holdout", type=int, default=15)
    p.add_argument("--must-holdout", default="",
                   help="逗号分隔基因名, 强制进留出集, 如 2026 目标重叠基因")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pert-col", default="target_gene")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    a = ad.read_h5ad(args.h5ad, backed="r")
    labels = a.obs[args.pert_col].astype(str).to_numpy()
    perts = sorted(set(labels) - {CONTROL})
    must = [g for g in args.must_holdout.split(",") if g]
    missing = set(must) - set(perts)
    if missing:
        raise SystemExit(f"must-holdout genes not in h5ad: {sorted(missing)}")
    n_random = max(0, args.n_holdout - len(must))
    rest = sorted(set(perts) - set(must))
    holdout = sorted(must + list(rng.choice(rest, n_random, replace=False)))
    train = sorted(set(perts) - set(holdout))

    os.makedirs(args.outdir, exist_ok=True)
    idx_ho = np.where(np.isin(labels, holdout + [CONTROL]))[0]
    write_sparse(a, idx_ho, os.path.join(args.outdir, "real_heldout.h5ad"))
    idx_tr = np.where(np.isin(labels, train))[0]
    write_sparse(a, idx_tr, os.path.join(args.outdir, "train_perturbed.h5ad"))
    idx_ct = np.where(labels == CONTROL)[0]
    write_sparse(a, idx_ct, os.path.join(args.outdir, "train_controls.h5ad"))

    pd.Series(holdout).to_csv(os.path.join(args.outdir, "holdout_genes.csv"),
                              index=False, header=["target_gene"])
    pd.Series(train).to_csv(os.path.join(args.outdir, "train_genes.csv"),
                            index=False, header=["target_gene"])
    print(f"perts {len(perts)}: holdout {len(holdout)}, train {len(train)}")
    print("wrote", args.outdir)


if __name__ == "__main__":
    main()
