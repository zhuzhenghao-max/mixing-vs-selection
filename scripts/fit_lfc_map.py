"""拟合 F -> 解码 lfc 的岭回归 (混合模型)

缓存来自 cache_map_data.py: LFC 是解码器用真实 dz 生成的扰动信号,
比真值 lfc 噪声小, 且天然落在解码器能表达的范围内
"""

import argparse
import os
import sys

sys.path.insert(0, "src")

import numpy as np

from linear_baseline import fit_ridge, EPS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--lams", default="1,10,100")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    m = np.load(args.cache)
    F_all, genes = m["F"], m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC = m["LFC"]
    filt = m["filt"]
    pos = {g: i for i, g in enumerate(genes)}
    F_tr = np.stack([F_all[pos[g]] for g in train_names])
    n = F_tr.shape[0]
    rng = np.random.default_rng(args.seed)
    folds = rng.permutation(n) % args.folds

    lams = [float(x) for x in args.lams.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    print(f"features {F_tr.shape[1]}, targets {len(train_names)}, genes {len(genes)}")
    for lam in lams:
        corrs = []
        for k in range(args.folds):
            te = folds == k
            tr = ~te
            W, b = fit_ridge(F_tr[tr], LFC[tr], lam)
            pred = F_tr[te] @ W + b
            y_true = LFC[te]
            for j in range(pred.shape[0]):
                f = filt & np.isfinite(pred[j]) & np.isfinite(y_true[j])
                x, y = pred[j][f], y_true[j][f]
                if np.std(y) > 1e-9 and np.std(x) > 1e-9:
                    corrs.append(np.corrcoef(x, y)[0, 1])
        print(f"lam {lam:g}: CV corr median {np.median(corrs):.3f} mean {np.mean(corrs):.3f}")

        W, b = fit_ridge(F_tr, LFC, lam)
        out = os.path.join(args.outdir, f"lfc_map_lam{lam:g}.npz")
        np.savez(out, W=W.astype(np.float32), b=b.astype(np.float32), genes=genes,
                 n_pcs=int(m["n_pcs"]),
                 prior_tf=str(m["prior_tf"]), prior_path=str(m["prior_path"]))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
