"""从扰动 h5ad 计算每基因 lfc 范数表 (per-gene 幅度校准用)

lfc 定义与 cell-eval2 一致: 每细胞 CPM 归一后均值, log2 比值; 自基因置零, clip ±4
范数在对照均值 > 5 CPM 的基因子集上取 (与 predict 的 filt 一致)
"""

import argparse
import sys

sys.path.insert(0, "src")

import anndata as ad
import numpy as np

from linear_baseline import CONTROL, EPS, mean_cpm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctrl", required=True)
    p.add_argument("--perturbed", required=True, help="逗号分隔, 逐扰动切片")
    p.add_argument("--max-cells-per-pert", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    ctrl = ad.read_h5ad(args.ctrl)
    base = mean_cpm(ctrl)
    filt = base > 5
    out = {}
    for path in args.perturbed.split(","):
        a = ad.read_h5ad(path, backed="r")
        pos = {g: i for i, g in enumerate(a.var_names.to_numpy().astype(str))}
        labs = a.obs["target_gene"].astype(str).to_numpy()
        for pert in sorted(set(labs) - {CONTROL}):
            if pert in out:
                raise SystemExit(f"duplicate pert across files: {pert}")
            idx = np.where(labs == pert)[0]
            if len(idx) > args.max_cells_per_pert:
                idx = rng.choice(idx, args.max_cells_per_pert, replace=False)
            m = mean_cpm(a[idx].to_memory())
            lfc = np.log2((m + EPS) / (base + EPS))
            if pert in pos:
                lfc[pos[pert]] = 0.0
            lfc = np.clip(np.nan_to_num(lfc), -4, 4)
            out[pert] = float(np.linalg.norm(lfc[filt]))
        print(f"{path}: {len(out)} perts cumulative")
    genes = np.array(sorted(out))
    np.savez(args.out, genes=genes, norms=np.array([out[g] for g in genes]))
    ns = np.array(list(out.values()))
    print(f"wrote {args.out}: {len(genes)} genes, norm med {np.median(ns):.2f} "
          f"p10 {np.percentile(ns, 10):.2f} p90 {np.percentile(ns, 90):.2f}")


if __name__ == "__main__":
    main()
