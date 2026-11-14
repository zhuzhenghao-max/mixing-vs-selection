"""预切 context 对照细胞 → 小 npz (热点上传 AutoDL 用).

对每个 context h5ad: 取 ≤max-cells 对照细胞, log1p(CP1M) 投影到
GWPS∩vocab 基因轴 (与 scgpt_zero_shot.py 的 read_controls 口径一致),
float16 存盘 (1000 细胞 × ~7.5k 基因 ≈ 15MB).
scgpt_zero_shot.py 检测到 .npz 路径时直接加载, 不再读原始 h5ad.
"""

import argparse
import csv
import json

import numpy as np

CONTROL_NAMES = {"control", "non-targeting", "non_targeting", "nt", "nontargeting",
                 "intergenic", "ctrl", "control_non-targeting"}
LABEL_COLS = ["perturbation", "gene_target", "target_gene", "gene", "target",
              "condition"]
CH = 20000


def read_controls(path, gene_axis, max_cells, seed):
    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    var_names = a.var_names.to_numpy().astype(str)
    gpos = {g: i for i, g in enumerate(gene_axis)}
    col = np.array([gpos.get(g, -1) for g in var_names])
    has = col >= 0

    lab_col = next((c for c in LABEL_COLS if c in a.obs.columns), None)
    if lab_col is None:
        cmask = np.ones(a.n_obs, dtype=bool)
    else:
        labels = a.obs[lab_col].to_numpy().astype(str)
        cmask = np.array([l.lower().replace(" ", "") in CONTROL_NAMES
                          or "ontrol" in l for l in labels])
    cidx = np.where(cmask)[0]
    rng = np.random.default_rng(seed)
    if len(cidx) > max_cells:
        cidx = cidx[rng.choice(len(cidx), max_cells, replace=False)]
        cidx.sort()  # dense h5ad 花式索引要求升序
    print(f"{path}: {len(cidx)} control cells of {a.n_obs}")

    n_genes = len(gene_axis)
    vals = np.zeros((len(cidx), n_genes), dtype=np.float32)
    for s in range(0, len(cidx), CH):
        e = min(s + CH, len(cidx))
        blk = a.X[cidx[s:e]]
        if hasattr(blk, "toarray"):
            blk = blk.toarray()
        blk = np.asarray(blk, dtype=np.float32)
        tot = blk.sum(1)
        cpm = blk / np.maximum(tot, 1)[:, None] * 1e6
        for i in range(e - s):
            vals[s + i, col[has]] = np.log1p(cpm[i, has])
    a.file.close()
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="configs/contexts_2026.csv")
    ap.add_argument("--gwps",
                    default="data/external/gwps/replogle_2022_k562_gwps.h5ad")
    ap.add_argument("--model-dir", default="models/scgpt/scGPT_human")
    ap.add_argument("--out-dir", default="outputs/ctrl_slices")
    ap.add_argument("--max-cells", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import os
    from pathlib import Path

    import anndata as ad

    vkeys = set(json.load(open(Path(args.model_dir) / "vocab.json")).keys())
    gw = ad.read_h5ad(args.gwps, backed="r")
    gwnames = gw.var_names.to_numpy().astype(str)
    gw.file.close()
    axis_genes = np.array([g for g in gwnames if g in vkeys])
    print(f"gene axis: {len(axis_genes)} (of {len(gwnames)} GWPS genes)")

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.contexts) as f:
        for row in csv.DictReader(f):
            ctx, path = row["context"], row["path"]
            vals = read_controls(path, axis_genes, args.max_cells, args.seed)
            op = out_dir / f"ctrl_{ctx}.npz"
            np.savez(op, values=vals.astype(np.float16), genes=axis_genes)
            print(f"saved {op} {op.stat().st_size / 1e6:.1f}MB "
                  f"{vals.shape}", flush=True)


if __name__ == "__main__":
    main()
