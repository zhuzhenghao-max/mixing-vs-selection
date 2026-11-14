"""单独重跑 predict 的 parts 合并 (预测已完成的场景, 避免重算)."""

import csv
import os

import anndata as ad

PART_DIR = "outputs/pred_2026_v11.h5ad.parts"
OUT = "outputs/pred_2026_v11.h5ad"


def main():
    with open("data/final/pert_counts.csv", newline="", encoding="utf-8") as f:
        targets = [r["target_gene"] for r in csv.DictReader(f)]
    with open("configs/contexts_2026_final.csv", newline="", encoding="utf-8") as f:
        ctxs = [r["context"] for r in csv.DictReader(f)]
    files = [os.path.join(PART_DIR, f"{c}_{t}.h5ad") for c in ctxs for t in targets]
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing {len(missing)} parts: {missing[:5]}")
    print(f"concat {len(files)} parts -> {OUT}", flush=True)
    ad.experimental.concat_on_disk(files, OUT, axis=0, join="outer")
    print("done", flush=True)


if __name__ == "__main__":
    main()
