"""Nadig 4 系统一处理 leave-one-line-out (同基因面板, 无平台混杂).

评测位点 = 18080 轴上的 eval mask ∩ Nadig 6118 个被测基因 (5952 位).
四个系都没有表达 profile, 跳过相似度加权方法.
"""

import sys

import numpy as np

sys.path.insert(0, "scripts")
from bench_common import load_axis, load_nadig_lines
from bench_transfer import loo_run, save_results

HOLDOUTS = ["k562", "rpe1", "hepg2", "jurkat"]


def main():
    genes, pos, gidx, m = load_axis()
    lines = load_nadig_lines(genes, gidx)
    z = np.load("outputs/nadig_rows.npz", allow_pickle=True)
    in_g = np.zeros(len(genes), dtype=bool)
    in_g[z["gpos"]] = True
    posn = pos[in_g[pos]]
    print(f"nadig eval positions {len(posn)} / {len(pos)}")
    summary, rows_out = loo_run(lines, [L for L in HOLDOUTS if L in lines],
                                genes, posn, gidx=gidx)
    save_results("outputs/bench_transfer_nadig_v2", summary, rows_out)


if __name__ == "__main__":
    main()
