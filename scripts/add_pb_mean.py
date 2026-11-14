"""把 r5 的 pb_mean (跨靶均值行) 加回 scGPT 零样本 delta.

r5 训练目标是中心化 pseudobulk delta, 零样本 --pred-mode absolute 输出
mean(mlm_output)-mean(ctrl) = 中心化预测; 与未中心化的 ref 行比较或生产
混入前必须加回 pb_mean (按基因名映射到 18080 轴).

用法:
  python scripts/add_pb_mean.py --pred outputs/scgpt_delta_check_r5.npz \
    --pb-mean outputs/scgpt_ft_r5/pb_mean.npz \
    --out outputs/scgpt_delta_check_r5_pbadd.npz
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pb-mean", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.pred, allow_pickle=True)
    pb = np.load(args.pb_mean, allow_pickle=True)
    axis = [str(g) for g in z["genes"].astype(str)]
    pm = {str(g): float(v) for g, v in zip(pb["genes"], pb["pb_mean"])}
    add = np.array([pm.get(g, 0.0) for g in axis], dtype=np.float32)
    hit = sum(g in pm for g in axis)
    print(f"pb_mean genes {len(pm)}, matched on axis {hit}/{len(axis)}, "
          f"add norm {np.linalg.norm(add):.3f}")

    out = {k: z[k] for k in z.files if not k.startswith("delta_")}
    for k in z.files:
        if k.startswith("delta_"):
            out[k] = (z[k] + add[None, :]).astype(np.float32)
    np.savez(args.out, **out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
