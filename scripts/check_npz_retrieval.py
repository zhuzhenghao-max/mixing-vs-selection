"""检索 pds 检查: 预测行 (scGPT 零样本 / 实测行) vs 参考实测行池.

AutoDL 验证步骤用 (rpe1 跨系零样本 / GWPS 自检). 本地对照跑同脚本同池,
数字直接可比.

- 查询: --pred 可为 scGPT 输出 npz (targets + delta_<ctx>) 或 rows npz (names+rows)
  (rows npz 用 --targets csv 取子集)
- 参考池: --ref rows npz 全体 (names + rows), 轴 = 18080
- 位置: --mask (默认 outputs/eval_pos_mask.npz, 10778 位, multi_line_loo 口径)
- pds: 每 target, q 与池中所有行 cos 相似度, 真值行 tie-midrank (分母 n-1)
- 两口径: ref 原样 unit / ref 减均值后 unit (q 均 unit)

用法:
  python scripts/check_npz_retrieval.py --pred outputs/scgpt_delta_rpe1.npz \
    --pred-key delta_rpe1 --ref outputs/rpe1_rows.npz \
    --targets data/rpe1_check_targets.csv
"""

import argparse
import csv

import numpy as np


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def read_targets_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    hdr = rows[0]
    ci = 0
    for c in ("target", "target_gene", "gene"):
        if c in hdr:
            ci = hdr.index(c)
            rows = rows[1:]
            break
    return [r[ci] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-key", default=None)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--targets", default=None, help="查询子集 csv; 缺省 = 全部查询行")
    ap.add_argument("--mask", default="outputs/eval_pos_mask.npz")
    ap.add_argument("--exclude-self", action="store_true",
                    help="每查询剔除其靶基因位置 (官方 pds 口径); query/ref 同剔后重归一化")
    ap.add_argument("--center-query", action="store_true",
                    help="查询行减去 common 靶的均值行 (去共享分量)")
    ap.add_argument("--axis", default="outputs/lfc_override_2026_v10.npz",
                    help="18080 基因轴来源 (pred 无 genes 键时, exclude-self 用)")
    args = ap.parse_args()

    mask = np.load(args.mask)["mask"]
    pos = np.where(mask)[0]

    ref = np.load(args.ref, allow_pickle=True)
    rnames = [str(x) for x in ref["names"].astype(str)]
    R = ref["rows"].astype(np.float64)[:, pos]
    ridx = {n: i for i, n in enumerate(rnames)}

    z = np.load(args.pred, allow_pickle=True)
    if "targets" in z.files:
        pnames = [str(x) for x in z["targets"].astype(str)]
        key = args.pred_key or [k for k in z.files if k.startswith("delta")][0]
        P = z[key].astype(np.float64)[:, pos]
        print(f"pred key: {key}")
    else:
        pnames = [str(x) for x in z["names"].astype(str)]
        P = z["rows"].astype(np.float64)[:, pos]

    axis_genes = ([str(g) for g in z["genes"].astype(str)]
                  if "genes" in z.files else None)
    if args.exclude_self and axis_genes is None:
        ax = np.load(args.axis, allow_pickle=True)
        axis_genes = [str(g) for g in ax["genes"].astype(str)]
    gpos = {g: i for i, g in enumerate(axis_genes)} if axis_genes else {}
    local_of_global = {int(g): i for i, g in enumerate(pos)}

    want = read_targets_csv(args.targets) if args.targets else pnames
    pidx = {n: i for i, n in enumerate(pnames)}
    missing_q = [t for t in want if t not in pidx]
    common = [t for t in want if t in pidx and t in ridx]
    missing_r = [t for t in want if t in pidx and t not in ridx]
    print(f"query rows {len(P)}, ref rows {len(rnames)}, common {len(common)}"
          f" (missing in pred: {len(missing_q)}, missing in ref: {len(missing_r)})")
    if missing_q:
        print(f"  pred missing: {missing_q[:10]}{'...' if len(missing_q) > 10 else ''}")
    if missing_r:
        print(f"  ref missing: {missing_r[:10]}{'...' if len(missing_r) > 10 else ''}")
    assert len(common) >= 5

    Puse = P
    if args.center_query:
        Puse = P - P[[pidx[t] for t in common]].mean(0, keepdims=True)
        print(f"center-query: mean row over {len(common)} targets subtracted")
    suffix = ("_selfex" if args.exclude_self else "") + ("_cq" if args.center_query else "")

    ref_raw = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
    Rc = R - R.mean(0)
    ref_res = Rc / np.maximum(np.linalg.norm(Rc, axis=1, keepdims=True), 1e-9)

    n = len(rnames)
    res = {}
    for tag, refs in (("raw", ref_raw), ("resid", ref_res)):
        pds = []
        for t in common:
            q = Puse[pidx[t]].copy()
            rr = refs
            if args.exclude_self:
                li = local_of_global.get(gpos.get(t, -1))
                if li is not None:
                    q[li] = 0.0
                    rr = refs.copy()
                    rr[:, li] = 0.0
                    rr = rr / np.maximum(
                        np.linalg.norm(rr, axis=1, keepdims=True), 1e-9)
            q = unit(q)
            s = rr @ q
            sv = s[ridx[t]]
            rank = 1.0 + np.sum(s > sv) + 0.5 * (np.sum(s == sv) - 1)
            pds.append(1.0 - (rank - 1.0) / (n - 1.0))
        pds = np.array(pds)
        res[tag] = pds
        print(f"[{tag}{suffix}] pds median {np.median(pds):.4f} "
              f"mean {pds.mean():.4f} (n={len(pds)})")

    out = args.pred.replace(".npz", "") + f"_pds{suffix}.npz"
    np.savez(out, targets=np.array(common), **res)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
