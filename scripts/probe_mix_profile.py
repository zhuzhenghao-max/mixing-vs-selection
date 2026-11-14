"""生产同款 unit-mix 行在 H1 真值上的方向/幅度分解.

对比臂: gwps / orion / unit-mix(w=0.4) / 均值响应, 每臂按 final 锚 (median filt-norm 12.90)
缩放, 扫全局尺度 s. 报告 profile 空间 cos 分布, mse 比值, PDS(cosine, panel 剔除).
"""

import json
import sys

import numpy as np

sys.path.insert(0, "src")
from linear_baseline import EPS  # noqa: E402

TS = 50000.0
ANCHOR = 12.90
W_ORION = 0.4


def main():
    d = np.load("outputs/proxy_profiles.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]
    genes = d["genes"].astype(str)
    gpos = {g: i for i, g in enumerate(genes)}
    n = len(names)
    ctrl_sum, ctrl_n, ctrl_total = d["ctrl_sum"], float(d["ctrl_n"]), float(d["ctrl_total"])
    sums, totals = d["sums"], d["totals"]
    gwps = d["gwps"].astype(np.float64)
    tpos = d["tpos"].astype(int)
    filt = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)["filt"].astype(bool)

    ctrl_rate = ctrl_sum / ctrl_n
    p_ctrl = ctrl_sum / ctrl_total
    ctrl_log = np.log1p(TS * p_ctrl)
    real_log = np.log1p(TS * (sums / totals[:, None]))
    B = real_log - ctrl_log
    B[:, tpos] = 0.0

    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    og = meta["genes"]
    ocol = {g: i for i, g in enumerate(og)}
    use = [i for i, g in enumerate(genes) if g in ocol]
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    rowmap = {}
    for r in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[r])], {})[int(pert_ids[r])] = r
    pid = meta["pert_to_id"]

    orion = np.zeros((n, len(genes)))
    has_o = np.zeros(n, bool)
    for i, t in enumerate(names):
        p = pid.get(t)
        if p is None:
            continue
        vs = []
        for cl in ("hct116", "hek293t"):
            r = rowmap[cl].get(p)
            if r is None:
                continue
            v = np.asarray(L[r, use], dtype=np.float64)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if np.abs(v).sum() > 0:
                vs.append(v)
        if vs:
            row = np.zeros(len(genes))
            row[use] = np.mean(vs, 0)
            row[~filt] = 0.0
            if np.abs(row).sum() > 0:
                orion[i] = row
                has_o[i] = True
    both = has_o & (np.linalg.norm(gwps[:, filt], axis=1) > 0)
    print(f"targets {n}, orion {has_o.sum()}, both {both.sum()}")

    def unit(v):
        return v / max(np.linalg.norm(v), 1e-9)

    mix = np.zeros_like(gwps)
    for i in range(n):
        if both[i]:
            mix[i] = W_ORION * unit(orion[i]) + (1 - W_ORION) * unit(gwps[i])

    def anchor_scale(rows, sub):
        med = np.median(np.linalg.norm(rows[sub][:, filt], axis=1))
        return ANCHOR / med, med

    arms = {}
    for nm, rows, sub in (
        ("gwps", gwps, np.ones(n, bool)),
        ("orion", orion, has_o),
        ("mix", mix, both),
    ):
        c, med = anchor_scale(rows, sub)
        arms[nm] = (np.clip(c * rows, -4, 4), sub)
        print(f"{nm}: natural median {med:.3f} -> c {c:.4g} (n={sub.sum()})")

    # 均值响应臂: 真实 delta 的行均值 (oracle shared response)
    mean_b = B[both].mean(0)
    mean_a = np.tile(mean_b, (n, 1))
    arms["mean"] = (mean_a, both)

    ss = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.63, 0.8, 1.0, 1.25, 1.5])
    print(f"\n{'arm':6s} {'s':5s} {'ratio':7s} {'cos_med':8s} {'cos_mean':8s} {'pds':6s}")
    for nm, (rows, sub) in arms.items():
        bb = B[sub]
        bn = np.linalg.norm(bb, axis=1)
        cos_at, ratio_at = {}, {}
        for s in ss:
            lfc = np.clip(s * rows[sub], -4, 4)
            rate = ctrl_rate[None, :] * np.exp2(lfc)
            p = rate / rate.sum(1, keepdims=True)
            A = np.log1p(TS * p) - ctrl_log[None, :]
            A[:, tpos[sub]] = 0.0
            an = np.linalg.norm(A, axis=1)
            cos = np.einsum("ij,ij->i", A, bb) / np.maximum(an * bn, 1e-12)
            ratio = float(np.sum((an ** 2 + bn ** 2 - 2 * cos * an * bn)) / np.sum(bn ** 2))
            # PDS: panel 剔除所有 proxy 靶基因
            keep = np.ones(len(genes), bool)
            keep[tpos] = False
            Ak, Bk = A[:, keep], bb[:, keep]
            An = Ak / np.maximum(np.linalg.norm(Ak, axis=1, keepdims=True), 1e-12)
            Bn = Bk / np.maximum(np.linalg.norm(Bk, axis=1, keepdims=True), 1e-12)
            sim = An @ Bn.T
            dist = 1.0 - sim
            m = int(np.count_nonzero(sub))
            ranks = np.empty(m)
            for i in range(m):
                row = dist[i]
                match = row[i]
                less = np.count_nonzero(row < match)
                eq = np.count_nonzero(row == match)
                ranks[i] = less + (eq - 1) / 2.0
            pds = float(np.mean(1.0 - ranks / (m - 1)))
            cos_at[s] = cos
            ratio_at[s] = (an ** 2 + bn ** 2 - 2 * cos * an * bn) / bn ** 2
            print(f"{nm:6s} {s:5.2f} {ratio:7.4f} {np.median(cos):8.4f} "
                  f"{np.mean(cos):8.4f} {pds:6.4f}")
        # oracle per-target magnitude bound: ratio_i = 1 - cos_i^2 at production s=1
        c1 = cos_at[1.0]
        w = bn ** 2 / np.sum(bn ** 2)
        oracle_scale = float(np.sum((1 - c1 ** 2) * w))
        grid_best = np.min(np.stack([ratio_at[s] for s in ss]), axis=0)
        oracle_grid = float(np.sum(grid_best * w))
        print(f"  [{nm}] oracle-scale ratio {oracle_scale:.4f} "
              f"(mean cos2 {np.mean(c1 ** 2):.4f}); oracle-grid scale ratio {oracle_grid:.4f}")

    np.savez("outputs/probe_mix_profile.npz", names=np.array(names), both=both,
             has_o=has_o, orion=orion, gwps=gwps, B=B, tpos=tpos)


if __name__ == "__main__":
    main()
