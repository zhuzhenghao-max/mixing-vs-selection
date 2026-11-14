"""内部代理实验: GWPS 行预测 H1 Training 真值, 在官方 bulk_lognorm 空间算 mse 比值.

对每个有 GWPS 行的 Training 靶:
  pred rate ∝ ctrl 原始均值 * 2^(s * thresholded_gwps_lfc), 归一化 (与 gen_cells 一致)
  pred_real 双方都取 log1p(TS * group_sum / group_total), TS=50000
  numerator = ||logp_pred - logp_real||^2 (剔除靶基因)
  denominator = ||logp_ctrl - logp_real||^2
聚合 ratio_of_sums, 扫描全局缩放 s 与弱效应阈值 T.
"""

import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from linear_baseline import CONTROL, EPS, mean_cpm  # noqa: E402

TS = 50000.0
PATH = "data/adata_Training_2025.h5ad"
GWPS = "outputs/gwps_rows.npz"
FILT_NPZ = "outputs/scvi_map_cache.npz"
CHUNK = 4000


def main():
    a = ad.read_h5ad(PATH, backed="r")
    var = a.var_names.to_numpy().astype(str)
    gpos = {g: i for i, g in enumerate(var)}
    labels = a.obs["target_gene"].astype(str).to_numpy()

    g = np.load(GWPS, allow_pickle=True)
    gnames = [str(x) for x in g["names"]]
    grows = g["rows"].astype(np.float64)
    gmap = {nm: i for i, nm in enumerate(gnames)}
    have = [t for t in sorted(set(labels) - {CONTROL}) if t in gmap]
    print(f"{len(have)} targets with GWPS rows", flush=True)

    filt = np.load(FILT_NPZ, allow_pickle=True)["filt"].astype(bool)
    assert len(filt) == len(var), (len(filt), len(var))
    # scvi filt 基于对照 mean-CPM > 5
    n = len(a)
    sums = {t: np.zeros(len(var)) for t in have}
    totals = {t: 0.0 for t in have}
    ctrl_sum = np.zeros(len(var))
    ctrl_total = 0.0
    ctrl_n = 0
    for s in range(0, n, CHUNK):
        part = a[s:s + CHUNK].to_memory()
        X = part.X
        lab = labels[s:s + CHUNK]
        row_tot = np.asarray(X.sum(1)).ravel()
        for t in np.unique(lab):
            if t == CONTROL:
                idx = np.where(lab == t)[0]
                sub = X[idx]
                ctrl_sum += np.asarray(sub.sum(0)).ravel()
                ctrl_total += float(row_tot[idx].sum())
                ctrl_n += len(idx)
        for t in have:
            idx = np.where(lab == t)[0]
            if len(idx):
                sub = X[idx]
                sums[t] += np.asarray(sub.sum(0)).ravel()
                totals[t] += float(row_tot[idx].sum())
        if (s // CHUNK) % 10 == 0:
            print(f"chunk {s}/{n}", flush=True)
    a.file.close()

    ctrl_rate = ctrl_sum / ctrl_n
    p_ctrl = ctrl_sum / ctrl_total
    log_ctrl = np.log1p(TS * p_ctrl)

    rows = {t: np.where(filt, grows[gmap[t]], 0.0) for t in have}
    norms = {t: float(np.linalg.norm(rows[t][filt])) for t in have}
    print("GWPS natural filt-norm median", np.median(list(norms.values())), flush=True)

    ts = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
    ss = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.63, 0.8, 1.0, 1.25, 1.5, 2.0])
    # 预计算真实 log profile 与靶基因位置
    real_log, denom, tpos = {}, {}, {}
    for t in have:
        p = sums[t] / totals[t]
        real_log[t] = np.log1p(TS * p)
        d = log_ctrl - real_log[t]
        pos = gpos[t]
        d[pos] = 0.0
        denom[t] = float(np.dot(d, d))
        tpos[t] = pos
    D = sum(denom.values())
    print(f"denominator sum {D:.1f}, per-target median {np.median(list(denom.values())):.1f}",
          flush=True)

    print(f"{'T':>5s} {'s':>5s} {'num/D':>7s} {'R2':>7s} {'zeros':>6s}")
    best = None
    for th in ts:
        for s in ss:
            num = 0.0
            nz = 0
            for t in have:
                lfc = rows[t].copy()
                if th > 0:
                    lfc[np.abs(lfc) < th] = 0.0
                if s == 0:
                    p_pred = p_ctrl.copy()
                else:
                    rate = ctrl_rate * np.exp2(np.clip(s * lfc, -4, 4))
                    p_pred = rate / rate.sum()
                nz += int(np.count_nonzero(lfc))
                d = np.log1p(TS * p_pred) - real_log[t]
                pos = tpos[t]
                d[pos] = 0.0
                num += float(np.dot(d, d))
            r = num / D
            if best is None or r < best[0]:
                best = (r, th, s)
            print(f"{th:5.2f} {s:5.2f} {r:7.4f} {1 - r:7.4f} {nz:6d}", flush=True)
    print("best", best)


if __name__ == "__main__":
    main()
