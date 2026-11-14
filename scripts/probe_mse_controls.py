"""正对照/符号诊断: 把 H1 Training 的 profile 缓存下来并测几种预测臂.

臂:
  oracle        p_pred = p_real (应给 ~0)
  cross         p_pred = 另一靶的真实 profile (H1 内部跨靶可迁移性)
  gwps_sX       GWPS 行, X 含负数 (符号诊断)
输出 outputs/proxy_profiles.npz 供后续快速扫描.
"""

import sys

import anndata as ad
import numpy as np

sys.path.insert(0, "src")
from linear_baseline import CONTROL  # noqa: E402

TS = 50000.0
PATH = "data/adata_Training_2025.h5ad"
CHUNK = 4000
OUT = "outputs/proxy_profiles.npz"


def main():
    a = ad.read_h5ad(PATH, backed="r")
    var = a.var_names.to_numpy().astype(str)
    gpos = {g: i for i, g in enumerate(var)}
    labels = a.obs["target_gene"].astype(str).to_numpy()
    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"]]
    grows = g["rows"].astype(np.float64)
    gmap = {nm: i for i, nm in enumerate(gnames)}
    have = [t for t in sorted(set(labels) - {CONTROL}) if t in gmap]
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
            idx = np.where(lab == t)[0]
            sub = X[idx]
            acc = np.asarray(sub.sum(0)).ravel()
            if t == CONTROL:
                ctrl_sum += acc
                ctrl_total += float(row_tot[idx].sum())
                ctrl_n += len(idx)
            elif t in sums:
                sums[t] += acc
                totals[t] += float(row_tot[idx].sum())
        if (s // CHUNK) % 20 == 0:
            print(f"chunk {s}/{n}", flush=True)
    a.file.close()

    p_ctrl = ctrl_sum / ctrl_total
    pref = np.log1p(TS * p_ctrl)
    P = np.stack([sums[t] / totals[t] for t in have])
    lreal = np.log1p(TS * P)
    tpos = np.array([gpos[t] for t in have])
    denom = np.empty(len(have))
    for i in range(len(have)):
        d = pref - lreal[i]
        d[tpos[i]] = 0.0
        denom[i] = np.dot(d, d)
    filt = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)["filt"].astype(bool)
    rows = np.stack([np.where(filt, grows[gmap[t]], 0.0) for t in have])
    np.savez(OUT, names=np.array(have), genes=var, ctrl_sum=ctrl_sum, ctrl_n=ctrl_n,
             ctrl_total=ctrl_total, sums=np.stack([sums[t] for t in have]),
             totals=np.array([totals[t] for t in have]), gwps=rows, tpos=tpos,
             denom=denom)

    def ratio_for(pp):
        num = 0.0
        for i in range(len(have)):
            lpred = np.log1p(TS * pp[i])
            d = lpred - lreal[i]
            d[tpos[i]] = 0.0
            num += float(np.dot(d, d))
        return num / denom.sum()

    print("oracle ratio (self profile):", ratio_for(P), flush=True)
    cross = np.roll(P, 1, axis=0)
    print("cross-target ratio:", ratio_for(cross), flush=True)

    ctrl_rate = ctrl_sum / ctrl_n
    for s in (-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2):
        pp = np.empty_like(P)
        for i in range(len(have)):
            rate = ctrl_rate * np.exp2(np.clip(s * rows[i], -4, 4))
            pp[i] = rate / rate.sum()
        print(f"gwps s={s:+.1f} ratio {ratio_for(pp):.4f}", flush=True)


if __name__ == "__main__":
    main()
