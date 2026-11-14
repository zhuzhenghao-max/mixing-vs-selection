"""GWPS (Replogle 2022 K562 genome-wide) → H1 迁移检验.

与 orion_transfer_check 同 harness (raw-pds 检索 + spec-corr vs H1-285 残差参考).
数据 = scripts/gwps_pseudobulk.py 产出的 outputs/gwps_rows.npz (our 18080-axis rows).
附加: 按细胞数分层 (n_cells>=50) 观察噪声; 报告对 145 未覆盖 panel 行的覆盖率.
"""

import sys

import numpy as np

sys.path.insert(0, "src")


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def main():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes_all = m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    n = len(train_names)

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    names = g["names"].astype(str)
    rows = g["rows"].astype(np.float64)
    ncells = g["n_cells"].astype(int)
    print(f"gwps table: {len(names)} perts, cells/p 5% {np.percentile(ncells, 5):.0f} "
          f"med {np.median(ncells):.0f} max {ncells.max()}")
    rmap = {nm: i for i, nm in enumerate(names)}

    base = filt & np.all(np.isfinite(LFC), 0)
    pos = np.where(base)[0]
    ref = LFC[:, pos]
    mvec = ref.mean(0)
    refn = np.zeros_like(ref)
    for i in range(n):
        refn[i] = resid(ref[i], mvec)
    refn /= np.maximum(np.linalg.norm(refn, axis=1, keepdims=True), 1e-9)

    # panel coverage diagnosis (vs v6 standing table)
    v5 = np.load("outputs/lfc_override_2026_v5.npz", allow_pickle=True)
    v6 = np.load("outputs/lfc_override_2026_v6.npz", allow_pickle=True)
    v3 = np.load("outputs/lfc_override_2026_v3.npz", allow_pickle=True)
    panel = [str(x) for x in v6["names"].astype(str)]
    injected = set(str(x) for x in v3["names"].astype(str))
    changed6 = set(panel[i] for i in range(len(panel))
                   if np.linalg.norm(v6["lfc"][i] - v5["lfc"][i]) > 1e-9)
    uncovered = [t for t in panel if t not in injected and t not in changed6]
    cov = [t for t in uncovered if t in rmap]
    print(f"v6 changed rows {len(changed6)} (expect 130), uncovered {len(uncovered)}, "
          f"gwps covers {len(cov)}/{len(uncovered)}")
    miss = [t for t in uncovered if t not in rmap]
    print(f"still missing: {len(miss)} {miss[:15]}")

    def eval_variant(name, sel):
        idx = [i for i, t in enumerate(train_names) if t in rmap and sel(rmap[t])]
        if len(idx) < 10:
            print(f"  {name:16s}: only {len(idx)} targets, skip")
            return
        P = np.stack([rows[rmap[train_names[i]]][pos] for i in idx])
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        S = Pn @ refn.T
        raws, cs = [], []
        for k, i in enumerate(idx):
            s = S[k]
            rank = 1.0 + np.sum(s > s[i]) + 0.5 * (np.sum(s == s[i]) - 1)
            raws.append(1.0 - (rank - 1.0) / (n - 1.0))
            a, b = P[k], refn[i]
            if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                cs.append(np.corrcoef(a, b)[0, 1])
        cs = np.array(cs)
        C = Pn @ Pn.T
        off = C[np.triu_indices(len(idx), 1)]
        print(f"  {name:16s} n={len(idx):3d} pds {np.median(raws):.4f} "
              f"spec med {np.median(cs):+.3f} mean {cs.mean():+.3f} "
              f"| xferpaircos {np.median(off):+.3f}", flush=True)

    eval_variant("gwps-all", lambda j: True)
    eval_variant("gwps-nc>=50", lambda j: ncells[j] >= 50)
    eval_variant("gwps-nc>=100", lambda j: ncells[j] >= 100)

    # same but excluding the 25 injected (pure zero-shot targets, like official val)
    inj = injected
    eval_variant("all-noinj", lambda j: names[j] not in inj)


if __name__ == "__main__":
    main()
