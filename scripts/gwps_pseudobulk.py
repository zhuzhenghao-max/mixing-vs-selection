"""GWPS (Replogle 2022 K562 genome-wide) h5ad -> our-axis log2FC rows.

估计量与 raw17 一致: 每细胞 CP1M 归一后按扰动取均值 m,
raw = log2((m + EPS) / (base + EPS)), base = 对照细胞 mean_cpm;
base CPM < 1 的基因置 0 (无信息屏蔽); NaN->0. 不做 clip (按 v6 惯例 clip 在 build 里
对缩放后的行做).
流式: backed 模式按块读 X, 稀疏 one-hot 矩阵乘累加.
输出 outputs/gwps_rows.npz {names, rows (n x 18080) float32 natural, n_cells}.

GWPS_PROBE=1 只打印 obs/var 结构后退出.
"""

import os

import numpy as np

EPS = 1e-9  # 同 linear_baseline.EPS (内联以支持独立部署)
H5 = "data/external/gwps/replogle_2022_k562_gwps.h5ad"
CONTROL_NAMES = {"control", "non-targeting", "non_targeting", "nt", "nontargeting",
                 "intergenic", "ctrl", "control_non-targeting"}
LABEL_COLS = ["perturbation", "gene_target", "target_gene", "gene", "target", "condition"]
CHUNK = 25000


def find_label_col(obs):
    for c in LABEL_COLS:
        if c in obs.columns:
            return c
    raise SystemExit(f"no label column found in {list(obs.columns)}")


def main():
    import scipy.sparse as sp
    import anndata as ad

    our = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes_all = our["genes"].astype(str)
    gpos = {g: i for i, g in enumerate(genes_all)}

    a = ad.read_h5ad(H5, backed="r")
    print(f"shape {a.shape}, X type {type(a.X).__name__}")
    print(f"layers: {list(a.layers.keys())}, raw: {a.raw is not None}")
    print(f"obs cols: {list(a.obs.columns)}")
    Xs = a.X[:200]
    if sp.issparse(Xs):
        vals = Xs.data
    else:
        vals = np.asarray(Xs).ravel()
    frac_int = float(np.mean(vals == np.floor(vals)))
    print(f"X sample: dtype {vals.dtype} min {vals.min():.3f} max {vals.max():.3f} "
          f"int-frac {frac_int:.3f} mean {vals.mean():.3f}")
    lab_col = find_label_col(a.obs)
    labels = a.obs[lab_col].astype(str).to_numpy()
    print(f"label col '{lab_col}', nunique {len(np.unique(labels))}")
    vc = a.obs[lab_col].astype(str).value_counts()
    print("top labels:", vc.head(5).to_dict())
    ctrl_vals = sorted({l for l in np.unique(labels)
                        if l.lower().replace(" ", "") in CONTROL_NAMES or "ontrol" in l})
    print(f"control-like labels: {ctrl_vals}")

    vn = a.var_names.to_numpy().astype(str)
    print(f"var sample: {vn[:5]}, var cols: {list(a.var.columns)}")
    ens_like = vn[0].startswith("ENS")
    if ens_like:
        symcol = next((c for c in ["gene_symbol", "symbol", "gene_name", "gene_symbols"]
                       if c in a.var.columns), None)
        print(f"ensembl var_names; symbol col -> {symcol}")
        vn = a.var[symcol].astype(str).to_numpy() if symcol else vn
    mapped = np.array([gpos.get(g, -1) for g in vn])
    print(f"var genes mapped to our axis: {(mapped >= 0).sum()}/{len(vn)}")

    if os.environ.get("GWPS_PROBE"):
        print("PROBE done")
        return

    keep_cols = np.where(mapped >= 0)[0]
    tgt_pos = mapped[keep_cols]
    ng = len(genes_all)
    print(f"using {len(keep_cols)} gene cols")

    uniq, codes = np.unique(labels, return_inverse=True)
    nU = len(uniq)
    uidx = {u: i for i, u in enumerate(uniq)}
    ctrl_codes = [uidx[c] for c in ctrl_vals if c in uidx]
    if not ctrl_codes:
        raise SystemExit("no control cells found")
    print(f"uniq labels {nU}, control codes {ctrl_codes} "
          f"({sum(int((codes == c).sum()) for c in ctrl_codes)} cells)")

    counts = np.zeros(nU, dtype=np.int64)
    ngk = len(keep_cols)
    Ssum = np.zeros((nU, ngk), dtype=np.float64)
    n = a.n_obs
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        X = a.X[i0:i1]
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        tot = X.sum(1)
        tot = np.where(tot > 0, tot, 1.0)
        X *= (1e6 / tot)[:, None]
        Xk = X[:, keep_cols]
        del X
        cc = codes[i0:i1]
        L = len(cc)
        G = sp.csr_matrix((np.ones(L, dtype=np.float32), (cc, np.arange(L))), shape=(nU, L))
        Ssum += np.asarray(G @ Xk, dtype=np.float64)
        counts += np.bincount(cc, minlength=nU)
        print(f"  chunk {i0}-{i1} / {n}", flush=True)

    base = np.zeros(ng, dtype=np.float64)
    ctot = sum(counts[c] for c in ctrl_codes)
    for c in ctrl_codes:
        base[tgt_pos] += Ssum[c] / max(ctot, 1)
    n_base_lt1 = int((base < 1.0).sum())
    print(f"base CPM<1 genes: {n_base_lt1}/{ng}")

    names, rows, nc = [], [], []
    for i, u in enumerate(uniq):
        if i in ctrl_codes:
            continue
        if counts[i] == 0:
            continue
        vec = np.zeros(ng, dtype=np.float64)
        vec[tgt_pos] = Ssum[i] / counts[i]
        raw = np.log2((vec + EPS) / (base + EPS))
        raw[base < 1.0] = 0.0
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        names.append(u)
        rows.append(raw.astype(np.float32))
        nc.append(int(counts[i]))
    rows = np.stack(rows)
    nc = np.array(nc)
    print(f"saved table {rows.shape}, cells/p med {np.median(nc):.0f}, min {nc.min()}")
    our_filt = our["filt"]
    nats = np.linalg.norm(rows[:, our_filt], axis=1)
    print(f"natural filt-norm: med {np.median(nats):.2f} p5 {np.percentile(nats, 5):.2f}")
    np.savez("outputs/gwps_rows.npz", names=np.array(names), rows=rows, n_cells=nc)
    print("saved outputs/gwps_rows.npz")


if __name__ == "__main__":
    main()
