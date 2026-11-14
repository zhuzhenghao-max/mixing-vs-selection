"""oran+GWPS v8 混合的微观精化检验 (train 代理, 只读).

在 n=123 (orion∩gwps train) 子集上对比:
  1. baseline v8: unit(Oavg) 0.4 + unit(G) 0.6
  2. orion 两系拆 unit 再平均 (line-split) 各权重
  3. orion FDR 掩码 (thr 0.05 / 0.01, either-line keep) 或 -log10 加权
  4. 非 rpe1 互补子集上的 O/G 权重扫描 (面板偏向非必需基因)
"""

import json
import sys

import numpy as np

sys.path.insert(0, "src")


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def main():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes = m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    n = len(train_names)
    gidx = {g: i for i, g in enumerate(genes)}

    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    o_genes = meta["genes"]
    pert_to_id = meta["pert_to_id"]
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    FD = np.load("data/external/orion_fdr.npy", mmap_mode="r")
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    rowmap = {}
    for r in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[r])], {})[int(pert_ids[r])] = r

    ocols, opos = [], []
    for oc, g in enumerate(o_genes):
        if g in gidx:
            ocols.append(oc)
            opos.append(gidx[g])
    ocols = np.array(ocols)
    opos = np.array(opos)

    def line_rows(t, cl):
        pid = pert_to_id.get(t)
        if pid is None:
            return None, None
        r = rowmap[cl].get(pid)
        if r is None:
            return None, None
        v = np.asarray(L[r, ocols], dtype=np.float64)
        f = np.asarray(FD[r, ocols], dtype=np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        f = np.nan_to_num(f, nan=1.0, posinf=1.0, neginf=1.0)
        if np.abs(v).sum() == 0:
            return None, None
        return v, f

    def full_axis(v):
        full = np.zeros(len(genes))
        full[opos] = v
        return full

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gmap = {str(nm): i for i, nm in enumerate(g["names"].astype(str))}
    grows = g["rows"].astype(np.float64)

    r1 = np.load("outputs/rpe1_rows.npz", allow_pickle=True)
    rmap = set(str(x) for x in r1["names"].astype(str))

    base = filt & np.all(np.isfinite(LFC), 0)
    pos = np.where(base)[0]
    ref = LFC[:, pos]
    mvec = ref.mean(0)
    refn = np.zeros_like(ref)
    for i in range(n):
        refn[i] = resid(ref[i], mvec)
    refn /= np.maximum(np.linalg.norm(refn, axis=1, keepdims=True), 1e-9)

    def eval_rows(name, rows, idx):
        P = np.stack([r[pos] for r in rows])
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
        print(f"  {name:34s} n={len(idx):3d} pds {np.median(raws):.4f} "
              f"spec med {np.median(cs):+.3f}", flush=True)

    cov = []
    for i, t in enumerate(train_names):
        vh, fh = line_rows(t, "hct116")
        vk, fk = line_rows(t, "hek293t")
        if (vh is None and vk is None) or t not in gmap:
            continue
        cov.append((i, t, vh, fh, vk, fk, grows[gmap[t]]))
    print(f"train targets with orion(nonzero)+gwps: {len(cov)}")

    idx = [c[0] for c in cov]
    Gv = [c[6] for c in cov]
    Ov = [full_axis(np.mean([v for v in (c[2], c[4]) if v is not None], 0)) for c in cov]

    eval_rows("base v8: unit Oavg .4 + G .6",
              [0.4 * unit(a) + 0.6 * unit(b) for a, b in zip(Ov, Gv)], idx)

    def line_split(c):
        vs = [unit(v) for v in (c[2], c[4]) if v is not None]
        return full_axis(np.mean(vs, 0))

    rows = []
    for c in cov:
        o = line_split(c)
        rows.append(0.4 * o + 0.6 * unit(c[6]))
    eval_rows("line-split (unit each line, avg)", rows, idx)

    # FDR masks on orion avg
    for thr in (0.05, 0.01):
        rows = []
        for c in cov:
            vs, fs = [], []
            for v, f in ((c[2], c[3]), (c[4], c[5])):
                if v is not None:
                    vs.append(v)
                    fs.append(f)
            keep = np.zeros_like(vs[0], dtype=bool)
            for f in fs:
                keep |= (f < thr)
            va = np.mean(vs, 0) * keep
            rows.append(0.4 * unit(full_axis(va)) + 0.6 * unit(c[6]))
        eval_rows(f"FDR<{thr} mask on O", rows, idx)

    def fdr_weighted(c):
        vs, fs = [], []
        for v, f in ((c[2], c[3]), (c[4], c[5])):
            if v is not None:
                vs.append(v)
                fs.append(f)
        wt = np.zeros_like(vs[0])
        for f in fs:
            wt = np.maximum(wt, np.clip(-np.log10(np.maximum(f, 1e-4)), 0, 4))
        return np.mean(vs, 0) * wt

    rows = []
    for c in cov:
        rows.append(0.4 * unit(full_axis(fdr_weighted(c))) + 0.6 * unit(c[6]))
    eval_rows("-log10(FDR) weight on O", rows, idx)
    print("weight sweep with FDR-weighted O:")
    for w in (0.3, 0.4, 0.5, 0.6):
        rows = [w * unit(full_axis(fdr_weighted(c))) + (1 - w) * unit(c[6])
                for c in cov]
        eval_rows(f"FW-O{w:.1f}+G{1-w:.1f}", rows, idx)

    # orion standalone: train targets with orion (any), FDR weight vs raw
    solo = []
    for i, t in enumerate(train_names):
        vh, fh = line_rows(t, "hct116")
        vk, fk = line_rows(t, "hek293t")
        if vh is None and vk is None:
            continue
        solo.append((i, t, vh, fh, vk, fk))
    print(f"orion-standalone train targets: {len(solo)}")
    sidx = [c[0] for c in solo]
    eval_rows("solo raw Oavg",
              [full_axis(np.mean([v for v in (c[2], c[4]) if v is not None], 0))
               for c in solo], sidx)
    eval_rows("solo -log10(FDR) Oavg",
              [full_axis(fdr_weighted(c)) for c in solo], sidx)

    # complement (non-rpe1) weight sweep
    idx2 = [k for k, c in enumerate(cov) if c[1] not in rmap]
    print(f"complement (not in rpe1): n={len(idx2)}")
    for w in (0.3, 0.4, 0.5, 0.6):
        rows = [w * unit(a) + (1 - w) * unit(b) for a, b in zip(Ov, Gv)]
        eval_rows(f"complement O{w:.1f}+G{1-w:.1f}",
                  [rows[k] for k in idx2], [cov[k][0] for k in idx2])


if __name__ == "__main__":
    main()
