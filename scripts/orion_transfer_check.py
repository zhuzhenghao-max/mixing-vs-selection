"""Orion (X-Atlas HCT116/HEK293T) → H1 迁移检验.

与 replogle_transfer_check 同 harness (raw-pds + spec-corr vs H1-285 残差参考).
用 lfc_true.npy (mmap) 取 train 靶的 log2FC 行, NaN->0, 限定到 filt∩finite 共享基因.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def main():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes_all = m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    n = len(train_names)

    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    o_genes = meta["genes"]
    syn = {name: i for i, name in enumerate(o_genes)}  # orion gene col index
    gidx = {g: i for i, g in enumerate(genes_all)}    # our col index

    ocols, opos = [], []
    for oc, g in enumerate(o_genes):
        if g in gidx:
            ocols.append(oc)
            opos.append(gidx[g])
    ocols = np.array(ocols)
    opos = np.array(opos)
    keep = filt[opos] & np.all(np.isfinite(LFC[:, opos]), 0)
    ocols, opos = ocols[keep], opos[keep]
    print(f"shared genes on filt∩finite axis: {len(opos)}")

    pert_to_id = meta["pert_to_id"]
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    print("lfc_true mmap:", L.shape, L.dtype)
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    lines = sorted(meta["context_to_id"], key=meta["context_to_id"].get)

    # rows per (line, target)
    rowmap = {cl: {} for cl in lines}
    for r in range(L.shape[0]):
        g = None
        pid = int(pert_ids[r])
        cid = int(ctx_ids[r])
        rowmap[ctx_name[cid]][pid] = r

    def vec(pid, cl):
        r = rowmap[cl].get(pid)
        if r is None:
            return None
        v = np.asarray(L[r, ocols], dtype=np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v

    base = np.all(np.isfinite(LFC[:, opos]), 0)
    pos = opos[base]
    ref = LFC[:, pos]
    mvec = ref.mean(0)
    refn = np.zeros_like(ref)
    for i in range(n):
        refn[i] = resid(ref[i], mvec)
    refn/= np.maximum(np.linalg.norm(refn, axis=1, keepdims=True), 1e-9)

    def eval_variant(name, rows_by_target):
        idx = [i for i, t in enumerate(train_names) if t in rows_by_target]
        if len(idx) < 10:
            print(f"  {name:22s}: only {len(idx)} targets, skip")
            return
        P = np.stack([rows_by_target[train_names[i]] for i in idx])
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
        print(f"  {name:22s} n={len(idx):3d} pds {np.median(raws):.4f} "
              f"spec med {np.median(cs):+.3f} mean {cs.mean():+.3f} "
              f"| xferpaircos {np.median(off):+.3f}", flush=True)

    for cl in lines:
        rows = {}
        for t in set(train_names):
            pid = pert_to_id.get(t)
            if pid is None:
                continue
            v = vec(pid, cl)
            if v is not None and np.abs(v).sum() > 0:
                rows[t] = v[base]
        eval_variant(f"orion-{cl}", rows)

    both = {}
    for t in set(train_names):
        pid = pert_to_id.get(t)
        if pid is None:
            continue
        vs = [vec(pid, cl) for cl in lines]
        vs = [v for v in vs if v is not None and np.abs(v).sum() > 0]
        if vs:
            both[t] = np.mean(vs, 0)[base]
    eval_variant("orion-avg", both)


if __name__ == "__main__":
    main()
