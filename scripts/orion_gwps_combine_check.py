"""orion + GWPS 合并行检验: 对 orion 覆盖的 train 靶, 对比
orion 跨系均值 / GWPS / 两者均值 的迁移 pds (同 harness), 判断 v8 (合并) 是否有增量.
只读, 不产出表.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "src")


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


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

    def orion_line(pid, cl):
        r = rowmap[cl].get(pid)
        if r is None:
            return None
        v = np.asarray(L[r, ocols], dtype=np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v

    def orion_avg(t):
        pid = pert_to_id.get(t)
        if pid is None:
            return None
        vs = [v for v in (orion_line(pid, cl) for cl in ("hct116", "hek293t"))
              if v is not None and np.abs(v).sum() > 0]
        if not vs:
            return None
        v = np.mean(vs, 0)
        full = np.zeros(len(genes))
        full[opos] = v
        return full

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"].astype(str)]
    grows = g["rows"].astype(np.float64)
    gmap = {nm: i for i, nm in enumerate(gnames)}

    base = filt & np.all(np.isfinite(LFC), 0)
    pos = np.where(base)[0]
    ref = LFC[:, pos]
    mvec = ref.mean(0)
    refn = np.zeros_like(ref)
    for i in range(n):
        refn[i] = resid(ref[i], mvec)
    refn /= np.maximum(np.linalg.norm(refn, axis=1, keepdims=True), 1e-9)

    cov = []
    for i, t in enumerate(train_names):
        ov = orion_avg(t)
        if ov is None:
            continue
        if t not in gmap:
            continue
        cov.append((i, t, ov, grows[gmap[t]]))
    print(f"orion∩gwps train targets: {len(cov)}")

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
        cs = np.array(cs)
        print(f"  {name:12s} n={len(idx):3d} pds {np.median(raws):.4f} "
              f"spec med {np.median(cs):+.3f}", flush=True)

    idx = [c[0] for c in cov]
    eval_rows("orion", [c[2] for c in cov], idx)
    eval_rows("gwps", [c[3] for c in cov], idx)
    eval_rows("mean2", [(c[2] + c[3]) / 2 for c in cov], idx)

    def unit(v):
        return v / max(np.linalg.norm(v), 1e-9)

    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        eval_rows(f"unit w={w}", [w * unit(c[2]) + (1 - w) * unit(c[3]) for c in cov], idx)


if __name__ == "__main__":
    main()
