"""多系岭回归对"完全无测量靶"的检索力 (U9 行替代判据).

对每个留出系 L0: 用其余系全部行训练 per-gene 岭回归 (scGPT emb -> z 行),
在 L0 的 (a) 有源测量靶 (≥2 src) 与 (b) 无任何源测量靶 上分别评估检索 pds.
(b) 对应官方 11 个 U9 行 (orion/gwps 均无测量的靶).
"""

import json
import sys

sys.path.insert(0, "src")

import numpy as np  # noqa: E402

LINES = ["k562", "hct116", "hek293t"]
LAM = 100.0
SEED = 0


def unit_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def pds_of(Pn, refn):
    S = Pn @ refn.T
    n = refn.shape[0]
    raws = []
    for k in range(n):
        s = S[k]
        rank = 1.0 + np.sum(s > s[k]) + 0.5 * (np.sum(s == s[k]) - 1)
        raws.append(1.0 - (rank - 1.0) / (n - 1.0))
    return np.median(raws)


def load_rows():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes = m["genes"].astype(str)
    base = m["filt"] & np.all(np.isfinite(m["LFC"]), 0)
    pos = np.where(base)[0]
    rows = {}
    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    rows["k562"] = {nm: r for nm, r in zip(
        [str(x) for x in g["names"].astype(str)], g["rows"].astype(np.float64))}
    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    o_genes = meta["genes"]
    gidx = {g_: i for i, g_ in enumerate(genes)}
    ocols, opos = [], []
    for oc, gg in enumerate(o_genes):
        if gg in gidx:
            ocols.append(oc)
            opos.append(gidx[gg])
    ocols = np.array(ocols)
    opos = np.array(opos)
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    rowmap = {}
    for r in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[r])], {})[int(pert_ids[r])] = r
    for cl in ("hct116", "hek293t"):
        d = {}
        for t, pid in meta["pert_to_id"].items():
            r = rowmap[cl].get(pid)
            if r is None:
                continue
            v = np.nan_to_num(np.asarray(L[r, ocols], dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
            if np.abs(v).sum() == 0:
                continue
            full = np.zeros(len(genes))
            full[opos] = v
            d[t] = full
        rows[cl] = d
    return rows, pos


def main():
    np.random.seed(SEED)
    rows, pos = load_rows()
    emb = np.load("outputs/scgpt_emb_all.npz", allow_pickle=True)
    epos = {str(n): i for i, n in enumerate(emb["names"].astype(str))}
    E = emb["E"].astype(np.float32)

    for L0 in LINES:
        src = [x for x in LINES if x != L0]
        t0 = [t for t in rows[L0] if t in epos]
        src_sets = [set(rows[o].keys()) for o in src]
        measured = [t for t in t0 if all(t in s for s in src_sets)]
        unmeasured = [t for t in t0 if not any(t in s for s in src_sets)]
        print(f"\n=== hold-out {L0}: measured {len(measured)}, "
              f"unmeasured {len(unmeasured)} ===")

        Ttr = np.concatenate([np.stack([rows[o][t][pos] for t in rows[o] if t in epos])
                              for o in src]).astype(np.float32)
        emb_idx = np.concatenate([[epos[t] for t in rows[o] if t in epos] for o in src])
        Etr = E[emb_idx]
        mu_g = Ttr.mean(0)
        sd_g = Ttr.std(0)
        sd_g[sd_g < 1e-6] = 1.0
        Ztr = ((Ttr - mu_g) / sd_g).astype(np.float32)
        Xm, Xs = Etr.mean(0), Etr.std(0) + 1e-6
        Xstd = (Etr - Xm) / Xs
        A = Xstd.T @ Xstd + LAM * np.eye(Xstd.shape[1])
        W = np.linalg.solve(A, Xstd.T @ Ztr)

        for tag, tlist in (("measured", measured), ("unmeasured", unmeasured)):
            if not tlist:
                continue
            ev = E[np.array([epos[t] for t in tlist])]
            pred = ((ev - Xm) / Xs) @ W
            ref = np.stack([rows[L0][t][pos] for t in tlist])
            refz = (ref - mu_g) / sd_g
            r = refz - refz.mean(0)
            pds = pds_of(unit_rows(pred), unit_rows(r))
            print(f"  {tag:10s} n={len(tlist):5d} ridge_z pds {pds:.4f}", flush=True)


if __name__ == "__main__":
    main()
