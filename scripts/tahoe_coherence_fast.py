"""Vectorized float32 equivalent of tahoe_coherence.py.

Produces the same per-condition columns (raw and centered coherence, pool
summaries, full and drug-excluded PDS) but avoids the float64 per-condition
loop, which is prohibitively slow on a single CPU core.

Usage:
  python tahoe_coherence_fast.py LINES_DIR OUT_CSV
"""

import glob
import os
import sys

import numpy as np
import pandas as pd


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)


def load_meta(path):
    d = np.load(path, allow_pickle=True)
    cond = [str(c) for c in d["cond"]]
    return dict(name=os.path.basename(path)[:-4], cond=cond,
                drug=np.asarray(d["drug"]).astype(str),
                index={c: i for i, c in enumerate(cond)})


def load_slice(path, idx):
    d = np.load(path, allow_pickle=True)
    return np.nan_to_num(d["X"][idx].astype(np.float32), nan=0.0)


def pds_rows(S, drug_keep):
    """PDS of each row of S against its column pool; S rows=battery."""
    n = S.shape[0]
    st = np.diag(S).copy()
    full = 1.0 - ((S > st[None, :]).sum(0) + 0.5 *
                  ((S == st[None, :]).sum(0) - 1)) / n
    exc = np.empty(n)
    for i in range(n):
        k = drug_keep[i].copy()
        k[i] = True
        s = S[k, i]
        sv = st[i]
        rank = (s > sv).sum() + 0.5 * ((s == sv).sum() - 1)
        exc[i] = 1.0 - rank / len(s)
    return full, exc


def main():
    lines_dir = sys.argv[1]
    out_csv = sys.argv[2]
    paths = sorted(glob.glob(os.path.join(lines_dir, "*.npz")))
    meta = {}
    for p in paths:
        m = load_meta(p)
        meta[m["name"]] = m
        print(m["name"], len(m["cond"]), "conditions", flush=True)
    names = list(meta)
    path_of = {os.path.basename(p)[:-4]: p for p in paths}
    common = set(meta[names[0]]["cond"])
    for n in names[1:]:
        common &= set(meta[n]["index"])
    common = sorted(common)
    idx = {n: np.array([meta[n]["index"][c] for c in common]) for n in names}
    act = []
    for n in names:
        Xs = load_slice(path_of[n], idx[n])
        act.append((np.abs(Xs) > 2).mean(axis=1))
        del Xs
    act = np.mean(act, axis=0)
    common = [c for c, a in zip(common, act) if a >= 0.005]
    print("common active conditions", len(common), flush=True)
    idx = {n: np.array([meta[n]["index"][c] for c in common]) for n in names}
    drugs = np.asarray([meta[names[0]]["drug"][i] for i in idx[names[0]]])
    drug_keep = np.asarray([drugs != d for d in drugs])
    nC = len(common)

    rows = []
    for L in names:
        srcs = [o for o in names if o != L]
        Y = load_slice(path_of[L], idx[L])
        V = np.empty((len(srcs), nC, Y.shape[1]), dtype=np.float32)
        for j, o in enumerate(srcs):
            M = load_slice(path_of[o], idx[o])
            V[j] = M
            del M
        mu = V.mean(axis=1)
        nr = np.linalg.norm(V, axis=-1)
        Rr = np.einsum("ild,jld->lij", V, V)
        G = Rr / np.maximum(nr.T[:, :, None] * nr.T[:, None, :], 1e-9)
        A = np.einsum("ild,jd->lij", V, mu)
        M2 = mu @ mu.T
        Gc = Rr - A - A.transpose(0, 2, 1) + M2[None, :, :]
        iu = np.triu_indices(len(srcs), 1)
        Gr = G[:, iu[0], iu[1]]
        nc = np.sqrt(np.maximum(np.diagonal(Gc, axis1=1, axis2=2), 1e-12))
        Gcc = Gc / (nc[:, :, None] * nc[:, None, :])
        Gcc = Gcc[:, iu[0], iu[1]]
        pair, pmin = Gr.mean(1), Gr.min(1)
        cpair, cmin = Gcc.mean(1), Gcc.min(1)

        mix = (V / np.maximum(nr, 1e-9)[:, :, None]).mean(axis=0)
        nrm = np.linalg.norm(mix, axis=1)
        mixc = np.zeros_like(mix)
        for i in range(len(srcs)):
            mixc += (V[i] - mu[i]) / np.maximum(nc[:, i], 1e-9)[:, None]
        mixc /= len(srcs)
        nrms = np.linalg.norm(mixc, axis=1)
        Yn = unit(Y)
        mt = np.sum(Yn * unit(mix), axis=1)
        ct = Y - Y.mean(axis=0)
        cmt = np.sum(unit(ct) * unit(mixc), axis=1)

        GY = Yn @ Yn.T
        np.fill_diagonal(GY, 0.0)
        pool_cos = GY.mean(1)
        S = Yn @ unit(mix).T
        diag = np.diag(S).copy()
        other_cos = (S.sum(0) - diag) / (nC - 1)
        np.fill_diagonal(S, -np.inf)
        q95 = np.quantile(S, 0.95, axis=0)
        q99 = np.quantile(S, 0.99, axis=0)
        np.fill_diagonal(S, diag)
        pfull, pexc = pds_rows(S, drug_keep)

        for i, c in enumerate(common):
            rows.append(dict(
                holdout=L, cond=c, drug=drugs[i],
                pair_cos=float(pair[i]), min_cos=float(pmin[i]),
                resultant=float(nrm[i]), cpair_cos=float(cpair[i]),
                cmin_cos=float(cmin[i]), cresultant=float(nrms[i]),
                mix_truth_cos=float(mt[i]), cmix_truth_cos=float(cmt[i]),
                pool_cos=float(pool_cos[i]), other_cos=float(other_cos[i]),
                other_q95=float(q95[i]), other_q99=float(q99[i]),
                pds_full=float(pfull[i]), pds_drug_excl=float(pexc[i])))
        print(f"{L:10s} full {pfull.mean():.4f} excl {pexc.mean():.4f} "
              f"pair {pair.mean():.4f}", flush=True)
        del V, Y, Yn, G, Gc, A, S, GY

    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False, float_format="%.6f")
    print("saved", out_csv, d.shape, flush=True)


if __name__ == "__main__":
    main()
