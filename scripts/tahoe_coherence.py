"""Per-condition source-response coherence on the Tahoe-100M drug panel.

Reads the per-line matrices written by tahoe_build_line.py (data/tahoe/lines
by default; set TAHOE_ROOT or pass LINES_DIR).  For each held-out line and each active
condition shared by all seven lines, unit-normalize the six source response
vectors, summarize their agreement (mean and minimum pairwise cosine,
resultant length), compare the equal mix with the held-out truth, and score
PDS under full and same-drug-excluded pools.  Output is a small per-condition
CSV for the regime diagnostic.

Usage:
  python tahoe_coherence.py [LINES_DIR] [OUT_CSV]
"""

import glob
import os
import sys

import numpy as np
import pandas as pd


def unit_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def load_meta(path):
    d = np.load(path, allow_pickle=True)
    cond = [str(c) for c in d["cond"]]
    drug = np.asarray(d["drug"]).astype(str)
    return dict(name=os.path.basename(path)[:-4], cond=cond, drug=drug,
                index={c: i for i, c in enumerate(cond)})


def load_slice(path, idx):
    d = np.load(path, allow_pickle=True)
    X = np.nan_to_num(d["X"][idx].astype(np.float32), nan=0.0)
    return X


def pds_pool(pred, Y, keep):
    """PDS of each prediction row against pool Y, keep[i] selects rows."""
    Yn = unit_rows(Y)
    Pn = unit_rows(pred)
    S = Yn @ Pn.T
    out = np.empty(len(pred))
    for i in range(len(pred)):
        k = keep[i].copy()
        k[i] = True
        s = S[k, i]
        sv = Yn[i] @ Pn[i]
        rank = np.sum(s > sv) + 0.5 * (np.sum(s == sv) - 1)
        out[i] = 1.0 - rank / len(s)
    return out


def summarize(V):
    G = V @ V.T
    iu = np.triu_indices(len(V), 1)
    pair = G[iu]
    mix = V.mean(0)
    return float(pair.mean()), float(pair.min()), float(np.linalg.norm(mix)), \
        mix


def main():
    lines_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.environ.get("TAHOE_ROOT", "data/tahoe"), "lines")
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "outputs/tahoe_coherence.csv"
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
    idx = {n: np.array([meta[n]["index"][c] for c in common])
           for n in names}
    act = []
    for n in names:
        Xs = load_slice(path_of[n], idx[n])
        act.append((np.abs(Xs) > 2).mean(axis=1))
        del Xs
    act = np.mean(act, axis=0)
    common = [c for c, a in zip(common, act) if a >= 0.005]
    print("common active conditions", len(common), flush=True)
    idx = {n: np.array([meta[n]["index"][c] for c in common])
           for n in names}
    drugs = np.array([meta[names[0]]["drug"][i] for i in idx[names[0]]])
    drug_keep = np.array([drugs != d for d in drugs])
    rows = []
    for L in names:
        srcs = [o for o in names if o != L]
        Y = load_slice(path_of[L], idx[L]).astype(np.float64)
        src_mats = [load_slice(path_of[o], idx[o]).astype(np.float64)
                    for o in srcs]
        src_means = [M.mean(0) for M in src_mats]
        pred = np.zeros_like(src_mats[0])
        for M in src_mats:
            pred += unit_rows(M)
        pred /= len(src_mats)
        Yn = unit_rows(Y)
        GY = Yn @ Yn.T
        np.fill_diagonal(GY, 0.0)
        pool_cos = GY.mean(1)
        pfull = pds_pool(pred, Y, np.ones((len(common), len(common)), bool))
        pexc = pds_pool(pred, Y, drug_keep)
        S_other = Yn @ unit_rows(pred).T
        diag = np.diag(S_other).copy()
        other_cos = (S_other.sum(0) - diag) / (len(common) - 1)
        np.fill_diagonal(S_other, -np.inf)
        other_q95 = np.quantile(S_other, 0.95, axis=0)
        other_q99 = np.quantile(S_other, 0.99, axis=0)
        np.fill_diagonal(S_other, diag)
        for i, c in enumerate(common):
            raw = np.stack([M[i] for M in src_mats])
            cen = np.stack([M[i] - mu for M, mu in zip(src_mats, src_means)])
            vr = unit_rows(raw)
            vc = unit_rows(cen)
            pair, pmin, res, mix = summarize(vr)
            cpair, cmin, cres, cmix = summarize(vc)
            truth = Y[i]
            tn = np.linalg.norm(truth)
            mt = float(mix @ truth / (np.linalg.norm(mix) * tn + 1e-12))
            ct = truth - Y.mean(0)
            cmt = float(cmix @ ct /
                        (np.linalg.norm(cmix) * np.linalg.norm(ct) + 1e-12))
            rows.append(dict(
                holdout=L, cond=c, drug=drugs[i], pair_cos=pair,
                min_cos=pmin, resultant=res, cpair_cos=cpair, cmin_cos=cmin,
                cresultant=cres, mix_truth_cos=mt, cmix_truth_cos=cmt,
                pool_cos=float(pool_cos[i]), other_cos=float(other_cos[i]),
                other_q95=float(other_q95[i]), other_q99=float(other_q99[i]),
                pds_full=float(pfull[i]), pds_drug_excl=float(pexc[i])))
        sub = [r for r in rows if r["holdout"] == L]
        print(f"{L:10s} full {pfull.mean():.4f} excl {pexc.mean():.4f} "
              f"pair {np.mean([r['pair_cos'] for r in sub]):.4f}", flush=True)
        del Y, src_mats, src_means, pred
    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False, float_format="%.6f")
    print("saved", out_csv, d.shape, flush=True)
    print(d.groupby("holdout")[["pair_cos", "min_cos", "resultant",
                                 "mix_truth_cos", "pool_cos",
                                 "pds_full", "pds_drug_excl"]]
          .mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
