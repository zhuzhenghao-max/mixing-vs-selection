"""Pilot on Tahoe-100M pseudobulk: condition overlap, baseline PDS, selection.

Loads per-line npz files built by build_line.py, keeps conditions shared by
all supplied lines, and for each held-out line compares equal-mix transfer
against kmeans/random calibration at k perturbations.
"""

import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV


def unit_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def load(path):
    d = np.load(path, allow_pickle=False)
    cond = [str(x) for x in d["cond"]]
    X = np.nan_to_num(d["X"].astype(np.float64), nan=0.0)
    padj = d["padj_frac"].astype(np.float64)
    return cond, X, padj


def pds(pred, Y, qidx):
    Yn = unit_rows(Y)
    Pn = unit_rows(pred)
    S = Yn @ Pn.T
    out = np.empty(len(qidx))
    for k, j in enumerate(qidx):
        col = S[:, k]
        sv = col[j]
        r = np.sum(col > sv) + 0.5 * (np.sum(col == sv) - 1)
        out[k] = 1.0 - r / len(Y)
    return out


def mix_feature(rows_by_line, srcs):
    return np.mean([unit_rows(rows_by_line[o]) for o in srcs], axis=0)


def kmeans_pick(X, k, rng):
    Xn = unit_rows(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xn)
    sel = []
    for c in km.cluster_centers_:
        d = np.linalg.norm(Xn - c, axis=1)
        d[sel] = np.inf
        sel.append(int(np.argmin(d)))
    if len(set(sel)) < k:
        rest = [i for i in range(len(X)) if i not in set(sel)]
        rng.shuffle(rest)
        sel += rest[:k - len(set(sel))]
    return np.array(sel[:k])


def main(paths):
    raw = {}
    for p in paths:
        name = p.split("/")[-1].replace(".npz", "")
        raw[name] = load(p)
        print(name, raw[name][1].shape)
    names = list(raw)
    common = set(raw[names[0]][0])
    for n in names[1:]:
        common &= set(raw[n][0])
    common = sorted(common)
    print("common conditions", len(common))
    data = {}
    for n in names:
        idx = {c: i for i, c in enumerate(raw[n][0])}
        rows = np.array([idx[c] for c in common])
        data[n] = dict(X=raw[n][1], padj=raw[n][2], rows=rows)

    rng = np.random.default_rng(0)
    n_eval = max(50, len(common) // 4)
    for L in names:
        srcs = [o for o in names if o != L]
        Y = data[L]["X"][data[L]["rows"]]
        qrows = np.arange(len(common))
        Q = mix_feature({o: data[o]["X"][data[o]["rows"]] for o in srcs},
                        srcs)
        base = pds(Q, Y, qrows)
        print(f"\n{L}: equal-mix PDS {np.nanmean(base):+.4f} "
              f"(sd {np.nanstd(base):.4f})")
        perm = rng.permutation(len(common))
        E, C = perm[:n_eval], perm[n_eval:]
        ncomp = min(512, len(C) - 1, Q.shape[1])
        pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0)
        ZC = pca.fit_transform(Q[C])
        ZE = pca.transform(Q[E])
        for k in (20, 50):
            if k >= len(C):
                continue
            for strat in ("kmeans", "random"):
                reps = 5 if strat == "random" else 1
                vals = []
                for _ in range(reps):
                    sel = (kmeans_pick(Q[C], k, rng) if strat == "kmeans"
                           else rng.choice(len(C), size=k, replace=False))
                    model = RidgeCV(alphas=np.logspace(-2, 4, 13),
                                    fit_intercept=False)
                    model.fit(ZC[sel], Y[qrows[C[sel]]])
                    P = model.predict(ZE)
                    vals.append(np.nanmean(pds(P, Y, qrows[E]) - base[E]))
                print(f"  k={k:3d} {strat:6s} dPDS {np.mean(vals):+.4f}")
        print(f"  equal-mix on eval {np.nanmean(base[E]):+.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
