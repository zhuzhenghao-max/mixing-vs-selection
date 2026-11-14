"""Diagnose transfer strength strata on the Tahoe external panel.

Reports per-query baseline PDS by cross-source agreement quartile, plus
calibration gains on an eval split stratified by agreement.
"""

import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "/root/autodl-tmp/tahoe")
from tahoe_external import (common_conditions, kmeans_pick, load_lines,
                            mix_rows, pds_fast, source_stats)


def main(paths):
    lines = load_lines(paths, "S")
    common = common_conditions(lines)
    names = list(lines)
    rows = {n: np.array([lines[n]["index"][c] for c in common]) for n in names}
    rng = np.random.default_rng(0)
    n = len(common)
    for L in names:
        srcs = [o for o in names if o != L]
        Q = mix_rows(lines, srcs, common)
        Y = lines[L]["X"][rows[L]]
        idx = np.arange(n)
        base = pds_fast(Q, Y, idx)
        neg = pds_fast(Q, Y, rng.permutation(n))
        agr = source_stats(lines, srcs, common)["agreement"]
        qs = np.quantile(agr, [0.25, 0.5, 0.75])
        strata = np.digitize(agr, qs)
        print(f"\n{L}: base PDS {base.mean():+.4f} "
              f"permuted {neg.mean():+.4f}")
        for s in range(4):
            m = strata == s
            print(f"  agreement q{s + 1} n={m.sum():3d} "
                  f"agr {agr[m].mean():.3f} PDS {base[m].mean():+.4f}")
        E = rng.permutation(n)[:n // 4]
        C = np.setdiff1d(idx, E)
        ncomp = min(512, len(C) - 1, Q.shape[1])
        pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0)
        ZC = pca.fit_transform(Q[C])
        ZE = pca.transform(Q[E])
        for k in (20, 50):
            for strat in ("kmeans", "random"):
                reps = 5 if strat == "random" else 1
                ds = []
                for _ in range(reps):
                    sel = (kmeans_pick(Q[C], k, rng) if strat == "kmeans"
                           else rng.choice(len(C), size=k, replace=False))
                    model = RidgeCV(alphas=np.logspace(-2, 4, 13),
                                    fit_intercept=False)
                    model.fit(ZC[sel], Y[C[sel]])
                    P = model.predict(ZE)
                    ds.append(pds_fast(P, Y, E) - base[E])
                d = np.mean(ds, axis=0)
                estr = strata[E]
                h = estr == 0
                print(f"  k={k:3d} {strat:6s} dPDS all {d.mean():+.4f} "
                      f"hard {d[h].mean():+.4f} easy {d[estr == 3].mean():+.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
