"""Full-source calibration on Nadig Tier A without self-coordinate exclusion.

The Tahoe drug panel has no target coordinate, so the regime contrast needs
a Nadig curve under the same policy. This mirrors the full-mix portion of
phase2_source_budget.py with plain cosine scoring.
"""

import csv
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "scripts")
from bench_common import load_axis, load_nadig_lines
from bench_supervised import TIERS, build_features, build_targets
from phase1_selection_pilot import _unit
from phase2_source_budget import kmeans_sel


class PlainRetrieval:
    def __init__(self, ref_rows, ref_names, pos):
        self.ridx = {t: i for i, t in enumerate(ref_names)}
        R = ref_rows[:, pos].astype(np.float32)
        self.Rn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True),
                                 1e-9)
        self.n = len(ref_names)

    def score_pos(self, qpos, qnames):
        Q = np.asarray(qpos, dtype=np.float32)
        Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
        S = self.Rn @ Qn.T
        out = np.full(len(qnames), np.nan, dtype=np.float64)
        for k, t in enumerate(qnames):
            ri = self.ridx.get(t)
            if ri is None:
                continue
            col = S[:, k]
            sv = col[ri]
            rank0 = np.sum(col > sv) + 0.5 * (np.sum(col == sv) - 1)
            out[k] = 1.0 - rank0 / self.n
        return out


def main():
    ks = [20, 50, 100, 200]
    splits_n = 3
    n_eval = 150
    repeats = 3
    genes, pos, gidx, m = load_axis()
    lines = load_nadig_lines(genes, gidx)
    lines.pop("_gidx", None)
    holdouts = list(TIERS["A"][0])
    rows = []
    for L in holdouts:
        ref = lines[L]
        srcs = [o for o in holdouts if o != L and o in lines]
        q = [t for t in ref["names"]
             if all(t in lines[o]["index"] for o in srcs)]
        print(f"\n=== {L} | sources {srcs} | covered {len(q)} ===", flush=True)
        Y = build_targets(ref, q, pos)
        R = PlainRetrieval(ref["rows"], ref["names"], pos)
        X = build_features(lines, srcs, q, pos, "mix")
        eq = R.score_pos(X, q)
        print(f"  equal-mix PDS {np.nanmean(eq):.4f}", flush=True)
        rng = np.random.default_rng(0)
        for split in range(splits_n):
            perm = rng.permutation(len(q))
            E = perm[:min(n_eval, max(20, len(q) // 3))]
            C = perm[min(n_eval, max(20, len(q) // 3)):]
            ncomp = min(1024, len(C) - 1, X.shape[1])
            pca = PCA(n_components=ncomp, svd_solver="randomized",
                      random_state=0).fit(X[C])
            ZC = pca.transform(X[C])
            ZE = pca.transform(X[E])
            Xn = _unit(X[C])
            for k in ks:
                for strat in ("kmeans", "random"):
                    reps = repeats if strat == "random" else 1
                    ds = []
                    for _ in range(reps):
                        sel = (kmeans_sel(Xn, k, rng) if strat == "kmeans"
                               else rng.choice(len(Xn), size=k,
                                               replace=False))
                        model = RidgeCV(
                            alphas=np.logspace(-2, 4, 13),
                            fit_intercept=False)
                        model.fit(ZC[sel], Y[C][sel])
                        P = model.predict(ZE)
                        ds.append(np.nanmean(
                            R.score_pos(P, [q[i] for i in E]) - eq[E]))
                    d = float(np.mean(ds))
                    rows.append(dict(holdout=L, k=k, strategy=strat,
                                     split=split, delta=d))
                    print(f"  {L} k={k:3d} {strat:6s} split{split} "
                          f"{d:+.4f}", flush=True)
    path = "outputs/phase2_fullmix_noexcl.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["holdout", "k", "strategy",
                                          "split", "delta"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
