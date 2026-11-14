"""ESM 嵌入 -> lfc 岭回归 LOO, 与 fit_lfc_map.py 完全同口径 (同 folds/同 filt/同 fit_ridge).

对照: F454 基线 CV corr 中位 0.322 (lam100). 判定: ESM 或其拼接须显著超过才值得下游.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from linear_baseline import fit_ridge  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--lams-f", default="1,10,100")
    p.add_argument("--lams-e", default="0.1,1,10,100")
    p.add_argument("--lams-c", default="1,10,100,1000")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    m = np.load("outputs/scvi_map_cache.npz")
    F_all, genes = m["F"], m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    posg = {g: i for i, g in enumerate(genes)}
    F_tr = np.stack([F_all[posg[g]] for g in train_names])

    emb = np.load(args.emb)
    enames, E = emb["names"].astype(str), emb["E"]
    epos = {g: i for i, g in enumerate(enames)}
    missing = [g for g in train_names if g not in epos]
    if missing:
        print(f"WARNING missing emb for {len(missing)}: {missing[:10]}")
    E_tr = np.stack([E[epos[g]] if g in epos else np.zeros(E.shape[1], np.float32)
                     for g in train_names])

    n = len(train_names)
    rng = np.random.default_rng(args.seed)
    folds = rng.permutation(n) % args.folds

    def cv(Feat, lams, tag):
        for lam in lams:
            corrs = []
            for k in range(args.folds):
                te = folds == k
                tr = ~te
                W, b = fit_ridge(Feat[tr], LFC[tr], lam)
                pred = Feat[te] @ W + b
                yt = LFC[te]
                for j in range(pred.shape[0]):
                    f = filt & np.isfinite(pred[j]) & np.isfinite(yt[j])
                    x, y = pred[j][f], yt[j][f]
                    if np.std(y) > 1e-9 and np.std(x) > 1e-9:
                        corrs.append(np.corrcoef(x, y)[0, 1])
            print(f"{tag} lam {lam:g}: median {np.median(corrs):.3f} mean {np.mean(corrs):.3f}",
                  flush=True)

    print(f"n={n}, F {F_tr.shape}, E {E_tr.shape}")
    cv(F_tr, [float(x) for x in args.lams_f.split(",")], "F454 ")
    cv(E_tr, [float(x) for x in args.lams_e.split(",")], "ESM  ")
    Fc = np.hstack([F_tr, E_tr])
    cv(Fc, [float(x) for x in args.lams_c.split(",")], "F+ESM")


if __name__ == "__main__":
    main()
