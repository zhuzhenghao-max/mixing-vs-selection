"""Check whether the Tahoe PDS ceiling survives removing same-drug doses.

The condition pool contains several concentrations per drug, so a query's
nearest reference row can be another dose of the same drug. Recompute PDS
with all other concentrations of the query's drug removed from the pool
(the true row stays). If transfer is still near ceiling, the saturation is
not an artifact of dose duplicates.
"""

import sys

import numpy as np

PATHS = ["lines/A549.npz", "lines/A498.npz", "lines/HS-578T.npz",
         "lines/HEC-1-A.npz", "lines/J82.npz", "lines/MIA_PaCa-2.npz",
         "lines/SK-MEL-2.npz"]


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def load(path):
    d = np.load(path, allow_pickle=True)
    cond = [str(c) for c in d["cond"]]
    return dict(path=path, X=d["X"].astype(np.float32),
                drug=np.asarray(d["drug"]).astype(str),
                index={c: i for i, c in enumerate(cond)}, cond=cond)


def pds_excl_drug(P, Y, drugs):
    Yn = unit(Y)
    Pn = unit(P)
    out = np.empty(len(Y))
    n1 = 0
    for i in range(len(Y)):
        pool = np.where(drugs != drugs[i])[0]
        pool = np.append(pool, i)
        s = Yn[pool] @ Pn[i]
        sv = s[-1]
        rank = int((s > sv).sum() + 0.5 * ((s == sv).sum() - 1))
        out[i] = 1.0 - rank / len(pool)
        n1 += rank == 0
    return out, n1


def main():
    lines = {}
    for p in PATHS:
        name = p.split("/")[-1][:-4]
        lines[name] = load(p)
        print(name, lines[name]["X"].shape, flush=True)
    names = list(lines)
    common = set(lines[names[0]]["cond"])
    for n in names[1:]:
        common &= set(lines[n]["index"])
    common = sorted(common)
    acts = []
    for n in names:
        idx = np.array([lines[n]["index"][c] for c in common])
        acts.append((np.abs(lines[n]["X"][idx]) > 2).mean(axis=1))
    act = np.mean(acts, axis=0)
    common = [c for c, a in zip(common, act) if a >= 0.005]
    print("common active", len(common), flush=True)
    drugs = np.array([lines[names[0]]["drug"][lines[names[0]]["index"][c]]
                      for c in common])
    print("drugs", len(set(drugs)), "max doses",
          max(np.sum(drugs == d) for d in set(drugs)), flush=True)
    save = {}
    for L in names:
        srcs = [o for o in names if o != L]
        rows = {n: np.array([lines[n]["index"][c] for c in common])
                for n in names}
        Y = lines[L]["X"][rows[L]]
        Q = np.mean([unit(lines[o]["X"][rows[o]]) for o in srcs], axis=0)
        full, n1 = pds_excl_drug(Q, Y,
                                 np.arange(len(common)).astype(str))
        exc, n1e = pds_excl_drug(Q, Y, drugs)
        save[L + " full"] = full
        save[L + " drug_excl"] = exc
        print(f"{L:12s} full mean {full.mean():.4f} "
              f"median {np.median(full):.4f} first {n1 / len(full):.3f} | "
              f"drug-excl mean {exc.mean():.4f} "
              f"median {np.median(exc):.4f} first {n1e / len(exc):.3f}",
              flush=True)
    np.savez("pilot_x.npz", **save)
    print("saved pilot_x.npz", flush=True)


if __name__ == "__main__":
    main()
