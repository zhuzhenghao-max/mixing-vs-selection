"""Candidate-pool composition control on the balanced Nadig panel.

The 638 queries, their predictions, and the true responses are held fixed.
Only the competitor pool changes, at sizes m = 50 and 100:

  random       m-1 uniformly drawn non-self candidates
  hard         m-1 candidates most similar to the prediction
  easy         m-1 candidates least similar to the prediction
  coherent     m-1 candidates with the strongest source-source agreement
  incoherent   m-1 candidates with the weakest source-source agreement
  strong       m-1 candidates with the largest source response norm
  quiet        m-1 candidates with the smallest source response norm

Each pool always contains the true response. Candidate selection uses only
source responses (the prediction and source statistics); the held-out
response supplies only the final rank. Predictions compared under identical
pools: equal mixing, per-source centered equal mixing, a single source, and
the target-agnostic mean. Scoring keeps the canonical target-coordinate
exclusion, per-reference normalization, and tie midrank.
"""

import csv
import sys

import numpy as np

sys.path.insert(0, "scripts")
from bench_common import boot_ci, load_axis, load_nadig_lines, unit_rows
from bench_supervised import build_features
from phase1_selection_pilot import FastRetrieval, source_stats
from pilot_small_pool import similarity_matrix

HOLDS = ("k562", "rpe1", "hepg2", "jurkat")
SIZES = (50, 100)
REPS = 200
SEED = 0


def build_methods(lines, srcs, ref, queries, pos):
    n = len(queries)
    d = len(pos)
    Xeq = build_features(lines, srcs, queries, pos, "mix").astype(np.float64)
    means = {o: lines[o]["rows"][:, pos].mean(0) for o in srcs}
    Xc = np.zeros((n, d))
    Xs = np.zeros((n, d))
    for i, t in enumerate(queries):
        acc = []
        for o in srcs:
            j = lines[o]["index"].get(t)
            if j is None:
                continue
            v = lines[o]["rows"][j, pos].astype(np.float64)
            acc.append(unit_rows((v - means[o])[None, :])[0])
            if np.linalg.norm(Xs[i]) == 0:
                Xs[i] = v
        if acc:
            Xc[i] = np.mean(acc, axis=0)
    mq = ref["rows"][:, pos].mean(0)
    Xm = np.tile(unit_rows(mq[None, :])[0], (n, 1))
    return {"equal": Xeq, "equal_center": Xc, "single": Xs,
            "mean": Xm}


def pool_pds(S, pools):
    p = np.empty(len(pools))
    for q, idx in enumerate(pools):
        s = S[idx, q]
        sv = S[q, q]
        rank = (s > sv).sum() + 0.5 * ((s == sv).sum() - 1)
        p[q] = 1.0 - rank / len(idx)
    return p


def fixed_pools(S, stats, m):
    n = len(S)
    out = {}
    for name in ("hard", "easy"):
        pools = []
        for q in range(n):
            others = np.delete(np.arange(n), q)
            col = S[others, q]
            order = others[np.argsort(-col if name == "hard" else col)]
            pools.append(np.concatenate(([q], order[:m - 1])))
        out[name] = pools
    for name, vals in (("coherent", stats["agreement"]),
                       ("incoherent", -stats["agreement"]),
                       ("strong", stats["raw_norm"]),
                       ("quiet", -stats["raw_norm"])):
        pools = []
        for q in range(n):
            others = np.delete(np.arange(n), q)
            order = others[np.argsort(-vals[others])]
            pools.append(np.concatenate(([q], order[:m - 1])))
        out[name] = pools
    return out


def random_pools(n, m, rng):
    pools = []
    for q in range(n):
        others = np.delete(np.arange(n), q)
        take = rng.choice(others, size=m - 1, replace=False)
        pools.append(np.concatenate(([q], take)))
    return pools


def summarize(tier, L, method, comp, m, reps, p):
    lo, hi, mu = boot_ci(p, n=4000, seed=SEED)
    return dict(tier=tier, holdout=L, method=method, composition=comp,
                m=m, reps=reps, n=len(p), mean=mu, lo=lo, hi=hi,
                median=float(np.median(p)),
                frac_ge_0_92=float(np.mean(p >= 0.92)))


def main():
    genes, pos, gidx, m_axis = load_axis()
    lines = load_nadig_lines(genes, gidx)
    lines.pop("_gidx", None)
    queries = sorted(set.intersection(
        *(set(lines[L]["names"]) for L in HOLDS)))
    n = len(queries)
    print(f"balanced panel: {n} queries", flush=True)
    rng = np.random.default_rng(SEED)
    rows, ranks = [], []
    for L in HOLDS:
        ref = lines[L]
        srcs = [o for o in HOLDS if o != L]
        balanced = ref["rows"][[ref["index"][t] for t in queries]]
        R = FastRetrieval(balanced, queries, pos, gidx)
        stats = source_stats(lines, srcs, queries, pos)
        methods = build_methods(lines, srcs, ref, queries, pos)
        S = {}
        for name, Q in methods.items():
            S[name] = similarity_matrix(R, Q, queries, "self").astype(
                np.float64)
        print(f"\n=== {L}: sources {srcs} ===", flush=True)

        full_idx = np.arange(n)
        for name in methods:
            p = pool_pds(S[name], [full_idx] * n)
            rows.append(summarize("Nadig", L, name, "full", n, 1, p))
        for m in SIZES:
            fixed = fixed_pools(S["equal"], stats, m)
            for comp, pools in fixed.items():
                for name in methods:
                    p = pool_pds(S[name], pools)
                    rows.append(summarize("Nadig", L, name, comp, m, 1, p))
            acc = {k: np.zeros(n) for k in methods}
            for rep in range(REPS):
                pools = random_pools(n, m, rng)
                for name in methods:
                    acc[name] += pool_pds(S[name], pools)
            for name in methods:
                p = acc[name] / REPS
                rows.append(summarize("Nadig", L, name, "random", m,
                                      REPS, p))

        for m in (n, *SIZES):
            comps = ["full"] if m == n else \
                ["random", "hard", "easy", "coherent", "incoherent",
                 "strong", "quiet"]
            for comp in comps:
                sub = [r for r in rows if r["holdout"] == L and r["m"] == m
                       and r["composition"] == comp]
                order = sorted(sub, key=lambda r: -r["mean"])
                for j, r in enumerate(order):
                    ranks.append(dict(tier="Nadig", holdout=L,
                                      composition=comp, m=m,
                                      method=r["method"],
                                      mean=r["mean"], rank=j + 1))
        for m in SIZES:
            print(f"  m={m}: " + "  ".join(
                f"{c} {np.mean([r['mean'] for r in rows if r['holdout'] == L and r['m'] == m and r['composition'] == c and r['method'] == 'equal']):.3f}"
                for c in ("random", "hard", "easy", "coherent",
                          "incoherent", "strong", "quiet")), flush=True)
        del R, S, methods

    with open("outputs/pool_composition_control.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open("outputs/pool_composition_rankings.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranks[0]))
        w.writeheader()
        w.writerows(ranks)
    print("\nwrote outputs/pool_composition_control.csv")
    print("wrote outputs/pool_composition_rankings.csv")

    print("\nEqual mix, mean PDS by composition (across lines):")
    for m in (n, *SIZES):
        print(f"  pool size {m}")
        comps = ["full"] if m == n else \
            ["random", "hard", "easy", "coherent", "incoherent",
             "strong", "quiet"]
        for c in comps:
            v = [r["mean"] for r in rows if r["m"] == m
                 and r["composition"] == c and r["method"] == "equal"]
            if v:
                print(f"    {c:11s} {np.mean(v):.4f} "
                      f"(min {np.min(v):.4f}, max {np.max(v):.4f})")
    print("\nMethod ranks at m=50 (1 = best):")
    for c in ("random", "hard", "easy", "coherent", "incoherent",
              "strong", "quiet"):
        sub = [r for r in ranks if r["m"] == 50 and r["composition"] == c]
        out = []
        for meth in ("equal", "equal_center", "single", "mean"):
            v = [r["rank"] for r in sub if r["method"] == meth]
            out.append(f"{meth}:{np.mean(v):.1f}")
        print(f"  {c:11s} " + "  ".join(out))


if __name__ == "__main__":
    main()
