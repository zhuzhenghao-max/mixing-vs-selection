"""Positive and negative controls for the PDS retrieval metric.

Fix a set of real queries and their true held-out responses, then score a
graded family of synthetic predictions with the canonical leave-one-line-out
PDS (target-coordinate exclusion, midrank ties, denominator n):

  truth        the true response itself (positive control, expected 1)
  mix          the equal source mixture (the benchmark baseline)
  mean         the target-agnostic mean reference row
  mismatch     another query's response (negative control, expected ~0.5)
  gene_shuf    the true response with genes permuted within the row
  sign_flip    the negated true response
  noise_s      truth + s * matched-norm Gaussian noise (graded control)

Also reports raw cosine to the truth so that rank-based and alignment-based
scores can be compared on the same predictions. Samples 300 queries per
holdout; all reference rows stay in the pool.
"""

import csv
import sys

import numpy as np

sys.path.insert(0, "scripts")
from bench_common import (Retrieval, boot_ci, load_axis, load_lines,
                          load_nadig_lines, unit_rows)
from bench_supervised import build_features

TIERS = {
    "A": (["k562", "rpe1", "hepg2", "jurkat"], load_nadig_lines),
    "B": (["h1", "k562", "rpe1", "hct116", "hek293t"], None),
}
N_SAMPLE = 300
SEED = 0


def derangement(n, rng):
    p = rng.permutation(n)
    fixed = np.where(p == np.arange(n))[0]
    if len(fixed) > 1:
        p[fixed] = np.roll(p[fixed], 1)
    return p


def full(P, pos, n_genes):
    out = np.zeros((P.shape[0], n_genes), dtype=np.float32)
    out[:, pos] = P.astype(np.float32)
    return out


def cosine_to_truth(P, Y):
    A = unit_rows(P)
    B = unit_rows(Y)
    return np.einsum("ij,ij->i", A, B)


def summarize(rec, p, cos):
    lo, hi, mu = boot_ci(p, n=4000, seed=SEED)
    rec.update(n=len(p), mean=mu, lo=lo, hi=hi,
               median=float(np.median(p)),
               win_above_half=float(np.mean(p > 0.5)),
               frac_ge_0_9=float(np.mean(p >= 0.9)),
               cosine_mean=float(np.mean(cos)))
    return rec


def run_tier(tier, holds, genes, pos, gidx, m):
    lines = (load_nadig_lines(genes, gidx) if tier == "A"
             else load_lines(genes, gidx, m))
    lines.pop("_gidx", None)
    out = []
    rng = np.random.default_rng(SEED + (0 if tier == "A" else 1000))
    for L in holds:
        ref = lines[L]
        srcs = [o for o in holds if o != L and o in lines]
        q = [t for t in ref["names"]
             if any(t in lines[o]["index"] for o in srcs)]
        if len(q) > N_SAMPLE:
            idx = np.sort(rng.choice(len(q), N_SAMPLE, replace=False))
            q = [q[i] for i in idx]
        print(f"\n=== Tier {tier} {L}: {len(q)} sampled queries, "
              f"pool {len(ref['names'])} ===", flush=True)
        X = build_features(lines, srcs, q, pos, "mix")
        Y = np.zeros((len(q), len(pos)), dtype=np.float32)
        for i, t in enumerate(q):
            v = ref["rows"][ref["index"][t], pos].astype(np.float64)
            nv = np.linalg.norm(v)
            if nv > 1e-9:
                Y[i] = v / nv
        R = Retrieval(ref["rows"], ref["names"], pos, gidx=gidx)
        mean = ref["rows"][:, pos].mean(0)
        mean = mean / max(np.linalg.norm(mean), 1e-9)

        preds = {}

        def put(name, P):
            preds[name] = np.asarray(P, dtype=np.float32)

        put("mix", X)
        put("mean", np.tile(mean, (len(q), 1)))
        perm = derangement(len(q), rng)
        put("mismatch", Y[perm])
        Sh = np.stack([row[rng.permutation(len(pos))] for row in Y])
        put("gene_shuf", Sh)
        put("sign_flip", -Y)
        for s in (0.25, 0.5, 1.0, 2.0, 4.0):
            E = rng.normal(size=Y.shape).astype(np.float64)
            E *= (np.linalg.norm(Y, axis=1, keepdims=True) /
                  np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12))
            put(f"noise_{s:g}", Y + s * E)

        truth_full = full(Y, pos, len(genes))
        p = R.score(truth_full, q, exclude="self", denominator="n")
        cos = cosine_to_truth(Y, Y)
        out.append(summarize(dict(tier=tier, holdout=L, method="truth"),
                             p, cos))
        print(f"  truth        {out[-1]['mean']:.4f}", flush=True)
        for name, P in preds.items():
            Pf = full(P, pos, len(genes))
            p = R.score(Pf, q, exclude="self", denominator="n")
            cos = cosine_to_truth(P, Y)
            rec = summarize(dict(tier=tier, holdout=L, method=name), p, cos)
            out.append(rec)
            print(f"  {name:12s} pds {rec['mean']:.4f} "
                  f"[{rec['lo']:.4f},{rec['hi']:.4f}] "
                  f"cos {rec['cosine_mean']:+.4f}", flush=True)
        del X, Y, preds, R
    return out


def main():
    genes, pos, gidx, m = load_axis()
    rows = []
    rows += run_tier("A", TIERS["A"][0], genes, pos, gidx, m)
    rows += run_tier("B", TIERS["B"][0], genes, pos, gidx, m)

    path = "outputs/metric_controls.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")

    print("\nAcross-holdout mean (Tier A / Tier B):")
    names = ["truth", "mix", "mean", "mismatch", "gene_shuf", "sign_flip",
             "noise_0.25", "noise_0.5", "noise_1", "noise_2", "noise_4"]
    for tier in ("A", "B"):
        print(f"  Tier {tier}")
        for name in names:
            v = [r["mean"] for r in rows
                 if r["tier"] == tier and r["method"] == name]
            c = [r["cosine_mean"] for r in rows
                 if r["tier"] == tier and r["method"] == name]
            if v:
                print(f"    {name:12s} PDS {np.mean(v):.4f} "
                      f"(min {np.min(v):.4f}, max {np.max(v):.4f})  "
                      f"cos {np.mean(c):+.4f}")


if __name__ == "__main__":
    main()
