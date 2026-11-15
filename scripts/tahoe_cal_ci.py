"""Bootstrap confidence intervals for the Tahoe calibration deltas.

Reads the per-condition paired deltas written by ``tahoe_external.py select
--out-deltas``. Conditions are resampled with replacement within each line,
budget, and selection rule; the cross-line mean resamples conditions within
each line and averages the per-line means, so the panel composition is
fixed. Writes outputs/tahoe_calibration_ci.csv.
"""

import csv
import sys
from collections import defaultdict

import numpy as np

REPS = 10000
SEED = 0
KS = (20, 50, 100, 200)
STRATS = ("kmeans", "random")


def read(path):
    out = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["holdout"], int(r["k"]), r["strategy"])].append(
                float(r["delta"]))
    return out


def boot_mean(v, rng):
    idx = rng.integers(0, len(v), (REPS, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5))


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/deltasX.csv"
    out_path = (sys.argv[2] if len(sys.argv) > 2
                else "outputs/tahoe_calibration_ci.csv")
    per = read(in_path)
    lines = sorted({k[0] for k in per})
    complete = [L for L in lines
                if all(per.get((L, k, s)) for k in KS for s in STRATS)]
    if len(complete) < len(lines):
        print("warning: incomplete lines excluded from the cross-line mean:",
              [L for L in lines if L not in complete])
    rng = np.random.default_rng(SEED)
    rows = []
    for L in lines:
        for k in KS:
            for s in STRATS:
                v = np.asarray(per.get((L, k, s), []))
                if not len(v):
                    continue
                m, lo, hi = boot_mean(v, rng)
                rows.append(dict(line=L, k=k, strategy=s, n=len(v),
                                 mean=m, lo=lo, hi=hi))
    for k in KS:
        for s in STRATS:
            arrs = [np.asarray(per[(L, k, s)]) for L in complete]
            if not arrs:
                continue
            m = float(np.mean([a.mean() for a in arrs]))
            means = np.empty(REPS)
            for b in range(REPS):
                means[b] = np.mean(
                    [a[rng.integers(0, len(a), len(a))].mean()
                     for a in arrs])
            rows.append(dict(line="ALL", k=k, strategy=s,
                             n=sum(len(a) for a in arrs), mean=m,
                             lo=float(np.percentile(means, 2.5)),
                             hi=float(np.percentile(means, 97.5))))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["line", "k", "strategy", "n",
                                          "mean", "lo", "hi"])
        w.writeheader()
        w.writerows(rows)
    print(f"{'line':12s} {'k':>3s} {'strategy':8s} {'n':>4s} "
          f"{'mean':>8s} {'95% CI':>17s}")
    for r in rows:
        print(f"{r['line']:12s} {r['k']:3d} {r['strategy']:8s} "
              f"{r['n']:4d} {r['mean']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
