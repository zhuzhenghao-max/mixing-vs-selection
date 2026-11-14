"""Consolidate Phase 1 selection scans into one table and CSV."""

import csv
import glob

import numpy as np

rows, seen = [], set()
for p in sorted(glob.glob("outputs/phase1_selection_pilot_*.csv")):
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("k", "delta_split_mean", "delta_split_sd", "vs_random",
                      "vs_random_lo", "vs_random_hi", "vs_random_p",
                      "win_vs_random"):
                r[k] = float(r[k])
            key = (r["tier"], r["holdout"], int(r["k"]), r["strategy"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

rows.sort(key=lambda r: (r["tier"], r["holdout"], int(r["k"]),
                         r["strategy"]))

print(f"{'tier':4s} {'line':8s} {'k':>4s} {'random':>8s} {'kmeans':>8s} "
      f"{'gain':>8s} {'CI':>20s} {'p':>9s} {'win':>5s}")
out = []
for r in rows:
    if r["strategy"] != "kmeans":
        continue
    key = (r["tier"], r["holdout"], int(r["k"]))
    d = {x["strategy"]: x for x in rows
         if (x["tier"], x["holdout"], int(x["k"])) == key}
    if "random" in d and "kmeans" in d:
        km, rn = d["kmeans"], d["random"]
        out.append(dict(
            tier=r["tier"], holdout=r["holdout"], k=int(r["k"]),
            random=rn["delta_split_mean"],
            kmeans=km["delta_split_mean"],
            gain=km["vs_random"] if np.isfinite(km["vs_random"]) else np.nan,
            lo=km["vs_random_lo"], hi=km["vs_random_hi"],
            p=km["vs_random_p"], win=km["win_vs_random"]))

for o in out:
    print(f"{o['tier']:4s} {o['holdout']:8s} {o['k']:4d} "
          f"{o['random']:+8.4f} {o['kmeans']:+8.4f} {o['gain']:+8.4f} "
          f"[{o['lo']:+.4f},{o['hi']:+.4f}] {o['p']:9.2g} {o['win']:5.2f}")

with open("outputs/phase1_selection_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print("\nwrote outputs/phase1_selection_summary.csv")
