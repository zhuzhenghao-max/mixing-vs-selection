"""Compare per-line calibration means in deltasX with selectX."""

import csv
import sys
from collections import defaultdict

import numpy as np

dpath = sys.argv[1] if len(sys.argv) > 1 else "outputs/deltasX_partial.csv"
spath = sys.argv[2] if len(sys.argv) > 2 else "outputs/selectX.csv"

d = defaultdict(list)
for r in csv.DictReader(open(dpath, newline="")):
    d[(r["holdout"], int(r["k"]), r["strategy"])].append(float(r["delta"]))

s = defaultdict(list)
for r in csv.DictReader(open(spath, newline="")):
    s[(r["holdout"], int(r["k"]), r["strategy"])].append(float(r["delta"]))

print(f"{'line':10s} {'k':>4s} {'strategy':8s} {'deltas_mean':>12s} "
      f"{'select_mean':>12s} {'diff':>9s} {'n_d':>5s} {'n_s':>4s}")
for key in sorted(d):
    if key not in s:
        continue
    a, b = float(np.mean(d[key])), float(np.mean(s[key]))
    print(f"{key[0]:10s} {key[1]:4d} {key[2]:8s} {a:+12.5f} "
          f"{b:+12.5f} {a - b:+9.5f} {len(d[key]):5d} {len(s[key]):4d}")
