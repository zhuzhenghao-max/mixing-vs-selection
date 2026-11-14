"""Print the cross-line and J82 calibration deltas with bootstrap CIs."""

import csv
import sys

path = (sys.argv[1] if len(sys.argv) > 1
        else "outputs/tahoe_calibration_ci.csv")
rows = list(csv.DictReader(open(path, newline="")))
ks = (20, 50, 100, 200)


def show(line, strategy):
    print(f"{line} / {strategy}")
    for k in ks:
        r = next((x for x in rows if x["line"] == line
                  and int(x["k"]) == k and x["strategy"] == strategy), None)
        if r:
            print(f"  k={k:3d} n={int(r['n']):4d} "
                  f"{float(r['mean']):+.4f} "
                  f"[{float(r['lo']):+.4f}, {float(r['hi']):+.4f}]")


for strategy in ("kmeans", "random"):
    show("ALL", strategy)
for strategy in ("kmeans", "random"):
    show("J82", strategy)
