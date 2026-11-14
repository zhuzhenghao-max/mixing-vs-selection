"""Build the calibration-budget, source-budget, and sensitivity tables.

Table 3     selection and full-mix calibration
Table 4     source subsets and exchange rates
Table S9    source-side allocation decisions
Table S10   calibration under no-exclusion scoring
"""

import csv
from collections import defaultdict

import numpy as np

OUT = "outputs/tables"


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def table3():
    rows = []
    for r in read("outputs/phase1_selection_summary.csv"):
        k = int(float(r["k"]))
        if k not in (20, 50):
            continue
        rows.append(dict(block="selection", tier=r["tier"],
                         holdout=r["holdout"], k=k,
                         contrast="kmeans-random",
                         delta=f"{float(r['gain']):+.4f}",
                         lo=f"{float(r['lo']):+.4f}",
                         hi=f"{float(r['hi']):+.4f}",
                         p=f"{float(r['p']):.2g}",
                         win=f"{float(r['win']):.2f}"))
    for r in read("outputs/phase1_selection_pilot_B_k20_all5lean.csv"):
        if r["strategy"] != "kmeans":
            continue
        rows.append(dict(block="selection", tier=r["tier"],
                         holdout=r["holdout"], k=int(float(r["k"])),
                         contrast="kmeans-random",
                         delta=f"{float(r['vs_random']):+.4f}",
                         lo=f"{float(r['vs_random_lo']):+.4f}",
                         hi=f"{float(r['vs_random_hi']):+.4f}",
                         p=f"{float(r['vs_random_p']):.2g}",
                         win=f"{float(r['win_vs_random']):.2f}"))
    for r in read("outputs/phase2_fullmix_calibration.csv"):
        k = int(float(r["k"]))
        if k not in (50, 100):
            continue
        for strat in ("kmeans", "random"):
            rows.append(dict(block="fullmix", tier="A",
                             holdout=r["holdout"], k=k, contrast=strat,
                             delta=f"{float(r[strat]):+.4f}",
                             lo=f"{float(r[strat + '_lo']):+.4f}",
                             hi=f"{float(r[strat + '_hi']):+.4f}",
                             p="", win=""))
    fields = ["block", "tier", "holdout", "k", "contrast", "delta", "lo",
              "hi", "p", "win"]
    write(f"{OUT}/table3_calibration_budget.csv", fields, rows)


def table4():
    rows = []
    seen = set()
    for r in read("outputs/phase2_source_budget_A.csv"):
        if not r["eq"]:
            continue
        key = (r["holdout"], r["subset"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(block="subsets", holdout=r["holdout"],
                         subset=r["subset"], srcs=r["srcs"],
                         n_src=r["n_src"], added="",
                         value=f"{float(r['eq']):.4f}", lo="", hi="",
                         kmeans_kstar="", random_kstar=""))
    rows.sort(key=lambda r: (r["holdout"], int(r["n_src"]), r["subset"]))
    for r in read("outputs/phase2_exchange.csv"):
        rows.append(dict(block="exchange", holdout=r["holdout"],
                         subset=r["subset"], srcs=r["srcs"],
                         n_src=r["n_src"], added=r["added"],
                         value=f"{float(r['source_delta']):+.4f}",
                         lo=f"{float(r['source_lo']):+.4f}",
                         hi=f"{float(r['source_hi']):+.4f}",
                         kmeans_kstar=(">200" if not np.isfinite(
                             float(r["kmeans_kstar"]))
                             else f"{float(r['kmeans_kstar']):.1f}"),
                         random_kstar=(">200" if not np.isfinite(
                             float(r["random_kstar"]))
                             else f"{float(r['random_kstar']):.1f}")))
    fields = ["block", "holdout", "subset", "srcs", "n_src", "added",
              "value", "lo", "hi", "kmeans_kstar", "random_kstar"]
    write(f"{OUT}/table4_source_budget.csv", fields, rows)


def tableS9():
    rows = []
    for r in read("outputs/phase2_exante_allocator.csv"):
        rows.append(dict(holdout=r["holdout"], state=r["state"],
                         added=r["added"],
                         actual=f"{float(r['actual']):+.4f}",
                         pseudo=f"{float(r['pseudo']):+.4f}",
                         pseudo_1=f"{float(r['pseudo_1']):+.4f}",
                         pseudo_2=f"{float(r['pseudo_2']):+.4f}"))
    fields = ["holdout", "state", "added", "actual", "pseudo",
              "pseudo_1", "pseudo_2"]
    write(f"{OUT}/tableS9_source_allocation.csv", fields, rows)


def tableS10():
    rows = read("outputs/phase2_fullmix_noexcl.csv")
    tab = defaultdict(list)
    for r in rows:
        tab[(r["holdout"], int(float(r["k"])), r["strategy"])].append(
            float(r["delta"]))
    out = []
    for L in ("k562", "rpe1", "hepg2", "jurkat"):
        for k in (20, 50, 100, 200):
            out.append(dict(holdout=L, k=k,
                            kmeans=f"{np.mean(tab[(L, k, 'kmeans')]):+.4f}",
                            random=f"{np.mean(tab[(L, k, 'random')]):+.4f}"))
    write(f"{OUT}/tableS10_calibration_noexcl.csv",
          ["holdout", "k", "kmeans", "random"], out)


if __name__ == "__main__":
    table3()
    table4()
    tableS9()
    tableS10()
