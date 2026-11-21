"""Summarize the weak vs saturated transfer regimes for the manuscript.

Reads the Nadig Tier A balanced-panel artifacts and the Tahoe-100M external
panel (log2FC main, Wald stat as sensitivity), prints the numbers used in
the regime-boundary section.
"""

import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

TAHOE_LINES = ["A549", "A498", "HS-578T", "HEC-1-A", "J82", "MIA_PaCa-2",
               "SK-MEL-2"]
NADIG_LINES = ["k562", "rpe1", "hepg2", "jurkat"]


def load_pilot():
    z = np.load("outputs/pilot_x.npz")
    n = np.load("outputs/pilot_pds_nadig_noexcl.npz")
    c = np.load("outputs/pilot_pds_nadig_corrected.npz")
    tahoe, excl = {}, {}
    for f in z.files:
        name, kind = f.rsplit(" ", 1)
        (tahoe if kind == "full" else excl)[name] = z[f]
    nadig = {L: n[L + "_noexcl"] for L in NADIG_LINES}
    nadig_excl = {L: c[L + "_excl"] for L in NADIG_LINES}
    return tahoe, excl, nadig, nadig_excl


def per_query(tahoe, excl, nadig):
    print("== per-query equal-mix PDS (log2FC) ==")
    print(f"{'line':12s} {'mean':>6s} {'median':>6s} {'top':>6s} "
          f"{'min':>6s}")
    for L in TAHOE_LINES:
        v = tahoe[L]
        print(f"{L:12s} {v.mean():.4f} {np.median(v):.4f} "
              f"{(v >= 1).mean():.3f} {v.min():6.3f}")
    for L in NADIG_LINES:
        v = nadig[L]
        print(f"{L:12s} {v.mean():.4f} {np.median(v):.4f} "
              f"{(v >= 1).mean():.3f} {v.min():6.3f}")


def drug_excl(tahoe, excl):
    print("\n== same-drug dose exclusion (Tahoe) ==")
    for L in TAHOE_LINES:
        a, b = tahoe[L], excl[L]
        print(f"{L:12s} full {a.mean():.4f} excl {b.mean():.4f} "
              f"delta {b.mean() - a.mean():+.4f}")


def saturation():
    print("\n== source saturation, mean PDS by |S| (log2FC) ==")
    rows = list(csv.DictReader(open("outputs/cacheX.csv")))
    tab = defaultdict(list)
    for r in rows:
        s = int(r["size"])
        if s:
            tab[(r["line"], s)].append(float(r["pds"]))
    for L in TAHOE_LINES:
        print(f"{L:12s} " + " ".join(f"{np.mean(tab[(L, s)]):.4f}"
                                     for s in range(1, 7)))
    print("Tahoe mean  " + " ".join(
        f"{np.mean([np.mean(tab[(L, s)]) for L in TAHOE_LINES]):.4f}"
        for s in range(1, 7)))
    rows = list(csv.DictReader(open("outputs/nadig_saturation_noexcl.csv")))
    ntab = defaultdict(list)
    for r in rows:
        ntab[(r["line"], int(r["size"]))].append(float(r["pds"]))
    for L in NADIG_LINES:
        print(f"{L:12s} " + " ".join(
            f"{np.mean(ntab[(L, s)]):.4f}" for s in (1, 2, 3)))
    print("Nadig mean  " + " ".join(
        f"{np.mean([np.mean(ntab[(L, s)]) for L in NADIG_LINES]):.4f}"
        for s in (1, 2, 3)))


def _cal_block(name, path, strategies):
    if not os.path.exists(path):
        print(f"{name}: {path} missing")
        return
    rows = list(csv.DictReader(open(path)))
    for strat in strategies:
        for k in (20, 50, 100, 200):
            per = []
            for L in TAHOE_LINES:
                v = [float(r["delta"]) for r in rows
                     if r["holdout"] == L and r["strategy"] == strat
                     and int(r["k"]) == k]
                if v:
                    per.append(np.mean(v))
            if per:
                print(f"{name:8s} {strat:12s} k={k:3d} "
                      f"{np.mean(per):+.4f} (sd {np.std(per):.4f}, "
                      f"n={len(per)})")


def calibration():
    print("\n== calibration dPDS vs k ==")
    _cal_block("Tahoe", "outputs/selectX.csv",
               ("kmeans", "strat", "random", "magnitude", "agreement",
                "disagreement"))
    _cal_block("Tahoe-S", "outputs/select7b.csv",
               ("kmeans", "random"))
    rows = list(csv.DictReader(open("outputs/phase2_fullmix_calibration.csv")))
    for strat in ("kmeans", "random"):
        for k in (20, 50, 100, 200):
            per = [float(r[strat]) for r in rows if int(r["k"]) == k]
            print(f"Nadig    {strat:12s} k={k:3d} {np.mean(per):+.4f} "
                  f"(sd {np.std(per):.4f}, n={len(per)})")
    if os.path.exists("outputs/phase2_fullmix_noexcl.csv"):
        rows = list(csv.DictReader(open("outputs/phase2_fullmix_noexcl.csv")))
        for strat in ("kmeans", "random"):
            for k in (20, 50, 100, 200):
                per = [float(r["delta"]) for r in rows
                       if r["strategy"] == strat and int(r["k"]) == k]
                print(f"Nadig-noX {strat:9s} k={k:3d} {np.mean(per):+.4f} "
                      f"(sd {np.std(per):.4f}, n={len(per)})")


def sscv():
    print("\n== source-side cross-validation ==")
    for name, path in (("Tahoe-X", "outputs/sscvX.csv"),
                       ("Tahoe-S", "outputs/sscv7.csv"),
                       ("Nadig", "outputs/phase2_exante_allocator.csv")):
        if not os.path.exists(path):
            print(f"{name}: {path} missing")
            continue
        rows = list(csv.DictReader(open(path)))
        a = np.array([float(r["actual"]) for r in rows])
        p = np.array([float(r["pseudo"]) for r in rows])
        rho, pv = spearmanr(p, a)
        print(f"{name}: n={len(rows)} rho={rho:+.3f} p={pv:.2g} "
              f"sign={np.mean((p > 0) == (a > 0)):.2f} "
              f"actual med={np.median(a):+.4f} pseudo med={np.median(p):+.4f}")


def pilot_diag(tahoe, nadig, k=50, tau=0.92, reps=50000, seed=0):
    print(f"\n== pilot diagnostic, k={k}, tau={tau} (log2FC) ==")
    for name, arrs, flip in (("Nadig", [nadig[L] for L in NADIG_LINES],
                              "stop"),
                             ("Tahoe", [tahoe[L] for L in TAHOE_LINES],
                              "calibrate")):
        probs = []
        for i, a in enumerate(arrs):
            rng = np.random.default_rng(seed + i)
            m = a[rng.integers(0, len(a), (reps, k))].mean(axis=1)
            probs.append(np.mean(m >= tau))
        probs = np.array(probs)
        err = probs if flip == "stop" else 1 - probs
        print(f"{name}: P(pilot>=tau) mean {probs.mean():.4f} "
              f"max {probs.max():.4f} min {probs.min():.4f} | "
              f"false {flip} mean {err.mean():.4f} max {err.max():.4f}")


def pilot_table(tahoe, nadig_excl, nadig, k=50, tau=0.92, reps=50000,
                seed=0):
    rows = []
    for regime, lines, arrs in (
            ("Tahoe", TAHOE_LINES, [tahoe[L] for L in TAHOE_LINES]),
            ("Nadig-exclusion", NADIG_LINES,
             [nadig_excl[L] for L in NADIG_LINES]),
            ("Nadig-noexclusion", NADIG_LINES,
             [nadig[L] for L in NADIG_LINES])):
        for i, (L, a) in enumerate(zip(lines, arrs)):
            rng = np.random.default_rng(seed + i)
            m = a[rng.integers(0, len(a), (reps, k))].mean(axis=1)
            p = float(np.mean(m >= tau))
            err = 1.0 - p if regime == "Tahoe" else p
            rows.append(dict(regime=regime, line=L, rate=p, error=err))
    with open("outputs/pilot_rates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["regime", "line", "rate", "error"])
        w.writeheader()
        w.writerows(rows)
    print("\n== pilot error table ==")
    for regime in dict.fromkeys(r["regime"] for r in rows):
        sub = [r for r in rows if r["regime"] == regime]
        e = np.array([r["error"] for r in sub])
        print(f"{regime:18s} mean {100 * e.mean():.2f}% "
              f"max {100 * e.max():.2f}% ({sub[int(e.argmax())]['line']})")
    print("wrote outputs/pilot_rates.csv")


def main():
    tahoe, excl, nadig, nadig_excl = load_pilot()
    per_query(tahoe, excl, nadig)
    drug_excl(tahoe, excl)
    saturation()
    calibration()
    sscv()
    pilot_diag(tahoe, nadig_excl)
    print("\n-- pilot under matched no-exclusion scoring --")
    pilot_diag(tahoe, nadig)
    pilot_table(tahoe, nadig_excl, nadig)


if __name__ == "__main__":
    main()
