"""Summarize Phase 2: source-line gain versus calibration budget.

Reads outputs/phase2_source_budget_{tier}.csv, writes the exchange-rate table
and Figure 8 (three panels). For every leave-one-source-out state, the best
source addition is expressed in units of calibration perturbations measured
on the target line.
"""

import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase2_source_budget import exchange_k

KS = [20, 50, 100, 200]
COLORS = {"k562": "#1f77b4", "rpe1": "#d62728",
          "hepg2": "#2ca02c", "jurkat": "#ff7f0e"}
NAME = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}


def load(tier):
    with open(f"outputs/phase2_source_budget_{tier}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n_src"] = int(r["n_src"])
        r["k"] = int(r["k"])
        for key in ("delta", "lo", "hi", "pds", "eq"):
            r[key] = float(r[key])
    return rows


def exchange_rows(rows):
    out = []
    for L in dict.fromkeys(r["holdout"] for r in rows):
        sub = [r for r in rows if r["holdout"] == L]
        n_tot = max(r["n_src"] for r in sub)
        states = sorted({r["subset"] for r in sub
                         if r["n_src"] == n_tot - 1})
        for S in states:
            st = [r for r in sub if r["subset"] == S]
            gains = [r for r in st
                     if r["strategy"] == "equal" and r["k"] == -1]
            if not gains:
                continue
            best = max(gains, key=lambda r: r["delta"])
            ks = sorted({r["k"] for r in st
                         if r["strategy"] in ("kmeans", "random")
                         and r["k"] > 0})
            grid = [0] + ks
            stars = {}
            kmax = {}
            for strat in ("kmeans", "random"):
                curve = {r["k"]: r["delta"] for r in st
                         if r["strategy"] == strat}
                vals = [0.0] + [curve.get(k, np.nan) for k in ks]
                stars[strat] = exchange_k(grid, vals, best["delta"])
                kmax[strat] = curve.get(ks[-1], np.nan)
            out.append(dict(
                tier=best["tier"], holdout=L, subset=S, srcs=best["srcs"],
                n_src=best["n_src"], added=best["added"],
                source_delta=best["delta"], source_lo=best["lo"],
                source_hi=best["hi"],
                kmeans_kstar=stars["kmeans"], random_kstar=stars["random"],
                kmeans_at_kmax=kmax["kmeans"],
                random_at_kmax=kmax["random"],
                source_gains=";".join(
                    f"{r['added']}:{r['delta']:+.4f}" for r in gains)))
    return out


def print_exchange(out):
    print(f"{'line':8s} {'subset':7s} {'added':7s} {'dsrc':>8s} "
          f"{'k* kmeans':>10s} {'k* random':>10s}")
    for r in out:
        fmt = lambda v: (f">{KS[-1]}" if not np.isfinite(v) else f"{v:.0f}")
        print(f"{r['holdout']:8s} {r['subset']:7s} {r['added']:7s} "
              f"{r['source_delta']:+8.4f} {fmt(r['kmeans_kstar']):>10s} "
              f"{fmt(r['random_kstar']):>10s}")


def fullmix_rows(rows):
    out = []
    for L in dict.fromkeys(r["holdout"] for r in rows):
        sub = [r for r in rows if r["holdout"] == L]
        n_tot = max(r["n_src"] for r in sub)
        full = [r for r in sub if r["n_src"] == n_tot
                and r["strategy"] in ("kmeans", "random", "equal")]
        for k in [0] + KS:
            kv = {}
            for r in full:
                if r["k"] == k:
                    key = r["strategy"]
                    kv[key] = r
            if kv:
                out.append(dict(
                    holdout=L, k=k,
                    eq=kv["equal"]["delta"] if "equal" in kv else 0.0,
                    kmeans=kv.get("kmeans", {}).get("delta", np.nan),
                    kmeans_lo=kv.get("kmeans", {}).get("lo", np.nan),
                    kmeans_hi=kv.get("kmeans", {}).get("hi", np.nan),
                    random=kv.get("random", {}).get("delta", np.nan),
                    random_lo=kv.get("random", {}).get("lo", np.nan),
                    random_hi=kv.get("random", {}).get("hi", np.nan)))
    return out


def figure(rows, exch, fm, prefix):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # Panel A: equal-mix PDS of every source subset
    ax = axes[0]
    order = ["0", "1", "2", "0-1", "0-2", "1-2", "0-1-2"]
    for L in dict.fromkeys(r["holdout"] for r in rows):
        eq = {r["subset"]: r["eq"] for r in rows
              if r["holdout"] == L}
        y = [eq.get(s, np.nan) for s in order]
        ax.plot(range(len(order)), y, marker="o", ms=4, lw=1.4,
                color=COLORS.get(L, "#555555"), label=L.upper())
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("equal-mix PDS")
    ax.set_title("source subsets", fontsize=10)
    ax.legend(frameon=False, fontsize=7)

    # Panel B: calibration curve at the full mix
    ax = axes[1]
    for L in dict.fromkeys(r["holdout"] for r in fm):
        sub = [r for r in fm if r["holdout"] == L]
        ks = [r["k"] for r in sub]
        c = COLORS.get(L, "#555555")
        ax.plot(ks, [r["kmeans"] for r in sub], marker="o", ms=4, lw=1.6,
                color=c, label=f"{L.upper()} kmeans")
        ax.plot(ks, [r["random"] for r in sub], marker="s", ms=4, lw=1.2,
                ls="--", color=c, alpha=0.75,
                label=f"{L.upper()} random")
    ax.axhline(0, color="0.7", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([1] + KS)
    ax.set_xticklabels(["0"] + [str(k) for k in KS])
    ax.set_xlabel("calibration perturbations measured (k)")
    ax.set_ylabel("ΔPDS vs equal mixing")
    ax.set_title("full source mix", fontsize=10)
    ax.legend(frameon=False, fontsize=6, ncol=2)

    # Panel C: exchange rate for states where the source addition helps
    ax = axes[2]
    pos = [r for r in exch if r["source_delta"] > 0]
    labels = [f"{NAME.get(r['holdout'], r['holdout'])} {r['subset']}"
              for r in pos]
    x = np.arange(len(pos))
    for j, mult in enumerate((0.82, 1.18)):
        strat = "kmeans" if j == 0 else "random"
        mk = "o" if strat == "kmeans" else "s"
        for xi, r in zip(x * mult, pos):
            v = r[f"{strat}_kstar"]
            col = COLORS.get(r["holdout"], "#555555")
            if np.isfinite(v):
                ax.scatter([xi], [v], c=[col], marker=mk, s=38, zorder=3)
            else:
                ax.scatter([xi], [205], c=[col], marker="^", s=44, zorder=3)
        ax.scatter([], [], c="#555555", marker=mk, s=38, label=strat)
    ax.scatter([], [], c="#555555", marker="^", s=44, label=">200")
    ax.axhline(200, color="0.6", lw=0.9, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("k* calibration perturbations ≈ one source line")
    ax.set_title("exchange rate (positive source gains)", fontsize=10)
    ax.set_ylim(0, 225)
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{prefix}.{ext}", dpi=300)
    print(f"\nwrote {prefix}.png/.pdf")


def main():
    all_rows, all_exch, all_fm = [], [], []
    for tier in ("A", "B"):
        try:
            rows = load(tier)
        except FileNotFoundError:
            continue
        all_rows += rows
        all_exch += exchange_rows(rows)
        all_fm += fullmix_rows(rows)
        print(f"\n=== Tier {tier} exchange rate ===")
        print_exchange([r for r in all_exch if r["tier"] == tier])

    print("\nFull-mix calibration: kmeans beats random at k<=50, "
          "random beats kmeans at k=200")
    for r in all_fm:
        if r["k"] == 0:
            continue
        win = ("kmeans" if r["kmeans"] > r["random"]
               else "random" if r["random"] > r["kmeans"] else "tie")
        print(f"  {r['holdout']:8s} k={r['k']:3d} "
              f"kmeans {r['kmeans']:+.4f} random {r['random']:+.4f} -> {win}")

    keys = list(all_exch[0].keys())
    with open("outputs/phase2_exchange.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_exch)
    print("\nwrote outputs/phase2_exchange.csv")

    fmkeys = list(all_fm[0].keys())
    with open("outputs/phase2_fullmix_calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fmkeys)
        w.writeheader()
        w.writerows(all_fm)
    print("wrote outputs/phase2_fullmix_calibration.csv")

    figure(all_rows, all_exch, all_fm, "figures/fig8_source_budget")


if __name__ == "__main__":
    main()
