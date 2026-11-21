"""Graphical abstract for the cross-line transfer paper.

(A) Equal-mix gain over the target-agnostic mean in nine holdouts and over a
GEARS adaptation in four Tier A folds, with paired 95% bootstrap intervals.
(B) Decomposition of the per-target oracle gain over equal mixing into the
split-half gain retained by one fixed rule and the remaining oracle gap.

Usage: python scripts/make_graphical_abstract.py
Writes figures/graphical_abstract.pdf + .png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

TIER_COLS = {"A": "#2A9D8F", "B": "#5C6B8A"}
GEARS_COL = "#D55E00"
GAP_COL = "#D3D3D3"

DISP = {"h1": "H1", "k562": "K562", "rpe1": "RPE1", "hct116": "HCT116",
        "hek293t": "HEK293T", "hepg2": "HepG2", "jurkat": "Jurkat"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def paired_ci(a, b, nboot=4000, seed=0):
    """Mean paired difference a - b with a percentile bootstrap interval."""
    common = sorted(set(a) & set(b))
    d = np.array([a[t] - b[t] for t in common], dtype=float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (nboot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def load_paired():
    out = {}
    for tier, path in (("A", "outputs/bench_transfer_nadig_v2.npz"),
                       ("B", "outputs/bench_transfer_v2.npz")):
        z = np.load(path, allow_pickle=True)
        tab = {}
        for h, m, t, p in zip(z["holdout"].astype(str),
                              z["method"].astype(str),
                              z["target"].astype(str),
                              z["pds_raw"].astype(float)):
            tab.setdefault((h, m), {})[t] = float(p)
        lines = {"A": ["k562", "rpe1", "hepg2", "jurkat"],
                 "B": ["h1", "k562", "rpe1", "hct116", "hek293t"]}[tier]
        for h in lines:
            out[(tier, h)] = paired_ci(tab[(h, "equal")], tab[(h, "mean")])
    return out


def load_gears():
    d = pd.read_csv("outputs/gears_summary_A.csv")
    return {r.holdout: (r.d_mean, r.ci_lo, r.ci_hi) for r in d.itertuples()}


def forest(ax, rows, xlim, title):
    y = np.arange(len(rows))[::-1]
    for yi, (lab, mean, lo, hi, c) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.3, zorder=3,
                solid_capstyle="round")
        ax.plot(mean, yi, "o", color=c, ms=4.2, mec="white", mew=0.5,
                zorder=4)
    ax.axvline(0, color="0.2", lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    for lab, c in zip(ax.get_yticklabels(), [r[4] for r in rows]):
        lab.set_color(c)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"$\Delta$ mean PDS vs equal mix", fontsize=6.8)
    ax.text(0.5, 1.10, title, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=6.8, color="0.25")
    ax.grid(axis="x", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def main():
    paired = load_paired()
    gears = load_gears()

    fig = plt.figure(figsize=(7.2, 2.75))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.36,
                             left=0.095, right=0.975, top=0.78, bottom=0.19)
    left = outer[0].subgridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.14)
    ax1 = fig.add_subplot(left[0])
    ax2 = fig.add_subplot(left[1])
    ax3 = fig.add_subplot(outer[1])

    rows = []
    for tier, h in [("A", "k562"), ("A", "rpe1"), ("A", "hepg2"),
                    ("A", "jurkat"), ("B", "h1"), ("B", "k562"),
                    ("B", "rpe1"), ("B", "hct116"), ("B", "hek293t")]:
        mean, lo, hi = paired[(tier, h)]
        rows.append((DISP[h], mean, lo, hi, TIER_COLS[tier]))
    forest(ax1, rows, (0, 0.37), "vs target-line mean (9 holdouts)")
    ax1.text(0.98, 0.05, "all p < $10^{-12}$", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=5.8, color="0.35")

    grows = [(DISP[h], *gears[h], GEARS_COL)
             for h in ("k562", "rpe1", "hepg2", "jurkat")]
    forest(ax2, grows, (0, 0.26), "vs GEARS (Tier A folds)")
    ax2.set_xlabel(r"$\Delta$ mean PDS vs equal mix", fontsize=6.8)

    fig.text(0.015, 0.955, "A", fontsize=10, fontweight="bold", va="top")
    fig.text(0.048, 0.958, "Training-free equal mixing beats every baseline",
             fontsize=7.6, fontweight="bold", va="top")

    # Panel B: oracle gain decomposition, averaged over holdouts.
    sel = pd.read_csv("outputs/tables/tableS11_selection_audit.csv")
    sel = sel[sel.kind == "variants"]
    dec = (sel.groupby("tier")
           .agg(total=("naive_vs_equal", "mean"),
                split=("honest_gain", "mean"),
                gap=("oracle_gap", "mean")))
    order = ["A", "B"]
    y = np.arange(len(order))[::-1]
    h = 0.52
    for yi, tier in zip(y, order):
        r = dec.loc[tier]
        c = TIER_COLS[tier]
        ax3.barh(yi, r.split, height=h, color=c, lw=0)
        ax3.barh(yi, r.gap, height=h, left=r.split, color=GAP_COL, lw=0)
        frac = 100 * r.split / r.total
        if r.split >= 0.03:
            ax3.text(r.split / 2, yi, f"{r.split:.3f}", ha="center",
                     va="center", fontsize=6, color="white")
        else:
            ax3.text(r.split + 0.004, yi, f"{r.split:.3f}", ha="left",
                     va="center", fontsize=6, color="0.25")
        ax3.text(r.total + 0.004, yi, f"{r.total:.3f}  ({frac:.0f}% kept)",
                 ha="left", va="center", fontsize=6.2, color="0.25")
    ax3.axvline(0, color="0.2", lw=0.8)
    ax3.set_yticks(y)
    ax3.set_yticklabels(["Tier A\n(4 lines)", "Tier B\n(5 lines)"],
                        fontsize=6.6)
    for lab, tier in zip(ax3.get_yticklabels(), order):
        lab.set_color(TIER_COLS[tier])
    ax3.set_ylim(-0.65, len(order) - 0.35)
    ax3.set_xlim(0, 0.175)
    ax3.set_xlabel("mean PDS gain over equal mix", fontsize=6.8)
    ax3.grid(axis="x", color="0.92", lw=0.5, zorder=0)
    ax3.set_axisbelow(True)
    ax3.tick_params(axis="y", length=0)
    handles = [
        Patch(color=TIER_COLS["A"], label="split-half gain (fixed rule)"),
        Patch(color=GAP_COL, label="oracle gap (not retained by a fixed rule)"),
    ]
    ax3.legend(handles=handles, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.6,
               handletextpad=0.3, columnspacing=0.9)
    fig.text(0.625, 0.955, "B", fontsize=10, fontweight="bold", va="top")
    fig.text(0.658, 0.958, "Per-target oracle gains rarely survive split-half",
             fontsize=7.6, fontweight="bold", va="top")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"graphical_abstract.{ext}", bbox_inches="tight",
                    pad_inches=0.03)
    plt.close(fig)
    print("wrote figures/graphical_abstract.pdf/.png")
    print(dec.to_string())


if __name__ == "__main__":
    main()
