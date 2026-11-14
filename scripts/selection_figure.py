"""Figure: per-target selection audit across holdouts and regimes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

TAB = pd.read_csv("outputs/tables/tableS11_selection_audit.csv")
KC = pd.read_csv("outputs/selection_audit_kcurve.csv")

DISP = {"h1": "H1", "k562": "K562", "rpe1": "RPE1", "hct116": "HCT116",
        "hek293t": "HEK293T", "hepg2": "HepG2", "jurkat": "Jurkat",
        "A498": "A498", "A549": "A549", "HEC-1-A": "HEC-1-A",
        "HS-578T": "HS-578T", "J82": "J82", "MIA_PaCa-2": "MIA-PaCa-2",
        "SK-MEL-2": "SK-MEL-2"}

ORDER = {"A": ["k562", "rpe1", "hepg2", "jurkat"],
         "B": ["h1", "k562", "rpe1", "hct116", "hek293t"]}

TIER_COLS = {"A": "#2A9D8F", "B": "#5C6B8A"}
COMP_COL = "#0072B2"
WITH_COL = "#E69F00"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def holdout_rows(kind):
    rows = []
    for tier, names in ORDER.items():
        sub = TAB[(TAB.kind == kind) & (TAB.tier == tier)]
        sub = sub.set_index("holdout").loc[names].reset_index()
        for _, r in sub.iterrows():
            rows.append((tier, r))
    return rows


def panel_letter(ax, letter, x=-0.16, y=1.10):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


fig = plt.figure(figsize=(7.2, 5.1))
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.50,
                      left=0.13, right=0.98, top=0.88, bottom=0.10)

# A: oracle gap vs honest gain per holdout
axA = fig.add_subplot(gs[0, 0])
rows = holdout_rows("variants")
y = np.arange(len(rows))[::-1]
gap = np.array([r["oracle_gap"] for _, r in rows], dtype=float)
hon = np.array([r["honest_gain"] for _, r in rows], dtype=float)
cols = [TIER_COLS[t] for t, _ in rows]
axA.barh(y, gap, height=0.62, color=cols, alpha=0.85, lw=0)
axA.scatter(hon, y, s=16, color="black", zorder=3, label="honest split − equal")
axA.axvline(0, color="black", lw=0.6)
axA.set_yticks(y)
axA.set_yticklabels([DISP[r["holdout"]] for _, r in rows])
for lab, (tier, _) in zip(axA.get_yticklabels(), rows):
    lab.set_color(TIER_COLS[tier])
axA.set_xlabel("PDS difference")
axA.set_xlim(-0.012, 0.135)
split = len(ORDER["A"]) - 0.5
axA.axhline(split, color="#BBBBBB", lw=0.6, ls=":")
panel_letter(axA, "A")

# B/C: k-curves
kg = KC.groupby(["tier", "k"]).agg(
    naive=("naive_vs_equal", "mean"),
    fixed=("best_fixed_vs_equal", "mean"),
    honest=("honest_gain", "mean"),
).reset_index()

for ax, tier, letter in ((fig.add_subplot(gs[0, 1]), "A", "B"),
                         (fig.add_subplot(gs[1, 0]), "B", "C")):
    d = kg[kg.tier == tier]
    ax.plot(d.k, d.naive, "-o", color=TIER_COLS[tier], ms=3, lw=1.0,
            label="per-target oracle − equal")
    ax.plot(d.k, d.fixed, ":", color=TIER_COLS[tier], lw=1.0,
            label="best fixed rule − equal")
    ax.plot(d.k, d.honest, "--s", color="#444444", ms=2.5, lw=0.9,
            label="honest split − equal")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("candidates in pool (k)")
    ax.set_ylabel("PDS difference vs equal")
    ax.set_xticks(sorted(d.k.unique()))
    ax.set_ylim(-0.006, 0.025)
    panel_letter(ax, letter)

axB = fig.axes[1]
axB.legend(loc="upper left", frameon=False)

# D: gate decomposition
axD = fig.add_subplot(gs[1, 1])
grows = holdout_rows("gate")
y = np.arange(len(grows))[::-1]
comp = np.array([r["composition"] for _, r in grows], dtype=float)
within = np.array([r["within"] for _, r in grows], dtype=float)
axD.barh(y + 0.19, comp, height=0.36, color=COMP_COL, lw=0,
         label="composition")
axD.barh(y - 0.19, within, height=0.36, color=WITH_COL, lw=0,
         label="within subset")
axD.axvline(0, color="black", lw=0.6)
axD.set_yticks(y)
axD.set_yticklabels([DISP[r["holdout"]] for _, r in grows])
for lab, (tier, _) in zip(axD.get_yticklabels(), grows):
    lab.set_color(TIER_COLS[tier])
axD.set_xlabel("PDS change")
split = len(ORDER["A"]) - 0.5
axD.axhline(split, color="#BBBBBB", lw=0.6, ls=":")
panel_letter(axD, "D")

handles = [
    plt.Line2D([], [], color=TIER_COLS["A"], lw=4, label="oracle − honest"),
    plt.Line2D([], [], color="black", marker="o", lw=0, ms=3,
               label="honest split − equal"),
    Patch(color=COMP_COL, label="gate: composition"),
    Patch(color=WITH_COL, label="gate: within subset"),
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.995),
           ncol=4, frameon=False)

fig.savefig(OUT / "fig10_selection_audit.png")
fig.savefig(OUT / "fig10_selection_audit.pdf")
print("wrote figures/fig10_selection_audit.png/.pdf")
