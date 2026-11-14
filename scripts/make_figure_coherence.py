"""Source-response coherence diagnostic figure (Figure 11).

(A) Within-holdout Spearman between per-target source coherence and equal-mix
PDS, raw and controlling response strength.  (B) Holdout-mean coherence versus
holdout-mean PDS across the nine genetic holdouts, plus the four Tier A means
under no-exclusion scoring and the seven drug-panel lines under full-pool
scoring.  (C) Mechanism: coherence versus alignment of the equal mix with the
held-out truth, binned within tier.  (D) Negative control: cross-line
control-profile similarity versus PDS.

Usage: python scripts/make_figure_coherence.py
Writes figures/fig11_coherence_diagnostic.pdf + .png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

TIER_COLS = {"A": "#2A9D8F", "B": "#5C6B8A"}
ORDER = [("A", "k562"), ("A", "rpe1"), ("A", "hepg2"), ("A", "jurkat"),
         ("B", "h1"), ("B", "k562"), ("B", "rpe1"), ("B", "hct116"),
         ("B", "hek293t")]
DISP = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat",
        "h1": "H1", "hct116": "HCT116", "hek293t": "HEK293T"}

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
})


def plabel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def main():
    d = pd.read_csv("outputs/regime_diagnostic_targets.csv")
    s = pd.read_csv("outputs/regime_diagnostic_holdouts.csv")
    s["key"] = list(zip(s.tier, s.holdout))
    tab = {(r.tier, r.holdout): r for r in s.itertuples()}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.09,
                        wspace=0.42, hspace=0.42)

    ax = axes[0, 0]
    for i, key in enumerate(ORDER):
        r = tab[key]
        c = TIER_COLS[key[0]]
        ax.plot([r.prho_pair_cos, r.rho_pair_cos], [i, i], "-", color=c,
                lw=0.7, alpha=0.55, zorder=1)
        ax.plot(r.rho_pair_cos, i, "o", color=c, ms=4.4, zorder=2)
        ax.plot(r.prho_pair_cos, i, "o", mfc="white", mec=c, mew=0.9, ms=4.0,
                zorder=2)
    ax.axvline(0, color="0.6", lw=0.6, ls=":")
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([f"{DISP[h]} ({t})" for t, h in ORDER])
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_xlabel("Spearman $\\rho$ (coherence vs PDS)")
    ax.plot([], [], "o", color="0.3", ms=4.4, label="raw")
    ax.plot([], [], "o", mfc="white", mec="0.3", mew=0.9, ms=4.0,
            label="controlling strength")
    ax.legend(loc="lower right", frameon=False)
    plabel(ax, "A")

    ax = axes[0, 1]
    off = {("A", "hepg2"): (4, -9), ("A", "k562"): (4, 3)}
    for t in ("A", "B"):
        g = s[s.tier == t]
        ax.plot(g.pair_cos, g.pds, "o", color=TIER_COLS[t], ms=4.6,
                label=f"Tier {t}")
        for r in g.itertuples():
            ax.annotate(DISP[r.holdout], (r.pair_cos, r.pds),
                        xytext=off.get((r.tier, r.holdout), (3, 3)),
                        textcoords="offset points", fontsize=6)
    rr = spearmanr(s.pair_cos, s.pds)
    xs = np.linspace(s.pair_cos.min(), s.pair_cos.max(), 20)
    ax.plot(xs, np.polyval(np.polyfit(s.pair_cos, s.pds, 1), xs), "--",
            color="0.5", lw=0.7)
    ga = s[(s.tier == "A") & s.pds_noexcl.notna()]
    ax.plot(ga.pair_cos, ga.pds_noexcl, "^", mfc="white", mec=TIER_COLS["A"],
            mew=0.9, ms=4.6, label="Tier A, no exclusion")
    th = pd.read_csv("outputs/tahoe_coherence_tail.csv")
    th = th.groupby("holdout")[["pair_cos", "pds_full"]].mean()
    ax.plot(th.pair_cos, th.pds_full, "s", color="#C0504D", ms=4.3,
            label="Tahoe drug panel")
    ax.annotate("Tahoe", (th.pair_cos.mean(), th.pds_full.min()),
                xytext=(4, -10), textcoords="offset points", fontsize=6,
                color="#C0504D")
    ax.text(0.02, 0.76, f"$\\rho$ = {rr.statistic:+.2f}  (p = {rr.pvalue:.3f})",
            transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("Mean source-response coherence")
    ax.set_ylabel("Mean equal-mix PDS")
    ax.legend(loc="lower right", frameon=False)
    plabel(ax, "B")

    ax = axes[1, 0]
    for t in ("A", "B"):
        g = d[d.tier == t]
        q = pd.qcut(g.pair_cos, 8, labels=False, duplicates="drop")
        bx, by, be = [], [], []
        for b in np.unique(q):
            h = g[q == b]
            bx.append(h.pair_cos.mean())
            by.append(h.mix_truth_cos.mean())
            be.append(h.mix_truth_cos.std() / np.sqrt(len(h)))
        ax.errorbar(bx, by, yerr=be, fmt="o-", color=TIER_COLS[t], ms=3.6,
                    lw=1.0, capsize=1.5, elinewidth=0.7,
                    label=f"Tier {t}")
    zz = []
    for _, g in d.groupby(["tier", "holdout"]):
        zz.append(np.arctanh(np.clip(spearmanr(g.pair_cos,
                                               g.mix_truth_cos).statistic,
                                     -0.999, 0.999)))
    ax.text(0.03, 0.92, f"mean within-holdout $\\rho$ = {np.tanh(np.mean(zz)):+.2f}",
            transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("Source-response coherence")
    ax.set_ylabel("Mix-truth cosine")
    ax.legend(loc="lower right", frameon=False)
    plabel(ax, "C")

    ax = axes[1, 1]
    cs = pd.read_csv("outputs/control_similarity_tierB.csv")
    ax.plot(cs.src_hold, cs.pds, "o", color=TIER_COLS["B"], ms=4.6)
    for r in cs.itertuples():
        ax.annotate(DISP[r.holdout], (r.src_hold, r.pds), xytext=(3, -7),
                    textcoords="offset points", fontsize=6)
    rr = spearmanr(cs.src_hold, cs.pds)
    ax.text(0.46, 0.84,
            f"$\\rho$ = {rr.statistic:+.2f}  (p = {rr.pvalue:.2f}, n = 5)",
            transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("Control-profile similarity to sources")
    ax.set_ylabel("Mean equal-mix PDS")
    plabel(ax, "D")

    fig.savefig(OUT / "fig11_coherence_diagnostic.pdf")
    fig.savefig(OUT / "fig11_coherence_diagnostic.png")
    print("wrote", OUT / "fig11_coherence_diagnostic.png")


if __name__ == "__main__":
    main()
