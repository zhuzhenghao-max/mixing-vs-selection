"""Source-response coherence diagnostic figure (Figure 10).

(A) Within-holdout Spearman between per-target source coherence and equal-mix
PDS, raw and controlling response strength.  (B) Holdout-mean coherence versus
holdout-mean PDS across the nine genetic holdouts, plus the four Tier A means
under no-exclusion scoring and the seven drug-panel lines under full-pool
scoring.  (C) Mechanism: coherence versus alignment of the equal mix with the
held-out truth, binned within tier.  (D) Negative control: cross-line
control-profile similarity versus PDS.

Usage: python scripts/make_figure_coherence.py
Writes figures/fig10_coherence_diagnostic.pdf + .png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
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


def _overlap_area(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(w, 0.0) * max(h, 0.0)


def place_labels(fig, ax, items, markers, extra=(), fontsize=6, pad=1.5):
    """Place point labels at the first candidate offset that clears other
    markers, labels, the legend, and pre-existing text."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    obstacles = []
    for x, y, ms in markers:
        cx, cy = ax.transData.transform((x, y))
        r = 0.5 * ms * px + pad
        obstacles.append(Bbox.from_extents(cx - r, cy - r, cx + r, cy + r))
    if ax.get_legend() is not None:
        obstacles.append(ax.get_legend().get_window_extent(rend))
    obstacles += [t.get_window_extent(rend) for t in extra]
    axbox = ax.get_window_extent(rend)
    cands = [
        (4.5, 0, "left", "center"), (-4.5, 0, "right", "center"),
        (0, 4.5, "center", "bottom"), (0, -4.5, "center", "top"),
        (4, 4, "left", "bottom"), (-4, 4, "right", "bottom"),
        (4, -4, "left", "top"), (-4, -4, "right", "top"),
        (9.5, 0, "left", "center"), (-9.5, 0, "right", "center"),
        (0, 9.5, "center", "bottom"), (0, -9.5, "center", "top"),
        (10, 8, "left", "bottom"), (-10, 8, "right", "bottom"),
        (10, -8, "left", "top"), (-10, -8, "right", "top"),
    ]
    placed = []
    for lab, x, y, color in items:
        best, best_pen = None, None
        for dx, dy, ha, va in cands:
            t = ax.annotate(lab, (x, y), xytext=(dx, dy),
                            textcoords="offset points", fontsize=fontsize,
                            color=color, ha=ha, va=va)
            bb = t.get_window_extent(rend)
            pen = 0.0
            if (bb.x0 < axbox.x0 + 1 or bb.x1 > axbox.x1 - 1
                    or bb.y0 < axbox.y0 + 1 or bb.y1 > axbox.y1 - 1):
                pen += 1e6
            for ob in obstacles:
                pen += _overlap_area(bb, ob)
            for ob in placed:
                pen += 3.0 * _overlap_area(bb, ob)
            if best_pen is None or pen < best_pen:
                if best is not None:
                    best.remove()
                best, best_pen = t, pen
            else:
                t.remove()
            if pen == 0:
                break
        placed.append(best.get_window_extent(rend))


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
    markers = []
    for t in ("A", "B"):
        g = s[s.tier == t]
        ax.plot(g.pair_cos, g.pds, "o", color=TIER_COLS[t], ms=4.6,
                label=f"Tier {t}")
        markers += [(x, y, 4.6) for x, y in zip(g.pair_cos, g.pds)]
    rr = spearmanr(s.pair_cos, s.pds)
    xs = np.linspace(s.pair_cos.min(), s.pair_cos.max(), 20)
    ax.plot(xs, np.polyval(np.polyfit(s.pair_cos, s.pds, 1), xs), "--",
            color="0.5", lw=0.7)
    ga = s[(s.tier == "A") & s.pds_noexcl.notna()]
    ax.plot(ga.pair_cos, ga.pds_noexcl, "^", mfc="white", mec=TIER_COLS["A"],
            mew=0.9, ms=4.6, label="Tier A, no exclusion")
    markers += [(x, y, 4.6) for x, y in zip(ga.pair_cos, ga.pds_noexcl)]
    th = pd.read_csv("outputs/tahoe_coherence_tail.csv")
    th = th.groupby("holdout")[["pair_cos", "pds_full"]].mean()
    ax.plot(th.pair_cos, th.pds_full, "s", color="#C0504D", ms=4.3,
            label="Tahoe drug panel")
    markers += [(x, y, 4.3) for x, y in zip(th.pair_cos, th.pds_full)]
    rho = ax.text(0.02, 0.76,
                  f"$\\rho$ = {rr.statistic:+.2f}  (p = {rr.pvalue:.3f})",
                  transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("Mean source-response coherence")
    ax.set_ylabel("Mean equal-mix PDS")
    ax.legend(loc="lower right", frameon=False)
    items = [(DISP[r.holdout], r.pair_cos, r.pds, "black")
             for r in s.itertuples()]
    items.append(("Tahoe", th.pair_cos.mean(), th.pds_full.min(),
                  "#C0504D"))
    place_labels(fig, ax, items, markers, extra=[rho])
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
    off_d = {"hek293t": (5, 4)}
    for r in cs.itertuples():
        ax.annotate(DISP[r.holdout], (r.src_hold, r.pds),
                    xytext=off_d.get(r.holdout, (3, -7)),
                    textcoords="offset points", fontsize=6)
    rr = spearmanr(cs.src_hold, cs.pds)
    ax.text(0.46, 0.84,
            f"$\\rho$ = {rr.statistic:+.2f}  (p = {rr.pvalue:.2f}, n = 5)",
            transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("Control-profile similarity to sources")
    ax.set_ylabel("Mean equal-mix PDS")
    plabel(ax, "D")

    fig.savefig(OUT / "fig10_coherence_diagnostic.pdf")
    fig.savefig(OUT / "fig10_coherence_diagnostic.png")
    print("wrote", OUT / "fig10_coherence_diagnostic.png")


if __name__ == "__main__":
    main()
