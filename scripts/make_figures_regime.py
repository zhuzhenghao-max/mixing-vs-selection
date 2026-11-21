"""Regime-boundary figure for the cross-line transfer paper.

Panels contrast the Nadig Tier A weak-transfer regime with the Tahoe-100M
saturated drug-response regime: per-query PDS, source saturation,
calibration gain, source-side cross-validation, and the pilot diagnostic.

Usage: python scripts/make_figures_regime.py [selectX.csv]
Writes figures/fig8_regime_boundary.pdf + .png.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

TAHOE = ["A549", "A498", "HS-578T", "HEC-1-A", "J82", "MIA_PaCa-2",
         "SK-MEL-2"]
NADIG = ["k562", "rpe1", "hepg2", "jurkat"]
DISP = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat",
        "A549": "A549", "A498": "A498", "HS-578T": "HS-578T",
        "HEC-1-A": "HEC-1-A", "J82": "J82", "MIA_PaCa-2": "MIA PaCa-2",
        "SK-MEL-2": "SK-MEL-2"}

WEAK = "#D55E00"
STRONG = "#0072B2"

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


def plabel(ax, letter, x=-0.01, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def load_pilot():
    z = np.load("outputs/pilot_x.npz")
    n = np.load("outputs/pilot_pds_nadig_noexcl.npz")
    c = np.load("outputs/pilot_pds_nadig_corrected.npz")
    tahoe = {}
    for f in z.files:
        name, kind = f.rsplit(" ", 1)
        if kind == "full":
            tahoe[name] = z[f]
    # No-exclusion scores have no per-reference division and are unchanged;
    # the exclusion scores predate the normalization fix, so use the rerun.
    noexcl = {L: n[L + "_noexcl"] for L in NADIG}
    excl = {L: c[L + "_excl"] for L in NADIG}
    return tahoe, noexcl, excl


def tahoe_saturation():
    rows = list(csv.DictReader(open("outputs/cacheX.csv")))
    tab = defaultdict(list)
    for r in rows:
        s = int(r["size"])
        if s:
            tab[(r["line"], s)].append(float(r["pds"]))
    per = {L: [np.mean(tab[(L, s)]) for s in range(1, 7)] for L in TAHOE}
    mean = [np.mean([per[L][i] for L in TAHOE]) for i in range(6)]
    return per, mean


def nadig_saturation():
    rows = list(csv.DictReader(open("outputs/nadig_saturation_noexcl.csv")))
    tab = defaultdict(list)
    for r in rows:
        tab[(r["line"], int(r["size"]))].append(float(r["pds"]))
    per = {L: [np.mean(tab[(L, s)]) for s in (1, 2, 3)] for L in NADIG}
    mean = [np.mean([per[L][i] for L in NADIG]) for i in range(3)]
    return per, mean


def calibration(path="outputs/selectX.csv"):
    ks = [20, 50, 100, 200]
    out = {}
    rows = list(csv.DictReader(open(path)))
    for strat in ("kmeans", "random"):
        per = {}
        for k in ks:
            vals = []
            for L in TAHOE:
                v = [float(r["delta"]) for r in rows
                     if r["holdout"] == L and r["strategy"] == strat
                     and int(r["k"]) == k]
                if v:
                    vals.append(np.mean(v))
            per[k] = vals
        out[("Tahoe", strat)] = per
    rows = list(csv.DictReader(open("outputs/phase2_fullmix_calibration.csv")))
    for strat in ("kmeans", "random"):
        per = {}
        for k in ks:
            per[k] = [float(r[strat]) for r in rows if int(r["k"]) == k]
        out[("Nadig", strat)] = per
    return ks, out


def sscv(path):
    rows = list(csv.DictReader(open(path)))
    a = np.array([float(r["actual"]) for r in rows])
    p = np.array([float(r["pseudo"]) for r in rows])
    return p, a, spearmanr(p, a)


def draw_ecdf(ax, v, color, thick=False):
    v = np.sort(v)
    xs = np.linspace(0.3, 1.0, 400)
    ys = np.searchsorted(v, xs, side="right") / len(v)
    ax.plot(xs, ys, color=color, lw=1.7 if thick else 0.6,
            alpha=1.0 if thick else 0.35, zorder=4 if thick else 2)


def _top_range(arrays):
    f = [100 * float((a >= 1.0).mean()) for a in arrays]
    return min(f), max(f)


def panel_distribution(ax, tahoe, nadig):
    for L in NADIG:
        draw_ecdf(ax, nadig[L], WEAK)
    draw_ecdf(ax, np.concatenate([nadig[L] for L in NADIG]), WEAK, True)
    for L in TAHOE:
        draw_ecdf(ax, tahoe[L], STRONG)
    draw_ecdf(ax, np.concatenate([tahoe[L] for L in TAHOE]), STRONG, True)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("equal-mix PDS")
    ax.set_ylabel("cumulative fraction")
    ax.text(0.04, 0.30, "Nadig (weak)", color=WEAK, fontsize=6.6,
            transform=ax.transAxes)
    ax.text(0.04, 0.21, "Tahoe (saturated)", color=STRONG, fontsize=6.6,
            transform=ax.transAxes)
    wn, wx = _top_range([nadig[L] for L in NADIG])
    tn, tx = _top_range([tahoe[L] for L in TAHOE])
    ax.text(0.04, 0.06,
            f"unique-top fraction: {wn:.0f}–{wx:.0f}% vs {tn:.0f}–{tx:.0f}%",
            transform=ax.transAxes, fontsize=5.8, color="0.35")
    ax.grid(color="0.93", lw=0.4, zorder=0)
    plabel(ax, "A")


def panel_saturation(ax, tper, tmean, nper, nmean):
    for L in TAHOE:
        ax.plot(range(1, 7), tper[L], color=STRONG, lw=0.6, alpha=0.35,
                zorder=2)
    ax.plot(range(1, 7), tmean, color=STRONG, lw=1.6, marker="o", ms=3.4,
            zorder=4)
    for L in NADIG:
        ax.plot(range(1, 4), nper[L], color=WEAK, lw=0.6, alpha=0.35,
                zorder=2)
    ax.plot(range(1, 4), nmean, color=WEAK, lw=1.6, marker="s", ms=3.2,
            zorder=4)
    ax.set_xticks(range(1, 7))
    ax.set_xlim(0.8, 6.2)
    ax.set_xlabel("source lines")
    ax.set_ylabel("mean PDS")
    gain_n = nmean[-1] - nmean[0]
    gain_t = tmean[-1] - tmean[0]
    ax.text(0.97, 0.52, f"+{gain_n:.3f} (1→3)", color=WEAK, fontsize=6.2,
            ha="right", transform=ax.transAxes)
    ax.text(0.97, 0.28, f"+{gain_t:.3f} (1→6)", color=STRONG, fontsize=6.2,
            ha="right", transform=ax.transAxes)
    ax.grid(axis="y", color="0.93", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    plabel(ax, "B")


def panel_calibration(ax, ks, cal):
    styles = {"kmeans": "-", "random": "--"}
    for regime, color in (("Nadig", WEAK), ("Tahoe", STRONG)):
        for strat in ("kmeans", "random"):
            per = cal[(regime, strat)]
            m = [np.nanmean(per[k]) for k in ks]
            s = [np.nanstd(per[k]) for k in ks]
            ax.errorbar(ks, m, yerr=s, color=color, ls=styles[strat],
                        lw=1.2, marker="o" if strat == "kmeans" else "s",
                        ms=3.2, capsize=1.5, elinewidth=0.6, zorder=3)
    ax.axhline(0, color="0.2", lw=0.7, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("calibration perturbations k")
    ax.set_ylabel("Δ mean PDS vs equal mix")
    ax.grid(axis="y", color="0.93", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], color=WEAK, lw=1.2, label="Nadig (weak)"),
        Line2D([], [], color=STRONG, lw=1.2, label="Tahoe (saturated)"),
        Line2D([], [], color="0.35", lw=1.0, ls="-", label="kmeans"),
        Line2D([], [], color="0.35", lw=1.0, ls="--", label="random"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.7, ncol=2,
              loc="lower left", handletextpad=0.3, columnspacing=0.8)
    plabel(ax, "C")


def panel_sscv(ax, p, a, res, color, letter, xlim, ylim):
    ax.scatter(p, a, s=9 if letter == "E" else 14, color=color,
               alpha=0.55 if letter == "E" else 0.9, edgecolors="none",
               zorder=3)
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.7, ls="--", zorder=1)
    ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    ax.axvline(0, color="0.85", lw=0.5, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("pseudo holdout gain")
    ax.set_ylabel("actual source-add gain")
    ax.text(0.04, 0.95,
            f"Spearman ρ = {res.statistic:+.2f}", transform=ax.transAxes,
            va="top", fontsize=6.3, color="0.3")
    ax.grid(color="0.94", lw=0.4, zorder=0)
    plabel(ax, letter)


def panel_pilot(ax, tahoe, nadig, k=50, reps=50000, seed=0):
    """Nadig uses row exclusion (benchmark policy); Tahoe has no target
    coordinate to remove."""
    rng = np.random.default_rng(seed)
    stats = []
    for L in NADIG:
        v = nadig[L]
        stats.append(v[rng.integers(0, len(v), (reps, k))].mean(axis=1))
    for L in TAHOE:
        v = tahoe[L]
        stats.append(v[rng.integers(0, len(v), (reps, k))].mean(axis=1))
    pos = list(range(1, 5)) + list(range(6, 13))
    bp = ax.boxplot(stats, positions=pos, widths=0.62, patch_artist=True,
                    showfliers=False, whis=(5, 95),
                    medianprops=dict(lw=1.0, color="0.15"),
                    whiskerprops=dict(lw=0.6, color="0.4"),
                    capprops=dict(lw=0.6, color="0.4"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(WEAK if i < len(NADIG) else STRONG)
        patch.set_alpha(0.35)
        patch.set_edgecolor(WEAK if i < len(NADIG) else STRONG)
        patch.set_linewidth(0.7)
    ax.axhline(0.92, color="0.25", lw=0.8, ls="--", zorder=1)
    ax.text(0.99, 1.015, "dashed: post-hoc 0.92 threshold",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.6, color="0.35")
    ax.axvline(5, color="0.8", lw=0.6, zorder=0)
    ax.set_xticks(pos)
    ax.set_xticklabels([DISP[L] for L in NADIG + TAHOE], rotation=40,
                       ha="right", fontsize=5.6)
    lo = min(np.percentile(s, 1) for s in stats) - 0.04
    ax.set_ylim(lo, 1.02)
    ax.set_xlim(0.4, 12.7)
    ax.set_ylabel("full-pool sampled mean PDS (k = 50)")
    ax.grid(axis="y", color="0.93", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    plabel(ax, "F")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight",
                    pad_inches=0.03)
    plt.close(fig)
    print(f"saved figures/{stem}.pdf/.png")


def main():
    tahoe, nadig_noexcl, nadig_excl = load_pilot()
    tper, tmean = tahoe_saturation()
    nper, nmean = nadig_saturation()
    cal_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/selectX.csv"
    ks, cal = calibration(cal_path)
    pn, an, rn = sscv("outputs/phase2_exante_allocator.csv")
    pt, at, rt = sscv("outputs/sscvX.csv")

    fig = plt.figure(figsize=(7.2, 4.85))
    gs = fig.add_gridspec(2, 3, hspace=0.56, wspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    panel_distribution(ax_a, tahoe, nadig_noexcl)
    panel_saturation(ax_b, tper, tmean, nper, nmean)
    panel_calibration(ax_c, ks, cal)
    panel_sscv(ax_d, pn, an, rn, WEAK, "D", (-0.05, 0.06), (-0.05, 0.06))
    panel_sscv(ax_e, pt, at, rt, STRONG, "E", (-0.001, 0.025),
               (-0.005, 0.019))
    panel_pilot(ax_f, tahoe, nadig_excl)
    save(fig, "fig8_regime_boundary")


if __name__ == "__main__":
    main()
