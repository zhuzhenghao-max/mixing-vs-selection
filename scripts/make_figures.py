"""Generate manuscript figures from the canonical v2 benchmark artifacts.

Usage: python scripts/make_figures.py
Writes figures/fig1..fig6 as PDF + PNG.

House style: Arial, 7-8 pt labels, bold panel letters, no in-figure titles,
muted colorblind-safe palettes, forest/dumbbell plots with zero lines and
95% bootstrap confidence intervals.
"""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.stats import spearmanr, wilcoxon

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

TIERS = {
    "A": dict(
        title="Tier A",
        npz="outputs/bench_transfer_nadig_v2.npz",
        triples="outputs/bench_transfer_nadig_v2_triples.csv",
        log="outputs/bench_transfer_nadig_v2.log",
        lines=["k562", "rpe1", "hepg2", "jurkat"],
    ),
    "B": dict(
        title="Tier B",
        npz="outputs/bench_transfer_v2.npz",
        triples="outputs/bench_transfer_v2_triples.csv",
        log="outputs/bench_transfer_v2.log",
        lines=["h1", "k562", "rpe1", "hct116", "hek293t"],
    ),
}

DISP = {
    "h1": "H1", "k562": "K562", "rpe1": "RPE1", "hct116": "HCT116",
    "hek293t": "HEK293T", "hepg2": "HepG2", "jurkat": "Jurkat",
}

TIER_COLS = {"A": "#2A9D8F", "B": "#5C6B8A"}

LINE_COLS = {
    "h1": "#7E57C2", "k562": "#0072B2", "rpe1": "#009E73",
    "hct116": "#D55E00", "hek293t": "#56B4E9", "hepg2": "#CC79A7",
    "jurkat": "#E69F00",
}

METHOD_COLS = {
    "mean": "#8C8C8C", "equal": "#1A1A1A", "equal_center": "#2A9D8F",
    "gated_equal": "#E69F00", "top2_selfw": "#7E57C2",
    "equal_raw": "#B0B0B0", "simT0.2": "#56B4E9",
}

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


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    D = {}
    for h, m, t, p in zip(z["holdout"].astype(str), z["method"].astype(str),
                          z["target"].astype(str),
                          z["pds_raw"].astype(np.float64)):
        D.setdefault((h, m), {})[t] = float(p)
    return D


def load_triples(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pool_sizes(log_path):
    out = {}
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    for h, n in re.findall(r"=== hold-out (\w+): pool (\d+)", text):
        out[h] = int(n)
    return out


def boot_mean_ci(v, rng, n=4000):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    bs = rng.choice(v, (n, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), int(len(v))


def paired_delta(D, h, a, b, rng, nboot=4000):
    da, db = D.get((h, a)), D.get((h, b))
    if not da or not db:
        return None
    common = sorted(set(da) & set(db))
    d = np.array([da[t] - db[t] for t in common], dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return None
    bs = rng.choice(d, (nboot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return dict(n=len(d), mean=float(d.mean()), median=float(np.median(d)),
                lo=float(lo), hi=float(hi), p=p,
                win=float(np.mean(d > 0)))


def best_single_values(D, h):
    singles = {m: d for (hh, m), d in D.items()
               if hh == h and m.startswith("single_")}
    if not singles:
        return []
    m = max(singles, key=lambda mm: np.nanmean(list(singles[mm].values())))
    return list(singles[m].values())


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight",
                    pad_inches=0.03)
    plt.close(fig)
    print(f"saved figures/{stem}.pdf/.png")


def plabel(ax, letter, x=-0.01, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def parse_variants(path):
    out = {}
    cur = None
    for line in Path(path).read_text(encoding="utf-8",
                                     errors="ignore").splitlines():
        m = re.match(r"=== (\w+):", line)
        if m:
            cur = m.group(1)
        m = re.match(r"  equal\s+med ([-\d.]+) mean ([-\d.]+)", line)
        if m and cur:
            out[cur] = (float(m.group(1)), float(m.group(2)))
    return out


def parse_centerq(path):
    out = {}
    cur = None
    for line in Path(path).read_text(encoding="utf-8",
                                     errors="ignore").splitlines():
        m = re.match(r"=== (\w+):", line)
        if m:
            cur = m.group(1)
        m = re.match(r"  (equal|equal_center|equal_center_q)\s+"
                     r"med ([-\d.]+) mean ([-\d.]+)", line)
        if m and cur:
            out.setdefault(cur, {})[m.group(1)] = float(m.group(3))
    return out


# --------------------------------------------------------------------------
# Figure 1: benchmark design


def draw_vector(ax, x0, y0, w, h, color, rng):
    vals = rng.normal(0.35, 0.33, 11)
    xs = np.linspace(x0, x0 + w, len(vals))
    ax.vlines(xs, y0, y0 + np.clip(vals, -0.12, 0.95) * h, color=color,
              lw=1.0, clip_on=False)
    ax.hlines(y0, x0, x0 + w, color="0.6", lw=0.5, clip_on=False)


def fig1_schematic(ax, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    ax.text(0.0, 3.85, "measured response vectors for target t",
            fontsize=6.6, color="0.25")
    src = ["K562", "RPE1", "HepG2"]
    for i, name in enumerate(src):
        y = 3.15 - i * 0.9
        ax.add_patch(FancyBboxPatch(
            (0.0, y), 1.55, 0.66,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            fc="#F7F7F7", ec="0.8", lw=0.6))
        ax.text(0.07, y + 0.57, name, fontsize=6.2, va="top", color="0.25")
        draw_vector(ax, 0.55, y + 0.17, 0.9, 0.3, LINE_COLS[name.lower()],
                    rng)
    ax.text(0.0, 0.18,
            "target t is measured in at least one source\n"
            "and in the held-out line",
            fontsize=5.9, color="0.35", va="bottom")
    ax.annotate("", xy=(2.0, 2.25), xytext=(1.62, 2.25),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.35",
                                shrinkA=0, shrinkB=0))
    ax.add_patch(FancyBboxPatch(
        (2.05, 1.35), 1.5, 1.7,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        fc="#EDF5F3", ec=TIER_COLS["A"], lw=0.7))
    ax.text(2.8, 2.78, "unit-normalize", fontsize=6.1, ha="center")
    ax.text(2.8, 2.45, "each source", fontsize=6.1, ha="center")
    ax.text(2.8, 1.98, "average", fontsize=6.1, ha="center")
    ax.text(2.8, 1.55, "ŷ", fontsize=9, ha="center")
    ax.annotate("", xy=(4.05, 2.25), xytext=(3.6, 2.25),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.35",
                                shrinkA=0, shrinkB=0))
    ax.add_patch(FancyBboxPatch(
        (4.1, 0.35), 5.8, 3.1,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        fc="#FAFAFA", ec="0.75", lw=0.7))
    ax.text(4.25, 3.26, "held-out line: reference pool", fontsize=6.5,
            color="0.2")
    ax.text(4.25, 2.99, "every measured target in the line", fontsize=5.9,
            color="0.4")
    for i in range(6):
        y = 2.65 - i * 0.28
        w = 1.2 + 0.5 * ((i * 3) % 4)
        ax.add_patch(Rectangle((4.25, y), w, 0.11, fc="0.78", ec="none"))
    ax.add_patch(Rectangle((4.2, 0.95), 5.0, 0.2, fc="#FDF0E3",
                           ec="#E69F00", lw=0.6))
    ax.add_patch(Rectangle((4.25, 0.99), 2.6, 0.11, fc="#E69F00",
                           ec="none"))
    ax.text(7.0, 1.05, "true response", fontsize=5.9, ha="left",
            va="center", color="#A66400")
    ax.text(4.25, 0.62, "score = 1 − rank0 / n", fontsize=6.4)
    ax.text(4.25, 0.44, "after removing the target gene's coordinate",
            fontsize=5.8, color="0.4")


def fig1_composition(ax):
    rows = ([("A", h) for h in TIERS["A"]["lines"]]
            + [("B", h) for h in TIERS["B"]["lines"]])
    pools = {t: pool_sizes(TIERS[t]["log"]) for t in ("A", "B")}
    Ds = {t: load_npz(TIERS[t]["npz"]) for t in ("A", "B")}
    y = np.arange(len(rows))[::-1]
    for yi, (t, h) in zip(y, rows):
        p = pools[t].get(h, np.nan)
        q = len(Ds[t].get((h, "equal"), {}))
        ax.plot([p, q], [yi, yi], color="0.8", lw=0.8, zorder=1)
        ax.plot(p, yi, "o", mfc="white", mec=LINE_COLS[h], ms=4.4, mew=1.0,
                zorder=3)
        ax.plot(q, yi, "o", color=LINE_COLS[h], ms=4.4, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(150, 25000)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{DISP[h]} ({t})" for t, h in rows], fontsize=6.4)
    ax.axhline(len(TIERS["B"]["lines"]) - 0.5, color="0.7", lw=0.6,
               ls="--", zorder=0)
    ax.set_xticks([200, 1000, 10000])
    ax.set_xticklabels(["200", "1,000", "10,000"])
    ax.minorticks_off()
    ax.set_xlabel("targets (log scale)")
    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec="0.4", ms=4,
               label="measured targets (pool)"),
        Line2D([], [], marker="o", ls="", mfc="0.4", mec="0.4", ms=4,
               label="evaluable queries"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.9, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.24),
              handletextpad=0.2, columnspacing=1.0)
    ax.grid(axis="x", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def overlap_heatmap(ax, tier, cbar=False):
    cfg = TIERS[tier]
    lines = cfg["lines"]
    n = len(lines)
    pool = pool_sizes(cfg["log"])
    counts = {}
    for r in load_triples(cfg["triples"]):
        counts[(r["holdout"], r["source"])] = \
            counts.get((r["holdout"], r["source"]), 0) + 1
    M = np.full((n, n), np.nan)
    for i, h in enumerate(lines):
        for j, s in enumerate(lines):
            if i != j:
                M[i, j] = counts.get((h, s), np.nan)
    im = ax.imshow(np.log10(M), cmap="Blues", vmin=0, vmax=4,
                   aspect="auto")
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="0.93",
                               edgecolor="white", lw=0.6, zorder=2))
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(M[i, j]):
                continue
            v = int(M[i, j])
            txt = f"{v / 1000:.1f}k" if v >= 1000 else f"{v}"
            col = "white" if np.log10(v) > 2.45 else "0.2"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.9,
                    color=col, zorder=3)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([DISP[l] for l in lines], rotation=38, ha="right",
                       fontsize=6.2)
    ax.set_yticklabels([f"{DISP[l]} ({pool.get(l, 0):,})" for l in lines],
                       fontsize=6.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("source line")
    ax.text(0.5, 1.07, cfg["title"], transform=ax.transAxes, fontsize=7,
            color="0.25", ha="center")
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_ticks([0, 1, 2, 3, 4])
        cb.set_ticklabels(["1", "10", "100", "1k", "10k"])
        cb.set_label("shared targets", fontsize=6.3)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_visible(False)


def load_gears_deltas(path="outputs/gears_summary_A.csv"):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["holdout"]] = dict(mean=float(r["d_mean"]),
                                     lo=float(r["ci_lo"]),
                                     hi=float(r["ci_hi"]))
    return out


def fig1_headline(ax, rng):
    stats = []
    for tier in ("A", "B"):
        D = load_npz(TIERS[tier]["npz"])
        for h in TIERS[tier]["lines"]:
            st = paired_delta(D, h, "equal", "mean", rng)
            stats.append((tier, h, st))
    x = np.arange(len(stats))
    for xi, (tier, h, st) in zip(x, stats):
        c = TIER_COLS[tier]
        ax.plot([xi, xi], [st["lo"], st["hi"]], color=c, lw=1.2, zorder=3)
        ax.plot(xi, st["mean"], "o" if tier == "A" else "s", color=c,
                ms=4.4, mec="white", mew=0.5, zorder=4)
    ax.axvline(3.5, color="0.75", lw=0.6, ls="--", zorder=1)
    ax.axhline(0, color="0.2", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([DISP[h] for _, h, _ in stats], fontsize=6.4)
    ax.set_xlim(-0.6, len(stats) - 0.4)
    ax.set_ylim(0, max(st["hi"] for _, _, st in stats) + 0.05)
    for xc, tier in ((1.5, "A"), (6.0, "B")):
        ax.text(xc, 1.015, f"Tier {tier}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6.6, color=TIER_COLS[tier])
    ax.set_ylabel("equal − mean\n(mean PDS, 95% CI)")
    ax.text(0.99, 0.97, "all p < 1e-12", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.9, color="0.35")
    ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def fig1_baselines(ax, rng):
    gears = load_gears_deltas()
    z = np.load("outputs/bench_scgpt_baseline.npz", allow_pickle=True)
    rows = [(f"GEARS · {DISP[h]}", gears[h], TIER_COLS["A"], "o")
            for h in TIERS["A"]["lines"]]
    st = scgpt_paired(z, "rpe1", "scgpt_r10", "equal", rng)
    rows.append(("scGPT · RPE1", st, "#7E57C2", "D"))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, stt, c, mk) in zip(y, rows):
        ax.plot([stt["lo"], stt["hi"]], [yi, yi], color=c, lw=1.2, zorder=3)
        ax.plot(stt["mean"], yi, mk, color=c, ms=3.8, mec="white", mew=0.5,
                zorder=4)
        ax.text(stt["hi"] + 0.008, yi, f"{stt['mean']:+.2f}", va="center",
                fontsize=5.8, color="0.3")
    ax.axhline(0.5, color="0.75", lw=0.6, ls="--", zorder=1)
    ax.axvline(0, color="0.2", lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 0.315)
    ax.set_xlabel("equal mix − baseline\n(Δ mean PDS, 95% CI)", fontsize=6.8)
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def fig1_overview():
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 1.05],
                          hspace=0.5, wspace=0.42,
                          left=0.105, right=0.985, top=0.94, bottom=0.075)
    ax_s = fig.add_subplot(gs[0, :])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_a = fig.add_subplot(gs[1, 1])
    ax_b = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[2, :2])
    ax_f = fig.add_subplot(gs[2, 2])
    fig1_schematic(ax_s, rng)
    plabel(ax_s, "A", x=0.0, y=1.0)
    fig1_composition(ax_c)
    plabel(ax_c, "B")
    overlap_heatmap(ax_a, "A")
    plabel(ax_a, "C")
    overlap_heatmap(ax_b, "B", cbar=True)
    plabel(ax_b, "D")
    fig1_headline(ax_e, rng)
    plabel(ax_e, "E")
    fig1_baselines(ax_f, rng)
    plabel(ax_f, "F")
    save(fig, "fig1_benchmark_overview")


# --------------------------------------------------------------------------
# Figure 2: method comparison matrices


def method_matrices():
    rows = [
        ("mean", "Mean response"),
        ("single_best", "Best single source"),
        ("equal_raw", "Raw mix"),
        ("equal", "Equal mix"),
        ("equal_center", "Equal mix + centering"),
        ("top2_selfw", "Top-2 self-weight"),
    ]
    cols = ([("A", h) for h in TIERS["A"]["lines"]]
            + [("B", h) for h in TIERS["B"]["lines"]])
    width = len(cols) + 1
    gap = len(TIERS["A"]["lines"])
    Mm = np.full((len(rows), width), np.nan)
    Md = np.full((len(rows), width), np.nan)
    Ds = {t: load_npz(TIERS[t]["npz"]) for t in ("A", "B")}
    for j, (tier, h) in enumerate(cols):
        jj = j if j < gap else j + 1
        D = Ds[tier]
        for i, (key, _) in enumerate(rows):
            if key == "single_best":
                v = best_single_values(D, h)
            else:
                v = list(D.get((h, key), {}).values())
            v = np.array(v, dtype=np.float64)
            v = v[np.isfinite(v)]
            if len(v):
                Mm[i, jj] = v.mean()
                Md[i, jj] = np.median(v)
    labels = [""] * width
    for j, (_, h) in enumerate(cols):
        labels[j if j < gap else j + 1] = DISP[h]
    return rows, labels, Mm, Md, gap


def fig2_method_comparison():
    rows, labels, Mm, Md, gap = method_matrices()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                                     gridspec_kw=dict(wspace=0.18))
    for ax, M, cblab, letter in ((ax_a, Mm, "mean PDS", "A"),
                                 (ax_b, Md, "median PDS", "B")):
        ax.set_facecolor("white")
        im = ax.imshow(M, cmap="viridis", vmin=0.45, vmax=1.0,
                       aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isfinite(M[i, j]):
                    continue
                r, g, b, _ = im.cmap(im.norm(M[i, j]))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.8, color="white" if lum < 0.5 else "0.15")
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=6.2)
        ax.set_yticks(range(M.shape[0]))
        if letter == "A":
            ax.set_yticklabels([r[1] for r in rows], fontsize=6.3)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylim(M.shape[0] - 0.5, -1.25)
        ax.text(1.5, -0.95, "Tier A", ha="center", fontsize=6.8,
                color="0.25")
        ax.text(7.0, -0.95, "Tier B", ha="center", fontsize=6.8,
                color="0.25")
        cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
        cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        cb.set_label(cblab, fontsize=6.3)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_visible(False)
        plabel(ax, letter, x=-0.18 if letter == "A" else -0.02)
    save(fig, "fig2_method_comparison")


# --------------------------------------------------------------------------
# Figure 3: paired comparisons against equal mixing


def collect_forest(D, holds, rows, rng):
    out = []
    for meth, label in rows:
        for h in holds:
            st = paired_delta(D, h, meth, "equal", rng)
            if st is not None:
                out.append((meth, label, h, st))
    return out


def draw_forest(ax, stats, rows, holds, xlim):
    nrow = len(rows)
    for r, (meth, label) in enumerate(rows):
        yr = nrow - 1 - r
        ax.axhspan(yr - 0.42, yr + 0.42,
                   color="0.96" if r % 2 == 0 else "white", zorder=0)
        offs = np.linspace(-0.25, 0.25, len(holds))
        for h, off in zip(holds, offs):
            rec = [s for s in stats if s[0] == meth and s[2] == h]
            if not rec:
                continue
            st = rec[0][3]
            y = yr + off
            ax.plot([st["lo"], st["hi"]], [y, y], color=LINE_COLS[h],
                    lw=1.0, alpha=0.85, zorder=3)
            ax.plot(st["mean"], y, "o", color=LINE_COLS[h], ms=3.6,
                    mec="white", mew=0.5, zorder=4)
    ax.axvline(0, color="0.2", lw=0.8, zorder=2)
    ax.set_xlim(xlim)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([r[1] for r in reversed(rows)], fontsize=6.5)
    ax.set_ylim(-0.6, nrow - 0.4)
    ax.set_xlabel("Δ mean PDS vs equal mix (95% CI)")
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def fig3_paired_forest():
    rng = np.random.default_rng(1)
    rows = [("equal_center", "Centering"), ("gated_equal", "Gated mix"),
            ("equal_raw", "Raw mix"), ("top2_selfw", "Top-2 self-weight")]
    rows_b = rows + [("simT0.2", "Similarity weight")]
    DA, DB = load_npz(TIERS["A"]["npz"]), load_npz(TIERS["B"]["npz"])
    sA = collect_forest(DA, TIERS["A"]["lines"], rows, rng)
    sB = collect_forest(DB, TIERS["B"]["lines"], rows_b, rng)
    allst = sA + sB
    lo = min(s[3]["lo"] for s in allst)
    hi = max(s[3]["hi"] for s in allst)
    pad = max(0.004, 0.06 * (hi - lo))
    xlim = (lo - pad, hi + pad)

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.35, 1.0], hspace=0.5,
                          wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    draw_forest(ax_a, sA, rows, TIERS["A"]["lines"], xlim)
    draw_forest(ax_b, sB, rows_b, TIERS["B"]["lines"], xlim)
    plabel(ax_a, "A")
    plabel(ax_b, "B")

    holds = TIERS["A"]["lines"] + TIERS["B"]["lines"]
    vals, los, his = [], [], []
    for tier in ("A", "B"):
        D = load_npz(TIERS[tier]["npz"])
        for h in TIERS[tier]["lines"]:
            st = paired_delta(D, h, "equal", "mean", rng)
            vals.append(st["mean"])
            los.append(st["lo"])
            his.append(st["hi"])
    x = np.arange(len(holds))
    for xi, h, v, l, u in zip(x, holds, vals, los, his):
        ax_c.plot([xi, xi], [l, u], color=LINE_COLS[h], lw=1.0, zorder=3)
        ax_c.plot(xi, v, "o", color=LINE_COLS[h], ms=4.2, mec="white",
                  mew=0.5, zorder=4)
    ax_c.axhline(0, color="0.2", lw=0.8, zorder=2)
    ax_c.axvline(3.5, color="0.6", lw=0.7, ls="--", zorder=1)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([DISP[h] for h in holds], fontsize=6.6)
    ax_c.set_ylabel("equal − mean\n(mean PDS)")
    ax_c.set_ylim(min(los) - 0.02, max(his) + 0.03)
    ax_c.text(0.99, 0.94, "all p < 1e-12", transform=ax_c.transAxes,
              ha="right", va="top", fontsize=6.2, color="0.35")
    ax_c.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    ax_c.set_axisbelow(True)
    plabel(ax_c, "C")

    for ax, tier in ((ax_a, "A"), (ax_b, "B")):
        handles = [Line2D([], [], marker="o", ls="", color=LINE_COLS[h],
                          ms=4, label=DISP[h])
                   for h in TIERS[tier]["lines"]]
        ax.legend(handles=handles, frameon=False, fontsize=5.9, ncol=2,
                  loc="lower right", handletextpad=0.2, columnspacing=0.8)
    save(fig, "fig3_paired_deltas")


# --------------------------------------------------------------------------
# Figure 4: gate decomposition and centering


def gate_and_center_rows(rng):
    rows = []
    for tier in ("A", "B"):
        D = load_npz(TIERS[tier]["npz"])
        for h in TIERS[tier]["lines"]:
            eq = D.get((h, "equal"), {})
            ga = D.get((h, "gated_equal"), {})
            common = sorted(set(eq) & set(ga))
            if len(common) < 5:
                continue
            eq_full = np.array(list(eq.values()))
            eq_sub = np.array([eq[t] for t in common])
            ga_sub = np.array([ga[t] for t in common])
            rows.append(dict(tier=tier, h=h,
                             sel=float(np.nanmean(eq_sub) - np.nanmean(eq_full)),
                             gate=float(np.nanmean(ga_sub) - np.nanmean(eq_sub)),
                             n_g=len(common), n_f=len(eq), D=D))
    rows.sort(key=lambda r: (r["tier"] != "A", r["sel"]))
    return rows


def fig4_gate_center():
    rng = np.random.default_rng(2)
    rows = gate_and_center_rows(rng)
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.48,
                          wspace=0.42)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        ax_a.plot([r["gate"], r["sel"]], [y, y], color="0.82", lw=1.0,
                  zorder=1)
    ax_a.plot([r["sel"] for r in rows], ys, "o", color="#0072B2", ms=4.6,
              label="selection effect", zorder=3)
    ax_a.plot([r["gate"] for r in rows], ys, "o", color="#E69F00", ms=4.6,
              label="gate effect (same targets)", zorder=3)
    ax_a.axvline(0, color="0.2", lw=0.8, zorder=2)
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([f"{DISP[r['h']]} ({r['tier']})  "
                          f"{r['n_g']:,}/{r['n_f']:,}" for r in rows],
                         fontsize=6.2)
    ax_a.set_ylim(-0.6, len(rows) - 0.4)
    nA = sum(r["tier"] == "A" for r in rows)
    ax_a.axhline(len(rows) - nA - 0.5, color="0.6", lw=0.7, ls="--",
                 zorder=1)
    ax_a.set_xlabel("mean PDS change")
    ax_a.set_ylabel("held-out line (gate / all targets)")
    ax_a.legend(frameon=False, fontsize=6.2, loc="lower left")
    ax_a.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax_a.set_axisbelow(True)
    plabel(ax_a, "A")

    yb = np.arange(len(rows))[::-1]
    cvals, clos, chis = [], [], []
    for r, y in zip(rows, yb):
        st = paired_delta(r["D"], r["h"], "equal_center", "equal", rng)
        cvals.append(st["mean"])
        clos.append(st["lo"])
        chis.append(st["hi"])
        ax_b.plot([st["lo"], st["hi"]], [y, y],
                  color=TIER_COLS[r["tier"]], lw=1.1, zorder=3)
        ax_b.plot(st["mean"], y, "o", color=TIER_COLS[r["tier"]], ms=4.2,
                  mec="white", mew=0.5, zorder=4)
    ax_b.axvline(0, color="0.2", lw=0.8, zorder=2)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([f"{DISP[r['h']]} ({r['tier']})" for r in rows],
                         fontsize=6.2)
    ax_b.set_ylim(-0.6, len(rows) - 0.4)
    ax_b.axhline(len(rows) - nA - 0.5, color="0.6", lw=0.7, ls="--",
                 zorder=1)
    ax_b.set_xlabel("Δ mean PDS (centered − equal)")
    ax_b.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="", color=TIER_COLS[t],
                      ms=4, label=f"Tier {t}") for t in ("A", "B")]
    ax_b.legend(handles=handles, frameon=False, fontsize=6, loc="lower right")
    plabel(ax_b, "B")

    cq = parse_centerq("outputs/bench_variants_centerq.log")
    cholds = [h for h in ("k562", "rpe1", "hct116") if h in cq]
    yc = np.arange(len(cholds))[::-1]
    for y, h in zip(yc, cholds):
        eq = cq[h]["equal"]
        full = cq[h]["equal_center"] - eq
        matched = cq[h]["equal_center_q"] - eq
        ax_c.plot([full, matched], [y, y], color="0.82", lw=1.0, zorder=1)
        ax_c.plot(full, y, "o", mfc="white", mec="#2A9D8F", mew=1.1,
                  ms=4.6, zorder=3)
        ax_c.plot(matched, y, "o", color="#2A9D8F", ms=4.6, zorder=4)
    ax_c.axvline(0, color="0.2", lw=0.8, zorder=2)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([DISP[h] for h in cholds], fontsize=6.4)
    ax_c.set_ylim(-0.6, len(cholds) - 0.4)
    ax_c.set_xlabel("Δ mean PDS (centered − equal)")
    ax_c.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax_c.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec="#2A9D8F", mew=1.1,
               ms=4, label="full source pool"),
        Line2D([], [], marker="o", ls="", color="#2A9D8F", ms=4,
               label="query-matched"),
    ]
    ax_c.legend(handles=handles, frameon=False, fontsize=5.9,
                loc="lower right")
    plabel(ax_c, "C")
    save(fig, "fig4_gate_and_centering")


# --------------------------------------------------------------------------
# Figure 5: metric sensitivity and transfer determinants


def fig5_metric_determinants():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8),
                             gridspec_kw=dict(wspace=0.4, hspace=0.52))
    ax_m, ax_u, ax_rho, ax_dec = axes.ravel()
    noex = parse_variants("outputs/bench_variants.log")
    selfex = parse_variants("outputs/bench_variants_selfex.log")
    holds = [h for h in ("k562", "rpe1", "hct116")
             if h in noex and h in selfex]
    ys = np.arange(len(holds))[::-1]
    for ax, idx, name, letter in ((ax_m, 0, "median", "A"),
                                  (ax_u, 1, "mean", "B")):
        for k, (y, h) in enumerate(zip(ys, holds)):
            a, b = noex[h][idx], selfex[h][idx]
            ax.plot([a, b], [y, y], color="0.6", lw=1.3, zorder=2)
            ax.plot(a, y, "o", mfc="white", mec="0.35", ms=5, zorder=3)
            ax.plot(b, y, "o", color="0.1", ms=5, zorder=4)
            ax.text((a + b) / 2, y - 0.22, f"{b - a:+.3f}", ha="center",
                    fontsize=5.8, color="0.3")
            if k == 0:
                ax.text(a, y + 0.2, "no exclusion", ha="center", va="bottom",
                        fontsize=5.6, color="0.35")
                ax.text(b, y + 0.2, "excluded", ha="center", va="bottom",
                        fontsize=5.6, color="0.2")
        ax.set_yticks(ys)
        ax.set_yticklabels([DISP[h] for h in holds], fontsize=6.4)
        ax.set_ylim(-0.7, len(holds) - 0.25)
        ax.set_xlabel(f"equal-mix {name} PDS")
        ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        plabel(ax, letter)

    rows = []
    for tier in ("A", "B"):
        cfg = TIERS[tier]
        tri = load_triples(cfg["triples"])
        pairs = sorted({(r["holdout"], r["source"]) for r in tri})
        for h, s in pairs:
            sub = [r for r in tri if r["holdout"] == h and r["source"] == s]
            pds = np.array([float(r["pds_raw"]) for r in sub])
            rn = np.array([float(r["row_norm"]) for r in sub])
            ok = np.isfinite(pds) & np.isfinite(rn)
            if ok.sum() < 20:
                continue
            rho = float(spearmanr(rn[ok], pds[ok]).statistic)
            rows.append(dict(tier=tier, holdout=h, source=s, rho=rho,
                             n=int(ok.sum())))
    groups = []
    y = 0
    for tier in ("B", "A"):
        for h in TIERS[tier]["lines"]:
            sub = [r for r in rows if r["tier"] == tier
                   and r["holdout"] == h]
            if not sub:
                continue
            sub.sort(key=lambda r: r["rho"])
            groups.append((tier, h, y, sub))
            y -= 1
    ax_rho.axvline(0, color="0.2", lw=0.8, zorder=2)
    for tier, h, yg, sub in groups:
        vals = [r["rho"] for r in sub]
        ax_rho.hlines(yg, min(vals), max(vals), color="0.85", lw=0.8,
                      zorder=1)
        ax_rho.plot(vals, [yg] * len(vals), "o", ms=3.8,
                    color=TIER_COLS[tier], zorder=3)
    ax_rho.set_yticks([g[2] for g in groups])
    ax_rho.set_yticklabels([f"{DISP[g[1]]} ({g[0]})" for g in groups],
                           fontsize=5.9)
    ax_rho.set_ylim(min(g[2] for g in groups) - 0.7,
                    max(g[2] for g in groups) + 0.7)
    posA = sum(r["rho"] > 0 for r in rows if r["tier"] == "A")
    totA = sum(1 for r in rows if r["tier"] == "A")
    posB = sum(r["rho"] > 0 for r in rows if r["tier"] == "B")
    totB = sum(1 for r in rows if r["tier"] == "B")
    ax_rho.set_xlabel("Spearman ρ (PDS vs response norm)\n"
                      f"positive in {posA}/{totA} Tier A and {posB}/{totB} "
                      "Tier B pairs")
    ax_rho.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax_rho.set_axisbelow(True)
    plabel(ax_rho, "C")

    bins = {"A": [[] for _ in range(10)], "B": [[] for _ in range(10)]}
    for tier in ("A", "B"):
        tri = load_triples(TIERS[tier]["triples"])
        pairs = sorted({(r["holdout"], r["source"]) for r in tri})
        for h, s in pairs:
            sub = [r for r in tri if r["holdout"] == h and r["source"] == s]
            v = np.array([float(r["self_lfc"]) for r in sub])
            p = np.array([float(r["pds_raw"]) for r in sub])
            ok = np.isfinite(v) & np.isfinite(p)
            v, p = v[ok], p[ok]
            if len(v) < 50:
                continue
            edges = np.quantile(v, np.linspace(0, 1, 11))
            for b in range(10):
                msk = (v >= edges[b]) & (v <= edges[b + 1] if b == 9
                                         else v < edges[b + 1])
                if msk.sum():
                    bins[tier][b].append((float(np.median(v[msk])),
                                          float(np.median(p[msk]))))
    xs = np.arange(10)
    for tier in ("A", "B"):
        vals = [np.array([a[1] for a in bins[tier][b]]) for b in range(10)]
        mu = np.array([v.mean() if len(v) else np.nan for v in vals])
        se = np.array([v.std(ddof=1) / np.sqrt(len(v))
                       if len(v) > 1 else 0 for v in vals])
        ax_dec.plot(xs, mu, "o-", color=TIER_COLS[tier], ms=3.6, lw=1.2)
        ax_dec.fill_between(xs, mu - se, mu + se, color=TIER_COLS[tier],
                            alpha=0.15, lw=0)
    ax_dec.text(9.0, np.nanmean([a[1] for a in bins["A"][9]]) + 0.012,
                "Tier A", color=TIER_COLS["A"], fontsize=6.2, ha="right")
    ax_dec.text(9.0, np.nanmean([a[1] for a in bins["B"][9]]) - 0.018,
                "Tier B", color=TIER_COLS["B"], fontsize=6.2, ha="right")
    ax_dec.set_xticks([0, 2, 4, 6, 8, 9])
    ax_dec.set_xlim(-0.4, 9.4)
    ax_dec.set_xlabel("self-LFC decile (0 = strong KD)")
    ax_dec.set_ylabel("median PDS")
    ax_dec.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    ax_dec.set_axisbelow(True)
    plabel(ax_dec, "D")
    save(fig, "fig5_metric_determinants")


# --------------------------------------------------------------------------
# Figure 6: retrieval versus scGPT


def scgpt_paired(z, ctx, tag, meth, rng):
    if meth is None:
        tkey, pkey = f"{ctx}|{tag}|targets", f"{ctx}|{tag}|pds"
    else:
        tkey = f"{ctx}|{tag}|{meth}|targets"
        pkey = f"{ctx}|{tag}|{meth}|pds"
    mt = z[tkey].astype(str)
    mp = z[pkey].astype(np.float64)
    bt = z[f"{ctx}|{tag}|targets"].astype(str)
    bp = z[f"{ctx}|{tag}|pds"].astype(np.float64)
    bmap = dict(zip(bt, bp))
    common = [t for t in mt if t in bmap]
    d = np.array([mp[list(mt).index(t)] - bmap[t] for t in common])
    d = d[np.isfinite(d)]
    bs = rng.choice(d, (4000, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi),
                n=len(d), win=float(np.mean(d > 0)))


def fig6_scgpt():
    z = np.load("outputs/bench_scgpt_baseline.npz", allow_pickle=True)
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.5)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_k = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[0, 2])
    for ax, ctx, tag, label, inset, letter in (
            (ax_r, "rpe1", "scgpt_r10", "RPE1 (cross-line)", None, "A"),
            (ax_k, "k562", "scgpt_check", "K562 (in-sample)",
             "#FDF0E3", "B")):
        x = z[f"{ctx}|{tag}|pds"].astype(np.float64)
        y = z[f"{ctx}|{tag}|equal|pds"].astype(np.float64)
        ax.plot([0, 1.05], [0, 1.05], color="0.65", lw=0.7, ls="--",
                zorder=1)
        ax.scatter(x, y, s=10, color="#5C6B8A", alpha=0.6,
                   edgecolors="none", zorder=3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("scGPT PDS")
        if letter == "A":
            ax.set_ylabel("equal-mix PDS")
        win = float(np.mean(y > x))
        ax.text(0.96, 0.04,
                f"median {np.median(x):.2f} → {np.median(y):.2f}\n"
                f"mean Δ {np.mean(y - x):+.2f}\n"
                f"equal wins {win * 100:.0f}%",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
                color="0.3",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.85))
        bbox = dict(boxstyle="round,pad=0.2", fc=inset if inset else "white",
                    ec="#E69F00" if inset else "0.8", lw=0.5, alpha=0.9)
        ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top",
                fontsize=6.3, color="0.25", bbox=bbox)
        ax.grid(color="0.93", lw=0.4, zorder=0)
        plabel(ax, letter)

    rows = [
        ("rpe1", "scgpt_r10", "mean", "mean", "#2A9D8F"),
        ("rpe1", "scgpt_r10", "equal", "equal mix", "#2A9D8F"),
        ("rpe1", "scgpt_r10", "equal_center", "equal + centering", "#2A9D8F"),
        ("k562", "scgpt_check", "mean", "mean", "#9E9E9E"),
        ("k562", "scgpt_check", "equal", "equal mix", "#9E9E9E"),
        ("k562", "scgpt_check", "equal_center", "equal + centering",
         "#9E9E9E"),
    ]
    ypos = [5, 4, 3, 1.6, 0.6, -0.4]
    for (ctx, tag, meth, label, col), y in zip(rows, ypos):
        st = scgpt_paired(z, ctx, tag, meth, rng)
        ax_s.plot([st["lo"], st["hi"]], [y, y], color=col, lw=1.1, zorder=3)
        ax_s.plot(st["mean"], y, "o", color=col, ms=4.2, mec="white",
                  mew=0.5, zorder=4)
        ax_s.text(st["hi"] + 0.012, y, f"{st['mean']:+.2f}", va="center",
                  fontsize=5.8, color="0.3")
    ax_s.axvline(0, color="0.2", lw=0.8, zorder=2)
    ax_s.axhline(2.2, color="0.7", lw=0.6, ls="--", zorder=1)
    ax_s.text(-0.30, 5.45, "RPE1 (cross-line)", fontsize=6.4,
              color="#1F7A6F", ha="left", va="center")
    ax_s.text(-0.30, 2.05, "K562 (in-sample)", fontsize=6.4,
              color="0.45", ha="left", va="center")
    ax_s.set_yticks(ypos)
    ax_s.set_yticklabels([r[3] for r in rows], fontsize=6.2)
    ax_s.set_ylim(-0.9, 5.8)
    ax_s.set_xlim(-0.32, 0.72)
    ax_s.set_xlabel("Δ mean PDS vs scGPT (95% CI)")
    ax_s.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax_s.set_axisbelow(True)
    plabel(ax_s, "C")
    save(fig, "fig6_scgpt_baseline")


def main():
    fig1_overview()
    fig2_method_comparison()
    fig3_paired_forest()
    fig4_gate_center()
    fig5_metric_determinants()
    fig6_scgpt()


if __name__ == "__main__":
    main()
