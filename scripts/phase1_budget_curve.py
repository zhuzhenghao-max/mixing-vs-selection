"""Assemble Figure 7: the calibration budget curve from Phase 1 pilot CSVs."""

import csv
import glob

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = sorted(glob.glob("outputs/phase1_selection_pilot_A_k*.csv"))
rows = []
for p in FILES:
    with open(p, newline="") as f:
        rows += list(csv.DictReader(f))

for r in rows:
    for k in ("k", "delta_split_mean", "delta_split_sd", "vs_random",
              "vs_random_lo", "vs_random_hi", "vs_random_p", "win_vs_random"):
        r[k] = float(r[k])

ks = sorted({int(r["k"]) for r in rows})
strategies = ["random", "kmeans", "shared", "magnitude",
              "diversity", "disagreement"]
holdouts = ["k562", "rpe1"]


def curve(h, s, metric="delta_split_mean"):
    out = []
    for k in ks:
        v = [r[metric] for r in rows
             if r["holdout"] == h and r["strategy"] == s and int(r["k"]) == k]
        out.append(v[0] if v else np.nan)
    return np.array(out, dtype=float)


def crossover(h, s):
    y = curve(h, s)
    for i in range(len(ks) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] * y[i + 1] <= 0:
            if y[i] == y[i + 1]:
                return float(ks[i])
            return float(ks[i] + (0 - y[i]) * (ks[i + 1] - ks[i])
                         / (y[i + 1] - y[i]))
    return np.nan


colors = {"random": "#444444", "kmeans": "#d62728", "shared": "#1f77b4",
          "magnitude": "#2ca02c", "diversity": "#9467bd",
          "disagreement": "#ff7f0e"}

print("Crossover k (delta PDS = 0), interpolated")
for h in holdouts:
    for s in strategies:
        print(f"  {h:6s} {s:12s} {crossover(h, s)}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, h in zip(axes[:2], holdouts):
    for s in strategies:
        ax.plot(ks, curve(h, s), marker="o", ms=4, lw=1.6,
                color=colors[s], label=s)
    ax.axhline(0, color="0.7", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("calibration perturbations measured (k)")
    ax.set_ylabel("ΔPDS vs equal mixing")
    ax.set_title(h.upper(), fontsize=10)
axes[0].legend(frameon=False, fontsize=7)

ax = axes[2]
for h, mk in (("k562", "o"), ("rpe1", "s")):
    gain = curve(h, "kmeans", "vs_random")
    lo = curve(h, "kmeans", "vs_random_lo")
    hi = curve(h, "kmeans", "vs_random_hi")
    ax.errorbar(ks, gain, yerr=[np.maximum(gain - lo, 0),
                                np.maximum(hi - gain, 0)],
                marker=mk, ms=5, lw=1.6, capsize=3, label=h.upper())
ax.axhline(0, color="0.7", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel("calibration perturbations measured (k)")
ax.set_ylabel("kmeans − random (ΔPDS)")
ax.set_title("selection gain", fontsize=10)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig7_calibration_selection.{ext}", dpi=300)
print("\nwrote figures/fig7_calibration_selection.png/.pdf")

out = "outputs/phase1_budget_curve.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}")
