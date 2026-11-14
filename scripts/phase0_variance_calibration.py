"""Phase 0: does target-line template dominance predict calibration benefit?

Uses existing benchmark artifacts only. For each held-out line we compute
several within-line variance-structure scalars (alignment of each response
with the line mean, mean/norm ratio, PC1 variance share) and correlate them
with the mean PDS gain of supervised calibration (ridge and residual MLP).

The hypothesis is that calibration helps less in lines whose responses are
dominated by a shared template, and more in lines with a larger
cell-line-by-perturbation interaction component.

Outputs:
  outputs/phase0_variance_calibration.csv
  outputs/phase0_variance_calibration.log
  figures/phase0_template_vs_delta.png / .pdf
"""

import csv
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, "scripts")
from bench_common import load_axis, load_lines

TIER_A = ["k562", "rpe1", "hepg2", "jurkat"]
TIER_B = ["h1", "k562", "rpe1", "hct116", "hek293t"]


def line_metrics(R):
    """Within-line variance-structure scalars for a response matrix."""
    R = np.asarray(R, dtype=np.float64)
    Rn = np.linalg.norm(R, axis=1)
    m = R.mean(0)
    mn = np.linalg.norm(m)
    if mn < 1e-12:
        cos2 = np.zeros(len(R))
    else:
        cos2 = (R @ m) ** 2 / np.maximum(Rn ** 2 * mn ** 2, 1e-24)
    X = R - R.mean(0, keepdims=True)
    _, S, _ = randomized_svd(X, n_components=1, random_state=0)
    tot = float(np.sum(X ** 2))
    pc1 = float(S[0] ** 2 / tot) if tot > 0 else np.nan
    return dict(
        n_rows=int(R.shape[0]),
        n_genes=int(R.shape[1]),
        cos2_mean=float(np.mean(cos2)),
        cos2_median=float(np.median(cos2)),
        norm_ratio=float(mn ** 2 / max(np.mean(Rn ** 2), 1e-24)),
        pc1_share=pc1,
    )


def read_ridge():
    out = {}
    paths = {
        "A": "outputs/bench_supervised_nadig_v2_mix_noint_pc1024.csv",
        "B": "outputs/bench_supervised_v2_mix_noint_pc1024.csv",
    }
    for tier, path in paths.items():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                out[(tier, row["holdout"].lower())] = float(row["delta_mean"])
    return out


def read_mlp():
    out = {}
    paths = {
        "A": "outputs/bench_supervised_nadig_v2_nl_mlp.npz",
        "B": "outputs/bench_supervised_v2_nl_mlp.npz",
    }
    for tier, path in paths.items():
        z = np.load(path, allow_pickle=True)
        h = z["holdout"].astype(str)
        for name in sorted(set(h)):
            mask = h == name
            d = z["pds_raw"][mask] - z["pds_equal"][mask]
            out[(tier, name.lower())] = float(np.nanmean(d))
    return out


def read_mlp_seed1():
    out = {}
    paths = {
        "A": "outputs/bench_supervised_nadig_v2_nl_mlp_seed1.npz",
        "B": "outputs/bench_supervised_v2_nl_mlp_seed1.npz",
    }
    for tier, path in paths.items():
        z = np.load(path, allow_pickle=True)
        h = z["holdout"].astype(str)
        for name in sorted(set(h)):
            mask = h == name
            d = z["pds_raw"][mask] - z["pds_equal"][mask]
            out[(tier, name.lower())] = float(np.nanmean(d))
    return out


def perm_spearman(x, y, n=20000, seed=0, alternative="less"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        yp = rng.permutation(y)
        r = spearmanr(x, yp).statistic
        if alternative == "less":
            cnt += r <= rho
        elif alternative == "greater":
            cnt += r >= rho
        else:
            cnt += abs(r) >= abs(rho)
    return rho, (cnt + 1) / (n + 1)


def loo_spearman(x, y, labels):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        rho = float(spearmanr(x[keep], y[keep]).statistic)
        out.append((labels[i], rho))
    return out


def strat_perm_spearman(x, y, tiers, n=20000, seed=0):
    """Permute y within tier, preserving the two-tier structure."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    tiers = np.asarray(tiers)
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        yp = y.copy()
        for t in set(tiers):
            idx = np.where(tiers == t)[0]
            yp[idx] = rng.permutation(yp[idx])
        r = spearmanr(x, yp).statistic
        cnt += r <= rho
    return rho, (cnt + 1) / (n + 1)


def report_corr(tag, x, y, labels, alt):
    rho, p = perm_spearman(x, y, alternative=alt)
    r, _ = pearsonr(x, y)
    print(f"  {tag:38s} n={len(x):2d}  Spearman {rho:+.3f} "
          f"(perm p={p:.4f}, {alt})  Pearson {r:+.3f}")
    return rho, p


def main():
    print("Phase 0: template dominance vs calibration benefit", flush=True)

    # Tier A compact matrix
    z = np.load("outputs/nadig_rows.npz", allow_pickle=True)
    rows_a = {}
    for L in TIER_A:
        rows_a[L] = z[f"{L}_rows"].astype(np.float64)
    met_a = {L: line_metrics(rows_a[L]) for L in TIER_A}

    # Tier B full axis, evaluation positions only
    genes, pos, gidx, m = load_axis()
    lines = load_lines(genes, gidx, m)
    lines.pop("_gidx", None)
    met_b = {}
    for L in TIER_B:
        if L not in lines:
            continue
        R = lines[L]["rows"][:, pos].astype(np.float64)
        met_b[L] = line_metrics(R)

    ridge = read_ridge()
    mlp = read_mlp()
    seed1 = read_mlp_seed1()

    rows = []
    for tier, names, met in (("A", TIER_A, met_a), ("B", TIER_B, met_b)):
        for L in names:
            if L not in met:
                continue
            rows.append(dict(
                tier=tier, holdout=L,
                n_rows=met[L]["n_rows"], n_genes=met[L]["n_genes"],
                cos2_mean=met[L]["cos2_mean"],
                cos2_median=met[L]["cos2_median"],
                norm_ratio=met[L]["norm_ratio"],
                pc1_share=met[L]["pc1_share"],
                ridge_delta=ridge.get((tier, L), np.nan),
                mlp_delta=mlp.get((tier, L), np.nan),
                mlp_delta_seed1=seed1.get((tier, L), np.nan),
            ))

    out_csv = "outputs/phase0_variance_calibration.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nPer-line variance structure and calibration gain")
    print(f"{'tier':4s} {'line':8s} {'rows':>6s} {'cos2':>7s} {'cos2med':>8s} "
          f"{'norm':>7s} {'pc1':>7s} {'ridge':>8s} {'mlp':>8s} {'mlp_s1':>8s}")
    for r in rows:
        s1 = ("%.4f" % r["mlp_delta_seed1"]
              if np.isfinite(r["mlp_delta_seed1"]) else "   -")
        print(f"{r['tier']:4s} {r['holdout']:8s} {r['n_rows']:6d} "
              f"{r['cos2_mean']:7.3f} {r['cos2_median']:8.3f} "
              f"{r['norm_ratio']:7.3f} {r['pc1_share']:7.3f} "
              f"{r['ridge_delta']:+8.4f} {r['mlp_delta']:+8.4f} {s1:>8s}")

    labels = [f"{r['tier']}:{r['holdout']}" for r in rows]
    cos2 = [r["cos2_mean"] for r in rows]
    pc1 = [r["pc1_share"] for r in rows]
    ridge_d = [r["ridge_delta"] for r in rows]
    mlp_d = [r["mlp_delta"] for r in rows]

    print("\nCorrelations (one-sided: higher template -> lower gain)")
    results = {}
    for tag, idx in (("all", np.arange(len(rows))),
                     ("Tier A", np.array([i for i, r in enumerate(rows)
                                          if r["tier"] == "A"])),
                     ("Tier B", np.array([i for i, r in enumerate(rows)
                                          if r["tier"] == "B"])),
                     ("Tier B no H1", np.array([i for i, r in enumerate(rows)
                                                if r["tier"] == "B"
                                                and r["holdout"] != "h1"])),
                     ("all no H1", np.array([i for i, r in enumerate(rows)
                                             if r["holdout"] != "h1"]))):
        if len(idx) < 3:
            continue
        print(f"[{tag}]")
        results[(tag, "mlp_cos2")] = report_corr(
            "MLP delta vs cos2", np.array(cos2)[idx], np.array(mlp_d)[idx],
            [labels[i] for i in idx], "less")
        results[(tag, "ridge_cos2")] = report_corr(
            "ridge delta vs cos2", np.array(cos2)[idx], np.array(ridge_d)[idx],
            [labels[i] for i in idx], "less")
        results[(tag, "mlp_pc1")] = report_corr(
            "MLP delta vs PC1 share", np.array(pc1)[idx], np.array(mlp_d)[idx],
            [labels[i] for i in idx], "less")
        results[(tag, "ridge_pc1")] = report_corr(
            "ridge delta vs PC1 share", np.array(pc1)[idx], np.array(ridge_d)[idx],
            [labels[i] for i in idx], "less")

    print("\nStratified tests (permute ΔPDS within tier)")
    tiers = [r["tier"] for r in rows]
    for mname in ("cos2_mean", "cos2_median", "norm_ratio", "pc1_share"):
        x = np.array([r[mname] for r in rows])
        z = np.zeros_like(x)
        for t in ("A", "B"):
            idx = np.array([i for i, r in enumerate(rows) if r["tier"] == t])
            z[idx] = (x[idx] - x[idx].mean()) / (x[idx].std(ddof=0) + 1e-12)
        for target in ("mlp_delta", "ridge_delta"):
            y = np.array([r[target] for r in rows])
            rho_raw, p_raw = strat_perm_spearman(x, y, tiers)
            rho_z, p_z = strat_perm_spearman(z, y, tiers)
            print(f"  {mname:12s} vs {target:11s}  raw {rho_raw:+.3f} "
                  f"(p={p_raw:.4f})  within-tier z {rho_z:+.3f} (p={p_z:.4f})")

    print("\nLeave-one-out (MLP delta vs cos2, all 9)")
    for lab, rho in loo_spearman(cos2, mlp_d, labels):
        print(f"  drop {lab:10s} Spearman {rho:+.3f}")
    print("\nLeave-one-out (MLP delta vs cos2, Tier A only)")
    ia = [i for i, r in enumerate(rows) if r["tier"] == "A"]
    for lab, rho in loo_spearman(np.array(cos2)[ia], np.array(mlp_d)[ia],
                                 [labels[i] for i in ia]):
        print(f"  drop {lab:10s} Spearman {rho:+.3f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        colors = {"A": "#1f77b4", "B": "#d62728"}
        for ax, yvals, title in (
                (axes[0], ridge_d, "Ridge calibration"),
                (axes[1], mlp_d, "Residual MLP calibration")):
            for tier in ("A", "B"):
                xs = [c for c, r in zip(cos2, rows) if r["tier"] == tier]
                ys = [v for v, r in zip(yvals, rows) if r["tier"] == tier]
                ax.scatter(xs, ys, s=42, color=colors[tier], label=f"Tier {tier}")
            for c, v, r in zip(cos2, yvals, rows):
                ax.annotate(r["holdout"], (c, v), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            ax.axhline(0, color="0.6", lw=0.8, ls="--")
            ax.set_xlabel("template alignment (mean cos$^2$ with line mean)")
            ax.set_title(title, fontsize=10)
        axes[0].set_ylabel("calibration ΔPDS")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"figures/phase0_template_vs_delta.{ext}", dpi=300)
        print("\nwrote figures/phase0_template_vs_delta.png/.pdf")
    except Exception as e:
        print("plot skipped:", e)

    print(f"\nwrote {out_csv}")
    print("\nPhase 0 readout:")
    rho, p = results[("all", "mlp_cos2")]
    print(f"  raw pooled: MLP delta vs cos2, all 9, Spearman {rho:+.3f}, p={p:.4f}")
    x = np.array(cos2)
    z = np.zeros_like(x)
    for t in ("A", "B"):
        idx = np.array([i for i, r in enumerate(rows) if r["tier"] == t])
        z[idx] = (x[idx] - x[idx].mean()) / (x[idx].std(ddof=0) + 1e-12)
    rho_s, p_s = strat_perm_spearman(z, np.array(mlp_d), tiers)
    print(f"  tier-stratified primary: within-tier z cos2 vs MLP, "
          f"Spearman {rho_s:+.3f}, p={p_s:.4f}")
    print("  go if |rho_s| >= 0.60 and p < 0.05 and LOO keeps the sign")


if __name__ == "__main__":
    main()
