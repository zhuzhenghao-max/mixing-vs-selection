"""Regime diagnostic: source-response coherence predicts equal-mix transfer.

For every query target, take the response vectors of the source lines that
measured it, unit-normalize them, and summarize their agreement with the mean
pairwise cosine, the minimum pairwise cosine, and the norm of their mean
(resultant length).  Versions with per-source centering (subtract the source's
mean response row, as in the equal_center rule) isolate shape agreement from
baseline differences.  All summaries are computable from source data alone,
before the held-out line is seen.  Join with the equal-mix PDS already
computed in the two leave-one-line-out benchmarks and test whether coherence
tracks transfer, with and without controlling for response strength.

Outputs
  outputs/regime_diagnostic_targets.csv
  outputs/regime_diagnostic_holdouts.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from bench_common import load_axis, load_lines, load_nadig_lines, unit_rows

OUT = "outputs"


def load_bench(path):
    z = np.load(path, allow_pickle=True)
    ho = z["holdout"].astype(str)
    me = z["method"].astype(str)
    tg = z["target"].astype(str)
    p = z["pds_raw"].astype(np.float64)
    keep = me == "equal"
    return {(h, t): v for h, t, v in zip(ho[keep], tg[keep], p[keep])}


def summarize(V):
    k = len(V)
    G = V @ V.T
    iu = np.triu_indices(k, 1)
    pair = G[iu]
    res = float(np.linalg.norm(V.mean(0)))
    mix = V.mean(0)
    return float(pair.mean()), float(pair.min()), res, mix


def run_tier(tier, lines, holds, pds):
    recs = []
    means = {o: lines[o]["rows"].mean(0) for o in holds if o in lines}
    for h in holds:
        if h not in lines:
            continue
        hline = lines[h]
        srcs = [o for o in holds if o != h and o in lines]
        for t, pv in pds.items():
            if t[0] != h:
                continue
            raw, cen, norms, selfs = [], [], [], []
            for o in srcs:
                oline = lines[o]
                j = oline["index"].get(t[1])
                if j is None:
                    continue
                r = oline["rows"][j].astype(np.float64)
                raw.append(r)
                cen.append(r - means[o])
                norms.append(float(np.linalg.norm(r)))
                s = oline["self_lfc"][j]
                if np.isfinite(s):
                    selfs.append(abs(float(s)))
            if len(raw) < 2:
                continue
            R = np.stack(raw)
            C = np.stack(cen)
            Vr = unit_rows(R)
            Vc = unit_rows(C)
            pair, pmin, res, mix = summarize(Vr)
            cpair, cmin, cres, cmix = summarize(Vc)
            truth = hline["rows"][hline["index"][t[1]]].astype(np.float64)
            tn = np.linalg.norm(truth)
            mt = float(mix @ truth / (np.linalg.norm(mix) * tn + 1e-12))
            ctruth = truth - means[h]
            ctn = np.linalg.norm(ctruth)
            cmt = float(cmix @ ctruth /
                        (np.linalg.norm(cmix) * ctn + 1e-12))
            recs.append(dict(
                tier=tier, holdout=h, target=t[1], pds=float(pv),
                n_src=len(raw), pair_cos=pair, min_cos=pmin, resultant=res,
                cpair_cos=cpair, cmin_cos=cmin, cresultant=cres,
                log_norm=float(np.log10(np.mean(norms))),
                abs_self=float(np.mean(selfs)) if selfs else np.nan,
                mix_truth_cos=mt, cmix_truth_cos=cmt,
                pool=len(hline["names"])))
    return pd.DataFrame(recs)


def partial_spearman(x, y, z):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rz = np.argsort(np.argsort(z)).astype(float)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(ex, ey).statistic


def partial_spearman2(x, y, z1, z2):
    def rk(v):
        return np.argsort(np.argsort(v)).astype(float)
    Z = np.column_stack([np.ones_like(rk(z1)), rk(z1), rk(z2)])
    rx, ry = rk(x), rk(y)
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return spearmanr(ex, ey).statistic


def main():
    genes, pos, gidx, m = load_axis()

    pds_b = load_bench(f"{OUT}/bench_transfer_v2.npz")
    lines_b = load_lines(genes, gidx, m)
    lines_b.pop("_gidx", None)
    db = run_tier("B", lines_b, ["h1", "k562", "rpe1", "hct116", "hek293t"],
                  pds_b)
    del lines_b

    pds_a = load_bench(f"{OUT}/bench_transfer_nadig_v2.npz")
    lines_a = load_nadig_lines(genes, gidx)
    da = run_tier("A", lines_a, ["k562", "rpe1", "hepg2", "jurkat"], pds_a)
    holds_a = ["k562", "rpe1", "hepg2", "jurkat"]
    noex = np.load(f"{OUT}/pilot_pds_nadig_noexcl.npz")
    key = {}
    for L in holds_a:
        srcs = [o for o in holds_a if o != L]
        q = [t for t in lines_a[L]["names"]
             if all(t in lines_a[o]["index"] for o in srcs)]
        key.update({(L, t): float(v) for t, v in
                    zip(q, noex[L + "_noexcl"])})
    da["pds_noexcl"] = [key.get((h, t), np.nan)
                        for h, t in zip(da.holdout, da.target)]

    d = pd.concat([da, db], ignore_index=True)
    d.to_csv(f"{OUT}/regime_diagnostic_targets.csv", index=False,
             float_format="%.6f")

    cols = ["pair_cos", "min_cos", "resultant", "cpair_cos", "cmin_cos",
            "cresultant", "log_norm", "abs_self", "mix_truth_cos",
            "cmix_truth_cos"]
    rows = []
    for (tier, h), g in d.groupby(["tier", "holdout"]):
        r = dict(tier=tier, holdout=h, n=len(g),
                 pool=int(g.pool.iloc[0]),
                 n_src_mean=float(g.n_src.mean()),
                 pds=float(g.pds.mean()), pds_median=float(g.pds.median()),
                 pds_noexcl=float(g.pds_noexcl.mean())
                 if g.pds_noexcl.notna().any() else np.nan)
        for c in cols:
            r[c] = float(g[c].mean())
        for c in ("pair_cos", "cpair_cos", "min_cos", "cmin_cos",
                  "resultant", "cresultant", "mix_truth_cos",
                  "cmix_truth_cos"):
            r["rho_" + c] = float(spearmanr(g[c], g.pds).statistic)
            r["prho_" + c] = float(partial_spearman(
                g[c].to_numpy(), g.pds.to_numpy(), g.log_norm.to_numpy()))
        for c in ("pair_cos", "cpair_cos"):
            ok = np.isfinite(g.abs_self.to_numpy())
            r["prho2_" + c] = float(partial_spearman(
                g[c].to_numpy()[ok], g.pds.to_numpy()[ok],
                g.abs_self.to_numpy()[ok])) if ok.sum() > 20 else np.nan
        rows.append(r)
    s = pd.DataFrame(rows).sort_values(["tier", "holdout"])
    s.to_csv(f"{OUT}/regime_diagnostic_holdouts.csv", index=False,
             float_format="%.6f")
    show = s[["tier", "holdout", "n", "n_src_mean", "pair_cos", "cpair_cos",
              "min_cos", "log_norm", "abs_self", "pds", "pds_noexcl",
              "rho_pair_cos", "rho_cpair_cos", "rho_min_cos",
              "prho_pair_cos", "prho_cpair_cos", "prho2_pair_cos",
              "prho2_cpair_cos", "rho_mix_truth_cos", "rho_cmix_truth_cos"]]
    print(show.to_string(index=False))

    print("\npooled within-holdout (mean of Fisher z):")
    for c in cols:
        zz = []
        for _, g in d.groupby(["tier", "holdout"]):
            r = spearmanr(g[c], g.pds).statistic
            if np.isfinite(r) and len(g) > 20:
                zz.append(np.arctanh(np.clip(r, -0.999, 0.999)))
        print(f"  {c:14s} mean rho {np.tanh(np.mean(zz)):+.3f} "
              f"({len(zz)} holdouts)")

    print("\nwithin-tier pooled Spearman (targets pooled, tiers apart):")
    for tier, g in d.groupby("tier"):
        for c in ("pair_cos", "cpair_cos", "min_cos", "cmin_cos",
                  "resultant", "cresultant", "mix_truth_cos"):
            r = spearmanr(g[c], g.pds).statistic
            print(f"  tier {tier} {c:14s} rho {r:+.3f} n={len(g)}")

    print("\nacross holdouts, Spearman of means (9 holdouts):")
    for c in cols + ["n_src_mean"]:
        r = spearmanr(s[c], s.pds).statistic
        print(f"  {c:14s} rho {r:+.3f}")
    r = spearmanr(np.log10(s.pool), s.pds).statistic
    print(f"  log10(pool)    rho {r:+.3f}")

    print("\nmechanism (within-holdout mean of Fisher z):")
    for a, b in (("pair_cos", "mix_truth_cos"), ("cpair_cos", "cmix_truth_cos"),
                 ("pair_cos", "log_norm")):
        zz = [np.arctanh(np.clip(spearmanr(g[a], g[b]).statistic, -0.999, 0.999))
              for _, g in d.groupby(["tier", "holdout"])
              if np.isfinite(spearmanr(g[a], g[b]).statistic)]
        print(f"  corr({a}, {b:14s}) mean rho {np.tanh(np.mean(zz)):+.3f}")
    for ctl in ("mix_truth_cos", "log_norm", "n_src"):
        zz = [np.arctanh(np.clip(partial_spearman(
                  g.pair_cos.to_numpy(), g.pds.to_numpy(), g[ctl].to_numpy()),
                  -0.999, 0.999))
              for _, g in d.groupby(["tier", "holdout"])]
        print(f"  pair_cos~pds | {ctl:14s} mean rho {np.tanh(np.mean(zz)):+.3f}")
    zz = [np.arctanh(np.clip(partial_spearman2(
              g.pair_cos.to_numpy(), g.pds.to_numpy(),
              g.log_norm.to_numpy(), g.n_src.to_numpy()), -0.999, 0.999))
          for _, g in d.groupby(["tier", "holdout"])]
    print(f"  pair_cos~pds | norm+n_src   mean rho {np.tanh(np.mean(zz)):+.3f}")
    zz = []
    for _, g in d.groupby(["tier", "holdout"]):
        r = partial_spearman(g.mix_truth_cos.to_numpy(), g.pds.to_numpy(),
                             g.pair_cos.to_numpy())
        zz.append(np.arctanh(np.clip(r, -0.999, 0.999)))
    print(f"  mix_truth_cos~pds | pair_cos mean rho {np.tanh(np.mean(zz)):+.3f}")

    print("\nacross-holdout means with p-values (n=9):")
    for c in ("pair_cos", "cpair_cos", "resultant", "cresultant", "mix_truth_cos",
              "log_norm", "n_src_mean"):
        r = spearmanr(s[c], s.pds)
        print(f"  {c:14s} rho {r.statistic:+.3f} p {r.pvalue:.4f}")

    print("\nNadig balanced 638-query panel, no-exclusion policy:")
    bal = d.dropna(subset=["pds_noexcl"])
    for (tier, h), g in bal.groupby(["tier", "holdout"]):
        rp = spearmanr(g.pair_cos, g.pds_noexcl).statistic
        rp2 = spearmanr(g.pair_cos, g.pds).statistic
        print(f"  {h:8s} n={len(g)} pair_cos {g.pair_cos.mean():.3f} "
              f"pds_noexcl {g.pds_noexcl.mean():.4f} "
              f"rho_noexcl {rp:+.3f}  rho_excl {rp2:+.3f}")
    zz = []
    for _, g in bal.groupby(["tier", "holdout"]):
        r = spearmanr(g.pair_cos, g.pds_noexcl).statistic
        zz.append(np.arctanh(np.clip(r, -0.999, 0.999)))
    print(f"  pooled rho_noexcl {np.tanh(np.mean(zz)):+.3f}")

    print("\nTier means:")
    print(d.groupby("tier")[["pair_cos", "cpair_cos", "min_cos",
                             "resultant", "mix_truth_cos", "pds",
                             "pds_noexcl"]]
          .mean().to_string())


if __name__ == "__main__":
    main()
