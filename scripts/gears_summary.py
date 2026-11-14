"""Paired GEARS-vs-equal-mix summary on matched Tier A targets.

Reads outputs/gears_baseline_A.npz (per-fold targets, predictions, PDS) and the
benchmark PDS for the equal mix, and reports per-fold mean and median PDS, the
paired difference (equal minus GEARS) with a 95% bootstrap CI, the win rate,
and a two-sided Wilcoxon p-value. Output: outputs/gears_summary_A.csv.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = "outputs"
HOLDS = ["k562", "rpe1", "hepg2", "jurkat"]


def best_epochs(path):
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return {}
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("tier holdout"):
            start = i
    out = {}
    if start is None:
        return out
    for line in lines[start + 1:]:
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "A":
            try:
                out[parts[1]] = int(parts[3])
            except ValueError:
                pass
    return out


def main():
    z = np.load(f"{OUT}/gears_baseline_A.npz", allow_pickle=True)
    b = np.load(f"{OUT}/bench_transfer_nadig_v2.npz", allow_pickle=True)
    be = best_epochs(f"{OUT}/gears_full_A.log")
    ho, me, tg = (b[k].astype(str) for k in ("holdout", "method", "target"))
    p = b["pds_raw"].astype(np.float64)
    keep = me == "equal"
    eq = {(h, t): v for h, t, v in zip(ho[keep], tg[keep], p[keep])}
    keep = me == "mean"
    mn = {(h, t): v for h, t, v in zip(ho[keep], tg[keep], p[keep])}
    rng = np.random.default_rng(0)
    rows = []
    for hold in HOLDS:
        if f"{hold}|targets" not in z.files:
            continue
        q = [str(x) for x in z[f"{hold}|targets"]]
        pds = z[f"{hold}|pds"].astype(np.float64)
        eqv = np.array([eq.get((hold, t), np.nan) for t in q])
        mnv = np.array([mn.get((hold, t), np.nan) for t in q])
        ok = np.isfinite(eqv) & np.isfinite(mnv)
        d = eqv[ok] - pds[ok]
        dm = pds[ok] - mnv[ok]
        boots = [float(np.mean(rng.choice(d, size=len(d), replace=True)))
                 for _ in range(2000)]
        lo, hi = np.quantile(boots, [0.025, 0.975])
        bootm = [float(np.mean(rng.choice(dm, size=len(dm), replace=True)))
                 for _ in range(2000)]
        mlo, mhi = np.quantile(bootm, [0.025, 0.975])
        pv = (wilcoxon(d).pvalue if len(d) > 10 else np.nan)
        pm = (wilcoxon(dm).pvalue if len(dm) > 10 else np.nan)
        rows.append(dict(
            holdout=hold, n=int(ok.sum()), best_epoch=be.get(hold),
            gears_mean=float(np.mean(pds[ok])),
            gears_median=float(np.median(pds[ok])),
            equal_mean=float(np.mean(eqv[ok])),
            equal_median=float(np.median(eqv[ok])),
            d_mean=float(np.mean(d)), d_median=float(np.median(d)),
            ci_lo=float(lo), ci_hi=float(hi),
            win=float(np.mean(d > 0)), p=float(pv),
            mean_pds=float(np.mean(mnv[ok])),
            gears_minus_mean=float(np.mean(dm)),
            ci_m_lo=float(mlo), ci_m_hi=float(mhi),
            win_vs_mean=float(np.mean(dm > 0)), p_vs_mean=float(pm)))
    s = pd.DataFrame(rows)
    s.to_csv(f"{OUT}/gears_summary_A.csv", index=False, float_format="%.6g")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
