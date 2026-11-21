"""H1 representation sensitivity for the Tier B equal-mix baseline.

Tier B uses 285 decoded H1 response rows, but only 150 correspond to the
official training targets and have a DESeq2 truth table. This recomputes the
equal-mix PDS for each Tier B holdout under three H1 source representations:

  full285     all decoded rows (the benchmark baseline)
  cache150    only the 150 official targets, decoded rows
  deseq150    only the 150 official targets, DESeq2 rows clipped at +/-10

Queries are paired across representations on the common target set. Only
the equal mix is recomputed; other methods are not re-fitted.
"""

import csv
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bench_common import (Retrieval, boot_ci, load_axis, load_lines,
                          unit_rows)
from scipy.stats import spearmanr

HOLDS = ["k562", "rpe1", "hct116", "hek293t"]
BASE = ["k562", "rpe1", "hct116", "hek293t"]


def mix_features(lines, srcs, q, pos):
    X = np.zeros((len(q), len(pos)))
    ns = np.zeros(len(q), dtype=int)
    for i, t in enumerate(q):
        acc = []
        for o in srcs:
            if o == "h1_var":
                continue
            j = lines[o]["index"].get(t)
            if j is None:
                continue
            v = lines[o]["rows"][j, pos]
            nv = np.linalg.norm(v)
            if nv > 1e-9:
                acc.append(v / nv)
                ns[i] += 1
        if "h1_var" in srcs:
            j = lines["h1_var"]["index"].get(t)
            if j is not None:
                v = lines["h1_var"]["rows"][j, pos]
                nv = np.linalg.norm(v)
                if nv > 1e-9:
                    acc.append(v / nv)
                    ns[i] += 1
        if acc:
            X[i] = np.mean(acc, axis=0)
    return X, ns


def common_queries(lines, srcs, ref):
    return [t for t in ref["names"]
            if any(t in lines[o]["index"] for o in srcs if o != "h1_var") or
            ("h1_var" in srcs and t in lines["h1_var"]["index"])]


def main():
    genes, pos, gidx, m = load_axis()
    lines = load_lines(genes, gidx, m)
    lines.pop("_gidx", None)
    v = pd.read_parquet("data/vccp_de_log2fc.parquet")
    vnames = [str(x) for x in v.index]
    V = np.clip(v.to_numpy(dtype=np.float64), -10, 10)
    h1full = lines["h1"]
    idx150 = [h1full["index"][t] for t in vnames if t in h1full["index"]]
    keep150 = [t for t in vnames if t in h1full["index"]]
    assert len(keep150) == 150

    variants = {
        "full285": dict(names=h1full["names"], rows=h1full["rows"]),
        "cache150": dict(names=keep150,
                         rows=h1full["rows"][idx150]),
        "deseq150": dict(names=keep150, rows=V[[vnames.index(t)
                                                for t in keep150]]),
    }
    rows = []
    for L in HOLDS:
        ref = lines[L]
        others = [o for o in BASE if o != L]
        lines["h1_var"] = dict(names=h1full["names"], rows=h1full["rows"],
                               index=h1full["index"])
        main_q = common_queries(lines, others + ["h1_var"], ref)
        X0, ns0 = mix_features(lines, others + ["h1_var"], main_q, pos)
        R = Retrieval(ref["rows"], ref["names"], pos, gidx=gidx)
        P0 = np.zeros((len(main_q), len(genes)))
        P0[:, pos] = X0
        p0 = R.score(P0, main_q, exclude="self", denominator="n")
        print(f"\n=== holdout {L} | baseline queries {len(main_q)} ===",
              flush=True)
        for name, h in variants.items():
            lines["h1_var"] = dict(
                names=h["names"], rows=h["rows"],
                index={t: i for i, t in enumerate(h["names"])})
            q = (main_q if name == "full285"
                 else common_queries(lines, others + ["h1_var"], ref))
            X, ns = mix_features(lines, others + ["h1_var"], q, pos)
            P = np.zeros((len(q), len(genes)))
            P[:, pos] = X
            p1 = R.score(P, q, exclude="self", denominator="n")
            q_common = [t for t in main_q if t in set(q)]
            a = np.array([p0[main_q.index(t)] for t in q_common])
            b = np.array([p1[q.index(t)] for t in q_common])
            d = b - a
            ok = np.isfinite(d)
            lo, hi, mu = boot_ci(d[ok], n=4000, seed=0)
            rho = spearmanr(a[ok], b[ok]).statistic if ok.sum() > 20 else np.nan
            rows.append(dict(holdout=L, variant=name, h1_rows=len(h["names"]),
                             n_queries=len(q), n_paired=int(ok.sum()),
                             mean_sources=float(np.mean(ns)),
                             pds_baseline=float(np.nanmean(a)),
                             pds_variant=float(np.nanmean(b)),
                             delta=mu, lo=lo, hi=hi,
                             spearman=float(rho),
                             mean_abs_change=float(np.mean(np.abs(d[ok])))))
            print(f"  {name:9s} rows {len(h['names']):3d} queries {len(q):5d} "
                  f"PDS {np.nanmean(b):.4f} (base {np.nanmean(a):.4f}) "
                  f"delta {mu:+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"rho {rho:+.3f}", flush=True)
        del lines["h1_var"]

    with open("outputs/h1_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote outputs/h1_sensitivity.csv")


if __name__ == "__main__":
    main()
