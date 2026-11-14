"""Secondary metrics for the Tier A benchmark and the GEARS baseline.

Rebuilds equal / mean / centered / gated predictions on the evaluation axis,
aligns them with the stored GEARS predictions, and scores every query with
cosine similarity, Pearson correlation, top-50 Jaccard overlap, and relative
MSE after per-query scalar calibration. Target coordinates are excluded from
both the prediction and the truth, matching the PDS protocol.

The script also recomputes PDS from the rebuilt predictions and checks it
against the stored benchmark values, so the metric table cannot silently use
misaligned arrays.
"""

import csv
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "scripts")
from bench_common import (Retrieval, boot_ci, load_axis, load_nadig_lines,
                          unit_rows)

HOLDS = ["k562", "rpe1", "hepg2", "jurkat"]
METHODS = ["equal", "equal_center", "mean", "gears"]
METRICS = ["cosine", "pearson", "jaccard50", "rel_mse_cal", "rel_mse_raw"]


def row_metrics(P, Y, cols, k=50):
    n = P.shape[0]
    Pz = np.asarray(P, dtype=np.float64).copy()
    Yz = np.asarray(Y, dtype=np.float64).copy()
    rows = np.arange(n)
    have = np.array([c is not None for c in cols])
    if have.any():
        cols_v = np.array([c for c in cols if c is not None], dtype=int)
        Pz[rows[have], cols_v] = 0.0
        Yz[rows[have], cols_v] = 0.0

    pn = np.linalg.norm(Pz, axis=1)
    yn = np.linalg.norm(Yz, axis=1)
    cos = (Pz * Yz).sum(1) / np.maximum(pn * yn, 1e-12)

    Pc = Pz - Pz.mean(1, keepdims=True)
    Yc = Yz - Yz.mean(1, keepdims=True)
    pear = (Pc * Yc).sum(1) / np.maximum(
        np.linalg.norm(Pc, axis=1) * np.linalg.norm(Yc, axis=1), 1e-12)

    kk = min(k, Yz.shape[1] - 1)
    ytop = np.argpartition(-np.abs(Yz), kk, axis=1)[:, :kk]
    ptop = np.argpartition(-np.abs(Pz), kk, axis=1)[:, :kk]
    inter = np.array([np.intersect1d(ytop[i], ptop[i]).size
                      for i in range(n)])
    jac = inter / (2 * kk - inter)

    yvar = (Yz ** 2).mean(1)
    rel_raw = ((Pz - Yz) ** 2).mean(1) / np.maximum(yvar, 1e-12)
    alpha = (Pz * Yz).sum(1) / np.maximum((Pz ** 2).sum(1), 1e-12)
    rel_cal = ((alpha[:, None] * Pz - Yz) ** 2).mean(1) / \
        np.maximum(yvar, 1e-12)

    return dict(cosine=cos, pearson=pear, jaccard50=jac,
                rel_mse_cal=rel_cal, rel_mse_raw=rel_raw)


def make_preds(lines, L, srcs, q, genes):
    ref = lines[L]
    src_means = {o: lines[o]["rows"].mean(0) for o in srcs}

    def equal_rows(subset, gate=False):
        out = {}
        for t in subset:
            acc = np.zeros(len(genes))
            wsum = 0.0
            for o in srcs:
                if t not in lines[o]["index"]:
                    continue
                s = lines[o]["self_lfc"][lines[o]["index"][t]]
                if gate:
                    if not np.isfinite(s) or s > -1.0:
                        continue
                acc += unit_rows(
                    lines[o]["rows"][lines[o]["index"][t]][None, :])[0]
                wsum += 1.0
            if wsum > 0:
                out[t] = acc / wsum
        return out

    def center_rows(subset):
        out = {}
        for t in subset:
            acc = np.zeros(len(genes))
            wsum = 0.0
            for o in srcs:
                if t not in lines[o]["index"]:
                    continue
                r = lines[o]["rows"][lines[o]["index"][t]] - src_means[o]
                acc += unit_rows(r[None, :])[0]
                wsum += 1.0
            if wsum > 0:
                out[t] = acc / wsum
        return out

    tot = np.zeros(len(genes))
    cnt = 0
    for o in srcs:
        tot += lines[o]["rows"].sum(0)
        cnt += len(lines[o]["rows"])
    mvec = unit_rows((tot / max(cnt, 1))[None, :])[0]

    q_gate = [t for t in q if any(
        t in lines[o]["index"] and
        np.isfinite(lines[o]["self_lfc"][lines[o]["index"][t]]) and
        lines[o]["self_lfc"][lines[o]["index"][t]] <= -1.0
        for o in srcs)]

    preds = {
        "equal": equal_rows(q),
        "equal_center": center_rows(q),
        "gated_equal": equal_rows(q_gate, gate=True),
        "mean": {t: mvec for t in q},
    }
    return preds, q, q_gate


def compact(preds, qs, posn, gidx, compact_input=False):
    P = np.stack([preds[t] for t in qs])
    if not compact_input:
        P = P[:, posn]
    cols = []
    local = {int(g): i for i, g in enumerate(posn)}
    for t in qs:
        j = gidx.get(t)
        cols.append(local.get(j) if j is not None else None)
    return P, cols


def pair_stats(diffs, seed=0):
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    lo, hi, mu = boot_ci(d, n=4000, seed=seed)
    p = float(wilcoxon(d, zero_method="wilcox").pvalue) if d.any() else np.nan
    return mu, lo, hi, float(np.mean(d > 0)), p


def main():
    genes, pos, gidx, m = load_axis()
    lines = load_nadig_lines(genes, gidx)
    z = np.load("outputs/nadig_rows.npz", allow_pickle=True)
    in_g = np.zeros(len(genes), dtype=bool)
    in_g[z["gpos"]] = True
    posn = pos[in_g[pos]]
    print(f"eval positions {len(posn)}", flush=True)

    gz = np.load("outputs/gears_baseline_A.npz", allow_pickle=True)
    eq = np.load("outputs/bench_transfer_nadig_v2.npz", allow_pickle=True)
    eq_ho, eq_me, eq_tg = (eq[k].astype(str) for k in
                           ("holdout", "method", "target"))
    pds_store = {(h, me, t): float(v) for h, me, t, v in
                 zip(eq_ho, eq_me, eq_tg, eq["pds_raw"])}

    summ, pairs, perq = [], [], {}
    for L in HOLDS:
        if L not in lines:
            continue
        ref = lines[L]
        srcs = [o for o in HOLDS if o != L and o in lines]
        q = [t for t in ref["names"]
             if any(t in lines[o]["index"] for o in srcs)]
        preds, q, q_gate = make_preds(lines, L, srcs, q, genes)

        gt = [str(x) for x in gz[f"{L}|targets"]]
        gpred = {t: gz[f"{L}|pred"][i].astype(np.float64)
                 for i, t in enumerate(gt)}
        preds["gears"] = gpred

        Yfull = ref["rows"].astype(np.float64)
        Y = Yfull[:, posn]

        for subset_name, qs in (
                ("all", [t for t in q if all(
                    t in preds[me] for me in METHODS)]),
                ("gated", [t for t in q_gate if all(
                    t in preds[me] for me in ("equal", "gated_equal",
                                              "gears"))])):
            if len(qs) < 5:
                continue
            methods_here = list(METHODS)
            if subset_name == "gated":
                methods_here = ["equal", "gated_equal", "equal_center",
                                "mean", "gears"]
            for me in methods_here:
                if me not in preds or any(t not in preds[me] for t in qs):
                    continue
                P, cols = compact(preds[me], qs, posn, gidx,
                                  compact_input=(me == "gears"))
                r = row_metrics(P, Y[[ref["index"][t] for t in qs]], cols)
                perq[(L, subset_name, me)] = r
                summ.append(dict(
                    holdout=L, subset=subset_name, method=me, n=len(qs),
                    cosine_mean=float(np.nanmean(r["cosine"])),
                    cosine_med=float(np.nanmedian(r["cosine"])),
                    pearson_mean=float(np.nanmean(r["pearson"])),
                    pearson_med=float(np.nanmedian(r["pearson"])),
                    jaccard50_mean=float(np.nanmean(r["jaccard50"])),
                    jaccard50_med=float(np.nanmedian(r["jaccard50"])),
                    rel_mse_cal_mean=float(np.nanmean(r["rel_mse_cal"])),
                    rel_mse_cal_med=float(np.nanmedian(r["rel_mse_cal"])),
                    rel_mse_raw_mean=float(np.nanmean(r["rel_mse_raw"])),
                ))
                print(f"{L:7s} {subset_name:5s} {me:13s} n={len(qs):4d} "
                      f"cos {np.nanmean(r['cosine']):.4f} "
                      f"pear {np.nanmean(r['pearson']):.4f} "
                      f"jac50 {np.nanmean(r['jaccard50']):.4f} "
                      f"relMSEcal {np.nanmean(r['rel_mse_cal']):.4f}",
                      flush=True)

            comps = (("equal", "gears"), ("equal", "mean"),
                     ("equal", "equal_center"))
            if subset_name == "gated":
                comps = (("equal", "gated_equal"),)
            for a, b in comps:
                if (L, subset_name, a) not in perq or \
                        (L, subset_name, b) not in perq:
                    continue
                ra = perq[(L, subset_name, a)]
                rb = perq[(L, subset_name, b)]
                for met in METRICS:
                    if met.startswith("rel_mse"):
                        d = rb[met] - ra[met]
                    else:
                        d = ra[met] - rb[met]
                    mu, lo, hi, win, p = pair_stats(d)
                    pairs.append(dict(
                        holdout=L, subset=subset_name, comparison=f"{a}_vs_{b}",
                        metric=met, n=len(d), mean_diff=mu, ci_lo=lo,
                        ci_hi=hi, win=win, p_wilcoxon=p,
                        median_diff=float(np.nanmedian(d))))

        # PDS audit: rebuilt equal must match the stored benchmark values,
        # and GEARS PDS from stored predictions must match its own record.
        local = {int(g): i for i, g in enumerate(posn)}
        lgidx = {t: local[gidx[t]] for t in ref["names"]
                 if t in gidx and gidx[t] in local}
        ret = Retrieval(Y, ref["names"], np.arange(len(posn)), gidx=lgidx)
        gpos = {t: i for i, t in enumerate(gt)}
        for me in ("equal", "equal_center", "mean", "gated_equal",
                   "gears"):
            if me not in preds:
                continue
            qs = [t for t in q if t in preds[me]]
            P, _ = compact(preds[me], qs, posn, gidx,
                           compact_input=(me == "gears"))
            pds = ret.score(P, qs, exclude="self", denominator="n")
            if me == "gears":
                refv = np.array([gz[f"{L}|pds"][gpos[t]] for t in qs])
            else:
                refv = np.array([pds_store.get((L, me, t), np.nan)
                                 for t in qs])
            ok = np.isfinite(refv)
            print(f"  PDS audit {L}/{me}: n={ok.sum()} "
                  f"max|diff|={np.nanmax(np.abs(pds[ok] - refv[ok])):.2e}",
                  flush=True)

    with open("outputs/secondary_metrics_A.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    with open("outputs/secondary_metrics_A_pairs.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)
    np.savez("outputs/secondary_metrics_A.npz",
             **{f"{L}|{s}|{me}|{k}": v
                for (L, s, me), d in perq.items() for k, v in d.items()})
    print(f"\nwrote {len(summ)} method rows and {len(pairs)} comparisons")


if __name__ == "__main__":
    main()
