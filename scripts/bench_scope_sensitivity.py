"""PDS exclusion-scope sensitivity: panel-wide exclusion with denominator n-1.

The canonical benchmark removes only the query's target gene and uses PDS
denominator n.  The cell-eval2 v2 / VCC 2026 preset removes every panel target
gene from the ranked feature space and uses denominator n-1.  This script
recomputes the key comparisons under that convention for both tiers.

Usage: python scripts/bench_scope_sensitivity.py
"""

import csv
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "scripts")
from bench_common import (Retrieval, boot_ci, load_axis, load_lines,
                          load_nadig_lines, unit_rows)

GATE = -1.0
TIER_A_HOLDOUTS = ["k562", "rpe1", "hepg2", "jurkat"]
TIER_B_HOLDOUTS = ["h1", "k562", "rpe1", "hct116", "hek293t"]


def build_predictions(srcs, lines, genes, q, q_gate):
    src_means = {o: lines[o]["rows"].mean(0) for o in srcs}
    tot = np.zeros(len(genes))
    cnt = 0
    for o in srcs:
        tot += lines[o]["rows"].sum(0)
        cnt += len(lines[o]["rows"])
    mvec = unit_rows((tot / max(cnt, 1))[None, :])[0]

    def mix(subset, center=False, gate=False):
        out = {}
        for t in subset:
            acc = np.zeros(len(genes))
            wsum = 0.0
            for o in srcs:
                if t not in lines[o]["index"]:
                    continue
                s = lines[o]["self_lfc"][lines[o]["index"][t]]
                if gate and (not np.isfinite(s) or s > GATE):
                    continue
                r = lines[o]["rows"][lines[o]["index"][t]]
                if center:
                    r = r - src_means[o]
                acc += unit_rows(r[None, :])[0]
                wsum += 1.0
            if wsum > 0:
                out[t] = acc / wsum
        return out

    def raw_mix(subset):
        out = {}
        for t in subset:
            acc = np.zeros(len(genes))
            wsum = 0.0
            for o in srcs:
                if t in lines[o]["index"]:
                    acc += lines[o]["rows"][lines[o]["index"][t]]
                    wsum += 1.0
            if wsum > 0:
                out[t] = acc / wsum
        return out

    def top2(subset):
        out = {}
        for t in subset:
            os_ = [o for o in srcs if t in lines[o]["index"]]
            if not os_:
                continue
            sl = {o: -lines[o]["self_lfc"][lines[o]["index"][t]]
                  for o in os_}
            fin = {o: v for o, v in sl.items()
                   if np.isfinite(v) and v > 0}
            use = sorted(fin, key=fin.get, reverse=True)[:2] or os_
            w = ({o: fin.get(o, 0.0) for o in use} if fin
                 else {o: 1.0 for o in use})
            acc = sum(w[o] * unit_rows(
                lines[o]["rows"][lines[o]["index"][t]][None, :])[0]
                for o in use)
            sw = sum(w.values())
            if sw > 0:
                out[t] = acc / sw
        return out

    return {
        "equal": mix(q),
        "mean": {t: mvec for t in q},
        "equal_raw": raw_mix(q),
        "equal_center": mix(q, center=True),
        "gated_equal": mix(q_gate, gate=True),
        "top2_selfw": top2(q),
    }


def score_panel(R, qrows, qnames, panel_cols, denominator="n-1", chunk=512):
    """Cosine PDS with a fixed panel-wide excluded column set."""
    out = np.full(len(qnames), np.nan)
    valid = [k for k, t in enumerate(qnames) if t in R.ridx]
    if panel_cols:
        factor = np.sqrt(np.maximum(
            1.0 - np.sum(R.Rn[:, panel_cols] ** 2, axis=1), 1e-12))
    else:
        factor = np.ones(R.n)
    for start in range(0, len(valid), chunk):
        ks = valid[start:start + chunk]
        Q = qrows[ks][:, R.pos].astype(np.float64)
        if panel_cols:
            Q[:, panel_cols] = 0.0
        Qn = Q / np.maximum(
            np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
        S = R.Rn @ Qn.T
        if panel_cols:
            S /= factor[:, None]
        for j, k in enumerate(ks):
            col = S[:, j]
            sv = col[R.ridx[qnames[k]]]
            rank0 = np.sum(col > sv) + 0.5 * (np.sum(col == sv) - 1)
            D = R.n if denominator == "n" else R.n - 1
            out[k] = 1.0 if D <= 0 else 1.0 - rank0 / D
    return out


def run_tier(tier, lines, holdouts, genes, pos, gidx, writer, npz_rows):
    seen = set()
    for L in holdouts:
        if L not in lines or not len(lines[L]["names"]):
            continue
        ref = lines[L]
        srcs = [o for o in holdouts if o != L and o in lines
                and len(lines[o]["names"]) > 0]
        R = Retrieval(ref["rows"], ref["names"], pos, gidx=gidx)
        q = [t for t in ref["names"]
             if any(t in lines[o]["index"] for o in srcs)]
        q_gate = [t for t in q if any(
            t in lines[o]["index"]
            and np.isfinite(lines[o]["self_lfc"][lines[o]["index"][t]])
            and lines[o]["self_lfc"][lines[o]["index"][t]] <= GATE
            for o in srcs)]
        panel_cols = sorted({R.local[gidx[t]] for t in ref["names"]
                             if t in gidx and gidx[t] in R.local})
        print(f"\n=== {tier} {L}: pool {len(ref['names'])}, queries {len(q)}, "
              f"gate {len(q_gate)}, eval positions {len(pos)}, "
              f"panel removes {len(panel_cols)}, remaining "
              f"{len(pos) - len(panel_cols)} ===")

        preds = build_predictions(srcs, lines, genes, q, q_gate)
        pds = {}
        for name, pred in preds.items():
            qs = list(pred.keys())
            Pmat = np.stack([pred[t] for t in qs])
            v = score_panel(R, Pmat, qs, panel_cols,
                            denominator="n-1")
            pds[name] = (qs, v)
            print(f"  {name:14s} n={len(qs):5d} "
                  f"mean {np.nanmean(v):.4f} median {np.nanmedian(v):.4f}")
            for t, x in zip(qs, v):
                npz_rows.append((tier, L, name, t, float(x)))

            if name == "equal":
                writer.writerow([tier, L, name, len(qs),
                                 np.nanmean(v), np.nanmedian(v),
                                 0.0, np.nan, np.nan, np.nan, np.nan,
                                 "all queries"])
                seen.add((tier, L, name))
                continue
            base_qs, base_v = pds["equal"]
            common = sorted(set(qs) & set(base_qs))
            fi = {t: i for i, t in enumerate(qs)}
            bi = {t: i for i, t in enumerate(base_qs)}
            a = np.array([v[fi[t]] for t in common])
            b = np.array([base_v[bi[t]] for t in common])
            d = a - b
            lo, hi, mu = boot_ci(d, n=4000)
            try:
                p = float(wilcoxon(d, alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
            win = float(np.mean(d > 0))
            print(f"    d({name:12s}-equal) n={len(common):5d} "
                  f"med {np.median(d):+.4f} mean {mu:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] p={p:.2e} win={win:.2f}")
            writer.writerow([tier, L, name, len(common),
                             np.nanmean(a), np.nanmedian(a),
                             mu, lo, hi, p, win,
                             "gate subset" if name == "gated_equal"
                             else "all queries"])


def main():
    genes, pos, gidx, _ = load_axis()
    lines_b = load_lines(genes, gidx, _)
    lines_b.pop("_gidx", None)
    lines_a = load_nadig_lines(genes, gidx)
    z = np.load("outputs/nadig_rows.npz", allow_pickle=True)
    in_g = np.zeros(len(genes), dtype=bool)
    in_g[z["gpos"]] = True
    pos_a = pos[in_g[pos]]

    rows = []
    npz_rows = []
    with open("outputs/bench_scope_sensitivity.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Tier", "Held-out", "Method", "n", "Mean", "Median",
                    "Delta mean vs equal", "CI low", "CI high", "p",
                    "Win rate", "Comparison set"])
        run_tier("A", lines_a, TIER_A_HOLDOUTS, genes, pos_a, gidx, w,
                 npz_rows)
        run_tier("B", lines_b, TIER_B_HOLDOUTS, genes, pos, gidx, w,
                 npz_rows)

    np.savez("outputs/bench_scope_sensitivity.npz",
             tier=np.array([r[0] for r in npz_rows]),
             holdout=np.array([r[1] for r in npz_rows]),
             method=np.array([r[2] for r in npz_rows]),
             target=np.array([r[3] for r in npz_rows]),
             pds=np.array([r[4] for r in npz_rows], dtype=np.float64))
    print("\nsaved outputs/bench_scope_sensitivity.csv/.npz")


if __name__ == "__main__":
    main()
