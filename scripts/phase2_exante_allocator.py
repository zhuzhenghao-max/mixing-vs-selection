"""Ex-ante test of the source-line gain: can source lines predict it?

For every leave-one-source-out state in the Phase 2 panel, the actual gain of
adding a source line is measured on the target line. Here we estimate the same
gain without touching the target line: treat each source line in the state as
a pseudo-target and measure how much the candidate line improves its transfer
from the other source lines. Correlation between pseudo-gain and actual gain
is the evidence for a source-only allocation rule.
"""

import csv

import numpy as np
from scipy.stats import spearmanr, pearsonr

import sys
sys.path.insert(0, "scripts")
from bench_common import load_axis, load_nadig_lines
from bench_supervised import build_features
from phase1_selection_pilot import FastRetrieval


def load_actual():
    out = {}
    with open("outputs/phase2_source_budget_A.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["strategy"] != "equal" or int(r["k"]) != -1:
                continue
            key = (r["holdout"], frozenset(r["srcs"].split(",")),
                   r["added"])
            out[key] = float(r["delta"])
    return out


def main():
    genes, pos, gidx, m = load_axis()
    lines = load_nadig_lines(genes, gidx)
    lines.pop("_gidx", None)
    names = list(lines)
    q = [t for t in lines[names[0]]["names"]
         if all(t in lines[o]["index"] for o in names)]
    print(f"shared queries: {len(q)}")

    R = {o: FastRetrieval(lines[o]["rows"], lines[o]["names"], pos, gidx)
         for o in names}
    actual = load_actual()

    rows = []
    for L in names:
        srcs = [o for o in names if o != L]
        for a in range(3):
            for b in range(a + 1, 3):
                S = (srcs[a], srcs[b])
                i = [o for o in srcs if o not in S][0]
                act = actual.get((L, frozenset(S), i))
                if act is None:
                    continue
                per_o = []
                for o in S:
                    base = [x for x in S if x != o]
                    aug = [x for x in S + (i,) if x != o]
                    Xb = build_features(lines, base, q, pos, "mix")
                    Xa = build_features(lines, aug, q, pos, "mix")
                    pb = R[o].score_pos(Xb, q)
                    pa = R[o].score_pos(Xa, q)
                    per_o.append(float(np.nanmean(pa - pb)))
                rows.append(dict(
                    holdout=L, state="+".join(S), added=i,
                    actual=act, pseudo=float(np.mean(per_o)),
                    pseudo_1=per_o[0], pseudo_2=per_o[1]))
                print(f"  {L:6s} {rows[-1]['state']:13s} +{i:6s} "
                      f"actual {act:+.4f} pseudo {rows[-1]['pseudo']:+.4f} "
                      f"({per_o[0]:+.4f},{per_o[1]:+.4f})", flush=True)

    a = np.array([r["actual"] for r in rows])
    p = np.array([r["pseudo"] for r in rows])
    rho, pv = spearmanr(p, a)
    r, rpv = pearsonr(p, a)
    print(f"\nn={len(rows)}  Spearman {rho:+.3f} (p={pv:.3g})  "
          f"Pearson {r:+.3f} (p={rpv:.3g})")
    for L in names:
        sub = [r for r in rows if r["holdout"] == L]
        aa = np.array([r["actual"] for r in sub])
        pp = np.array([r["pseudo"] for r in sub])
        if len(sub) >= 3:
            rr, ppv = spearmanr(pp, aa)
        else:
            rr, ppv = np.nan, np.nan
        best_p = sub[int(np.argmax(pp))]["added"]
        best_a = sub[int(np.argmax(aa))]["added"]
        print(f"  {L:6s} rho {rr:+.3f} (p={ppv:.2f})  "
              f"pseudo-best {best_p:6s}  actual-best {best_a:6s}  "
              f"{'match' if best_p == best_a else 'miss'}")

    sign = np.mean((p > 0) == (a > 0))
    print(f"sign agreement: {sign:.2f} ({int(sign * len(rows))}/{len(rows)})")

    with open("outputs/phase2_exante_allocator.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote outputs/phase2_exante_allocator.csv")


if __name__ == "__main__":
    main()
