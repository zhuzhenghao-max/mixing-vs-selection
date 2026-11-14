"""快速比较混合策略 (Tier B, 每留出系抽样 limit 个查询, 同一靶集).

变体: unit 等权 / raw 等权 / gate / best_self / top2_selfw / consensus / source 去均值.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "scripts")
from bench_common import (Retrieval, load_axis, load_lines, profile_sim,
                          unit_rows)

GATE = -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdouts", default="k562,rpe1,hct116")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", choices=("none", "self"), default="self")
    ap.add_argument("--denominator", choices=("n", "n-1"), default="n")
    ap.add_argument("--kinds", default="",
                    help="逗号分隔的额外变体; equal 总是包含")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    genes, pos, gidx, m = load_axis()
    lines = load_lines(genes, gidx, m)
    lines.pop("_gidx", None)
    holds = args.holdouts.split(",")
    print(f"exclude={args.exclude} denominator={args.denominator}")

    for L in holds:
        if L not in lines:
            continue
        ref = lines[L]
        srcs = [o for o in holds_all(lines) if o != L]
        R = Retrieval(ref["rows"], ref["names"], pos,
                      gidx=gidx if args.exclude == "self" else None)
        q = [t for t in ref["names"] if any(t in lines[o]["index"] for o in srcs)]
        q = list(rng.permutation(q))[:args.limit]
        print(f"\n=== {L}: {len(q)} sampled queries, sources {srcs} ===")

        means = {}
        means_q = {}
        norms = {}
        for o in srcs:
            means[o] = lines[o]["rows"].mean(0)
            qo = [lines[o]["index"][t] for t in q if t in lines[o]["index"]]
            means_q[o] = (lines[o]["rows"][qo].mean(0) if qo
                          else means[o])
            norms[o] = np.linalg.norm(lines[o]["rows"][:, pos], axis=1)

        def sources_for(t):
            return [o for o in srcs if t in lines[o]["index"]]

        def build(kind):
            P = np.zeros((len(q), len(genes)))
            for i, t in enumerate(q):
                os_ = sources_for(t)
                if not os_:
                    continue
                rows = {o: lines[o]["rows"][lines[o]["index"][t]] for o in os_}

                def unit(r):
                    return unit_rows(r[None, :])[0]

                if kind == "equal":
                    P[i] = np.mean([unit(rows[o]) for o in os_], axis=0)
                elif kind == "equal_raw":
                    P[i] = np.mean([rows[o] for o in os_], axis=0)
                elif kind == "equal_center":
                    P[i] = np.mean([unit(rows[o] - means[o]) for o in os_],
                                   axis=0)
                elif kind == "equal_center_q":
                    P[i] = np.mean([unit(rows[o] - means_q[o]) for o in os_],
                                   axis=0)
                elif kind == "gated":
                    g = [o for o in os_
                         if np.isfinite(lines[o]["self_lfc"][lines[o]["index"][t]])
                         and lines[o]["self_lfc"][lines[o]["index"][t]] <= GATE]
                    use = g or os_
                    P[i] = np.mean([unit(rows[o]) for o in use], axis=0)
                elif kind == "gated_raw":
                    g = [o for o in os_
                         if np.isfinite(lines[o]["self_lfc"][lines[o]["index"][t]])
                         and lines[o]["self_lfc"][lines[o]["index"][t]] <= GATE]
                    use = g or os_
                    P[i] = np.mean([rows[o] for o in use], axis=0)
                elif kind == "best_self":
                    sl = {o: lines[o]["self_lfc"][lines[o]["index"][t]]
                          for o in os_}
                    fin = {o: v for o, v in sl.items() if np.isfinite(v)}
                    o = min(fin, key=fin.get) if fin else os_[0]
                    P[i] = rows[o]
                elif kind == "top2_selfw":
                    sl = {o: -lines[o]["self_lfc"][lines[o]["index"][t]]
                          for o in os_}
                    fin = {o: v for o, v in sl.items() if np.isfinite(v) and v > 0}
                    use = sorted(fin, key=fin.get, reverse=True)[:2] or os_
                    if not fin:
                        w = {o: 1.0 for o in use}
                    else:
                        w = {o: fin.get(o, 0.0) for o in use}
                    acc = sum(w[o] * unit(rows[o]) for o in use)
                    P[i] = acc / sum(w.values())
                elif kind == "consensus":
                    if len(os_) == 1:
                        P[i] = rows[os_[0]]
                    else:
                        U = np.stack([unit(rows[o]) for o in os_])
                        S = U @ U.T
                        pick_j = int(np.argmax(S.sum(1)))
                        P[i] = U[pick_j]
                elif kind == "median":
                    P[i] = np.median([unit(rows[o]) for o in os_], axis=0)
                elif kind == "trimmed":
                    U = np.stack([unit(rows[o]) for o in os_])
                    if len(os_) <= 2:
                        P[i] = U.mean(0)
                    else:
                        mu = U.mean(0)
                        mu /= max(np.linalg.norm(mu), 1e-9)
                        drop = int(np.argmin(U @ mu))
                        P[i] = np.delete(U, drop, axis=0).mean(0)
                elif kind == "concordw":
                    U = np.stack([unit(rows[o]) for o in os_])
                    if len(os_) == 1:
                        P[i] = U[0]
                    else:
                        S = U @ U.T
                        np.fill_diagonal(S, 0.0)
                        w = np.maximum(S.sum(1), 0.0)
                        if w.sum() < 1e-9:
                            w = np.ones(len(os_))
                        P[i] = (w[:, None] * U).sum(0) / w.sum()
                elif kind == "normw":
                    w = {o: float(norms[o][lines[o]["index"][t]]) for o in os_}
                    acc = sum(w[o] * unit(rows[o]) for o in os_)
                    P[i] = acc / sum(w.values())
                else:
                    raise ValueError(kind)
            return P

        kinds = ["equal"] + [k for k in args.kinds.split(",") if k] if \
            args.kinds else ["equal", "equal_raw", "equal_center",
                             "equal_center_q", "gated", "gated_raw",
                             "best_self", "top2_selfw", "consensus", "normw",
                             "median", "trimmed", "concordw"]
        res = {}
        for kind in kinds:
            P = build(kind)
            pds = R.score(P, q, exclude=args.exclude,
                          denominator=args.denominator)
            res[kind] = pds
            print(f"  {kind:13s} med {np.nanmedian(pds):.4f} "
                  f"mean {np.nanmean(pds):.4f}")
        for kind in kinds:
            if kind == "equal":
                continue
            d = res[kind] - res["equal"]
            print(f"  d({kind:11s}-equal) med {np.nanmedian(d):+.4f} "
                  f"mean {np.nanmean(d):+.4f} win {np.mean(d > 0):.2f}")


def holds_all(lines):
    return [k for k in ("h1", "k562", "rpe1", "hct116", "hek293t") if k in lines]


if __name__ == "__main__":
    main()
