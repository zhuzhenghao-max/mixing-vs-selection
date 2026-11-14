"""5 系 leave-one-line-out: 单源 / 等权 / 相似度加权 / self-gate / 均值基线.

每个留出系用全池检索 PDS (剔除查询靶基因), 方法之间在共同靶子集上再比较.
"""

import csv
import sys

import numpy as np

sys.path.insert(0, "scripts")
from bench_common import (Retrieval, boot_ci, load_axis, load_lines,
                          profile_sim, unit_rows)

HOLDOUTS = ["h1", "k562", "rpe1", "hct116", "hek293t"]
SIM_TS = [0.05, 0.1, 0.2, 0.5, 1.0]
GATE = -1.0


def loo_run(lines, holdouts, genes, pos, sim_ts=SIM_TS, gate=GATE, verbose=True,
            gidx=None, exclude="self", denominator="n"):
    if exclude == "self" and gidx is None:
        raise ValueError("exclude='self' requires gidx")
    rows_out = {}
    summary = []
    for L in holdouts:
        if L not in lines or len(lines[L]["names"]) == 0:
            continue
        ref = lines[L]
        srcs = [o for o in holdouts if o != L and o in lines
                and len(lines[o]["names"]) > 0]
        R = Retrieval(ref["rows"], ref["names"], pos,
                      gidx=gidx if exclude == "self" else None)
        Rr = Retrieval(ref["rows"], ref["names"], pos, resid=True,
                       gidx=gidx if exclude == "self" else None)
        q = [t for t in ref["names"]
             if any(t in lines[o]["index"] for o in srcs)]
        sim = {}
        for o in srcs:
            a, b = ref["profile"], lines[o]["profile"]
            sim[o] = (profile_sim(a, b)
                      if a is not None and b is not None else np.nan)
        sim_ok = all(np.isfinite(sim[o]) for o in srcs)
        if verbose:
            print(f"\n=== hold-out {L}: pool {len(ref['names'])}, queries {len(q)}, "
                  f"sources {srcs} ===")
            print("  sim " + " ".join(f"{o}:{sim[o]:+.2f}" for o in srcs))

        def preds_single(o):
            return {t: lines[o]["rows"][lines[o]["index"][t]]
                    for t in q if t in lines[o]["index"]}

        src_means = {o: lines[o]["rows"].mean(0) for o in srcs}

        def preds_equal(subset=None, gate=False, selfw=False):
            out = {}
            for t in (subset if subset is not None else q):
                acc = np.zeros(len(genes))
                wsum = 0.0
                for o in srcs:
                    if t not in lines[o]["index"]:
                        continue
                    s = lines[o]["self_lfc"][lines[o]["index"][t]]
                    w = 1.0
                    if gate:
                        if not np.isfinite(s) or s > GATE:
                            continue
                        if selfw:
                            w = -s
                    acc += w * unit_rows(
                        lines[o]["rows"][lines[o]["index"][t]][None, :])[0]
                    wsum += w
                if wsum > 0:
                    out[t] = acc / wsum
            return out

        def preds_equal_raw(subset=None):
            out = {}
            for t in (subset if subset is not None else q):
                acc = np.zeros(len(genes))
                wsum = 0.0
                for o in srcs:
                    if t not in lines[o]["index"]:
                        continue
                    acc += lines[o]["rows"][lines[o]["index"][t]]
                    wsum += 1.0
                if wsum > 0:
                    out[t] = acc / wsum
            return out

        def preds_equal_center(subset=None):
            out = {}
            for t in (subset if subset is not None else q):
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

        def preds_top2_selfw(subset=None):
            out = {}
            for t in (subset if subset is not None else q):
                os_ = [o for o in srcs if t in lines[o]["index"]]
                if not os_:
                    continue
                sl = {o: -lines[o]["self_lfc"][lines[o]["index"][t]]
                      for o in os_}
                fin = {o: v for o, v in sl.items()
                       if np.isfinite(v) and v > 0}
                use = sorted(fin, key=fin.get, reverse=True)[:2] or os_
                w = {o: fin.get(o, 0.0) for o in use} if fin else \
                    {o: 1.0 for o in use}
                acc = sum(w[o] * unit_rows(
                    lines[o]["rows"][lines[o]["index"][t]][None, :])[0]
                    for o in use)
                sw = sum(w.values())
                if sw > 0:
                    out[t] = acc / sw
            return out

        def preds_sim(T):
            out = {}
            for t in q:
                acc = np.zeros(len(genes))
                wsum = 0.0
                for o in srcs:
                    if t not in lines[o]["index"]:
                        continue
                    w = np.exp(sim[o] / T)
                    acc += w * unit_rows(
                        lines[o]["rows"][lines[o]["index"][t]][None, :])[0]
                    wsum += w
                if wsum > 0:
                    out[t] = acc / wsum
            return out

        q_gate = [t for t in q if any(
            t in lines[o]["index"] and
            np.isfinite(lines[o]["self_lfc"][lines[o]["index"][t]]) and
            lines[o]["self_lfc"][lines[o]["index"][t]] <= GATE
            for o in srcs)]

        tot = np.zeros(len(genes))
        cnt = 0
        for o in srcs:
            tot += lines[o]["rows"].sum(0)
            cnt += len(lines[o]["rows"])
        mvec = unit_rows((tot / max(cnt, 1))[None, :])[0]

        methods = {}
        for o in srcs:
            methods[f"single_{o}"] = preds_single(o)
        methods["equal"] = preds_equal()
        methods["equal_raw"] = preds_equal_raw()
        methods["equal_center"] = preds_equal_center()
        methods["gated_equal"] = preds_equal(q_gate, gate=True)
        methods["gated_selfw"] = preds_equal(q_gate, gate=True, selfw=True)
        methods["top2_selfw"] = preds_top2_selfw()
        if sim_ok:
            for T in sim_ts:
                methods[f"simT{T:g}"] = preds_sim(T)
        methods["mean"] = {t: mvec for t in q}

        P = {}
        for name, preds in methods.items():
            qs = [t for t in q if t in preds]
            if len(qs) < 5:
                continue
            Pmat = np.stack([preds[t] for t in qs])
            raw = R.score(Pmat, qs, exclude=exclude,
                          denominator=denominator)
            res = Rr.score(Pmat, qs, exclude=exclude,
                           denominator=denominator)
            P[name] = (qs, raw, res)
            if verbose:
                print(f"  {name:14s} n={len(qs):4d} "
                      f"pds_raw {np.nanmedian(raw):.4f} ({np.nanmean(raw):.4f}) "
                      f"pds_res {np.nanmedian(res):.4f} ({np.nanmean(res):.4f})")
            rows_out[(L, name)] = dict(zip(qs, zip(raw, res)))

        # 共同子集: 混合方法之间 (单源覆盖太窄, 交集常为空)
        mix_names = [n for n in P if not n.startswith("single_")]
        common_mix = (sorted(set.intersection(
            *[set(P[n][0]) for n in mix_names])) if mix_names else [])
        common_all = sorted(set.intersection(*[set(P[n][0]) for n in P]))
        blocks = [("common_mix", common_mix)]
        if len(common_all) >= 5 and len(common_all) != len(common_mix):
            blocks.append(("common_all", common_all))
        for tag, subset in blocks:
            if len(subset) < 5:
                continue
            if verbose:
                print(f"  --- {tag} n={len(subset)} ---")
            names_here = ([n for n in P if not n.startswith("single_")]
                          if tag == "common_mix" else list(P))
            for name in names_here:
                qs, raw, res = P[name]
                qset = set(qs)
                if not all(t in qset for t in subset):
                    continue
                idx = [qs.index(t) for t in subset]
                r = np.array(raw)[idx]
                lo, hi, mu = boot_ci(r)
                summary.append(dict(holdout=L, method=name, n=len(subset),
                                    pds_raw_med=float(np.nanmedian(r)),
                                    pds_raw_mean=mu, ci_lo=lo, ci_hi=hi,
                                    pds_res_med=float(np.nanmedian(np.array(res)[idx])),
                                    subset=tag))
                if verbose:
                    print(f"    {name:14s} pds_raw {np.nanmedian(r):.4f} "
                          f"[{lo:.4f},{hi:.4f}]")
        # 各自子集也记一份
        for name, (qs, raw, res) in P.items():
            summary.append(dict(holdout=L, method=name, n=len(qs),
                                pds_raw_med=float(np.nanmedian(raw)),
                                pds_raw_mean=float(np.nanmean(raw)),
                                ci_lo=np.nan, ci_hi=np.nan,
                                pds_res_med=float(np.nanmedian(res)),
                                subset="own"))

    return summary, rows_out


def save_results(prefix, summary, rows_out):
    flat = [(L, name, t, raw, res)
            for (L, name), d in rows_out.items()
            for t, (raw, res) in d.items()]
    np.savez(f"{prefix}.npz",
             holdout=np.array([f[0] for f in flat]),
             method=np.array([f[1] for f in flat]),
             target=np.array([f[2] for f in flat]),
             pds_raw=np.array([f[3] for f in flat], dtype=np.float64),
             pds_res=np.array([f[4] for f in flat], dtype=np.float64))
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nsaved {prefix}.csv / .npz")


def main():
    genes, pos, gidx, m = load_axis()
    lines = load_lines(genes, gidx, m)
    lines.pop("_gidx", None)
    summary, rows_out = loo_run(lines, HOLDOUTS, genes, pos, gidx=gidx)
    save_results("outputs/bench_transfer_v2", summary, rows_out)


if __name__ == "__main__":
    main()
