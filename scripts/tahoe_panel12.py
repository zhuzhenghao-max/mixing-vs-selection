"""Extension analyses on the 12-line Tahoe-100M drug panel.

Reuses the published PDS definitions on a fixed condition panel so the
original seven lines can be compared with and without the five added lines.
Panel statistics (full mix, saturation, drug exclusion, pilot) use the
log2 fold-change matrix, matching the published Tahoe analysis; calibration
(tahoe_external.py select) uses the Wald statistic.

Usage:
  python tahoe_panel12.py check PANEL_CSV LINES...
  python tahoe_panel12.py full  PANEL_CSV OUT_CSV COND_NPZ LINES...
  python tahoe_panel12.py sat   PANEL_CSV OUT_CSV REPS LINES...
  python tahoe_panel12.py delta PANEL_CSV OUT_CSV LINES...
  python tahoe_panel12.py pilot COND_NPZ OUT_CSV LINES...
"""

import os
import sys

import numpy as np
import pandas as pd

METRIC = "X"


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def pds_fast(P, Y):
    Yn = unit(Y.astype(np.float32))
    Pn = unit(np.asarray(P, dtype=np.float32))
    S = Yn @ Pn.T
    st = np.diag(S).copy()
    rank = (S > st[None, :]).sum(axis=0) + 0.5 * (
        (S == st[None, :]).sum(axis=0) - 1)
    return 1.0 - rank / len(Y)


def pds_drug_excl(P, Y, drugs):
    Yn = unit(Y)
    Pn = unit(P)
    out = np.empty(len(Y))
    for i in range(len(Y)):
        pool = np.where(drugs != drugs[i])[0]
        pool = np.append(pool, i)
        s = Yn[pool] @ Pn[i]
        sv = s[-1]
        rank = int((s > sv).sum() + 0.5 * ((s == sv).sum() - 1))
        out[i] = 1.0 - rank / len(pool)
    return out


def panel_conditions(path):
    d = pd.read_csv(path)
    return list(dict.fromkeys(d["cond"].astype(str)))


def load_lines(paths, panel):
    out = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        name = os.path.basename(p)[:-4]
        cond = [str(c) for c in d["cond"]]
        index = {c: i for i, c in enumerate(cond)}
        miss = [c for c in panel if c not in index]
        rows = np.array([index[c] for c in panel if c in index])
        out[name] = dict(path=p, X=d[METRIC].astype(np.float32)[rows],
                         drug=np.asarray(d["drug"]).astype(str)[rows],
                         cond=[c for c in panel if c in index],
                         n_missing=len(miss), missing=miss[:5])
        print(f"  {name:12s} {out[name]['X'].shape} "
              f"missing={len(miss)} {miss[:3]}", flush=True)
    return out


def align(lines):
    names = list(lines)
    common = set(lines[names[0]]["cond"])
    for n in names[1:]:
        common &= set(lines[n]["cond"])
    common = [c for c in lines[names[0]]["cond"] if c in common]
    for n in names:
        idx = {c: i for i, c in enumerate(lines[n]["cond"])}
        rows = np.array([idx[c] for c in common])
        lines[n]["X"] = lines[n]["X"][rows]
        lines[n]["drug"] = lines[n]["drug"][rows]
        lines[n]["cond"] = common
    print(f"aligned panel {len(common)}", flush=True)
    return lines


def mix(lines, srcs):
    return np.mean([unit(lines[o]["X"]) for o in srcs], axis=0)


def cmd_check(panel, paths):
    lines = load_lines(paths, panel)
    names = list(lines)
    sets = [set(lines[n]["cond"]) for n in names]
    common = set.intersection(*sets)
    print(f"panel {len(panel)}  common-across-lines {len(common)}")
    for n in names:
        print(f"  {n:12s} conds {len(lines[n]['cond'])} "
              f"missing {lines[n]['n_missing']}")
    lines = align(lines)
    act = np.mean([(np.abs(lines[n]["X"]) > 2).mean(axis=1)
                   for n in names], axis=0)
    print("active>=0.005 on common:", int((act >= 0.005).sum()))


def cmd_full(panel, out, cond_out, paths):
    lines = align(load_lines(paths, panel))
    names = list(lines)
    rows, conds = [], {}
    for L in names:
        srcs = [o for o in names if o != L]
        Y = lines[L]["X"]
        full = pds_fast(mix(lines, srcs), Y)
        exc = pds_drug_excl(mix(lines, srcs), Y, lines[L]["drug"])
        single = [float(pds_fast(unit(lines[o]["X"]), Y).mean())
                  for o in srcs]
        conds[L] = full
        rows.append(dict(holdout=L, n_src=len(srcs),
                         pds_full=float(full.mean()),
                         pds_drug_excl=float(exc.mean()),
                         pds_full_median=float(np.median(full)),
                         pds_excl_median=float(np.median(exc)),
                         first_frac=float((full > 0.999).mean()),
                         pds_single=float(np.mean(single)),
                         pds_single_min=float(np.min(single)),
                         pds_single_max=float(np.max(single))))
        print(f"{L:12s} full {full.mean():.4f} excl {exc.mean():.4f} "
              f"single {np.mean(single):.4f}", flush=True)
        del Y
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    np.savez(cond_out, **conds)
    print("saved", out, cond_out)


def cmd_sat(panel, out, reps, paths):
    lines = align(load_lines(paths, panel))
    names = list(lines)
    rng = np.random.default_rng(0)
    rows = []
    for L in names:
        srcs = [o for o in names if o != L]
        Y = lines[L]["X"]
        for rep in range(reps):
            order = list(rng.permutation(srcs))
            acc = np.zeros_like(Y, dtype=np.float32)
            for k in range(1, len(order) + 1):
                acc += unit(lines[order[k - 1]]["X"])
                v = float(pds_fast(acc / k, Y).mean())
                rows.append(dict(holdout=L, rep=rep, k=k, pds=v))
            print(f"{L:12s} rep{rep} last {rows[-1]['pds']:.4f}", flush=True)
        del Y
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print("saved", out)


def cmd_delta(panel, out, paths):
    lines = align(load_lines(paths, panel))
    names = list(lines)
    rows = []
    for L in names:
        srcs = [o for o in names if o != L]
        Y = lines[L]["X"]
        base = pds_fast(mix(lines, srcs), Y)
        for drop in srcs:
            rest = [o for o in srcs if o != drop]
            v = pds_fast(mix(lines, rest), Y)
            d = float(np.mean(base - v))
            rows.append(dict(holdout=L, dropped=drop, delta=d))
        print(f"{L:12s} delta min {min(r['delta'] for r in rows if r['holdout'] == L):+.4f}",
              flush=True)
        del Y
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print("saved", out)


def cmd_pilot(cond_npz, out, reps=50000, k=50, tau=0.92, seed=0):
    z = np.load(cond_npz)
    rng = np.random.default_rng(seed)
    rows = []
    for L in z.files:
        v = z[L]
        idx = rng.integers(0, len(v), (reps, k))
        means = v[idx].mean(axis=1)
        rows.append(dict(holdout=L, k=k, tau=tau,
                         below=float((means < tau).mean()),
                         mean=float(means.mean()),
                         min=float(means.min()), max=float(means.max())))
        print(f"{L:12s} below {tau} {rows[-1]['below'] * 100:.3f}%",
              flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print("saved", out)


def main():
    cmd = sys.argv[1]
    if cmd == "check":
        cmd_check(panel_conditions(sys.argv[2]), sys.argv[3:])
    elif cmd == "full":
        cmd_full(panel_conditions(sys.argv[2]), sys.argv[3], sys.argv[4],
                 sys.argv[5:])
    elif cmd == "sat":
        cmd_sat(panel_conditions(sys.argv[2]), sys.argv[3], int(sys.argv[4]),
                sys.argv[5:])
    elif cmd == "delta":
        cmd_delta(panel_conditions(sys.argv[2]), sys.argv[3], sys.argv[4:])
    elif cmd == "pilot":
        cmd_pilot(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
