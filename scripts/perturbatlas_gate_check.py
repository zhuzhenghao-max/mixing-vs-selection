"""PA 靶级门槛 + 权重检验 (离线, 用已保存的聚合行).

对同一靶子集比较 mix vs gwps 的 PDS 增益; 门槛基于源内质量信号:
  self_lfc (敲低效率), n_comp (对比数), coverage.
"""

import re

import numpy as np

LOG = "outputs/perturbatlas_multi_check.log"
ANCHOR = None


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def parse_self():
    txt = open(LOG, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"^  (\S+)\s+n=(\d+) coverage=\[([^\]]*)\] "
                         r"self_lfc=\[([^\]]*)\]", txt, re.M):
        t = m.group(1)
        cov = [int(x) for x in m.group(3).split(",") if x.strip()]
        sf = []
        for x in m.group(4).split(","):
            x = x.strip()
            sf.append(np.nan if x == "nan" else float(x))
        out[t] = dict(cov=cov, self=sf)
    return out


def parse_pair():
    txt = open(LOG, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"^  (\S+)\s+n=(\d+) pair_cos mean ([+-][0-9.]+) "
                         r"min ([+-][0-9.]+) max ([+-][0-9.]+)", txt, re.M):
        out[m.group(1)] = float(m.group(3))
    return out


def parse_coverage():
    txt = open(LOG, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"^  (\S+)\s+n=(\d+) coverage=\[([^\]]*)\]", txt, re.M):
        out[m.group(1)] = [int(x) for x in m.group(3).split(",") if x.strip()]
    return out


def purity_reach(rank_scores, real, signs, p0=0.9):
    order = np.argsort(-rank_scores)
    order = [j for j in order if real[j]]
    if not order:
        return 0.0
    agree = np.cumsum([1.0 if signs[j] else 0.0 for j in order])
    purity = agree / np.arange(1, len(order) + 1)
    ok = np.where(purity >= p0)[0]
    if len(ok) == 0:
        return 0.0
    return float(ok[-1] + 1) / float(real.sum())


def main():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes = m["genes"].astype(str)
    tn = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    pos = np.where(filt & np.all(np.isfinite(LFC), 0))[0]
    ref = LFC[:, pos].astype(np.float64)
    R = ref - ref.mean(0)
    refn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
    gidx = {g: i for i, g in enumerate(genes)}
    tidx = {t: i for i, t in enumerate(tn)}

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"].astype(str)]
    grows = g["rows"].astype(np.float64)
    gi = {t: i for i, t in enumerate(gnames)}
    gwps = {t: grows[gi[t]] for t in tn if t in gi}

    d = np.load("outputs/perturbatlas_multi_h1.npz", allow_pickle=True)
    agg = {}
    for label in ("single_max", "mean_all", "mean_cov5k", "med_cov5k",
                  "selfw_cov5k"):
        ks = [str(x) for x in d[f"{label}_names"]]
        agg[label] = {t: d[label][i].astype(np.float64) for i, t in enumerate(ks)}

    meta = parse_self()
    cov = parse_coverage()
    pair = parse_pair()
    print(f"parsed meta {len(meta)} targets")

    def pds(vec, t):
        i = tidx[t]
        a = vec[pos]
        an = unit(a)
        s = refn @ an
        rank = 1.0 + np.sum(s > s[i]) + 0.5 * (np.sum(s == s[i]) - 1)
        return 1.0 - (rank - 1.0) / (len(tn) - 1.0)

    # precompute per-target pds at each (agg, w)
    P = {}
    ws = (0.2, 0.3, 0.4, 0.5)
    for label, vecs in agg.items():
        for w in ws:
            for t, v in vecs.items():
                if t not in gwps:
                    continue
                mix = w * unit(v) + (1 - w) * unit(gwps[t])
                P[(label, w, t)] = pds(mix, t)
    base = {t: pds(gwps[t], t) for t in gwps}

    def gate(t, kind):
        mm = meta.get(t, {})
        sf = mm.get("self", [])
        cv = mm.get("cov", [])
        fin = [x for x in sf if np.isfinite(x)]
        mins = min(fin) if fin else np.inf
        ncomp = len(sf)
        if kind == "all":
            return True
        if kind == "self1":
            return mins <= -1.0
        if kind.startswith("cons"):
            thr = float(kind[4:]) / 10.0
            return pair.get(t, -9) >= thr
        if kind.startswith("self1_or_cons"):
            thr = float(kind.split("cons")[1]) / 10.0
            return mins <= -1.0 or pair.get(t, -9) >= thr
        if kind == "self1_in5k":
            fin5 = [x for x, c in zip(sf, cv) if c >= 5000 and np.isfinite(x)]
            return bool(fin5) and min(fin5) <= -1.0
        if kind == "self05":
            return mins <= -0.5
        if kind == "ncomp2":
            return ncomp >= 2
        if kind == "self1_n2":
            return mins <= -1.0 and ncomp >= 2
        if kind == "cov10k":
            return any(c >= 10000 for c in cv)
        if kind == "self05_cov5k":
            return mins <= -0.5 and any(c >= 5000 for c in cv)
        if kind == "neg_majority":
            return ncomp > 0 and sum(x < 0 for x in fin) >= (len(fin) + 1) // 2
        raise ValueError(kind)

    print("\n=== fair delta vs gwps on gated subset (PDS raw) ===")
    for kind in ("self1", "cons2", "cons3", "cons5", "self1_or_cons2",
                 "self1_or_cons5"):
        for label in ("single_max", "mean_cov5k", "selfw_cov5k", "med_cov5k"):
            for w in (0.2, 0.3, 0.4, 0.5):
                ts = [t for t in agg[label] if t in gwps and gate(t, kind)]
                if len(ts) < 8:
                    continue
                deltas = [P[(label, w, t)] - base[t] for t in ts]
                print(f"  {kind:14s} {label:12s} w{w:g} n={len(ts):2d} "
                      f"mean {np.mean(deltas):+.4f} med {np.median(deltas):+.4f} "
                      f"pos {sum(d > 0.02 for d in deltas):2d} "
                      f"neg {sum(d < -0.02 for d in deltas):2d}")
        print()

    # DE metrics for gated rules
    d2 = np.load("outputs/probe_mix_profile.npz", allow_pickle=True)
    bnames = [str(x) for x in d2["names"]]
    B = d2["B"].astype(np.float64)
    tpos = d2["tpos"].astype(int)

    def one_de(v, i):
        real = (np.abs(B[i]) >= 0.5) & filt
        real[tpos[i]] = False
        pred = np.zeros_like(real)
        order = np.argsort(-np.abs(v))
        pred[order[:300]] = True
        pred &= filt
        pred[tpos[i]] = False
        inter = pred & real
        k = int(np.count_nonzero(inter & (np.sign(v) == np.sign(B[i]))))
        fid = k / max(pred.sum(), real.sum()) if inter.sum() else 0.0
        jac = inter.sum() / max((pred | real).sum(), 1) if inter.sum() else 0.0
        rank = np.where(pred, np.abs(v), -1.0)
        reach = purity_reach(rank, real, np.sign(v) == np.sign(B[i]))
        return fid, jac, reach

    def de(vecs, w, kind):
        df, dj, dr = [], [], []
        for i, t in enumerate(bnames):
            if t not in vecs or t not in gwps or not gate(t, kind):
                continue
            gf, gj, gr = one_de(gwps[t], i)
            v = w * unit(vecs[t]) + (1 - w) * unit(gwps[t])
            mf, mj, mr = one_de(v, i)
            df.append(mf - gf)
            dj.append(mj - gj)
            dr.append(mr - gr)
        return np.mean(df), np.mean(dj), np.mean(dr), len(df)

    print("=== DE deltas (mix - gwps) on gated subset ===")
    for kind in ("self1", "cons2", "cons3", "cons5", "self1_or_cons2",
                 "self1_or_cons5"):
        for label in ("selfw_cov5k", "mean_cov5k", "single_max"):
            if label not in agg:
                continue
            for w in (0.3, 0.4):
                r = de(agg[label], w, kind)
                print(f"  {kind:14s} {label:12s} w{w:g} n={r[3]:2d}: "
                      f"dfid {r[0]:+.4f} djac {r[1]:+.4f} dreach {r[2]:+.4f}")


if __name__ == "__main__":
    main()
