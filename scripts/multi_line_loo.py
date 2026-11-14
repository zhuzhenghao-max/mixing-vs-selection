"""多细胞系 LOO 代理: 留出一系, 用其余系的实测行做 (相似度加权) 混合,
在留出系实测行上做检索 pds. 判据: 相似度加权 vs 等权 vs 最优单源.

- 4 系 v1: k562 (gwps) / hct116 / hek293t (orion) / rpe1 (replogle essential)
- 行均为 natural (未缩放), 混合时 unit 化; 检索参考 = 留出系自身实测行
  (resid 掉均值后 unit 化, 同 combine_check 口径)
- 权重 w_i = softmax(cos(prof_L, prof_i) / T), 仅在被测源上归一化
"""

import json
import sys

import numpy as np

sys.path.insert(0, "src")

LINES = ["k562", "hct116", "hek293t", "rpe1"]
TS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
# 若某些系没有对照谱 (如 rpe1 未下载), 只能作为源不能作为留出系
REQUIRE_PROFILE = True


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def main():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes = m["genes"].astype(str)
    filt = m["filt"]
    n_genes = len(genes)

    rows = {}

    # k562
    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    rows["k562"] = {nm: r for nm, r in zip(
        [str(x) for x in g["names"].astype(str)], g["rows"].astype(np.float64))}

    # orion per line
    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    o_genes = meta["genes"]
    gidx = {g_: i for i, g_ in enumerate(genes)}
    ocols, opos = [], []
    for oc, gg in enumerate(o_genes):
        if gg in gidx:
            ocols.append(oc)
            opos.append(gidx[gg])
    ocols = np.array(ocols)
    opos = np.array(opos)
    pert_to_id = meta["pert_to_id"]
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    rowmap = {}
    for r in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[r])], {})[int(pert_ids[r])] = r
    for cl in ("hct116", "hek293t"):
        d = {}
        for t, pid in pert_to_id.items():
            r = rowmap[cl].get(pid)
            if r is None:
                continue
            v = np.nan_to_num(np.asarray(L[r, ocols], dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
            if np.abs(v).sum() == 0:
                continue
            full = np.zeros(n_genes)
            full[opos] = v
            d[t] = full
        rows[cl] = d
        print(f"{cl}: {len(d)} rows")

    # rpe1
    rp = np.load("outputs/rpe1_rows.npz", allow_pickle=True)
    rows["rpe1"] = {nm: r for nm, r in zip(
        [str(x) for x in rp["names"].astype(str)], rp["rows"].astype(np.float64))}
    print(f"rpe1: {len(rows['rpe1'])} rows")

    # profiles
    pr = np.load("outputs/line_profiles.npz", allow_pickle=True)
    pnames = [str(x) for x in pr["names"].astype(str)]
    P = {nm: np.log1p(pr["prof"][pnames.index(nm)].astype(np.float64))
         for nm in LINES if nm in pnames}
    sim = {}
    for a in LINES:
        for b in LINES:
            if a == b or a not in P or b not in P:
                continue
            mask = (P[a] > 0) & (P[b] > 0)
            if mask.sum() < 100:
                sim[(a, b)] = 0.0
                continue
            sim[(a, b)] = float(np.dot(P[a][mask], P[b][mask]) /
                                (np.linalg.norm(P[a][mask]) * np.linalg.norm(P[b][mask])))
    print("profile cosine:")
    for a in LINES:
        print(f"  {a:8s} " + " ".join(f"{sim.get((a, b), 0.0):+.2f}" for b in LINES if b != a))

    # retrieval base: for each hold-out line, restrict to shared finite genes
    base = filt & np.all(np.isfinite(m["LFC"]), 0)
    pos = np.where(base)[0]
    print(f"\neval positions: {len(pos)}")

    results = {}
    holdouts = [x for x in LINES if x in rows and (not REQUIRE_PROFILE or x in P)]
    for L0 in holdouts:
        others = [x for x in LINES if x != L0]
        src = [o for o in others if o in P]
        targets = [t for t in rows[L0]
                   if sum(1 for o in src if t in rows[o]) >= 2]
        print(f"\n=== hold-out {L0}: {len(targets)} targets measured in >=2 profiled others "
              f"(src={src}) ===")
        ref_raw = np.stack([rows[L0][t][pos] for t in targets])
        ref = ref_raw - ref_raw.mean(0)
        refn = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
        n = len(targets)

        def pds_of(Pn):
            S = Pn @ refn.T
            raws = []
            for k in range(n):
                s = S[k]
                rank = 1.0 + np.sum(s > s[k]) + 0.5 * (np.sum(s == s[k]) - 1)
                raws.append(1.0 - (rank - 1.0) / (n - 1.0))
            return np.median(raws)

        def build_mix(weights_fn):
            out = []
            for t in targets:
                acc = np.zeros(len(pos))
                wsum = 0.0
                for o in src:
                    if t not in rows[o]:
                        continue
                    w = weights_fn(o)
                    acc += w * unit(rows[o][t][pos])
                    wsum += w
                out.append(acc / max(wsum, 1e-9))
            return np.stack(out)

        res = {}
        # best single source
        for o in others:
            Pn = np.stack([unit(rows[o][t][pos]) if t in rows[o]
                           else np.zeros(len(pos)) for t in targets])
            res[f"single_{o}"] = pds_of(Pn)
        # equal
        res["equal"] = pds_of(build_mix(lambda o: 1.0))
        # similarity weighted
        for T in TS:
            res[f"simT{T:g}"] = pds_of(build_mix(lambda o, T=T: np.exp(sim[(L0, o)] / T)))
        results[L0] = res
        for k, v in res.items():
            print(f"  {k:10s} pds {v:.4f}")

    print("\n=== summary (mean pds across hold-outs) ===")
    methods = [f"simT{T:g}" for T in TS] + ["equal"] + [f"single_{o}" for o in LINES]
    for k in methods:
        vs = [results[L0].get(k) for L0 in results if k in results[L0]]
        print(f"  {k:10s} {np.mean(vs):.4f}  ({len(vs)} lines)")


if __name__ == "__main__":
    main()
