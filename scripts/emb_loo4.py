"""最终集成候选多粒度检验: 精选成员池 + 8 候选 × 5 种子.

候选来源: C3(现行 v4), C3s, G7(种子0贪心, 已被独立复检选为最稳之一),
S7G5/S42G(种子7/42贪心得到的 GF-ridge 系), 两个并集 U9/U12, TOP8(独立pds前8).
判据: 跨种子 mean±SE, 最优种子(去发现种子)边际, floor.
"""

import os
import sys

import numpy as np

sys.path.insert(0, "src")
from linear_baseline import fit_ridge  # noqa: E402


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def orth_rows(P, mvec):
    out = P.copy()
    for j in range(P.shape[0]):
        mv = mvec[j] if mvec.ndim == 2 else mvec
        f = np.isfinite(out[j]) & np.isfinite(mv)
        out[j][f] = resid(out[j][f], mv[f])
    return out


def knn_from_S(S, train_LFC, k):
    P = np.zeros((S.shape[0], train_LFC.shape[1]), dtype=np.float64)
    for j in range(S.shape[0]):
        top = np.argsort(-S[j])[:k]
        w = np.maximum(S[j][top], 0)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        P[j] = (train_LFC[top] * w[:, None]).sum(0) / w.sum()
    return P


def main():
    m = np.load("outputs/scvi_map_cache.npz")
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    n = len(train_names)

    def emb(path):
        e = np.load(path)
        p = {g: i for i, g in enumerate(e["names"].astype(str))}
        return np.stack([e["E"][p[g]] for g in train_names])

    spaces = {"E650": emb("outputs/esm650_emb.npz"),
              "E150": emb("outputs/esm150_emb.npz"),
              "SC": emb("outputs/scgpt_emb.npz"),
              "GF316": emb("outputs/gf316_emb.npz")}
    print("spaces:", list(spaces), flush=True)

    pool = [
        ("E650r100", "ridge", "E650", 100.0),
        ("E650r30", "ridge", "E650", 30.0),
        ("E650k5", "knn", "E650", 5),
        ("E150r100", "ridge", "E150", 100.0),
        ("E150k20", "knn", "E150", 20),
        ("E150k40", "knn", "E150", 40),
        ("SCk10", "knn", "SC", 10),
        ("SCk20", "knn", "SC", 20),
        ("SCr100", "ridge", "SC", 100.0),
        ("CatESr100", "ridge", ("cat", "E650", "SC"), 100.0),
        ("GF316r100", "ridge", "GF316", 100.0),
        ("GF316r30", "ridge", "GF316", 30.0),
        ("GF316r300", "ridge", "GF316", 300.0),
        ("GF316k20", "knn", "GF316", 20),
        ("CatGFSCr100", "ridge", ("cat", "GF316", "SC"), 100.0),
    ]

    def feat_of(key):
        if isinstance(key, tuple):
            return np.hstack([spaces[k] for k in key[1:]])
        return spaces[key]

    def build_members(folds):
        mvec_row = np.zeros_like(LFC)
        for k in range(5):
            mvec_row[folds == k] = LFC[folds != k].mean(0)
        members = {}
        for name, kind, key, param in pool:
            Feat = feat_of(key)
            P = np.zeros_like(LFC)
            if kind == "ridge":
                for k in range(5):
                    te = folds == k
                    W, b = fit_ridge(Feat[~te], LFC[~te], param)
                    P[te] = Feat[te] @ W + b
            else:
                for k in range(5):
                    te = folds == k
                    Tn = Feat[~te] / np.maximum(np.linalg.norm(Feat[~te], axis=1, keepdims=True), 1e-9)
                    En = Feat[te] / np.maximum(np.linalg.norm(Feat[te], axis=1, keepdims=True), 1e-9)
                    P[te] = knn_from_S(En @ Tn.T, LFC[~te], param)
            members[name] = orth_rows(P, mvec_row)
        return members, mvec_row

    base = filt & np.all(np.isfinite(LFC), 0)
    ref = LFC[:, base]
    refn = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)

    def score(P, mvec_row):
        yr = np.zeros((n, base.sum()))
        for i in range(n):
            yr[i] = resid(LFC[i][base], mvec_row[i][base])
        yr_std = yr.std(1)
        Pb = P[:, base]
        Pb = Pb / np.maximum(np.linalg.norm(Pb, axis=1, keepdims=True), 1e-9)
        S = Pb @ refn.T
        raws = np.empty(n)
        for i in range(n):
            s = S[i]
            sii = s[i]
            rank = 1.0 + np.sum(s > sii) + 0.5 * (np.sum(s == sii) - 1)
            raws[i] = 1.0 - (rank - 1.0) / (n - 1.0)
        cs = []
        for i in range(n):
            x = P[i][base]
            if np.std(x) < 1e-12 or yr_std[i] < 1e-12:
                continue
            cs.append(np.corrcoef(x, yr[i])[0, 1])
        cs = np.array(cs)
        return float(np.median(raws)), float(np.median(cs)), float(np.mean(cs))

    core = ["GF316r100", "E650r30", "CatGFSCr100"]
    s7g5 = core + ["GF316r300", "SCr100"]
    s42g = core + ["GF316r30", "GF316k20", "E150k20", "CatESr100"]
    candidates = {
        "C3":    ["E650r100", "SCk10", "E150k20", "E650k5"],
        "C3s":   ["E650r100", "SCk20", "E150k20", "E650k5"],
        "G7":    ["SCk20", "E150k20", "SCr100", "E650r30", "E150r100", "CatESr100", "E650r100"],
        "S7G5":  s7g5,
        "S42G":  s42g,
        "U9":    sorted(set(s7g5) | set(s42g)),
        "U12":   sorted(set(s7g5) | set(s42g) | {"SCk20", "E150r100", "E650r100"}),
        "TOP8":  ["SCk20", "SCk10", "CatESr100", "GF316r100", "E150k20", "E650r100", "E650k5", "E150k40"],
    }
    for c, comp in candidates.items():
        missing = [x for x in comp if x not in dict((p[0], 1) for p in pool)]
        assert not missing, (c, missing)
    seeds = [0, 7, 42, 123, 2024]
    results = {}
    for seed in seeds:
        folds = np.random.default_rng(seed).permutation(n) % 5
        members, mvec_row = build_members(folds)
        print(f"seed {seed}: members ready", flush=True)
        for cname, comp in candidates.items():
            P = sum(members[c] for c in comp) / len(comp)
            pm, sm, smu = score(P, mvec_row)
            results.setdefault(cname, []).append((pm, sm, smu))
            print(f"  seed {seed} {cname:5s} pds {pm:.4f} spec {sm:+.3f}/{smu:+.3f}", flush=True)

    print("\n== summary ==")
    c3 = np.array([r[0] for r in results["C3"]])
    for cname in candidates:
        rs = np.array(results[cname])
        pm, sm, smu = rs[:, 0], rs[:, 1], rs[:, 2]
        d = pm - c3
        print(f"  {cname:5s} pds {pm.mean():.4f}±{pm.std(ddof=1)/np.sqrt(len(seeds)):.4f} "
              f"(vs C3 {d.mean():+.4f}, win {int(np.sum(pm > c3))}/{len(seeds)}, floor {pm.min():.4f}) "
              f"| spec {sm.mean():+.3f}(med)/{smu.mean():+.3f}(mean) | per-seed pds {np.round(pm, 4)}")


if __name__ == "__main__":
    main()
