"""跨折叠种子稳定性检验: 固定候选组合 vs C3 (v4).

对每种种子的折叠划分重算全部成员 OOF 预测 (ridge/kNN 重训练 + mvec 正交化),
评分候选组合的 median raw-pds 与 spec-corr, 报告 mean±SE 及与 C3 的差值.
单种子内评分差异 ≈1σ (median SE≈0.015), 需跨种子排序一致性才可信.
附带: 种子 7/42 的贪心轨迹 (诊断选择稳定性).
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
              "SC": emb("outputs/scgpt_emb.npz")}
    for path, tag in [("outputs/gf316_emb.npz", "GF316"), ("outputs/gf316_ctx_emb.npz", "GF316C"),
                      ("outputs/gf104_emb.npz", "GF104")]:
        if os.path.exists(path):
            spaces[tag] = emb(path)
    print("spaces:", list(spaces), flush=True)

    def pool_defs():
        """(name, kind, space_key_or_matrix, param)"""
        defs = [
            ("E650r100", "ridge", "E650", 100.0),
            ("E650r30", "ridge", "E650", 30.0),
            ("E650r300", "ridge", "E650", 300.0),
            ("E150r100", "ridge", "E150", 100.0),
            ("SCr100", "ridge", "SC", 100.0),
            ("E650k5", "knn", "E650", 5),
            ("E650k10", "knn", "E650", 10),
            ("E650k20", "knn", "E650", 20),
            ("E150k10", "knn", "E150", 10),
            ("E150k20", "knn", "E150", 20),
            ("E150k40", "knn", "E150", 40),
            ("SCk10", "knn", "SC", 10),
            ("SCk20", "knn", "SC", 20),
            ("CatESr100", "ridge", ("cat", "E650", "SC"), 100.0),
        ]
        if "GF316" in spaces:
            defs += [
                ("GF316r30", "ridge", "GF316", 30.0),
                ("GF316r100", "ridge", "GF316", 100.0),
                ("GF316r300", "ridge", "GF316", 300.0),
                ("GF316k10", "knn", "GF316", 10),
                ("GF316k20", "knn", "GF316", 20),
                ("CatGFSCr100", "ridge", ("cat", "GF316", "SC"), 100.0),
                ("CatGFESr100", "ridge", ("cat", "GF316", "E650"), 100.0),
            ]
        if "GF316C" in spaces:
            defs += [
                ("GF316Cr100", "ridge", "GF316C", 100.0),
                ("GF316Ck10", "knn", "GF316C", 10),
                ("GF316Ck20", "knn", "GF316C", 20),
            ]
        return defs

    def feat_of(key):
        if isinstance(key, tuple):
            return np.hstack([spaces[k] for k in key[1:]])
        return spaces[key]

    def build_members(folds):
        mvec_row = np.zeros_like(LFC)
        for k in range(5):
            mvec_row[folds == k] = LFC[folds != k].mean(0)
        members = {}
        for name, kind, key, param in pool_defs():
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
        return members

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

    candidates = {
        "C3":     ["E650r100", "SCk10", "E150k20", "E650k5"],
        "C3s":    ["E650r100", "SCk20", "E150k20", "E650k5"],
        "C3g":    ["E650r100", "SCk10", "E150k20", "E650k5", "GF316r100"],
        "C3sg":   ["E650r100", "SCk20", "E150k20", "E650k5", "GF316r100"],
        "G3":     ["SCk20", "E150k20", "SCr100"],
        "G3g":    ["SCk20", "E150k20", "SCr100", "GF316r100"],
        "G7":     ["SCk20", "E150k20", "SCr100", "E650r30", "E150r100", "CatESr100", "E650r100"],
        "G7g":    ["SCk20", "E150k20", "SCr100", "E650r30", "E150r100", "CatESr100", "E650r100", "GF316r100"],
        "G7GF":   ["SCk20", "E150k20", "SCr100", "GF316r100", "E650r30", "GF316Cr100", "GF316Ck10"],
        "G6GF":   ["SCk20", "E150k20", "SCr100", "GF316r100", "E650r30", "GF316Cr100"],
        "SP4":    ["CatESr100", "GF316r100", "SCk20", "E650r100"],
        "S7G5":   ["E650r30", "CatGFSCr100", "GF316r300", "SCr100", "GF316r100"],
        "S7G6":   ["E650r30", "CatGFSCr100", "GF316r300", "SCr100", "GF316r100", "GF316r30"],
        "S7G5c":  ["E650r30", "CatGFSCr100", "GF316r300", "SCr100", "GF316r100", "CatESr100"],
        "S42G":   ["GF316r100", "E150k20", "E650r30", "CatGFSCr100", "GF316k20", "CatESr100", "GF316r30"],
    }

    seeds = [0, 7, 42, 123, 2024]
    results = {c: [] for c in candidates}
    for seed in seeds:
        folds = np.random.default_rng(seed).permutation(n) % 5
        members = build_members(folds)
        for name in ("ENSALL",):
            candidates[name] = list(members)
        mvec_row = np.zeros_like(LFC)
        for k in range(5):
            mvec_row[folds == k] = LFC[folds != k].mean(0)
        print(f"seed {seed}: members ready", flush=True)
        for cname, comp in candidates.items():
            P = sum(members[c] for c in comp) / len(comp)
            pm, sm, smu = score(P, mvec_row)
            results.setdefault(cname, []).append((pm, sm, smu))
            print(f"  seed {seed} {cname:6s} pds {pm:.4f} spec {sm:+.3f}/{smu:+.3f}", flush=True)

    print("\n== summary (mean±SE over seeds) ==")
    c3p = np.array([r[0] for r in results["C3"]])
    for cname in list(candidates):
        rs = np.array(results[cname])
        pm, sm, smu = rs[:, 0], rs[:, 1], rs[:, 2]
        d = pm - c3p
        wins = int(np.sum(pm > c3p))
        print(f"  {cname:6s} pds {pm.mean():.4f}±{pm.std(ddof=1)/np.sqrt(len(seeds)):.4f} "
              f"(vs C3 {d.mean():+.4f}, win {wins}/{len(seeds)}) | "
              f"spec med {sm.mean():+.3f}±{sm.std(ddof=1)/np.sqrt(len(seeds)):.3f} "
              f"mean {smu.mean():+.3f}")

    if os.environ.get("GREEDY") == "1":
        for seed in (7, 42):
            folds = np.random.default_rng(seed).permutation(n) % 5
            members = build_members(folds)
            mvec_row = np.zeros_like(LFC)
            for k in range(5):
                mvec_row[folds == k] = LFC[folds != k].mean(0)
            chosen, pool = [], set(members)
            cur = None
            for step in range(7):
                best = None
                for nm in sorted(pool):
                    cand = members[nm].copy() if cur is None else (cur * len(chosen) + members[nm]) / (len(chosen) + 1)
                    pm, sm, smu = score(cand, mvec_row)
                    if best is None or pm > best[0]:
                        best = (pm, sm, smu, nm)
                pm, sm, smu, nm = best
                chosen.append(nm)
                pool.discard(nm)
                cur = members[chosen[0]].copy() if len(chosen) == 1 else (cur * (len(chosen) - 1) + members[nm]) / len(chosen)
                print(f"greedy seed {seed} [{step + 1}] +{nm:12s} -> pds {pm:.4f} spec {sm:+.3f}", flush=True)


if __name__ == "__main__":
    main()
