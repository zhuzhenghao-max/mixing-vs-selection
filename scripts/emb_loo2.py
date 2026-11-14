"""集成成员扩展扫描: 成员池 + 贪心前向选择, 目标 = raw-pds 代理.

285 train pert 5-fold LOO; 每个成员 = (特征空间, 方法) 训练折拟合→测试折预测, 对折均值
响应正交化. 组合 = 等权平均 (投影线性 → 组合仍正交). 评价: raw-pds 代理 (filt 空间 cosine
检索 midrank) + spec-corr. Geneformer 嵌入存在时自动入池 (outputs/gf316_emb.npz 等).
对照: C3 (v4 当前 4 成员) 应复现 spec +0.090.
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

    folds = np.random.default_rng(0).permutation(n) % 5
    mvec_row = np.zeros_like(LFC)
    for k in range(5):
        mvec_row[folds == k] = LFC[folds != k].mean(0)

    members = {}

    def add(name, P):
        members[name] = orth_rows(P, mvec_row)
        print(f"  member {name} ready", flush=True)

    def ridge_member(name, Feat, lam):
        P = np.zeros_like(LFC)
        for k in range(5):
            te = folds == k
            W, b = fit_ridge(Feat[~te], LFC[~te], lam)
            P[te] = Feat[te] @ W + b
        add(name, P)

    def knn_member(name, Feat, kk):
        P = np.zeros_like(LFC)
        for k in range(5):
            te = folds == k
            Tn = Feat[~te] / np.maximum(np.linalg.norm(Feat[~te], axis=1, keepdims=True), 1e-9)
            En = Feat[te] / np.maximum(np.linalg.norm(Feat[te], axis=1, keepdims=True), 1e-9)
            P[te] = knn_from_S(En @ Tn.T, LFC[~te], kk)
        add(name, P)

    ridge_member("E650r100", spaces["E650"], 100.0)
    ridge_member("E650r30", spaces["E650"], 30.0)
    ridge_member("E650r300", spaces["E650"], 300.0)
    ridge_member("E150r100", spaces["E150"], 100.0)
    ridge_member("SCr100", spaces["SC"], 100.0)
    knn_member("E650k5", spaces["E650"], 5)
    knn_member("E650k10", spaces["E650"], 10)
    knn_member("E650k20", spaces["E650"], 20)
    knn_member("E150k10", spaces["E150"], 10)
    knn_member("E150k20", spaces["E150"], 20)
    knn_member("E150k40", spaces["E150"], 40)
    knn_member("SCk10", spaces["SC"], 10)
    knn_member("SCk20", spaces["SC"], 20)
    ridge_member("CatESr100", np.hstack([spaces["E650"], spaces["SC"]]), 100.0)
    if "GF316" in spaces:
        ridge_member("GF316r100", spaces["GF316"], 100.0)
        knn_member("GF316k10", spaces["GF316"], 10)
        knn_member("GF316k20", spaces["GF316"], 20)
    if "GF316C" in spaces:
        ridge_member("GF316Cr100", spaces["GF316C"], 100.0)
        knn_member("GF316Ck10", spaces["GF316C"], 10)
        knn_member("GF316Ck20", spaces["GF316C"], 20)
    if "GF104" in spaces:
        ridge_member("GF104r100", spaces["GF104"], 100.0)
        knn_member("GF104k10", spaces["GF104"], 10)

    base = filt & np.all(np.isfinite(LFC), 0)
    ref = LFC[:, base]
    refn = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
    yr = np.zeros((n, base.sum()))
    for i in range(n):
        yr[i] = resid(LFC[i][base], mvec_row[i][base])
    yr_std = yr.std(1)

    def score(P):
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

    names = list(members)
    print(f"{len(names)} members; standalone:")
    for nm in names:
        pm, sm, smu = score(members[nm])
        print(f"  {nm:10s} pds {pm:.3f} | spec {sm:+.3f}/{smu:+.3f}", flush=True)

    chosen = []
    pool = set(names)
    cur = None
    for step in range(7):
        best = None
        for nm in sorted(pool):
            cand = members[nm].copy() if cur is None else (cur * len(chosen) + members[nm]) / (len(chosen) + 1)
            pm, sm, smu = score(cand)
            if best is None or pm > best[0]:
                best = (pm, sm, smu, nm)
        pm, sm, smu, nm = best
        chosen.append(nm)
        pool.discard(nm)
        cur = members[chosen[0]].copy() if len(chosen) == 1 else (cur * (len(chosen) - 1) + members[nm]) / len(chosen)
        print(f"greedy[{step + 1}] +{nm:10s} -> pds {pm:.4f} spec {sm:+.3f}/{smu:+.3f} | {chosen}",
              flush=True)
        if not pool:
            break

    c3 = ["E650r100", "SCk10", "E150k20", "E650k5"]
    if all(c in members for c in c3):
        P = sum(members[c] for c in c3) / len(c3)
        pm, sm, smu = score(P)
        print(f"C3 (v4 当前 4 成员) -> pds {pm:.4f} spec {sm:+.3f}/{smu:+.3f}")


if __name__ == "__main__":
    main()
