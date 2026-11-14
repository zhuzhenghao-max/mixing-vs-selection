"""LOO 决策实验: F454 ridge 与嵌入特异分量的组合方式对比 (285 pert).

每 fold: mvec=训练均值响应; F454 ridge100 + 4 模型特异方向集成 (与 build_emb_add 同构,
全部只用训练折拟合). 候选组合:
  A ridgeonly   : F454 ridge
  B add_g       : ridge + g*ens
  C repl_g      : projm(ridge) + g*ens     (去掉 F454 弱特异肩, 换成 ens 肩)
  D repl_2g / E repl_hg / G add_2g         剂量响应
  F ensonly     : ens + g*... 纯特异 (无共享分量, 参考上界)
评价: raw-pds (在 285 真值面板内 cosine 检索, midrank), spec-corr, plain corr.
"""

import sys

import numpy as np

sys.path.insert(0, "src")
from linear_baseline import fit_ridge  # noqa: E402


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def orth_rows(P, mvec):
    out = P.copy()
    for j in range(P.shape[0]):
        f = np.isfinite(out[j]) & np.isfinite(mvec)
        out[j][f] = resid(out[j][f], mvec[f])
    return out


def knn_test(Te, Tr, LFCtr, k):
    En = Te / np.maximum(np.linalg.norm(Te, axis=1, keepdims=True), 1e-9)
    Tn = Tr / np.maximum(np.linalg.norm(Tr, axis=1, keepdims=True), 1e-9)
    S = En @ Tn.T
    P = np.zeros((Te.shape[0], LFCtr.shape[1]), dtype=np.float64)
    for j in range(Te.shape[0]):
        top = np.argsort(-S[j])[:k]
        w = np.maximum(S[j][top], 0)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        P[j] = (LFCtr[top] * w[:, None]).sum(0) / w.sum()
    return P


def main():
    m = np.load("outputs/scvi_map_cache.npz")
    F_all, genes = m["F"], m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    posg = {g: i for i, g in enumerate(genes)}
    F_tr = np.stack([F_all[posg[g]] for g in train_names])
    n = len(train_names)

    def emb(path):
        e = np.load(path)
        p = {g: i for i, g in enumerate(e["names"].astype(str))}
        return np.stack([e["E"][p[g]] for g in train_names])

    E650 = emb("outputs/esm650_emb.npz")
    E150 = emb("outputs/esm150_emb.npz")
    ESC = emb("outputs/scgpt_emb.npz")

    folds = np.random.default_rng(0).permutation(n) % 5
    mvec_row = np.zeros_like(LFC)  # per-row fold train-mean (no leakage)
    pred454 = np.zeros_like(LFC)
    ens = np.zeros_like(LFC)
    for k in range(5):
        te = folds == k
        mvec = LFC[~te].mean(0)
        mvec_row[te] = mvec
        W, b = fit_ridge(F_tr[~te], LFC[~te], 100.0)
        pred454[te] = F_tr[te] @ W + b
        W2, b2 = fit_ridge(E650[~te], LFC[~te], 100.0)
        p1 = E650[te] @ W2 + b2
        p2 = knn_test(ESC[te], ESC[~te], LFC[~te], 10)
        p3 = knn_test(E150[te], E150[~te], LFC[~te], 20)
        p4 = knn_test(E650[te], E650[~te], LFC[~te], 5)
        ens[te] = np.mean([orth_rows(p, mvec) for p in (p1, p2, p3, p4)], 0)

    n4 = np.median(np.linalg.norm(orth_rows(pred454, mvec_row.mean(0))[:, filt], axis=1))
    ne = np.median(np.linalg.norm(ens[:, filt], axis=1))
    g = n4 / max(ne, 1e-9)
    print(f"F454 spec norm median {n4:.2f}, ens norm median {ne:.2f} -> gamma {g:.3f}")

    base = filt & np.all(np.isfinite(LFC), 0) \
        & np.all(np.isfinite(pred454), 0) & np.all(np.isfinite(ens), 0)
    print(f"base genes {base.sum()}")

    def projm(M):
        out = M.copy()
        for i in range(n):
            f = base
            out[i][f] = resid(out[i][f], mvec_row[i][f])
        return out

    repl = projm(pred454)
    share = pred454 - repl  # (ridge·m̂)m̂: 共享分量, 丢掉噪声特异肩
    configs = {
        "A ridgeonly ": pred454,
        "B add_g     ": pred454 + g * ens,
        "S1 share_g  ": share + g * ens,
        "S2 share_2g ": share + 2 * g * ens,
        "S4 share_4g ": share + 4 * g * ens,
        "D repl_2g   ": repl + 2 * g * ens,
        "F ensonly   ": g * ens,
    }

    ref = LFC[:, base]
    refn = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
    for tag, M in configs.items():
        Mb = M[:, base]
        Mb = Mb / np.maximum(np.linalg.norm(Mb, axis=1, keepdims=True), 1e-9)
        S = Mb @ refn.T  # n x n cosine
        raws = np.zeros(n)
        for i in range(n):
            s = S[i]
            sii = s[i]
            rank = 1.0 + np.sum(s > sii) + 0.5 * (np.sum(s == sii) - 1)
            raws[i] = 1.0 - (rank - 1.0) / (n - 1.0)
        vals_plain, vals_spec = [], []
        for i in range(n):
            f = base
            y = LFC[i][f]
            x = M[i][f]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            vals_plain.append(np.corrcoef(x, y)[0, 1])
            mv = mvec_row[i][f]
            xr, yr = resid(x, mv), resid(y, mv)
            if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
                continue
            vals_spec.append(np.corrcoef(xr, yr)[0, 1])
        vals_plain, vals_spec = np.array(vals_plain), np.array(vals_spec)
        print(f"{tag}: raw-pds med {np.median(raws):.3f} mean {np.mean(raws):.3f} "
              f"| member med {2 * np.median(raws) - 1:+.3f} "
              f"| spec med {np.median(vals_spec):+.3f} mean {np.mean(vals_spec):+.3f} "
              f"| plain med {np.median(vals_plain):+.3f} mean {np.mean(vals_plain):+.3f}",
              flush=True)


if __name__ == "__main__":
    main()
