"""跨细胞系迁移检验: Replogle-Nadig (4 系) delta → H1 方向迁移.

留出检验: 对 285 个 H1 训练靶中在 Replogle 被测的靶, 用其 K562/RPE1/... 的
pseudobulk delta 向量 (限定到共享基因 & filt) 当作 "H1 预测", 用 raw-pds + 谱相关
harness (与 emb_loo4 相同) 评分, 对照零样本嵌入成员的 0.55-0.62 代理区间.

变体: 每系单独 / 跨系均值; 迁移向量是否对 H1 mvec 正交化; 行标准化.
只读 all_delta.parquet 需要的行 (pyarrow 行组流式过滤), 峰值内存小.
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "src")


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def main():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes_all = m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    n = len(train_names)

    feat = [str(x) for x in __import__("json").load(open("data/external/feature_names.json"))]
    gidx = {g: i for i, g in enumerate(genes_all)}
    feat_ok = np.array([f in gidx for f in feat])
    print(f"features {len(feat)}, mapped to our genes {feat_ok.sum()}, filt+finite pass ", end="")
    our_pos = np.array([gidx[f] for f in feat if f in gidx])
    use = filt[our_pos] & np.all(np.isfinite(LFC[:, our_pos]), 0)
    our_pos = our_pos[use]
    feat_sel = [f for f, u in zip((f for f in feat if f in gidx), use) if u]
    print(f"{len(our_pos)}")

    targets = set(train_names)
    pf = pq.ParquetFile("data/external/all_delta.parquet")
    print("rows total:", pf.metadata.num_rows, "| cols:", pf.metadata.num_columns)

    t = pf.read(columns=["cell_line", "gem_group", "gene"])
    genes_col = t.column("gene").to_pylist()
    keep = np.array([g in targets for g in genes_col])
    meta_cell = np.array(t.column("cell_line").to_pylist(), dtype=object)[keep]
    meta_gg = np.array(t.column("gem_group").to_pylist(), dtype=object)[keep]
    meta_gene = np.array(genes_col, dtype=object)[keep]
    nk = int(keep.sum())
    print(f"kept {nk} rows for train targets")
    D = np.empty((nk, len(feat_sel)), dtype=np.float32)
    B = 256
    for i in range(0, len(feat_sel), B):
        cols = feat_sel[i:i + B]
        tb = pf.read(columns=cols)
        for j, c in enumerate(cols):
            D[:, i + j] = tb.column(c).to_numpy(zero_copy_only=False)[keep]
        print(f"  cols {i}-{i + len(cols)}/{len(feat_sel)}", flush=True)

    df = pd.DataFrame({"cell_line": meta_cell, "gem_group": meta_gg, "gene": meta_gene})
    print("cell lines:", df.cell_line.value_counts().to_dict())
    print("targets covered:", df.gene.nunique(), "/", n)

    base = np.all(np.isfinite(LFC[:, our_pos]), 0)
    pos = our_pos[base]
    ref = LFC[:, pos]
    mvec = ref.mean(0)
    refn = np.zeros_like(ref)
    for i in range(n):
        refn[i] = resid(ref[i], mvec)
    refn = refn / np.maximum(np.linalg.norm(refn, axis=1, keepdims=True), 1e-9)

    lines = sorted(df.cell_line.unique())
    # pool per (line, target): mean over gem_groups
    pooled = {}
    for cl in lines:
        sub = df.cell_line.values == cl
        tgt = meta_gene[sub]
        Dc = D[sub]
        acc = {}
        for t in np.unique(tgt):
            acc[t] = Dc[tgt == t].mean(0)[base]
        pooled[cl] = acc
        print(f"  {cl}: {len(acc)} targets pooled", flush=True)

    def pds_of(P, idx):
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        S = Pn @ refn.T
        raws = []
        for k, i in enumerate(idx):
            s = S[k]
            rank = 1.0 + np.sum(s > s[i]) + 0.5 * (np.sum(s == s[i]) - 1)
            raws.append(1.0 - (rank - 1.0) / (n - 1.0))
        return float(np.median(raws)), S

    def eval_variant(name, rows_by_target):
        idx = [i for i, t in enumerate(train_names) if t in rows_by_target]
        if len(idx) < 10:
            print(f"  {name:20s}: only {len(idx)} targets, skip")
            return
        P = np.stack([rows_by_target[train_names[i]] for i in idx])
        # raw variant
        for tag, Pv in [("raw", P), ("orth", np.stack([resid(r, mvec) for r in P]))]:
            pds, S = pds_of(Pv, idx)
            cs = []
            for k, i in enumerate(idx):
                a = Pv[k]
                b = refn[i]
                if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                    cs.append(np.corrcoef(a, b)[0, 1])
            cs = np.array(cs)
            extra = ""
            if tag == "raw":
                Pnn = Pv / np.maximum(np.linalg.norm(Pv, axis=1, keepdims=True), 1e-9)
                C = Pnn @ Pnn.T
                off = C[np.triu_indices(len(idx), 1)]
                pds_c, _ = pds_of(np.repeat(Pnn.mean(0)[None], len(idx), 0), idx)
                rng = np.random.default_rng(0)
                perm = rng.permutation(len(idx))
                while np.any(perm == np.arange(len(idx))):
                    perm = rng.permutation(len(idx))
                pds_s, _ = pds_of(Pnn[perm], idx)
                extra = (f" | xferpaircos med {np.median(off):+.3f} | pds_const {pds_c:.3f} "
                         f"pds_shuf {pds_s:.3f}")
            print(f"  {name:20s} [{tag:4s}] n={len(idx):3d} pds {pds:.4f} "
                  f"spec med {np.median(cs):+.3f} mean {cs.mean():+.3f}{extra}", flush=True)

    for cl in lines:
        eval_variant(cl, pooled[cl])
    common = set.intersection(*[set(p.keys()) for p in pooled.values()])
    if common:
        avg = {t: np.mean([pooled[cl][t] for cl in lines], 0) for t in common}
        eval_variant("all-lines-avg", avg)
    for pair in (("k562", "rpe1"), ("k562", "rpe1", "hepg2", "jurkat")):
        if all(l in pooled for l in pair):
            its = set.intersection(*[set(pooled[l].keys()) for l in pair])
            avg = {t: np.mean([pooled[l][t] for l in pair], 0) for t in its}
            eval_variant("+".join(pair), avg)

    np.savez("outputs/replogle_pooled.npz",
             genes=np.array(feat_sel), our_pos=our_pos, base=base,
             **{f"{cl}_names": np.array(sorted(pooled[cl]))
                for cl in lines},
             **{f"{cl}_vecs": np.stack([pooled[cl][t] for t in sorted(pooled[cl])])
                for cl in lines})
    print("saved outputs/replogle_pooled.npz")

    panel = set(pd.read_csv("data/pert_counts.csv")["target_gene"].astype(str))
    all_t = set(genes_col)
    print(f"parquet unique knockdown targets: {len(all_t)}")
    print(f"panel ∩ replogle targets: {len(panel & all_t)} / 300")
    for cl in lines:
        n_cl = sum(1 for t in panel if t in pooled[cl])
        print(f"  {cl}: panel covered {n_cl}")


if __name__ == "__main__":
    main()
