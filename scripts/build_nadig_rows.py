"""构建 Nadig 2024 四系 (K562/RPE1/HepG2/Jurkat) 每靶行.

每个 (cell_line, target) 对 gem_group 取均值, 存映射到 18080 轴的紧凑矩阵
(只用 6118 个被测基因列; 加载时再展开).
流式 iter_batches, 避免整表 (约 6GB) 进内存.
输出 outputs/nadig_rows.npz: gpos, gpos_names, <line>_names, <line>_rows.
"""

import numpy as np
import pyarrow.parquet as pq

SRC = "data/external/all_delta.parquet"
AXIS = "outputs/scvi_map_cache.npz"
OUT = "outputs/nadig_rows.npz"
LINES = ("k562", "rpe1", "hepg2", "jurkat")
BATCH = 2048


def main():
    genes = np.load(AXIS, allow_pickle=True)["genes"].astype(str)
    gidx = {g: i for i, g in enumerate(genes)}

    f = pq.ParquetFile(SRC)
    gene_cols = f.schema_arrow.names[3:]
    mapped = [(i, gidx[g], g) for i, g in enumerate(gene_cols) if g in gidx]
    src_cols = np.array([m[0] for m in mapped])
    tgt_cols = np.array([m[1] for m in mapped])
    print(f"gene cols {len(gene_cols)} -> mapped {len(mapped)}")

    row_of = {L: {} for L in LINES}
    sums = {L: [] for L in LINES}
    cnts = {L: [] for L in LINES}

    for batch in f.iter_batches(batch_size=BATCH):
        line = batch.column(0).to_pylist()
        tgt = batch.column(2).to_pylist()
        V = np.column_stack([
            batch.column(3 + int(i)).to_numpy(zero_copy_only=False)
            for i in src_cols]).astype(np.float64)
        for k, (L, t) in enumerate(zip(line, tgt)):
            if L not in row_of:
                continue
            j = row_of[L].get(t)
            if j is None:
                j = len(sums[L])
                row_of[L][t] = j
                sums[L].append(np.zeros(len(tgt_cols)))
                cnts[L].append(0)
            sums[L][j] += V[k]
            cnts[L][j] += 1

    out = {"gpos": tgt_cols, "gpos_names": np.array([m[2] for m in mapped])}
    for L in LINES:
        if not row_of[L]:
            continue
        names_L = sorted(row_of[L], key=row_of[L].get)
        S = np.stack(sums[L]) / np.array(cnts[L])[:, None]
        out[f"{L}_names"] = np.array(names_L)
        out[f"{L}_rows"] = S.astype(np.float32)
        print(f"{L}: {len(names_L)} targets x {S.shape[1]} genes, "
              f"groups/target min {min(cnts[L])} max {max(cnts[L])}")
    np.savez_compressed(OUT, **out)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
