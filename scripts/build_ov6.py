"""组装 v6 override 表: v5 基础上, Orion (X-Atlas HCT116/HEK293T) 覆盖的 panel 行
换成跨系实测迁移 lfc_true 行.

- 迁移行: orion gene 轴 -> 我们 18080 轴 (symbol 匹配), NaN->0, filt 外置 0, clip ±4,
  跨系均值 (hct116/hek293t 都有时), 再用 c6 把中位 filt-范数对齐到 raw17 的 13.78
- 25 注入行 (raw17 H1 实测) 保持 bit-identical; 未覆盖行保持 v5 的 c5·U9
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")


def main():
    v5 = np.load("outputs/lfc_override_2026_v5.npz", allow_pickle=True)
    panel = list(v5["names"].astype(str))
    genes = v5["genes"].astype(str)
    tab = v5["lfc"].copy()
    v3 = np.load("outputs/lfc_override_2026_v3.npz", allow_pickle=True)
    v3_tab = {g: v3["lfc"][i] for i, g in enumerate(v3["names"].astype(str))}
    injected = set(v3_tab)  # 25 行, 不许动
    print(f"panel {len(panel)}, injected rows {len(injected)}")

    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    filt = m["filt"]

    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    o_genes = meta["genes"]
    gidx = {g: i for i, g in enumerate(genes)}
    ocols, opos = [], []
    for oc, g in enumerate(o_genes):
        if g in gidx:
            ocols.append(oc)
            opos.append(gidx[g])
    ocols = np.array(ocols)
    opos = np.array(opos)
    print(f"orion gene cols {len(o_genes)}, mapped to ours {len(ocols)}")

    pert_to_id = meta["pert_to_id"]
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    rowmap = {}
    for r in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[r])], {})[int(pert_ids[r])] = r

    def line_vec(pid, cl):
        r = rowmap[cl].get(pid)
        if r is None:
            return None
        v = np.asarray(L[r, ocols], dtype=np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v

    lines = ["hct116", "hek293t"]
    raw17 = [g for g in panel if g in injected]  # 17 raw + 8 decoded
    n_raw = np.median([np.linalg.norm(tab[[i for i, g in enumerate(panel) if g in injected][k]][filt])
                       for k, g in enumerate(raw17)])  # == 13.78 anchor
    print(f"raw17 anchor median filt-norm: {n_raw:.2f}")

    cov_rows = {}
    stats = {"hct116": 0, "hek293t": 0, "both": 0}
    for g in panel:
        if g in injected:
            continue
        pid = pert_to_id.get(g)
        if pid is None:
            continue
        vs, used = [], []
        for cl in lines:
            v = line_vec(pid, cl)
            if v is not None and np.abs(v).sum() > 0:
                vs.append(v)
                used.append(cl)
        if not vs:
            continue
        stats["both" if len(vs) == 2 else used[0]] += 1
        v = np.mean(vs, 0)
        full = np.zeros(len(genes))
        full[opos] = v
        full[~filt] = 0.0
        if np.abs(full).sum() == 0:
            continue
        cov_rows[g] = full
        if len(used) == 1:
            pass
    print(f"orion-covered new rows: {len(cov_rows)} (hct-only {stats['hct116']}, "
          f"hek-only {stats['hek293t']}, both {stats['both']})")
    assert not (set(cov_rows) & injected)

    nats = np.array([np.linalg.norm(v[filt]) for v in cov_rows.values()])
    c6 = n_raw / np.median(nats)
    print(f"orion natural median filt-norm {np.median(nats):.2f} -> c6 {c6:.3f}")
    weak = [(g, float(np.linalg.norm(cov_rows[g][filt]))) for g in cov_rows
            if np.linalg.norm(cov_rows[g][filt]) < 0.05 * np.median(nats)]
    print(f"near-zero rows: {len(weak)} {weak[:5]}")

    for g, v in cov_rows.items():
        i = panel.index(g)
        tab[i] = np.clip(c6 * v, -4, 4).astype(np.float32)

    iidx = [i for i, g in enumerate(panel) if g in injected]
    assert np.array_equal(tab[iidx], np.stack([v3_tab[panel[i]] for i in iidx]))
    changed = len(cov_rows)
    d = np.array([np.linalg.norm(tab[panel.index(g)][filt] - v5["lfc"][panel.index(g)][filt])
                  for g in cov_rows])
    print(f"v6 table {tab.shape}, max|.| {np.abs(tab).max():.2f}, finite {np.isfinite(tab).all()}, "
          f"changed {changed} rows, ||v6-v5|| on changed med {np.median(d):.2f}")
    np.savez("outputs/lfc_override_2026_v6.npz", names=np.array(panel), genes=genes, lfc=tab)
    print("saved outputs/lfc_override_2026_v6.npz")


if __name__ == "__main__":
    main()
