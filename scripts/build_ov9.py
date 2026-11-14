"""组装 v9 override 表: v8 基础上, 130 个 orion 行的 orion 分量加 -log10(FDR) 权重.

- 113 混合行: O 分量换 fdr 加权版, 混法不变 (0.4*unit(O_fdr) + 0.6*unit(G))
  (代理 pds 0.7500 -> 0.7887, 同 harness; w=0.4 仍最优)
- 17 orion-only 行: 也换 fdr 加权 (单独代理 0.6901 -> 0.7113)
- 其余全部行 bit-identical (25 注入 + 134 gwps + 11 U9), 注入 == v3 断言
- fdr 权重: wt = clip(-log10(max(fdr,1e-4)), 0, 4), 两系取逐基因 max, 乘在均值 lfc 上
- 混合后 filt 外置 0, c9 对齐 raw17 锚 (~13.57), clip ±4
"""

import json
import sys

import numpy as np

sys.path.insert(0, "src")

W_ORION = 0.4


def main():
    v8 = np.load("outputs/lfc_override_2026_v8.npz", allow_pickle=True)
    v7 = np.load("outputs/lfc_override_2026_v7.npz", allow_pickle=True)
    v6 = np.load("outputs/lfc_override_2026_v6.npz", allow_pickle=True)
    v5 = np.load("outputs/lfc_override_2026_v5.npz", allow_pickle=True)
    v3 = np.load("outputs/lfc_override_2026_v3.npz", allow_pickle=True)
    panel = [str(x) for x in v8["names"].astype(str)]
    genes = v8["genes"].astype(str)
    tab = v8["lfc"].copy()
    injected = set(str(x) for x in v3["names"].astype(str))
    v3_tab = {g: v3["lfc"][i] for i, g in enumerate(v3["names"].astype(str))}

    changed6 = set(panel[i] for i in range(len(panel))
                   if np.linalg.norm(v6["lfc"][i] - v5["lfc"][i]) > 1e-9)
    print(f"panel {len(panel)}, injected {len(injected)}, orion rows {len(changed6)}")

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
    pert_to_id = meta["pert_to_id"]
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    FD = np.load("data/external/orion_fdr.npy", mmap_mode="r")
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    rowmap = {}
    for r in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[r])], {})[int(pert_ids[r])] = r

    def line_rows(t, cl):
        pid = pert_to_id.get(t)
        if pid is None:
            return None, None
        r = rowmap[cl].get(pid)
        if r is None:
            return None, None
        v = np.asarray(L[r, ocols], dtype=np.float64)
        f = np.asarray(FD[r, ocols], dtype=np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        f = np.nan_to_num(f, nan=1.0, posinf=1.0, neginf=1.0)
        return v, f

    def orion_fdr_full(t):
        vs, fs = [], []
        for cl in ("hct116", "hek293t"):
            v, f = line_rows(t, cl)
            if v is None or np.abs(v).sum() == 0:
                continue
            vs.append(v)
            fs.append(f)
        if not vs:
            return None
        wt = np.zeros_like(vs[0])
        for f in fs:
            wt = np.maximum(wt, np.clip(-np.log10(np.maximum(f, 1e-4)), 0, 4))
        full = np.zeros(len(genes))
        full[opos] = np.mean(vs, 0) * wt
        return full

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"].astype(str)]
    grows = g["rows"].astype(np.float64)
    gmap = {nm: i for i, nm in enumerate(gnames)}

    iidx = [i for i, t in enumerate(panel) if t in injected]
    n_raw = np.median([np.linalg.norm(tab[i][filt]) for i in iidx])
    print(f"raw17 anchor median filt-norm: {n_raw:.2f}")

    def unit(v):
        return v / max(np.linalg.norm(v), 1e-9)

    new_rows = {}
    for t in changed6:
        ov = orion_fdr_full(t)
        if ov is None:
            continue
        j = gmap.get(t)
        if j is not None:
            mix = W_ORION * unit(ov) + (1 - W_ORION) * unit(np.nan_to_num(grows[j]))
        else:
            mix = unit(ov)
        new_rows[t] = np.where(filt, mix, 0.0)
    print(f"new rows {len(new_rows)} (mixed {sum(1 for t in new_rows if t in gmap)}, "
          f"orion-only {sum(1 for t in new_rows if t not in gmap)})")

    nats = np.array([np.linalg.norm(v[filt]) for v in new_rows.values()])
    c9 = n_raw / np.median(nats)
    print(f"new natural median filt-norm {np.median(nats):.4f} -> c9 {c9:.3f}")

    for t, v in new_rows.items():
        tab[panel.index(t)] = np.clip(c9 * v, -4, 4).astype(np.float32)

    assert np.array_equal(tab[iidx], np.stack([v3_tab[panel[i]] for i in iidx]))
    keep = [i for i, t in enumerate(panel) if t not in new_rows]
    assert np.array_equal(tab[keep], v8["lfc"][keep]), "non-orion rows changed!"
    d = np.array([np.linalg.norm(tab[panel.index(t)][filt] - v8["lfc"][panel.index(t)][filt])
                  for t in new_rows])
    print(f"v9 table {tab.shape}, max|.| {np.abs(tab).max():.2f}, "
          f"changed {len(new_rows)} rows, ||v9-v8|| on changed med {np.median(d):.2f}")
    np.savez("outputs/lfc_override_2026_v9.npz", names=np.array(panel), genes=genes, lfc=tab)
    print("saved outputs/lfc_override_2026_v9.npz")


if __name__ == "__main__":
    main()
