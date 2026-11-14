"""组装 v8 override 表: v7 基础上, 130 个 orion 行换成 orion+GWPS 单位范数混合.

- 混合: unit(orion 跨系均值 natural) 与 unit(gwps natural) 按 w=0.4/0.6 加权
  (orion_gwps_combine_check: w=0.4 代理 pds 0.7500 vs 单源 0.69)
- 无 gwps 覆盖的 orion 行保持 v7 bit-identical; 其余全部行 bit-identical (25 注入 + 134 gwps + 11 U9)
- 混合后 filt 外置 0, c8 对齐 raw17 锚 (~13.57), clip ±4
"""

import json
import sys

import numpy as np

sys.path.insert(0, "src")

W_ORION = 0.4


def main():
    v7 = np.load("outputs/lfc_override_2026_v7.npz", allow_pickle=True)
    v6 = np.load("outputs/lfc_override_2026_v6.npz", allow_pickle=True)
    v5 = np.load("outputs/lfc_override_2026_v5.npz", allow_pickle=True)
    v3 = np.load("outputs/lfc_override_2026_v3.npz", allow_pickle=True)
    panel = [str(x) for x in v7["names"].astype(str)]
    genes = v7["genes"].astype(str)
    tab = v7["lfc"].copy()
    injected = set(str(x) for x in v3["names"].astype(str))
    v3_tab = {g: v3["lfc"][i] for i, g in enumerate(v3["names"].astype(str))}

    changed6 = set(panel[i] for i in range(len(panel))
                   if np.linalg.norm(v6["lfc"][i] - v5["lfc"][i]) > 1e-9)
    print(f"panel {len(panel)}, injected {len(injected)}, orion rows {len(changed6)}")

    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    filt = m["filt"]

    # rebuild orion natural rows (same mapping as build_ov6)
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
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    rowmap = {}
    for r in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[r])], {})[int(pert_ids[r])] = r

    def orion_natural(t):
        pid = pert_to_id.get(t)
        if pid is None:
            return None
        vs = []
        for cl in ("hct116", "hek293t"):
            r = rowmap[cl].get(pid)
            if r is None:
                continue
            v = np.asarray(L[r, ocols], dtype=np.float64)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if np.abs(v).sum() > 0:
                vs.append(v)
        if not vs:
            return None
        full = np.zeros(len(genes))
        full[opos] = np.mean(vs, 0)
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

    mixed = {}
    orion_only = []
    for t in changed6:
        ov = orion_natural(t)
        if ov is None:
            orion_only.append(t)
            continue
        j = gmap.get(t)
        if j is None:
            orion_only.append(t)
            continue
        mix = W_ORION * unit(ov) + (1 - W_ORION) * unit(np.nan_to_num(grows[j]))
        mixed[t] = np.where(filt, mix, 0.0)
    print(f"mixed rows {len(mixed)}, orion-only kept {len(orion_only)} "
          f"{orion_only[:8]}")

    nats = np.array([np.linalg.norm(v[filt]) for v in mixed.values()])
    c8 = n_raw / np.median(nats)
    print(f"mixed natural median filt-norm {np.median(nats):.3f} -> c8 {c8:.3f}")

    for t, v in mixed.items():
        tab[panel.index(t)] = np.clip(c8 * v, -4, 4).astype(np.float32)

    assert np.array_equal(tab[iidx], np.stack([v3_tab[panel[i]] for i in iidx]))
    keep = [i for i, t in enumerate(panel) if t not in mixed]
    assert np.array_equal(tab[keep], v7["lfc"][keep]), "non-mixed rows changed!"
    d = np.array([np.linalg.norm(tab[panel.index(t)][filt] - v7["lfc"][panel.index(t)][filt])
                  for t in mixed])
    print(f"v8 table {tab.shape}, max|.| {np.abs(tab).max():.2f}, "
          f"changed {len(mixed)} rows, ||v8-v7|| on changed med {np.median(d):.2f}")
    np.savez("outputs/lfc_override_2026_v8.npz", names=np.array(panel), genes=genes, lfc=tab)
    print("saved outputs/lfc_override_2026_v8.npz")


if __name__ == "__main__":
    main()
