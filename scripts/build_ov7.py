"""组装 v7 override 表: v6 基础上, GWPS (Replogle 2022 K562 全基因组, 9867 靶)
覆盖的未覆盖 panel 行换成跨系实测 lfc 行.

- 未覆盖行 (= v6 与 v5 逐位相同, 且非注入): outputs/gwps_rows.npz 的 our-axis 行
  (symbol 匹配已在 pseudobulk 完成), NaN->0, filt 外置 0, clip ±4,
  c7 把中位 filt-范数对齐到 raw17 锚 (~13.78)
- 25 注入行 (v3) + 130 orion 行 (v6) 保持 bit-identical
- usage: python scripts/build_ov7.py [min_cells=10]
"""

import sys

import numpy as np

sys.path.insert(0, "src")


def main():
    min_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    v6 = np.load("outputs/lfc_override_2026_v6.npz", allow_pickle=True)
    v5 = np.load("outputs/lfc_override_2026_v5.npz", allow_pickle=True)
    v3 = np.load("outputs/lfc_override_2026_v3.npz", allow_pickle=True)
    panel = [str(x) for x in v6["names"].astype(str)]
    genes = v6["genes"].astype(str)
    tab = v6["lfc"].copy()
    injected = set(str(x) for x in v3["names"].astype(str))
    v3_tab = {g: v3["lfc"][i] for i, g in enumerate(v3["names"].astype(str))}

    changed6 = set(panel[i] for i in range(len(panel))
                   if np.linalg.norm(v6["lfc"][i] - v5["lfc"][i]) > 1e-9)
    uncovered = [t for t in panel if t not in injected and t not in changed6]
    print(f"panel {len(panel)}, injected {len(injected)}, orion {len(changed6)}, "
          f"uncovered {len(uncovered)}")

    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    filt = m["filt"]

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"].astype(str)]
    grows = g["rows"].astype(np.float64)
    gcells = g["n_cells"].astype(int)
    rmap = {nm: i for i, nm in enumerate(gnames)}

    iidx = [i for i, t in enumerate(panel) if t in injected]
    n_raw = np.median([np.linalg.norm(tab[i][filt]) for i in iidx])
    print(f"raw17 anchor median filt-norm: {n_raw:.2f}")

    new_rows = {}
    low_cells = []
    for t in uncovered:
        j = rmap.get(t)
        if j is None:
            continue
        if gcells[j] < min_cells:
            low_cells.append((t, int(gcells[j])))
            continue
        v = np.nan_to_num(grows[j], nan=0.0, posinf=0.0, neginf=0.0)
        v = np.where(filt, v, 0.0)
        if np.abs(v).sum() == 0:
            continue
        new_rows[t] = v
    print(f"gwps-covered new rows: {len(new_rows)}/{len(uncovered)} "
          f"(low-cells<{min_cells} dropped {len(low_cells)} {low_cells[:8]})")
    if not new_rows:
        print("nothing to do")
        return

    nats = np.array([np.linalg.norm(v[filt]) for v in new_rows.values()])
    c7 = n_raw / np.median(nats)
    print(f"gwps natural median filt-norm {np.median(nats):.2f} -> c7 {c7:.3f}")
    weak = [(t, float(np.linalg.norm(new_rows[t][filt]))) for t in new_rows
            if np.linalg.norm(new_rows[t][filt]) < 0.05 * np.median(nats)]
    print(f"near-zero rows: {len(weak)} {weak[:5]}")

    for t, v in new_rows.items():
        tab[panel.index(t)] = np.clip(c7 * v, -4, 4).astype(np.float32)

    assert np.array_equal(tab[iidx], np.stack([v3_tab[panel[i]] for i in iidx]))
    oidx = [i for i, t in enumerate(panel) if t in changed6]
    assert np.array_equal(tab[oidx], v6["lfc"][oidx])
    d = np.array([np.linalg.norm(tab[panel.index(t)][filt] - v6["lfc"][panel.index(t)][filt])
                  for t in new_rows])
    print(f"v7 table {tab.shape}, max|.| {np.abs(tab).max():.2f}, "
          f"finite {np.isfinite(tab).all()}, changed {len(new_rows)} rows, "
          f"||v7-v6|| on changed med {np.median(d):.2f}")
    np.savez("outputs/lfc_override_2026_v7.npz", names=np.array(panel), genes=genes, lfc=tab)
    print("saved outputs/lfc_override_2026_v7.npz")


if __name__ == "__main__":
    main()
