"""终局面板: 统计 override 行来源/空洞, 并检查 9 个未覆盖靶是否有其它实测源."""

import csv

import numpy as np
import pyarrow.parquet as pq

PANEL = "data/final/pert_counts.csv"
OV = "outputs/lfc_override_2026_final.npz"
FILT = "outputs/scvi_map_cache.npz"


def main():
    panel = [r["target_gene"].strip() for r in csv.DictReader(open(PANEL))]
    print(f"panel {len(panel)} unique {len(set(panel))}")

    ov = np.load(OV, allow_pickle=True)
    names = [str(x) for x in ov["names"]]
    lfc = ov["lfc"]
    filt = np.load(FILT, allow_pickle=True)["filt"].astype(bool)
    gpos = {g: i for i, g in enumerate(ov["genes"].astype(str))}
    norms = np.linalg.norm(lfc[:, filt], axis=1)
    zero = [names[i] for i in range(len(names)) if np.abs(lfc[i][filt]).max() == 0]
    print(f"override {lfc.shape}; zero rows {len(zero)} {zero}")
    small = [(names[i], float(norms[i])) for i in np.argsort(norms)[:15]]
    print("smallest norms:", small)
    print("median filt-norm", float(np.median(norms)))

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = set(str(x) for x in g["names"])
    try:
        r = np.load("outputs/rpe1_rows.npz", allow_pickle=True)
        rnames = set(str(x) for x in r["names"])
    except FileNotFoundError:
        rnames = set()
        print("no rpe1_rows.npz")
    try:
        e = np.load("outputs/lfc_add_emb_ens.npz", allow_pickle=True)
        enames = set(str(x) for x in e["names"])
    except FileNotFoundError:
        enames = set()

    import json
    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    onames = set(meta["genes"])

    def iso(name):
        return [t for t in panel if t not in gnames and t not in onames]

    unc = iso(panel)
    print(f"\nnot in GWPS and not in orion: {len(unc)}")
    for t in unc:
        if t in gpos:
            print(f"  {t:10s} norm {norms[names.index(t)]:7.3f} "
                  f"gwps={t in gnames} orion={t in onames} rpe1={t in rnames} u9={t in enames}")
        else:
            print(f"  {t:10s} (missing from override axis)")

    pf = pq.ParquetFile("data/external/all_delta.parquet")
    tab = pf.read(columns=["cell_line", "gene"])
    lines = tab.column("cell_line").to_pylist()
    pert = tab.column("gene").to_pylist()
    pairs = set(zip(lines, pert))
    pert_set = set(pert)
    print(f"\nall_delta rows {len(pert)}, unique perts {len(pert_set)}, "
          f"lines {sorted(set(lines))}")
    hits = [t for t in panel if t in pert_set]
    print(f"panel targets in all_delta pert rows: {len(hits)} {hits}")
    unc_hits = [t for t in unc if t in pert_set]
    print(f"uncovered hits in all_delta: {unc_hits}")
    for t in unc_hits:
        print(f"  {t}: lines {sorted({ln for ln, g in pairs if g == t})}")
    hits_rpe1 = [t for t in panel if t in rnames]
    print(f"panel targets in rpe1_rows: {len(hits_rpe1)}")
    print(f"uncovered in rpe1: {[t for t in unc if t in rnames]}")
    print(f"uncovered in gwps: {[t for t in unc if t in gnames]}")
    print(f"uncovered in orion: {[t for t in unc if t in onames]}")


if __name__ == "__main__":
    main()
