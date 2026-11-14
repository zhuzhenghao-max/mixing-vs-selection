"""PerturbAtlas 第三源 -> H1 迁移检验 (与 orion_transfer_check 同口径).

- 每个 H1 靶取覆盖最大的 KD/KO 对比 (outputs/perturbatlas_h1_sel.json)
- DEG xlsx (ENSG) 经 HGNC 映射到 18080 轴; 变体: raw / clip4 / bm>=10 / bm>=50
- 参考 = H1 285 train 行残差 (scvi_map_cache LFC - mean), filt & finite 位
- raw-pds: 查询行 unit 后与参考池 cos, 真值行 tie-midrank
- spec corr: 查询行 vs 对应 H1 残差行 Pearson
- 对照: 同靶 GWPS 行 / unit-mix(PA,GWPS,w)
"""

import csv
import json
import os

import numpy as np
import pandas as pd


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def load_hgnc_map():
    sym_of = {}
    with open("data/external/hgnc_complete_set.txt", encoding="utf-8") as f:
        for x in csv.DictReader(f, delimiter="\t"):
            e = x.get("ensembl_gene_id")
            if e and e not in sym_of:
                sym_of[e] = x["symbol"]
    return sym_of


def build_rows(genes, sym_of, sel):
    gidx = {g: i for i, g in enumerate(genes)}
    out = {}
    for t, c in sel.items():
        path = f"outputs/perturbatlas_deg/{c['perturb_id']}_degs.xlsx"
        if not os.path.exists(path):
            print(f"  missing {t} {c['perturb_id']}")
            continue
        df = pd.read_excel(path)
        df["sym"] = df["gene"].map(sym_of)
        df = df[df["sym"].isin(gidx)]
        ax = df["sym"].map(gidx).to_numpy()
        v = np.zeros(len(genes))
        bm = np.zeros(len(genes))
        cnt = np.zeros(len(genes))
        np.add.at(v, ax, df["log2FoldChange"].to_numpy())
        np.add.at(bm, ax, df["baseMean"].to_numpy())
        np.add.at(cnt, ax, 1.0)
        v = v / np.maximum(cnt, 1.0)
        bm = bm / np.maximum(cnt, 1.0)
        out[t] = dict(row=v, base=bm, cond=c["condition"], tissue=c["tissue"],
                      pid=c["perturb_id"], total=c["total"],
                      n_axis=int((cnt > 0).sum()))
        print(f"  {t:9s} {c['perturb_id']} {str(c['tissue'])[:28]:28s} "
              f"genes_on_axis {int((cnt > 0).sum()):5d}")
    return out


def eval_variant(name, vecs, tn, pos, refn):
    idx = [i for i, t in enumerate(tn) if t in vecs]
    if len(idx) < 5:
        print(f"  {name:24s}: only {len(idx)} targets")
        return None
    P = np.stack([vecs[tn[i]][pos] for i in idx])
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    S = Pn @ refn.T
    n = len(tn)
    raws, cs = [], []
    for k, i in enumerate(idx):
        s = S[k]
        rank = 1.0 + np.sum(s > s[i]) + 0.5 * (np.sum(s == s[i]) - 1)
        raws.append(1.0 - (rank - 1.0) / (n - 1.0))
        a, b = P[k], refn[i]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            cs.append(np.corrcoef(a, b)[0, 1])
    cs = np.array(cs)
    print(f"  {name:24s} n={len(idx):3d} pds_med {np.median(raws):.4f} "
          f"pds_mean {np.mean(raws):.4f} spec_med {np.median(cs):+.3f} "
          f"spec_mean {cs.mean():+.3f}")
    return dict(n=len(idx), pds_med=float(np.median(raws)),
                pds_mean=float(np.mean(raws)), spec_med=float(np.median(cs)),
                spec_mean=float(cs.mean()))


def main():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes = m["genes"].astype(str)
    tn = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    n = len(tn)

    sel = json.load(open("outputs/perturbatlas_h1_sel.json", encoding="utf-8"))
    sym_of = load_hgnc_map()
    print("building PerturbAtlas rows:")
    rows = build_rows(genes, sym_of, sel)

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"].astype(str)]
    grows = g["rows"].astype(np.float64)
    gi = {t: i for i, t in enumerate(gnames)}
    gwps = {t: grows[gi[t]] for t in rows if t in gi}
    print(f"gwps overlap with PA targets: {len(gwps)}/{len(rows)}")

    pos = np.where(filt & np.all(np.isfinite(LFC), 0))[0]
    ref = LFC[:, pos].astype(np.float64)
    mvec = ref.mean(0)
    R = ref - mvec
    refn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
    print(f"eval positions: {len(pos)}, reference rows: {n}")

    raw = {t: v["row"] for t, v in rows.items()}
    clip = {t: np.clip(r, -4, 4) for t, r in raw.items()}
    bm10 = {t: v["row"] * (v["base"] >= 10) for t, v in rows.items()}
    bm50 = {t: v["row"] * (v["base"] >= 50) for t, v in rows.items()}

    print("\n=== PerturbAtlas variants ===")
    eval_variant("pa_raw", raw, tn, pos, refn)
    eval_variant("pa_clip4", clip, tn, pos, refn)
    eval_variant("pa_bm10", bm10, tn, pos, refn)
    eval_variant("pa_bm50", bm50, tn, pos, refn)

    print("\n=== GWPS baseline (same targets) ===")
    eval_variant("gwps", gwps, tn, pos, refn)

    print("\n=== unit mixes (w*unit(PA) + (1-w)*unit(GWPS)) ===")
    common = [t for t in rows if t in gwps]
    print(f"common targets: {len(common)}")
    for w in (0.2, 0.3, 0.4, 0.5, 0.6):
        mix = {t: w * unit(clip[t]) + (1 - w) * unit(gwps[t]) for t in common}
        eval_variant(f"mix_w{w:g}", mix, tn, pos, refn)
    eval_variant("gwps_common", {t: gwps[t] for t in common}, tn, pos, refn)

    names = sorted(rows)
    np.savez("outputs/perturbatlas_h1_rows.npz",
             names=np.array(names),
             rows=np.stack([rows[t]["row"] for t in names]).astype(np.float32),
             base=np.stack([rows[t]["base"] for t in names]).astype(np.float32),
             cond=np.array([rows[t]["cond"] for t in names]),
             tissue=np.array([rows[t]["tissue"] for t in names]),
             pid=np.array([rows[t]["pid"] for t in names]),
             genes=genes)
    print(f"\nsaved outputs/perturbatlas_h1_rows.npz ({len(names)} rows)")


if __name__ == "__main__":
    main()
