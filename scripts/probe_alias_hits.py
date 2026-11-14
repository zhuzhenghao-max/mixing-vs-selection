"""终局面板别名挖掘: 检查 300 靶是否以别名出现在 H1/GWPS/orion 的扰动标签里.

H1 标签集 (Training deseq2 150 + holdout 15 + Validation 50 + train_perturbed 35)
优先于外部源; GWPS/orion 命中需要自基因 KD 验证.
"""

import csv
import gzip
import json
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd

PANEL = "data/final/pert_counts.csv"


def load_gene_info(path="data/external/Homo_sapiens.gene_info.gz"):
    alias_to_sym = defaultdict(set)
    sym_aliases = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 5 or p[0] != "9606":
                continue
            sym = p[2]
            syn = [s for s in p[4].split("|") if s and s != "-"]
            als = {sym, *syn}
            sym_aliases[sym] = als
            for a in als:
                alias_to_sym[a].add(sym)
    return sym_aliases, alias_to_sym


def als_for(t, sym_aliases, alias_to_sym):
    if t in sym_aliases:
        return set(sym_aliases[t])
    out = set()
    for sym in alias_to_sym.get(t, ()):
        out |= sym_aliases.get(sym, set())
        out.add(sym)
    return out


def main():
    panel = [r["target_gene"].strip() for r in csv.DictReader(open(PANEL))]
    sym_aliases, alias_to_sym = load_gene_info()

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"]]
    gset = set(gnames)
    gc = g["n_cells"].astype(int)
    gr = g["rows"].astype(np.float64)
    gm = {n: i for i, n in enumerate(gnames)}

    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    oset = set(meta["pert_to_id"].keys())

    train = set(map(str, pd.read_parquet("data/vccp_de_log2fc.parquet").index))
    hold = set(pd.read_csv("data/internal/holdout_genes.csv")["target_gene"].astype(str))
    val = set(r[0].strip() for r in
              list(csv.reader(open("data/pert_counts_Validation_2025.csv")))[1:])
    a_tp = ad.read_h5ad("data/internal/train_perturbed.h5ad", backed="r")
    tp = set(a_tp.obs["target_gene"].astype(str).to_numpy())
    a_tp.file.close()
    h1 = train | hold | val | tp
    print(f"H1 labels: train {len(train)} hold {len(hold)} val {len(val)} tp {len(tp)} "
          f"union {len(h1)}")

    gpos = None
    ov = np.load("outputs/lfc_override_2026_final.npz", allow_pickle=True)
    genes = ov["genes"].astype(str)
    gpos = {x: i for i, x in enumerate(genes)}
    u9 = np.load("outputs/lfc_add_emb_ens.npz", allow_pickle=True)
    e9 = set(str(x) for x in u9["names"])

    uncovered = [t for t in panel if t not in gset and t not in oset]
    print(f"\nuncovered by name (GWPS or orion pert): {len(uncovered)} {uncovered}")
    for t in uncovered:
        print(f"  {t}: in_u9={t in e9}, in_axis={t in gpos}")

    print("\n=== alias hits ===")
    for t in panel:
        als = als_for(t, sym_aliases, alias_to_sym)
        if not als:
            continue
        h1h = sorted(als & h1)
        gh = sorted(als & gset)
        oh = sorted(als & oset)
        if h1h or gh or oh:
            print(f"{t}: aliases={sorted(als)}")
            if h1h:
                print(f"   H1 hit {h1h} (train={[a in train for a in h1h]} "
                      f"hold={[a in hold for a in h1h]} val={[a in val for a in h1h]} "
                      f"tp={[a in tp for a in h1h]})")
            if gh:
                for a in gh:
                    j = gm[a]
                    kid = gr[j, gpos[t]] if t in gpos else float("nan")
                    print(f"   GWPS hit {a}: n={gc[j]} self_KD_at_{t}={kid:.2f}")
            if oh:
                print(f"   orion hit {oh}")


if __name__ == "__main__":
    main()
