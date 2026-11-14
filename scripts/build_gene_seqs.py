"""人参考蛋白组 FASTA -> 目标基因符号的蛋白序列表.

覆盖: 285 训练扰动靶基因 (scvi_map_cache train_names) + 300 val 面板靶基因 (pert_counts.csv).
UniProt reference proteome UP000005640_9606, 每符号取最长序列. 输出 data/proteomes/gene_seqs.json
"""

import gzip
import json
import re

import numpy as np
import pandas as pd

FASTA = "data/proteomes/UP000005640_9606.fasta.gz"
GN = re.compile(r"GN=([^\s]+)")

cache = np.load("outputs/scvi_map_cache.npz")
train_names = cache["train_names"].astype(str).tolist()
panel = pd.read_csv("data/pert_counts.csv")["target_gene"].astype(str).tolist()
wanted = sorted(set(train_names) | set(panel))
print(f"wanted {len(wanted)} (train {len(set(train_names))} + panel {len(set(panel))})")

seqs = {}
cur_gn = None
cur = []
with gzip.open(FASTA, "rt") as f:
    for line in f:
        if line.startswith(">"):
            if cur_gn is not None:
                s = "".join(cur)
                if cur_gn not in seqs or len(s) > len(seqs[cur_gn]):
                    seqs[cur_gn] = s
            m = GN.search(line)
            cur_gn = m.group(1) if m else None
            cur = []
        else:
            cur.append(line.strip())
    if cur_gn is not None:
        s = "".join(cur)
        if cur_gn not in seqs or len(s) > len(seqs[cur_gn]):
            seqs[cur_gn] = s

print(f"proteome symbols: {len(seqs)}")
out = {}
missing = []
for g in wanted:
    if g in seqs:
        out[g] = seqs[g]
    else:
        missing.append(g)
print(f"mapped {len(out)}/{len(wanted)}; missing {len(missing)}:")
for g in missing[:40]:
    print("  ", g)
lens = [len(v) for v in out.values()]
print(f"seq len: min {min(lens)} median {int(np.median(lens))} max {max(lens)}")

with open("data/proteomes/gene_seqs.json", "w") as f:
    json.dump(out, f)
print("saved data/proteomes/gene_seqs.json")
