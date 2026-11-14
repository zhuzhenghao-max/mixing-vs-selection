"""fair-esm 本地权重 -> 基因蛋白平均池化嵌入.

torch>=2.6 的 torch.load 默认 weights_only=True 拒绝 fair-esm 检查点 -> 装载前打补丁.
输出 npz: names (sorted), E (n x d), 均为最后一层 per-residue 表示的掩码均值.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_orig_load = torch.load


def _load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _load

sys.path.insert(0, "src")
import esm  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--seqs", default="data/proteomes/gene_seqs.json")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1022)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    with open(args.seqs) as f:
        seqs = json.load(f)
    names = sorted(seqs)
    items = [(g, seqs[g][:args.max_len]) for g in names]
    lens = np.array([len(s) for _, s in items])
    print(f"{len(items)} seqs, len median {int(np.median(lens))} max {lens.max()}")

    model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.weights)
    model.eval()
    n_layers = model.num_layers
    d = model.embed_dim
    print(f"model loaded: layers {n_layers}, dim {d}")
    batch_converter = alphabet.get_batch_converter()

    order = np.argsort(lens)
    E = np.zeros((len(items), d), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, len(order), args.batch):
            idx = order[k:k + args.batch]
            batch = [items[i] for i in idx]
            _, _, toks = batch_converter(batch)
            out = model(toks, repr_layers=[n_layers], return_contacts=False)
            rep = out["representations"][n_layers]
            for bi, i in enumerate(idx):
                L = len(items[i][1])
                E[i] = rep[bi, 1:L + 1].mean(0).numpy()
            if k % (args.batch * 25) == 0:
                print(f"  {k}/{len(order)}", flush=True)
    np.savez(args.out, names=np.array(names), E=E)
    print(f"saved {args.out} {E.shape}")


if __name__ == "__main__":
    main()
