"""提取 scGPT vocab 嵌入到全部训练靶基因 (gwps + orion + rpe1 并集).

输出 outputs/scgpt_emb_all.npz: names, E (n x 512 float32)
"""

import json
import sys

sys.path.insert(0, "src")

import torch  # noqa: E402  (torch 必须先于 numpy, WinError 1114)

import numpy as np  # noqa: E402


def main():
    ck = torch.load("models/scgpt/scGPT_human/best_model.pt",
                    map_location="cpu", weights_only=False)
    W = ck["encoder.embedding.weight"].detach().cpu().numpy().astype(np.float32)
    vocab = json.load(open("models/scgpt/scGPT_human/vocab.json", encoding="utf-8"))
    # vocab.json: list of tokens; embedding row i <-> vocab[i]
    vocab = [str(v) for v in vocab]
    vpos = {v: i for i, v in enumerate(vocab)}
    print(f"embedding {W.shape}, vocab {len(vocab)}")

    need = set()
    for f, key in (("outputs/gwps_rows.npz", "names"),
                   ("outputs/rpe1_rows.npz", "names")):
        g = np.load(f, allow_pickle=True)
        need |= {str(x) for x in g[key].astype(str)}
    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    need |= {str(t) for t in meta["pert_to_id"].keys()}
    print(f"union targets: {len(need)}")

    names, E = [], []
    missing = 0
    for t in sorted(need):
        if t in vpos:
            names.append(t)
            E.append(W[vpos[t]])
        else:
            missing += 1
    print(f"found {len(names)}, missing {missing}: "
          f"{[t for t in sorted(need) if t not in vpos][:20]}")

    np.savez("outputs/scgpt_emb_all.npz",
             names=np.array(names), E=np.stack(E))
    print("saved outputs/scgpt_emb_all.npz")


if __name__ == "__main__":
    main()
