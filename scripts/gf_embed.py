"""Geneformer V2-316M 嵌入提取 (564 基因: 285 train + 300 panel).

两条路线:
  (a) token 嵌入 = word_embeddings 直接行 (类比 scGPT 提取);
  (b) 上下文嵌入 = 单基因输入 [CLS, g] 的最后一层 hidden state (类比 ESM 均值池化).
symbol -> ENSG (gene_name_id_dict_gc104M, TAZ->TAFAZZIN) -> token id (token_dictionary).
输出 outputs/gf316_emb.npz {names, E} 和 outputs/gf316_ctx_emb.npz.
"""

import pickle

import torch  # 必须先于 numpy/pandas 导入 (否则 c10.dll WinError 1114)
import numpy as np
import pandas as pd
from safetensors import safe_open
from transformers import BertConfig, BertModel

MODEL = "models/geneformer/v2_316m/model.safetensors"
ALIAS = {"TAZ": "TAFAZZIN"}


def main():
    td = pickle.load(open("models/geneformer/dicts/token_dictionary_gc104M.pkl", "rb"))
    nd = pickle.load(open("models/geneformer/dicts/gene_name_id_dict_gc104M.pkl", "rb"))
    ens2tok = {k: v for k, v in td.items() if isinstance(k, str) and k.startswith("ENSG")}

    m = np.load("outputs/scvi_map_cache.npz")
    train = list(m["train_names"].astype(str))
    panel = pd.read_csv("data/pert_counts.csv")["target_gene"].astype(str).tolist()
    names = train + [g for g in panel if g not in set(train)]
    print(f"{len(names)} genes")

    toks = np.array([ens2tok[nd[ALIAS.get(g, g)]] for g in names])
    assert len(set(toks)) == len(toks)

    f = safe_open(MODEL, framework="pt")
    W = f.get_tensor("bert.embeddings.word_embeddings.weight").numpy()
    print("word_embeddings", W.shape)
    E = W[toks]
    np.savez("outputs/gf316_emb.npz", names=np.array(names), E=E.astype(np.float32))
    print("saved outputs/gf316_emb.npz", E.shape)

    cfg = BertConfig.from_json_file("models/geneformer/v2_316m/config.json")
    model = BertModel(cfg, add_pooling_layer=False)
    sd = {k[5:]: f.get_tensor(k).clone() for k in f.keys() if k.startswith("bert.")}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("encoder state dict loaded, running contextual forward")
    cls = td["<cls>"]
    ids = torch.tensor([[cls, t] for t in toks], dtype=torch.long)
    with torch.no_grad():
        out = model(ids).last_hidden_state
    Ec = out[:, 1, :].numpy()
    np.savez("outputs/gf316_ctx_emb.npz", names=np.array(names), E=Ec.astype(np.float32))
    print("saved outputs/gf316_ctx_emb.npz", Ec.shape)


if __name__ == "__main__":
    main()
