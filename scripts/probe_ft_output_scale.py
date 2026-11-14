"""探针: FT ckpt 的 mlm_output 是绝对表达还是 delta.

AffineExprDecoder = coeff(x)*values + bias(x). 残差训练 pred=mlm_output+输入
等于又加一次输入; 若控制样本的 mean(mlm^2) ≈ 训练 loss (≈3.5), 说明模型实际
学的是"基因均值表达", 残差公式双重计数, delta 训练未生效.

用法:
  python scripts/probe_ft_output_scale.py --model-dir models/scgpt/scGPT_human \
    --ckpt outputs/scgpt_ft_r2/model_ep1.pt --ctrl outputs/ctrl_slices/ctrl_rpe1_300.npz \
    --target ATP6V0B --n-cells 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab

PAD_VALUE = 0
PERT_PAD_ID = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--target", default="")
    ap.add_argument("--n-cells", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(args.ctrl)
    genes = [str(g) for g in z["genes"]]
    vals = z["values"][: args.n_cells].astype(np.float32)

    vocab = GeneVocab.from_file(str(Path(args.model_dir) / "vocab.json"))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=int)
    mc = json.load(open(Path(args.model_dir) / "args.json"))
    model = TransformerGenerator(
        len(vocab), mc["embsize"], mc["nheads"], mc["d_hid"], mc["nlayers"],
        nlayers_cls=mc.get("n_layers_cls", 3), n_cls=1, vocab=vocab, dropout=0.2,
        pad_token="<pad>", pad_value=PAD_VALUE, pert_pad_id=PERT_PAD_ID,
        use_fast_transformer=torch.cuda.is_available(),
    )
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    model.to(device).eval()

    vv = torch.from_numpy(vals).to(device)
    gids = torch.tensor([gene_ids], device=device).repeat(vv.shape[0], 1)
    mask = torch.zeros_like(vv, dtype=torch.bool, device=device)
    ff0 = torch.zeros_like(vv, dtype=torch.long, device=device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        out0 = model(gids, vv, ff0, src_key_padding_mask=mask,
                     CLS=False, CCE=False, MVC=False, ECS=False)["mlm_output"].float()

    def stat(name, t):
        t = t.float()
        print(f"{name:18s} mean_abs={t.abs().mean():.4f} norm={t.norm():.2f} "
              f"mean_sq={t.pow(2).mean():.4f}")

    stat("input values", vv)
    stat("mlm_out flag0", out0)
    stat("out0+input", out0 + vv)
    c = torch.nn.functional.cosine_similarity(
        out0.flatten(), vv.flatten(), dim=0).item()
    print(f"cos(out0, input) = {c:.4f}")
    print(f"control-sample loss mean((out0+v-in)^2) = "
          f"{(out0 + vv - vv).pow(2).mean().item():.4f}")

    if args.target:
        if args.target not in genes:
            raise SystemExit(f"{args.target} not in axis")
        fi = genes.index(args.target)
        ff1 = torch.zeros_like(vv, dtype=torch.long, device=device)
        ff1[:, fi] = 1
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out1 = model(gids, vv, ff1, src_key_padding_mask=mask,
                         CLS=False, CCE=False, MVC=False, ECS=False)["mlm_output"].float()
        d = (out1 - out0).mean(0)
        print(f"flag1-flag0 delta on {args.target}: norm={d.norm():.4f} "
              f"absmax={d.abs().max():.4f} own={d[fi]:.4f} "
              f"cos(d,out0.mean0)={torch.nn.functional.cosine_similarity(d, out0.mean(0), dim=0).item():.4f}")


if __name__ == "__main__":
    main()
