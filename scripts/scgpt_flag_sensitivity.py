"""flag 敏感度快检: pert flag 是否真的影响输出 (GPU 秒级, CPU 用 --max-genes 冒烟).

第一轮 FT 的 gate delta 行间 cos=1.0000 / rel 1.5e-5 — flag 路径被网络抵消, gate 无效.
重训前后各跑一次本脚本: flag=1 vs flag=0 的前向输出差必须 >> 数值噪声 (rel > 1e-3).

兼容两种 ckpt:
- 预训练 ckpt (scGPT_human/best_model.pt): 旧式 flag_encoder 行灌入 pert_encoder 0:2,
  用于确认预训练模型对 flag 是否有先天响应.
- FT ckpt (outputs/scgpt_ft/*.pt): 整表加载, 不再注入.
两种 ckpt 的 self_attn 键可能带/不带 ._impl 包装, 自动双向重映射.

用法 (GPU, 全序列 18080 基因):
  python scripts/scgpt_flag_sensitivity.py --model-dir models/scgpt/scGPT_human \
    --ckpt outputs/scgpt_ft/best_model.pt --ctrl outputs/ctrl_slices/ctrl_rpe1.npz
用法 (CPU 冒烟, 基因子集):
  ... --max-genes 2048 --n-cells 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab

PAD_VALUE = 0


def load_ckpt_into(model, ckpt):
    """按形状匹配 + _impl 双向重映射载入, 返回 (n_loaded, n_total)."""
    sd = model.state_dict()
    variants = [
        lambda k: k,
        lambda k: k.replace(".self_attn.", ".self_attn._impl."),
        lambda k: k.replace(".self_attn._impl.", ".self_attn."),
        # CPU (use_fast_transformer=False) 的 nn.MultiheadAttention 参数名:
        # flash 的 Wqkv → in_proj_weight/in_proj_bias, out_proj 同名
        lambda k: k.replace(".self_attn._impl.Wqkv.weight", ".self_attn.in_proj_weight")
                   .replace(".self_attn._impl.Wqkv.bias", ".self_attn.in_proj_bias")
                   .replace(".self_attn._impl.", ".self_attn."),
    ]

    def count(f):
        return sum(1 for k, v in ckpt.items() if f(k) in sd and sd[f(k)].shape == v.shape)

    remap = max(variants, key=count)
    new = {}
    for k, v in ckpt.items():
        k2 = remap(k)
        if k2 in sd and sd[k2].shape == v.shape:
            new[k2] = v
    sd.update(new)
    model.load_state_dict(sd, strict=False)
    return len(new), len(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--targets", default="", help="逗号分隔; 默认轴上前 3 个等距基因")
    ap.add_argument("--n-cells", type=int, default=2)
    ap.add_argument("--max-genes", type=int, default=0, help="0=全轴; CPU 冒烟给 2048")
    ap.add_argument("--pert-pad-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred-mode", choices=["absolute", "delta"], default="absolute",
                    help="delta: 残差训练模型, 输出即 delta; 判据改为 absmax+靶间 cos")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    z = np.load(args.ctrl)
    axis_genes = [str(g) for g in z["genes"]]
    vals = z["values"][: args.n_cells].astype(np.float32)
    if args.max_genes > 0:
        axis_genes = axis_genes[: args.max_genes]
        vals = vals[:, : args.max_genes]
    n_g = len(axis_genes)

    vocab = GeneVocab.from_file(str(Path(args.model_dir) / "vocab.json"))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in axis_genes], dtype=int)

    mc = json.load(open(Path(args.model_dir) / "args.json"))
    model = TransformerGenerator(
        len(vocab), mc["embsize"], mc["nheads"], mc["d_hid"], mc["nlayers"],
        nlayers_cls=mc.get("n_layers_cls", 3), n_cls=1, vocab=vocab, dropout=0.2,
        pad_token="<pad>", pad_value=PAD_VALUE, pert_pad_id=args.pert_pad_id,
        use_fast_transformer=torch.cuda.is_available(),
    )
    ck = torch.load(args.ckpt, map_location="cpu")
    n_loaded, n_total = load_ckpt_into(model, ck)
    print(f"ckpt params loaded: {n_loaded}/{n_total}")
    if n_loaded < n_total * 0.9:
        raise SystemExit("ckpt 载入率过低 — 检查 flash-attn/_impl 键或 ckpt 版本")

    if "pert_encoder.weight" not in ck and "flag_encoder.weight" in ck:
        fe = ck["flag_encoder.weight"].float()
        with torch.no_grad():
            model.pert_encoder.weight.data[:2] = fe[:2]
        print(f"injected pretrained flag_encoder -> pert_encoder rows 0:2 "
              f"(norms {fe[0].norm():.2f}, {fe[1].norm():.2f})")

    model.to(device).eval()
    with torch.no_grad():
        pw = model.pert_encoder.weight
        print("pert_encoder row norms: " +
              " ".join(f"{i}={pw[i].norm():.2f}" for i in range(pw.shape[0])))

    if args.targets:
        tg = [t for t in args.targets.split(",") if t]
        for t in tg:
            if t not in axis_genes:
                raise SystemExit(f"target {t} 不在基因轴")
    else:
        tg = [axis_genes[i] for i in np.linspace(0, n_g - 1, 4).astype(int)[1:]]

    vv = torch.from_numpy(vals).to(device)
    gids = torch.tensor(
        [[vocab[g] if g in vocab else vocab["<pad>"] for g in axis_genes]], dtype=torch.long,
        device=device).repeat(vv.shape[0], 1)
    mask = torch.zeros_like(vv, dtype=torch.bool, device=device)

    def fwd(ff):
        torch.manual_seed(args.seed)  # 配对采样噪声, 差值只剩 flag 效应
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model(gids, vv, ff, src_key_padding_mask=mask,
                        CLS=False, CCE=False, MVC=False, ECS=False, do_sample=True)
        return out["mlm_output"].float().mean(0)

    with torch.no_grad():
        base = fwd(torch.zeros_like(vv, dtype=torch.long, device=device))
        denom = base.abs().mean().item() + 1e-8
        dds = []
        rels = []
        absmaxs = []
        coss = []
        for t in tg:
            fi = axis_genes.index(t)
            ff = torch.zeros_like(vv, dtype=torch.long, device=device)
            ff[:, fi] = 1
            d = fwd(ff) - base
            rel = d.abs().max().item() / denom
            peak = int(d.abs().argmax().item())
            dds.append(d.cpu().numpy())
            rels.append(rel)
            absmaxs.append(d.abs().max().item())
            print(f"{t:12s} fi={fi:5d} rel={rel:.2e} absmax={d.abs().max():.3e} "
                  f"peak_at={peak} ({axis_genes[peak]})")
        for i in range(len(dds)):
            for j in range(i + 1, len(dds)):
                a, b = dds[i], dds[j]
                c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                coss.append(c)
                print(f"cos({tg[i]},{tg[j]}) = {c:.4f}")

    med = float(np.median(rels))
    print(f"median rel = {med:.2e}")
    if args.pred_mode == "delta":
        # delta 模式 base≈0, rel 无意义; target 特异性的判据 = 有幅度 + 靶间 cos 明显 <1
        alive = max(absmaxs) > 1e-2 and (not coss or min(coss) < 0.9)
        print(f"NOTE: delta 模式; max absmax={max(absmaxs):.3e}, "
              f"min cos={min(coss) if coss else float('nan'):.4f}")
    else:
        alive = med > 1e-3
    print("VERDICT: FLAG_ALIVE" if alive else "VERDICT: FLAG_DEAD")


if __name__ == "__main__":
    main()
