"""scGPT 零样本扰动预测 (AutoDL GPU): 微调后模型 + 目标系对照细胞.

对每个 context (A/B/C): 取对照细胞, 逐目标基因置 pert flag,
pred = mean_cells(pred_pert) - mean_cells(ctrl_values), 输出到 18080 override 轴.
全基因序列一次前向 (同 tutorial pred_perturb 口径, 全序列注意力传播 flag 效应;
需要 flash-attn, GPU 24GB 下 batch 8-16 无压力).
contexts CSV 的 path 可为原始 h5ad, 也可为 slice_controls.py 预切的 npz
(热点上传场景, 免传 645MB context 文件).

用法:
  python scripts/scgpt_zero_shot.py \
    --model-dir models/scgpt/scGPT_human \
    --ckpt outputs/scgpt_ft/best_model.pt \
    --contexts configs/contexts_2026.csv \
    --targets data/pert_counts.csv \
    --axis outputs/lfc_override_2026_v10.npz \
    --out outputs/scgpt_delta_2026.npz \
    --max-cells 1000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

PAD_VALUE = 0
PERT_PAD_ID = 0


CONTROL_NAMES = {"control", "non-targeting", "non_targeting", "nt", "nontargeting",
                 "intergenic", "ctrl", "control_non-targeting"}
LABEL_COLS = ["perturbation", "gene_target", "target_gene", "gene", "target",
              "condition"]


def read_controls(path, gene_axis, vocab, max_cells, seed):
    """返回 (values float32 [n_cells, n_gene_axis], cell_idx, axis_genes 用于映射)."""
    if path.endswith(".npz"):
        z = np.load(path)
        if "genes" in z:
            assert [str(g) for g in z["genes"]] == list(gene_axis), \
                "npz 基因轴与 GWPS∩vocab 不一致"
        vals = z["values"].astype(np.float32)
        print(f"{path}: {vals.shape[0]} control cells (presliced)")
        return vals

    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    var_names = a.var_names.to_numpy().astype(str)
    gpos = {g: i for i, g in enumerate(gene_axis)}
    col = np.array([gpos.get(g, -1) for g in var_names])
    has = col >= 0

    lab_col = next((c for c in LABEL_COLS if c in a.obs.columns), None)
    if lab_col is None:
        cmask = np.ones(a.n_obs, dtype=bool)
    else:
        labels = a.obs[lab_col].to_numpy().astype(str)
        cmask = np.array([l.lower().replace(" ", "") in CONTROL_NAMES
                          or "ontrol" in l for l in labels])
    cidx = np.where(cmask)[0]
    rng = np.random.default_rng(seed)
    if len(cidx) > max_cells:
        cidx = cidx[rng.choice(len(cidx), max_cells, replace=False)]
        cidx.sort()  # dense h5ad 花式索引要求升序
    print(f"{path}: {len(cidx)} control cells")

    n_genes = len(gene_axis)
    vals = np.zeros((len(cidx), n_genes), dtype=np.float32)
    CH = 20000
    for s in range(0, len(cidx), CH):
        e = min(s + CH, len(cidx))
        blk = a.X[cidx[s:e]]
        if hasattr(blk, "toarray"):
            blk = blk.toarray()
        blk = np.asarray(blk, dtype=np.float32)
        tot = blk.sum(1)
        cpm = blk / np.maximum(tot, 1)[:, None] * 1e6
        for i in range(e - s):
            vals[s + i, col[has]] = np.log1p(cpm[i, has])
    a.file.close()
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--axis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cells", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred-mode", choices=["absolute", "delta"], default="absolute",
                    help="absolute: 输出为表达, delta=mean(pred)-mean(ctrl); "
                         "delta: 残差训练模型, 输出本身即 delta")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, pred mode: {args.pred_mode}")

    vocab = GeneVocab.from_file(str(Path(args.model_dir) / "vocab.json"))
    vocab.set_default_index(vocab["<pad>"])

    import csv

    contexts = []
    with open(args.contexts) as f:
        for row in csv.DictReader(f):
            contexts.append((row["context"], row["path"]))
    print(f"contexts: {contexts}")

    # 基因轴 = 对照切片 npz 的 genes (免 GWPS 大文件依赖, 与训练轴解耦)
    if not contexts[0][1].endswith(".npz"):
        raise SystemExit("首行 context 需为预切 npz (基因轴来源)")
    z0 = np.load(contexts[0][1])
    axis_genes = [str(g) for g in z0["genes"]]
    print(f"gene axis: {len(axis_genes)} (from {contexts[0][1]})")

    ax = np.load(args.axis, allow_pickle=True)
    axnames = [str(x) for x in ax["genes"].astype(str)]
    axpos = {g: i for i, g in enumerate(axnames)}
    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in axis_genes], dtype=int)

    model_configs = json.load(open(Path(args.model_dir) / "args.json"))
    model = TransformerGenerator(
        len(vocab),
        model_configs["embsize"],
        model_configs["nheads"],
        model_configs["d_hid"],
        model_configs["nlayers"],
        nlayers_cls=model_configs.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=0.2,
        pad_token="<pad>",
        pad_value=PAD_VALUE,
        pert_pad_id=PERT_PAD_ID,
        use_fast_transformer=torch.cuda.is_available(),
    )
    ft_ckpt = torch.load(args.ckpt, map_location="cpu")
    md = model.state_dict()
    loaded = sum(1 for k in ft_ckpt if k in md and md[k].shape == ft_ckpt[k].shape)
    print(f"ckpt params loaded: {loaded}/{len(ft_ckpt)}")
    assert loaded > len(ft_ckpt) * 0.9, "ckpt 载入率过低 — 检查 flash-attn 是否安装"
    model.load_state_dict(ft_ckpt, strict=False)
    model.to(device)
    model.eval()

    targets = []
    with open(args.targets) as f:
        for row in csv.DictReader(f):
            targets.append(row.get("target_gene") or row.get("target") or
                           row.get("gene") or list(row.values())[0])
    print(f"targets: {len(targets)}")

    n_g = len(axis_genes)
    delta = {ctx: np.zeros((len(targets), len(axnames)), dtype=np.float32)
             for ctx, _ in contexts}
    axis_set = set(axis_genes)
    gsel_all = np.arange(n_g)
    gids_all = torch.from_numpy(
        map_raw_id_to_vocab_id(gsel_all, gene_ids)).to(device)

    with torch.no_grad():
        for ctx, path in contexts:
            ctrl = read_controls(path, axis_genes, vocab, args.max_cells, args.seed)
            n_cells = ctrl.shape[0]
            ctrl_t = torch.from_numpy(ctrl).to(device)
            ctrl_mean = ctrl_t.mean(0)
            print(f"context {ctx}: {n_cells} ctrl cells, full seq {n_g} genes")

            for ti, t in enumerate(targets):
                flag = 1 if t in axis_set else 0
                fi = axis_genes.index(t) if flag else -1
                acc = torch.zeros(n_g, device=device)
                for s in range(0, n_cells, args.batch):
                    e = min(s + args.batch, n_cells)
                    vv = ctrl_t[s:e]
                    ff = torch.zeros_like(vv, dtype=torch.long, device=device)
                    if flag:
                        ff[:, fi] = 1
                    gids = gids_all.repeat(e - s, 1)
                    mask = torch.zeros_like(vv, dtype=torch.bool, device=device)
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        out = model(gids, vv, ff, src_key_padding_mask=mask,
                                    CLS=False, CCE=False, MVC=False, ECS=False,
                                    do_sample=True)
                    acc += out["mlm_output"].float().sum(0)
                pred = acc / n_cells
                d = (pred if args.pred_mode == "delta" else pred - ctrl_mean).cpu().numpy()
                for j, g in enumerate(axis_genes):
                    if g in axpos:
                        delta[ctx][ti, axpos[g]] = d[j]
                if (ti + 1) % 50 == 0:
                    print(f"  {ctx} {ti + 1}/{len(targets)}", flush=True)

    np.savez(args.out, targets=np.array(targets),
             genes=np.array(axnames),
             **{f"delta_{ctx}": delta[ctx] for ctx, _ in contexts})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
