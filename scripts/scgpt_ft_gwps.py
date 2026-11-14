"""scGPT perturbation 微调 (在 AutoDL GPU 实例上运行): GWPS K562 2M 细胞.

基于 scGPT 官方 Tutorial_Perturbation.ipynb, 数据换成 GWPS h5ad, 并修正两处:
- 细胞值 = log1p(CP1M), 基因轴 = GWPS var_names ∩ scGPT vocab
- pert_flags: 对照细胞全 0; 扰动细胞在被靶基因位置置 1
- 对照→扰动配对训练: 扰动样本的输入换成随机对照细胞的表达, 目标为该扰动细胞的
  表达. tutorial 的全重建自编码让 flag 完全冗余 (网络学会抵消, 第一轮 gate 行间
  cos=1.0000 即此因); 配对后 flag 是目标信息的唯一来源, 强制网络学其效应.
  对照样本保留自身表达, 充当无 flag 的恒等样本.
- 每 batch 抽 max_seq_len=1536 基因子集时强制并入本 batch 所有 target 位置
  (纯随机子集下 target 缺席约 89%, flag 几乎学不到).
- pert_encoder row1 用 std=0.03 初始化 (norm≈0.68): 预训练基因嵌入行范数仅≈1.54,
  默认 N(0,1) 的 flag 向量 norm≈22.6, 在 target 位置把基因身份彻底淹没 — 第二轮
  FT 实测不同靶 delta cos≈0.996 即此因. 小初始化让 src[target] 重新可见.
- 残差训练: pred = mlm_output + 输入表达, 对 target 做 MSE. 控制样本的残差目标恰为 0,
  网络容量集中在扰动 delta 而非绝对表达重建; 推理时 mlm_output 即 delta
  (scgpt_zero_shot.py --pred-mode delta).
- **--pbdelta (r3)**: 单细胞 target 的 control-vs-perturbed 差值被细胞噪声淹没
  (norm ≈ 输入表达范数), MSE 只能学到边际均值 + 自基因 KD. 改为 pseudobulk delta:
  每靶 d = log2((mean_cpm_pert+eps)/(mean_cpm_ctrl+eps)), ctrl<1CPM 位置屏蔽,
  clip ±4; pred = mlm_output - 输入表达; 靶基因位置从 loss 剔除 (断掉自 KD 捷径).
  推理 = mean(mlm_output) - mean(输入) = zero_shot 的 --pred-mode absolute.
  AffineExprDecoder 是 A·x+b, 该形式下 coeff→1, bias→靶特异 delta.
- **--center-pb (r5/r6)**: pb 目标减去 loss 加权的跨靶均值 (常数预测的最优解),
  去掉"所有靶共享的均值分量"这条捷径, loss 只能靠靶特异分量下降.
  简单行均值不够: cell 数加权残差仍有 row norm 的 ~18%. r5 用行均值, r6 改加权.
  pb_mean 存到 out_dir/pb_mean.npz, 推理时需加回.
- TransformerGenerator 加载预训练 backbone (encoder/value_encoder/transformer_encoder),
  AffineExprDecoder 从零训练; masked_mse_loss 全位置重建

用法 (AutoDL):
  pip install scgpt torch --index-url ... (CUDA 版 torch)
  python scripts/scgpt_ft_gwps.py \
    --data data/external/gwps_e/k562_gwps_essential_labeled.h5ad \
    --model-dir models/scgpt/scGPT_human \
    --out-dir outputs/scgpt_ft \
    --epochs 3 --batch-size 48 --cells-per-pert 80 --max-cells 0
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from scgpt.model import TransformerGenerator
from scgpt.loss import masked_mse_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

MAX_SEQ_LEN = 1536
PAD_VALUE = 0
PERT_PAD_ID = 0
MASK_POSITIONS = True  # 全位置重建 (tutorial 口径)
CONTROL_NAMES = {"control", "non-targeting", "non_targeting", "nt", "nontargeting",
                 "intergenic", "ctrl", "control_non-targeting"}


def load_gwps(data_path, model_dir, cells_per_pert, max_cells, seed):
    """返回 (values float16 [n_cells, n_genes], pert_flags uint8, gene_names, ctrl_idx)."""
    import anndata as ad

    vocab = GeneVocab.from_file(str(Path(model_dir) / "vocab.json"))
    vocab.set_default_index(vocab["<pad>"])

    a = ad.read_h5ad(data_path, backed="r")
    var_names = a.var_names.to_numpy().astype(str)
    if not any(g in vocab for g in var_names) and "gene_name" in a.var.columns:
        var_names = a.var["gene_name"].to_numpy().astype(str)
        print("var_names 非符号, 改用 var['gene_name']")
    keep = np.array([g in vocab for g in var_names])
    seen = set()
    for i in np.where(keep)[0]:
        if var_names[i] in seen:
            keep[i] = False
        else:
            seen.add(var_names[i])
    genes = var_names[keep]
    print(f"genes in vocab: {keep.sum()}/{len(var_names)} (deduped)")

    pert = a.obs["gene"].to_numpy().astype(str)
    gpos = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(seed)

    # 选细胞: 每扰动至多 cells_per_pert, 对照 (non-targeting) 至多 30000
    sel = []
    for label, idx in _group_indices(pert).items():
        idx = np.asarray(idx)
        if label.lower().replace(" ", "") in CONTROL_NAMES:
            take = min(len(idx), 30000)
        elif label in gpos:
            take = min(len(idx), cells_per_pert) if cells_per_pert > 0 else len(idx)
        else:
            continue
        if take < len(idx):
            idx = idx[rng.choice(len(idx), take, replace=False)]
        sel.extend(idx)
    sel = np.array(sel, dtype=np.int64)
    if max_cells > 0 and len(sel) > max_cells:
        sel = sel[rng.choice(len(sel), max_cells, replace=False)]
    sel.sort()
    print(f"selected {len(sel)} cells (of {len(pert)})")

    # 值矩阵 (chunked CP1M)
    n_genes = keep.sum()
    values = np.zeros((len(sel), n_genes), dtype=np.float16)
    fl = np.zeros((len(sel), n_genes), dtype=np.uint8)
    labels = pert[sel]
    CH = 20000
    for s in range(0, len(sel), CH):
        e = min(s + CH, len(sel))
        blk = a.X[sel[s:e]]
        if hasattr(blk, "toarray"):
            blk = blk.toarray()
        blk = np.asarray(blk, dtype=np.float32)
        blk = blk[:, keep]
        tot = blk.sum(1)
        vals = np.log1p(blk / np.maximum(tot, 1)[:, None] * 1e6)
        values[s:e] = vals.astype(np.float16)
        for i, lab in enumerate(labels[s:e]):
            if lab.lower().replace(" ", "") not in CONTROL_NAMES and lab in gpos:
                fl[s + i, gpos[lab]] = 1
        if (s // CH) % 10 == 0:
            print(f"  cells {s}/{len(sel)}", flush=True)
    a.file.close()
    is_ctrl = np.array([lab.lower().replace(" ", "") in CONTROL_NAMES for lab in labels])
    ctrl_idx = np.where(is_ctrl)[0]
    print(f"controls: {len(ctrl_idx)} / {len(sel)}")
    return values, fl, genes, ctrl_idx


def _group_indices(arr):
    d = {}
    for i, v in enumerate(arr):
        d.setdefault(v, []).append(i)
    return d


def counterfactual_diag(model, values, flags, ctrl_idx, ctrl_mask, pb_t, valid_t,
                        cell_row, train_idx, n_genes, gene_ids, args, device, tag):
    """固定 batch 上的反事实 loss 分解: 实际 pred / flag 清零 / 零预测 / 常数预测.

    回答"loss 下降到底来自什么": 若 l_flag0 ≈ l_model, flag 无贡献;
    若 l_const_gene 解释大部分降幅, 模型在学逐基因常数 (均值捷径).
    """
    rng = np.random.default_rng(123)
    bs = min(args.batch_size, len(train_idx))
    sel = train_idx[rng.permutation(len(train_idx))[:bs]]
    f_np = flags[sel].astype(np.int64)
    tpos = np.nonzero(f_np)[1]
    gsel_np = np.unique(np.concatenate(
        [rng.permutation(n_genes)[:MAX_SEQ_LEN], tpos]))
    pick = ctrl_idx[rng.integers(len(ctrl_idx), size=bs)]
    v_np = values[pick].astype(np.float32)
    own = ctrl_mask[sel]
    v_np[own] = values[sel[own]]

    f = torch.from_numpy(f_np).to(device)
    vv_in = torch.from_numpy(v_np[:, gsel_np]).to(device)
    ff = f[:, torch.from_numpy(gsel_np).to(device)]
    gids = torch.from_numpy(map_raw_id_to_vocab_id(gsel_np, gene_ids)).to(
        device).repeat(bs, 1)
    mask = torch.zeros_like(vv_in, dtype=torch.bool, device=device)
    rows = cell_row[torch.from_numpy(sel).to(device)]
    pert = rows >= 0
    target = torch.zeros_like(vv_in)
    valid = torch.ones_like(vv_in, dtype=torch.bool)
    loc = None
    if bool(pert.any()):
        target[pert] = pb_t[rows[pert]][:, torch.from_numpy(gsel_np).to(device)]
        valid[pert] = valid_t[rows[pert]][:, torch.from_numpy(gsel_np).to(device)]
        tg = f.argmax(1)
        loc = torch.searchsorted(torch.from_numpy(gsel_np).to(device), tg)
        if not args.keep_target_gene:
            valid[torch.arange(bs, device=device)[pert], loc[pert]] = False
    w = 1.0 + args.nonzero_weight * (target.abs() > 0.1).float()
    if loc is not None and args.target_gene_weight != 1.0:
        w[torch.arange(bs, device=device)[pert], loc[pert]] *= \
            args.target_gene_weight
    denom = valid.sum().clamp(min=1)

    def loss_of(pred):
        return (((pred - target) ** 2 * w * valid).sum() / denom).item()

    model.eval()
    amp_dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda",
                                                 dtype=amp_dtype):
        def fwd(flags_):
            return model(gids, vv_in, flags_, src_key_padding_mask=mask,
                         CLS=False, CCE=False, MVC=False, ECS=False)[
                             "mlm_output"].float() - vv_in
        pred = fwd(ff)
        pred0 = fwd(torch.zeros_like(ff))
        l_model, l_flag0 = loss_of(pred), loss_of(pred0)
        l_zero = loss_of(torch.zeros_like(vv_in))
        c_gene = (w * target * valid).sum(0) / (w * valid).sum(0).clamp(min=1)
        l_const_gene = loss_of(c_gene.unsqueeze(0).expand_as(vv_in))
        c_glob = (w * target * valid).sum() / (w * valid).sum().clamp(min=1)
        l_const_glob = loss_of(torch.full_like(vv_in, c_glob))
        if bool(pert.any()):
            pv = pred0[pert][valid[pert]]
            tv = target[pert][valid[pert]]
            cos0 = torch.nn.functional.cosine_similarity(pv, tv, dim=0).item()
            tnorm = tv.norm().item()
            ri = torch.arange(bs, device=device)[pert]
            self_pred = pred[ri, loc[pert]].mean().item()
            self_tgt = target[ri, loc[pert]].mean().item()
        else:
            cos0 = tnorm = self_pred = self_tgt = 0.0
        dnorm = (pred - pred0).norm().item()
        pnorm = pred.norm().item()
        p0norm = pred0.norm().item()
    model.train()
    print(f"  diag[{tag}] l_model={l_model:.5f} l_flag0={l_flag0:.5f} "
          f"l_zero={l_zero:.5f} l_const_gene={l_const_gene:.5f} "
          f"l_const_glob={l_const_glob:.5f}", flush=True)
    print(f"  diag[{tag}] |pred|={pnorm:.2f} |pred0|={p0norm:.2f} "
          f"|flagdelta|={dnorm:.2f} |target|={tnorm:.2f} cos(pred0,target)={cos0:.4f}",
          flush=True)
    print(f"  diag[{tag}] self_pred={self_pred:.3f} self_target={self_tgt:.3f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", default="outputs/scgpt_ft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cells-per-pert", type=int, default=80)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fast", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no-residual", action="store_true",
                    help="关闭残差训练 (pred=mlm_output 绝对表达); 默认残差 pred=mlm_output+输入")
    ap.add_argument("--pbdelta", action="store_true",
                    help="r3: pseudobulk delta 目标 + pred=mlm_output-输入 + 靶基因位置不入 loss")
    ap.add_argument("--min-cells-per-target", type=int, default=20,
                    help="--pbdelta 时每靶最少细胞数")
    ap.add_argument("--nonzero-weight", type=float, default=5.0,
                    help="--pbdelta 时 |delta|>0.1 基因的 loss 权重 (1=不加权)")
    ap.add_argument("--id-init", action="store_true",
                    help="decoder 末层初始化为 coeff=1, bias=0 → 起始 out=输入, pred=out-输入=0")
    ap.add_argument("--init-scale", type=float, default=0.0,
                    help="--id-init 时末层权重用 N(0,scale²) 而非精确 0: 严格 0 会让 "
                         "dL/dx=W4^T·... 为 0, transformer/flag 通路从第一步就断梯度")
    ap.add_argument("--pert-init-std", type=float, default=0.03,
                    help="pert_encoder 行初始化标准差 (默认 0.03, tutorial 用 N(0,1))")
    ap.add_argument("--keep-target-gene", action="store_true",
                    help="靶基因位置保留在 loss 中: flag 主要影响其自身位置的表示, "
                         "mask 掉后下游无梯度, flag 通路永远启动不了; 评测 selfex 本就剔除该位置")
    ap.add_argument("--target-gene-weight", type=float, default=1.0,
                    help="靶基因位置的 loss 权重倍数: 自效应大且方向一致, 加权可快速"
                         "启动 flag 通路; 评测 selfex 剔除该位置, 训练加权不影响得分")
    ap.add_argument("--tgw-decay-batches", type=int, default=0,
                    help=">0: 靶基因权重从 target-gene-weight 线性衰减到 1 的 batch 数 "
                         "(curriculum: 先强制学会自效应, 再转向下游)")
    ap.add_argument("--bf16", action="store_true",
                    help="用 bf16 autocast (4090D 支持), 避免 fp16 梯度 inf 导致的跳步")
    ap.add_argument("--center-pb", action="store_true",
                    help="pbdelta 目标减去 loss 加权跨靶均值 (去掉均值捷径, loss 只能靠靶特异分量下降)")
    ap.add_argument("--holdout-targets", default="",
                    help="csv 中的扰动靶完全不参与训练 (留出靶泛化检验)")
    ap.add_argument("--diag-only", default="",
                    help="加载该 ckpt, 只跑反事实 loss 诊断后退出 (不训练)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    values, flags, genes, ctrl_idx = load_gwps(
        args.data, args.model_dir, args.cells_per_pert, args.max_cells, args.seed)
    n_cells, n_genes = values.shape
    ctrl_mask = np.zeros(n_cells, dtype=bool)
    ctrl_mask[ctrl_idx] = True

    pb_t = valid_t = cell_row = None
    if args.pbdelta:
        EPS = 1e-9
        ctrl_cpm = np.expm1(values[ctrl_idx].astype(np.float32)).mean(0)
        gene_ok = ctrl_cpm >= 1.0
        tgt_gene = flags.argmax(1).astype(np.int64)
        pert_mask = ~ctrl_mask
        uniq, inv, counts = np.unique(
            tgt_gene[pert_mask], return_inverse=True, return_counts=True)
        keep_t = counts >= args.min_cells_per_target
        if args.holdout_targets:
            import csv
            with open(args.holdout_targets) as f:
                rd = csv.DictReader(f)
                col = next((c for c in ("target_gene", "target", "gene")
                            if c in (rd.fieldnames or [])), None)
                hold = {(row[col] if col else list(row.values())[0]).strip()
                        for row in rd}
            hold_idx = np.array([str(genes[u]) in hold for u in uniq])
            found = {str(genes[u]) for u in uniq if str(genes[u]) in hold}
            print(f"holdout: {hold_idx.sum()} targets in data, "
                  f"{len(hold - found)} listed but absent")
            keep_t &= ~hold_idx
        print(f"pbdelta targets: {keep_t.sum()}/{len(uniq)} "
              f"(>= {args.min_cells_per_target} cells), gene_ok {gene_ok.sum()}/{n_genes}")
        pb = np.zeros((len(uniq), n_genes), dtype=np.float32)
        for j in np.where(keep_t)[0]:
            cells = np.where(pert_mask & (tgt_gene == uniq[j]))[0]
            m = np.expm1(values[cells].astype(np.float32)).mean(0)
            d = np.log2((m + EPS) / (ctrl_cpm + EPS))
            d[~gene_ok] = 0.0
            pb[j] = np.clip(d, -4, 4)
        pb_mean = None
        if args.center_pb:
            # 去除的是 loss 加权的跨靶均值 (常数预测的最优解), 不是简单行均值:
            # 行均值消除后 cell 数加权残差仍有 row norm 的 ~18%, 模型会学它当捷径
            kpb = pb[keep_t]
            s_t = counts[keep_t].astype(np.float64) * (
                n_genes + args.nonzero_weight * (np.abs(kpb) > 0.1).sum(1))
            den = s_t.sum() + len(ctrl_idx) * n_genes
            pb_mean = ((kpb * s_t[:, None]).sum(0) / den).astype(np.float32)
            pb[keep_t] -= pb_mean
            print(f"center-pb(weighted): removed mean norm "
                  f"{np.linalg.norm(pb_mean):.3f}, row-mean norm "
                  f"{np.linalg.norm(kpb.mean(0)):.3f}, centered row norm median "
                  f"{np.median(np.linalg.norm(pb[keep_t], axis=1)):.2f}")
        # pb 按 uniq 完整位置索引 (未保留的靶行为零), cell_row 必须存 uniq 位置;
        # 之前存的是压缩后的行号, 导致丢弃 min_cells_per_target 个靶之后全部行错位
        cell_row = np.full(n_cells, -1, dtype=np.int64)
        cell_row[pert_mask] = np.where(keep_t[inv], inv, -1)
        self_d = pb[keep_t][np.arange(keep_t.sum()), uniq[keep_t]]
        print(f"pb self-delta median {np.median(self_d):.2f} "
              f"mean {self_d.mean():.2f} (应为强负)")
        frac = float((np.abs(pb[keep_t]) > 0.1).mean())
        norm = np.linalg.norm(pb[keep_t], axis=1)
        print(f"pb delta: |d|>0.1 frac {frac:.3f}, row norm median {np.median(norm):.2f}")
        pb_t = torch.from_numpy(pb).to(device)
        valid_t = torch.from_numpy(
            np.repeat(gene_ok[None], len(uniq), 0)).to(device)
        train_idx = np.where(ctrl_mask | (cell_row >= 0))[0]
        cell_row = torch.from_numpy(cell_row).to(device)
        print(f"pbdelta cells used: {len(train_idx)}/{n_cells} "
              f"(dropped {n_cells - len(train_idx)} below min-cells)")
    elif args.no_residual:
        print("NOTE: --no-residual 仅对 cell 模式有效; pbdelta 恒为 mlm_output-输入")

    vocab = GeneVocab.from_file(str(Path(args.model_dir) / "vocab.json"))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=int)

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
        dropout=args.dropout,
        pad_token="<pad>",
        pad_value=PAD_VALUE,
        pert_pad_id=PERT_PAD_ID,
        use_fast_transformer=not args.no_fast,
    )
    ckpt = torch.load(Path(args.model_dir) / "best_model.pt", map_location="cpu")
    prefixes = ["encoder", "value_encoder", "transformer_encoder"]
    # flash-attn>=2.8 移除旧 FlashMHA, scGPT master 兼容层包成 self._impl →
    # 模型键 self_attn._impl.Wqkv.*, 官方 ckpt 键 self_attn.Wqkv.*, 需重映射
    sd = model.state_dict()
    pretrained = {}
    for k, v in ckpt.items():
        if not any(k.startswith(p) for p in prefixes):
            continue
        if k not in sd or sd[k].shape != v.shape:
            k2 = k.replace(".self_attn.", ".self_attn._impl.")
            if k2 != k and k2 in sd and sd[k2].shape == v.shape:
                k = k2
        pretrained[k] = v
    missing = [k for k in pretrained if k not in sd or sd[k].shape != pretrained[k].shape]
    print(f"pretrained params to load: {len(pretrained)}, skipped: {len(missing)}")
    if missing:
        print("  missing sample:", missing[:6])
    sd.update({k: v for k, v in pretrained.items() if k not in missing})
    model.load_state_dict(sd, strict=False)
    # flag 行小初始化: 预训练基因嵌入 norm≈1.54, 默认 N(0,1) 的 flag norm≈22.6 会淹没 target 身份
    with torch.no_grad():
        model.pert_encoder.weight.normal_(0.0, args.pert_init_std)
        model.pert_encoder.weight[PERT_PAD_ID].zero_()
        print(f"pert row norms init: " +
              " ".join(f"{i}={model.pert_encoder.weight[i].norm():.3f}"
                       for i in range(model.pert_encoder.weight.shape[0])))
        if args.id_init:
            for dec, val in ((model.decoder.coeff_decoder, 1.0),
                             (model.decoder.bias_decoder, 0.0)):
                if args.init_scale > 0:
                    dec.fc[-1].weight.normal_(0.0, args.init_scale)
                else:
                    dec.fc[-1].weight.zero_()
                dec.fc[-1].bias.fill_(val)
            print(f"decoder init: coeff=1(+N(0,{args.init_scale}^2)), bias=0")
            w4 = model.decoder.coeff_decoder.fc[-1].weight
            print(f"  init |W4| coeff {w4.norm():.4f} bias "
                  f"{model.decoder.bias_decoder.fc[-1].weight.norm():.4f}")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and not args.bf16))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "ft_args.json", "w"), indent=2)
    if args.center_pb:
        np.savez(out_dir / "pb_mean.npz", genes=genes, pb_mean=pb_mean)
        print(f"saved pb_mean (norm {np.linalg.norm(pb_mean):.3f}) "
              f"to {out_dir / 'pb_mean.npz'}")
    print(f"residual training: {not args.no_residual}")
    if not args.pbdelta:
        train_idx = np.arange(n_cells)
    n_batches = len(train_idx) // args.batch_size
    print(f"cells {len(train_idx)}/{n_cells}, genes {n_genes}, "
          f"batches/epoch {n_batches}")

    probe_rows = None
    if args.pbdelta:
        rows_keep = np.where(keep_t)[0]
        probe_rows = rows_keep[np.linspace(
            0, len(rows_keep) - 1, 4).astype(int)[1:]]
        if args.diag_only:
            model.load_state_dict(torch.load(args.diag_only, map_location=device),
                                  strict=False)
            model.to(device)
            counterfactual_diag(model, values, flags, ctrl_idx, ctrl_mask, pb_t,
                                valid_t, cell_row, train_idx, n_genes, gene_ids,
                                args, device, "ckpt")
            return
        counterfactual_diag(model, values, flags, ctrl_idx, ctrl_mask, pb_t,
                            valid_t, cell_row, train_idx, n_genes, gene_ids,
                            args, device, "init")

    gb = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train_idx))
        tot = 0.0
        tot_zero = 0.0
        tot_grad = 0.0
        tot_sp = 0.0
        tot_st = 0.0
        n_grad = 0
        n_inf = 0
        for b in range(n_batches):
            idx = train_idx[perm[b * args.batch_size:(b + 1) * args.batch_size].numpy()]
            bs = len(idx)
            f = torch.from_numpy(flags[idx].astype(np.int64)).to(device)

            # 对照→扰动配对: 扰动样本输入 = 随机对照细胞表达 (flag 是唯一目标信息);
            # 对照样本保留自身表达, 构成无 flag 的恒等样本
            pick = ctrl_idx[torch.randint(len(ctrl_idx), (bs,)).numpy()]
            v_np = values[pick].astype(np.float32)
            own = ctrl_mask[idx]
            v_np[own] = values[idx[own]]
            v = torch.from_numpy(v_np).to(device)

            # 基因子集强制并入本 batch 所有 target 位置, 保证 flag 在每个样本中可见
            gsel = torch.randperm(n_genes, device=device)[:MAX_SEQ_LEN]
            tpos = f.nonzero(as_tuple=False)[:, 1]
            gsel = torch.unique(torch.cat([gsel, tpos]))
            vv = v[:, gsel]
            ff = f[:, gsel]
            gids = map_raw_id_to_vocab_id(gsel.cpu().numpy(), gene_ids)
            gids = torch.from_numpy(gids).to(device).repeat(bs, 1)
            mask = torch.zeros_like(vv, dtype=torch.bool, device=device)

            tgw = 1.0
            if args.pbdelta:
                rows = cell_row[torch.from_numpy(idx).to(device)]
                pert = rows >= 0
                target = torch.zeros_like(vv)
                valid = torch.ones_like(vv, dtype=torch.bool)
                loc = None
                if bool(pert.any()):
                    target[pert] = pb_t[rows[pert]][:, gsel]
                    valid[pert] = valid_t[rows[pert]][:, gsel]
                    tg = f.argmax(1)
                    loc = torch.searchsorted(gsel, tg)
                    if not args.keep_target_gene:
                        valid[torch.arange(bs, device=device)[pert], loc[pert]] = False
            else:
                tgt = torch.from_numpy(values[idx].astype(np.float32)).to(device)
                target = tgt[:, gsel]

            with torch.cuda.amp.autocast(enabled=device.type == "cuda",
                                         dtype=amp_dtype):
                out = model(gids, vv, ff, src_key_padding_mask=mask,
                            CLS=False, CCE=False, MVC=False, ECS=False)
                pred = out["mlm_output"].float()
                if args.pbdelta:
                    pred = pred - vv
                    w = 1.0 + args.nonzero_weight * (target.abs() > 0.1).float()
                    tgw = args.target_gene_weight
                    if args.tgw_decay_batches > 0:
                        tgw = 1.0 + (args.target_gene_weight - 1.0) * max(
                            0.0, 1.0 - gb / args.tgw_decay_batches)
                    if loc is not None and tgw != 1.0:
                        w[torch.arange(bs, device=device)[pert], loc[pert]] *= tgw
                    se = (pred - target) ** 2 * w * valid
                    loss = se.sum() / valid.sum().clamp(min=1)
                else:
                    if not args.no_residual:
                        pred = pred + vv
                    pos = torch.ones_like(vv, dtype=torch.bool) if MASK_POSITIONS else None
                    loss = masked_mse_loss(pred, target, pos)
            zero_loss = (target ** 2 * w * valid).sum() / valid.sum().clamp(min=1) \
                if args.pbdelta else loss.detach()
            if args.pbdelta and loc is not None:
                ri = torch.arange(bs, device=device)[pert]
                tot_sp += float(pred[ri, loc[pert]].mean())
                tot_st += float(target[ri, loc[pert]].mean())

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            tot_zero += float(zero_loss)
            g = float(gnorm)
            if math.isfinite(g):
                tot_grad += g
                n_grad += 1
            else:
                n_inf += 1
            if b % 200 == 0:
                w4c = model.decoder.coeff_decoder.fc[-1].weight.norm().item()
                w4b = model.decoder.bias_decoder.fc[-1].weight.norm().item()
                print(f"  ep{epoch} batch {b}/{n_batches} loss {tot / (b + 1):.5f} "
                      f"zero {tot_zero / (b + 1):.5f} "
                      f"self {tot_sp / (b + 1):.3f}/{tot_st / (b + 1):.3f} "
                      f"gnorm {tot_grad / max(n_grad, 1):.2f} inf {n_inf} "
                      f"|W4| {w4c:.4f}/{w4b:.4f} tgw {tgw:.1f} "
                      f"{time.time() - t0:.0f}s", flush=True)
            gb += 1
        scheduler.step()
        print(f"epoch {epoch} done, avg loss {tot / n_batches:.5f}, "
              f"zero {tot_zero / n_batches:.5f}, "
              f"self {tot_sp / n_batches:.3f}/{tot_st / n_batches:.3f}, "
              f"gnorm {tot_grad / max(n_grad, 1):.2f}, inf {n_inf}, "
              f"{time.time() - t0:.0f}s", flush=True)
        if probe_rows is not None:
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(
                    enabled=device.type == "cuda", dtype=amp_dtype):
                cells = torch.from_numpy(
                    values[ctrl_idx[:4]].astype(np.float32)).to(device)
                gids_full = torch.from_numpy(map_raw_id_to_vocab_id(
                    np.arange(n_genes), gene_ids)).to(device).repeat(cells.shape[0], 1)
                mfull = torch.zeros_like(cells, dtype=torch.bool)
                zflag = torch.zeros_like(cells, dtype=torch.long)
                base = model(gids_full, cells, zflag, src_key_padding_mask=mfull,
                             CLS=False, CCE=False, MVC=False, ECS=False)["mlm_output"].float()
                ds = []
                for r in probe_rows:
                    g = int(uniq[r])
                    ff = torch.zeros_like(cells, dtype=torch.long)
                    ff[:, g] = 1
                    o = model(gids_full, cells, ff, src_key_padding_mask=mfull,
                              CLS=False, CCE=False, MVC=False, ECS=False)["mlm_output"].float()
                    d = (o - base).mean(0)
                    ds.append(d)
                    print(f"  probe {str(genes[g]):10s} absmax {d.abs().max():.3f} "
                          f"peak {genes[int(d.abs().argmax())]}")
                for i in range(len(ds)):
                    for j in range(i + 1, len(ds)):
                        c = torch.nn.functional.cosine_similarity(
                            ds[i], ds[j], dim=0).item()
                        print(f"  probe cos {i},{j} = {c:.4f}")
            model.train()
        if args.pbdelta:
            counterfactual_diag(model, values, flags, ctrl_idx, ctrl_mask, pb_t,
                                valid_t, cell_row, train_idx, n_genes, gene_ids,
                                args, device, f"ep{epoch}")
        torch.save(model.state_dict(), out_dir / f"model_ep{epoch}.pt")

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
