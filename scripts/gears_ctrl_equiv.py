"""Check that GEARS uses the cell-graph input only as an additive residual.

The released GEARS model adds the per-cell expression vector x to the decoder
output (``out = ... + x.reshape(-1, 1)``) and never feeds it through a
nonlinearity.  Training on delta targets with x = 0 is therefore the same
model as training on absolute targets with x = control expression, provided
the loss shifts with x.  This script builds the K562 Tier A fold exactly as
scripts/gears_baseline.py does, takes one training batch, and compares

    (x = 0, y = delta)          vs      (x = c, y = delta + c)

for an arbitrary nonzero c.  Predictions must differ by c on every row and
MSE gradients must agree to floating point.  Results go to
outputs/gears_ctrl_equiv_A.log.
"""

import sys
import types

import numpy as np
import torch

sys.path.insert(0, "scripts")
import gears_baseline as gb  # noqa: E402


def flatten_grads(model):
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()
                      if p.grad is not None])


def main():
    torch.set_num_threads(4)
    args = types.SimpleNamespace(
        tier="A", fold="k562", epochs=1, batch=16, hidden=64,
        lr=1e-3, wd=5e-4, seed=0, threads=4, device="cpu",
        direction_lambda=0.1, patience=0, max_train=0, max_targets=0,
        max_train_perts=0, agg="mean", center=False, val_frac=0.08,
        val_mode="rows", coexpr_genes=2000, coexpr_rows=2500,
        no_unit=False, loss="mse", norm="bn", bn_cum=False,
        bn_recalib=False, diag_train=False, pert_lr_mult=1.0,
    )

    lines, _, pos, gidx, genes = gb.tier_data("A")
    gene_names = np.array(genes)[pos]
    perts = sorted({t for L in lines.values() for t in L["names"]})
    sets = gb.load_gene_sets(gb.GS_FILES)
    go_edges = gb.jaccard_edges(perts, sets, k=20)
    built = gb.build_fold(lines, "k562", pos, gidx, gene_names, perts, args)
    ce = gb.coexpr_edges(built["R"][built["train_idx"]], built["G"],
                         n_hvg=args.coexpr_genes, n_rows=args.coexpr_rows,
                         seed=args.seed)
    g = gb.make_model(built["pd"], go_edges, ce, args, "cpu", args.seed)
    m = g.model

    torch.manual_seed(123)
    batch = next(iter(g.dataloader["train_loader"]))
    B, G = batch.y.shape
    torch.manual_seed(7)
    c = torch.randn(G)

    state = {k: v.detach().clone() for k, v in m.state_dict().items()}

    b0 = batch.clone()
    b0.x = torch.zeros(B, G)
    bc = batch.clone()
    bc.x = c.unsqueeze(0).repeat(B, 1)
    m.eval()
    with torch.no_grad():
        p0 = m(b0)
        pc = m(bc)
    pred_err = float((pc - p0 - c.unsqueeze(0)).abs().max())
    shift = float((pc - p0).abs().max())

    m.load_state_dict(state)
    m.zero_grad(set_to_none=True)
    m.train()
    torch.manual_seed(11)
    loss0 = ((m(b0) - batch.y) ** 2).mean()
    loss0.backward()
    g0 = flatten_grads(m)
    m.load_state_dict(state)
    m.zero_grad(set_to_none=True)
    m.train()
    torch.manual_seed(11)
    lossc = ((m(bc) - (batch.y + c.unsqueeze(0))) ** 2).mean()
    lossc.backward()
    gc = flatten_grads(m)
    grad_err = float((g0 - gc).abs().max())
    grad_scale = float(g0.abs().max())

    lines_out = [
        f"fold k562  batch {B}  genes {G}",
        f"max |pred(c) - pred(0) - c| = {pred_err:.3e}",
        f"max |pred(c) - pred(0)|     = {shift:.3e}",
        f"loss zero-input {float(loss0):.8f}  control-input {float(lossc):.8f}",
        f"max |grad diff| = {grad_err:.3e}  grad scale = {grad_scale:.3e}",
    ]
    text = "\n".join(lines_out)
    print(text)
    with open("outputs/gears_ctrl_equiv_A.log", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
