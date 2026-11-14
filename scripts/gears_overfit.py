"""Single-batch overfit test for the GEARS adaptation.

Trains the model on one fixed batch of 16 source rows for 120 Adam steps.
If the loss does not approach zero, the wiring or optimizer is broken rather
than the data being hard.

Usage: python scripts/gears_overfit.py [--no-unit] [--loss mse|quartic]
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "scripts")
import gears_baseline as gb


def main():
    args = gb.parse_args()
    args.tier, args.fold = "A", "k562"
    torch.set_num_threads(args.threads)
    device = "cpu"

    lines, holds, pos, gidx, genes = gb.tier_data(args.tier)
    gene_names = np.array(genes)[pos]
    perts = sorted({t for L in lines.values() for t in L["names"]})
    sets = gb.load_gene_sets(gb.GS_FILES)
    go_edges = gb.jaccard_edges(perts, sets, k=20)
    built = gb.build_fold(lines, "k562", pos, gidx, gene_names, perts,
                          args)
    ce = gb.coexpr_edges(built["R"][built["train_idx"]], built["G"],
                         n_hvg=args.coexpr_genes, n_rows=args.coexpr_rows,
                         seed=args.seed)
    g = gb.make_model(built["pd"], go_edges, ce, args, device, args.seed)

    opt = torch.optim.Adam(g.model.parameters(), lr=args.lr,
                           weight_decay=args.wd)
    g.model.train()
    batch = next(iter(g.dataloader["train_loader"])).to(device)
    y = batch.y
    print(f"y |mean| {y.abs().mean():.4f} mean y^2 {(y ** 2).mean():.6f}",
          flush=True)

    def run_loss():
        pred = g.model(batch)
        if args.loss == "mse":
            return ((pred - y) ** 2).mean(), pred
        return gb.loss_fct(pred, y, batch.pert, ctrl=g.ctrl_expression,
                           direction_lambda=args.direction_lambda,
                           dict_filter=g.dict_filter), pred

    with torch.no_grad():
        l0, p0 = run_loss()
    print(f"step 000 loss {l0:.6f} pred |mean| {p0.abs().mean():.4f} "
          f"pred rms {p0.pow(2).mean().sqrt():.4f}", flush=True)
    t0 = time.time()
    for step in range(1, 121):
        opt.zero_grad()
        loss, pred = run_loss()
        loss.backward()
        if step == 1:
            for name, p in g.model.named_parameters():
                if name in ("indv_b1", "indv_b2", "gene_emb.weight",
                            "pert_emb.weight"):
                    print(f"  grad {name:18s} norm {p.grad.norm():.3e}",
                          flush=True)
        torch.nn.utils.clip_grad_value_(g.model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            with torch.no_grad():
                lv, pv = run_loss()
            print(f"step {step:03d} loss {lv:.6f} "
                  f"pred |mean| {pv.abs().mean():.4f} "
                  f"pred rms {pv.pow(2).mean().sqrt():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    g.model.train()
    with torch.no_grad():
        lt, pt = run_loss()
    print(f"train-mode loss {lt:.6f} pred rms {pt.pow(2).mean().sqrt():.4f}")
    g.model.eval()
    with torch.no_grad():
        le, pe = run_loss()
    print(f"eval-mode  loss {le:.6f} pred rms {pe.pow(2).mean().sqrt():.4f}")

    seen = {}

    def hook(m, inp, out):
        x = inp[0].detach()
        seen[id(m)] = (x.var(0).mean().item(),
                       x.mean(0).abs().mean().item())

    handles = [m.register_forward_hook(hook) for m in g.model.modules()
               if isinstance(m, torch.nn.BatchNorm1d)]
    g.model.train()
    with torch.no_grad():
        run_loss()
    for h in handles:
        h.remove()
    for m in g.model.modules():
        if isinstance(m, torch.nn.BatchNorm1d) and id(m) in seen:
            bv, bm = seen[id(m)]
            print(f"  BN({m.num_features:3d}) batch var {bv:.4e} "
                  f"run var {m.running_var.mean():.4e} "
                  f"batch |mean| {bm:.4e} run mean {m.running_mean.abs().mean():.4e} "
                  f"n {int(m.num_batches_tracked)}")


if __name__ == "__main__":
    main()
