"""One-batch diagnostic for the GEARS adaptation: output scale and gradients.

Builds the k562 fold, runs one training batch forward/backward, and prints
prediction and gradient norms by module.  Used to diagnose why the quartic
loss on log-fold-change targets fails to move the model.

Usage: python scripts/gears_diag.py [--no-unit]
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

    t0 = time.time()
    lines, holds, pos, gidx, genes = gb.tier_data(args.tier)
    gene_names = np.array(genes)[pos]
    perts = sorted({t for L in lines.values() for t in L["names"]})
    print(f"data {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    sets = gb.load_gene_sets(gb.GS_FILES)
    go_edges = gb.jaccard_edges(perts, sets, k=20)
    print(f"go {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    built = gb.build_fold(lines, "k562", pos, gidx, gene_names, perts, args)
    print(f"fold {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    ce = gb.coexpr_edges(built["R"][built["train_idx"]], built["G"],
                         n_hvg=args.coexpr_genes, n_rows=args.coexpr_rows,
                         seed=args.seed)
    print(f"coexpr {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    g = gb.make_model(built["pd"], go_edges, ce, args, device, args.seed)
    print(f"model {time.time() - t0:.0f}s", flush=True)

    g.model.train()
    batch = next(iter(g.dataloader["train_loader"])).to(device)
    print("batch", batch.y.shape, "pert", list(batch.pert)[:6], flush=True)
    t0 = time.time()
    pred = g.model(batch)
    loss = gb.loss_fct(pred, batch.y, batch.pert, ctrl=g.ctrl_expression,
                       direction_lambda=args.direction_lambda,
                       dict_filter=g.dict_filter)
    loss.backward()
    print(f"fwd/bwd {time.time() - t0:.1f}s", flush=True)

    y = batch.y.detach().numpy()
    p = pred.detach().numpy()
    print(f"loss {loss.item():.5f}")
    print(f"y    |mean {np.abs(y).mean():.4f} norm {np.linalg.norm(y, axis=1).mean():.4f}")
    print(f"pred |mean {np.abs(p).mean():.4f} norm {np.linalg.norm(p, axis=1).mean():.4f}")

    def gnorm(name, param):
        if param.grad is None:
            print(f"  {name:22s} grad None")
            return
        gv = param.grad
        print(f"  {name:22s} norm {gv.norm():.3e} max {gv.abs().max():.3e}")

    for name, param in g.model.named_parameters():
        if name in ("indv_b1", "indv_b2", "indv_w1", "indv_w2",
                    "gene_emb.weight", "pert_emb.weight"):
            gnorm(name, param)
    gnorm("recovery_w.0.weight",
          g.model.recovery_w.network[0].weight)
    gnorm("cross_gene_state.0.weight",
          g.model.cross_gene_state.network[0].weight)


if __name__ == "__main__":
    main()
