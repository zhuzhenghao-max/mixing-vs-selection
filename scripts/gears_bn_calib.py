"""Test BN running-stat recalibration for the GEARS adaptation.

Trains briefly, then compares eval-mode PDS with (a) the running stats
accumulated during training, (b) stats recalibrated by one cumulative
(momentum=None) pass over the training data, and (c) batch statistics.
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
import gears_baseline as gb


def pds_pass(model, dl, device, mode):
    model.train(mode == "train")
    P, Y = [], []
    with torch.no_grad():
        for b in dl:
            b = b.to(device)
            P.append(model(b).cpu().numpy())
            Y.append(b.y.cpu().numpy())
    return gb.pds_plain(np.concatenate(P), np.concatenate(Y))


def main():
    args = gb.parse_args()
    args.tier, args.fold = "A", "k562"
    args.max_train_perts, args.max_targets = 300, 40
    args.epochs, args.batch = 6, 64
    args.patience = 0
    torch.set_num_threads(args.threads)
    device = "cpu"

    lines, holds, pos, gidx, genes = gb.tier_data(args.tier)
    gene_names = np.array(genes)[pos]
    perts = sorted({t for L in lines.values() for t in L["names"]})
    sets = gb.load_gene_sets(gb.GS_FILES)
    go_edges = gb.jaccard_edges(perts, sets, k=20)
    built = gb.build_fold(lines, "k562", pos, gidx, gene_names, perts, args)
    ce = gb.coexpr_edges(built["R"][built["train_idx"]], built["G"],
                         n_hvg=args.coexpr_genes, n_rows=args.coexpr_rows,
                         seed=args.seed)
    g = gb.make_model(built["pd"], go_edges, ce, args, device, args.seed)
    gb.train_loop(g, args, device, built["val_truth"], built["val_coords"])

    ds = g.dataloader["train_loader"].dataset
    idx = list(range(1, len(ds)))[::max(1, (len(ds) - 1) // 256)][:256]
    dl = gb.DataLoader([ds[i] for i in idx], batch_size=128, shuffle=False)

    print("eval running   ", pds_pass(g.model, dl, device, "eval"), flush=True)
    print("train batch    ", pds_pass(g.model, dl, device, "train"), flush=True)

    bns = [m for m in g.model.modules()
           if isinstance(m, torch.nn.BatchNorm1d)]
    saved = [(m.momentum, m.num_batches_tracked.clone()) for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    full = gb.DataLoader(list(ds), batch_size=128, shuffle=False)
    g.model.train()
    with torch.no_grad():
        for b in full:
            g.model(b.to(device))
    print("eval calibrated", pds_pass(g.model, dl, device, "eval"), flush=True)
    for m, (mom, nb) in zip(bns, saved):
        m.momentum = mom
        m.num_batches_tracked.copy_(nb)


if __name__ == "__main__":
    main()
