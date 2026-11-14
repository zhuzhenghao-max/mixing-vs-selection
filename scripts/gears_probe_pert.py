"""Check that perturbation identity changes GEARS predictions.

Builds a tiny fold, inspects how PyG collates pert_idx, and compares
predictions for two different perturbations before and after training.
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
import gears_baseline as gb


def main():
    args = gb.parse_args()
    args.tier, args.fold = "A", "k562"
    args.max_train_perts, args.max_targets = 80, 10
    args.epochs, args.batch = 2, 32
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

    batch = next(iter(g.dataloader["train_loader"])).to(device)
    print("pert_idx", type(batch.pert_idx), len(batch.pert_idx),
          batch.pert_idx[:6])

    def diff_two(p0, p1):
        graphs = []
        for p in (p0, p1):
            graphs.append(gb.Data(x=torch.zeros(built["G"]),
                                  pert_idx=[p], y=torch.zeros(1, built["G"]),
                                  de_idx=np.array([-1] * 20), pert="x"))
        bt = next(iter(gb.DataLoader(graphs, batch_size=2)))
        out = []
        for mode in ("eval", "train"):
            getattr(g.model, mode)()
            with torch.no_grad():
                o = g.model(bt).numpy()
            out.append((o[0], o[1], np.linalg.norm(o[0] - o[1])))
        return out

    pi = {p: i for i, p in enumerate(perts)}
    a, b = pi[perts[10]], pi[perts[20]]
    for tag in ("untrained",):
        for mode, (o0, o1, d) in zip(("eval", "train"), diff_two(a, b)):
            print(f"{tag} {mode}: |out0| {np.linalg.norm(o0):.4e} "
                  f"|out1| {np.linalg.norm(o1):.4e} |diff| {d:.4e}")

    gb.train_loop(g, args, device, built["val_truth"], built["val_coords"])
    for mode, (o0, o1, d) in zip(("eval", "train"), diff_two(a, b)):
        print(f"trained {mode}: |out0| {np.linalg.norm(o0):.4e} "
              f"|out1| {np.linalg.norm(o1):.4e} |diff| {d:.4e}")

    ds = g.dataloader["train_loader"].dataset
    idx = list(range(1, len(ds)))[:64]
    dl = gb.DataLoader([ds[i] for i in idx], batch_size=64, shuffle=False)
    bt = next(iter(dl))
    for mode in ("eval", "train"):
        getattr(g.model, mode)()
        with torch.no_grad():
            P = g.model(bt).numpy()
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        C = Pn @ Pn.T
        iu = np.triu_indices(len(P), 1)
        print(f"{mode}: pred pair cos mean {C[iu].mean():.4f} "
              f"row std {P.std(0).mean():.4e} "
              f"pds {gb.pds_plain(P, bt.y.numpy()):.4f}")


if __name__ == "__main__":
    main()
