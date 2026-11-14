"""GEARS baseline trained on source lines under the canonical PDS harness.

The benchmark stores pseudobulk LFCs, not single-cell expression, and the
GEARS data pipeline expects AnnData with condition/cell_type/gene_name plus
per-cell graphs.  This script rebuilds the pieces GEARS needs without its
download-dependent PertData constructor:

  * gene axis = the tier's PDS evaluation positions;
  * one graph per (source line, target) row, x = 0 (so the model predicts
    the response directly), y = the row, L2-normalized to match cosine
    scoring (toggle with --no-unit);
  * perturbation graph = top-20 Jaccard neighbours among target genes over
    MSigDB Hallmark, KEGG, Reactome and ChEA gene sets (stand-ins for GO);
  * co-expression graph = top-20 absolute correlations among the 2000 most
    variable evaluation genes, computed over the source rows of the fold;
  * MSE + direction loss exactly as in gears.utils.loss_fct, early stopping
    on a held-out subset of source-only perturbations.

For each leave-one-line-out fold the model never sees the held-out line.
Predictions are scored with bench_common.Retrieval and paired against the
equal-mix PDS recorded in the tier's benchmark npz.

Usage:
  python scripts/gears_baseline.py --tier A --epochs 15
  python scripts/gears_baseline.py --tier A --fold k562 --max-train 300 \
      --max-targets 80 --epochs 2          # smoke

Outputs:
  outputs/gears_baseline_<tier>.npz
  outputs/gears_baseline_<tier>.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
from bench_common import (Retrieval, load_axis, load_lines, load_nadig_lines,
                          unit_rows)

GEARS_PATH = os.environ.get("GEARS_PATH", "external/GEARS")
sys.path.insert(0, GEARS_PATH)
from gears.gears import GEARS  # noqa: E402
from gears.utils import (create_cell_graph_for_prediction,  # noqa: E402
                         loss_fct)
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

OUT = "outputs"
GS_FILES = ["data/gs_MSigDB_Hallmark_2020.txt", "data/gs_KEGG_2021_Human.txt",
            "data/gs_Reactome_2022.txt",
            "data/gs_ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X.txt"]
HOLDS_B = ["h1", "k562", "rpe1", "hct116", "hek293t"]
HOLDS_A = ["k562", "rpe1", "hepg2", "jurkat"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["A", "B"], default="A")
    p.add_argument("--fold", default="", help="comma list; default all folds")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--direction-lambda", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-targets", type=int, default=0,
                   help="smoke: cap query targets per fold")
    p.add_argument("--max-train-perts", type=int, default=0,
                   help="smoke: cap training perturbations")
    p.add_argument("--agg", choices=["rows", "mean"], default="mean",
                   help="per-line training rows, or one source-mean row per "
                        "perturbation (Bayes target without a line covariate)")
    p.add_argument("--center", action="store_true",
                   help="subtract the source-mean response before training "
                        "and add it back at prediction time")
    p.add_argument("--val-frac", type=float, default=0.08)
    p.add_argument("--val-mode", choices=["perts", "rows"], default="perts",
                   help="validate on held-out perturbations, or on one "
                        "held-out source row per seen perturbation (matches "
                        "the matched-target transfer task)")
    p.add_argument("--coexpr-genes", type=int, default=2000)
    p.add_argument("--coexpr-rows", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-unit", action="store_true")
    p.add_argument("--loss", choices=["quartic", "mse"], default="mse",
                   help="official loss_fct quartic, or MSE on delta targets")
    p.add_argument("--norm", choices=["bn", "ln", "none"], default="bn",
                   help="keep batch norm, swap to layer norm, or drop norm")
    p.add_argument("--bn-cum", action="store_true",
                   help="accumulate BatchNorm statistics cumulatively "
                        "(momentum=None) instead of an EMA")
    p.add_argument("--bn-recalib", action="store_true",
                   help="re-estimate BatchNorm statistics on the training "
                        "set before each validation and at inference")
    p.add_argument("--diag-train", action="store_true")
    p.add_argument("--pert-lr-mult", type=float, default=1.0)
    p.add_argument("--out", default="")
    return p.parse_args()


def tier_data(tier):
    genes, pos, gidx, m = load_axis()
    if tier == "B":
        lines = load_lines(genes, gidx, m)
        lines.pop("_gidx", None)
        holds = [h for h in HOLDS_B if h in lines]
        return lines, holds, pos, gidx, genes
    lines = load_nadig_lines(genes, gidx)
    z = np.load(f"{OUT}/nadig_rows.npz", allow_pickle=True)
    in_g = np.zeros(len(genes), dtype=bool)
    in_g[z["gpos"]] = True
    return lines, [h for h in HOLDS_A if h in lines], pos[in_g[pos]], gidx, \
        genes


def load_gene_sets(paths):
    sets = []
    for path in paths:
        for line in open(path, encoding="utf-8"):
            parts = [x for x in line.rstrip("\n").split("\t") if x]
            if len(parts) < 3:
                continue
            sets.append(set(parts[1:]))
    return sets


def jaccard_edges(perts, sets, k=20):
    """Top-k Jaccard neighbours among perturbation genes."""
    pidx = {g: i for i, g in enumerate(perts)}
    members = []
    for s in sets:
        idx = np.array([pidx[g] for g in s if g in pidx], dtype=np.int64)
        if len(idx) >= 2:
            members.append(idx)
    n = len(perts)
    deg = np.zeros(n, dtype=np.int64)
    sets_of = [[] for _ in range(n)]
    for si, idx in enumerate(members):
        deg[idx] += 1
        for j in idx:
            sets_of[j].append(si)
    src, dst, wt = [], [], []
    for i in range(n):
        if not sets_of[i]:
            continue
        cnt = np.bincount(np.concatenate([members[s] for s in sets_of[i]]),
                          minlength=n)
        jac = cnt / np.maximum(deg[i] + deg - cnt, 1)
        jac[i] = 0.0
        jac[cnt < 2] = 0.0
        if jac.max() <= 0:
            continue
        kk = min(k, int((jac > 0).sum()))
        top = np.argpartition(-jac, kk - 1)[:kk]
        src.extend([i] * kk)
        dst.extend(top.tolist())
        wt.extend(jac[top].tolist())
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), \
        np.array(wt, dtype=np.float32)


def coexpr_edges(R, n_genes, k=20, threshold=0.4, n_hvg=2000, n_rows=2500,
                 seed=0):
    """Top-k |Pearson| neighbours among the most variable genes."""
    rng = np.random.default_rng(seed)
    if len(R) > n_rows:
        R = R[rng.choice(len(R), n_rows, replace=False)]
    var = R.var(0)
    hv = np.argsort(-var)[:min(n_hvg, n_genes)]
    hv = hv[var[hv] > 1e-12]
    if len(hv) < 2:
        e = np.arange(n_genes)
        return e, e, np.ones(n_genes, dtype=np.float32)
    Z = R[:, hv].astype(np.float64)
    Z -= Z.mean(0)
    sd = Z.std(0)
    keep = sd > 1e-12
    hv = hv[keep]
    Z = Z[:, keep] / sd[keep]
    C = np.abs(Z.T @ Z / max(len(Z), 1))
    np.fill_diagonal(C, 0.0)
    kk = min(k, C.shape[0] - 1)
    idx = np.argpartition(-C, kk - 1, axis=1)[:, :kk]
    src, dst, wt = [], [], []
    for i in range(C.shape[0]):
        for j in idx[i]:
            if C[i, j] > threshold:
                src.append(int(hv[i]))
                dst.append(int(hv[j]))
                wt.append(float(C[i, j]))
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), \
        np.array(wt, dtype=np.float32)


class PD:
    pass


def build_fold(lines, hold, pos, gidx, gene_names, perts, args):
    G = len(pos)
    srcs = [o for o in lines if o != hold]
    ref = lines[hold]
    q = [t for t in ref["names"] if any(t in lines[o]["index"] for o in srcs)]
    if args.max_targets:
        q = q[:args.max_targets]
    qset = set(q)
    pert_index = {p: i for i, p in enumerate(perts)}
    have = set(perts)

    rows, row_t, row_line = [], [], []
    by_target = {}
    for o in srcs:
        L = lines[o]
        for t, j in L["index"].items():
            if t not in have:
                continue
            v = L["rows"][j][pos].astype(np.float32)
            if not np.isfinite(v).all() or np.abs(v).sum() == 0:
                continue
            rows.append(v)
            row_t.append(t)
            row_line.append(o)
            by_target.setdefault(t, []).append(len(rows) - 1)
    if not rows:
        return None
    R = np.stack(rows)
    if not args.no_unit:
        R = unit_rows(R.astype(np.float64)).astype(np.float32)

    val_truth_rows, val_names = None, []
    R_orig = R
    if args.agg == "mean":
        targets = sorted(by_target)
        if args.val_mode == "rows" and args.val_frac > 0:
            tr_rows, tr_t, val_rows_orig = [], [], []
            for t in targets:
                js = by_target[t]
                if len(js) >= 2:
                    tr_rows.extend(js[:-1])
                    tr_t.extend([t] * (len(js) - 1))
                    val_names.append(t)
                    val_rows_orig.append(js[-1])
                else:
                    tr_rows.extend(js)
                    tr_t.append(t)
            groups = {}
            for i, t in zip(tr_rows, tr_t):
                groups.setdefault(t, []).append(i)
            tr_targets = sorted(groups)
            R = np.stack([R[groups[t]].mean(0) for t in tr_targets])
            by_target = {t: [i] for i, t in enumerate(tr_targets)}
            row_t = tr_targets
            row_line = ["mix"] * len(tr_targets)
            val_truth_rows = R_orig[val_rows_orig].astype(np.float64)
        else:
            R = np.stack([R[by_target[t]].mean(0) for t in targets])
            by_target = {t: [i] for i, t in enumerate(targets)}
            row_t = targets
            row_line = ["mix"] * len(targets)

    mu = R.mean(0)
    if args.center:
        R = R - mu
        if val_truth_rows is not None:
            val_truth_rows = val_truth_rows - mu

    targets = sorted(by_target)
    if args.max_train_perts and len(targets) > args.max_train_perts:
        rng0 = np.random.default_rng(args.seed + 17)
        sel = np.sort(rng0.choice(len(targets), args.max_train_perts,
                                  replace=False))
        keep_t = [targets[i] for i in sel]
        keep_rows = [i for t in keep_t for i in by_target[t]]
        remap = {old: new for new, old in enumerate(keep_rows)}
        R = R[keep_rows]
        by_target = {t: [remap[i] for i in by_target[t]] for t in keep_t}
        row_t = [row_t[i] for i in keep_rows]
        row_line = [row_line[i] for i in keep_rows]
        targets = keep_t
    rng = np.random.default_rng(args.seed)
    if val_truth_rows is not None:
        train_t = sorted(by_target)
        train_idx = [by_target[t][0] for t in train_t]
        val_idx = []
        val_t = set(val_names)
    else:
        non_q = [t for t in targets if t not in qset]
        rng.shuffle(non_q)
        n_val = max(1, int(round(args.val_frac * len(non_q))))
        val_t = set(non_q[:n_val])
        train_t = [t for t in targets if t not in val_t]
        keep = [by_target[t][0] for t in train_t]
        extra = [i for t in train_t for i in by_target[t][1:]]
        extra = list(rng.permutation(extra))
        if args.max_train:
            extra = extra[:max(0, args.max_train - len(keep))]
        train_idx = keep + extra
        val_idx = [i for t in sorted(val_t) for i in by_target[t]]
    n_val_rows = len(val_truth_rows) if val_truth_rows is not None \
        else len(val_idx)
    print(f"  fold {hold}: queries {len(q)}, train rows {len(train_idx)} "
          f"({len(train_t)} targets), val rows {n_val_rows} "
          f"({len(val_t)} targets), gene axis {G}", flush=True)

    mean_resp = {t: R[by_target[t]].mean(0) for t in targets}
    de_idx = {t: np.argsort(-np.abs(mean_resp[t]))[:20].astype(np.int64)
              for t in targets}

    obs = {"condition": np.array(["ctrl"] + [row_t[i] for i in train_idx]),
           "condition_name": np.array(["ctrl"] + [row_t[i]
                                                   for i in train_idx]),
           "cell_type": np.array(["ctrl"] + [row_line[i]
                                             for i in train_idx])}
    import anndata as ad
    adata = ad.AnnData(X=np.zeros((len(train_idx) + 1, G), dtype=np.float32),
                       obs=pd.DataFrame(obs),
                       var=pd.DataFrame(index=[str(g) for g in gene_names]))
    adata.var["gene_name"] = adata.var.index.values
    adata.uns["non_zeros_gene_idx"] = {
        t: np.where(np.abs(mean_resp[t]) > 1e-8)[0] for t in targets}
    adata.uns["non_zeros_pert"] = {t: True for t in targets}

    # PyG >= 2.5 flattens 1-D y on collation; keep y as (1, G) so a batch
    # is (B, G) as gears.utils.loss_fct expects.
    x0 = torch.zeros(G)
    train_graphs = [Data(x=x0, pert_idx=[-1], y=torch.zeros(1, G),
                         de_idx=np.array([-1] * 20), pert="ctrl")]
    for i in train_idx:
        train_graphs.append(Data(
            x=x0, pert_idx=[pert_index[row_t[i]]],
            y=torch.from_numpy(R[i].copy()).unsqueeze(0),
            de_idx=de_idx[row_t[i]], pert=row_t[i]))
    if val_truth_rows is not None:
        keep_v = [j for j, t in enumerate(val_names) if t in by_target]
        val_names = [val_names[j] for j in keep_v]
        val_truth_rows = val_truth_rows[keep_v]
        val_t = set(val_names)
        val_graphs = [Data(x=x0, pert_idx=[pert_index[t]],
                           y=torch.from_numpy(
                               val_truth_rows[j].astype(np.float32)
                           ).unsqueeze(0),
                           de_idx=de_idx[t], pert=t)
                      for j, t in enumerate(val_names)]
        val_truth_out = val_truth_rows.astype(np.float64)
    else:
        val_names = [t for t in sorted(val_t) for _ in by_target[t]]
        val_rows = [i for t in sorted(val_t) for i in by_target[t]]
        val_graphs = [Data(x=x0, pert_idx=[pert_index[t]],
                           y=torch.from_numpy(R[i].copy()).unsqueeze(0),
                           de_idx=de_idx[t], pert=t)
                      for t, i in zip(val_names, val_rows)]
        val_truth_out = R[val_rows].astype(np.float64) if val_rows else None
    g2pos = {int(g): j for j, g in enumerate(pos)}
    val_coords = [g2pos.get(int(gidx[t]), -1) if t in gidx else -1
                  for t in val_names]

    pd_obj = PD()
    pd_obj.adata = adata
    pd_obj.gene_names = pd.Series([str(g) for g in gene_names])
    pd_obj.pert_names = np.array(perts)
    pd_obj.node_map = {g: i for i, g in enumerate(pd_obj.gene_names)}
    pd_obj.node_map_pert = pert_index
    pd_obj.data_path = ""
    pd_obj.dataset_name = "pds"
    pd_obj.split = "no_test"
    pd_obj.seed = args.seed
    pd_obj.train_gene_set_size = 0.9
    pd_obj.set2conditions = {"train": ["ctrl"] + train_t, "val": sorted(val_t)}
    pd_obj.subgroup = None
    pd_obj.default_pert_graph = False
    pd_obj.dataloader = {
        "train_loader": DataLoader(train_graphs, batch_size=args.batch,
                                   shuffle=True),
        "val_loader": DataLoader(val_graphs, batch_size=args.batch,
                                 shuffle=False)}
    return dict(pd=pd_obj, q=q, ref=ref, R=R, train_idx=train_idx, G=G,
                mu=mu, val_truth=val_truth_out, val_coords=val_coords)


def swap_norm(module, kind="ln"):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm1d):
            if kind == "ln":
                setattr(module, name, torch.nn.LayerNorm(child.num_features))
            else:
                setattr(module, name, torch.nn.Identity())
        else:
            swap_norm(child, kind)


def make_model(pd_obj, go_edges, co_edges, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    src, dst, wt = co_edges
    g = GEARS(pd_obj, device=device)
    g.model_initialize(
        hidden_size=args.hidden, num_go_gnn_layers=1, num_gene_gnn_layers=1,
        decoder_hidden_size=16, num_similar_genes_go_graph=20,
        num_similar_genes_co_express_graph=20, coexpress_threshold=0.4,
        uncertainty=False, uncertainty_reg=1,
        direction_lambda=args.direction_lambda,
        G_go=torch.tensor(np.stack(go_edges[:2]), dtype=torch.long),
        G_go_weight=torch.tensor(go_edges[2], dtype=torch.float32),
        G_coexpress=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        G_coexpress_weight=torch.tensor(wt, dtype=torch.float32))
    if args.norm != "bn":
        swap_norm(g.model, args.norm)
    if args.bn_cum:
        for m in g.model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.momentum = None
    return g


def val_pds(P, Y, coords):
    P = np.asarray(P, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).copy()
    out = []
    K = len(Y)
    for i in range(len(P)):
        c = coords[i]
        p = P[i].copy()
        if c >= 0:
            p[c] = 0.0
        p /= max(np.linalg.norm(p), 1e-9)
        s = Y @ p
        st = s[i]
        rank = np.sum(s > st) + 0.5 * (np.sum(s == st) - 1)
        out.append(1.0 - rank / K)
    return float(np.mean(out))


def pds_plain(P, Y):
    P = np.asarray(P, dtype=np.float64)
    Yn = np.asarray(Y, dtype=np.float64)
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    Yn = Yn / np.maximum(np.linalg.norm(Yn, axis=1, keepdims=True), 1e-9)
    S = Yn @ Pn.T
    n = len(Yn)
    r = np.sum(S > np.diag(S)[None, :], axis=0)
    e = np.sum(S == np.diag(S)[None, :], axis=0)
    return float(np.mean(1.0 - (r + 0.5 * (e - 1)) / n))


def diag_train_pds(g, device, k=256):
    ds = g.dataloader["train_loader"].dataset
    idx = list(range(1, len(ds)))[::max(1, (len(ds) - 1) // k)][:k]
    dl = DataLoader([ds[i] for i in idx], batch_size=64, shuffle=False)
    model = g.model

    def pass_pds():
        P, Y = [], []
        with torch.no_grad():
            for b in dl:
                b = b.to(device)
                P.append(model(b).cpu().numpy())
                Y.append(b.y.cpu().numpy())
        return pds_plain(np.concatenate(P), np.concatenate(Y))

    model.eval()
    pe = pass_pds()
    bns = [m for m in model.modules()
           if isinstance(m, torch.nn.BatchNorm1d)]
    saved = [(m.running_mean.clone(), m.running_var.clone(),
              m.num_batches_tracked.clone()) for m in bns]
    model.train()
    pt = pass_pds()
    for m, (rm, rv, nb) in zip(bns, saved):
        m.running_mean.copy_(rm)
        m.running_var.copy_(rv)
        m.num_batches_tracked.copy_(nb)
    return pe, pt


def bn_recalibrate(model, loader, device):
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    if not bns:
        return
    moms = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    with torch.no_grad():
        for b in loader:
            model(b.to(device))
    for m, mom in zip(bns, moms):
        m.momentum = mom


def train_loop(g, args, device, val_truth=None, val_coords=None):
    pert = [g.model.pert_emb.weight]
    rest = [p for n, p in g.model.named_parameters()
            if n != "pert_emb.weight"]
    opt = torch.optim.Adam(
        [{"params": rest, "lr": args.lr},
         {"params": pert, "lr": args.lr * args.pert_lr_mult}],
        weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.7)
    best, best_state, best_ep = -float("inf"), None, -1
    Yv = np.asarray(val_truth, dtype=np.float64) if val_truth is not None \
        else None
    if Yv is not None:
        for j, c in enumerate(val_coords):
            if c >= 0:
                Yv[j, c] = 0.0
        Yv /= np.maximum(np.linalg.norm(Yv, axis=1, keepdims=True), 1e-9)
    tl = g.dataloader["train_loader"]
    vl = g.dataloader["val_loader"]
    for ep in range(args.epochs):
        t0 = time.time()
        g.model.train()
        tr_loss, nb = 0.0, 0
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            pred = g.model(batch)
            if args.loss == "mse":
                loss = ((pred - batch.y) ** 2).mean()
            else:
                loss = loss_fct(pred, batch.y, batch.pert,
                                ctrl=g.ctrl_expression,
                                direction_lambda=args.direction_lambda,
                                dict_filter=g.dict_filter)
            loss.backward()
            torch.nn.utils.clip_grad_value_(g.model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss)
            nb += 1
        sched.step()
        if args.bn_recalib:
            bn_recalibrate(g.model, tl, device)
        g.model.eval()
        vse, vpred = [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                p = g.model(batch)
                vse.append(float(((p - batch.y) ** 2).mean()))
                vpred.append(p.cpu().numpy())
        vm = float(np.mean(vse)) if vse else 0.0
        vp = val_pds(np.concatenate(vpred), Yv, val_coords) \
            if Yv is not None and vpred else float("nan")
        vpt = float("nan")
        if args.diag_train and Yv is not None:
            bns = [m for m in g.model.modules()
                   if isinstance(m, torch.nn.BatchNorm1d)]
            saved = [(m.running_mean.clone(), m.running_var.clone(),
                      m.num_batches_tracked.clone()) for m in bns]
            g.model.train()
            tpred = []
            with torch.no_grad():
                for batch in vl:
                    batch = batch.to(device)
                    tpred.append(g.model(batch).cpu().numpy())
            for m, (rm, rv, nb) in zip(bns, saved):
                m.running_mean.copy_(rm)
                m.running_var.copy_(rv)
                m.num_batches_tracked.copy_(nb)
            vpt = val_pds(np.concatenate(tpred), Yv, val_coords)
        improved = vp > best
        if improved:
            best, best_ep = vp, ep
            best_state = {k: t.detach().clone()
                          for k, t in g.model.state_dict().items()}
        if args.diag_train:
            tpe, tpt = diag_train_pds(g, device)
        else:
            tpe = tpt = float("nan")
        print(f"    epoch {ep + 1:02d} loss {tr_loss / max(nb, 1):.4f} "
              f"val_mse {vm:.5f} val_pds {vp:.4f}/{vpt:.4f} "
              f"train_pds {tpe:.4f}/{tpt:.4f}"
              f"{' *' if improved else ''} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if args.patience and ep - best_ep >= args.patience:
            print(f"    early stop at epoch {ep + 1}", flush=True)
            break
    if best_state is not None:
        g.model.load_state_dict(best_state)
    return best_ep


def predict(g, q, perts, G, device, batch=64, mu=None):
    g.model.eval()
    pi = {p: i for i, p in enumerate(perts)}
    out = {}
    for s in range(0, len(q), batch):
        chunk = q[s:s + batch]
        graphs = [create_cell_graph_for_prediction(
            np.zeros(G, dtype=np.float32), [pi[t]], [t]) for t in chunk]
        dl = DataLoader(graphs, batch_size=len(chunk), shuffle=False)
        bt = next(iter(dl)).to(device)
        with torch.no_grad():
            p = g.model(bt).cpu().numpy()
        if mu is not None:
            p = p + mu
        for t, row in zip(chunk, p):
            out[t] = row
    return out


def equal_pds(tier):
    path = (f"{OUT}/bench_transfer_v2.npz" if tier == "B"
            else f"{OUT}/bench_transfer_nadig_v2.npz")
    z = np.load(path, allow_pickle=True)
    ho, me, tg = (z[k].astype(str) for k in ("holdout", "method", "target"))
    p = z["pds_raw"].astype(np.float64)
    keep = me == "equal"
    return {(h, t): v for h, t, v in zip(ho[keep], tg[keep], p[keep])}


def main():
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device {device}", flush=True)
    torch.set_num_threads(max(1, args.threads))

    lines, holds, pos, gidx, genes = tier_data(args.tier)
    if args.fold:
        want = set(args.fold.split(","))
        holds = [h for h in holds if h in want]
    gene_names = np.array(genes)[pos]
    perts = sorted({t for L in lines.values() for t in L["names"]})
    print(f"tier {args.tier}: {len(holds)} folds, {len(pos)} genes, "
          f"{len(perts)} perturbations", flush=True)

    sets = load_gene_sets(GS_FILES)
    print(f"gene sets {len(sets)}", flush=True)
    t0 = time.time()
    go_edges = jaccard_edges(perts, sets, k=20)
    print(f"GO graph {len(go_edges[0])} edges ({time.time() - t0:.0f}s)",
          flush=True)

    eq = equal_pds(args.tier)
    store, summ = {}, []
    for hold in holds:
        t_fold = time.time()
        built = build_fold(lines, hold, pos, gidx, gene_names, perts, args)
        if built is None:
            continue
        pd_obj = built["pd"]
        ce = coexpr_edges(built["R"][built["train_idx"]], built["G"],
                          n_hvg=args.coexpr_genes, n_rows=args.coexpr_rows,
                          seed=args.seed)
        print(f"  coexpr graph {len(ce[0])} edges", flush=True)
        g = make_model(pd_obj, go_edges, ce, args, device, args.seed)
        best_ep = train_loop(g, args, device, built["val_truth"],
                             built["val_coords"])
        preds = predict(g, built["q"], perts, built["G"], device,
                        mu=built["mu"] if args.center else None)
        q = [t for t in built["q"] if t in preds]
        Psub = np.stack([preds[t] for t in q])
        P = np.zeros((len(q), len(genes)), dtype=np.float64)
        P[:, pos] = Psub
        ref = built["ref"]
        ret = Retrieval(ref["rows"], ref["names"], pos, gidx=gidx)
        pds = ret.score(P, q, exclude="self", denominator="n")
        eqv = np.array([eq.get((hold, t), np.nan) for t in q])
        ok = np.isfinite(eqv)
        d = eqv[ok] - pds[ok]
        store[f"{hold}|targets"] = np.array(q)
        store[f"{hold}|pred"] = Psub.astype(np.float32)
        store[f"{hold}|pds"] = pds
        summ.append(dict(tier=args.tier, holdout=hold, n=len(q),
                         best_epoch=best_ep,
                         pds_med=float(np.nanmedian(pds)),
                         pds_mean=float(np.nanmean(pds)),
                         equal_med=float(np.nanmedian(eqv)),
                         equal_mean=float(np.nanmean(eqv)),
                         d_med=float(np.nanmedian(d)),
                         d_mean=float(np.nanmean(d)),
                         win=float(np.mean(d > 0))))
        print(f"  === {hold}: n={len(q)} gears {np.nanmedian(pds):.4f} "
              f"({np.nanmean(pds):.4f}) vs equal "
              f"{np.nanmedian(eqv):.4f} ({np.nanmean(eqv):.4f}) "
              f"d_med {np.nanmedian(d):+.4f} win {np.mean(d > 0):.2f} "
              f"[{time.time() - t_fold:.0f}s]", flush=True)
        np.savez(f"{OUT}/gears_baseline_{args.tier}.npz", **store)
        pd.DataFrame(summ).to_csv(f"{OUT}/gears_baseline_{args.tier}.csv",
                                  index=False)
    print(pd.DataFrame(summ).to_string(index=False))


if __name__ == "__main__":
    main()
