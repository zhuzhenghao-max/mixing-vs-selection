"""Nonlinear supervised calibration baselines on the held-out line.

Two models, both fitted per outer fold on the held-out line's other
measured targets and scored with the canonical PDS:

  rff  PCA -> random Fourier features -> ridge readout of the residual
  mlp  PCA -> residual MLP with inner-validation early stopping

The outer folds, equal-mix features, targets, and scoring match
bench_supervised.py, so results are directly comparable with the
linear ridge and kNN baselines.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import pairwise_distances

sys.path.insert(0, "scripts")
from bench_common import (Retrieval, boot_ci, load_axis, load_lines,
                          load_nadig_lines)
from bench_supervised import (TIERS, build_features, build_targets,
                              load_equal_pds, paired_stats, zero_self)


def folds_of(n, n_folds, seed):
    return np.array_split(np.random.default_rng(seed).permutation(n), n_folds)


def median_gamma(Z, seed, cap=300):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(cap, len(Z)), replace=False)
    md = np.median(pairwise_distances(Z[idx], squared=True))
    return 1.0 / (2.0 * max(md, 1e-9))


def cv_rff(X, Y, folds, n_comp, n_rff, gamma_mult, whiten, seed):
    P = np.zeros_like(Y, dtype=np.float32)
    for fi, val in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(X)), val, assume_unique=False)
        tr = tr[np.linalg.norm(Y[tr], axis=1) > 1e-6]
        k = min(n_comp, len(tr) - 1, X.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=seed,
                  whiten=whiten)
        Ztr = pca.fit_transform(X[tr].astype(np.float64)).astype(np.float32)
        Zv = pca.transform(X[val].astype(np.float64)).astype(np.float32)
        gamma = gamma_mult * median_gamma(Ztr, seed + fi)
        rff = RBFSampler(gamma=gamma, n_components=n_rff,
                         random_state=seed + fi)
        Ftr = rff.fit_transform(Ztr)
        Fv = rff.transform(Zv)
        t0 = time.time()
        model = RidgeCV(alphas=np.logspace(-2, 4, 7))
        model.fit(Ftr, Y[tr] - X[tr])
        P[val] = (X[val] + model.predict(Fv)).astype(np.float32)
        print(f"  fold {fi}: comp {k} rff {n_rff} gamma {gamma:.3g} "
              f"alpha {model.alpha_:.3g} fit {time.time() - t0:.1f}s",
              flush=True)
    return P


class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden, dropout):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        out = nn.Linear(prev, d_out)
        nn.init.zeros_(out.bias)
        nn.init.normal_(out.weight, std=0.01)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def cv_mlp(X, Y, folds, n_comp, whiten, hidden, dropout, weight_decay,
           lr, batch, max_epochs, patience, seed):
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    P = np.zeros_like(Y, dtype=np.float32)
    d_out = Y.shape[1]
    for fi, val in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(X)), val, assume_unique=False)
        tr = tr[np.linalg.norm(Y[tr], axis=1) > 1e-6]
        k = min(n_comp, len(tr) - 1, X.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=seed,
                  whiten=whiten)
        Ztr = pca.fit_transform(X[tr].astype(np.float64)).astype(np.float32)
        Zv = pca.transform(X[val].astype(np.float64)).astype(np.float32)

        rng = np.random.default_rng(seed + fi)
        perm = rng.permutation(len(tr))
        n_in = max(32, int(0.1 * len(tr)))
        inner = perm[:n_in]
        fit = perm[n_in:]
        Xfit = torch.from_numpy(Ztr[fit])
        Tfit = torch.from_numpy((Y[tr] - X[tr])[fit])
        Xinn = torch.from_numpy(Ztr[inner])
        Tinn = torch.from_numpy((Y[tr] - X[tr])[inner])
        Xval = torch.from_numpy(Zv)

        torch.manual_seed(seed + fi)
        model = ResidualMLP(k, d_out, hidden, dropout)
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        lossf = nn.MSELoss()
        best = float("inf")
        best_state = {k_: v.detach().clone()
                      for k_, v in model.state_dict().items()}
        bad = 0
        t0 = time.time()
        for ep in range(1, max_epochs + 1):
            model.train()
            order = torch.randperm(len(fit))
            for start in range(0, len(fit), batch):
                idx = order[start:start + batch]
                opt.zero_grad()
                loss = lossf(model(Xfit[idx]), Tfit[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(lossf(model(Xinn), Tinn))
            if vl < best - 1e-7:
                best = vl
                best_state = {k_: v.detach().clone()
                              for k_, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if ep == 1 or ep % 10 == 0:
                print(f"    fold {fi} ep {ep:3d} inner {vl:.5f} "
                      f"best {best:.5f} ({time.time() - t0:.0f}s)", flush=True)
            if bad >= patience:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(Xval).numpy()
        P[val] = (X[val] + pred).astype(np.float32)
        print(f"  fold {fi}: comp {k} hidden {hidden} epochs {ep} "
              f"inner {best:.5f} ({time.time() - t0:.0f}s)", flush=True)
    return P


def run_tier(tier, model, n_folds, n_comp, seed, only, selfzero, shuffle,
             n_rff, gamma_mult, whiten, hidden, dropout, weight_decay, lr,
             batch, max_epochs, patience, tag):
    genes, pos, gidx, m = load_axis()
    lines = (load_lines(genes, gidx, m) if tier == "B"
             else load_nadig_lines(genes, gidx))
    lines.pop("_gidx", None)
    all_holdouts, eq_path = TIERS[tier]
    holdouts = ([h for h in all_holdouts if h in only] if only
                else list(all_holdouts))
    eq = load_equal_pds(eq_path)

    summary, flat = [], []
    print(f"\n######## Tier {tier} | {model} | {n_folds}-fold ########",
          flush=True)
    for L in holdouts:
        if L not in lines or len(lines[L]["names"]) == 0:
            continue
        ref = lines[L]
        srcs = [o for o in all_holdouts if o != L and o in lines
                and len(lines[o]["names"]) > 0]
        q = [t for t in ref["names"]
             if any(t in lines[o]["index"] for o in srcs)]
        print(f"\n=== hold-out {L}: pool {len(ref['names'])}, "
              f"queries {len(q)}, sources {srcs} ===", flush=True)
        if len(q) < 10:
            continue

        X = build_features(lines, srcs, q, pos, "mix")
        Y = build_targets(ref, q, pos)
        if selfzero:
            X, Y = zero_self(X, Y, q, gidx, pos)
        if shuffle:
            perm = np.random.default_rng(seed + 991).permutation(len(q))
            Y = Y[perm]
        print(f"  features {X.shape}, targets {Y.shape}", flush=True)

        folds = folds_of(len(q), n_folds, seed)
        if model == "rff":
            P = cv_rff(X, Y, folds, n_comp, n_rff, gamma_mult, whiten, seed)
        else:
            P = cv_mlp(X, Y, folds, n_comp, whiten, hidden, dropout,
                       weight_decay, lr, batch, max_epochs, patience, seed)
        del X, Y

        R = Retrieval(ref["rows"], ref["names"], pos, gidx=gidx)
        P_full = np.zeros((len(q), len(genes)), dtype=np.float32)
        P_full[:, pos] = P
        pds = R.score(P_full, q, exclude="self", denominator="n")
        del P, P_full

        method = f"{model}_mix"
        common = [t for t in q if (L, t) in eq and np.isfinite(eq[(L, t)])]
        ps = np.array([pds[q.index(t)] for t in common])
        pe = np.array([eq[(L, t)] for t in common])
        st = paired_stats(ps, pe)
        st.update(tier=tier, holdout=L, method=method, n=len(common),
                  mean=float(np.nanmean(ps)), median=float(np.nanmedian(ps)),
                  equal_mean=float(np.nanmean(pe)),
                  equal_median=float(np.nanmedian(pe)))
        summary.append(st)
        print(f"  {method:12s} n={len(common):4d} "
              f"mean {st['mean']:.4f} (equal {st['equal_mean']:.4f}) "
              f"median {st['median']:.4f} (equal {st['equal_median']:.4f})",
              flush=True)
        print(f"  delta vs equal {st['delta_mean']:+.4f} "
              f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}] "
              f"p={st['p']:.2g} win={st['win']:.2f}", flush=True)
        for t, p, e in zip(common, ps, pe):
            flat.append((L, method, t, p, e))

    prefix = ("outputs/bench_supervised_v2" if tier == "B"
              else "outputs/bench_supervised_nadig_v2")
    prefix += f"_nl_{model}"
    if tag:
        prefix += f"_{tag}"
    if selfzero:
        prefix += "_selfzero"
    if shuffle:
        prefix += "_yshuf"
    np.savez(f"{prefix}.npz",
             holdout=np.array([f[0] for f in flat]),
             method=np.array([f[1] for f in flat]),
             target=np.array([f[2] for f in flat]),
             pds_raw=np.array([f[3] for f in flat], dtype=np.float64),
             pds_equal=np.array([f[4] for f in flat], dtype=np.float64))
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nsaved {prefix}.csv / .npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B", "all"], default="A")
    ap.add_argument("--model", choices=["rff", "mlp"], default="rff")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pcs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--selfzero", type=int, choices=[0, 1], default=0)
    ap.add_argument("--shuffle", type=int, choices=[0, 1], default=0)
    ap.add_argument("--rff-features", type=int, default=2000)
    ap.add_argument("--gamma-mult", type=float, default=1.0)
    ap.add_argument("--whiten", type=int, choices=[0, 1], default=1)
    ap.add_argument("--hidden", default="512,256")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    only = set(args.holdout.split(",")) if args.holdout else None
    hidden = tuple(int(h) for h in args.hidden.split(","))
    tiers = ["B", "A"] if args.tier == "all" else [args.tier]
    for t in tiers:
        run_tier(t, args.model, args.folds, args.pcs, args.seed, only,
                 bool(args.selfzero), bool(args.shuffle), args.rff_features,
                 args.gamma_mult, bool(args.whiten), hidden, args.dropout,
                 args.weight_decay, args.lr, args.batch, args.max_epochs,
                 args.patience, args.tag)


if __name__ == "__main__":
    main()
