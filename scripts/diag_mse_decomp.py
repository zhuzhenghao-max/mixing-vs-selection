"""Diagnose the mse gap: compare oracle emission vs D2 emission vs real bulk-lognorm profiles.

mu = log1p(BULK_TS * pseudobulk_mean_g / sum_g pseudobulk_mean_g), BULK_TS = 5e4.
"""
import sys
import numpy as np
import anndata as ad

BULK_TS = 50000.0
PERT_COL = "target_gene"
CTRL = "non-targeting"

def bulk_mu(path, want_ctrl=False):
    a = ad.read_h5ad(path)
    labels = a.obs[PERT_COL].to_numpy().astype(str)
    genes = a.var_names.to_numpy().astype(str)
    X = a.X
    mu = {}
    for lab in np.unique(labels):
        if lab == CTRL and not want_ctrl:
            continue
        m = labels == lab
        n = int(m.sum())
        s = np.asarray(X[m].sum(axis=0)).ravel().astype(np.float64) / n
        mu[lab] = np.log1p(BULK_TS * s / s.sum())
    return mu, genes

real, genes = bulk_mu("data/internal/real_heldout.h5ad")
orac, _ = bulk_mu("outputs/h1_holdout/baseline/baseline_pred.h5ad")
d2, _ = bulk_mu("outputs/pred_h1_lfcmap_const_s10_eval.h5ad")
print(f"real perts: {len(real)}, oracle: {len(orac)}, d2: {len(d2)}")

perts = sorted(real.keys())
print(f"\n{'pert':<14} {'|d(orc,real)|^2':>14} {'|d(d2,real)|^2':>13} {'vs panel-mean':>13}")
prof = np.mean([real[p] for p in perts], axis=0)  # ideal lognorm-space panel mean
sums = {"orc": [], "d2": [], "prof": []}
for p in perts:
    r = real[p]
    do = float(np.sum((orac[p] - r) ** 2))
    dd = float(np.sum((d2[p] - r) ** 2))
    dp = float(np.sum((prof - r) ** 2))
    sums["orc"].append(do); sums["d2"].append(dd); sums["prof"].append(dp)
    print(f"{p:<14} {do:>14.3f} {dd:>13.3f} {dp:>13.3f}")
for k, v in sums.items():
    print(f"MEAN {k}: {np.mean(v):.3f}")

# gene-level decomposition: where is D2 vs oracle
gap = np.mean([(d2[p] - real[p]) ** 2 - (orac[p] - real[p]) ** 2 for p in perts], axis=0)
print("\n-- top 20 genes with D2 worse than oracle (mean sq diff) --")
idx = np.argsort(-gap)[:20]
for i in idx:
    print(f"  {genes[i]:<12} gap {gap[i]:+.4f}  mu_real {np.mean([real[p][i] for p in perts]):.3f}")
print("-- top 20 genes with D2 better than oracle --")
idx = np.argsort(gap)[:20]
for i in idx:
    print(f"  {genes[i]:<12} gap {gap[i]:+.4f}  mu_real {np.mean([real[p][i] for p in perts]):.3f}")

# check: is the oracle == profile in its own space? compare orac[p] to mean of oracle over perts
orac_mean = np.mean([orac[p] for p in perts], axis=0)
od = [float(np.sum((orac[p] - orac_mean) ** 2)) for p in perts]
print(f"\noracle self-spread |orac_p - orac_mean|^2 mean: {np.mean(od):.4f}")
# and real spread around panel mean
rd = [float(np.sum((real[p] - prof) ** 2)) for p in perts]
print(f"real spread around panel mean: {np.mean(rd):.4f}")

# norms / scale sanity
print("\n-- norms --")
print(f"|prof| (lognorm panel mean): {np.linalg.norm(prof):.2f}")
print(f"|mu_real| mean: {np.mean([np.linalg.norm(real[p]) for p in perts]):.2f}")
print(f"|mu_oracle| mean: {np.mean([np.linalg.norm(orac[p]) for p in perts]):.2f}")
print(f"|mu_d2| mean: {np.mean([np.linalg.norm(d2[p]) for p in perts]):.2f}")

# cosine of perturbation deltas (subtract control) - direction check in lognorm space
ctrl, _ = bulk_mu("data/internal/real_heldout.h5ad", want_ctrl=True)
c = ctrl[CTRL]
print("\n-- delta (mu_p - mu_ctrl) cosine to real delta --")
cos = {"orc": [], "d2": [], "prof": []}
for p in perts:
    dr = real[p] - c
    for k, mat in (("orc", orac), ("d2", d2), ("prof", None)):
        dv = (prof - c) if mat is None else (mat[p] - c)
        cs = float(np.dot(dv, dr) / (np.linalg.norm(dv) * np.linalg.norm(dr) + 1e-12))
        cos[k].append(cs)
for k, v in cos.items():
    print(f"  {k}: {np.mean(v):.3f}")
print("DONE_MSE_DECOMP")
