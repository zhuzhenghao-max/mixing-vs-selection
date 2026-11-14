"""Tiny synthetic Perturb-seq dataset to smoke-test the eval pipeline.

Control cells draw from a Poisson per-gene base rate. Perturbed cells are
control-like except the target gene drops by 90% and a small set of
downstream genes shifts. Both h5ads go to the same dir so cell-eval2 can
score them as a (pred, real) pair.
"""

import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)

N_GENES = 2000
GENES = [f"g{i}" for i in range(N_GENES)]
PERTS = [f"g{i}" for i in range(10, 210, 10)]
CONTROL = "non-targeting"
N_CTRL = 600
N_PERT = 300
UMI = 4000

base_rate = rng.lognormal(mean=0.0, sigma=1.2, size=N_GENES)
base_rate /= base_rate.sum()

downstream = {p: rng.choice(N_GENES, size=40, replace=False) for p in PERTS}


def make_cells(target):
    rates = base_rate.copy()
    if target != CONTROL:
        i = int(target[1:])
        rates[i] *= rng.uniform(0.05, 0.4)
        # 每个扰动随机效应强度, 否则扰动间太同质, baseline 臂的方向指标会退化
        sigma = rng.uniform(0.2, 0.8)
        for j in downstream[target]:
            rates[j] *= float(np.exp(rng.normal(0, sigma)))
    rates /= rates.sum()
    n = N_CTRL if target == CONTROL else N_PERT
    X = rng.multinomial(UMI, rates, size=n).astype(np.float32)
    obs = pd.DataFrame({"target_gene": [target] * n, "context": ["A"] * n},
                       index=[f"{target}_{k}" for k in range(n)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES))


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    real = ad.concat([make_cells(CONTROL)] + [make_cells(p) for p in PERTS])
    real.write_h5ad(os.path.join(outdir, "real.h5ad"))

    # control paste: every perturbation gets the mean control profile
    ctrl = real[real.obs["target_gene"] == CONTROL].X.toarray()
    mean_ctrl = np.round(ctrl.mean(0)).astype(np.float32)
    pred = ad.concat([make_cells(CONTROL)] +
                     [ad.AnnData(X=np.tile(mean_ctrl, (N_PERT, 1)),
                                 obs=pd.DataFrame({"target_gene": [p] * N_PERT,
                                                   "context": ["A"] * N_PERT},
                                                  index=[f"{p}_{k}" for k in range(N_PERT)]),
                                 var=pd.DataFrame(index=GENES))
                      for p in PERTS])
    pred.write_h5ad(os.path.join(outdir, "pred_controlpaste.h5ad"))
    print("wrote", os.path.join(outdir, "real.h5ad"),
          os.path.join(outdir, "pred_controlpaste.h5ad"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/synth")
