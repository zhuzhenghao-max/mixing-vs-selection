"""Independent truth-pipeline confirmation on the 150 official H1 targets.

The H1 cache rows used in Tier B are decoded responses for 285 targets. The
150 official training targets also have a DESeq2 pseudobulk log2 fold-change
table that the benchmark never used. Here the same cross-line source mixes
are scored against (i) the decoded cache truth and (ii) the DESeq2 truth,
with identical queries and pools, so the truth pipeline is the only change.
A second scope scores the DESeq2 truth against the full 285-row cache pool.

The DESeq2 tail is heavy, so the primary analysis clips fold changes at
+/-10; raw and +/-5 sensitivities are reported. A Hallmark pathway-profile
analysis asks whether the same predictions recover phenotype-level responses
beyond individual genes, with mismatched-target profiles as the negative
control.
"""

import csv
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bench_common import (Retrieval, boot_ci, load_axis, load_lines,
                          unit_rows)

VCCP = "data/vccp_de_log2fc.parquet"
HALL = "data/gs_MSigDB_Hallmark_2020.txt"
BENCH = "outputs/bench_transfer_v2.npz"
SOURCES = ["k562", "rpe1", "hct116", "hek293t"]
CLIPS = {"vccp_raw": None, "vccp_clip5": 5.0, "vccp_clip10": 10.0}


def load_hallmark(genes):
    sets = []
    for line in open(HALL, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        name = parts[0]
        members = [g for g in parts[2:] if g]
        idx = np.array([i for i, g in enumerate(genes) if g in set(members)],
                       dtype=int)
        if len(idx) >= 10:
            sets.append((name, idx))
    return sets


def fill(P, pos, ng):
    out = np.zeros((len(P), ng))
    out[:, pos] = P
    return out


def build_predictions(lines, queries, pos, ng):
    n = len(queries)
    Xeq = np.zeros((n, len(pos)))
    Xc = np.zeros((n, len(pos)))
    Xs = np.zeros((n, len(pos)))
    ns = np.zeros(n, dtype=int)
    means = {o: lines[o]["rows"][:, pos].mean(0) for o in SOURCES}
    for i, t in enumerate(queries):
        acc, cen = [], []
        for o in SOURCES:
            j = lines[o]["index"].get(t)
            if j is None:
                continue
            v = lines[o]["rows"][j, pos]
            nv = np.linalg.norm(v)
            if nv <= 1e-9:
                continue
            acc.append(v / nv)
            cen.append(unit_rows((v - means[o])[None, :])[0])
            if ns[i] == 0:
                Xs[i] = v
            ns[i] += 1
        if acc:
            Xeq[i] = np.mean(acc, axis=0)
            Xc[i] = np.mean(cen, axis=0)
    return Xeq, Xc, Xs, ns


def path_scores(R, As, gidx_t):
    counts = As.sum(1)
    S = R @ As.T
    out = np.zeros_like(S)
    for i, g in enumerate(gidx_t):
        if g is None:
            out[i] = S[i] / counts
            continue
        hit = As[:, g].astype(bool)
        num = S[i] - R[i, g] * hit
        den = np.maximum(counts - hit, 1)
        out[i] = num / den
    return out


def corr_rows(A, B):
    out = np.full(len(A), np.nan)
    for i in range(len(A)):
        a, b = A[i], B[i]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        out[i] = np.corrcoef(a, b)[0, 1]
    return out


def derangement(n, rng):
    p = rng.permutation(n)
    fixed = np.where(p == np.arange(n))[0]
    if len(fixed) > 1:
        p[fixed] = np.roll(p[fixed], 1)
    return p


def main():
    genes, pos, gidx, m = load_axis()
    ng = len(genes)
    z = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    cnames = [str(x) for x in z["train_names"].astype(str)]
    Cfull = z["LFC"].astype(np.float64)
    v = pd.read_parquet(VCCP)
    vnames = [str(x) for x in v.index]
    assert list(v.columns) == list(genes)
    Vfull = v.to_numpy(dtype=np.float64)
    queries = sorted(set(vnames) & set(cnames))
    qi = [vnames.index(t) for t in queries]
    ci = [cnames.index(t) for t in queries]
    V = Vfull[qi]
    C = Cfull[ci]

    lines = load_lines(genes, gidx, m)
    lines.pop("_gidx", None)
    Xeq, Xc, Xs, ns = build_predictions(lines, queries, pos, ng)
    keep = np.where(ns > 0)[0]
    print(f"targets {len(queries)}, with >=1 source {len(keep)}, "
          f"with >=2 {int((ns >= 2).sum())}", flush=True)

    cpool = C  # 150 cache rows aligned to queries
    c285 = Cfull
    c285_names = cnames
    truth = {"cache": C,
             "vccp_raw": V,
             "vccp_clip5": np.clip(V, -5, 5),
             "vccp_clip10": np.clip(V, -10, 10)}
    rows = []
    per_target = {}
    scopes = [("cache150", "cache", cpool, queries),
              ("vccp150_raw", "vccp_raw", truth["vccp_raw"], queries),
              ("vccp150_clip5", "vccp_clip5", truth["vccp_clip5"], queries),
              ("vccp150_clip10", "vccp_clip10", truth["vccp_clip10"],
               queries),
              ("cache285", "cache", c285, c285_names),
              ("cache285", "vccp_clip10", c285, c285_names),
              ("cache285", "vccp_raw", c285, c285_names)]
    rng = np.random.default_rng(0)
    for scope, tname, pool, pnames in scopes:
        R = Retrieval(pool, pnames, pos, gidx=gidx)
        mq = unit_rows(pool[:, pos].mean(0)[None, :])[0]
        mean_pred = fill(np.tile(mq, (len(queries), 1)), pos, ng)
        preds = {"equal": fill(Xeq, pos, ng),
                 "equal_center": fill(Xc, pos, ng),
                 "single": fill(Xs, pos, ng),
                 "mean": mean_pred}
        T = truth[tname]
        Tf = T
        perm = derangement(len(queries), rng)
        preds["truth"] = Tf
        preds["mismatch"] = Tf[perm]
        print(f"\n=== scope {scope} | truth {tname} | pool {len(pnames)} ===",
              flush=True)
        for meth, P in preds.items():
            p = R.score(P, queries, exclude="self", denominator="n")
            qy = keep if meth in ("equal", "equal_center", "single",
                                  "mean") else np.arange(len(queries))
            cos = np.einsum("ij,ij->i", unit_rows(P[qy][:, pos]),
                            unit_rows(Tf[qy][:, pos]))
            lo, hi, mu = boot_ci(p[qy], n=4000, seed=0)
            rows.append(dict(scope=scope, truth=tname, method=meth,
                             n=len(qy), mean=mu, lo=lo, hi=hi,
                             median=float(np.median(p[qy])),
                             win_above_half=float(np.mean(p[qy] > 0.5)),
                             frac_ge_0_9=float(np.mean(p[qy] >= 0.9)),
                             cosine_mean=float(np.mean(cos))))
            print(f"  {meth:13s} PDS {mu:.4f} [{lo:.4f},{hi:.4f}] "
                  f"cos {np.mean(cos):+.4f}", flush=True)
            if scope == "cache150" and meth == "equal":
                per_target["pds_cache"] = p
            if scope == "vccp150_clip10" and meth == "equal":
                per_target["pds_vccp"] = p
        del R

    pd.DataFrame(rows).to_csv("outputs/vccp_confirm_pds.csv", index=False,
                              float_format="%.6f")

    b = np.load(BENCH, allow_pickle=True)
    bench = {(str(h), str(t)): float(p) for h, mth, t, p in
             zip(b["holdout"], b["method"], b["target"], b["pds_raw"])
             if str(mth) == "equal" and str(h) == "h1"}
    per_target["pds_bench"] = np.array([bench.get(("h1", t), np.nan)
                                        for t in queries])

    hall = load_hallmark(genes)
    print(f"\nHallmark sets {len(hall)}", flush=True)
    As = np.zeros((len(hall), ng))
    for j, (_, idx) in enumerate(hall):
        As[j, idx] = 1.0
    gt = [gidx.get(t) for t in queries]
    P_path = path_scores(fill(np.clip(Xeq, -10, 10), pos, ng), As, gt)
    V_path = path_scores(np.clip(V, -10, 10), As, gt)
    C_path = path_scores(C, As, gt)
    perm = derangement(len(queries), rng)
    good = np.array([np.std(P_path[i]) > 0 and np.std(V_path[i]) > 0
                     for i in range(len(queries))])
    valid = good & (ns > 0)
    rp = corr_rows(P_path, V_path)
    rm = corr_rows(P_path, V_path[perm])
    rc = corr_rows(C_path, V_path)
    print(f"pathway profile corr (n={int(valid.sum())}): "
          f"mix {np.nanmean(rp[valid]):+.4f}, "
          f"mismatch {np.nanmean(rm[valid]):+.4f}, "
          f"cache-vccp {np.nanmean(rc[valid]):+.4f}", flush=True)
    per_target["path_corr_mix"] = rp
    per_target["path_corr_mismatch"] = rm
    per_target["path_corr_cache"] = rc

    pw = []
    for j, (name, _) in enumerate(hall):
        a, b_ = P_path[valid, j], V_path[valid, j]
        c_ = C_path[valid, j]
        pw.append(dict(pathway=name,
                       corr_mix=float(np.corrcoef(a, b_)[0, 1])
                       if np.std(a) > 0 and np.std(b_) > 0 else np.nan,
                       corr_cache=float(np.corrcoef(c_, b_)[0, 1])
                       if np.std(c_) > 0 and np.std(b_) > 0 else np.nan))
    pd.DataFrame(pw).sort_values("corr_mix", ascending=False).to_csv(
        "outputs/vccp_confirm_pathways.csv", index=False,
        float_format="%.6f")

    out = pd.DataFrame(dict(target=queries, n_src=ns,
                            pds_cache=per_target["pds_cache"],
                            pds_vccp=per_target["pds_vccp"],
                            pds_bench=per_target["pds_bench"],
                            path_corr_mix=rp, path_corr_mismatch=rm,
                            path_corr_cache=rc))
    out.to_csv("outputs/vccp_confirm_targets.csv", index=False,
               float_format="%.6f")

    ok = (ns > 0) & np.isfinite(out.pds_cache) & np.isfinite(out.pds_vccp)
    from scipy.stats import spearmanr
    print(f"\ncache vs DESeq2 PDS: means {out.pds_cache[ok].mean():.4f} "
          f"/ {out.pds_vccp[ok].mean():.4f}, "
          f"Spearman {spearmanr(out.pds_cache[ok], out.pds_vccp[ok]).statistic:+.3f}",
          flush=True)
    print(f"benchmark H1 subset: {out.pds_bench[ok].mean():.4f}, "
          f"corr with cache150 {spearmanr(out.pds_bench[ok], out.pds_cache[ok]).statistic:+.3f}",
          flush=True)
    print("wrote outputs/vccp_confirm_pds.csv, _targets.csv, _pathways.csv")


if __name__ == "__main__":
    main()
