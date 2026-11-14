"""PerturbAtlas 多对比聚合 -> H1 迁移检验.

每个 H1 靶有 1 个 sel (最大覆盖) + 最多 3 个 extra 对比.
变体: single / mean / median / self-LFC 加权 / 一致性过滤 / 最佳 self-LFC.
评估与 perturbatlas_transfer_check 相同: raw PDS + spec, 以及与 GWPS 的 unit mix.
DE 代理指标 (fid/jac/reach) 用 probe_mix_profile 的 B 真值.
"""

import csv
import json

import numpy as np
import pandas as pd


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def load_hgnc_map():
    sym_of = {}
    with open("data/external/hgnc_complete_set.txt", encoding="utf-8") as f:
        for x in csv.DictReader(f, delimiter="\t"):
            e = x.get("ensembl_gene_id")
            if e and e not in sym_of:
                sym_of[e] = x["symbol"]
    return sym_of


def build_one(path, genes, sym_of, gidx, tgt):
    df = pd.read_excel(path)
    df["sym"] = df["gene"].map(sym_of)
    df = df[df["sym"].isin(gidx)]
    ax = df["sym"].map(gidx).to_numpy()
    v = np.zeros(len(genes))
    cnt = np.zeros(len(genes))
    np.add.at(v, ax, df["log2FoldChange"].to_numpy())
    np.add.at(cnt, ax, 1.0)
    v = v / np.maximum(cnt, 1.0)
    ti = gidx.get(tgt)
    self_lfc = float(v[ti]) if ti is not None and cnt[ti] > 0 else np.nan
    padj = df["padj"].to_numpy() if "padj" in df.columns else np.zeros(len(df))
    n_sig = int(np.nansum(padj < 0.05))
    return dict(row=v, self_lfc=self_lfc, n_sig=n_sig,
                n_axis=int((cnt > 0).sum()), nrow=len(df))


def eval_variant(name, vecs, tn, pos, refn):
    idx = [i for i, t in enumerate(tn) if t in vecs]
    if len(idx) < 5:
        print(f"  {name:24s}: only {len(idx)} targets")
        return None
    P = np.stack([vecs[tn[i]][pos] for i in idx])
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    S = Pn @ refn.T
    n = len(tn)
    raws, cs = [], []
    for k, i in enumerate(idx):
        s = S[k]
        rank = 1.0 + np.sum(s > s[i]) + 0.5 * (np.sum(s == s[i]) - 1)
        raws.append(1.0 - (rank - 1.0) / (n - 1.0))
        a, b = P[k], refn[i]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            cs.append(np.corrcoef(a, b)[0, 1])
    cs = np.array(cs)
    print(f"  {name:24s} n={len(idx):3d} pds_med {np.median(raws):.4f} "
          f"pds_mean {np.mean(raws):.4f} spec_med {np.median(cs):+.3f} "
          f"spec_mean {cs.mean():+.3f}")
    return dict(name=name, n=len(idx), pds_med=float(np.median(raws)),
                pds_mean=float(np.mean(raws)), spec_med=float(np.median(cs)),
                spec_mean=float(cs.mean()), raw=dict(zip([tn[i] for i in idx], raws)))


def purity_reach(rank_scores, real, signs, p0=0.9):
    order = np.argsort(-rank_scores)
    order = [j for j in order if real[j]]
    if not order:
        return 0.0
    agree = np.cumsum([1.0 if signs[j] else 0.0 for j in order])
    purity = agree / np.arange(1, len(order) + 1)
    ok = np.where(purity >= p0)[0]
    if len(ok) == 0:
        return 0.0
    return float(ok[-1] + 1) / float(real.sum())


def de_metrics(name, vecs, filt, B, tpos, bnames):
    ok_idx = [i for i, t in enumerate(bnames) if t in vecs]
    if len(ok_idx) < 5:
        print(f"  {name:24s}: only {len(ok_idx)} DE targets")
        return None
    fids, jacs, reaches, qs = [], [], [], []
    for i in ok_idx:
        v = vecs[bnames[i]]
        real = (np.abs(B[i]) >= 0.5) & filt
        real[tpos[i]] = False
        pred = np.zeros_like(real)
        order = np.argsort(-np.abs(v))
        pred[order[:300]] = True
        pred &= filt
        pred[tpos[i]] = False
        inter = pred & real
        if inter.sum() == 0:
            fids.append(0.0)
            jacs.append(0.0)
            qs.append(np.nan)
        else:
            q = np.mean(np.sign(v[inter]) == np.sign(B[i][inter]))
            k = int(np.count_nonzero(inter & (np.sign(v) == np.sign(B[i]))))
            qs.append(q)
            jacs.append(inter.sum() / max((pred | real).sum(), 1))
            fids.append(k / max(pred.sum(), real.sum()))
        rank = np.where(pred, np.abs(v), -1.0)
        reaches.append(purity_reach(rank, real, np.sign(v) == np.sign(B[i])))
    print(f"  {name:24s} n={len(ok_idx):3d} fid {np.mean(fids):.4f} "
          f"jac {np.mean(jacs):.4f} reach {np.mean(reaches):.4f} "
          f"q {np.nanmean(qs):.3f}")
    return dict(fid=float(np.mean(fids)), jac=float(np.mean(jacs)),
                reach=float(np.mean(reaches)))


def main():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes = m["genes"].astype(str)
    tn = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    gidx = {g: i for i, g in enumerate(genes)}

    sel = json.load(open("outputs/perturbatlas_h1_sel.json", encoding="utf-8"))
    extra = json.load(open("outputs/perturbatlas_h1_extra.json", encoding="utf-8"))
    comps = []
    for t, c in sel.items():
        comps.append(dict(target=t, pid=c["perturb_id"], cond=c["condition"],
                          tissue=c["tissue"], total=c["total"], src="sel"))
    for c in extra:
        comps.append(dict(target=c["target"], pid=c["perturb_id"],
                          cond=c["condition"], tissue=c["tissue"],
                          total=c["total"], src="extra"))
    seen = set()
    uniq = []
    for c in comps:
        if c["pid"] in seen:
            continue
        seen.add(c["pid"])
        uniq.append(c)
    comps = uniq
    print(f"comparisons: {len(comps)} over {len(set(c['target'] for c in comps))} targets")

    sym_of = load_hgnc_map()
    rows = {}
    print("building comparison rows:")
    for c in comps:
        path = f"outputs/perturbatlas_deg/{c['pid']}_degs.xlsx"
        try:
            r = build_one(path, genes, sym_of, gidx, c["target"])
        except Exception as e:
            print(f"  FAIL {c['target']} {c['pid']}: {e}")
            continue
        rows.setdefault(c["target"], []).append(dict(c, **r))
    for t in sorted(rows):
        rr = rows[t]
        selfs = ", ".join("nan" if not np.isfinite(x["self_lfc"]) else f"{x['self_lfc']:+.2f}"
                          for x in rr)
        print(f"  {t:9s} n={len(rr)} coverage={[x['total'] for x in rr]} "
              f"self_lfc=[{selfs}]")

    pos = np.where(filt & np.all(np.isfinite(LFC), 0))[0]
    ref = LFC[:, pos].astype(np.float64)
    R = ref - ref.mean(0)
    refn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
    print(f"\neval positions: {len(pos)}, reference rows: {len(tn)}")

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"].astype(str)]
    grows = g["rows"].astype(np.float64)
    gi = {t: i for i, t in enumerate(gnames)}
    gwps = {t: grows[gi[t]] for t in rows if t in gi}
    print(f"gwps overlap: {len(gwps)}/{len(rows)}")

    # ---- aggregation variants ----
    def agg(t, min_total=0, method="mean", cos_thr=0.0):
        rr = [x for x in rows[t] if x["total"] >= min_total]
        if not rr:
            return None
        U = np.stack([unit(x["row"][pos]) for x in rr])
        if method == "max":
            j = int(np.argmax([x["total"] for x in rr]))
            return U[j]
        if method == "mean":
            return U.mean(0)
        if method == "median":
            return np.median(U, 0)
        if method == "selfw":
            w = np.array([max(0.0, -x["self_lfc"]) if np.isfinite(x["self_lfc"])
                          else 0.0 for x in rr])
            if w.sum() <= 1e-9:
                w = np.ones(len(rr))
            return (U * w[:, None]).sum(0) / w.sum()
        if method == "best_lfc":
            j = int(np.argmin([x["self_lfc"] if np.isfinite(x["self_lfc"])
                               else 0.0 for x in rr]))
            return U[j]
        if method == "cos":
            if len(rr) == 1:
                return U[0]
            C = U @ U.T
            keep = C.mean(1) >= cos_thr
            if keep.sum() == 0:
                keep = np.ones(len(rr), bool)
            return U[keep].mean(0)
        raise ValueError(method)

    def embed(vp):
        v = np.zeros(len(genes))
        v[pos] = vp
        return v

    variants = {}
    for meth, mt, ct, label in [
        ("max", 0, 0, "single_max"),
        ("mean", 0, 0, "mean_all"),
        ("mean", 5000, 0, "mean_cov5k"),
        ("mean", 10000, 0, "mean_cov10k"),
        ("median", 5000, 0, "med_cov5k"),
        ("selfw", 5000, 0, "selfw_cov5k"),
        ("best_lfc", 2000, 0, "bestlfc"),
        ("cos", 5000, 0.10, "cos0.10"),
        ("cos", 5000, 0.20, "cos0.20"),
    ]:
        vecs = {}
        for t in rows:
            vp = agg(t, mt, meth, ct)
            if vp is not None:
                vecs[t] = embed(vp)
        variants[label] = vecs

    print("\n=== aggregation alone ===")
    for label, vecs in variants.items():
        eval_variant(label, vecs, tn, pos, refn)

    print("\n=== mixes with GWPS (unit mix) ===")
    for label, vecs in variants.items():
        for w in (0.2, 0.3, 0.4):
            common = [t for t in vecs if t in gwps]
            mix = {t: w * unit(vecs[t]) + (1 - w) * unit(gwps[t]) for t in common}
            eval_variant(f"{label}+w{w:g}", mix, tn, pos, refn)
    gwps_r = eval_variant("gwps", gwps, tn, pos, refn)

    # ---- DE metrics ----
    d = np.load("outputs/probe_mix_profile.npz", allow_pickle=True)
    bnames = [str(x) for x in d["names"]]
    B = d["B"].astype(np.float64)
    tpos = d["tpos"].astype(int)
    print("\n=== DE proxy (needs B) ===")
    de_metrics("gwps", gwps, filt, B, tpos, bnames)
    for label in ("single_max", "mean_cov5k", "mean_all", "selfw_cov5k", "cos0.10"):
        vecs = variants[label]
        de_metrics(label, vecs, filt, B, tpos, bnames)
        for w in (0.3, 0.4):
            common = [t for t in vecs if t in gwps]
            mix = {t: w * unit(vecs[t]) + (1 - w) * unit(gwps[t]) for t in common}
            de_metrics(f"{label}+w{w:g}", mix, filt, B, tpos, bnames)

    # consistency diagnostic
    print("\n=== pairwise cosine within target (cov>=5k) ===")
    for t in sorted(rows):
        rr = [x for x in rows[t] if x["total"] >= 5000]
        if len(rr) < 2:
            continue
        U = np.stack([unit(x["row"][pos]) for x in rr])
        C = U @ U.T
        iu = np.triu_indices(len(rr), 1)
        print(f"  {t:9s} n={len(rr)} pair_cos mean {C[iu].mean():+.3f} "
              f"min {C[iu].min():+.3f} max {C[iu].max():+.3f}")

    # per-target PDS deltas vs gwps for leading variants
    print("\n=== per-target PDS delta vs gwps (w0.3 / w0.4) ===")
    for label in ("mean_cov5k", "selfw_cov5k", "single_max"):
        vecs = variants[label]
        for w in (0.3, 0.4):
            common = [t for t in vecs if t in gwps]
            mix = {t: w * unit(vecs[t]) + (1 - w) * unit(gwps[t]) for t in common}
            r = eval_variant(f"_{label}_w{w:g}", mix, tn, pos, refn)
            if r is None:
                continue
            deltas = sorted(((r["raw"][t] - gwps_r["raw"][t], t) for t in r["raw"]),
                            reverse=True)
            pos_d = [d for d, _ in deltas if d > 0.02]
            neg_d = [d for d, _ in deltas if d < -0.02]
            print(f"    gain>0.02 {len(pos_d)}, loss>0.02 {len(neg_d)}, "
                  f"mean {np.mean([d for d, _ in deltas]):+.4f}")
            print("    top: " + ", ".join(f"{t}{d:+.2f}" for d, t in deltas[:8]))
            print("    bot: " + ", ".join(f"{t}{d:+.2f}" for d, t in deltas[-8:]))

    # save aggregates (only targets present in each variant)
    out = {"genes": genes}
    for label in ("single_max", "mean_all", "mean_cov5k", "med_cov5k",
                  "selfw_cov5k"):
        ks = sorted(variants[label])
        out[f"{label}_names"] = np.array(ks)
        out[label] = np.stack([variants[label][t] for t in ks]).astype(np.float32)
    np.savez("outputs/perturbatlas_multi_h1.npz", **out)
    print("\nsaved outputs/perturbatlas_multi_h1.npz")


if __name__ == "__main__":
    main()
