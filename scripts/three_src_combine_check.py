"""orion + GWPS + RPE1 三源合并检验: 在 train 靶交集上评估
单源 / 两两 unit-mix / 三源 unit-mix 网格的迁移 pds (同 harness).
另报 rpe1 的 panel 覆盖 (含 17 orion-only 与 11 missing 名单的命中).
只读, 不产出表.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "src")


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def main():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes = m["genes"].astype(str)
    train_names = m["train_names"].astype(str)
    LFC, filt = m["LFC"], m["filt"]
    n = len(train_names)
    gidx = {g: i for i, g in enumerate(genes)}

    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    o_genes = meta["genes"]
    pert_to_id = meta["pert_to_id"]
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    rowmap = {}
    for r in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[r])], {})[int(pert_ids[r])] = r

    ocols, opos = [], []
    for oc, g in enumerate(o_genes):
        if g in gidx:
            ocols.append(oc)
            opos.append(gidx[g])
    ocols = np.array(ocols)
    opos = np.array(opos)

    def orion_avg(t):
        pid = pert_to_id.get(t)
        if pid is None:
            return None
        vs = []
        for cl in ("hct116", "hek293t"):
            r = rowmap[cl].get(pid)
            if r is None:
                continue
            v = np.asarray(L[r, ocols], dtype=np.float64)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if np.abs(v).sum() > 0:
                vs.append(v)
        if not vs:
            return None
        full = np.zeros(len(genes))
        full[opos] = np.mean(vs, 0)
        return full

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gmap = {str(nm): i for i, nm in enumerate(g["names"].astype(str))}
    grows = g["rows"].astype(np.float64)

    r1 = np.load("outputs/rpe1_rows.npz", allow_pickle=True)
    rmap = {str(nm): i for i, nm in enumerate(r1["names"].astype(str))}
    rrows = r1["rows"].astype(np.float64)
    print(f"gwps {len(gmap)} rows, rpe1 {len(rmap)} rows")

    base = filt & np.all(np.isfinite(LFC), 0)
    pos = np.where(base)[0]
    ref = LFC[:, pos]
    mvec = ref.mean(0)
    refn = np.zeros_like(ref)
    for i in range(n):
        refn[i] = resid(ref[i], mvec)
    refn /= np.maximum(np.linalg.norm(refn, axis=1, keepdims=True), 1e-9)

    # ---- panel coverage of rpe1 ----
    v7 = np.load("outputs/lfc_override_2026_v7.npz", allow_pickle=True)
    v6 = np.load("outputs/lfc_override_2026_v6.npz", allow_pickle=True)
    v5 = np.load("outputs/lfc_override_2026_v5.npz", allow_pickle=True)
    v3 = np.load("outputs/lfc_override_2026_v3.npz", allow_pickle=True)
    panel = [str(x) for x in v7["names"].astype(str)]
    injected = set(str(x) for x in v3["names"].astype(str))
    changed6 = set(panel[i] for i in range(len(panel))
                   if np.linalg.norm(v6["lfc"][i] - v5["lfc"][i]) > 1e-9)
    gwps_cov = set(panel[i] for i in range(len(panel))
                   if np.linalg.norm(v7["lfc"][i] - v6["lfc"][i]) > 1e-9)
    orion_only = changed6 - gwps_cov
    uncovered = set(panel) - injected - changed6          # v7 里被 gwps 补的
    missing11 = uncovered - gwps_cov                      # 两源都缺
    print(f"panel {len(panel)}: injected {len(injected)}, orion {len(changed6)}, "
          f"gwps-cov {len(gwps_cov)}, orion-only {len(orion_only)}, "
          f"uncovered145 {len(uncovered)}, missing {len(missing11)}")
    for nm, s in (("orion-only", orion_only), ("missing11", missing11),
                  ("uncovered145", uncovered)):
        hit = sorted(t for t in s if t in rmap)
        print(f"  rpe1 covers {nm}: {len(hit)}/{len(s)}  {hit[:12]}")

    # ---- eval harness ----
    def eval_rows(name, rows, idx):
        P = np.stack([r[pos] for r in rows])
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        S = Pn @ refn.T
        raws, cs = [], []
        for k, i in enumerate(idx):
            s = S[k]
            rank = 1.0 + np.sum(s > s[i]) + 0.5 * (np.sum(s == s[i]) - 1)
            raws.append(1.0 - (rank - 1.0) / (n - 1.0))
            a, b = P[k], refn[i]
            if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                cs.append(np.corrcoef(a, b)[0, 1])
        print(f"  {name:26s} n={len(idx):3d} pds {np.median(raws):.4f} "
              f"spec med {np.median(cs):+.3f}", flush=True)
        return np.median(raws)

    # ---- train intersection ----
    cov = []
    for i, t in enumerate(train_names):
        ov = orion_avg(t)
        if ov is None or t not in gmap or t not in rmap:
            continue
        cov.append((i, t, ov, grows[gmap[t]], rrows[rmap[t]]))
    print(f"\ntrain targets in ALL three sources: {len(cov)}")
    if not cov:
        return
    idx = [c[0] for c in cov]
    O = [c[2] for c in cov]
    G = [c[3] for c in cov]
    R = [c[4] for c in cov]

    eval_rows("orion", O, idx)
    eval_rows("gwps", G, idx)
    eval_rows("rpe1", R, idx)
    best = {}
    for wa in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        best[f"O{wa:.1f}G{1-wa:.1f}"] = eval_rows(
            f"unit O{wa:.1f}+G{1-wa:.1f}", [wa * unit(a) + (1 - wa) * unit(b)
                                            for a, b in zip(O, G)], idx)
    for wa in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        best[f"O{wa:.1f}R{1-wa:.1f}"] = eval_rows(
            f"unit O{wa:.1f}+R{1-wa:.1f}", [wa * unit(a) + (1 - wa) * unit(b)
                                            for a, b in zip(O, R)], idx)
    for wa in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        best[f"G{wa:.1f}R{1-wa:.1f}"] = eval_rows(
            f"unit G{wa:.1f}+R{1-wa:.1f}", [wa * unit(a) + (1 - wa) * unit(b)
                                            for a, b in zip(G, R)], idx)
    print("triple grid (wO,wG,wR):")
    for wo in (0.2, 0.3, 0.4, 0.5):
        for wg in (0.2, 0.3, 0.4, 0.5):
            wr = round(1 - wo - wg, 2)
            if wr < 0.1:
                continue
            best[f"T{wo:.1f}{wg:.1f}{wr:.1f}"] = eval_rows(
                f"unit T O{wo:.1f}G{wg:.1f}R{wr:.1f}",
                [wo * unit(a) + wg * unit(b) + wr * unit(c)
                 for a, b, c in zip(O, G, R)], idx)
    top = sorted(best.items(), key=lambda kv: -kv[1])[:8]
    print("\nTOP configs:")
    for k, v in top:
        print(f"  {k:14s} {v:.4f}")


if __name__ == "__main__":
    main()
