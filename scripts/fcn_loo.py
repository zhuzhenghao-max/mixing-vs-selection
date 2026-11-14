"""XLearning 式基因级 FCN: [emb_t, ctrl_g] -> delta(g).

LOO-系验证 (k562/hct116/hek293t): 留出一系, 其余系全部行训练,
留出系实测行上 tie-midrank cosine 检索 pds (同 multi_line_loo 口径).

方法:
- single_{src} / equal: 原始空间基线 (raw rows, 同 harness)
- equal_z: z 空间 (per-gene 去均值+标准化) 的 equal mix — 与 FCN-z 同空间对照
- ridge_z: 闭式 per-gene 岭回归 emb_t -> z 行 (信号强度参考)
- fcn_plain: MSE (基因级采样), 报告 z 空间 + raw 空间 pds
- fcn_cos: 行级 cosine loss (z 空间), 报告 z 空间 + raw 空间 pds
"""

import json
import sys

sys.path.insert(0, "src")

import torch  # noqa: E402  (torch 必须先于 numpy, WinError 1114)

import numpy as np  # noqa: E402

LINES = ["k562", "hct116", "hek293t"]
HID = 256
EPOCHS_MSE = 12
EPOCHS_COS = 8
BATCH = 8192
G_PER_SAMPLE = 32
COS_B = 64
SEED = 0


def resid(x, m):
    return x - ((x @ m) / (m @ m)) * m


def unit_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def pds_of(Pn, refn):
    S = Pn @ refn.T
    n = refn.shape[0]
    raws = []
    for k in range(n):
        s = S[k]
        rank = 1.0 + np.sum(s > s[k]) + 0.5 * (np.sum(s == s[k]) - 1)
        raws.append(1.0 - (rank - 1.0) / (n - 1.0))
    return np.median(raws)


def eval_all(Pn, refn):
    """Pn 与 refn 均按行; refn 先 resid 后 unit, Pn 直接 unit."""
    r = refn - refn.mean(0)
    r = unit_rows(r)
    return pds_of(unit_rows(Pn), r)


def load_rows():
    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    genes = m["genes"].astype(str)
    base = m["filt"] & np.all(np.isfinite(m["LFC"]), 0)
    pos = np.where(base)[0]

    rows = {}
    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    rows["k562"] = {nm: r for nm, r in zip(
        [str(x) for x in g["names"].astype(str)], g["rows"].astype(np.float64))}

    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    o_genes = meta["genes"]
    gidx = {g_: i for i, g_ in enumerate(genes)}
    ocols, opos = [], []
    for oc, gg in enumerate(o_genes):
        if gg in gidx:
            ocols.append(oc)
            opos.append(gidx[gg])
    ocols = np.array(ocols)
    opos = np.array(opos)
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    rowmap = {}
    for r in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[r])], {})[int(pert_ids[r])] = r
    for cl in ("hct116", "hek293t"):
        d = {}
        for t, pid in meta["pert_to_id"].items():
            r = rowmap[cl].get(pid)
            if r is None:
                continue
            v = np.nan_to_num(np.asarray(L[r, ocols], dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
            if np.abs(v).sum() == 0:
                continue
            full = np.zeros(len(genes))
            full[opos] = v
            d[t] = full
        rows[cl] = d
    return rows, pos, genes


def make_model(indim):
    return torch.nn.Sequential(
        torch.nn.Linear(indim, HID), torch.nn.ReLU(),
        torch.nn.Linear(HID, HID), torch.nn.ReLU(),
        torch.nn.Linear(HID, HID), torch.nn.ReLU(),
        torch.nn.Linear(HID, 1))


def train_mse(model, batch_fn, n_batches, nep, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(nep):
        model.train()
        tot, nb = 0.0, 0
        for b in range(n_batches):
            xb, yb = batch_fn(ep, b)
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.shape[0]
            nb += xb.shape[0]
        print(f"    ep {ep + 1}/{nep} loss {tot / nb:.5f}", flush=True)
    model.eval()


def train_cos(model, batch_fn, n_batches, nep, nv, lr=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(nep):
        model.train()
        tot, nb = 0.0, 0
        for b in range(n_batches):
            xb, yrow = batch_fn(ep, b)
            out = model(xb).reshape(yrow.shape[0], nv)
            denom = out.norm(2, 1) * yrow.norm(2, 1) + 1e-8
            loss = 1.0 - (out * yrow).sum(1).div(denom).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"    ep {ep + 1}/{nep} cos {tot / nb:.5f}", flush=True)
    model.eval()


def predict_rows(model, Eev_t, Egv, ctrl_v, n, nv, npos, vg, use_embg=False):
    pred = np.zeros((n, npos), dtype=np.float64)
    with torch.no_grad():
        for ti in range(n):
            et = np.repeat(Eev_t[ti][None, :], nv, axis=0)
            if use_embg:
                xb = np.concatenate([et, Egv, ctrl_v[:, None]], 1).astype(np.float32)
            else:
                xb = np.concatenate([et, ctrl_v[:, None]], 1).astype(np.float32)
            pred[ti, vg] = model(torch.from_numpy(xb))[:, 0].numpy()
    return pred


def main():
    np.random.seed(SEED)
    rows, pos, genes = load_rows()
    npos = len(pos)
    gnames = genes[pos]
    print(f"eval positions: {npos}")

    emb = np.load("outputs/scgpt_emb_all.npz", allow_pickle=True)
    epos = {str(n): i for i, n in enumerate(emb["names"].astype(str))}
    E = emb["E"].astype(np.float32)
    gemb = np.array([epos.get(g, -1) for g in gnames])
    vg = np.where(gemb >= 0)[0]
    Egv = E[gemb[vg]]
    nv = len(vg)
    print(f"pos genes with emb: {nv}/{npos}")

    pr = np.load("outputs/line_profiles.npz", allow_pickle=True)
    pnames = [str(x) for x in pr["names"].astype(str)]
    P = np.log1p(pr["prof"].astype(np.float64)[:, pos])
    mu_c, sd_c = float(P.mean()), float(P.std()) + 1e-6
    Ctrl = {nm: (P[pnames.index(nm)] - mu_c) / sd_c for nm in LINES}

    D = {}
    for Ln in LINES:
        tnames = [t for t in rows[Ln] if t in epos]
        T = np.stack([rows[Ln][t][pos] for t in tnames])
        D[Ln] = (tnames, T.astype(np.float32))
        print(f"{Ln}: {len(tnames)} rows with emb")

    results = {}
    for L0 in LINES:
        src = [x for x in LINES if x != L0]
        tset = set(D[L0][0])
        for o in src:
            tset &= set(D[o][0])
        targets = sorted(tset)
        n = len(targets)
        print(f"\n=== hold-out {L0}: {n} targets ===")

        ref_raw = np.stack([rows[L0][t][pos] for t in targets])
        tpos = {t: i for i, t in enumerate(D[L0][0])}
        ref_idx = np.array([tpos[t] for t in targets])

        Ttr = np.concatenate([D[o][1] for o in src]).astype(np.float32)
        tr_lines = np.concatenate([[src.index(o)] * len(D[o][0]) for o in src])
        emb_idx = np.concatenate([[epos[t] for t in D[o][0]] for o in src])
        Etr = E[emb_idx]
        mu_g = Ttr.mean(0)
        sd_g = Ttr.std(0)
        sd_g[sd_g < 1e-6] = 1.0
        Ztr = ((Ttr - mu_g) / sd_g).astype(np.float32)
        ctrl_tr = np.stack([Ctrl[src[l]] for l in tr_lines]).astype(np.float32)
        Ntr = Ttr.shape[0]
        print(f"    train samples {Ntr}")

        refz = (ref_raw - mu_g) / sd_g
        res = {}

        # raw-space baselines
        for o in src:
            Pn = np.stack([rows[o][t][pos] for t in targets])
            res[f"single_{o}"] = eval_all(Pn, ref_raw)
        mix = np.stack([np.mean([rows[o][t][pos] for o in src], 0) for t in targets])
        res["equal"] = eval_all(mix, ref_raw)
        # z-space mix
        mix_z = np.stack([np.mean([(rows[o][t][pos] - mu_g) / sd_g for o in src], 0)
                          for t in targets])
        res["equal_z"] = eval_all(mix_z, refz)
        print(f"    equal {res['equal']:.4f}  equal_z {res['equal_z']:.4f}", flush=True)

        # ridge-z closed form
        Xm, Xs = Etr.mean(0), Etr.std(0) + 1e-6
        Xstd = (Etr - Xm) / Xs
        lam = 100.0
        A = Xstd.T @ Xstd + lam * np.eye(Xstd.shape[1])
        W = np.linalg.solve(A, Xstd.T @ Ztr)
        ev_eidx = np.array([epos[t] for t in targets])
        ev = E[ev_eidx]
        ridge_z = (ev - Xm) / Xs @ W
        res["ridge_z"] = eval_all(ridge_z, refz)
        print(f"    ridge_z {res['ridge_z']:.4f}", flush=True)

        ctrl_ev = Ctrl[L0].astype(np.float32)
        ctrl_v = ctrl_ev[vg]
        Eev_t = E[ev_eidx]

        def eval_fcn(zhat, tag):
            zhat = zhat.astype(np.float64)
            res[f"{tag}_z"] = eval_all(zhat, refz)
            raw = zhat * sd_g + mu_g
            res[f"{tag}_raw"] = eval_all(raw, ref_raw)
            print(f"    {tag}_z {res[f'{tag}_z']:.4f}  {tag}_raw {res[f'{tag}_raw']:.4f}",
                  flush=True)

        # fcn_plain: MSE 基因级采样
        torch.manual_seed(SEED)
        n_batches = Ntr * G_PER_SAMPLE // BATCH

        def mse_batch(ep, b):
            np.random.seed(SEED * 1000 + ep * 100 + b)
            idx_s = np.random.randint(0, Ntr, BATCH)
            idx_v = np.random.randint(0, nv, BATCH)
            xb = np.concatenate(
                [Etr[idx_s], ctrl_tr[idx_s, vg[idx_v]][:, None]], 1).astype(np.float32)
            yb = Ztr[idx_s, vg[idx_v]][:, None].astype(np.float32)
            return torch.from_numpy(xb), torch.from_numpy(yb)

        model = make_model(513)
        train_mse(model, mse_batch, n_batches, EPOCHS_MSE, lr=1e-3)
        eval_fcn(predict_rows(model, Eev_t, Egv, ctrl_v, n, nv, npos, vg), "fcn_mse")

        # fcn_cos: 行级 cosine loss (z 空间)
        torch.manual_seed(SEED)
        n_cos_batches = Ntr // COS_B

        def cos_batch(ep, b):
            np.random.seed(SEED * 2000 + ep * 100 + b)
            tidx = np.random.randint(0, Ntr, COS_B)
            et = torch.from_numpy(Etr[tidx]).repeat_interleave(nv, dim=0)
            cb = torch.from_numpy(ctrl_tr[tidx][:, vg].reshape(-1, 1))
            xb = torch.cat([et, cb], 1)
            yrow = torch.from_numpy(Ztr[tidx][:, vg])
            return xb, yrow

        model = make_model(513)
        train_cos(model, cos_batch, n_cos_batches, EPOCHS_COS, nv)
        eval_fcn(predict_rows(model, Eev_t, Egv, ctrl_v, n, nv, npos, vg), "fcn_cos")

        results[L0] = res
        print(f"  -- hold-out {L0} done --")

    print("\n=== summary (mean pds across hold-outs) ===")
    for k in ["fcn_cos_z", "fcn_mse_z", "ridge_z", "equal_z", "equal",
              "fcn_cos_raw", "fcn_mse_raw"] + [f"single_{o}" for o in LINES]:
        vs = [results[L0].get(k) for L0 in results if k in results[L0]]
        if vs:
            print(f"  {k:12s} {np.mean(vs):.4f}  ({len(vs)} lines)")


if __name__ == "__main__":
    main()
