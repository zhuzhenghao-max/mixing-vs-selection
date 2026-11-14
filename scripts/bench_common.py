"""跨细胞系扰动迁移 benchmark: 共享数据加载与全池检索 PDS.

参考池 = 留出系全部实测行; 查询 = 留出系中同时被源系覆盖的靶.
PDS 用 tie-midrank, 分母 = 全池行数 - 1.
"""

import json

import numpy as np

AXIS = "outputs/scvi_map_cache.npz"
MASK = "outputs/eval_pos_mask.npz"
PROFILES = "outputs/line_profiles.npz"
ORION_META = "data/external/orion_meta.json"


def unit_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def load_axis():
    m = np.load(AXIS, allow_pickle=True)
    genes = m["genes"].astype(str)
    mask = np.load(MASK)["mask"]
    assert len(mask) == len(genes)
    pos = np.where(mask)[0]
    return genes, pos, {g: i for i, g in enumerate(genes)}, m


def load_profiles():
    d = np.load(PROFILES, allow_pickle=True)
    names = [str(x) for x in d["names"].astype(str)]
    return {n: np.log1p(np.clip(d["prof"][i].astype(np.float64), 0, None))
            for i, n in enumerate(names)}


def profile_sim(a, b):
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return 0.0
    return float(np.dot(a[mask], b[mask]) /
                 (np.linalg.norm(a[mask]) * np.linalg.norm(b[mask]) + 1e-9))


def _self_lfc(rows, names, gidx):
    out = np.full(len(names), np.nan)
    for i, t in enumerate(names):
        j = gidx.get(t)
        if j is not None:
            out[i] = rows[i, j]
    return out


def _add_line(out, name, names, rows, profile, n_cells=None):
    out[name] = dict(names=names, rows=rows,
                     index={t: i for i, t in enumerate(names)},
                     self_lfc=_self_lfc(rows, names, out["_gidx"]),
                     profile=profile, n_cells=n_cells)
    print(f"  {name:8s} {len(names)} rows")


def load_lines(genes, gidx, m):
    prof = load_profiles()
    out = {"_gidx": gidx}

    h1_names = [str(x) for x in m["train_names"].astype(str)]
    h1_rows = m["LFC"].astype(np.float64)
    _add_line(out, "h1", h1_names, h1_rows,
              np.log1p(np.clip(m["base"].astype(np.float64), 0, None)))

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    _add_line(out, "k562", [str(x) for x in g["names"].astype(str)],
              g["rows"].astype(np.float64), prof.get("k562"),
              g["n_cells"].astype(int))

    r = np.load("outputs/rpe1_rows.npz", allow_pickle=True)
    _add_line(out, "rpe1", [str(x) for x in r["names"].astype(str)],
              r["rows"].astype(np.float64), prof.get("rpe1"),
              r["n_cells"].astype(int))

    meta = json.load(open(ORION_META, encoding="utf-8"))
    ocols, opos = [], []
    for oc, gg in enumerate(meta["genes"]):
        if gg in gidx:
            ocols.append(oc)
            opos.append(gidx[gg])
    ocols = np.array(ocols)
    opos = np.array(opos)
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    rowmap = {}
    for i in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[i])], {})[int(pert_ids[i])] = i
    for line in ("hct116", "hek293t"):
        names, mats = [], []
        for t, pid in meta["pert_to_id"].items():
            ri = rowmap.get(line, {}).get(pid)
            if ri is None:
                continue
            v = np.nan_to_num(np.asarray(L[ri, ocols], dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
            if np.abs(v).sum() == 0:
                continue
            full = np.zeros(len(genes))
            full[opos] = v
            names.append(t)
            mats.append(full)
        if names:
            _add_line(out, line, names, np.stack(mats), prof.get(line))
    return out


def load_nadig_lines(genes, gidx):
    """Nadig 4 系统一处理版, 紧凑行展开到 18080 轴; 未被测基因列为 0."""
    z = np.load("outputs/nadig_rows.npz", allow_pickle=True)
    gpos = z["gpos"]
    measured = set(str(x) for x in z["gpos_names"])
    prof = load_profiles()
    out = {}
    for L in ("k562", "rpe1", "hepg2", "jurkat"):
        if f"{L}_names" not in z.files:
            continue
        names = [str(x) for x in z[f"{L}_names"].astype(str)]
        R = np.zeros((len(names), len(genes)), dtype=np.float64)
        R[:, gpos] = z[f"{L}_rows"].astype(np.float64)
        self_lfc = np.full(len(names), np.nan)
        for i, t in enumerate(names):
            if t in measured and t in gidx:
                self_lfc[i] = R[i, gidx[t]]
        out[L] = dict(names=names, rows=R,
                      index={t: i for i, t in enumerate(names)},
                      self_lfc=self_lfc, profile=prof.get(L), n_cells=None)
        print(f"  {L:8s} {len(names)} rows (nadig)")
    return out


class Retrieval:
    """全池 cosine 检索; 支持剔除靶基因/panel 基因并重归一化."""

    def __init__(self, ref_rows, ref_names, pos, resid=False, gidx=None):
        self.names = ref_names
        self.ridx = {t: i for i, t in enumerate(ref_names)}
        self.pos = pos
        self.gidx = gidx
        self.local = {int(g): i for i, g in enumerate(pos)}
        R = ref_rows[:, pos].astype(np.float64)
        self.mean = R.mean(0) if resid else None
        if resid:
            R = R - self.mean
        self.Rn = unit_rows(R)
        self.n = len(ref_names)

    def _cols(self, exclude, qname, panel_names):
        if exclude == "self":
            j = self.gidx.get(qname) if self.gidx else None
            li = self.local.get(j)
            return [li] if li is not None else []
        if exclude == "panel":
            return [self.local[self.gidx[u]] for u in panel_names
                    if u in (self.gidx or {}) and self.gidx[u] in self.local]
        return []

    def score(self, qrows, qnames, exclude="self", panel_names=None, chunk=512,
              denominator="n"):
        out = np.full(len(qnames), np.nan)
        valid = [k for k, t in enumerate(qnames) if t in self.ridx]
        for start in range(0, len(valid), chunk):
            ks = valid[start:start + chunk]
            Q = qrows[ks][:, self.pos].astype(np.float64)
            if self.mean is not None:
                Q -= self.mean[None, :]
            cols_list = [self._cols(exclude, qnames[k], panel_names or [])
                         for k in ks]
            for row, cols in zip(Q, cols_list):
                if cols:
                    row[cols] = 0.0
            Qn = Q / np.maximum(
                np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
            S = self.Rn @ Qn.T
            for j, cols in enumerate(cols_list):
                if cols:
                    S[:, j] /= np.sqrt(np.maximum(
                        1.0 - np.sum(self.Rn[:, cols] ** 2, axis=1), 1e-12))
            for j, k in enumerate(ks):
                col = S[:, j]
                sv = col[self.ridx[qnames[k]]]
                rank0 = np.sum(col > sv) + 0.5 * (np.sum(col == sv) - 1)
                D = self.n if denominator == "n" else self.n - 1
                out[k] = 1.0 if D <= 0 else 1.0 - rank0 / D
        return out


def boot_ci(x, n=1000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.mean(x))
