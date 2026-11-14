"""零行填充估算: 共享响应行在 H1 代理上的 DE 方向指标.

对 136 个有 GWPS 行的 H1 靶:
  - 用 GWPS 行 (按生产锚 12.9 缩放) 作为"典型预测行", 其行均值作为共享填充.
  - 真值 DE 集用 |B_i| 阈值近似 (多个阈值).
  - 填充行 DE 集用 |fill| 阈值近似; 计算 k/n_pred/n_conf, fid_yield_raw,
    sig_jaccard, 以及 adjudicated purity 曲线上的 k*/n_conf (reach).
零行的 raw 值均为 0. 报告每靶均值 + 6/300 折算的成员分增量 (b: fid 0.50,
reach 0.025, jac 0.06; 成员分=(raw-b)/(1-b)).
"""

import numpy as np

ANCHOR = 12.90
TS = 50000.0
N_PANEL = 300
N_ZERO = 6


def purity_reach(rank_scores, real_sig, signs, p0=0.9):
    """未加权 purity 曲线; 返回最深 k 使 purity>=p0, 及该点 k/n_conf."""
    order = np.argsort(-rank_scores)
    order = [j for j in order if real_sig[j]]
    if not order:
        return 0, 0.0
    agree = np.cumsum([1.0 if signs[j] else 0.0 for j in order])
    kk = np.arange(1, len(order) + 1)
    purity = agree / kk
    ok = np.where(purity >= p0)[0]
    if len(ok) == 0:
        return 0, 0.0
    kstar = int(ok[-1] + 1)
    return kstar, kstar / float(real_sig.sum())


def main():
    d = np.load("outputs/proxy_profiles.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]
    genes = d["genes"].astype(str)
    gwps = d["gwps"].astype(np.float64)
    tpos = d["tpos"].astype(int)
    filt = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)["filt"].astype(bool)
    B = np.load("outputs/probe_mix_profile.npz", allow_pickle=True)["B"]
    n, ng = gwps.shape
    fg = np.where(filt)[0]
    s = ANCHOR / np.median(np.linalg.norm(gwps[:, filt], axis=1))
    rows = np.clip(s * gwps, -4, 4)
    print(f"scaled GWPS rows: factor {s:.4f}, median filt-norm "
          f"{np.median(np.linalg.norm(rows[:, filt], axis=1)):.2f}")

    fills = {}
    m = rows.mean(0)
    fills["pred_mean"] = m * ANCHOR / max(np.linalg.norm(m[filt]), 1e-9)
    mb = B.mean(0)
    fills["true_mean"] = mb * ANCHOR / max(np.linalg.norm(mb[filt]), 1e-9)
    for nm, v in fills.items():
        v = v.copy()
        v[~filt] = 0.0
        v = np.clip(v, -4, 4)
        fills[nm] = v
        print(f"{nm}: filt-norm {np.linalg.norm(v[filt]):.2f}, "
              f"nz {np.count_nonzero(v[filt])}")

    real_thr = (0.3, 0.5, 0.8)
    pred_ntop = (150, 300, 600, 1000)
    print("\nreal DE size by |B| threshold (filt genes, target excluded):")
    for t in real_thr:
        sizes = []
        for i in range(n):
            ok = filt.copy()
            ok[tpos[i]] = False
            sizes.append(np.count_nonzero((np.abs(B[i]) >= t) & ok))
        print(f"  |B|>={t}: median {np.median(sizes):.0f}, "
              f"mean {np.mean(sizes):.0f}, p10 {np.percentile(sizes, 10):.0f}, "
              f"p90 {np.percentile(sizes, 90):.0f}")

    # 只用 filt 基因做 DE/ranking; 每个靶排除自身靶基因
    for nm, v in fills.items():
        print(f"\n=== fill {nm} ===")
        for nt in pred_ntop:
            order = np.argsort(-np.abs(v))
            pred_sig = np.zeros(ng, bool)
            pred_sig[order[:nt]] = True
            pred_sig &= filt
            # target gene 不参与 (每靶排除); 先算不含 tpos 的版本
            for rt in real_thr:
                fids, jacs, reaches, fids_raw, qs = [], [], [], [], []
                for i in range(n):
                    ok = pred_sig.copy()
                    ok[tpos[i]] = False
                    real = np.zeros(ng, bool)
                    real[(np.abs(B[i]) >= rt) & filt] = True
                    real[tpos[i]] = False
                    inter = ok & real
                    if inter.sum() == 0:
                        fids.append(0.0)
                        jacs.append(0.0)
                        qs.append(np.nan)
                    else:
                        q = np.mean(np.sign(v[inter]) == np.sign(B[i][inter]))
                        k = int(np.count_nonzero(
                            inter & (np.sign(v) == np.sign(B[i]))))
                        qs.append(q)
                        jacs.append(inter.sum() / max((ok | real).sum(), 1))
                        fids.append(k / max(ok.sum(), real.sum()))
                        fids_raw.append(k / max(ok.sum(), 1))
                    rank = np.where(ok, np.abs(v), -1.0)
                    # 只在 real sig 内排序 (adjudicated), 按 |v| 降序
                    kstar, reach = purity_reach(rank, real,
                                                np.sign(v) == np.sign(B[i]))
                    reaches.append(reach)
                print(f"  n_pred={nt:4d} real|B|>={rt}: "
                      f"fid_yield_raw mean {np.mean(fids):.3f} "
                      f"med {np.median(fids):.3f}; q {np.nanmean(qs):.3f}; "
                      f"jac {np.mean(jacs):.3f}; reach {np.mean(reaches):.3f}")

    # 折算: 零行 raw=0, 其成员分为 (0-b)/(1-b); 填充行的增量 = raw/(1-b).
    # 6/300 行 -> 成员增量 = g_row * 6/300; overall 增量 = 成员增量/6.
    print("\n=== implied overall gain for 6/300 rows (zero -> fill) ===")
    b = {"fid": 0.50, "reach": 0.025, "jac": 0.06}
    for nm, v in fills.items():
        for nt in pred_ntop:
            order = np.argsort(-np.abs(v))
            pred_sig = np.zeros(ng, bool)
            pred_sig[order[:nt]] = True
            pred_sig &= filt
            fr, rr, jr = [], [], []
            for i in range(n):
                ok = pred_sig.copy()
                ok[tpos[i]] = False
                real = np.zeros(ng, bool)
                real[(np.abs(B[i]) >= 0.5) & filt] = True
                real[tpos[i]] = False
                inter = ok & real
                k = int(np.count_nonzero(inter & (np.sign(v) == np.sign(B[i]))))
                fr.append(k / max(ok.sum(), real.sum()))
                jr.append(inter.sum() / max((ok | real).sum(), 1))
                rank = np.where(ok, np.abs(v), -1.0)
                _, reach = purity_reach(rank, real, np.sign(v) == np.sign(B[i]))
                rr.append(reach)
            raws = {"fid": np.mean(fr), "reach": np.mean(rr), "jac": np.mean(jr)}
            total = 0.0
            parts = []
            for k_, raw in raws.items():
                g_row = raw / (1 - b[k_])          # zero 行 -> fill 行的成员分增量
                overall = g_row * N_ZERO / N_PANEL / 6.0
                total += overall
                parts.append(f"{k_} raw {raw:.3f} gain/row {g_row:+.3f} "
                             f"overall {overall:+.4f}")
            print(f"  {nm} n_pred={nt}: " + "; ".join(parts))
            print(f"    -> total overall {total:+.4f}")


if __name__ == "__main__":
    main()
