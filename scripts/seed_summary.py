"""Multi-seed summary of the Tier A GEARS baseline.

Reads the seed 0 CSV plus the merged seed 1/2 CSVs and lite npz files,
reports per-fold d_mean across seeds, cross-seed agreement of per-target
PDS, and writes outputs/seed_runs/multiseed_summary.csv.
"""

import csv

import numpy as np

HOLDS = ["k562", "rpe1", "hepg2", "jurkat"]
FOLDS = list(zip("A A A A".split(), HOLDS))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["holdout"]): r for r in csv.DictReader(f)}


def main():
    src = {
        0: read_csv("outputs/gears_baseline_A.csv"),
        1: read_csv("outputs/seed_runs/gears_baseline_A_seed1_full.csv"),
        2: read_csv("outputs/seed_runs/gears_baseline_A_seed2_full.csv"),
    }
    lite = {
        s: np.load(f"outputs/seed_runs/gears_baseline_A_seed{s}_full_lite.npz",
                   allow_pickle=True) for s in (1, 2)
    }
    eqz = np.load("outputs/bench_transfer_nadig_v2.npz", allow_pickle=True)
    eq_pds = {(str(h), str(t)): float(v) for h, me, t, v in
              zip(eqz["holdout"], eqz["method"], eqz["target"],
                  eqz["pds_raw"]) if str(me) == "equal"}

    rows = []
    for L in HOLDS:
        per = {s: src[s][L] for s in (0, 1, 2)}
        dm = [float(per[s]["d_mean"]) for s in (0, 1, 2)]
        dmed = [float(per[s]["d_med"]) for s in (0, 1, 2)]
        win = [float(per[s]["win"]) for s in (0, 1, 2)]
        pds = [float(per[s]["pds_mean"]) for s in (0, 1, 2)]
        rows.append(dict(holdout=L, n=int(per[0]["n"]),
                         pds_mean_s0=pds[0], pds_mean_s1=pds[1],
                         pds_mean_s2=pds[2],
                         best_epoch_s0=int(per[0]["best_epoch"]),
                         best_epoch_s1=int(per[1]["best_epoch"]),
                         best_epoch_s2=int(per[2]["best_epoch"]),
                         d_mean_s0=dm[0], d_mean_s1=dm[1], d_mean_s2=dm[2],
                         d_mean_avg=float(np.mean(dm)),
                         d_mean_sd=float(np.std(dm, ddof=1)),
                         d_med_avg=float(np.mean(dmed)),
                         win_avg=float(np.mean(win)),
                         equal_mean=float(per[0]["equal_mean"])))

        z0 = np.load("outputs/gears_baseline_A.npz", allow_pickle=True)
        t0 = [str(x) for x in z0[f"{L}|targets"]]
        p0 = {t: float(v) for t, v in zip(t0, z0[f"{L}|pds"])}
        common = list(t0)
        for s in (1, 2):
            ts = [str(x) for x in lite[s][f"{L}|targets"]]
            common = [t for t in common if t in set(ts)]
        pa = np.array([p0[t] for t in common])
        for s in (1, 2):
            ts = [str(x) for x in lite[s][f"{L}|targets"]]
            ps = {t: float(v) for t, v in zip(ts, lite[s][f"{L}|pds"])}
            pb = np.array([ps[t] for t in common])
            r = np.corrcoef(pa, pb)[0, 1]
            sp = np.corrcoef(np.argsort(np.argsort(pa)),
                             np.argsort(np.argsort(pb)))[0, 1]
            rows[-1][f"pearson_s0_s{s}"] = float(r)
            rows[-1][f"spearman_s0_s{s}"] = float(sp)
            print(f"{L:7s} seed0 vs seed{s}: n={len(common)} "
                  f"pearson={r:.4f} spearman={sp:.4f}")
        # per-target spread of the three seed PDS values
        t1 = [str(x) for x in lite[1][f"{L}|targets"]]
        p1 = {t: float(v) for t, v in zip(t1, lite[1][f"{L}|pds"])}
        t2 = [str(x) for x in lite[2][f"{L}|targets"]]
        p2 = {t: float(v) for t, v in zip(t2, lite[2][f"{L}|pds"])}
        stack = np.array([[p0[t], p1[t], p2[t]] for t in common])
        eqt = np.array([eq_pds[(L, t)] for t in common])
        sd = stack.std(1, ddof=1)
        beat = stack < eqt[:, None]
        rows[-1]["seed_sd_median"] = float(np.median(sd))
        rows[-1]["seed_sd_p90"] = float(np.percentile(sd, 90))
        rows[-1]["win_s0"] = float(np.mean(beat[:, 0]))
        rows[-1]["win_s1"] = float(np.mean(beat[:, 1]))
        rows[-1]["win_s2"] = float(np.mean(beat[:, 2]))
        rows[-1]["win_all3"] = float(np.mean(beat.all(1)))
        rows[-1]["win_any"] = float(np.mean(beat.any(1)))
        print(f"{L:7s} per-target seed sd: median {np.median(sd):.4f} "
              f"p90 {np.percentile(sd, 90):.4f}; win "
              f"{np.mean(beat, 0)}, all3 {np.mean(beat.all(1)):.3f}")

    with open("outputs/seed_runs/multiseed_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nper-fold d_mean across seeds")
    for r in rows:
        print(f"{r['holdout']:7s} s0 {r['d_mean_s0']:.4f} "
              f"s1 {r['d_mean_s1']:.4f} s2 {r['d_mean_s2']:.4f} "
              f"avg {r['d_mean_avg']:.4f} sd {r['d_mean_sd']:.4f} "
              f"win {r['win_avg']:.3f}")
    dm = np.array([[r["d_mean_s0"], r["d_mean_s1"], r["d_mean_s2"]]
                   for r in rows])
    print(f"\noverall d_mean {dm.mean():.4f} "
          f"(fold-seed range {dm.min():.4f}-{dm.max():.4f})")


if __name__ == "__main__":
    main()
