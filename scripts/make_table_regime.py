"""Build the regime-contrast table (Table 5) from the local artifacts.

Both regimes are scored without a target coordinate so that the drug panel
and the genetic panel are compared under the same policy. Nadig calibration
and saturation numbers come from the balanced 638-query Phase 2 panel.
"""

import csv
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

TAHOE = ["A549", "A498", "HS-578T", "HEC-1-A", "J82", "MIA_PaCa-2",
         "SK-MEL-2"]
NADIG = ["k562", "rpe1", "hepg2", "jurkat"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pilot_rates(tahoe, nadig, k=50, tau=0.92, reps=50000, seed=0):
    out = {}
    for name, arrs in (("Tahoe", [tahoe[L] for L in TAHOE]),
                       ("Nadig", [nadig[L] for L in NADIG])):
        ps = []
        for i, a in enumerate(arrs):
            rng = np.random.default_rng(seed + i)
            m = a[rng.integers(0, len(a), (reps, k))].mean(axis=1)
            ps.append(float(np.mean(m >= tau)))
        out[name] = np.array(ps)
    return out


def cal_curve(path, holdouts, strategies=("kmeans", "random")):
    rows = read(path)
    per = defaultdict(list)
    for r in rows:
        per[(r["strategy"], int(r["k"]))].append(float(r["delta"]))
    return {k: np.mean(v) for k, v in per.items() if k[0] in strategies}


def sat_gain(path, holdouts, sizes):
    rows = read(path)
    tab = defaultdict(list)
    for r in rows:
        s = int(r["size"])
        if s in sizes:
            tab[(r["line"], s)].append(float(r["pds"]))
    per = {L: [np.mean(tab[(L, s)]) for s in sizes] for L in holdouts}
    mean = [np.mean([per[L][i] for L in holdouts]) for i in range(len(sizes))]
    return per, mean


def rho(path):
    rows = read(path)
    a = np.array([float(r["actual"]) for r in rows])
    p = np.array([float(r["pseudo"]) for r in rows])
    return spearmanr(p, a)


def main():
    z = np.load("outputs/pilot_x.npz")
    n = np.load("outputs/pilot_pds_nadig_noexcl.npz")
    tahoe = {f.rsplit(" ", 1)[0]: z[f] for f in z.files
             if f.endswith(" full")}
    nadig = {L: n[L + "_noexcl"] for L in NADIG}
    out = {}
    for name, arrs in (("Nadig", [nadig[L] for L in NADIG]),
                       ("Tahoe", [tahoe[L] for L in TAHOE])):
        mean = [float(a.mean()) for a in arrs]
        med = [float(np.median(a)) for a in arrs]
        top = [float((a >= 1).mean()) for a in arrs]
        out[name] = dict(
            lines=len(arrs),
            mean=f"{min(mean):.3f}-{max(mean):.3f}",
            median=f"{min(med):.3f}-{max(med):.3f}",
            top=f"{100 * min(top):.0f}-{100 * max(top):.0f}",
            mean_lo=min(mean), mean_hi=max(mean),
            top_lo=min(top), top_hi=max(top))
    _, tmean = sat_gain("outputs/cacheX.csv", TAHOE, (1, 2, 6))
    _, nmean = sat_gain("outputs/nadig_saturation_noexcl.csv", NADIG,
                        (1, 2, 3))
    out["Tahoe"]["sat12"] = tmean[1] - tmean[0]
    out["Tahoe"]["satfull"] = tmean[2] - tmean[0]
    out["Tahoe"]["sat_n"] = "1->6"
    out["Nadig"]["sat12"] = nmean[1] - nmean[0]
    out["Nadig"]["satfull"] = nmean[2] - nmean[0]
    out["Nadig"]["sat_n"] = "1->3"
    tc = cal_curve("outputs/selectX.csv", TAHOE)
    nk = defaultdict(list)
    for r in read("outputs/phase2_fullmix_calibration.csv"):
        for s in ("kmeans", "random"):
            nk[(s, int(r["k"]))].append(float(r[s]))
    nc = {k: np.mean(v) for k, v in nk.items()}
    tk = {k: v for k, v in tc.items() if k[0] == "kmeans"}
    out["Tahoe"]["cal_k50"] = tc[("kmeans", 50)]
    out["Tahoe"]["cal_best"] = max(tk.values())
    out["Tahoe"]["cal_best_at"] = max(tk, key=tk.get)
    nkm = {k: v for k, v in nc.items() if k[0] == "kmeans"}
    out["Nadig"]["cal_k50"] = nc[("kmeans", 50)]
    out["Nadig"]["cal_best"] = max(nkm.values())
    out["Nadig"]["cal_best_at"] = max(nkm, key=nkm.get)
    rt = rho("outputs/sscvX.csv")
    rn = rho("outputs/phase2_exante_allocator.csv")
    out["Tahoe"]["rho"] = rt.statistic
    out["Nadig"]["rho"] = rn.statistic
    pr = pilot_rates(tahoe, nadig)
    out["Tahoe"]["pilot_err"] = float(1 - pr["Tahoe"].mean())
    out["Nadig"]["pilot_err"] = float(pr["Nadig"].mean())
    cols = [("Regime", "name"), ("Lines", "lines"),
            ("Equal-mix mean PDS", "mean"),
            ("Equal-mix median PDS", "median"),
            ("Unique-top (%)", "top"),
            ("Saturation gain", "satfull"),
            ("Saturation span", "sat_n"),
            ("kmeans dPDS k=50", "cal_k50"),
            ("Best kmeans dPDS", "cal_best"),
            ("SSCV rho", "rho"),
            ("Pilot error (%)", "pilot_err")]
    print("  ".join(c for c, _ in cols))
    rows = []
    for name in ("Nadig", "Tahoe"):
        o = dict(out[name], name=name)
        o["satfull"] = f"{o['satfull']:+.3f}"
        o["cal_k50"] = f"{o['cal_k50']:+.3f}"
        o["cal_best"] = f"{o['cal_best']:+.3f} at k={o['cal_best_at'][1]}"
        o["rho"] = f"{o['rho']:+.3f}"
        o["pilot_err"] = f"{100 * o['pilot_err']:.2f}"
        vals = [str(o[key]) for _, key in cols]
        print("  ".join(vals))
        rows.append(vals)
    path = "outputs/tables/table5_regime.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in cols])
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
