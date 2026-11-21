"""Check every headline number in the abstract and supplements against
canonical outputs.

Each entry prints the claim, the recomputed value, and a pass flag. The
script reads only files that were used to build the manuscript tables, so a
failure means the text and the stored results disagree.
"""

import csv
import numpy as np
from scipy.stats import spearmanr

TOL = 1e-3
results = []


def check(name, claim, value, tol=TOL):
    ok = abs(float(claim) - float(value)) <= tol
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} claim {claim:+.4f} "
          f"value {float(value):+.4f}")


def check_range(name, lo, hi, vlo, vhi, tol=TOL):
    ok = (vlo >= lo - tol) and (vhi <= hi + tol)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} claim [{lo}, {hi}] "
          f"value [{vlo:.4f}, {vhi:.4f}]")


def check_upper(name, hi, value, tol=TOL):
    ok = value <= hi + tol
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} claim <= {hi:+.4f} "
          f"value {value:+.4f}")


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return np.array([float(r[key]) for r in rows])


def main():
    # Query counts.
    nA = nB = 0
    for path, tag in (("outputs/bench_transfer_nadig_v2.npz", "A"),
                      ("outputs/bench_transfer_v2.npz", "B")):
        z = np.load(path, allow_pickle=True)
        ho, me = z["holdout"].astype(str), z["method"].astype(str)
        n = sum(np.sum((ho == h) & (me == "equal")) for h in np.unique(ho))
        check(f"Tier {tag} equal-mix query count",
              5486 if tag == "A" else 18678, n, tol=0)

    # Equal minus mean, per holdout.
    diffs = []
    for path in ("outputs/bench_transfer_nadig_v2.npz",
                 "outputs/bench_transfer_v2.npz"):
        z = np.load(path, allow_pickle=True)
        ho, me = z["holdout"].astype(str), z["method"].astype(str)
        for h in np.unique(ho):
            e = z["pds_raw"][(ho == h) & (me == "equal")].mean()
            m = z["pds_raw"][(ho == h) & (me == "mean")].mean()
            diffs.append(e - m)
    check_range("equal - mean over nine holdouts", 0.087, 0.329,
                min(diffs), max(diffs))

    # Variant scans excluding gating and centering: max improvement.
    improves = []
    for r in read("outputs/tables/tableS1_all_methods.csv"):
        if r["Method"] in ("equal", "equal_center", "gated_equal",
                           "gated_selfw", "mean") or \
                r["Method"].startswith("single_") or \
                r["Method"].startswith("equal_center"):
            continue
        improves.append(float(list(r.values())[6]))
    for r in read("outputs/tables/tableS3_variant_scans.csv"):
        v = r["Variant"]
        if v == "equal" or "center" in v or v == "gated":
            continue
        try:
            improves.append(float(list(r.values())[6]))
        except ValueError:
            pass
    improves = np.array(improves)
    check_upper("max non-centering variant gain", 0.008, improves.max())
    print(f"      {len(improves)} variant/holdout rows checked; "
          f"range [{improves.min():+.4f}, {improves.max():+.4f}]")

    # Selection audit.
    sel = read("outputs/tables/tableS11_selection_audit.csv")
    audits = [r for r in sel if r["kind"] in ("variants", "singles")]
    gaps = col(audits, "oracle_gap")
    check_range("best-of vs cross-fitted gap, genetic tiers",
                0.03, 0.12, gaps.min(), gaps.max(), tol=0.001)
    a_var = [r for r in sel if r["tier"] == "A" and r["kind"] == "variants"]
    check_range("Tier A honest gain", 0.032, 0.094,
                col(a_var, "honest_gain").min(),
                col(a_var, "honest_gain").max())
    print("      Tier A best fixed rule:",
          sorted(set(r["best_method"] for r in a_var)))
    b_var = [r for r in sel if r["tier"] == "B" and r["kind"] == "variants"]
    hb = col(b_var, "honest_gain")
    print(f"      Tier B honest gains {np.round(hb, 4).tolist()}; "
          f"positive count {int((hb > 0.002).sum())} of {len(hb)}")

    # scGPT (supplement; no longer an abstract headline).
    sc = read("outputs/tables/tableS4_scgpt_baseline.csv")
    row = [r for r in sc if r["Context"] == "RPE1" and
           r["scGPT run"] == "scgpt_r10" and r["Method"] == "equal"][0]
    check("scGPT r10 vs equal on RPE1", 0.190,
          float(row["Δ mean vs scGPT"]), tol=0.0005)

    # GEARS multi-seed.
    ms = read("outputs/seed_runs/multiseed_summary.csv")
    dm = np.array([[float(r["d_mean_s0"]), float(r["d_mean_s1"]),
                    float(r["d_mean_s2"])] for r in ms])
    check_range("GEARS advantage over 12 fold-seed runs",
                0.163, 0.193, dm.min(), dm.max(), tol=0.001)

    # Coherence: genetic pooled (Fisher z of per-holdout rho).
    rd = read("outputs/regime_diagnostic_holdouts.csv")
    rz = np.arctanh(col(rd, "rho_pair_cos"))
    gen_pool = np.tanh(rz.mean())
    check("genetic pooled coherence (Fisher z mean)", 0.34, gen_pool,
          tol=0.005)
    # Drug panel: per-line Spearman, mean and Fisher pool.
    tx = read("outputs/tahoe_coherence.csv")
    by = {}
    for r in tx:
        by.setdefault(r["holdout"], []).append(
            (float(r["pair_cos"]), float(r["pds_full"])))
    rhos = [spearmanr(np.array(v)[:, 0], np.array(v)[:, 1]).statistic
            for v in by.values()]
    print(f"      drug per-line rho {np.round(rhos, 3).tolist()}; "
          f"mean {np.mean(rhos):+.4f}, Fisher {np.tanh(np.arctanh(rhos).mean()):+.4f}")
    check("drug-panel pooled coherence (mean rho)", 0.30,
          np.mean(rhos), tol=0.01)
    check_range("drug-panel within-line coherence rho (2 dp)", 0.23, 0.34,
                round(min(rhos), 2), round(max(rhos), 2), tol=0.001)
    pc = read("outputs/tahoe_partial_corrected.csv")
    check("drug-panel coherence | alignment (Fisher)", -0.35,
          np.tanh(np.arctanh(col(pc, "rho_adj_alignment")).mean()), tol=0.005)
    check("drug-panel alignment | coherence (Fisher)", 0.54,
          np.tanh(np.arctanh(col(pc, "rho_alignment_adj_coherence")).mean()),
          tol=0.005)

    # Exchange rate.
    ex = [r for r in read("outputs/tables/table4_source_budget.csv")
          if r["block"] == "exchange"]
    rates = []
    for r in ex:
        for k in ("kmeans_kstar", "random_kstar"):
            try:
                v = float(r[k])
            except ValueError:
                continue
            if v > 0:
                rates.append(v)
    rates = np.array(rates)
    check_range("source line worth k calibration labels", 46, 97,
                rates.min(), rates.max(), tol=0.5)
    print(f"      finite exchange rates {np.round(np.sort(rates), 1).tolist()}")

    # Regime pilot.
    t5 = read("outputs/tables/table5_regime.csv")
    print("      pilot error: " + ", ".join(
        f"{r['Regime']} {r['Pilot error (%)']}%" for r in t5))

    n_fail = int(np.sum(~np.asarray(results)))
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
