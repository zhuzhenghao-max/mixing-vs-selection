"""Build manuscript tables (main + supplementary) from v2 artifacts.

Writes CSV + Markdown to outputs/tables/.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, "scripts")
from make_figures import (DISP, TIERS, load_npz, load_triples, paired_delta,
                          pool_sizes)


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

OUT = Path("outputs/tables")
OUT.mkdir(parents=True, exist_ok=True)
GENE_AXIS = {"A": 6118, "B": 18080}
EVAL_POS = {"A": 5952, "B": 10778}


def md_table(header, rows, path):
    text = ["| " + " | ".join(header) + " |",
            "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        text.append("| " + " | ".join(str(x) for x in r) + " |")
    Path(path).write_text("\n".join(text) + "\n", encoding="utf-8")


def csv_table(header, rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def fmt_d(st):
    return f"{st['mean']:+.3f}{stars(st['p'])}"


def table1():
    header = ["Tier", "Held-out line", "Pool (targets)", "Evaluable queries",
              "Sources", "Gene axis", "Eval positions"]
    rows = []
    for tier in ("A", "B"):
        cfg = TIERS[tier]
        pool = pool_sizes(cfg["log"])
        D = load_npz(cfg["npz"])
        for h in cfg["lines"]:
            src = ", ".join(DISP[o] for o in cfg["lines"] if o != h)
            rows.append([tier, DISP[h], pool.get(h, ""),
                         len(D.get((h, "equal"), {})), src,
                         GENE_AXIS[tier], EVAL_POS[tier]])
    csv_table(header, rows, OUT / "table1_composition.csv")
    md_table(header, rows, OUT / "table1_composition.md")
    print("table1_composition")


def table2():
    rng = np.random.default_rng(10)
    header = ["Tier", "Held-out", "n", "Equal mix mean (median)",
              "Mean response Δ", "Best single Δ (source)", "Gated Δ (n)",
              "Centered Δ", "Top-2 self-weight Δ"]
    rows = []
    for tier in ("A", "B"):
        cfg = TIERS[tier]
        D = load_npz(cfg["npz"])
        for h in cfg["lines"]:
            eq = D.get((h, "equal"), {})
            eqv = np.array(list(eq.values()), dtype=float)
            st_mean = paired_delta(D, h, "mean", "equal", rng)
            singles = {m: d for (hh, m), d in D.items()
                       if hh == h and m.startswith("single_")}
            best_m = max(singles,
                         key=lambda mm: np.nanmean(list(singles[mm].values())))
            st_best = paired_delta(D, h, best_m, "equal", rng)
            st_gate = paired_delta(D, h, "gated_equal", "equal", rng)
            st_cent = paired_delta(D, h, "equal_center", "equal", rng)
            st_top2 = paired_delta(D, h, "top2_selfw", "equal", rng)
            rows.append([
                tier, DISP[h], len(eq),
                f"{eqv.mean():.3f} ({np.median(eqv):.3f})",
                fmt_d(st_mean),
                f"{fmt_d(st_best)} ({DISP[best_m.replace('single_', '')]})",
                f"{fmt_d(st_gate)} ({st_gate['n']})",
                fmt_d(st_cent), fmt_d(st_top2),
            ])
    csv_table(header, rows, OUT / "table2_main_results.csv")
    md_table(header, rows, OUT / "table2_main_results.md")
    print("table2_main_results")


def table_s1():
    rng = np.random.default_rng(11)
    header = ["Tier", "Held-out", "Method", "n", "Mean", "Median",
              "Δ mean vs equal", "CI low", "CI high", "p", "Win rate",
              "Δ median vs equal"]
    rows = []
    for tier in ("A", "B"):
        cfg = TIERS[tier]
        D = load_npz(cfg["npz"])
        for h in cfg["lines"]:
            base = D.get((h, "equal"), {})
            out = []
            for (hh, m), d in D.items():
                if hh != h:
                    continue
                v = np.array(list(d.values()), dtype=float)
                st = (paired_delta(D, h, m, "equal", rng)
                      if m != "equal" else None)
                if st:
                    out.append([tier, DISP[h], m, len(v),
                                f"{v.mean():.4f}", f"{np.median(v):.4f}",
                                f"{st['mean']:+.4f}", f"{st['lo']:+.4f}",
                                f"{st['hi']:+.4f}", f"{st['p']:.2e}",
                                f"{st['win']:.2f}", f"{st['median']:+.4f}"])
                else:
                    out.append([tier, DISP[h], m, len(v),
                                f"{v.mean():.4f}", f"{np.median(v):.4f}",
                                "", "", "", "", "", ""])
            out.sort(key=lambda r: -float(r[4]))
            rows.extend(out)
    csv_table(header, rows, OUT / "tableS1_all_methods.csv")
    md_table(header, rows, OUT / "tableS1_all_methods.md")
    print(f"tableS1_all_methods ({len(rows)} rows)")


def table_s2():
    header = ["Tier", "Held-out", "Source", "n",
              "ρ (PDS, row norm)", "p", "ρ (PDS, self-LFC)", "p"]
    rows = []
    for tier in ("A", "B"):
        cfg = TIERS[tier]
        tri = load_triples(cfg["triples"])
        pairs = sorted({(r["holdout"], r["source"]) for r in tri})
        for h, s in pairs:
            sub = [r for r in tri if r["holdout"] == h and r["source"] == s]
            pds = np.array([float(r["pds_raw"]) for r in sub])
            rn = np.array([float(r["row_norm"]) for r in sub])
            sl = np.array([float(r["self_lfc"]) for r in sub])
            okn = np.isfinite(pds) & np.isfinite(rn)
            okf = np.isfinite(pds) & np.isfinite(sl)
            rn_rho, rn_p = spearmanr(rn[okn], pds[okn])
            sl_rho, sl_p = spearmanr(sl[okf], pds[okf])
            rows.append([tier, DISP[h], DISP[s], int(okn.sum()),
                         f"{rn_rho:+.3f}", f"{rn_p:.1e}",
                         f"{sl_rho:+.3f}", f"{sl_p:.1e}"])
    csv_table(header, rows, OUT / "tableS2_determinants.csv")
    md_table(header, rows, OUT / "tableS2_determinants.md")
    print(f"tableS2_determinants ({len(rows)} rows)")


def parse_scan(path, scan):
    out = {}
    cur = None
    for line in Path(path).read_text(encoding="utf-8",
                                     errors="ignore").splitlines():
        m = re.match(r"=== (\w+):", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"  d\((.+?)\s*-equal\) med ([+\-\d.]+) "
                     r"mean ([+\-\d.]+) win ([\d.]+)", line)
        if m and cur:
            key = (scan, cur, m.group(1))
            out.setdefault(key, {}).update(d_med=float(m.group(2)),
                                           d_mean=float(m.group(3)),
                                           win=float(m.group(4)))
            continue
        m = re.match(r"  (\S+)\s+med ([-\d.]+) mean ([-\d.]+)", line)
        if m and cur:
            out[(scan, cur, m.group(1))] = dict(med=float(m.group(2)),
                                                mean=float(m.group(3)))
    return out


def table_s3():
    scans = parse_scan("outputs/bench_variants_centerq.log", "centerq")
    scans.update(parse_scan("outputs/bench_variants_robust.log", "robust"))
    header = ["Scan", "Held-out", "Variant", "Median", "Mean",
              "Δ median vs equal", "Δ mean vs equal", "Win rate"]
    rows = []
    for (scan, h, v), d in sorted(scans.items(),
                                  key=lambda kv: (kv[0][0], kv[0][1],
                                                  -kv[1]["mean"])):
        rows.append([scan, DISP[h], v, f"{d['med']:.4f}", f"{d['mean']:.4f}",
                     f"{d['d_med']:+.4f}" if "d_med" in d else "",
                     f"{d['d_mean']:+.4f}" if "d_mean" in d else "",
                     f"{d['win']:.2f}" if "win" in d else ""])
    csv_table(header, rows, OUT / "tableS3_variant_scans.csv")
    md_table(header, rows, OUT / "tableS3_variant_scans.md")
    print(f"tableS3_variant_scans ({len(rows)} rows)")


def table_s4():
    z = np.load("outputs/bench_scgpt_baseline.npz", allow_pickle=True)
    header = ["Context", "scGPT run", "Method", "n",
              "scGPT mean (median)", "Method mean (median)",
              "Δ mean vs scGPT", "CI low", "CI high", "p", "Win rate"]
    methods = {}
    for k in z.files:
        if not k.endswith("|pds"):
            continue
        parts = k[:-4].split("|")
        if len(parts) == 3:
            methods.setdefault((parts[0], parts[1]), []).append(parts[2])
    rows = []
    rng = np.random.default_rng(12)
    for (ctx, tag), meths in methods.items():
        base_t = z[f"{ctx}|{tag}|targets"].astype(str)
        base_p = z[f"{ctx}|{tag}|pds"].astype(np.float64)
        bmap = dict(zip(base_t, base_p))
        for meth in sorted(meths):
            mt = z[f"{ctx}|{tag}|{meth}|targets"].astype(str)
            mp = z[f"{ctx}|{tag}|{meth}|pds"].astype(np.float64)
            bp = np.array([bmap[t] for t in mt], dtype=np.float64)
            d = mp - bp
            bs = rng.choice(d, (4000, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            rows.append([DISP.get(ctx, ctx), tag, meth, len(d),
                         f"{bp.mean():.4f} ({np.median(bp):.4f})",
                         f"{mp.mean():.4f} ({np.median(mp):.4f})",
                         f"{d.mean():+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                         f"{p:.2e}", f"{np.mean(d > 0):.2f}"])
    csv_table(header, rows, OUT / "tableS4_scgpt_baseline.csv")
    md_table(header, rows, OUT / "tableS4_scgpt_baseline.md")
    print(f"tableS4_scgpt_baseline ({len(rows)} rows)")


def parse_equal(path):
    out = {}
    cur = None
    for line in Path(path).read_text(encoding="utf-8",
                                     errors="ignore").splitlines():
        m = re.match(r"=== (\w+):", line)
        if m:
            cur = m.group(1)
        m = re.match(r"  equal\s+med ([-\d.]+) mean ([-\d.]+)", line)
        if m and cur:
            out[cur] = (float(m.group(1)), float(m.group(2)))
    return out


def table_s5():
    no = parse_equal("outputs/bench_variants.log")
    se = parse_equal("outputs/bench_variants_selfex.log")
    header = ["Held-out", "Metric", "No exclusion", "Self excluded", "Δ"]
    rows = []
    for h in ("k562", "rpe1", "hct116"):
        if h in no and h in se:
            rows.append([DISP[h], "median", f"{no[h][0]:.4f}",
                         f"{se[h][0]:.4f}", f"{se[h][0] - no[h][0]:+.4f}"])
            rows.append([DISP[h], "mean", f"{no[h][1]:.4f}",
                         f"{se[h][1]:.4f}", f"{se[h][1] - no[h][1]:+.4f}"])
    csv_table(header, rows, OUT / "tableS5_self_exclusion.csv")
    md_table(header, rows, OUT / "tableS5_self_exclusion.md")
    print("tableS5_self_exclusion")


def table_s5b():
    src = Path("outputs/bench_self_lfc_corr_summary.csv")
    if not src.exists():
        print("tableS5b_self_lfc_corr (missing source)")
        return
    header = ["Held-out", "n sampled", "n excludable", "Spearman ρ",
              "95% CI", "p", "Median self-LFC", "Median per-query drop"]
    rows = []
    with open(src, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append([
                DISP[r["holdout"]], r["n_sampled"], r["n_excludable"],
                f"{float(r['spearman_excludable']):+.3f}",
                f"[{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}]",
                f"{float(r['p_excludable']):.1e}",
                f"{float(r['self_lfc_median_excludable']):+.3f}",
                f"{float(r['drop_median_excludable']):+.4f}",
            ])
    csv_table(header, rows, OUT / "tableS5b_self_lfc_corr.csv")
    md_table(header, rows, OUT / "tableS5b_self_lfc_corr.md")
    print("tableS5b_self_lfc_corr")


def table_s6():
    rng = np.random.default_rng(13)
    header = ["Tier", "Held-out", "n gate", "n all",
              "Equal mean (all)", "Equal mean (gate subset)",
              "Gated mean (gate subset)", "Selection Δ", "Gate Δ",
              "Gate p", "Gate win rate"]
    rows = []
    for tier in ("A", "B"):
        cfg = TIERS[tier]
        D = load_npz(cfg["npz"])
        for h in cfg["lines"]:
            eq = D.get((h, "equal"), {})
            ga = D.get((h, "gated_equal"), {})
            common = sorted(set(eq) & set(ga))
            if len(common) < 5:
                continue
            eq_full = np.array(list(eq.values()), dtype=float)
            eq_sub = np.array([eq[t] for t in common], dtype=float)
            ga_sub = np.array([ga[t] for t in common], dtype=float)
            st = paired_delta(D, h, "gated_equal", "equal", rng)
            rows.append([tier, DISP[h], len(common), len(eq),
                         f"{eq_full.mean():.4f}", f"{eq_sub.mean():.4f}",
                         f"{ga_sub.mean():.4f}",
                         f"{eq_sub.mean() - eq_full.mean():+.4f}",
                         f"{ga_sub.mean() - eq_sub.mean():+.4f}",
                         f"{st['p']:.2e}", f"{st['win']:.2f}"])
    csv_table(header, rows, OUT / "tableS6_gate_decomposition.csv")
    md_table(header, rows, OUT / "tableS6_gate_decomposition.md")
    print("tableS6_gate_decomposition")


METHOD_DISP = {
    "equal": "Equal mix",
    "mean": "Mean response",
    "equal_raw": "Raw mix",
    "equal_center": "Centered mix",
    "gated_equal": "Gated mix",
    "top2_selfw": "Top-2 self-weight",
}


def table_s7():
    path = Path("outputs/bench_scope_sensitivity.csv")
    if not path.exists():
        print("tableS7_scope_sensitivity skipped (run bench_scope_sensitivity.py)")
        return
    removal = {}
    log = Path("outputs/bench_scope_sensitivity.log")
    if log.exists():
        for line in log.read_text(encoding="utf-8",
                                  errors="ignore").splitlines():
            m = re.search(r"=== (\w+) (\w+): .*panel removes (\d+), "
                          r"remaining (\d+) ===", line)
            if m:
                removal[(m.group(1), m.group(2))] = (int(m.group(3)),
                                                     int(m.group(4)))
    header = ["Tier", "Held-out", "Method", "n", "Panel removed",
              "Remaining positions", "Mean", "Median",
              "Δ mean vs equal", "CI low", "CI high", "p", "Win rate",
              "Comparison set"]
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            is_base = r["Method"] == "equal"
            rm = removal.get((r["Tier"], r["Held-out"]), ("", ""))
            rows.append([
                r["Tier"], DISP[r["Held-out"]],
                METHOD_DISP.get(r["Method"], r["Method"]), r["n"],
                rm[0], rm[1],
                f"{float(r['Mean']):.4f}", f"{float(r['Median']):.4f}",
                "" if is_base else f"{float(r['Delta mean vs equal']):+.4f}",
                "" if is_base else f"{float(r['CI low']):+.4f}",
                "" if is_base else f"{float(r['CI high']):+.4f}",
                "" if is_base else f"{float(r['p']):.2e}",
                "" if is_base else f"{float(r['Win rate']):.2f}",
                r["Comparison set"],
            ])
    csv_table(header, rows, OUT / "tableS7_scope_sensitivity.csv")
    md_table(header, rows, OUT / "tableS7_scope_sensitivity.md")
    print(f"tableS7_scope_sensitivity ({len(rows)} rows)")


SUPERVISED = [
    ("A", "Ridge (1,024 PCs, no intercept)",
     "outputs/bench_supervised_nadig_v2_mix_noint_pc1024.csv"),
    ("A", "Ridge (256 PCs, intercept)",
     "outputs/bench_supervised_nadig_v2_mix.csv"),
    ("A", "kNN residual (k = 25)",
     "outputs/bench_supervised_nadig_v2_knn_resid.csv"),
    ("A", "Residual MLP (512, 256)",
     "outputs/bench_supervised_nadig_v2_nl_mlp.csv"),
    ("A", "Residual MLP, seed 1 (512, 256)",
     "outputs/bench_supervised_nadig_v2_nl_mlp_seed1.csv"),
    ("B", "Ridge (1,024 PCs, no intercept)",
     "outputs/bench_supervised_v2_mix_noint_pc1024.csv"),
    ("B", "Ridge (256 PCs, intercept)",
     "outputs/bench_supervised_v2_mix.csv"),
    ("B", "kNN residual (k = 25)",
     "outputs/bench_supervised_v2_knn_resid_k25_full.csv"),
    ("B", "Residual MLP (512, 256)",
     "outputs/bench_supervised_v2_nl_mlp.csv"),
    ("B", "Residual MLP, seed 1 (512, 256)",
     "outputs/bench_supervised_v2_nl_mlp_seed1.csv"),
]


def table_s8():
    header = ["Tier", "Held-out", "Model", "n", "Mean (median)", "Equal mean",
              "Δ mean vs equal", "CI low", "CI high", "p", "Win rate",
              "Δ median vs equal"]
    rows = []
    for tier, name, path in SUPERVISED:
        if not Path(path).exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                h = r["holdout"]
                rows.append([
                    tier, DISP[h], name, r["n"],
                    f"{float(r['mean']):.4f} ({float(r['median']):.4f})",
                    f"{float(r['equal_mean']):.4f}",
                    f"{float(r['delta_mean']):+.4f}",
                    f"{float(r['ci_lo']):+.4f}",
                    f"{float(r['ci_hi']):+.4f}",
                    f"{float(r['p']):.2e}",
                    f"{float(r['win']):.2f}",
                    f"{float(r['delta_median']):+.4f}",
                ])
    csv_table(header, rows, OUT / "tableS8_supervised.csv")
    md_table(header, rows, OUT / "tableS8_supervised.md")
    print(f"tableS8_supervised ({len(rows)} rows)")


def main():
    table1()
    table2()
    table_s1()
    table_s2()
    table_s3()
    table_s4()
    table_s5()
    table_s5b()
    table_s6()
    table_s7()
    table_s8()


if __name__ == "__main__":
    main()
