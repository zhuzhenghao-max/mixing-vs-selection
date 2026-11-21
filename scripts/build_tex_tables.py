"""Generate LaTeX table blocks for the manuscript from the main-table CSVs.

Writes manuscript/tex_tables/table{1..5}.tex. Each file is a tabular block
without a caption; the caption text lives in draft.tex.
"""

import csv
import os
import re

OUT = "manuscript/tex_tables"
LINES = {
    "k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat",
    "h1": "H1", "hct116": "HCT116", "hek293t": "HEK293T",
}
ORDER = ["k562", "rpe1", "hepg2", "jurkat", "h1", "hct116", "hek293t"]


def read(name):
    with open(f"outputs/tables/{name}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def line(v):
    return LINES.get(v.strip().lower(), v)


def num(v):
    return v.replace("-", "$-$", 1) if v.startswith("-") else v


def esc(v):
    return (v.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("->", r" $\to$ "))


def fmt_int(v):
    return f"{int(str(v).replace(',', '')):,}".replace(",", r"\,")


def fmt_p(v):
    v = v.strip()
    if "e" in v:
        a, b = v.split("e")
        return rf"${a} \times 10^{{{int(b)}}}$"
    return v


def fmt_gated(v):
    m = re.search(r"\((\d+)\)\s*$", v)
    if not m:
        return v
    core = v[:m.start()].strip().replace("ns", r"$^{\mathrm{ns}}$")
    return f"{core} ({fmt_int(m.group(1))})"


def fmt_kstar(v):
    v = v.strip()
    if v.startswith(">"):
        return r"$>$" + v[1:]
    x = float(v)
    return f"{x:.0f}" if x > 0 else "0"


def wrap(rows, cols, align, size=r"\footnotesize", sep="4pt"):
    body = "\n".join(" & ".join(c(r) for c in cols) + r" \\" for r in rows)
    return (f"{{\\{size}\n\\setlength{{\\tabcolsep}}{{{sep}}}\n"
            f"\\renewcommand{{\\arraystretch}}{{1.0}}\n"
            f"\\begin{{tabular}}{{{align}}}\n\\toprule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n}}\n")


def table1():
    rows = read("table1_composition")
    body = []
    for i, r in enumerate(rows):
        if i == 4:
            body.append(r"\midrule")
        body.append(" & ".join([
            r["Tier"], r["Held-out line"], fmt_int(r["Pool (targets)"]),
            fmt_int(r["Evaluable queries"]), esc(r["Sources"]),
            fmt_int(r["Gene axis"]), fmt_int(r["Eval positions"]),
        ]) + r" \\")
    return (r"{\footnotesize" "\n" r"\setlength{\tabcolsep}{4pt}"
            "\n" r"\renewcommand{\arraystretch}{1.0}" "\n"
            r"\begin{tabular}{@{}llrrp{2.1in}rr@{}}" "\n"
            r"\toprule" "\n"
            r"Tier & Held-out line & Pool & Evaluable queries & Sources &"
            r" Gene axis & Eval.\ positions \\" "\n" r"\midrule" "\n"
            + "\n".join(body) + "\n" r"\bottomrule" "\n"
            r"\end{tabular}" "\n}\n")


def table2():
    rows = read("table2_main_results")
    body = []
    for i, r in enumerate(rows):
        if i == 4:
            body.append(r"\midrule")
        centered = r["Centered Δ"].replace("ns", r"$^{\mathrm{ns}}$")
        top2 = r["Top-2 self-weight Δ"].replace("ns", r"$^{\mathrm{ns}}$")
        body.append(" & ".join([
            r["Tier"], r["Held-out"], fmt_int(r["n"]),
            f"{r['Equal mix mean (median)']}",
            num(r["Mean response Δ"]),
            num(r["Best single Δ (source)"]),
            fmt_gated(num(r["Gated Δ (n)"])),
            num(centered),
            num(top2),
        ]) + r" \\")
    return (r"{\scriptsize" "\n" r"\setlength{\tabcolsep}{3pt}"
            "\n" r"\renewcommand{\arraystretch}{1.05}" "\n"
            r"\begin{tabular}{@{}llrcccccc@{}}" "\n"
            r"\toprule" "\n"
            r"Tier & Held-out & $n$ & \makecell{Equal mix\\mean (median)} &"
            r" \makecell{Mean response\\$\Delta$PDS} &"
            r" \makecell{Best single\\source $\Delta$PDS} &"
            r" \makecell{Gated mix\\$\Delta$PDS ($n$)} &"
            r" \makecell{Mean centering\\$\Delta$PDS} &"
            r" \makecell{Top-2 self-\\weighting $\Delta$PDS} \\" "\n"
            r"\midrule" "\n" + "\n".join(body) + "\n" r"\bottomrule" "\n"
            r"\end{tabular}" "\n}\n")


def table3():
    rows = read("table3_calibration_budget")
    sel = [r for r in rows if r["block"] == "selection"]
    full = [r for r in rows if r["block"] == "fullmix"]
    sel.sort(key=lambda r: (r["tier"], ORDER.index(r["holdout"].lower()),
                            int(r["k"])))
    full.sort(key=lambda r: (ORDER.index(r["holdout"].lower()), int(r["k"])))
    a = []
    for r in sel:
        a.append(" & ".join([
            r["tier"], line(r["holdout"]), r["k"], num(r["delta"]),
            f"[{num(r['lo'])}, {num(r['hi'])}]", fmt_p(r["p"]), r["win"],
        ]) + r" \\")
    b = []
    km = {(r["holdout"].lower(), r["k"]): r for r in full
          if r["contrast"] == "kmeans"}
    rd = {(r["holdout"].lower(), r["k"]): r for r in full
          if r["contrast"] == "random"}
    for key in sorted(km, key=lambda x: (ORDER.index(x[0]), int(x[1]))):
        r1, r2 = km[key], rd[key]
        b.append(" & ".join([
            line(r1["holdout"]), r1["k"],
            f"{num(r1['delta'])} [{num(r1['lo'])}, {num(r1['hi'])}]",
            f"{num(r2['delta'])} [{num(r2['lo'])}, {num(r2['hi'])}]",
        ]) + r" \\")
    part_a = (r"{\footnotesize" "\n" r"\setlength{\tabcolsep}{4pt}" "\n"
              r"\renewcommand{\arraystretch}{1.0}" "\n"
              r"\begin{tabular}{@{}llrcccc@{}}" "\n" r"\toprule" "\n"
              r"Tier & Held-out & $k$ & \makecell{$\Delta$PDS\\(k-means $-$ random)} &"
              r" 95\% CI & $p$ & Win rate \\" "\n" r"\midrule" "\n"
              + "\n".join(a) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n}\n")
    part_b = (r"{\footnotesize" "\n" r"\setlength{\tabcolsep}{4pt}" "\n"
              r"\renewcommand{\arraystretch}{1.0}" "\n"
              r"\begin{tabular}{@{}lrcc@{}}" "\n" r"\toprule" "\n"
              r"Held-out & $k$ & \makecell{k-means $\Delta$PDS\\{[95\% CI]}} &"
              r" \makecell{random $\Delta$PDS\\{[95\% CI]}} \\" "\n"
              r"\midrule" "\n" + "\n".join(b) + "\n" r"\bottomrule" "\n"
              r"\end{tabular}" "\n}\n")
    return (r"\noindent\textit{(A) Selection experiment}" "\n\n"
            + part_a + "\n"
            + r"\noindent\textit{(B) Full-mix calibration}" "\n\n"
            + part_b)


def table4():
    rows = read("table4_source_budget")
    sub = [r for r in rows if r["block"] == "subsets"]
    exc = [r for r in rows if r["block"] == "exchange"]
    sub.sort(key=lambda r: (ORDER.index(r["holdout"].lower()),
                            r["n_src"], r["subset"]))
    exc.sort(key=lambda r: (ORDER.index(r["holdout"].lower()), r["subset"]))
    a, prev = [], None
    for r in sub:
        h = line(r["holdout"])
        if prev is not None and h != prev:
            a.append(r"\midrule")
        prev = h
        srcs = ", ".join(line(s) for s in r["srcs"].split(","))
        a.append(" & ".join([h, r["subset"], srcs, r["n_src"],
                             f"{float(r['value']):.4f}"]) + r" \\")
    b, prev = [], None
    for r in exc:
        h = line(r["holdout"])
        if prev is not None and h != prev:
            b.append(r"\midrule")
        prev = h
        srcs = ", ".join(line(s) for s in r["srcs"].split(","))
        kk = fmt_kstar(r["kmeans_kstar"])
        kr = fmt_kstar(r["random_kstar"])
        b.append(" & ".join([
            h, srcs, line(r["added"]), num(r["value"]),
            f"[{num(r['lo'])}, {num(r['hi'])}]", kk, kr,
        ]) + r" \\")
    part_a = (r"{\footnotesize" "\n" r"\setlength{\tabcolsep}{4pt}" "\n"
              r"\renewcommand{\arraystretch}{0.95}" "\n"
              r"\begin{tabular}{@{}lllrr@{}}" "\n" r"\toprule" "\n"
              r"Held-out & Subset & Sources & $n$ & Equal-mix PDS \\" "\n"
              r"\midrule" "\n" + "\n".join(a) + "\n" r"\bottomrule" "\n"
              r"\end{tabular}" "\n}\n")
    part_b = (r"{\footnotesize" "\n" r"\setlength{\tabcolsep}{4pt}" "\n"
              r"\renewcommand{\arraystretch}{1.0}" "\n"
              r"\begin{tabular}{@{}lllcccc@{}}" "\n" r"\toprule" "\n"
              r"Held-out & Existing sources & Added & $\Delta$PDS & 95\% CI &"
              r" \makecell{$k^*$\\k-means} & \makecell{$k^*$\\random} \\" "\n"
              r"\midrule" "\n" + "\n".join(b) + "\n" r"\bottomrule" "\n"
              r"\end{tabular}" "\n}\n")
    return (r"\noindent\textit{(A) Equal-mix PDS by source subset}" "\n\n"
            + part_a + "\n"
            + r"\noindent\textit{(B) Source addition and exchange rate}" "\n\n"
            + part_b)


def table5():
    rows = read("table5_regime")
    body = []
    for r in rows:
        v = {
            "Nadig": ("Nadig (no exclusion)", "4", "0.820--0.869", "0.941--0.992",
                      "13--26", r"+0.025 (1$\to$3)", r"+0.011",
                      r"+0.027 ($k$=100)", r"+0.790", "1.90"),
            "Tahoe": ("Tahoe-100M", "7", "0.970--0.994", "0.998--1.000",
                      "42--61", r"+0.023 (1$\to$6)", r"$-$0.013",
                      r"$-$0.006 ($k$=200)", r"$-$0.031", "0.05"),
        }[r["Regime"]]
        body.append(" & ".join(v) + r" \\")
    return (r"{\scriptsize" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
            r"\renewcommand{\arraystretch}{1.1}" "\n"
            r"\begin{tabular}{@{}llcccrcccc@{}}" "\n" r"\toprule" "\n"
            r"Regime & Lines & \makecell{Equal-mix\\mean PDS} &"
            r" \makecell{Equal-mix\\median PDS} & \makecell{Unique top\\(\%)} &"
            r" \makecell{Saturation\\gain} & \makecell{Mean $\Delta$PDS\\$k=50$} &"
            r" \makecell{Best mean\\$\Delta$PDS} & \makecell{Source-\\allocation $\rho$} &"
            r" \makecell{Pilot error\\(\%)} \\" "\n" r"\midrule" "\n"
            + "\n".join(body) + "\n" r"\bottomrule" "\n"
            r"\end{tabular}" "\n}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    for i, f in enumerate((table1, table2, table3, table4, table5), 1):
        path = f"{OUT}/table{i}.tex"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f())
        print("wrote", path)


if __name__ == "__main__":
    main()
