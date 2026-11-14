"""Insert the Tahoe bootstrap CIs into the Results 2.11 calibration passage.

Reads outputs/tahoe_calibration_ci.csv (from scripts/tahoe_cal_ci.py) and
rewrites the cross-line and J82 sentences in draft.md and draft.tex. All
means are taken from the CI table, so the text cannot drift from the data.
"""

import csv
import io
import sys

CI_PATH = "outputs/tahoe_calibration_ci.csv"
FILES = ("manuscript/draft.md", "manuscript/draft.tex")
KS = (20, 50, 100, 200)
MINUS = "−"

A_OLD = (
    "Averaged over lines, the change in mean PDS was negative at every "
    "budget: −0.033 at k = 20, −0.013 at k = 50, −0.008 at "
    "k = 100, and −0.006 at k = 200 with k-means selection, and "
    "slightly lower with random selection at every budget (−0.043, "
    "−0.019, −0.012, and −0.008; Fig. 9C, Table 5). "
    "Six of the seven lines lost PDS under both rules at every budget. "
    "J82 was the only exception, and a small one: k-means reached +0.0004 "
    "at k = 50, +0.0048 at k = 100, and +0.0079 at k = 200, and random "
    "reached +0.0065 at k = 200."
)


def fmt(x, nd, plus):
    s = f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"
    return s.replace("-", MINUS)


def main():
    rows = list(csv.DictReader(io.open(CI_PATH, newline="")))
    tab = {(r["line"], int(r["k"]), r["strategy"]): r for r in rows}
    lines = sorted({r["line"] for r in rows} - {"ALL"})
    missing = [(L, k, s) for L in lines for k in KS
               for s in ("kmeans", "random") if (L, k, s) not in tab]
    missing += [("ALL", k, s) for k in KS for s in ("kmeans", "random")
                if ("ALL", k, s) not in tab]
    if missing:
        sys.exit(f"CI table incomplete, missing: {missing}")
    bad = [k for k, r in tab.items()
           if k[0] != "ALL" and int(r["n"]) != 486]
    if bad:
        sys.exit(f"unexpected n in CI table: {bad}")

    def ci(r, nd=3):
        return (f"{fmt(float(r['lo']), nd, True)} to "
                f"{fmt(float(r['hi']), nd, True)}")

    def mean(r, nd=3, plus=False):
        return fmt(float(r["mean"]), nd, plus)

    km = {k: tab[("ALL", k, "kmeans")] for k in KS}
    rd = {k: tab[("ALL", k, "random")] for k in KS}
    six = [(L, k, s) for L in lines if L != "J82" for k in KS
           for s in ("kmeans", "random")]
    resolved = [key for key in six if float(tab[key]["hi"]) < 0]
    rand_zero = [k for k in KS if float(rd[k]["lo"]) <= 0 <= float(rd[k]["hi"])]

    a_new = (
        "Averaged over lines, the change in mean PDS was negative at every "
        f"budget: {mean(km[20])} (95% CI {ci(km[20])}) at k = 20, "
        f"{mean(km[50])} ({ci(km[50])}) at k = 50, "
        f"{mean(km[100])} ({ci(km[100])}) at k = 100, and "
        f"{mean(km[200])} ({ci(km[200])}) at k = 200 with k-means "
        "selection; random selection was slightly lower at every budget "
        f"({mean(rd[20])}, {mean(rd[50])}, {mean(rd[100])}, and "
        f"{mean(rd[200])}), "
        + ("and its intervals excluded zero at every budget"
           if not rand_zero else
           "although the interval included zero at k = "
           + ", ".join(str(k) for k in rand_zero))
        + " (Fig. 9C, Table 5). "
    )
    if len(resolved) == len(six):
        a_new += (
            f"Six of the seven lines lost PDS under both rules at every "
            f"budget, and the 95% CI excluded zero for all {len(six)} of "
            "those line-budget-rule means. "
        )
    else:
        a_new += (
            f"Six of the seven lines lost PDS under both rules at every "
            f"budget; the 95% CIs lay entirely below zero in "
            f"{len(resolved)} of the {len(six)} line-budget-rule means. "
        )

    j82 = {(k, s): tab[("J82", k, s)] for k in KS
           for s in ("kmeans", "random")}
    b_new = (
        "J82 was the only exception, and a small one: k-means reached "
        f"{mean(j82[(50, 'kmeans')], 4, True)} "
        f"({ci(j82[(50, 'kmeans')], 4)}) at k = 50, "
        f"{mean(j82[(100, 'kmeans')], 4, True)} "
        f"({ci(j82[(100, 'kmeans')], 4)}) at k = 100, and "
        f"{mean(j82[(200, 'kmeans')], 4, True)} "
        f"({ci(j82[(200, 'kmeans')], 4)}) at k = 200, while random "
        f"reached {mean(j82[(200, 'random')], 4, True)} "
        f"({ci(j82[(200, 'random')], 4)}) at k = 200."
    )
    jkeys = [(50, "kmeans"), (100, "kmeans"), (200, "kmeans"),
             (200, "random")]
    jzero = [k for k, s in jkeys
             if float(j82[(k, s)]["lo"]) <= 0 <= float(j82[(k, s)]["hi"])]
    if not jzero:
        b_new += " All four intervals excluded zero."
    elif len(jzero) == len(jkeys):
        b_new += " None of the four intervals excluded zero."
    elif len(jzero) == 1:
        b_new += f" The interval at k = {jzero[0]} still included zero."
    else:
        b_new += (" The intervals at k = "
                  + " and ".join(str(k) for k in jzero)
                  + " still included zero.")

    for path in FILES:
        pct = r"\%" if path.endswith(".tex") else "%"
        new = (a_new + b_new).replace("%", pct)
        s = io.open(path, encoding="utf-8").read()
        n = s.count(A_OLD)
        if n != 1:
            sys.exit(f"{path}: expected 1 match of old passage, found {n}")
        io.open(path, "w", encoding="utf-8").write(s.replace(A_OLD, new))
        print("updated", path)
    print()
    print((a_new + b_new).replace("%", "%"))


if __name__ == "__main__":
    main()
