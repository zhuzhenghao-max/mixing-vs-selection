"""Static consistency checks for the manuscript sources."""

import os
import re
from collections import Counter

TEX = "manuscript/draft.tex"
MD = "manuscript/draft.md"
FIGDIR = "manuscript/overleaf_upload/figures"


def main():
    tex = open(TEX, encoding="utf-8").read()
    md = open(MD, encoding="utf-8").read()

    envs = re.findall(r"\\(begin|end)\{([^}]+)\}", tex)
    beg = Counter(e for k, e in envs if k == "begin")
    end = Counter(e for k, e in envs if k == "end")
    bad = [e for e in sorted(set(beg) | set(end)) if beg[e] != end[e]]
    print("env mismatch:", bad or "none")

    t = tex.replace("\\\\", "")
    t = re.sub(r"\\[{}]", "", t)
    t = re.sub(r"(?<!\\)%.*", "", t)
    print("brace balance (0 expected):", t.count("{") - t.count("}"))
    depth = 0
    for n, s in enumerate(tex.splitlines(), 1):
        s2 = s.replace("\\\\", "")
        s2 = re.sub(r"\\[{}]", "", s2)
        s2 = re.sub(r"(?<!\\)%.*", "", s2)
        depth += s2.count("{") - s2.count("}")
        if depth < 0:
            print("  negative depth at line %d: %r" % (n, s[:100]))
            break

    missing = []
    for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        if not os.path.exists(os.path.join(FIGDIR, m)):
            missing.append(m)
    print("missing graphics:", missing or "none")

    legs = sorted(set(int(x) for x in re.findall(r"\\textbf\{Figure (\d+)\.", tex)))
    print("figure legends:", legs)
    figrefs = sorted(set(int(x) for x in re.findall(r"Fig\. (\d+)", tex)))
    print("figure refs:", figrefs)
    print("figs without legend:", [f for f in figrefs if f not in legs] or "none")

    tabs = sorted(set(re.findall(r"\\textbf\{Table (S?\d+)\.", tex)))
    tabrefs = sorted(set(re.findall(r"Table (S?\d+)", tex)))
    print("table legends:", tabs)
    print("tables without legend:", [x for x in tabrefs if x not in set(tabs)] or "none")

    print("Section 2.x refs:", sorted(set(int(x) for x in re.findall(r"Section 2\.(\d+)", tex))))

    res = tex.split("\\section{Results}")[1].split("\\section{Discussion}")[0]
    subs = re.findall(r"\\subsection\{([^}]+)\}", res)
    print("Results subsections:", len(subs))
    for i, s in enumerate(subs, 1):
        print("  2.%d %s" % (i, s))

    mdh = re.findall(r"^### (2\.\d+) (.*)$", md, re.M)
    print("md headings:", [h[0] for h in mdh])
    if len(mdh) != len(subs):
        print("heading count mismatch:", len(mdh), len(subs))

    print("Fig. 10 refs md/tex:", md.count("Fig. 10"), tex.count("Fig. 10"))
    print("Table S11 refs md/tex:", md.count("Table S11"), tex.count("Table S11"))


if __name__ == "__main__":
    main()
