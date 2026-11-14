"""Check citations, figure references, and table references in draft.tex."""

import re

path = "manuscript/draft.tex"
s = open(path, encoding="utf-8").read()

cites = set()
for m in re.finditer(r"\[([0-9,–\- ]+)\]", s):
    for part in m.group(1).split(","):
        part = part.strip()
        if re.fullmatch(r"\d+", part):
            cites.add(int(part))
        elif re.fullmatch(r"\d+\s*[–-]\s*\d+", part):
            a, b = re.split(r"[–-]", part)
            cites.update(range(int(a), int(b) + 1))
bibs = {int(x) for x in re.findall(r"\\bibitem\{r(\d+)\}", s)}
print("citations:", sorted(cites))
print("bibitems:", sorted(bibs))
print("cited but missing bib:", sorted(cites - bibs))
print("bib but never cited:", sorted(bibs - cites))

figs = sorted({int(x) for x in re.findall(r"Fig\.?\s+(\d+)", s)})
tabs = sorted({int(x) for x in re.findall(r"Table\s+(\d+)", s)})
sfigs = sorted({int(x) for x in re.findall(r"Figure\s+(\d+)\.", s)})
stabs = sorted({int(x) for x in re.findall(r"Table\s+(\d+)\.", s)})
print("Fig. references:", figs)
print("Figure legends:", sfigs)
print("Table references:", tabs)
print("Table legends:", stabs)
print("inline graphics:",
      re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s))
