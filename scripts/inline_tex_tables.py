"""Inline the generated table blocks into manuscript/draft.tex.

Run after scripts/build_tex_tables.py so the Overleaf upload is a single
.tex file plus the figure PNGs.
"""

import os

PATH = "manuscript/draft.tex"

src = open(PATH, encoding="utf-8").read()
for n in range(1, 6):
    key = rf"\input{{tex_tables/table{n}}}"
    tpath = f"manuscript/tex_tables/table{n}.tex"
    if key not in src:
        print(f"table {n}: already inlined")
        continue
    block = open(tpath, encoding="utf-8").read().rstrip()
    src = src.replace(key, block)
    print(f"table {n}: inlined {os.path.getsize(tpath)} bytes")

open(PATH, "w", encoding="utf-8").write(src)
print("wrote", PATH)
