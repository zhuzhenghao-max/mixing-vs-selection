"""Assemble the Overleaf upload folder: draft.tex plus figures/.

Overleaf projects can be uploaded as a folder tree (draft.tex at the root,
figures/ beside it), which avoids the zip upload path.
"""

import glob
import os
import shutil

OUT = "manuscript/overleaf_upload"
TEX = "manuscript/draft.tex"
FIG_GLOB = "figures/fig*_*.png"

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(os.path.join(OUT, "figures"))
shutil.copy2(TEX, os.path.join(OUT, "draft.tex"))
figs = sorted(glob.glob(FIG_GLOB))
for path in figs:
    shutil.copy2(path, os.path.join(OUT, "figures", os.path.basename(path)))
print(f"wrote {OUT}/draft.tex and {len(figs)} figures")
