"""Package draft.tex and the figure PNGs for a single Overleaf upload."""

import glob
import os
import zipfile

OUT = "manuscript/overleaf_upload.zip"
TEX = "manuscript/draft.tex"
FIG_GLOB = "figures/fig*_*.png"

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(TEX, "draft.tex")
    for path in sorted(glob.glob(FIG_GLOB)):
        z.write(path, "figures/" + os.path.basename(path))
print("wrote", OUT)
