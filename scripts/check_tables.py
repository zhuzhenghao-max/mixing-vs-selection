"""Check tabular column counts in a LaTeX file."""

import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "manuscript/draft.tex"
src = open(path, encoding="utf-8").read()


def split_depth(s, sep):
    """Split on sep only at brace depth zero."""
    out, buf, depth, i = [], [], 0, 0
    while i < len(s):
        if depth == 0 and s.startswith(sep, i):
            out.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        if s[i] == "\\" and i + 1 < len(s):
            buf.append(s[i:i + 2])
            i += 2
            continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        buf.append(s[i])
        i += 1
    out.append("".join(buf))
    return out


def count_cols(spec):
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r"![^,{}]*", "", spec)
    spec = spec.replace(" ", "")
    n = 0
    i = 0
    while i < len(spec):
        c = spec[i]
        if c in "lcrX":
            n += 1
            i += 1
        elif c in "pmb":
            j = spec.find("{", i)
            depth = 0
            while j < len(spec):
                if spec[j] == "{":
                    depth += 1
                elif spec[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            n += 1
            i = j + 1
        elif c == "*":
            print("star spec not handled:", spec)
            break
        else:
            i += 1
    return n


for m in re.finditer(r"\\begin\{tabular\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
                     src):
    spec = m.group(1)
    ncol = count_cols(spec)
    start = m.end()
    end = src.find(r"\end{tabular}", start)
    body = src[start:end]
    rows = split_depth(body, r"\\")
    print(f"tabular at char {m.start()}: {ncol} columns, {len(rows)} rows")
    for row in rows:
        r = row.strip()
        if not r or r.startswith(("\\toprule", "\\midrule", "\\bottomrule",
                                  "\\endhead", "\\addlinespace")):
            continue
        cells = split_depth(r, "&")
        if len(cells) != ncol:
            print(f"  MISMATCH {len(cells)} cells (want {ncol}): {r[:110]}")
