"""Quick structural check of manuscript/draft.tex (not a compiler)."""

import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "manuscript/draft.tex"
src = open(path, encoding="utf-8").read()

# Drop comments: % not preceded by a backslash.
lines = []
for ln in src.splitlines():
    out = []
    i = 0
    while i < len(ln):
        if ln[i] == "\\" and i + 1 < len(ln):
            out.append(ln[i:i + 2])
            i += 2
            continue
        if ln[i] == "%":
            break
        out.append(ln[i])
        i += 1
    lines.append("".join(out))
text = "\n".join(lines)

depth = 0
for i, c in enumerate(text):
    if c == "{":
        depth += 1
    elif c == "}":
        depth -= 1
        if depth < 0:
            print("negative brace depth at char", i)
print("final brace depth:", depth)

stack = []
errs = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
    kind, name = m.group(1), m.group(2)
    if kind == "begin":
        stack.append(name)
    elif not stack or stack[-1] != name:
        errs.append((name, stack[-3:]))
    else:
        stack.pop()
print("unclosed environments:", stack[-5:])
print("environment mismatches:", errs[:5])

body = re.sub(r"\\\[.*?\\\]", "", text, flags=re.S)
print("dollar count (body):", body.count("$"))

for pat in (r"\*\*", r"^#", r"\]\(", r"```"):
    print(pat, len(re.findall(pat, text, re.M)))

for m in re.finditer(r"(?<!\\)%", text):
    s = max(0, m.start() - 40)
    print("UNESCAPED %", repr(text[s:m.end() + 20]))

# underscores in text outside math and texttt: report line numbers
for n, ln in enumerate(text.splitlines(), 1):
    stripped = re.sub(r"\$[^$]*\$", "", ln)
    stripped = re.sub(r"\\texttt\{[^}]*\}", "", stripped)
    stripped = re.sub(r"\\begin\{tabular\}\{[^}]*\}", "", stripped)
    stripped = re.sub(r"\\url\{[^}]*\}", "", stripped)
    stripped = re.sub(r"\\(label|ref|cite)\{[^}]*\}", "", stripped)
    if re.search(r"(?<!\\)_", stripped):
        print("UNDERSCORE line", n, repr(stripped[:100]))

# characters that are only valid escaped (outside math and tabular)
nomath = re.sub(r"\\\[.*?\\\]", "", text, flags=re.S)
nomath = re.sub(r"\$[^$]*\$", "", nomath)
for ch in "#~^":
    for n, ln in enumerate(nomath.splitlines(), 1):
        if re.search(rf"(?<!\\){re.escape(ch)}", ln):
            print(f"RAW {ch!r} line", n, repr(ln[:120]))
for n, ln in enumerate(text.splitlines(), 1):
    if re.search(r"(?<!\\)&", ln) and "tabular" not in ln and " & " not in ln:
        print("RAW '&' line", n, repr(ln[:120]))
