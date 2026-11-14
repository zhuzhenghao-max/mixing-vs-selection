import io
import re
import sys

for path in sys.argv[1:]:
    s = io.open(path, encoding="utf-8").read()
    if path.endswith(".tex"):
        m = re.search(r"\\section\*\{Abstract\}(.*?)\\noindent\\textbf\{Keywords",
                      s, re.S)
        t = m.group(1)
        t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", t)
    else:
        m = re.search(r"^## Abstract\s*$(.*?)^\*\*Keywords", s, re.S | re.M)
        t = m.group(1)
    words = [w for w in re.split(r"\s+", t) if re.search(r"[A-Za-z0-9]", w)]
    print(f"{path}: {len(words)} words")
