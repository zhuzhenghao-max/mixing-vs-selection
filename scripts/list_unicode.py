"""List non-ASCII characters in a text file with counts and line numbers."""

import collections
import sys
import unicodedata

path = sys.argv[1] if len(sys.argv) > 1 else "manuscript/draft.tex"
s = open(path, encoding="utf-8").read()
cnt = collections.Counter(c for c in s if ord(c) > 127)
for c, n in cnt.most_common():
    print(f"U+{ord(c):04X} {c!r} {n:4d} {unicodedata.name(c, '?')}")
