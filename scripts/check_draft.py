import re

s = open("manuscript/draft.tex", encoding="utf-8").read()
print("begin/end", s.count("\\begin{"), s.count("\\end{"))
print("braces", s.count("{"), s.count("}"))
print("Secondary metrics", s.count("\\subsection*{Secondary metrics}"))
print("Table S15", s.count("Table S15"), "Table S16", s.count("Table S16"))

a = s.split("\\section*{Abstract}")[1].split("\\noindent\\textbf{Keywords")[0]
a = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", a)
a = re.sub(r"[^A-Za-z0-9'\-]+", " ", a)
print("abstract words", len(a.split()))
