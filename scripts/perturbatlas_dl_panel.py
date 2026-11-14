"""下载 final panel 各靶的 PerturbAtlas KD/KO DEG 表 (每靶最多 4 个, 按覆盖降序)."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

OUT = "outputs/perturbatlas_deg"
BASE = "https://perturbatlas.kratoss.site/api/download"


def valid(path):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        return f.read(2) == b"PK"


def fetch(pid):
    path = f"{OUT}/{pid}_degs.xlsx"
    if valid(path):
        return pid, "have"
    url = f"{BASE}?id={pid}&kind=degs"
    for _ in range(3):
        r = subprocess.run(["curl", "-sL", "--fail", "--max-time", "180",
                            "-o", path, url], capture_output=True)
        if r.returncode == 0 and valid(path):
            return pid, "ok"
    if os.path.exists(path) and not valid(path):
        os.remove(path)
    return pid, "fail"


def main():
    comps = json.load(open("outputs/perturbatlas_panel_comps.json", encoding="utf-8"))
    totals = {x["perturb_id"]: x["total"] for x in
              json.load(open("outputs/perturbatlas_cand_totals.json", encoding="utf-8"))}
    per = {}
    for x in comps:
        if x["condition"] not in ("KD", "KO"):
            continue
        pid = x["perturb_id"]
        if pid not in totals:
            continue
        per.setdefault(x["target"], []).append((totals[pid], pid))
    sel = []
    for t, vs in sorted(per.items()):
        vs.sort(reverse=True)
        for total, pid in vs[:4]:
            sel.append((t, total, pid))
    print(f"panel targets {len(per)}, files to ensure {len(sel)}")
    have = sum(valid(f"{OUT}/{pid}_degs.xlsx") for _, _, pid in sel)
    print(f"already valid {have}")
    fails = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for pid, st in ex.map(fetch, [pid for _, _, pid in sel]):
            if st == "fail":
                fails.append(pid)
    for t, total, pid in sel:
        ok = valid(f"{OUT}/{pid}_degs.xlsx")
        print(f"  {t:9s} {pid:14s} total={total:6d} {'ok' if ok else 'MISSING'}")
    print(f"\nvalid {sum(valid(f'{OUT}/{pid}_degs.xlsx') for _, _, pid in sel)}/{len(sel)}"
          f", fails {fails}")


if __name__ == "__main__":
    sys.exit(main())
