import os
import shutil
import subprocess
import sys
import threading
import time

def curl_range(url, start, end, out, tries=20):
    size = end - start + 1
    tmp = out + ".tmp"
    # 手动续传: 这个 curl 版本不允许 -C - 和 -r 同时用
    for attempt in range(tries):
        done = os.path.getsize(out) if os.path.exists(out) else 0
        if done >= size:
            return True
        # --fail 保证 HTTP 错误时不把错误页写进文件; 超时退出时 tmp 里是合法前缀
        subprocess.run(
            ["curl", "-s", "--fail", "-r", f"{start + done}-{end}",
             "--max-time", "240", "-o", tmp, url],
            capture_output=True,
        )
        if os.path.exists(tmp):
            with open(out, "ab") as fo, open(tmp, "rb") as fi:
                shutil.copyfileobj(fi, fo)
            os.remove(tmp)
        done = os.path.getsize(out) if os.path.exists(out) else 0
        if done >= size:
            return True
        time.sleep(3)
    return False

def part_sizes(total, n_threads):
    chunk = total // n_threads
    return [chunk if i < n_threads - 1 else total - (n_threads - 1) * chunk
            for i in range(n_threads)]


def main(url, dest, n_threads=4, rounds=20):
    head = subprocess.run(
        ["curl", "-sI", url], capture_output=True, text=True
    )
    total = None
    for line in head.stdout.splitlines():
        if line.lower().startswith("content-length"):
            total = int(line.split(":")[1].strip())
    if total is None:
        print("no content-length, abort")
        return
    if os.path.exists(dest) and os.path.getsize(dest) == total:
        print("already complete:", dest)
        return
    # 分片范围由线程数决定, 换线程数必须换目录, 否则旧文件内容对不上新范围
    tmpdir = f"{dest}.parts{n_threads}"
    os.makedirs(tmpdir, exist_ok=True)
    sizes = part_sizes(total, n_threads)
    for rnd in range(rounds):
        threads = []
        for i, size in enumerate(sizes):
            start = sum(sizes[:i])
            end = start + size - 1
            part = os.path.join(tmpdir, f"part_{i:03d}")
            if os.path.exists(part) and os.path.getsize(part) == size:
                continue
            t = threading.Thread(target=curl_range, args=(url, start, end, part))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        missing = [(i, size) for i, size in enumerate(sizes)
                   if not (os.path.exists(os.path.join(tmpdir, f"part_{i:03d}"))
                           and os.path.getsize(os.path.join(tmpdir, f"part_{i:03d}")) == size)]
        if not missing:
            break
        done_mb = sum(s for i, s in enumerate(sizes) if (i, s) not in missing) / 1e6
        print(f"round {rnd}: {len(missing)} parts incomplete, {done_mb:.0f}MB done")
    else:
        print("gave up after", rounds, "rounds, parts dir:", tmpdir)
        return
    with open(dest, "wb") as out:
        for i in range(n_threads):
            with open(os.path.join(tmpdir, f"part_{i:03d}"), "rb") as f:
                out.write(f.read())
    print("done:", dest)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
