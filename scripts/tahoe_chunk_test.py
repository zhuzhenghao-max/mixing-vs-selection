"""Measure multi-connection range download throughput for one Tahoe shard."""

import concurrent.futures as cf
import sys
import time

import requests

URL = ("https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/"
       "metadata/pseudobulk_differential_expression/train-00302-of-01026.parquet")


def size_of(url):
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    return int(r.headers["Content-Range"].split("/")[-1]), r.url


def part(final, a, b, out, log):
    t0 = time.time()
    got = 0
    with requests.get(final, headers={"Range": f"bytes={a}-{b}"},
                      stream=True, timeout=180) as r:
        if r.status_code not in (200, 206):
            log.append((a, b, -1, r.status_code, 0))
            return
        with open(out, "wb") as fh:
            for chunk in r.iter_content(2 ** 20):
                if chunk:
                    fh.write(chunk)
                    got += len(chunk)
    log.append((a, b, got, r.status_code, time.time() - t0))


def main():
    global URL
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    if len(sys.argv) > 2:
        URL = sys.argv[2]
    total, final = size_of(URL)
    step = (total + n - 1) // n
    log = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(n) as ex:
        futs = []
        for i in range(n):
            a, b = i * step, min((i + 1) * step - 1, total - 1)
            futs.append(ex.submit(part, final, a, b, f"/tmp/part{i}", log))
        for f in futs:
            f.result()
    dt = time.time() - t0
    mb = sum(x[2] for x in log) / 1e6
    print(f"n={n} total={total/1e6:.1f}MB wall={dt:.1f}s "
          f"aggregate={mb/dt:.2f}MB/s")
    for x in log:
        print(f"  range {x[0]}-{x[1]} got {x[2]/1e6:.1f}MB "
              f"http {x[3]} {x[2]/1e6/max(x[4], 1e-9):.2f}MB/s")


if __name__ == "__main__":
    main()
