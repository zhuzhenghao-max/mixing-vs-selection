"""Download all shards of one Tahoe cell line and extract condition x gene matrices.

Caches each shard extraction so a crash loses at most the shard in flight.
Stores log2FoldChange and Wald stat. The gene axis is read from
$TAHOE_ROOT/meta/axis_genes.txt (default data/tahoe), and per-line matrices
are written to $TAHOE_ROOT/lines.

Usage:
  python scripts/tahoe_build_line.py <cell line> <first shard> <last shard>
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT = os.environ.get("TAHOE_ROOT", "data/tahoe")
AXIS = [x for x in
        open(os.path.join(ROOT, "meta", "axis_genes.txt")).read().split('\n')
        if x]
PAT = ("{base}/datasets/tahoebio/Tahoe-100M/resolve/{rev}/metadata/"
       "pseudobulk_differential_expression/train-%05d-of-01026.parquet")
SOURCES = [
    ("ms", PAT.format(base="https://modelscope.cn", rev="master"), False),
    ("hf", PAT.format(base="https://huggingface.co", rev="main"), True),
    ("mir", PAT.format(base="https://hf-mirror.com", rev="main"), False),
]
SESS = {}


def session(proxy):
    key = bool(proxy)
    if key not in SESS:
        s = requests.Session()
        s.trust_env = proxy
        SESS[key] = s
    return SESS[key]
OUT = os.path.join(ROOT, "lines")
TMP = os.path.join(ROOT, "tmp")
CACHE = f'{TMP}/extracted'
COLS = ['Cell_Name_Vevo', 'gene_name', 'log2FoldChange', 'stat', 'padj',
        'drug', 'concentration', 'concentration_unit', 'plate',
        'n_cells_trt', 'n_cells_ctrl']


def safe(line):
    return line.replace(' ', '_').replace('/', '_')


def pth(i):
    return f'{TMP}/shard_{i:05d}.parquet'


def size_of(url, sess):
    r = sess.get(url, headers={'Range': 'bytes=0-0'}, allow_redirects=True,
                 timeout=60)
    if r.status_code not in (200, 206) or 'Content-Range' not in r.headers:
        raise RuntimeError(f'HEAD HTTP {r.status_code}')
    return int(r.headers['Content-Range'].split('/')[-1]), r.url


def fetch_file(i):
    last = None
    for attempt in range(9):
        _, url, proxy = SOURCES[attempt % len(SOURCES)]
        sess = session(proxy)
        try:
            total, final = size_of(url % i, sess)
            pos = os.path.getsize(pth(i)) if os.path.exists(pth(i)) else 0
            if pos > total:
                os.remove(pth(i))
                pos = 0
            while pos < total:
                end = min(pos + 8 * 2**20 - 1, total - 1)
                r = sess.get(final, headers={'Range': f'bytes={pos}-{end}'},
                             stream=True, timeout=180)
                if r.status_code == 200:
                    with open(pth(i), 'wb') as fh:
                        for chunk in r.iter_content(2**20):
                            if chunk:
                                fh.write(chunk)
                    pos = os.path.getsize(pth(i))
                    break
                if r.status_code != 206:
                    raise RuntimeError(f'HTTP {r.status_code}')
                with open(pth(i), 'ab' if pos else 'wb') as fh:
                    for chunk in r.iter_content(2**20):
                        if chunk:
                            fh.write(chunk)
                pos = os.path.getsize(pth(i))
            if pos != total:
                raise RuntimeError(f'short {pos}/{total}')
            try:
                pq.read_metadata(pth(i))
            except Exception:
                os.remove(pth(i))
                raise
            return
        except Exception as e:
            last = e
            print(f'  dl retry {i} try {attempt} {type(e).__name__}: {e}',
                  flush=True)
            time.sleep(min(8 * (attempt + 1), 45))
    raise RuntimeError(f'download failed {i}: {last}')


def extract(path):
    t = pq.read_table(path, columns=COLS)
    df = t.to_pandas()
    line = df['Cell_Name_Vevo'].iloc[0]
    df = df[df['gene_name'].isin(AXIS)]
    df['cond'] = (df['drug'].astype(str) + '|' +
                  df['concentration'].map(lambda x: f'{x:.6g}'))
    def piv(col):
        return df.pivot_table(index='cond', columns='gene_name', values=col,
                              aggfunc='mean').reindex(columns=AXIS)
    L, S = piv('log2FoldChange'), piv('stat')
    meta = df.groupby('cond').agg(
        plate=('plate', 'first'), drug=('drug', 'first'),
        conc=('concentration', 'first'), unit=('concentration_unit', 'first'),
        n_trt=('n_cells_trt', 'first'), n_ctrl=('n_cells_ctrl', 'first'),
        padj_frac=('padj', lambda s: float(np.mean(np.isfinite(s)))))
    return line, L, S, meta


def process_shard(i, target):
    cf = f'{CACHE}/{safe(target)}_{i:05d}.npz'
    if os.path.exists(cf):
        d = np.load(cf, allow_pickle=True)
        return str(d['line']), True
    if not os.path.exists(pth(i)):
        fetch_file(i)
    ln, L, S, meta = extract(pth(i))
    os.remove(pth(i))
    if ln != target:
        np.savez(cf, line=ln, foreign=True)
        return ln, False
    np.savez(cf, line=ln, foreign=False,
             X=np.nan_to_num(L.to_numpy(dtype=np.float32), nan=0.0),
             S=np.nan_to_num(S.to_numpy(dtype=np.float32), nan=0.0),
             cond=np.array(L.index.tolist()), drug=meta['drug'].values,
             conc=meta['conc'].values, unit=meta['unit'].values,
             plate=meta['plate'].values, n_trt=meta['n_trt'].values,
             n_ctrl=meta['n_ctrl'].values,
             padj_frac=meta['padj_frac'].values)
    return ln, False


def main():
    line, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    for i in range(start, end + 1):
        t0 = time.time()
        ln, cached = process_shard(i, line)
        print(f'  shard {i} {ln} {"cached" if cached else "ok"} '
              f'{time.time() - t0:.0f}s', flush=True)
    Xs, Ss, metas = [], [], []
    for i in range(start, end + 1):
        cf = f'{CACHE}/{safe(line)}_{i:05d}.npz'
        if not os.path.exists(cf):
            continue
        d = np.load(cf, allow_pickle=True)
        if bool(d['foreign']) or str(d['line']) != line:
            continue
        Xs.append(d['X'])
        Ss.append(d['S'])
        metas.append(pd.DataFrame({
            'cond': d['cond'], 'drug': d['drug'], 'conc': d['conc'],
            'unit': d['unit'], 'plate': d['plate'], 'n_trt': d['n_trt'],
            'n_ctrl': d['n_ctrl'], 'padj_frac': d['padj_frac'],
        }).set_index('cond'))
    if not Xs:
        print(f'NO DATA for {line}')
        return
    X, S = np.vstack(Xs), np.vstack(Ss)
    meta = pd.concat(metas)
    keep = ~meta.index.duplicated(keep='first')
    X, S, meta = X[keep], S[keep], meta[keep]
    out = f'{OUT}/{safe(line)}.npz'
    np.savez_compressed(out, X=X, S=S, cond=np.array(meta.index.tolist()),
                        drug=meta['drug'].values, conc=meta['conc'].values,
                        unit=meta['unit'].values, plate=meta['plate'].values,
                        n_trt=meta['n_trt'].values, n_ctrl=meta['n_ctrl'].values,
                        padj_frac=meta['padj_frac'].values)
    print(f'DONE {line}: {X.shape} -> {out} '
          f'({os.path.getsize(out) / 1e6:.0f}MB)', flush=True)


if __name__ == '__main__':
    main()
