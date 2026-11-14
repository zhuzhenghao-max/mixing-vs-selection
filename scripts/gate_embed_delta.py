"""靶基因 embedding -> 响应 delta 的 LOO gate (93 个 rpe1 check 靶).

训练: GWPS 517 个有 embedding 的靶 (rows 实测 delta).
预测: 每个 gate 靶留一并从其 embedding 近邻/岭回归预测其 delta 行.
评测: check_npz_retrieval.py (rpe1 rows 池, selfex/cq).
"""
import argparse
import csv
import subprocess
import sys

import numpy as np

AXIS = 'outputs/lfc_override_2026_v10.npz'
MASK = 'outputs/eval_pos_mask.npz'


def load_emb(path):
    z = np.load(path, allow_pickle=True)
    names = [str(x) for x in z['names']]
    E = z['E'].astype(np.float64)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    return names, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default='knn5,knn10,knn20,ridge10,ridge100,ridge1000')
    ap.add_argument('--out-prefix', default='outputs/gate_embed')
    ap.add_argument('--run-checker', action='store_true')
    args = ap.parse_args()

    gw = np.load('outputs/gwps_rows.npz', allow_pickle=True)
    gwn = [str(x) for x in gw['names']]
    rows = gw['rows']
    mask = np.load(MASK)['mask']
    pos = np.where(mask)[0]
    genes_axis = np.load(AXIS, allow_pickle=True)['genes']
    gpos = {str(g): i for i, g in enumerate(genes_axis)}

    gate = [r['target'].strip() for r in
            csv.DictReader(open('data/rpe1_check_targets.csv'))]
    gset = set(gate)
    gidx = {n: i for i, n in enumerate(gwn)}

    embs = {}
    for tag, path in (('esm', 'outputs/esm150_emb.npz'),
                      ('gf', 'outputs/gf316_emb.npz'),
                      ('gfc', 'outputs/gf316_ctx_emb.npz')):
        nm, E = load_emb(path)
        embs[tag] = (nm, E)
    idxs = {tag: {n: i for i, n in enumerate(embs[tag][0])}
            for tag in embs}
    # 训练/预测共同覆盖的 GWPS 靶 + gate 靶
    covered = [n for n in embs['esm'][0]
               if all(n in idxs[tag] for tag in idxs) and n in gidx]
    print(f'gate {len(gate)}, gwps covered by emb {len(covered)}')

    def feat(tag, n):
        return embs[tag][1][idxs[tag][n]]

    def build_pred(method):
        P = np.zeros((len(gate), len(genes_axis)), dtype=np.float32)
        for gi, t in enumerate(gate):
            tr = [n for n in covered if n != t]
            ytr = rows[[gidx[n] for n in tr]]
            if method.startswith('knn'):
                k = int(method[3:])
                X = np.stack([feat('esm', n) for n in tr])
                q = feat('esm', t)
                s = X @ q / (np.linalg.norm(X, axis=1) * np.linalg.norm(q) + 1e-9)
                top = np.argsort(-s)[:k]
                w = np.maximum(s[top], 0) ** 3
                if w.sum() <= 0:
                    w = np.ones_like(w)
                pred = (ytr[top] * w[:, None]).sum(0) / w.sum()
            else:
                alpha = float(method.replace('ridge', ''))
                X = np.stack([np.concatenate([feat('esm', n), feat('gf', n),
                                              feat('gfc', n)]) for n in tr])
                q = np.concatenate([feat('esm', t), feat('gf', t),
                                    feat('gfc', t)])
                mu = X.mean(0)
                sd = X.std(0) + 1e-6
                Xs = (X - mu) / sd
                qs = (q - mu) / sd
                Ym = ytr.mean(0)
                Yc = (ytr - Ym).astype(np.float64)
                A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
                B = Xs.T @ Yc
                W = np.linalg.solve(A, B)
                pred = qs @ W + Ym
            P[gi, pos] = pred[pos].astype(np.float32)
        return P

    for method in args.methods.split(','):
        print(f'== {method} ==', flush=True)
        P = build_pred(method)
        out = f'{args.out_prefix}_{method}.npz'
        np.savez(out, names=np.array(gate), rows=P, genes=genes_axis)
        if args.run_checker:
            for extra in (['--exclude-self', '--center-query'],
                          ['--exclude-self'], []):
                r = subprocess.run(
                    [sys.executable, 'scripts/check_npz_retrieval.py',
                     '--pred', out, '--ref', 'outputs/rpe1_rows.npz',
                     '--targets', 'data/rpe1_check_targets.csv', *extra],
                    capture_output=True, text=True)
                for line in r.stdout.splitlines():
                    if 'pds' in line or 'common' in line:
                        print(' ', line)
                if r.returncode != 0:
                    print(r.stderr[-800:])


if __name__ == '__main__':
    main()
