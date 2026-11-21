"""Local Nadig sensitivity to genuinely small response reference pools.

Each draw selects m distinct shared targets without replacement. Those same
m targets are both the queries and the entire held-out response reference
pool. Source-only mixtures are fixed before sampling: unit source responses
on the benchmark mask, then equal averaging, then retrieval normalization.
No calibration is fitted. Cached pairwise similarities (not full-pool PDS)
are sliced on BOTH axes and tie-midrank is recomputed with denominator m.

self: remove each query's target coordinate from that query AND every
candidate response, renormalizing each candidate separately. none: retain
all masked coordinates. Ties use exact floating-point equality, as in
FastRetrieval. Unique-top means the correct response is the sole maximum.
Subset quantiles are NOT confidence intervals. The 0.92 threshold is
post hoc, with no external validation; Tahoe response matrices are absent.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from bench_supervised import build_features, build_targets
from phase1_selection_pilot import FastRetrieval

ROOT = Path(__file__).resolve().parents[1]
RESPONSES = ROOT / "outputs/nadig_rows.npz"
AXIS = ROOT / "outputs/scvi_map_cache.npz"
MASK = ROOT / "outputs/eval_pos_mask.npz"
LEGACY_PDS = ROOT / "outputs/pilot_pds_nadig_noexcl.npz"
OUTPUT = ROOT / "outputs/pilot_small_pool.csv"
HOLDOUTS = ("k562", "rpe1", "hepg2", "jurkat")


def similarity_matrix(retriever, features, queries, policy):
    """Return reference-by-query cosines; preserve FastRetrieval arithmetic."""
    assert policy in ("self", "none")
    Q = np.asarray(features, dtype=np.float32).copy()
    cols = [retriever.local.get(retriever.gidx.get(t)) for t in queries]
    if policy == "self":
        for k, col in enumerate(cols):
            if col is not None:
                Q[k, col] = 0.0
    Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
    S = retriever.Rn @ Q.T
    if policy == "self":
        for k, col in enumerate(cols):
            if col is not None:
                S[:, k] /= np.sqrt(np.maximum(
                    1.0 - retriever.Rn[:, col] ** 2, 1e-12))
    assert np.isfinite(S).all()
    return S


def rank_scores(S):
    """Aligned square pool: diagonal is the correct candidate for each query."""
    assert S.ndim == 2 and S.shape[0] == S.shape[1] and len(S) > 0
    assert np.isfinite(S).all()
    m = len(S)
    true_sim = np.diag(S)
    greater = (S > true_sim[None, :]).sum(axis=0)
    ties = (S == true_sim[None, :]).sum(axis=0)
    rank0 = greater + 0.5 * (ties - 1)
    scores = 1.0 - rank0 / m
    assert np.all(ties >= 1)
    assert np.all((scores >= 1.0 / m - 1e-12) & (scores <= 1.0))
    unique_top = (greater == 0) & (ties == 1)
    return scores, unique_top, ties > 1


def subset_scores(S, selected):
    selected = np.asarray(selected, dtype=int)
    assert len(np.unique(selected)) == len(selected)
    # This is the essential small-pool operation, not resampling full-pool PDS.
    return rank_scores(S[np.ix_(selected, selected)])


def synthetic_tests():
    for m in (1, 2, 50, 100):
        p, top, ties = rank_scores(np.ones((m, m)))
        np.testing.assert_allclose(p, 0.5 + 1.0 / (2 * m), atol=1e-14)
        assert np.all(top == (m == 1))
        assert np.all(ties == (m > 1))
        np.testing.assert_array_equal(rank_scores(np.eye(m))[0], np.ones(m))
        np.testing.assert_allclose(rank_scores(-np.eye(m))[0], 1.0 / m)
    p, top, ties = rank_scores(np.array([[1.0, 0.0], [1.0, 2.0]]))
    np.testing.assert_array_equal(p, [0.75, 1.0])
    np.testing.assert_array_equal(top, [False, True])
    np.testing.assert_array_equal(ties, [True, False])
    rng = np.random.default_rng(71)
    S = rng.normal(size=(9, 9))
    perm = rng.permutation(9)
    np.testing.assert_array_equal(subset_scores(S, perm)[0], rank_scores(S)[0][perm])
    try:
        subset_scores(S, [0, 0])
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicate target subset was not rejected")

    # Check unit-before-average even when source magnitudes differ greatly.
    lines = {
        "a": {"rows": np.array([[100.0, 0.0]]), "index": {"t": 0}},
        "b": {"rows": np.array([[0.0, 1.0]]), "index": {"t": 0}},
    }
    np.testing.assert_array_equal(
        build_features(lines, ["a", "b"], ["t"], np.arange(2), "mix"),
        [[0.5, 0.5]])

    # Regression test for per-reference (not scalar pooled) normalization.
    R = np.array([[0.8, 0.6, 0.0], [0.0, 0.4, 0.3], [0.0, 0.0, 1.0]],
                 dtype=np.float32)
    Q = np.array([[0.9, 0.3, 0.1]], dtype=np.float32)
    retriever = FastRetrieval(R, ["t", "u", "v"], np.arange(3), {"t": 0})
    expected_R = R.astype(np.float64).copy()
    expected_Q = Q.astype(np.float64).copy()
    expected_R[:, 0] = 0.0
    expected_Q[:, 0] = 0.0
    expected_R /= np.linalg.norm(expected_R, axis=1, keepdims=True)
    expected_Q /= np.linalg.norm(expected_Q, axis=1, keepdims=True)
    expected = expected_R @ expected_Q.T
    actual = similarity_matrix(retriever, Q, ["t"], "self")
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    rank0 = (actual[:, 0] > actual[0, 0]).sum() + 0.5 * (
        (actual[:, 0] == actual[0, 0]).sum() - 1)
    np.testing.assert_array_equal(retriever.score_pos(Q, ["t"]), [1 - rank0 / 3])
    print("Synthetic assertions passed", flush=True)


def load_compact():
    """Equivalent to load_axis/load_nadig_lines without 18080-axis zero padding."""
    with np.load(AXIS, allow_pickle=True) as axis:
        genes = axis["genes"].astype(str)
    with np.load(MASK, allow_pickle=True) as mask_file:
        mask = mask_file["mask"]
        np.testing.assert_array_equal(mask_file["genes"].astype(str), genes)
    assert len(mask) == len(genes) and mask.dtype == bool
    with np.load(RESPONSES, allow_pickle=True) as z:
        gpos = z["gpos"]
        measured = z["gpos_names"].astype(str)
        np.testing.assert_array_equal(genes[gpos], measured)
        assert len(gpos) == 6118 and np.all(np.diff(gpos) > 0)
        # Same 5952 columns and order as bench_transfer_nadig.py:19-25.
        pos = np.flatnonzero(mask[gpos])
        assert len(pos) == 5952
        gidx = {str(t): i for i, t in enumerate(measured)}
        lines = {}
        for L in HOLDOUTS:
            names = list(z[L + "_names"].astype(str))
            rows = z[L + "_rows"].astype(np.float32)
            assert rows.shape == (len(names), 6118)
            assert len(set(names)) == len(names) and np.isfinite(rows).all()
            lines[L] = dict(names=names, rows=rows,
                            index={t: i for i, t in enumerate(names)})
    queries = sorted(set.intersection(*(set(lines[L]["names"]) for L in HOLDOUTS)))
    assert len(queries) == 638
    with np.load(LEGACY_PDS, allow_pickle=True) as z:
        legacy = {}
        for L in HOLDOUTS:
            assert z[L + "_noexcl"].shape == (638,)
            legacy[L] = float(z[L + "_noexcl"].mean())
    return lines, queries, pos, gidx, legacy


def summarize(scores, top, ties):
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return dict(mean=float(np.mean(scores)), subset_mean_q025=float(lo),
                subset_mean_q975=float(hi),
                p_mean_ge_0_92=float(np.mean(scores >= 0.92)),
                unique_top_fraction=float(np.mean(top)),
                correct_similarity_tie_fraction=float(np.mean(ties)))


def run(reps, seed):
    synthetic_tests()
    lines, queries, pos, gidx, legacy = load_compact()
    n = len(queries)
    rng = np.random.default_rng(seed)
    # Identical draws across holdouts and policies for paired comparisons.
    subsets = {m: [rng.choice(n, size=m, replace=False) for _ in range(reps)]
               for m in (50, 100)}
    summaries = []
    for L in HOLDOUTS:
        srcs = [s for s in HOLDOUTS if s != L]
        # Only source responses are used to construct features, before any draw.
        X = build_features(lines, srcs, queries, pos, "mix")
        ref = lines[L]
        balanced_rows = ref["rows"][[ref["index"][t] for t in queries]]
        Y = build_targets(ref, queries, pos)
        assert np.all(np.linalg.norm(X, axis=1) > 1e-9)
        assert np.all(np.linalg.norm(Y, axis=1) > 1e-9)
        for policy in ("self", "none"):
            R = FastRetrieval(balanced_rows, queries, pos,
                              gidx if policy == "self" else {})
            # Raw-response float32 normalization matches FastRetrieval exactly;
            # build_targets uses float64 norms before casting to float32.
            np.testing.assert_allclose(R.Rn, Y, rtol=2e-6, atol=2e-7)
            S = similarity_matrix(R, X, queries, policy)
            full_pds, full_top, full_ties = rank_scores(S)
            np.testing.assert_array_equal(full_pds, R.score_pos(X, queries))
            all_indices = np.arange(n)
            np.testing.assert_array_equal(subset_scores(S, all_indices)[0], full_pds)
            # Also verify real small pools against direct retrieval, independently
            # of the cached similarity slicing (no unselected response participates).
            for m in (50, 100):
                idx = subsets[m][0]
                direct = FastRetrieval(balanced_rows[idx], [queries[i] for i in idx],
                                       pos, gidx if policy == "self" else {})
                np.testing.assert_array_equal(
                    subset_scores(S, idx)[0],
                    direct.score_pos(X[idx], [queries[i] for i in idx]))
            meta = dict(dataset="Nadig", holdout=L, policy=policy, seed=seed,
                        balanced_query_count=n, original_line_reference_count=len(ref["names"]),
                        measured_gene_count=6118, evaluation_gene_count=len(pos),
                        queries_with_masked_self_coordinate=sum(
                            gidx.get(t) in R.local for t in queries),
                        source_count=len(srcs),
                        legacy_noexcl_original_line_pool_mean=legacy[L],
                        threshold_status="posthoc_not_externally_validated",
                        quantile_status="subset_distribution_not_confidence_interval",
                        responses_path=str(RESPONSES), axis_path=str(AXIS),
                        mask_path=str(MASK), legacy_pds_only_path=str(LEGACY_PDS))
            baseline = dict(meta, reference_scope="balanced638_reference_baseline",
                            m=n, n_subsets=1, chance_mean=0.5 + 1.0 / (2 * n),
                            **summarize([full_pds.mean()], [full_top.mean()], [full_ties.mean()]))
            summaries.append(baseline)
            print(f"{L:7s} {policy:4s} balanced638 mean={full_pds.mean():.6f} "
                  f"unique_top={full_top.mean():.6f}", flush=True)
            for m in (50, 100):
                means = np.empty(reps)
                tops = np.empty(reps)
                tied = np.empty(reps)
                for b, idx in enumerate(subsets[m]):
                    pds, top, ties = subset_scores(S, idx)
                    means[b], tops[b], tied[b] = pds.mean(), top.mean(), ties.mean()
                row = dict(meta, reference_scope="small_reference_pool", m=m,
                           n_subsets=reps, chance_mean=0.5 + 1.0 / (2 * m),
                           **summarize(means, tops, tied))
                summaries.append(row)
                print(f"  m={m:3d} mean={row['mean']:.6f} "
                      f"subset_q=[{row['subset_mean_q025']:.6f},"
                      f"{row['subset_mean_q975']:.6f}] "
                      f"P(mean>=0.92)={row['p_mean_ge_0_92']:.4f} "
                      f"unique_top={row['unique_top_fraction']:.6f} "
                      f"tie={row['correct_similarity_tie_fraction']:.6f}", flush=True)
    assert OUTPUT.parent.is_dir()
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"All balanced and direct-subset retrieval assertions passed; saved {OUTPUT}",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 1000 <= args.reps <= 5000 or args.seed < 0:
        parser.error("--reps must be 1000..5000 and --seed must be nonnegative")
    run(args.reps, args.seed)
