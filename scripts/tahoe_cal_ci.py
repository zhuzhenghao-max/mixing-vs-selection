"""Bootstrap confidence intervals for the Tahoe calibration deltas.

Without flags, preserve the original row-pooled calculation, RNG sequence,
CSV columns and positional input/output interface. That calculation resamples
all delta rows, including repeated conditions from existing splits, and uses
independent within-line row resampling for the fixed-line panel mean.

With --sensitivity, the query unit is a unique condition ID, not a scored row:
average duplicate (split, cond) scores first, then average the available split
scores equally within each condition. Bootstrap these unique conditions 10000
times. Separately bootstrap unique conditions within each existing split at
alpha = 0.05 / n_splits (two-sided percentile intervals), then take the minimum
lower and maximum upper endpoint. This Bonferroni split-CI envelope is only a
conservative sensitivity range over existing fits, NOT uncertainty from new
splits, retraining, or inference to a population of cell lines. Multiplicity
adjustment is over splits within an output row, not over lines or budgets.

ALL always weights the fixed, complete lines equally. If condition supports
align, average the line scores for each condition before shared resampling.
Otherwise resample the UNION of observed condition IDs, sharing each ID's
multinomial count across lines, normalize within each line's observed support,
and average line means equally. This preserves shared-ID dependence without
restricting to a potentially tiny intersection or imputing missing scores.
Reject a draw only if it omits an entire line; report the rejection count.
Use the same fixed-line/shared-condition procedure separately in each split.
All query inference is conditional on the observed line-specific query
supports and already fitted models. Missing split scores are not imputed.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPS = 10000
SEED = 0
KS = (20, 50, 100, 200)
STRATS = ("kmeans", "random")


def read(path):
    out = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["holdout"], int(r["k"]), r["strategy"])].append(
                float(r["delta"]))
    return out


def boot_mean(v, rng):
    idx = rng.integers(0, len(v), (REPS, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5))


def pooled_rows(per):
    """Legacy calculation; retain its row order and RNG consumption exactly."""
    lines = sorted({k[0] for k in per})
    complete = [L for L in lines
                if all(per.get((L, k, s)) for k in KS for s in STRATS)]
    if len(complete) < len(lines):
        print("warning: incomplete lines excluded from the cross-line mean:",
              [L for L in lines if L not in complete])
    rng = np.random.default_rng(SEED)
    rows = []
    for L in lines:
        for k in KS:
            for s in STRATS:
                v = np.asarray(per.get((L, k, s), []))
                if not len(v):
                    continue
                m, lo, hi = boot_mean(v, rng)
                rows.append(dict(line=L, k=k, strategy=s, n=len(v),
                                 mean=m, lo=lo, hi=hi))
    for k in KS:
        for s in STRATS:
            arrs = [np.asarray(per[(L, k, s)]) for L in complete]
            if not arrs:
                continue
            m = float(np.mean([a.mean() for a in arrs]))
            means = np.empty(REPS)
            for b in range(REPS):
                means[b] = np.mean(
                    [a[rng.integers(0, len(a), len(a))].mean()
                     for a in arrs])
            rows.append(dict(line="ALL", k=k, strategy=s,
                             n=sum(len(a) for a in arrs), mean=m,
                             lo=float(np.percentile(means, 2.5)),
                             hi=float(np.percentile(means, 97.5))))
    return rows


def query_scores(records):
    """Collapse duplicate split/query rows before equal-weight split averaging."""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts = defaultdict(int)
    for r in records:
        key = (r["holdout"], int(r["k"]), r["strategy"])
        value = float(r["delta"])
        if not r["cond"] or not r["split"] or not np.isfinite(value):
            raise ValueError("Sensitivity requires nonempty cond/split and finite delta")
        raw[key][r["split"]][r["cond"]].append(value)
        counts[key] += 1
    per = {key: {sp: {cond: float(np.mean(v)) for cond, v in queries.items()}
                 for sp, queries in splits.items()}
           for key, splits in raw.items()}
    return per, counts


def average_splits(splits):
    by_cond = defaultdict(list)
    for queries in splits.values():
        for cond, value in queries.items():
            by_cond[cond].append(value)
    return {cond: float(np.mean(v)) for cond, v in by_cond.items()}


def shared_query_boot(line_scores, rng, alpha=0.05, reps=REPS):
    """Ordinary query bootstrap with fixed equal line weights and shared IDs."""
    if not line_scores or any(not scores for scores in line_scores):
        raise ValueError("Every fixed panel line must have query scores")
    supports = [set(scores) for scores in line_scores]
    union = sorted(set.union(*supports))
    common = set.intersection(*supports)
    mean = float(np.mean([np.mean(list(scores.values()))
                          for scores in line_scores]))
    means = np.empty(reps)
    rejected = 0
    if all(support == supports[0] for support in supports):
        # Averaging each shared query first exactly preserves cross-line pairing.
        values = np.mean([[scores[c] for c in union]
                          for scores in line_scores], axis=0)
        for start in range(0, reps, 256):
            size = min(256, reps - start)
            idx = rng.integers(0, len(union), (size, len(union)))
            means[start:start + size] = values[idx].mean(axis=1)
        scope = ("unique_condition" if len(line_scores) == 1
                 else "shared_conditions_fixed_equal_line_mean")
    else:
        # A shared union count acts on every observed occurrence of that ID.
        # Ratio normalization retains each line's original observed support.
        present = np.asarray([[c in scores for scores in line_scores]
                              for c in union], dtype=float)
        values = np.asarray([[scores.get(c, 0.0) for scores in line_scores]
                             for c in union])
        probabilities = np.full(len(union), 1.0 / len(union))
        done = 0
        while done < reps:
            size = min(256, reps - done)
            weights = rng.multinomial(len(union), probabilities, size=size)
            denominators = weights @ present
            valid = np.all(denominators > 0, axis=1)
            rejected += int(np.count_nonzero(~valid))
            draws = ((weights[valid] @ values) / denominators[valid]).mean(axis=1)
            means[done:done + len(draws)] = draws
            done += len(draws)
            if rejected > 100 * reps:
                raise ValueError("Too many union draws omit an entire fixed line")
        scope = "shared_union_weights_line_support_normalized_fixed_equal_line_mean"
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(mean=mean, lo=float(lo), hi=float(hi),
                n_unique_cond=len(union), n_common_cond=len(common),
                n_line_cond=sum(map(len, supports)), n_rejected=rejected,
                resampling=scope)


def sensitivity_rows(pooled, per, counts, reps=REPS, seed=SEED):
    lines = sorted({key[0] for key in per})
    complete = [line for line in lines
                if all(per.get((line, k, s)) for k in KS for s in STRATS)]
    rng = np.random.default_rng(seed)
    rows = []
    for original in pooled:
        line, k, strategy = (original[name] for name in ("line", "k", "strategy"))
        panel = complete if line == "ALL" else [line]
        groups = [per[(name, k, strategy)] for name in panel]
        split_sets = [set(group) for group in groups]
        if any(splits != split_sets[0] for splits in split_sets):
            raise ValueError("ALL requires the same split labels in every fixed line")
        splits = sorted(split_sets[0])
        unique = shared_query_boot([average_splits(group) for group in groups],
                                   rng, reps=reps)
        alpha = 0.05 / len(splits)
        details = []
        for split in splits:
            result = shared_query_boot([group[split] for group in groups], rng,
                                       alpha=alpha, reps=reps)
            details.append(dict(split=split, **result))
        n_pairs = sum(len(queries) for group in groups
                      for queries in group.values())
        n_rows = sum(counts[(name, k, strategy)] for name in panel)
        assert n_rows == original["n"]
        row = dict(original)
        row.update(unique_mean=unique["mean"], unique_lo=unique["lo"],
                   unique_hi=unique["hi"],
                   split_mean_min=min(d["mean"] for d in details),
                   split_mean_max=max(d["mean"] for d in details),
                   envelope_lo=min(d["lo"] for d in details),
                   envelope_hi=max(d["hi"] for d in details),
                   n_unique_cond=unique["n_unique_cond"],
                   n_line_cond=unique["n_line_cond"],
                   n_common_cond=unique["n_common_cond"],
                   n_cond_split=n_pairs, duplicate_cond_split_rows=n_rows - n_pairs,
                   n_splits=len(splits), n_lines=len(panel),
                   bonferroni_alpha=alpha, bootstrap_reps=reps, seed=seed,
                   unique_rejected_draws=unique["n_rejected"],
                   panel_lines=";".join(panel), query_resampling=unique["resampling"],
                   scope="observed_queries_existing_fits_fixed_lines_no_retraining",
                   split_details=json.dumps(details, sort_keys=True,
                                            separators=(",", ":")))
        rows.append(row)
    return rows


def sensitivity_self_test():
    """Small synthetic asserts; no files or global RNG state are changed."""
    records = [dict(holdout="L", k="20", strategy="kmeans", split=sp,
                    cond=cond, delta=value)
               for sp, cond, value in [("0", "a", 0.25), ("0", "b", 0.75),
                                       ("1", "a", 0.25), ("2", "a", 0.25)]]
    per, counts = query_scores(records)
    old = dict(line="L", k=20, strategy="kmeans", n=4,
               mean=0.375, lo=0.25, hi=0.75)
    rows = sensitivity_rows([old], per, counts, reps=4000, seed=11)
    assert rows == sensitivity_rows([old], per, counts, reps=4000, seed=11)
    row = rows[0]
    assert row["n_unique_cond"] == 2 and row["n_cond_split"] == 4
    assert row["unique_mean"] == 0.5 and row["n_splits"] == 3
    assert row["bonferroni_alpha"] == 0.05 / 3
    duplicated = records + [records[0]] * 9
    per2, counts2 = query_scores(duplicated)
    assert per2 == per
    row2 = sensitivity_rows([dict(old, n=13)], per2, counts2,
                            reps=4000, seed=11)[0]
    assert row2["duplicate_cond_split_rows"] == 9
    for name in ("unique_mean", "unique_lo", "unique_hi", "n_unique_cond",
                 "split_mean_min", "split_mean_max", "envelope_lo", "envelope_hi"):
        assert row2[name] == row[name]
    details = json.loads(row["split_details"])
    assert row["envelope_lo"] == min(d["lo"] for d in details)
    assert row["envelope_hi"] == max(d["hi"] for d in details)
    for scores, expected in [([{"a": 0.125, "b": 0.125}] * 2, 0.125),
                             ([{"a": 1.0, "b": -1.0},
                               {"a": -1.0, "b": 1.0}], 0.0)]:
        result = shared_query_boot(scores, np.random.default_rng(7), reps=4000)
        assert result["mean"] == result["lo"] == result["hi"] == expected
        assert result["n_unique_cond"] == 2 and result["n_line_cond"] == 4
    constant = {("L", k, s): {str(sp): {"a": 0.125, "b": 0.125}
                             for sp in range(3)} for k in KS for s in STRATS}
    constant_counts = {key: 6 for key in constant}
    constant_row = sensitivity_rows(
        [dict(old, line="ALL", n=6, mean=0.125, lo=0.125, hi=0.125)],
        constant, constant_counts, reps=4000)[0]
    assert (constant_row["unique_lo"] == constant_row["unique_hi"]
            == constant_row["envelope_lo"] == constant_row["envelope_hi"] == 0.125)
    partial = [{"a": 1.0, "b": 3.0}, {"b": 5.0, "c": 7.0}]
    result = shared_query_boot(partial, np.random.default_rng(7), reps=4000)
    assert result == shared_query_boot(partial, np.random.default_rng(7), reps=4000)
    assert result["mean"] == 4.0 and result["n_unique_cond"] == 3
    assert result["n_common_cond"] == 1 and result["n_rejected"] > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_path", nargs="?", default="outputs/deltasX.csv")
    parser.add_argument("out_path", nargs="?")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Write query/split sensitivity to a NEW CSV; preserve pooled CI")
    args = parser.parse_args()
    out_path = args.out_path or ("outputs/tahoe_calibration_ci_sensitivity.csv"
                                 if args.sensitivity else "outputs/tahoe_calibration_ci.csv")
    if args.sensitivity:
        output = Path(out_path)
        if not output.parent.is_dir():
            parser.error(f"Output parent directory does not exist: {output.parent}")
        if output.exists():
            parser.error(f"Sensitivity output must be a new file: {output}")
        sensitivity_self_test()
        print("sensitivity synthetic assertions passed (seed, duplicates, constants, shared IDs)")
    rows = pooled_rows(read(args.in_path))
    fields = ["line", "k", "strategy", "n", "mean", "lo", "hi"]
    if args.sensitivity:
        with open(args.in_path, newline="") as f:
            per, counts = query_scores(csv.DictReader(f))
        rows = sensitivity_rows(rows, per, counts)
        fields = list(rows[0]) if rows else fields
    with open(out_path, "x" if args.sensitivity else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{'line':12s} {'k':>3s} {'strategy':8s} {'n':>4s} "
          f"{'mean':>8s} {'95% CI':>17s}")
    for r in rows:
        print(f"{r['line']:12s} {r['k']:3d} {r['strategy']:8s} "
              f"{r['n']:4d} {r['mean']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")
    if args.sensitivity:
        for name, lo, hi in [("pooled", "lo", "hi"),
                             ("unique-query", "unique_lo", "unique_hi"),
                             ("split envelope", "envelope_lo", "envelope_hi")]:
            excluded = sum(r[lo] > 0 or r[hi] < 0 for r in rows)
            print(f"{name}: {excluded}/{len(rows)} intervals exclude zero")
        for r in rows:
            if r["line"] == "ALL":
                print(f"ALL k={r['k']} {r['strategy']}: unique={r['unique_mean']:+.6f} "
                      f"[{r['unique_lo']:+.6f}, {r['unique_hi']:+.6f}], "
                      f"split means=[{r['split_mean_min']:+.6f}, "
                      f"{r['split_mean_max']:+.6f}], "
                      f"envelope=[{r['envelope_lo']:+.6f}, {r['envelope_hi']:+.6f}]")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
