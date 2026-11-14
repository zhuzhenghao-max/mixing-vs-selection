"""Selection audit for the cross-line mixing benchmark.

Quantifies how per-target candidate selection and conditional subsets inflate
mean PDS: oracle per-target max vs best fixed rule, a permutation null, and a
split-half protocol that chooses one rule on half the targets and scores it on
the other half.  Also decomposes the gate gain into composition and
within-subset parts, and audits post-hoc budget selection on Tahoe-100M.

Outputs
  outputs/tables/tableS11_selection_audit.csv/.md
  outputs/selection_audit_kcurve.csv
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TAB = OUT / "tables"

MIX_A = ["equal", "equal_raw", "equal_center", "top2_selfw"]
MIX_B = MIX_A + ["simT0.05", "simT0.1", "simT0.2", "simT0.5", "simT1"]
SIM_B = ["equal"] + ["simT0.05", "simT0.1", "simT0.2", "simT0.5", "simT1"]
KCURVE_A = ["equal", "equal_raw", "top2_selfw"]


def load_long(path, tier):
    d = np.load(path, allow_pickle=True)
    return pd.DataFrame({
        "tier": tier,
        "holdout": d["holdout"].astype(str),
        "method": d["method"].astype(str),
        "target": d["target"].astype(str),
        "pds": d["pds_raw"].astype(float),
    })


def score_matrix(df, holdout, pool):
    sub = df[(df.holdout == holdout) & df.method.isin(pool)]
    S = sub.pivot(index="target", columns="method", values="pds")
    keep = [m for m in pool if m in S.columns]
    S = S[keep].dropna()
    if "equal" in S.columns:
        S = S[["equal"] + [c for c in S.columns if c != "equal"]]
    return S


def permute_rows(rng, X):
    try:
        return rng.permuted(X, axis=1)
    except AttributeError:
        Y = X.copy()
        for i in range(Y.shape[0]):
            rng.shuffle(Y[i])
        return Y


def null_inflation(X, perms, rng):
    vals = np.empty(perms)
    for i in range(perms):
        P = permute_rows(rng, X)
        vals[i] = P.max(1).mean() - P.mean(0).max()
    return vals


def split_honest(X, reps, rng):
    n = X.shape[0]
    vals = np.empty(reps)
    gains = np.empty(reps)
    for i in range(reps):
        idx = rng.permutation(n)
        h = n // 2
        A, B = idx[:h], idx[h:]
        m = int(np.argmax(X[A].mean(0)))
        vals[i] = X[B, m].mean()
        gains[i] = vals[i] - X[B, 0].mean()
    return float(vals.mean()), float(gains.mean())


def bootstrap_inflation(X, reps, rng):
    n = X.shape[0]
    vals = np.empty(reps)
    for i in range(reps):
        Xb = X[rng.integers(0, n, n)]
        vals[i] = Xb.max(1).mean() - Xb.mean(0).max()
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def audit(S, reps=200, perms=200, boots=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = S.to_numpy()
    naive = float(X.max(1).mean())
    fixed = X.mean(0)
    best_fixed = float(fixed.max())
    best_method = str(S.columns[int(fixed.argmax())])
    inflation = naive - best_fixed
    null = null_inflation(X, perms, rng)
    honest_mean, honest_gain = split_honest(X, reps, rng)
    ci_lo, ci_hi = bootstrap_inflation(X, boots, rng)
    return dict(
        k=S.shape[1], n=S.shape[0],
        equal_mean=float(X[:, 0].mean()),
        naive=naive, best_fixed=best_fixed, best_method=best_method,
        inflation=inflation, inf_ci_lo=ci_lo, inf_ci_hi=ci_hi,
        null_mean=float(null.mean()),
        null_q95=float(np.quantile(null, 0.95)),
        excess=inflation - float(null.mean()),
        honest_mean=honest_mean, honest_gain=honest_gain,
        optimism=best_fixed - honest_mean,
        oracle_gap=naive - honest_mean,
        naive_vs_equal=naive - float(X[:, 0].mean()),
    )


def gate_audit(df, holdout):
    sub = df[df.holdout == holdout]
    eq = sub[sub.method == "equal"].set_index("target")["pds"]
    ga = sub[sub.method == "gated_equal"].set_index("target")["pds"]
    common = eq.index.intersection(ga.index)
    eq_all = float(eq.mean())
    eq_gate = float(eq.loc[common].mean())
    gated_gate = float(ga.loc[common].mean())
    return dict(
        k=np.nan, n=len(common), equal_mean=eq_all,
        naive=gated_gate, best_fixed=np.nan, best_method="gated_equal",
        inflation=gated_gate - eq_all, inf_ci_lo=np.nan, inf_ci_hi=np.nan,
        null_mean=np.nan, null_q95=np.nan, excess=np.nan,
        honest_mean=np.nan, honest_gain=np.nan, optimism=np.nan,
        n_all=len(eq), n_gate=len(common),
        equal_gate=eq_gate,
        composition=eq_gate - eq_all,
        within=gated_gate - eq_gate,
    )


def tahoe_audit(path):
    d = pd.read_csv(path)
    d["cand"] = d["k"].astype(int).astype(str) + "_" + d["strategy"].astype(str)
    rows = []
    for (H, split), g in d.groupby(["holdout", "split"]):
        S = g.pivot_table(index="cond", columns="cand", values="delta",
                          aggfunc="mean").dropna()
        if S.shape[0] < 40 or S.shape[1] < 2:
            continue
        rng = np.random.default_rng(0)
        X = S.to_numpy()
        naive = float(X.max(1).mean())
        fixed = X.mean(0)
        best_fixed = float(fixed.max())
        inflation = naive - best_fixed
        null = null_inflation(X, 200, rng)
        # no equal column here: baseline delta is 0
        n = X.shape[0]
        vals = np.empty(200)
        for i in range(200):
            idx = rng.permutation(n)
            h = n // 2
            A, B = idx[:h], idx[h:]
            m = int(np.argmax(X[A].mean(0)))
            vals[i] = X[B, m].mean()
        rows.append(dict(
            holdout=H, split=split, n=n, k=X.shape[1],
            naive=naive, best_fixed=best_fixed,
            best_method=str(S.columns[int(fixed.argmax())]),
            inflation=inflation, null_mean=float(null.mean()),
            excess=inflation - float(null.mean()),
            honest_mean=float(vals.mean()),
        ))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    agg = t.groupby("holdout").agg(
        n=("n", "mean"), k=("k", "max"),
        naive=("naive", "mean"), best_fixed=("best_fixed", "mean"),
        inflation=("inflation", "mean"), null_mean=("null_mean", "mean"),
        excess=("excess", "mean"), honest_mean=("honest_mean", "mean"),
    ).reset_index()
    agg["tier"] = "Tahoe"
    agg["method"] = "budget_rule"
    return agg


def main():
    dfa = load_long(OUT / "bench_transfer_nadig_v2.npz", "A")
    dfb = load_long(OUT / "bench_transfer_v2.npz", "B")
    df = pd.concat([dfa, dfb], ignore_index=True)

    rows = []
    for tier, frame in (("A", dfa), ("B", dfb)):
        for H in sorted(frame.holdout.unique()):
            pool = MIX_A if tier == "A" else MIX_B
            S = score_matrix(frame, H, pool)
            r = audit(S)
            r.update(tier=tier, holdout=H, kind="variants",
                     pool="mix4" if tier == "A" else "mix9")
            rows.append(r)

            g = gate_audit(frame, H)
            g.update(tier=tier, holdout=H, kind="gate", pool="gated_equal")
            rows.append(g)

            if tier == "A":
                singles = sorted(frame[frame.holdout == H].method.unique())
                singles = [m for m in singles if m.startswith("single_")]
                Ss = score_matrix(frame, H, singles)
                if Ss.shape[0] >= 50:
                    r = audit(Ss)
                    r.update(tier=tier, holdout=H, kind="singles",
                             pool="source_lines")
                    rows.append(r)

    t = pd.DataFrame(rows)
    cols = ["tier", "holdout", "kind", "pool", "k", "n", "equal_mean",
            "naive", "best_fixed", "best_method", "inflation",
            "inf_ci_lo", "inf_ci_hi", "null_mean", "null_q95", "excess",
            "honest_mean", "honest_gain", "optimism", "oracle_gap",
            "naive_vs_equal",
            "n_all", "n_gate", "equal_gate", "composition", "within"]
    for c in cols:
        if c not in t.columns:
            t[c] = np.nan
    t = t[cols]

    th = tahoe_audit(OUT / "deltasX.csv")
    if not th.empty:
        for c in cols:
            if c not in th.columns:
                th[c] = np.nan
        th["kind"] = "tahoe"
        th["pool"] = "budget_rules"
        th["equal_mean"] = 0.0
        th["honest_gain"] = th["honest_mean"]
        th["optimism"] = th["best_fixed"] - th["honest_mean"]
        th["oracle_gap"] = th["naive"] - th["honest_mean"]
        th["naive_vs_equal"] = th["naive"]
        th["inf_ci_lo"] = np.nan
        th["inf_ci_hi"] = np.nan
        th["null_q95"] = np.nan
        th["best_method"] = th["method"]
        th["n_all"] = np.nan
        th["n_gate"] = np.nan
        th["equal_gate"] = np.nan
        th["composition"] = np.nan
        th["within"] = np.nan
        t = pd.concat([t, th[cols]], ignore_index=True)

    TAB.mkdir(exist_ok=True)
    t.to_csv(TAB / "tableS11_selection_audit.csv", index=False, float_format="%.6f")

    show = t.copy()
    for c in ["equal_mean", "naive", "best_fixed", "inflation", "inf_ci_lo",
              "inf_ci_hi", "null_mean", "null_q95", "excess", "honest_mean",
              "honest_gain", "optimism", "oracle_gap", "naive_vs_equal",
              "equal_gate", "composition", "within"]:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.4f}")
    show["k"] = show["k"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    show["n"] = show["n"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    show.to_csv(TAB / "tableS11_selection_audit.md", index=False, sep="|")

    # k-curve: pool size vs oracle inflation and honest split gain
    krows = []
    for tier, frame, pool in (("A", dfa, KCURVE_A), ("B", dfb, SIM_B)):
        for H in sorted(frame.holdout.unique()):
            S = score_matrix(frame, H, pool)
            extras = [c for c in pool if c != "equal"]
            for kk in range(1, len(extras) + 1):
                for combo in itertools.combinations(extras, kk):
                    cols = ["equal"] + list(combo)
                    X = S[cols].to_numpy()
                    rng = np.random.default_rng(0)
                    null = null_inflation(X, 50, rng)
                    vals = np.empty(50)
                    n = X.shape[0]
                    for i in range(50):
                        idx = rng.permutation(n)
                        h = n // 2
                        A, B = idx[:h], idx[h:]
                        m = int(np.argmax(X[A].mean(0)))
                        vals[i] = X[B, m].mean() - X[B, 0].mean()
                    krows.append(dict(
                        tier=tier, holdout=H, k=kk + 1,
                        inflation=float(X.max(1).mean() - X.mean(0).max()),
                        null_mean=float(null.mean()),
                        honest_gain=float(vals.mean()),
                        naive_vs_equal=float(X.max(1).mean() - X[:, 0].mean()),
                        best_fixed_vs_equal=float(
                            X.mean(0).max() - X[:, 0].mean()),
                    ))
    kc = pd.DataFrame(krows)
    kc.to_csv(OUT / "selection_audit_kcurve.csv", index=False,
              float_format="%.6f")
    print(show.to_string(index=False))
    print("\nkcurve means")
    print(kc.groupby(["tier", "k"]).agg(
        inflation=("inflation", "mean"),
        null_mean=("null_mean", "mean"),
        honest_gain=("honest_gain", "mean"),
        naive_vs_equal=("naive_vs_equal", "mean"),
        best_fixed_vs_equal=("best_fixed_vs_equal", "mean"),
        n=("holdout", "size")).to_string())


if __name__ == "__main__":
    main()
