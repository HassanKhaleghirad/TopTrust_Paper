"""
InTAS diagnostics for TopTrust  (read-only: no training code, no DP code, no split changes).

Step 2  Audit of Train / Validation / Test:
        per-feature stats (mean, std, percentiles, invalid values), overall and per class,
        univariate ROC-AUC per feature, sender-label purity, and scale check of the
        sender-receiver offset versus global coordinate scale.
Step 3  Independent baselines on the SAME official split:
        Logistic Regression (and optionally RandomForest / HistGradientBoosting).
        Scaler is fitted on Train only. Test is NOT touched unless --report_test is given.

Feature extraction mirrors toptrust_revised3._intas_node_features exactly
(x, y, z, speed, accel, sin(hed), cos(hed) for sender and receiver) so that
baseline "A" sees the same 14 numbers the GNN sees. Baseline "B" additionally
gets derived relative features (offset, distance, speed/heading differences, delay).

Run (from the folder that contains spliteDataset.py and dataset/):
    python intas_diagnostics.py
    python intas_diagnostics.py --trees
    python intas_diagnostics.py --report_test      # only at the very end
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

BASE_NAMES = ["x", "y", "z", "spd", "acl", "sin_hed", "cos_hed"]
GNN_NAMES = [f"s_{n}" for n in BASE_NAMES] + [f"r_{n}" for n in BASE_NAMES]
DERIVED_NAMES = ["dx", "dy", "dz", "dist_xy", "dist_xyz", "d_spd", "d_acl",
                 "hed_diff_cos", "hed_diff_sin", "delay_s"]
ALL_NAMES = GNN_NAMES + DERIVED_NAMES
PCTS = [1, 5, 25, 50, 75, 95, 99]


# ---------------------------------------------------------------------------------------------
# Feature extraction (same arithmetic as toptrust_revised3)
# ---------------------------------------------------------------------------------------------
def _xyz(value):
    if not isinstance(value, str):
        raise ValueError("pos is not a string")
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("pos does not have 3 coordinates")
    return [float(p.strip()) for p in parts]


def _node(obj):
    if not isinstance(obj, dict):
        raise ValueError("node is not an object")
    x, y, z = _xyz(obj.get("pos"))
    hed = math.radians(float(obj["hed"]))
    return [x, y, z, float(obj["spd"]), float(obj["acl"]), math.sin(hed), math.cos(hed)], float(obj["hed"])


def records_to_matrix(records, split_name):
    rows, labels, senders, bad = [], [], [], {}
    for rec in records:
        try:
            s, s_hed = _node(rec["sender"])
            r, r_hed = _node(rec["receiver"])
            y = int(rec["attacker"])
            if y not in (0, 1):
                raise ValueError("label")
            delay = max(0.0, (int(rec["rcvTime"]) - int(rec["sendTime"])) / 1e9)
            dh = math.radians(s_hed - r_hed)
            derived = [s[0] - r[0], s[1] - r[1], s[2] - r[2],
                       math.hypot(s[0] - r[0], s[1] - r[1]),
                       math.sqrt((s[0] - r[0]) ** 2 + (s[1] - r[1]) ** 2 + (s[2] - r[2]) ** 2),
                       s[3] - r[3], s[4] - r[4], math.cos(dh), math.sin(dh), delay]
            row = s + r + derived
            rows.append(row)
            labels.append(y)
            senders.append(str(rec["sender_id"]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            key = f"{type(exc).__name__}: {exc}"[:80]
            bad[key] = bad.get(key, 0) + 1
    X = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(ALL_NAMES)))
    return X, np.asarray(labels, dtype=int), np.asarray(senders), bad


# ---------------------------------------------------------------------------------------------
# Metrics without extra dependencies on torch
# ---------------------------------------------------------------------------------------------
def roc_auc(score, y):
    y = np.asarray(y)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def avg_precision(score, y):
    from sklearn.metrics import average_precision_score
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


# ---------------------------------------------------------------------------------------------
# Step 2: audit
# ---------------------------------------------------------------------------------------------
def describe(col):
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan, **{f"P{p}": np.nan for p in PCTS})
    out = dict(n=int(finite.size), mean=float(finite.mean()), std=float(finite.std()),
               min=float(finite.min()), max=float(finite.max()))
    for p, v in zip(PCTS, np.percentile(finite, PCTS)):
        out[f"P{p}"] = float(v)
    return out


def audit_split(name, X, y, senders, bad, writer):
    print("\n" + "=" * 88)
    print(f"AUDIT {name}: records={len(y):,} benign={(y == 0).sum():,} malicious={(y == 1).sum():,} "
          f"malicious_share={(y == 1).mean() if len(y) else float('nan'):.4f}")
    print("=" * 88)
    if bad:
        print("Records skipped while parsing:")
        for k, v in sorted(bad.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>8,}  {k}")
    nonfinite = int((~np.isfinite(X)).sum())
    print(f"Non-finite feature values: {nonfinite:,}")

    # sender label purity
    ids = np.unique(senders)
    pure0 = pure1 = mixed = 0
    for sid in ids:
        m = y[senders == sid]
        if m.min() == m.max():
            pure1 += int(m[0] == 1)
            pure0 += int(m[0] == 0)
        else:
            mixed += 1
    print(f"Senders: {len(ids):,} | pure-benign={pure0:,} pure-malicious={pure1:,} mixed-label={mixed:,}")

    print(f"\n{'feature':<14}{'class':<10}{'mean':>14}{'std':>14}{'P1':>14}{'P50':>14}{'P99':>14}  uniAUC")
    for j, fname in enumerate(ALL_NAMES):
        auc = roc_auc(np.nan_to_num(X[:, j]), y) if len(y) else float("nan")
        for cname, mask in (("all", np.ones(len(y), bool)), ("benign", y == 0), ("malicious", y == 1)):
            d = describe(X[mask, j])
            writer.writerow([name, fname, cname] + [d[k] for k in ("n", "mean", "std", "min", *(f"P{p}" for p in PCTS), "max")]
                            + [auc])
            if cname == "all":
                continue
            tail = f"  {auc:.4f}" if cname == "malicious" else ""
            print(f"{fname:<14}{cname:<10}{d['mean']:>14.4f}{d['std']:>14.4f}{d['P1']:>14.4f}"
                  f"{d['P50']:>14.4f}{d['P99']:>14.4f}{tail}")


def scale_check(Xtr):
    """Does the sender-receiver offset survive standardisation by the GLOBAL coordinate std?"""
    print("\n" + "=" * 88)
    print("SCALE CHECK (Train): what the GNN sees after global z-scoring of x, y")
    print("=" * 88)
    jd = ALL_NAMES.index("dist_xy")
    dist = Xtr[:, jd]
    print(f"global std of x (sender+receiver pooled) = {np.std(np.concatenate([Xtr[:, 0], Xtr[:, 7]])):.2f}")
    print(f"global std of y (sender+receiver pooled) = {np.std(np.concatenate([Xtr[:, 1], Xtr[:, 8]])):.2f}")
    print(f"sender-receiver distance_xy: median={np.median(dist):.2f} P5={np.percentile(dist, 5):.2f} "
          f"P95={np.percentile(dist, 95):.2f}")
    sx = np.std(np.concatenate([Xtr[:, 0], Xtr[:, 7]]))
    print(f"median distance / global x-std = {np.median(dist) / sx:.5f}  "
          "(if far below ~0.01, the offset is buried in z-scored absolute coordinates)")


# ---------------------------------------------------------------------------------------------
# Step 3: baselines
# ---------------------------------------------------------------------------------------------
def fit_eval(name, model, Xtr, ytr, evals):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)                      # fitted on Train only
    model.fit(sc.transform(Xtr), ytr)
    print(f"\n[{name}]")
    for split, (X, y) in evals.items():
        s = model.predict_proba(sc.transform(X))[:, 1]
        print(f"  {split:<11} ROC-AUC={roc_auc(s, y):.4f}  AP={avg_precision(s, y):.4f}  "
              f"(prevalence={y.mean():.4f})")
    return model


def run_baselines(data, use_trees, report_test):
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr = data["Train"][0], data["Train"][1]
    evals = {"Train": (Xtr, ytr), "Validation": (data["Validation"][0], data["Validation"][1])}
    if report_test:
        evals["Test"] = (data["Test"][0], data["Test"][1])

    def sub(X, cols):
        idx = [ALL_NAMES.index(c) for c in cols]
        return X[:, idx]

    print("\n" + "=" * 88)
    print("BASELINES (official split, scaler fitted on Train, "
          + ("Test included" if report_test else "Test NOT used") + ")")
    print("=" * 88)
    for label, cols in (("A: the 14 GNN inputs", GNN_NAMES), ("B: 14 + derived relative features", ALL_NAMES)):
        ev = {k: (sub(X, cols), y) for k, (X, y) in evals.items()}
        fit_eval(f"LogReg  {label}", LogisticRegression(max_iter=2000, class_weight="balanced"),
                 sub(Xtr, cols), ytr, ev)
        if use_trees:
            from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
            fit_eval(f"RandForest {label}",
                     RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                            class_weight="balanced", random_state=0),
                     sub(Xtr, cols), ytr, ev)
            fit_eval(f"HistGB {label}", HistGradientBoostingClassifier(random_state=0),
                     sub(Xtr, cols), ytr, ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", action="store_true", help="also run RandomForest and HistGradientBoosting")
    ap.add_argument("--report_test", action="store_true", help="include Test in baseline metrics (final report only)")
    ap.add_argument("--out", default="intas_feature_audit.csv")
    a = ap.parse_args()

    from spliteDataset import load_dataset
    tr, va, te = load_dataset()
    data = {}
    for name, recs in (("Train", tr), ("Validation", va), ("Test", te)):
        data[name] = records_to_matrix(recs, name)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "feature", "class", "n", "mean", "std", "min", *(f"P{p}" for p in PCTS), "max", "uni_auc"])
        for name in ("Train", "Validation", "Test"):
            audit_split(name, *data[name], w)
    print(f"\nFull table saved to {Path(a.out).resolve()}")

    scale_check(data["Train"][0])
    run_baselines({k: v[:2] for k, v in data.items()}, a.trees, a.report_test)


if __name__ == "__main__":
    main()