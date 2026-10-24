"""
InTAS diagnostics, part 3  (read-only: no TopTrust training, DP, or split changes).

Questions answered:
  1. Do vehicles shared between official splits keep the same label? (possible identity leakage)
  2. How is sender.distance_to_road_edge distributed per class?
  3. Does a transparent "map" feature work on its own: distance from the claimed sender position to the
     nearest position actually observed in Train (Train receivers + Train benign senders)?
     Also reports map coverage of every split WITHOUT using class labels.
  4. Honest within-Train generalisation: GroupKFold by sender_id with HistGB.

Test labels are not used unless --report_test is given. Requires intas_diagnostics.py and
intas_diagnostics2.py in the same folder.

Run:  python intas_diagnostics3.py
"""
import argparse

import numpy as np

from intas_diagnostics import ALL_NAMES, records_to_matrix, roc_auc, avg_precision
from intas_diagnostics2 import edge_column

PCTS = [1, 5, 25, 50, 75, 95, 99]


def pct_line(v):
    q = np.percentile(v, PCTS)
    return " ".join(f"P{p}={x:.2f}" for p, x in zip(PCTS, q))


def label_consistency(senders, labels):
    print("\n" + "=" * 88)
    print("1. LABEL CONSISTENCY of vehicles shared between splits")
    print("=" * 88)
    per = {}
    for split in senders:
        d = {}
        for s, y in zip(senders[split], labels[split]):
            d.setdefault(s, set()).add(int(y))
        per[split] = d
    names = list(senders)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = names[i], names[j]
            shared = set(per[a]) & set(per[b])
            same = sum(per[a][s] == per[b][s] for s in shared)
            mal_a = sum(per[a][s] == {1} for s in shared)
            mal_b = sum(per[b][s] == {1} for s in shared)
            print(f"{a} & {b}: shared={len(shared)}  identical label set={same}  "
                  f"malicious in {a}={mal_a}, in {b}={mal_b}")


def edge_distribution(X, y, edge, split):
    ok = np.isfinite(edge)
    print(f"\n[{split}] distance_to_road_edge: finite={ok.sum():,}/{len(edge):,}")
    for name, m in (("benign", y == 0), ("malicious", y == 1)):
        v = edge[ok & m]
        if v.size:
            print(f"  {name:<10} n={v.size:>6,} frac<0={np.mean(v < 0):.3f}  {pct_line(v)}")
    print(f"  uni-AUC={roc_auc(np.nan_to_num(edge), y):.4f}   |value| uni-AUC={roc_auc(np.nan_to_num(np.abs(edge)), y):.4f}")


def map_feature(data, report_test):
    from scipy.spatial import cKDTree
    print("\n" + "=" * 88)
    print("3. MAP FEATURE: distance (m) from claimed sender position to nearest Train-observed position")
    print("=" * 88)
    ix = {n: i for i, n in enumerate(ALL_NAMES)}
    Xtr, ytr = data["Train"][0], data["Train"][1]
    ref = np.vstack([Xtr[:, [ix["r_x"], ix["r_y"]]],
                     Xtr[ytr == 0][:, [ix["s_x"], ix["s_y"]]]])
    tree = cKDTree(ref)
    print(f"reference points: {len(ref):,} (Train receivers + Train benign senders)")
    for split in ("Validation", "Test"):
        X, y = data[split][0], data[split][1]
        d, _ = tree.query(X[:, [ix["s_x"], ix["s_y"]]], k=1)
        print(f"\n[{split}] coverage (all records, labels not used): {pct_line(d)}")
        print(f"   share farther than 25 m: {np.mean(d > 25):.3f} | 50 m: {np.mean(d > 50):.3f} | 100 m: {np.mean(d > 100):.3f}")
        if split == "Validation" or report_test:
            for name, m in (("benign", y == 0), ("malicious", y == 1)):
                print(f"   {name:<10} {pct_line(d[m])}")
            print(f"   AUC of this single feature: {roc_auc(d, y):.4f}  AP={avg_precision(d, y):.4f}")


def grouped_cv(data):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    print("\n" + "=" * 88)
    print("4. GroupKFold(5) by sender_id inside Train (honest generalisation to unseen vehicles)")
    print("=" * 88)
    X, y, g = data["Train"]
    ix = {n: i for i, n in enumerate(ALL_NAMES)}
    sets = {"sender position (s_x,s_y)": [ix["s_x"], ix["s_y"]],
            "all 14 GNN inputs": [ix[n] for n in ALL_NAMES[:14]]}
    for name, cols in sets.items():
        oof = np.zeros(len(y))
        for tr_i, te_i in GroupKFold(n_splits=5).split(X, y, g):
            m = HistGradientBoostingClassifier(random_state=0).fit(X[tr_i][:, cols], y[tr_i])
            oof[te_i] = m.predict_proba(X[te_i][:, cols])[:, 1]
        print(f"  {name:<30} out-of-fold AUC={roc_auc(oof, y):.4f}  AP={avg_precision(oof, y):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_test", action="store_true")
    a = ap.parse_args()

    from spliteDataset import load_dataset
    tr, va, te = load_dataset()
    raw = {"Train": tr, "Validation": va, "Test": te}

    data, senders, labels = {}, {}, {}
    for name, recs in raw.items():
        X, y, s, bad = records_to_matrix(recs, name)
        if bad:
            print(f"[WARN] {name}: {sum(bad.values())} records skipped")
        data[name] = (X, y, s)
        senders[name], labels[name] = s, y

    label_consistency(senders, labels)

    print("\n" + "=" * 88)
    print("2. distance_to_road_edge by class")
    print("=" * 88)
    for name in ("Train", "Validation") + (("Test",) if a.report_test else ()):
        recs = raw[name]
        if len(recs) == len(data[name][1]):
            edge_distribution(data[name][0], data[name][1], edge_column(recs), name)

    map_feature(data, a.report_test)
    grouped_cv(data)


if __name__ == "__main__":
    main()
