"""
InTAS diagnostics, part 2  (still read-only: no TopTrust training, DP, or split changes).

Why: tree models reach AUC ~0.98 on Validation with the 14 GNN inputs while Logistic Regression
is at ~0.51. This script checks WHERE that signal comes from and whether it can leak:

  1. Identity overlap of sender_id / receiver_id / sender_alias across official splits.
  2. HistGradientBoosting ablations by feature group
     (absolute position, relative features, speed/heading, optional distance_to_road_edge).
  3. A centralized MLP on the 14 GNN inputs (no federation, no graph) as a capacity check.

Test is not used unless --report_test is given. Requires intas_diagnostics.py next to this file.

Run:   python intas_diagnostics2.py
       python intas_diagnostics2.py --mlp
       python intas_diagnostics2.py --report_test     # final, once
"""
import argparse

import numpy as np

from intas_diagnostics import (ALL_NAMES, GNN_NAMES, records_to_matrix, roc_auc, avg_precision)

EDGE = "road_edge"   # sender.distance_to_road_edge (extra field, not used by toptrust_revised3)


def ids(records, key, nested=None):
    out = set()
    for r in records:
        v = r.get(nested, {}).get(key) if nested else r.get(key)
        if v is not None and str(v).strip() != "":
            out.add(str(v))
    return out


def identity_overlap(raw):
    print("\n" + "=" * 88)
    print("IDENTITY OVERLAP across official splits (counts of shared IDs)")
    print("=" * 88)
    names = ("Train", "Validation", "Test")
    for label, key in (("sender_id", "sender_id"), ("receiver_id", "receiver_id"), ("sender_alias", "sender_alias")):
        sets = {n: ids(raw[n], key) for n in names}
        print(f"{label}: " + ", ".join(f"{n}={len(sets[n]):,}" for n in names))
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = names[i], names[j]
                inter = sets[a] & sets[b]
                print(f"   {a} & {b}: {len(inter):,}" + (f"  e.g. {sorted(inter)[:4]}" if inter else ""))
    # a sender can also appear as a receiver of another split
    snd = {n: ids(raw[n], "sender_id") for n in names}
    rcv = {n: {f"veh_{x}" for x in ids(raw[n], "receiver_id")} for n in names}
    cross = [(a, b, len(snd[a] & rcv[b])) for a in names for b in names if a != b]
    print("sender_id(split A) as veh_<receiver_id>(split B): "
          + ", ".join(f"{a}->{b}={c}" for a, b, c in cross))


def edge_column(records):
    col = []
    for r in records:
        try:
            col.append(float(r["sender"]["distance_to_road_edge"]))
        except (KeyError, TypeError, ValueError):
            col.append(np.nan)
    return np.asarray(col)


def hgb_eval(name, cols, data, report_test):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr = data["Train"][0][:, cols], data["Train"][1]
    m = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    parts = []
    for split in ("Train", "Validation") + (("Test",) if report_test else ()):
        X, y = data[split][0][:, cols], data[split][1]
        s = m.predict_proba(X)[:, 1]
        parts.append(f"{split[:5]} AUC={roc_auc(s, y):.4f} AP={avg_precision(s, y):.4f}")
    print(f"  {name:<44} " + " | ".join(parts))


def ablations(data, report_test):
    print("\n" + "=" * 88)
    print("HistGB ABLATIONS (fit on Train; " + ("Test included" if report_test else "Test NOT used") + ")")
    print("=" * 88)
    ix = {n: i for i, n in enumerate(data["names"])}
    groups = [
        ("all 14 GNN inputs", GNN_NAMES),
        ("absolute positions only (s_x,s_y,r_x,r_y)", ["s_x", "s_y", "r_x", "r_y"]),
        ("sender position only (s_x,s_y)", ["s_x", "s_y"]),
        ("receiver position only (r_x,r_y)", ["r_x", "r_y"]),
        ("14 inputs WITHOUT absolute positions",
         [n for n in GNN_NAMES if n not in ("s_x", "s_y", "r_x", "r_y", "s_z", "r_z")]),
        ("sender speed/accel/heading only", ["s_spd", "s_acl", "s_sin_hed", "s_cos_hed"]),
        ("relative features only",
         ["dx", "dy", "dist_xy", "d_spd", "d_acl", "hed_diff_cos", "hed_diff_sin", "delay_s"]),
    ]
    if EDGE in ix:
        groups += [("all 14 + distance_to_road_edge (CAUTION)", GNN_NAMES + [EDGE]),
                   ("distance_to_road_edge only (CAUTION)", [EDGE])]
    for name, cols in groups:
        hgb_eval(name, [ix[c] for c in cols], data, report_test)


def mlp_check(data, report_test):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    ix = {n: i for i, n in enumerate(data["names"])}
    cols = [ix[c] for c in GNN_NAMES]
    sc = StandardScaler().fit(data["Train"][0][:, cols])          # Train only
    print("\n" + "=" * 88)
    print("CENTRALIZED MLP on the 14 GNN inputs (no graph, no federation)")
    print("=" * 88)
    for hidden in ((16,), (64, 64)):
        m = MLPClassifier(hidden_layer_sizes=hidden, early_stopping=True, validation_fraction=0.1,
                          max_iter=300, random_state=0).fit(sc.transform(data["Train"][0][:, cols]), data["Train"][1])
        parts = []
        for split in ("Train", "Validation") + (("Test",) if report_test else ()):
            X, y = data[split][0][:, cols], data[split][1]
            s = m.predict_proba(sc.transform(X))[:, 1]
            parts.append(f"{split[:5]} AUC={roc_auc(s, y):.4f} AP={avg_precision(s, y):.4f}")
        print(f"  MLP{hidden!s:<10} " + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--report_test", action="store_true")
    a = ap.parse_args()

    from spliteDataset import load_dataset
    tr, va, te = load_dataset()
    raw = {"Train": tr, "Validation": va, "Test": te}
    identity_overlap(raw)

    data = {}
    for name, recs in raw.items():
        X, y, s, bad = records_to_matrix(recs, name)
        if bad or len(y) != len(recs):
            print(f"[WARN] {name}: {sum(bad.values())} records skipped; distance_to_road_edge disabled")
            data["names"] = ALL_NAMES
        else:
            X = np.column_stack([X, edge_column(recs)])
            data.setdefault("names", ALL_NAMES + [EDGE])
        data[name] = (X, y)
    if len(data["names"]) != data["Train"][0].shape[1]:
        data["names"] = ALL_NAMES
        for n in ("Train", "Validation", "Test"):
            data[n] = (data[n][0][:, :len(ALL_NAMES)], data[n][1])
    ablations(data, a.report_test)
    if a.mlp:
        mlp_check(data, a.report_test)


if __name__ == "__main__":
    main()
