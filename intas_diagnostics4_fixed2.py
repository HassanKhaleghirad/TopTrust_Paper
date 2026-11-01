"""
InTAS diagnostics, part 4 (fixed)  -- read-only: no TopTrust training, DP, or split changes. Test labels are NOT used.

Question: which input features survive the spatial shift between Train and Test (Test lies ~1.8 km from anything
in Train)?  Two simulations of that shift are run INSIDE Train, plus the ordinary Train -> Validation score:

  1. Leave-one-band-out: Train is cut into --bands bands along the receiver coordinate (--axis). The model is
     trained on the other bands and tested on the held-out band. Rows are dropped when the RECEIVER or the SENDER
     lies within --buffer metres of the held-out band (sender positions can be falsified and are up to one
     communication range away from the receiver, so excluding only by receiver position leaks neighbours).
  2. One-sided extrapolation (closest analogue of Test): train on the lowest 50% of the axis, test on the top 20%
     (and the mirror image), with the real gap between them printed.

What changed w.r.t. the previous version
  * AUC from sklearn (no dependence on the argument order of a local roc_auc helper); average precision added.
  * Exclusion by BOTH receiver and sender coordinate; default buffer 300 m (was 100 m, only receiver-based).
  * The effective gap and the kept fraction are printed, so you can see how far the simulated shift is from 1.8 km.
  * Extrapolation test added (bands with data on both sides understate the Test shift).
  * Optional --drop_shared_senders removes training rows of vehicles that also appear in the held-out part.
  * Deterministic classifier (no random internal early-stopping split); CSV output; leakage warning for 'edge'.

Requires intas_diagnostics.py, intas_diagnostics2.py and spliteDataset.py next to this file.
Run:  python intas_diagnostics4_fixed.py [--axis y] [--bands 5] [--buffer 300] [--drop_shared_senders] [--csv out.csv]
"""
import argparse
import csv
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from intas_diagnostics import ALL_NAMES, GNN_NAMES, records_to_matrix
from intas_diagnostics2 import edge_column

REL = ["dx", "dy", "dist_xy", "d_spd", "d_acl", "hed_diff_cos", "hed_diff_sin", "delay_s"]
KIN = ["s_spd", "s_acl", "s_sin_hed", "s_cos_hed", "r_spd", "r_acl", "r_sin_hed", "r_cos_hed"]
NAMES = ALL_NAMES + ["edge"]
SETS = {
    "abs_pos": ["s_x", "s_y"],
    "edge": ["edge"],
    "rel": REL,
    "rel+kin": REL + KIN,
    "edge+rel": ["edge"] + REL,
    "edge+rel+kin": ["edge"] + REL + KIN,
    "all14+edge": GNN_NAMES + ["edge"],
}
MIN_TRAIN_ROWS = 500


def load(recs, name):
    X, y, s, bad = records_to_matrix(recs, name)
    if bad or len(y) != len(recs):
        raise SystemExit(f"{name}: records were skipped during parsing; alignment with the edge column is unsafe")
    edge = np.asarray(edge_column(recs), dtype=float)
    if edge.shape[0] != len(y):
        raise SystemExit(f"{name}: edge column length {edge.shape[0]} != {len(y)} records")
    X = np.asarray(X, dtype=float)
    if X.shape[1] != len(ALL_NAMES):
        raise SystemExit(f"{name}: matrix has {X.shape[1]} columns, ALL_NAMES has {len(ALL_NAMES)}")
    groups = None
    if s is not None:
        g = np.asarray(s)
        if g.ndim == 1 and len(g) == len(y):
            groups = g
    return np.column_stack([X, edge]), np.asarray(y).astype(int), groups


def fit_scores(Xa, ya, Xb, yb, cols):
    """Return (ROC-AUC, average precision); NaN when a split has one class or too few training rows."""
    if len(ya) < MIN_TRAIN_ROWS or len(set(ya.tolist())) < 2 or len(set(yb.tolist())) < 2:
        return float("nan"), float("nan")
    m = HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=0)
    p = m.fit(Xa[:, cols], ya).predict_proba(Xb[:, cols])[:, 1]
    return float(roc_auc_score(yb, p)), float(average_precision_score(yb, p))


def nan_stat(f, v):
    v = [x for x in v if not np.isnan(x)]
    return f(v) if v else float("nan")


def band_masks(pos, spos, lo, hi, buffer, groups, drop_shared):
    held = (pos >= lo) & (pos < hi)
    near = lambda v: (v >= lo - buffer) & (v < hi + buffer)
    keep = ~near(pos) & ~near(spos)                       # receiver AND sender must be outside the buffer
    if drop_shared and groups is not None:
        keep &= ~np.isin(groups, np.unique(groups[held]))
    d = lambda v: np.where(v < lo, lo - v, v - hi)
    gap = float(min(d(pos[keep]).min(), d(spos[keep]).min())) if keep.any() else float("nan")
    return keep, held, gap


def extrap_masks(pos, spos, side, q_train=0.5, q_test=0.8):
    """One-sided extrapolation: train on the near half of the axis, test on the far 20%."""
    if side == "hi":
        t_tr, t_te = np.quantile(pos, [q_train, q_test])
        return (pos < t_tr) & (spos < t_tr), pos >= t_te, float(t_te - t_tr)
    t_te, t_tr = np.quantile(pos, [1 - q_test, 1 - q_train])
    return (pos >= t_tr) & (spos >= t_tr), pos < t_te, float(t_tr - t_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["x", "y", "road"], default="road",
                    help="coordinate for the spatial cut; 'road' = first principal axis of the receiver positions "
                         "(use this when the road is not aligned with x or y)")
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--buffer", type=float, default=300.0,
                    help="metres excluded around the held-out band (>= communication range)")
    ap.add_argument("--drop_shared_senders", action="store_true",
                    help="also drop training rows of senders that appear in the held-out part (stricter than Test)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    from spliteDataset import load_dataset
    tr, va, _ = load_dataset()                            # the Test split is loaded by the helper but never used
    Xtr, ytr, gtr = load(tr, "Train")
    Xva, yva, _ = load(va, "Validation")
    ix = {n: i for i, n in enumerate(NAMES)}
    rx, ry_, sx, sy_ = (Xtr[:, ix[k]] for k in ("r_x", "r_y", "s_x", "s_y"))
    print(f"receiver x range: {rx.min():.0f}..{rx.max():.0f} (span {np.ptp(rx):.0f} m) | "
          f"y range: {ry_.min():.0f}..{ry_.max():.0f} (span {np.ptp(ry_):.0f} m)")
    if a.axis == "road":
        c = np.array([rx.mean(), ry_.mean()])
        d = np.linalg.svd(np.column_stack([rx - c[0], ry_ - c[1]]), full_matrices=False)[2][0]
        pos = (rx - c[0]) * d[0] + (ry_ - c[1]) * d[1]
        spos = (sx - c[0]) * d[0] + (sy_ - c[1]) * d[1]
        print(f"road axis direction (dx,dy)=({d[0]:.3f},{d[1]:.3f}); span along road {np.ptp(pos):.0f} m")
    else:
        pos, spos = (rx, sx) if a.axis == "x" else (ry_, sy_)
    # equal-WIDTH bands: equal-count bands collapse to a few metres when most rows share one stretch of road
    edges = np.linspace(pos.min(), pos.max(), a.bands + 1)
    edges[-1] += 1e-6
    if a.drop_shared_senders and gtr is None:
        print("[warn] sender ids unavailable from records_to_matrix; --drop_shared_senders ignored")

    print("=" * 96)
    print(f"SPATIAL SHIFT SIMULATION on Train (axis={a.axis}, {a.bands} bands, buffer={a.buffer:.0f} m, "
          f"range along cut: {pos.min():.0f}..{pos.max():.0f} m)")
    print("=" * 96)
    plans = []
    for b in range(a.bands):
        keep, held, gap = band_masks(pos, spos, edges[b], edges[b + 1], a.buffer, gtr, a.drop_shared_senders)
        plans.append((f"band{b}", keep, held))
        warn = "   <-- too few training rows kept: lower --buffer or --bands" if keep.mean() < 0.3 else ""
        mal = ytr[held].mean() if held.any() else float("nan")
        print(f"band {b}: [{edges[b]:.0f}, {edges[b + 1]:.0f})  held-out rows={held.sum():,} "
              f"(malicious {mal:.3f})  train rows kept={keep.mean():.1%}  real gap={gap:.0f} m{warn}")
    for side in ("hi", "lo"):
        keep, held, gap = extrap_masks(pos, spos, side)
        plans.append((f"extrap_{side}", keep, held))
        print(f"extrap_{side}: train rows={keep.sum():,}  test rows={held.sum():,} (malicious {ytr[held].mean():.3f})  "
              f"gap={gap:.0f} m   (Test is ~1800 m away)")

    cols_hdr = [p[0] for p in plans]
    results, table = {}, []
    for name, cols_n in SETS.items():
        cols = [ix[c] for c in cols_n]
        res = {}
        for pname, keep, held in plans:
            res[pname] = fit_scores(Xtr[keep], ytr[keep], Xtr[held], ytr[held], cols)
        res["train_val"] = fit_scores(Xtr, ytr, Xva, yva, cols)
        results[name] = res

    for k, label in ((0, "ROC-AUC"), (1, "AVERAGE PRECISION")):
        print(f"\n{label}")
        print(f"{'feature set':<14}" + "".join(f"{c:>11}" for c in cols_hdr) + f"{'worst':>9}{'Train->Val':>12}")
        for name in SETS:
            vals = [results[name][c][k] for c in cols_hdr]
            print(f"{name:<14}" + "".join(f"{v:>11.4f}" for v in vals)
                  + f"{nan_stat(min, vals):>9.4f}{results[name]['train_val'][k]:>12.4f}")

    ranking = sorted(SETS, key=lambda n: -nan_stat(min, [results[n][c][0] for c in cols_hdr]))
    print("\nRanking by WORST-CASE ROC-AUC over bands and extrapolation (higher = safer under spatial shift):")
    for n in ranking:
        w = nan_stat(min, [results[n][c][0] for c in cols_hdr])
        print(f"  {n:<14} worst={w:.4f}  Train->Val={results[n]['train_val'][0]:.4f}")
    if results["edge"]["train_val"][0] > 0.97:
        print("\n[warning] 'edge' alone reaches AUC > 0.97 on Validation. If distance_to_road_edge is computed from the "
              "TRUE position it leaks the label; if it is computed from the REPORTED position it is a legitimate "
              "map-plausibility feature. Confirm which before using it.")
    print("\nRead: prefer feature sets whose WORST score stays high; bands with data on both sides understate the "
          "Test shift, so extrap_hi / extrap_lo (large gap) are the more relevant columns. With --buffer small or "
          "gaps far below 1800 m the simulated shift is milder than the real one.")

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feature_set", "plan", "roc_auc", "average_precision"])
            for n in SETS:
                for c in cols_hdr + ["train_val"]:
                    w.writerow([n, c, results[n][c][0], results[n][c][1]])
        print("saved", a.csv)


if __name__ == "__main__":
    main()
