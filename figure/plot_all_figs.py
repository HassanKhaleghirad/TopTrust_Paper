#!/usr/bin/env python3
"""
All result figures of the TOPTrust paper (Figs. 2-9) from REAL experiment logs.

  python plot_all_figs.py demo                       # layout test with SYNTHETIC data (watermarked)
  python plot_all_figs.py all --data-dir results/    # every figure whose CSV exists in results/
  python plot_all_figs.py fig6 --csv results/fig6_per_round.csv
  python plot_all_figs.py parse-log --log run.txt --method TOPTrust --epsilon 0.5 --seed 0 \
                                    --out results/fig6_per_round.csv

Fig. 1 (architecture) is a drawing, not a plot, so it is not generated here.

CSV files expected in --data-dir (one row per run / seed; std is computed over seeds, ddof=1):
  fig2_methods.csv          method, setting, seed, f1            (setting: e.g. "eps=inf", "eps=0.5", "" )
  fig3_privacy_utility.csv  method, epsilon, seed, f1            (epsilon: 0.5, 1, 2, 5 or inf)
  fig4_latency_scaling.csv  method, n_vehicles, seed, latency_ms
  fig5a_latency_methods.csv method, seed, latency_ms
  fig5b_robustness.csv      kind, level, seed, f1                (kind: gradient_noise | malicious_ratio)
  fig6_per_round.csv        method, epsilon, seed, round, f1     (method: TOPTrust | Uniform-DP)
  fig7_coldstart.csv        init, seed, n_interactions, f1       (init: bootstrap | random)
  fig8_datasets.csv         dataset, seed, precision, recall, f1
  fig9_sensitivity.csv      param, value, seed, f1               (param: L | alpha | E)
All F1 / precision / recall values are in percent.

Nothing in this file contains results: every number in a figure comes from your CSVs.
Summaries needed for the text (R^2 of Fig. 4, convergence rounds of Fig. 6, interactions to 90% F1 of
Fig. 7) are printed to the console and saved as *_summary.csv.
"""
import argparse
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

warnings.filterwarnings("ignore", message=".*PostScript backend does not support transparency.*")

R_PLANNED = 78        # rounds for which sigma was calibrated (Table II)
TOL = 1.0             # Fig. 6: convergence = within TOL points of final F1
TARGET_F1_COLD = 90.0 # Fig. 7: threshold for "interactions needed"
COL = ["#1F77B4", "#D62728", "#009E73", "#E69F00", "#9467BD", "#8C564B", "#7F7F7F", "#17BECF"]
HATCH = ["//", "\\\\", "..", "--", "xx", "++", "oo", "**"]
MARK = ["o", "s", "^", "D", "v", "P", "X"]


# --------------------------------------------------------------------------- helpers
def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "mathtext.fontset": "cm",
        "axes.grid": True, "grid.linestyle": ":", "grid.linewidth": 0.5, "grid.alpha": 0.7,
    })


def light(c, w=0.82):
    return (1 - w) * np.array(to_rgb(c)) + w * np.ones(3)   # EPS has no alpha: blend with white


def save(fig, name, outdir, demo):
    os.makedirs(outdir, exist_ok=True)
    if demo:
        fig.text(0.5, 0.5, "", ha="center", va="center", rotation=25,
                 fontsize=14, color="red", alpha=0.35)
        name += "_DEMO"
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  saved {outdir}/{name}.[pdf|eps|png]")


def read(path):
    df = pd.read_csv(path)
    if "epsilon" in df.columns:
        df["epsilon"] = df["epsilon"].astype(str).str.lower().map(
            lambda x: np.inf if x in ("inf", "infinity", "none", "nan") else float(x))
    return df


def ms(df, by, val):
    g = df.groupby(by, sort=False)[val]
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1)}).reset_index()
    if out["std"].isna().all():
        print("  WARNING: a single seed per point -> std not defined (shown as 0). Use >=5 seeds.")
    out["std"] = out["std"].fillna(0.0)
    return out


def new(w=3.45, h=2.5, **kw):
    style()
    return plt.subplots(figsize=(w, h), **kw)


# --------------------------------------------------------------------------- figures
def fig2(df, out, demo):
    d = ms(df, [c for c in ("method", "setting") if c in df.columns], "f1")
    if "setting" in d.columns:
        s = d["setting"].fillna("").astype(str)
        d["label"] = np.where(s == "", d["method"], d["method"] + "\n" + s)
    else:
        d["label"] = d["method"]
    fig, ax = new(3.6, 2.7)
    for i, r in d.iterrows():
        ax.bar(i, r["mean"], yerr=r["std"], capsize=2, color=COL[i % len(COL)], hatch=HATCH[i % len(HATCH)],
               edgecolor="black", lw=0.6, zorder=3)
        ax.text(i, r["mean"] + r["std"] + 0.15, f"{r['mean']:.1f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["label"], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("F1-score (%)")
    ax.set_ylim(np.floor((d["mean"] - d["std"]).min() - 2), 100)
    save(fig, "fig2_f1_methods", out, demo)


def fig3(df, out, demo):
    d = ms(df, ["method", "epsilon"], "f1")
    fig, ax = new()
    for i, (m, g) in enumerate(d.groupby("method", sort=False)):
        fin = g[np.isfinite(g["epsilon"])].sort_values("epsilon")
        ax.errorbar(fin["epsilon"], fin["mean"], yerr=fin["std"], color=COL[i % len(COL)], marker=MARK[i % 7],
                    ms=3.5, lw=1.2, capsize=2, label=m)
        inf = g[~np.isfinite(g["epsilon"])]
        if len(inf):
            ax.axhline(inf["mean"].iloc[0], color=COL[i % len(COL)], ls=":", lw=0.9)
    eps = sorted(df.loc[np.isfinite(df["epsilon"]), "epsilon"].unique())
    ax.set_xscale("log")
    ax.set_xticks(eps)
    ax.set_xticklabels([f"{e:g}" for e in eps])
    ax.minorticks_off()
    ax.set_xlabel(r"Privacy budget $\epsilon$ ($\delta=10^{-5}$)")
    ax.set_ylabel("F1-score (%)")
    ax.legend(loc="lower right")
    save(fig, "fig3_privacy_utility", out, demo)


def fig4(df, out, demo):
    d = ms(df, ["method", "n_vehicles"], "latency_ms")
    fig, ax = new()
    rows = []
    for i, (m, g) in enumerate(d.groupby("method", sort=False)):
        g = g.sort_values("n_vehicles")
        ax.errorbar(g["n_vehicles"], g["mean"], yerr=g["std"], color=COL[i % len(COL)], marker=MARK[i % 7],
                    ms=3.5, lw=1.2, capsize=2, label=m)
        if len(g) >= 3:
            p = np.polyfit(g["n_vehicles"], g["mean"], 1)
            res = g["mean"] - np.polyval(p, g["n_vehicles"])
            r2 = 1 - (res ** 2).sum() / ((g["mean"] - g["mean"].mean()) ** 2).sum()
            rows.append({"method": m, "slope_ms_per_vehicle": p[0], "R2": r2})
    ax.set_xlabel("Number of vehicles")
    ax.set_ylabel("Detection latency (ms)")
    ax.legend(loc="upper left")
    save(fig, "fig4_latency_scaling", out, demo)
    if rows:
        s = pd.DataFrame(rows)
        print(s.to_string(index=False))
        s.to_csv(os.path.join(out, "fig4_summary.csv"), index=False)


def fig5a(df, out, demo, ref=100.0):
    d = ms(df, ["method"], "latency_ms")
    fig, ax = new(3.4, 2.6)
    for i, r in d.iterrows():
        ax.bar(i, r["mean"], yerr=r["std"], capsize=2, color=COL[i % len(COL)], hatch=HATCH[i % len(HATCH)],
               edgecolor="black", lw=0.6, zorder=3)
        ax.text(i, r["mean"] + r["std"] + 5, f"{r['mean']:.0f}", ha="center", fontsize=6)
    ax.axhline(ref, color="red", ls="--", lw=0.9, zorder=4)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["method"], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("Detection latency (ms)")
    save(fig, "fig5a_latency_methods", out, demo)


def fig5b(df, out, demo):
    labels = {"gradient_noise": "Gradient noise level (%)", "malicious_ratio": "Malicious vehicles (%)"}
    kinds = [k for k in labels if k in set(df["kind"])]
    style()
    fig, axs = plt.subplots(1, len(kinds), figsize=(3.45 * len(kinds) / (1.0 if len(kinds) == 1 else 1.5), 2.4))
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, kinds):
        d = ms(df[df["kind"] == k], ["level"], "f1").sort_values("level")
        ax.errorbar(d["level"], d["mean"], yerr=d["std"], color=COL[1], marker="o", ms=3.5, lw=1.2, capsize=2)
        for _, r in d.iterrows():
            ax.annotate(f"{r['mean']:.1f}", (r["level"], r["mean"]), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=6)
        ax.set_xlabel(labels[k])
        ax.set_ylabel("F1-score (%)")
    save(fig, "fig5b_robustness", out, demo)


def first_reach(x, y, thr):
    idx = np.where(np.asarray(y) >= thr)[0]
    return None if len(idx) == 0 else float(np.asarray(x)[idx[0]])


def fig6(df, out, demo):
    sty = {("TOPTrust", np.inf): (r"TOPTrust, $\epsilon=\infty$", "black", "-", None),
           ("TOPTrust", 5.0): (r"TOPTrust, $\epsilon=5$", "#009E73", "--", None),
           ("TOPTrust", 2.0): (r"TOPTrust, $\epsilon=2$", "#9467BD", (0, (5, 1, 1, 1, 1, 1)), None),
           ("TOPTrust", 1.0): (r"TOPTrust, $\epsilon=1$", "#E69F00", "-.", None),
           ("TOPTrust", 0.5): (r"TOPTrust, $\epsilon=0.5$", "#D62728", ":", None),
           ("Uniform-DP", 0.5): (r"Uniform-DP, $\epsilon=0.5$", "#1F77B4", "-", "o")}
    agg, rows = {}, []
    for (m, e), g in df.groupby(["method", "epsilon"]):
        p = g.pivot_table(index="round", columns="seed", values="f1")
        agg[(m, e)] = pd.DataFrame({"mean": p.mean(axis=1), "std": p.std(axis=1, ddof=1)}).fillna(0.0)
    fig, ax = new()
    ref = agg.get(("TOPTrust", 0.5))
    tgt = float(ref["mean"].iloc[-1]) if ref is not None else None
    for key, (lab, c, ls, mk) in sty.items():
        if key not in agg:
            continue
        d = agg[key]
        if not np.isinf(key[1]):
            d = d[d.index < R_PLANNED]            # DP curves stop at the planned budget
        x, mu, sd = d.index.values, d["mean"].values, d["std"].values
        ax.fill_between(x, mu - sd, mu + sd, color=light(c), lw=0, zorder=1)
        ax.plot(x, mu, color=c, ls=ls, lw=1.3, marker=mk, ms=2.5, markevery=max(1, len(x) // 10),
                label=lab, zorder=3)
        fin = mu[-1]
        rows.append({"method": key[0], "epsilon": key[1], "last_round": int(x[-1]), "final_f1": round(fin, 2),
                     f"round_within_{TOL:g}pt_of_final": first_reach(x, mu, fin - TOL),
                     "round_to_reach_TOPTrust_eps0.5_final": first_reach(x, mu, tgt) if tgt else None})
    ax.axvline(R_PLANNED, color="gray", lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(R_PLANNED - 1, 0.45, r"budget exhausted ($R=78$)", rotation=90, transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=6, color="gray")
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("F1-score (%)")
    ax.set_xlim(0, None)
    ax.legend(loc="lower right", framealpha=1, edgecolor="0.7")
    save(fig, "fig6_convergence", out, demo)
    s = pd.DataFrame(rows)
    print(s.to_string(index=False))
    s.to_csv(os.path.join(out, "fig6_summary.csv"), index=False)


def fig7(df, out, demo):
    fig, ax = new()
    rows = []
    for i, (init, g) in enumerate(df.groupby("init", sort=False)):
        p = g.pivot_table(index="n_interactions", columns="seed", values="f1")
        mu, sd = p.mean(axis=1), p.std(axis=1, ddof=1).fillna(0.0)
        c = COL[i % len(COL)]
        ax.fill_between(p.index, mu - sd, mu + sd, color=light(c), lw=0, zorder=1)
        ax.plot(p.index, mu, color=c, marker=MARK[i % 7], ms=2.8, lw=1.2, ls="-" if i == 0 else "--",
                label=str(init), zorder=3)
        per_seed = [first_reach(p.index.values, p[s].values, TARGET_F1_COLD) for s in p.columns]
        ok = [v for v in per_seed if v is not None]
        rows.append({"init": init, "interactions_to_90_mean": np.mean(ok) if ok else None,
                     "interactions_to_90_std": np.std(ok, ddof=1) if len(ok) > 1 else None,
                     "seeds_reaching_90": f"{len(ok)}/{len(per_seed)}"})
    ax.axhline(TARGET_F1_COLD, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Number of observed interactions")
    ax.set_ylabel("F1-score (%)")
    ax.legend(loc="lower right")
    save(fig, "fig7_coldstart", out, demo)
    s = pd.DataFrame(rows)
    print(s.to_string(index=False))
    s.to_csv(os.path.join(out, "fig7_summary.csv"), index=False)


def fig8(df, out, demo):
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
    datasets = list(dict.fromkeys(df["dataset"]))
    fig, ax = new(3.6, 2.5)
    w = 0.26
    for j, (col, lab) in enumerate(metrics):
        d = ms(df, ["dataset"], col).set_index("dataset").loc[datasets]
        x = np.arange(len(datasets)) + (j - 1) * w
        ax.bar(x, d["mean"], w, yerr=d["std"], capsize=2, color=COL[j], edgecolor="black", lw=0.5, label=lab, zorder=3)
        for xi, m in zip(x, d["mean"]):
            ax.text(xi, m + 0.7, f"{m:.1f}", ha="center", fontsize=5.5, rotation=90, va="bottom")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=7)
    ax.set_ylabel("Percentage (%)")
    lo = min(ms(df, ["dataset"], c)["mean"].min() for c, _ in metrics)
    ax.set_ylim(np.floor(lo - 6), 102)
    ax.legend(loc="upper right", ncol=3)
    save(fig, "fig8_datasets", out, demo)


def fig9(df, out, demo):
    names = {"L": "GNN layers $L$", "alpha": r"Clipping factor $\alpha$", "E": "Local epochs $E$"}
    params = [p for p in names if p in set(df["param"])]
    style()
    fig, axs = plt.subplots(1, len(params), figsize=(3.45 * 1.9, 2.1), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, p in zip(axs, params):
        d = ms(df[df["param"] == p], ["value"], "f1").sort_values("value")
        ax.errorbar(d["value"], d["mean"], yerr=d["std"], color=COL[0], marker="o", ms=3.5, lw=1.2, capsize=2)
        b = d.loc[d["mean"].idxmax()]
        ax.plot(b["value"], b["mean"], "o", ms=6, mfc="none", mec=COL[1])
        ax.set_xlabel(names[p])
        ax.set_xticks(d["value"])
    axs[0].set_ylabel("F1-score (%)")
    save(fig, "fig9_sensitivity", out, demo)


# --------------------------------------------------------------------------- log parser
def parse_log(a):
    pat = re.compile(r"^(?:round|epoch)\s+(\d+)\s+val F1=([0-9.]+)")
    rows = []
    with open(a.log) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                rows.append((a.method, a.epsilon, a.seed, int(m.group(1)), 100 * float(m.group(2))))
    if not rows:
        sys.exit("no 'round N val F1=...' lines found")
    new_file = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=["method", "epsilon", "seed", "round", "f1"]).to_csv(
        a.out, mode="a", header=new_file, index=False)
    print(f"appended {len(rows)} rows to {a.out}")
    if len(rows) < 30:
        print("NOTE: your script prints val F1 only every 10 rounds -> curves are coarse. "
              "Evaluate and print/log every round for Fig. 6.")


# --------------------------------------------------------------------------- demo data (NOT results)
def demo_frames():
    rng = np.random.default_rng(0)
    S = range(5)
    f2 = pd.DataFrame([(m, s_, sd, v + rng.normal(0, .4)) for m, s_, v in
                       [("Ensemble-IDS", "", 96.8), ("QART", "", 96.6), ("FL-IDS", "", 93.8),
                        ("Fed-GNN", "", 95.8), ("TOPTrust", "eps=inf", 97.4), ("TOPTrust", "eps=0.5", 94.1)]
                       for sd in S], columns=["method", "setting", "seed", "f1"])
    f3 = pd.DataFrame([(m, e, sd, v + rng.normal(0, .4)) for m, vals in
                       {"TOPTrust": {.5: 94.1, 1: 95.2, 2: 96.3, 5: 96.9, np.inf: 97.4},
                        "Uniform-DP": {.5: 90.5, 1: 92.8, 2: 94.7, 5: 96.0, np.inf: 97.4}}.items()
                       for e, v in vals.items() for sd in S], columns=["method", "epsilon", "seed", "f1"])
    f4 = pd.DataFrame([(m, n, sd, k * n + b + rng.normal(0, 4)) for m, k, b in
                       [("TOPTrust", .14, 45), ("Fed-GNN", .41, 40)] for n in range(250, 1001, 125) for sd in S],
                      columns=["method", "n_vehicles", "seed", "latency_ms"])
    f5a = pd.DataFrame([(m, sd, v + rng.normal(0, 8)) for m, v in
                        [("Ensemble-IDS", 320), ("QART", 290), ("FL-IDS", 410), ("Fed-GNN", 452), ("TOPTrust", 187)]
                        for sd in S], columns=["method", "seed", "latency_ms"])
    f5b = pd.DataFrame(
        [("gradient_noise", l, sd, 97.4 - 0.2 * l + rng.normal(0, .3)) for l in (0, 5, 10, 15, 20, 25) for sd in S] +
        [("malicious_ratio", l, sd, 97.4 - 0.16 * l + rng.normal(0, .3)) for l in (0, 10, 20, 30, 40) for sd in S],
        columns=["kind", "level", "seed", "f1"])
    r6 = []
    for m, e, fe, k in [("TOPTrust", np.inf, 97.4, .085), ("TOPTrust", 5, 96.9, .07), ("TOPTrust", 1, 95.2, .06),
                        ("TOPTrust", .5, 94.1, .05), ("Uniform-DP", .5, 91.0, .04)]:
        for sd in S:
            for r in range(140 if np.isinf(e) else R_PLANNED):
                r6.append((m, e, sd, r, 60 + (fe - 60) * (1 - np.exp(-k * r)) + rng.normal(0, .4)))
    f6 = pd.DataFrame(r6, columns=["method", "epsilon", "seed", "round", "f1"])
    f7 = pd.DataFrame([(i, sd, n, (60 if i == "random" else 78) + (96 - (60 if i == "random" else 78)) *
                        (1 - np.exp(-(.2 if i == "random" else .6) * n)) + rng.normal(0, .5))
                       for i in ("bootstrap", "random") for sd in S for n in range(1, 21)],
                      columns=["init", "seed", "n_interactions", "f1"])
    f8 = pd.DataFrame([(d, sd, p + rng.normal(0, .5), r + rng.normal(0, .5), f + rng.normal(0, .5))
                       for d, p, r, f in [("VeReMi", 97.6, 97.2, 97.4), ("Dataset-B", 94.5, 93.9, 94.2),
                                          ("pNEUMA", 95.4, 95.0, 95.2)] for sd in S],
                      columns=["dataset", "seed", "precision", "recall", "f1"])
    f9 = pd.DataFrame(
        [("L", v, sd, f + rng.normal(0, .3)) for v, f in zip((1, 2, 3, 4, 5), (94.2, 96.5, 97.4, 97.1, 96.3)) for sd in S] +
        [("alpha", v, sd, f + rng.normal(0, .3)) for v, f in zip((.5, .75, 1, 1.25, 1.5), (94.8, 96.2, 97.4, 96.9, 95.7)) for sd in S] +
        [("E", v, sd, f + rng.normal(0, .3)) for v, f in zip((2, 4, 6, 8, 10), (95.0, 96.8, 97.3, 97.4, 97.4)) for sd in S],
        columns=["param", "value", "seed", "f1"])
    return f2, f3, f4, f5a, f5b, f6, f7, f8, f9


FIGS = {"fig2": ("fig2_methods.csv", fig2), "fig3": ("fig3_privacy_utility.csv", fig3),
        "fig4": ("fig4_latency_scaling.csv", fig4), "fig5a": ("fig5a_latency_methods.csv", fig5a),
        "fig5b": ("fig5b_robustness.csv", fig5b), "fig6": ("fig6_per_round.csv", fig6),
        "fig7": ("fig7_coldstart.csv", fig7), "fig8": ("fig8_datasets.csv", fig8),
        "fig9": ("fig9_sensitivity.csv", fig9)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["demo", "all", "parse-log"] + list(FIGS))
    ap.add_argument("--csv")
    ap.add_argument("--data-dir", default="results")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--log"); ap.add_argument("--method", default="TOPTrust")
    ap.add_argument("--epsilon", default="inf"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.cmd == "parse-log":
        a.out = a.out if a.out.endswith(".csv") else "results/fig6_per_round.csv"
        parse_log(a)
    elif a.cmd == "demo":
        for (key, (_, fn)), df in zip(FIGS.items(), demo_frames()):
            print(key); fn(df, a.out, True)
    elif a.cmd == "all":
        for key, (fname, fn) in FIGS.items():
            p = os.path.join(a.data_dir, fname)
            if os.path.exists(p):
                print(key); fn(read(p), a.out, False)
            else:
                print(f"{key}: skipped (missing {p})")
    else:
        if not a.csv:
            ap.error("--csv is required")
        FIGS[a.cmd][1](read(a.csv), a.out, False)
