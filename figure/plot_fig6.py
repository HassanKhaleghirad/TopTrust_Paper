#!/usr/bin/env python3
"""
Fig. 6 for TOPTrust: convergence under different privacy budgets
(+ Uniform-DP at eps = 0.5 with the same sigma, q and R).

Usage
-----
  python plot_fig6.py --csv results_per_round.csv          # real data (use this for the paper)
  python plot_fig6.py --demo                               # synthetic curves, ONLY to test the layout

Input CSV (one row per method / epsilon / seed / round), columns:
  method   : "TOPTrust" or "Uniform-DP"
  epsilon  : 0.5, 1, 2, 5, or inf
  seed     : 0..4 (five independent runs)
  round    : 0..R-1  (communication round)
  f1       : validation F1-score in percent (e.g. 94.1)

Outputs: fig6.pdf / fig6.eps / fig6.png and convergence_summary.csv
(the summary gives the numbers for the \\todo{X, Y, Z, W} in Section V-F).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

R_PLANNED = 78          # rounds for which sigma was calibrated (Table II)
TOL = 1.0               # convergence = first round within TOL points of the final F1
TARGET_F1 = 94.1        # F1 reached by TOPTrust at eps = 0.5 (for "rounds to reach")

# (method, epsilon) -> (label, color, linestyle, marker)
STYLE = {
    ("TOPTrust", np.inf): (r"TOPTrust, $\epsilon=\infty$ (no DP)", "black",   "-",  None),
    ("TOPTrust", 5.0):    (r"TOPTrust, $\epsilon=5$",              "#009E73", "--", None),
    ("TOPTrust", 1.0):    (r"TOPTrust, $\epsilon=1$",              "#E69F00", "-.", None),
    ("TOPTrust", 0.5):    (r"TOPTrust, $\epsilon=0.5$",            "#D62728", ":",  None),
    ("Uniform-DP", 0.5):  (r"Uniform-DP, $\epsilon=0.5$",          "#1F77B4", "-",  "o"),
}


def light(color, w=0.82):
    """Blend a color with white. EPS has no transparency, so we avoid alpha."""
    r, g, b = to_rgb(color)
    return (1 - w) * np.array([r, g, b]) + w * np.ones(3)


def demo_data(seed=0):
    """Synthetic curves, ONLY to test the plotting code. NOT results."""
    rng = np.random.default_rng(seed)
    rows = []
    # (method, eps): (final F1, rate)
    spec = {("TOPTrust", np.inf): (97.4, 0.085), ("TOPTrust", 5.0): (96.9, 0.07),
            ("TOPTrust", 1.0): (95.2, 0.06), ("TOPTrust", 0.5): (94.1, 0.05),
            ("Uniform-DP", 0.5): (91.0, 0.04)}
    for (m, e), (f_end, k) in spec.items():
        rounds = np.arange(0, 140 if np.isinf(e) else R_PLANNED)
        for s in range(5):
            curve = 60 + (f_end - 60) * (1 - np.exp(-k * rounds))
            curve = curve + rng.normal(0, 0.4, len(rounds))
            for r_, v in zip(rounds, curve):
                rows.append((m, e, s, r_, v))
    return pd.DataFrame(rows, columns=["method", "epsilon", "seed", "round", "f1"])


def load(csv):
    df = pd.read_csv(csv)
    df["epsilon"] = df["epsilon"].astype(str).str.lower().replace({"inf": "inf", "infinity": "inf"})
    df["epsilon"] = df["epsilon"].apply(lambda x: np.inf if x == "inf" else float(x))
    return df


def aggregate(df, ddof=1):
    out = {}
    for (m, e), g in df.groupby(["method", "epsilon"]):
        p = g.pivot_table(index="round", columns="seed", values="f1")
        out[(m, e)] = pd.DataFrame({"mean": p.mean(axis=1), "std": p.std(axis=1, ddof=ddof)})
    return out


def first_round(mean, thr):
    idx = np.where(mean.values >= thr)[0]
    return int(mean.index[idx[0]]) if len(idx) else None


def summary(agg):
    rows = []
    for (m, e), d in agg.items():
        # DP curves must not extend past the planned budget
        if not np.isinf(e):
            d = d[d.index < R_PLANNED]
        final = d["mean"].iloc[-1]
        rows.append({
            "method": m, "epsilon": e,
            "last_round": int(d.index[-1]),
            "final_f1": round(final, 2),
            f"round_within_{TOL:g}pt_of_final": first_round(d["mean"], final - TOL),
            f"round_to_reach_{TARGET_F1}": first_round(d["mean"], TARGET_F1),
        })
    return pd.DataFrame(rows)


def plot(agg, out="fig6", demo=False):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.usetex": False, "mathtext.fontset": "cm",
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.5))  # IEEE single column
    for key, (label, c, ls, mk) in STYLE.items():
        if key not in agg:
            continue
        d = agg[key]
        if not np.isinf(key[1]):                  # DP: stop at the planned budget
            d = d[d.index < R_PLANNED]
        x, mu, sd = d.index.values, d["mean"].values, d["std"].fillna(0).values
        ax.fill_between(x, mu - sd, mu + sd, color=light(c), lw=0, zorder=1)
        ax.plot(x, mu, color=c, ls=ls, lw=1.3, marker=mk, ms=2.5,
                markevery=max(1, len(x) // 10), label=label, zorder=3)

    ax.axvline(R_PLANNED, color="gray", lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(R_PLANNED - 1, 0.45, r"budget exhausted ($R=78$)", rotation=90,
            transform=ax.get_xaxis_transform(), ha="right", va="top",
            fontsize=6, color="gray")
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("F1-score (%)")
    ax.set_xlim(0, None)
    ax.grid(True, ls=":", lw=0.5, alpha=0.7)
    ax.legend(loc="lower right", frameon=True, framealpha=1, edgecolor="0.7")
    if demo:
        ax.text(0.5, 0.5, "DEMO - SYNTHETIC DATA", transform=ax.transAxes, ha="center",
                va="center", rotation=25, fontsize=14, color="red", alpha=0.35, zorder=10)
        out += "_DEMO"
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print(f"saved {out}.pdf / .eps / .png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="per-round results CSV (see docstring)")
    ap.add_argument("--demo", action="store_true", help="synthetic data, layout test only")
    ap.add_argument("--out", default="fig6")
    a = ap.parse_args()
    if not a.demo and not a.csv:
        ap.error("give --csv results.csv (real runs) or --demo")
    df = demo_data() if a.demo else load(a.csv)
    agg = aggregate(df)
    plot(agg, a.out, demo=a.demo)
    s = summary(agg)
    s.to_csv("convergence_summary_DEMO.csv" if a.demo else "convergence_summary.csv", index=False)
    print(s.to_string(index=False))
