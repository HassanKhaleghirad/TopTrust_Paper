"""Runs the four extra experiments requested by the reviewers and writes LaTeX tabulars to results/.
  clip   : clipping-rule ablation + adaptive-quantile baseline (Andrew et al.) + Uniform-DP
  byz    : Byzantine / collusion / degree-inflation attacks, with and without a robust RSU aggregator
  audit  : canary privacy audit (per degree group), topology-leak audit, loss-threshold membership inference
Usage:  python run_experiments.py --smoke                      (synthetic data, only checks that the code runs)
        python run_experiments.py --adapter my_adapter:build   (your GNN and data, see README.md)"""
import argparse, importlib, json, os
import numpy as np
from scipy import stats
from clipping import (UniformClip, DegreeClip, ShuffledDegreeClip, QuantileClip, match_alpha)
from fl import run_fl
from attacks import Attack, make_byzantine, reset_byzantine
from audit import audit_vehicle, audit_topology_leak, mia_auc

# noise multipliers from Table I of the paper (N=1000, q=0.05, R=78, delta=1e-5); other settings: accountant.calibrate_z
Z_TABLE = {0.5: 3.66, 1.0: 2.12, 2.0: 1.36, 5.0: 0.87, float("inf"): 0.0}
CFG = dict(q=0.05, rounds=78, epochs=5, lr=0.05, alpha=1.0, c_max=1.0, tau_min=3, delta=1e-5,
           seeds=5, eval_every=78, byz_frac=0.2, byz_eps=2.0, clip_eps=(0.5, 2.0))

def pm(a): return "$%.1f\\pm%.1f$" % (100 * np.mean(a), 100 * np.std(a))
def write_tab(path, colspec, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n" % (colspec, " & ".join(header)))
        for r in rows: f.write(" & ".join(str(x) for x in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

def rule_factories(clients, cfg):
    topo = lambda s: DegreeClip(cfg["alpha"], cfg["c_max"], cfg["tau_min"], "log")
    mean_c = float(np.mean([topo(0).bound(c) for c in clients]))
    degs = [c.public_deg for c in clients]
    a_sqrt = match_alpha("sqrt", cfg["c_max"], cfg["tau_min"], degs, mean_c)
    a_lin = match_alpha("linear", cfg["c_max"], cfg["tau_min"], degs, mean_c)
    return {
        "Uniform-DP ($C_{\\max}$)": lambda s: UniformClip(cfg["c_max"]),
        "Uniform, mean-matched": lambda s: UniformClip(mean_c),
        "Shuffled degrees (control)": lambda s: ShuffledDegreeClip(cfg["alpha"], cfg["c_max"], cfg["tau_min"], clients, seed=s),
        "Degree, $\\sqrt{\\deg}$": lambda s: DegreeClip(a_sqrt, cfg["c_max"], cfg["tau_min"], "sqrt"),
        "Degree, linear": lambda s: DegreeClip(a_lin, cfg["c_max"], cfg["tau_min"], "linear"),
        "Adaptive quantile \\cite{Andrew2021}": lambda s: QuantileClip(c0=cfg["c_max"]),
        "\\textsc{Top}Trust ($\\log$ degree)": topo,
    }

def final_f1(hist): return hist[-1][1]["f1"]

def one(model, clients, test, clipper, eps, cfg, seed, attack=None, aggregator="mean"):
    rng = np.random.default_rng(seed)
    p, h = run_fl(model, clients, test, clipper, Z_TABLE[eps], cfg["q"], cfg["rounds"], cfg["epochs"], cfg["lr"],
                  rng, attack, aggregator, eval_every=cfg["eval_every"])
    return p, final_f1(h)

def exp_clip(model, clients, test, cfg, out):
    facts = rule_factories(clients, cfg); res = {}; reset_byzantine(clients)
    for name, mk in facts.items():
        for eps in cfg["clip_eps"]:
            res[(name, eps)] = [one(model, clients, test, mk(s), eps, cfg, s)[1] for s in range(cfg["seeds"])]
    ref = [k for k in facts if k.startswith("\\textsc{Top}Trust")][0]
    rows = []
    for name in facts:
        cells = []
        for eps in cfg["clip_eps"]:
            c = pm(res[(name, eps)])
            if name != ref:
                p = stats.ttest_rel(res[(ref, eps)], res[(name, eps)]).pvalue
                c += "$^{%s}$" % ("\\dagger" if p < 0.05 / (len(facts) - 1) else "")
            cells.append(c)
        rows.append([name] + cells)
    write_tab(out + "/table_clip_ablation.tex", "l" + "c" * len(cfg["clip_eps"]),
              ["\\textbf{Clipping rule}"] + ["$\\epsilon=%g$" % e for e in cfg["clip_eps"]], rows)
    json.dump({"%s|%s" % k: v for k, v in res.items()}, open(out + "/raw_clip.json", "w"))

def exp_byz(model, clients, test, cfg, out):
    eps = cfg["byz_eps"]; facts = rule_factories(clients, cfg)
    topo_name = [k for k in facts if k.startswith("\\textsc{Top}Trust")][0]; uni_name = "Uniform-DP ($C_{\\max}$)"
    cols = [("TopTrust, mean", topo_name, "mean"), ("TopTrust, trimmed mean", topo_name, "trimmed"), ("Uniform-DP, mean", uni_name, "mean")]
    scen = [("No attack", None, False), ("Sign flipping ($\\times10$)", "signflip", False), ("Scaled update ($\\times10$)", "scale", False),
            ("Gaussian noise", "gauss", False), ("Collusion: ALIE", "alie", False), ("Collusion: aligned", "aligned", False),
            ("Sign flipping + degree inflation", "signflip", True)]
    rows = []
    for label, kind, infl in scen:
        row = [label]
        for _, rname, agg in cols:
            vals = []
            for s in range(cfg["seeds"]):
                rng = np.random.default_rng(1000 + s); reset_byzantine(clients)
                if kind: make_byzantine(clients, cfg["byz_frac"], rng, inflate_degree=infl)
                vals.append(one(model, clients, test, facts[rname](s), eps, cfg, s, Attack(kind) if kind else None, agg)[1])
            row.append(pm(vals))
        rows.append(row)
    reset_byzantine(clients)
    write_tab(out + "/table_byz.tex", "lccc", ["\\textbf{Scenario}"] + ["\\textbf{%s}" % c[0] for c in cols], rows)

def exp_audit(model, clients, test, cfg, out, n_trials=20000, dim=50):
    rng = np.random.default_rng(7); eps = cfg["byz_eps"]; z = Z_TABLE[eps]; delta = cfg["delta"]
    order = sorted(clients, key=lambda c: c.deg)
    groups = [("low", order[len(order) // 10]), ("median", order[len(order) // 2]), ("high", order[-len(order) // 10 - 1])]
    facts = rule_factories(clients, cfg)
    rows = []
    for rname in ("\\textsc{Top}Trust ($\\log$ degree)", "Uniform-DP ($C_{\\max}$)"):
        cl = facts[rname](0)
        for g, c in groups:
            r1 = audit_vehicle(cl, c, z, dim, 1, n_trials, delta, rng)
            r4 = audit_vehicle(cl, c, z, dim, 4, n_trials, delta, rng)
            rows.append([rname, g, c.deg, "%.2f" % r1["bound"], "%.2f / %.2f" % (r1["eps_theory"], r1["eps_emp"]),
                         "%.2f / %.2f" % (r4["eps_theory"], r4["eps_emp"])])
    leak = audit_topology_leak(lambda: DegreeClip(cfg["alpha"], cfg["c_max"], cfg["tau_min"]), groups[0][1].deg, z, 500, 10000, delta, rng)
    write_tab(out + "/table_audit.tex", "llcccc",
              ["\\textbf{Rule}", "\\textbf{Degree group}", "$\\deg$", "$C_i$", "$k{=}1$: theory / audit", "$k{=}4$: theory / audit"], rows)
    open(out + "/audit_topology_leak.txt", "w").write("empirical eps lower bound when the degree itself changes by one: %.2f\n" % leak)
    # membership inference on trained global models
    mrows = []
    rng2 = np.random.default_rng(11)
    members = [c for c in clients if not c.byz][: max(5, len(clients) // 4)]
    for e in cfg["clip_eps"] + (float("inf"),):
        row = ["$\\epsilon=%s$" % ("\\infty" if e == float("inf") else "%g" % e)]
        for rname in ("\\textsc{Top}Trust ($\\log$ degree)", "Uniform-DP ($C_{\\max}$)"):
            aucs = []
            for s in range(cfg["seeds"]):
                p, _ = one(model, clients, test, facts[rname](s), e, cfg, s)
                lm = np.concatenate([model.per_sample_loss(p, c.data) for c in members])
                ln = model.per_sample_loss(p, test)
                aucs.append(mia_auc(lm, ln))
            row.append("$%.3f\\pm%.3f$" % (np.mean(aucs), np.std(aucs)))
        mrows.append(row)
    write_tab(out + "/table_mia.tex", "lcc", ["$\\epsilon$", "\\textbf{Top}Trust AUC", "Uniform-DP AUC"], mrows)

def load(adapter, smoke):
    if smoke:
        from toy import ToyModel, build_toy
        clients, test = build_toy(); return ToyModel(), clients, test, {}
    mod, fn = adapter.split(":"); return getattr(importlib.import_module(mod), fn)()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="clip,byz,audit"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    model, clients, test, over = load(a.adapter, a.smoke)
    cfg = dict(CFG); cfg.update(over)
    kw = {}
    if a.smoke:
        cfg.update(q=0.5, rounds=12, seeds=2, eval_every=12, lr=0.5, c_max=0.5, tau_min=2); kw = dict(n_trials=2000, dim=20)
    for name in a.only.split(","):
        print("running", name)
        if name == "clip": exp_clip(model, clients, test, cfg, a.out)
        if name == "byz": exp_byz(model, clients, test, cfg, a.out)
        if name == "audit": exp_audit(model, clients, test, cfg, a.out, **kw)
    print("tables written to", a.out)
