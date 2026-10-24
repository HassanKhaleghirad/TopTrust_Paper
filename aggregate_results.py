"""Aggregate logs/*.log from run_campaign.sh into mean +- std tables (Validation and Test)."""
import ast, csv, glob, re, sys
from collections import defaultdict
import numpy as np

KEYS = ["auc", "pr_auc", "f1", "mcc", "precision", "recall"]
PAT = {"Validation": re.compile(r"^VALIDATION \(threshold selected here\):\s*(\{.*\})", re.M),
       "Test": re.compile(r"^TEST \(official InTAS split; message-level\):\s*(\{.*?\})", re.M)}

def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for split, pat in PAT.items():
        m = pat.search(text)
        if m:
            out[split] = ast.literal_eval(m.group(1))
    return out

groups = defaultdict(list)
for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "logs/*.log")):
    name = re.sub(r"_seed\d+\.log$", "", path.split("/")[-1])
    r = parse(path)
    if r:
        groups[name].append(r)
    else:
        print(f"[skip] no metrics found in {path}")

rows = []
for split in ("Validation", "Test"):
    print(f"\n{split}  (mean +- std over seeds; threshold chosen on Validation)")
    print(f"{'run':<26}{'n':>3}" + "".join(f"{k:>16}" for k in KEYS))
    for name, runs in groups.items():
        runs = [r[split] for r in runs if split in r]
        if not runs:
            continue
        cells = []
        for k in KEYS:
            v = np.array([r[k] for r in runs], dtype=float)
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            cells.append(f"{v.mean():.3f}+-{sd:.3f}")
            rows.append([split, name, len(v), k, v.mean(), sd])
        print(f"{name:<26}{len(runs):>3}" + "".join(f"{c:>16}" for c in cells))
with open("campaign_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["split", "run", "n", "metric", "mean", "std"]); w.writerows(rows)
print("\nsaved campaign_summary.csv")
