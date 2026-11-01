# TopTrust extra experiments (reviewer-driven)

Pure NumPy/SciPy harness (+ `opacus` only for the accountant). It reproduces the training protocol of the paper
(Poisson sampling, per-vehicle clip + Gaussian noise, RSU/TA aggregation normalised over participants) and runs:

| script call | what it produces | paper location |
|---|---|---|
| `--only clip`  | Uniform-DP, mean-matched uniform, shuffled-degree control, sqrt/linear degree, adaptive quantile clipping (Andrew et al. 2021) vs TopTrust, paired t-tests | `results/table_clip_ablation.tex` |
| `--only byz`   | sign-flip, scaling, Gaussian, ALIE and aligned collusion, degree inflation; mean vs trimmed-mean RSU aggregation | `results/table_byz.tex` |
| `--only audit` | canary privacy audit per degree group (k=1,4 releases), topology-leak audit, loss-threshold membership inference | `results/table_audit.tex`, `table_mia.tex`, `audit_topology_leak.txt` |

## Use
1. `pip install numpy scipy opacus`
2. Copy `adapter_template.py`, implement `TopTrustModel` (flat parameter vector, `local_update`, `evaluate`, `per_sample_loss`) and `build()`.
3. `python run_experiments.py --adapter my_adapter:build --out results`
4. Copy `results/*.tex` next to the paper; the paper includes them with `\inputresult{...}` and the red-free `\tbd{}` markers disappear
   once the files exist. Replace the remaining `\tbd{}` numbers in the prose by hand.

`python run_experiments.py --smoke` runs everything on synthetic data. It only proves the code executes; **never report smoke-test numbers**.

## Validity checks already done (sandbox)
* `accountant.calibrate_z` reproduces Table I: eps 0.5/1/2/5 -> z = 3.66/2.12/1.36/0.87 (q=0.05, R=78, delta=1e-5).
* Exact Gaussian-DP epsilon for the RSU view (no amplification) is below the RDP values of Table I (k=4: 2.20 vs 2.39; k=10: 3.70 vs 4.00).
* The audit returns lower bounds that stay below the exact epsilon for known Gaussian shifts, and flags a mechanism whose
  noise is 3x too small (audit 8.39 > claimed 6.89).

## Caveats
* The audit uses worst-case canaries (distance 2C), so it validates the *implementation* of the mechanism, not real-data leakage;
  the membership-inference AUC is the real-data check. Audit bounds are necessarily below theory at delta=1e-5.
* QuantileClip is audited only through the update part; the bit release is accounted by the z_delta split.
* Attackers are omniscient about other selected updates (conservative). Adaptive attacks against the clipping rule are not covered.
