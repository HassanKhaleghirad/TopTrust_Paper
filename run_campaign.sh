#!/usr/bin/env bash
# Frozen-configuration campaign for toptrust_revised4.py (InTAS).
# Edit ONLY the values below, after the Validation-based tuning is finished, then run once:
#     bash run_campaign.sh
# Test is evaluated here (--report_test) for the frozen configuration only.
set -euo pipefail

C=${C:-1.0}                    # --cmax chosen on Validation
SLR=${SLR:-0.25}               # --server_lr chosen on Validation (used only for finite epsilon)
LR=${LR:-0.1}
ROUNDS=${ROUNDS:-60}
FEATURES=${FEATURES:-edge_rel_kin}
SEEDS=${SEEDS:-"1 2 3 4 5"}
EPS_LIST=${EPS_LIST:-"0.5 1 2 5"}
PY=${PY:-python}

mkdir -p logs
for s in $SEEDS; do
  echo "== seed $s: non-private federated (eps=inf)"
  $PY toptrust_revised4.py --intas --mode none --epsilon inf --rounds "$ROUNDS" --lr "$LR" \
      --intas_features "$FEATURES" --seed "$s" --report_test > "logs/fed_eps_inf_seed${s}.log"
  for e in $EPS_LIST; do
    echo "== seed $s: uniform DP eps=$e"
    $PY toptrust_revised4.py --intas --mode uniform --epsilon "$e" --rounds "$ROUNDS" --lr "$LR" \
        --cmax "$C" --server_lr "$SLR" --intas_features "$FEATURES" --seed "$s" --report_test \
        > "logs/uniform_eps_${e}_seed${s}.log"
  done
  echo "== seed $s: centralized references (input ablation)"
  for f in edge_rel_kin gnn7; do
    $PY toptrust_revised4.py --intas --mode none --epsilon inf --centralized_epochs 30 --lr "$LR" \
        --intas_features "$f" --seed "$s" --report_test > "logs/central_${f}_seed${s}.log"
  done
done
echo "done; now: python aggregate_results.py"
