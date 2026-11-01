#!/bin/bash
# Resumable full per-degree study on Split MNIST (run from the repository root).
#   pip install -e . && bash experiments/run_per_degree_study.sh
# Each (strategy, sampling) cell is written to results/per_degree/cell_<strategy>_<sampling>.json and
# skipped if it exists, so an interrupted run can simply be started again. Selection seeds 0-2,
# test seeds 3-9, 3 epochs per experience. The last stage re-runs the shared and the selected vector
# of every cell at 10 epochs (no re-selection). Expect roughly 8-10 minutes per cell on one CPU core.
set -euo pipefail
R=results/per_degree
COMMON="--dataset split_mnist"
mkdir -p "$R"
[ -f "$R/sampling_e3_lr0.1.json" ] || python experiments/per_degree_lr.py $COMMON --experiment sampling \
  --seeds 0 1 2 3 4 --output "$R/sampling_e3_lr0.1.json"
for st in ordinary ordinary_no_replay small_old_lr frozen_old strategy_b strategy_b_new; do
  for sp in class_balanced shuffled; do
    [ -f "$R/cell_${st}_${sp}.json" ] || python experiments/per_degree_lr.py $COMMON --experiment per-degree \
      --strategies "$st" --samplings "$sp" --output "$R/cell_${st}_${sp}.json"
  done
done
echo "STUDY DONE"
for st in ordinary ordinary_no_replay small_old_lr frozen_old strategy_b strategy_b_new; do
  [ -f "$R/epochs10_${st}.json" ] || python experiments/per_degree_lr.py $COMMON --experiment epochs \
    --epochs 10 --strategies "$st" --output "$R/epochs10_${st}.json"
done
echo "ALL DONE"
