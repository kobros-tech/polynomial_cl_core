#!/bin/bash
# Resumable full per-degree study on Split MNIST (run from the repository root).
#   pip install -e . && bash experiments/run_per_degree_study.sh
# Each (strategy, sampling) cell is written to results/per_degree/cell_<strategy>_<sampling>.json and
# skipped if it exists, so an interrupted run can simply be started again. Selection seeds 0-2,
# test seeds 3-9, 3 epochs per experience. The last stage re-runs the shared and the selected vector
# of every cell at 10 epochs (no re-selection). Expect roughly 8-10 minutes per cell on one CPU core.
# Progress (one line per finished run) is printed and also saved to results/per_degree/logs/.
set -euo pipefail
R=results/per_degree
mkdir -p "$R/logs"

run() {  # run <output-json> <log-name> <script arguments...>
  local out=$1 name=$2
  shift 2
  if [ -f "$out" ]; then echo "skip $out (exists)"; return; fi
  echo "=== $name  ($(date '+%H:%M:%S'))"
  python -u experiments/per_degree_lr.py --dataset split_mnist "$@" --output "$out" 2>&1 | tee "$R/logs/$name.log"
}

run "$R/sampling_e3_lr0.1.json" sampling --experiment sampling --seeds 0 1 2 3 4
for st in ordinary ordinary_no_replay small_old_lr frozen_old strategy_b strategy_b_new; do
  for sp in class_balanced shuffled; do
    run "$R/cell_${st}_${sp}.json" "cell_${st}_${sp}" --experiment per-degree --strategies "$st" --samplings "$sp"
  done
done
echo "STUDY DONE"
for st in ordinary ordinary_no_replay small_old_lr frozen_old strategy_b strategy_b_new; do
  run "$R/epochs10_${st}.json" "epochs10_${st}" --experiment epochs --epochs 10 --strategies "$st"
done
echo "ALL DONE"
