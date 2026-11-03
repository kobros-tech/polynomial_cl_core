# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Choose per-feature learning rates on a validation split, then report the test result.

For every (feature power p, base learning rate) cell the class-incremental comparison is run on
a validation split carved out of the training set. Each condition (or strategy) gets the cell
with the best validation final mean accuracy, so a baseline is tuned as much as the method it is
compared with. Only the selected cells are then evaluated on the test set. Feature power p gives
feature i the step size ``lr * (max_rms / rms_i) ** p``; p = 0 is one shared rate.
"""

import argparse
import json
from pathlib import Path

from common import add_dataset_arguments, load_dataset

from polynomial_cl.continual import (
    holdout_split,
    make_feature_data,
    run_comparison,
    run_strategy_comparison,
)
from polynomial_cl.continual.experiment import (
    EPOCHS,
    MEMORY_PER_CLASS,
    N_REF_PER_CLASS,
)

METRICS = ("final_mean_accuracy", "final_current_accuracy", "mean_forgetting")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(p)
    p.add_argument(
        "--experiment", choices=("conditions", "strategies"), default="conditions"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--powers", nargs="+", type=float, default=[0, 1, 2, 3])
    p.add_argument("--lrs", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0])
    p.add_argument(
        "--select-seeds",
        nargs="+",
        type=int,
        default=[0, 1],
        help="seeds used on the validation split to choose (p, lr)",
    )
    p.add_argument(
        "--test-seeds",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        help="seeds for the reported test result; must be disjoint from --select-seeds",
    )
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--memory-per-class", type=int, default=MEMORY_PER_CLASS)
    p.add_argument(
        "--n-ref-per-class", type=int, default=N_REF_PER_CLASS, help="strategy_b only"
    )
    args = p.parse_args()
    shared = sorted(set(args.select_seeds) & set(args.test_seeds))
    if shared:
        p.error(
            f"--select-seeds and --test-seeds overlap on {shared}: the same stochastic "
            "runs would choose the rates and then be reported as the test estimate"
        )

    run = run_strategy_comparison if args.experiment == "strategies" else run_comparison
    extra = (
        {"n_ref_per_class": args.n_ref_per_class}
        if args.experiment == "strategies"
        else {}
    )

    def compare(data, seeds, power, lr):
        return run(
            data,
            seeds,
            epochs=args.epochs,
            learning_rate=lr,
            memory_per_class=args.memory_per_class,
            feature_power=power,
            **extra,
        )["summary"]

    train_x, train_y, test_x, test_y, data_settings = load_dataset(args)
    fit = holdout_split(train_x, train_y, args.validation_fraction)
    validation_data = make_feature_data(*fit)

    grid = {}
    for power in args.powers:
        for lr in args.lrs:
            summary = compare(validation_data, args.select_seeds, power, lr)
            grid[(power, lr)] = {
                n: s["final_mean_accuracy"]["mean"] for n, s in summary.items()
            }
            print(
                f"validation p={power:g} lr={lr:g}:",
                {n: round(a, 3) for n, a in grid[(power, lr)].items()},
            )
    names = list(next(iter(grid.values())))
    chosen = {n: max(grid, key=lambda cell: grid[cell][n]) for n in names}

    test_data = make_feature_data(train_x, train_y, test_x, test_y)
    rows, cache = [], {}
    for name, cell in chosen.items():
        if cell not in cache:
            cache[cell] = compare(test_data, args.test_seeds, *cell)
        s = cache[cell][name]
        rows.append(
            {
                "name": name,
                "feature_power": cell[0],
                "learning_rate": cell[1],
                "validation_accuracy": grid[cell][name],
                **{m: s[m] for m in METRICS},
            }
        )
        print(
            f"{name:20s} p={cell[0]:g} lr={cell[1]:g} val={grid[cell][name]:.3f} "
            f"test={s['final_mean_accuracy']['mean']:.3f}+/-{s['final_mean_accuracy']['std']:.3f} "
            f"current={s['final_current_accuracy']['mean']:.3f} "
            f"forgetting={s['mean_forgetting']['mean']:.3f}"
        )
    payload = {
        "experiment": args.experiment,
        "protocol": {**vars(args), "output": str(args.output), **data_settings},
        "validation_grid": [
            {"feature_power": c[0], "learning_rate": c[1], "final_mean_accuracy": v}
            for c, v in grid.items()
        ],
        "selected": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
