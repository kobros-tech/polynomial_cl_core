# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Class-incremental comparison (fixed_linear / expanding / fixed_capacity) -> JSON.

Datasets: ``synthetic_2d`` (ten Gaussian classes on a ring) or ``split_mnist`` (MNIST digits
under a fixed 16-dimensional random projection). See polynomial_cl/continual/experiment.py
for the protocol; all logic lives in the package, this script only parses arguments and
writes the result. Argument and dataset helpers shared with the other runners: common.py.
"""

import argparse
import json
from pathlib import Path

from common import add_dataset_arguments, load_dataset

from polynomial_cl.continual import (
    TASK_ORDER,
    holdout_split,
    make_feature_data,
    run_comparison,
    run_strategy_comparison,
)
from polynomial_cl.continual.experiment import (
    BATCH_SIZE,
    DEGREES,
    EPOCHS,
    LEARNING_RATE,
    MEMORY_PER_CLASS,
    N_REF_PER_CLASS,
    STRATEGIES,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(p)
    p.add_argument(
        "--experiment",
        choices=("conditions", "strategies"),
        default="conditions",
        help="conditions: fixed_linear/expanding/fixed_capacity; strategies: update rules on 'expanding'",
    )
    p.add_argument(
        "--n-ref-per-class", type=int, default=N_REF_PER_CLASS, help="strategy_b only"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--memory-per-class", type=int, default=MEMORY_PER_CLASS)
    p.add_argument(
        "--feature-power",
        type=float,
        default=0.0,
        help="per-feature step sizes lr*(max_rms/rms_i)**power; 0 = one shared rate",
    )
    p.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="hold out this fraction of the training set and evaluate on it, not on the test set",
    )
    args = p.parse_args()
    if not args.seeds:
        p.error("at least one seed is required")
    if min(args.epochs, args.batch_size, args.memory_per_class) < 1:
        p.error("epochs, batch-size and memory-per-class must be positive")

    train_x, train_y, test_x, test_y, data_settings = load_dataset(args)
    if args.validation_fraction:
        # Train on part of the training set and report on the held-out part (test set unused).
        train_x, train_y, test_x, test_y = holdout_split(
            train_x, train_y, args.validation_fraction
        )
        data_settings["evaluated_on"] = f"validation split ({args.validation_fraction})"
    settings = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "memory_per_class": args.memory_per_class,
        "feature_power": args.feature_power,
    }
    data = make_feature_data(train_x, train_y, test_x, test_y)
    if args.experiment == "strategies":
        result = run_strategy_comparison(
            data, args.seeds, n_ref_per_class=args.n_ref_per_class, **settings
        )
    else:
        result = run_comparison(data, args.seeds, **settings)
    payload = {
        "dataset": args.dataset,
        "protocol": {
            "seeds": args.seeds,
            "task_order": [list(pair) for pair in TASK_ORDER],
            "epochs_per_experience": args.epochs,
            "learning_rate": args.lr,
            "feature_power": args.feature_power,
            "batch_size": args.batch_size,
            "memory_per_class": args.memory_per_class,
            "experiment": args.experiment,
            "degrees_by_condition": {name: list(d) for name, d in DEGREES.items()},
            "strategies": list(STRATEGIES) if args.experiment == "strategies" else None,
            "n_ref_per_class": args.n_ref_per_class,
            **data_settings,
        },
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
