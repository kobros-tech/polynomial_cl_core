# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Split MNIST: sampling-policy comparison and the per-degree learning-rate study -> JSON.

``--experiment sampling``: every strategy under ``shuffled`` and ``class_balanced`` sampling
with one shared rate (test data, seeds ``--seeds``).

``--experiment epochs``: re-runs the shared reference and the selected vector of every cell found
in ``--cells-dir`` (``cell_*.json`` from the per-degree experiment) at ``--epochs`` epochs on the
test seeds, without re-selecting rates.

``--experiment per-degree``: for every (strategy, sampling) the rate vectors are selected on a
validation split with ``--select-seeds`` and then run on the test set with ``--test-seeds``
(disjoint). See polynomial_cl/continual/degree_study.py for the protocol; this script
only parses arguments and writes the result.
"""

import argparse
import json
from pathlib import Path

from common import add_dataset_arguments, load_dataset

from polynomial_cl.continual import (
    holdout_split,
    load_study_cells,
    make_feature_data,
    run_epoch_robustness,
    run_per_degree_study,
    run_sampling_comparison,
)
from polynomial_cl.continual.experiment import (
    BATCH_SIZE,
    EPOCHS,
    MEMORY_PER_CLASS,
    N_REF_PER_CLASS,
    STRATEGIES,
)
from polynomial_cl.continual.sampling import SAMPLING_POLICIES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(p)
    p.add_argument(
        "--experiment", choices=("sampling", "per-degree", "epochs"), required=True
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES
    )
    p.add_argument(
        "--samplings",
        nargs="+",
        default=list(SAMPLING_POLICIES),
        choices=SAMPLING_POLICIES,
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="sampling experiment",
    )
    p.add_argument(
        "--lr", type=float, default=0.1, help="sampling experiment: shared rate"
    )
    p.add_argument("--select-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--test-seeds", nargs="+", type=int, default=[3, 4, 5, 6, 7, 8, 9])
    p.add_argument(
        "--min-current-accuracy", type=float, default=None,
        help="per-degree: validation floor on newest-experience accuracy for selection",
    )  # fmt: skip
    p.add_argument("--cells-dir", type=Path, default=Path("results/per_degree"))
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--memory-per-class", type=int, default=MEMORY_PER_CLASS)
    p.add_argument("--n-ref-per-class", type=int, default=N_REF_PER_CLASS)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.memory_per_class) < 1:
        p.error("epochs, batch-size and memory-per-class must be positive")
    shared = sorted(set(args.select_seeds) & set(args.test_seeds))
    if args.experiment == "per-degree" and shared:
        p.error(f"--select-seeds and --test-seeds overlap on {shared}")

    train_x, train_y, test_x, test_y, data_settings = load_dataset(args)
    settings = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "memory_per_class": args.memory_per_class,
        "n_ref_per_class": args.n_ref_per_class,
    }

    def log(message):
        print(message, flush=True)

    protocol = {**vars(args), "output": str(args.output), **data_settings}
    if args.experiment == "sampling":
        data = make_feature_data(train_x, train_y, test_x, test_y)
        result = run_sampling_comparison(
            data,
            args.seeds,
            policies=args.samplings,
            strategies=args.strategies,
            learning_rate=args.lr,
            log=log,
            **settings,
        )
    elif args.experiment == "epochs":
        study = load_study_cells(args.cells_dir)
        if study is None:
            p.error(f"no cell_*.json files in {args.cells_dir}")
        result = run_epoch_robustness(
            make_feature_data(train_x, train_y, test_x, test_y),
            study,
            args.epochs,
            batch_size=args.batch_size,
            memory_per_class=args.memory_per_class,
            n_ref_per_class=args.n_ref_per_class,
            strategies=args.strategies,
            samplings=args.samplings,
            log=log,
        )
    else:
        fit = holdout_split(train_x, train_y, args.validation_fraction)
        result = run_per_degree_study(
            make_feature_data(*fit),
            make_feature_data(train_x, train_y, test_x, test_y),
            args.select_seeds,
            args.test_seeds,
            strategies=args.strategies,
            samplings=args.samplings,
            min_current_accuracy=args.min_current_accuracy,
            log=log,
            progress=lambda c: print(
                f"{c['sampling']:15s} {c['strategy']:20s} base={c['base_rate']:g} "
                f"selected={c['selected']} pareto={c['pareto']}",
                flush=True,
            ),
            **settings,
        )
    payload = {"dataset": args.dataset, "protocol": protocol, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=1, allow_nan=False, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
