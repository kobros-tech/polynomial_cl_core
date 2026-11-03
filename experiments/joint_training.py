# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Joint-training reference: all data available at once (NOT a continual-learning run).

``--experiment binary``     pooled training on the binary synthetic target (sinusoidal or
                            circular) for several polynomial degrees and epoch budgets;
``--experiment ten_class``   the ten-class features (``--dataset split_mnist`` or ``synthetic_2d``)
                            trained the ordinary way.

No forgetting is computed: there is no sequence of experiences. All logic lives in the package
(``run_joint_study``, ``run_joint_comparison``); this script parses arguments and writes JSON.
"""

import argparse
import json
from pathlib import Path

from common import add_dataset_arguments, load_dataset

from polynomial_cl.binary import make_experiences, run_joint_study
from polynomial_cl.continual import make_feature_data, run_joint_comparison
from polynomial_cl.data import circular_target, sinusoidal_target

TARGETS = {"sinusoidal": sinusoidal_target, "circular": circular_target}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=("binary", "ten_class"), default="binary")
    p.add_argument("--output", type=Path, help="write the JSON result here")
    binary = p.add_argument_group("binary synthetic target")
    binary.add_argument("--target", choices=sorted(TARGETS), default="sinusoidal")
    binary.add_argument("--seed", type=int, default=0)
    binary.add_argument(
        "--n-train", type=int, default=200, help="training samples per experience"
    )
    binary.add_argument(
        "--n-test", type=int, default=500, help="test samples per experience"
    )
    binary.add_argument(
        "--label-noise", type=float, default=0.0, help="training label-flip probability"
    )
    binary.add_argument(
        "--binary-degrees", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10]
    )
    binary.add_argument("--binary-epochs", type=int, nargs="+", default=[5000])
    binary.add_argument("--binary-lr", type=float, default=0.5)
    mnist = p.add_argument_group("ten-class (Split MNIST features)")
    add_dataset_arguments(mnist, required=False)
    mnist.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3])
    mnist.add_argument("--epochs", type=int, nargs="+", default=[3, 10, 30])
    mnist.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    mnist.add_argument("--learning-rate", type=float, default=0.1)
    args = p.parse_args()
    if args.experiment == "ten_class" and args.dataset is None:
        p.error("--experiment ten_class requires --dataset")

    if args.experiment == "binary":
        experiences = make_experiences(
            seed=args.seed,
            n_train=args.n_train,
            n_test=args.n_test,
            noise=args.label_noise,
            target_function=TARGETS[args.target],
        )
        rows = run_joint_study(
            experiences,
            args.binary_degrees,
            args.binary_epochs,
            args.binary_lr,
            seed=args.seed,
        )
        for row in rows:
            for key in (
                "loss_history",
                "model",
            ):  # for notebooks; the JSON keeps the scalars
                row.pop(key)
            print(
                f"degree {row['degree']:2d} epochs {row['epochs']:5d}: "
                f"mean accuracy {row['mean_accuracy']:.3f} (min {row['min_accuracy']:.3f})"
            )
        result = {"protocol": {**vars(args), "output": None}, "rows": rows}
    else:
        train_x, train_y, test_x, test_y, protocol = load_dataset(args)
        data = make_feature_data(train_x, train_y, test_x, test_y)
        rows = run_joint_comparison(
            data,
            args.degrees,
            args.epochs,
            args.seeds,
            learning_rate=args.learning_rate,
        )
        for row in rows:
            print(
                f"degree {row['degree']} epochs {row['epochs']:3d}: "
                f"test accuracy {row['test_accuracy_mean']:.3f}"
            )
        result = {"protocol": {**protocol, "joint_reference": True}, "rows": rows}

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
