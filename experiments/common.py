# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Command-line helpers shared by the class-incremental experiment scripts.

Only argument definitions and dataset loading live here. The scripts in this directory are
entry points: they run as ``python experiments/<name>.py`` from the repository root, which puts
this directory on ``sys.path``, and they import this module as ``common``.
"""

from polynomial_cl.data import load_projected_mnist, make_synthetic_2d
from polynomial_cl.data.mnist import N_COMPONENTS


def add_dataset_arguments(p, required=True):
    p.add_argument(
        "--dataset", choices=("synthetic_2d", "split_mnist"), required=required
    )
    mnist = p.add_argument_group("split_mnist")
    mnist.add_argument("--data-dir", default="data/mnist")
    mnist.add_argument("--projection-seed", type=int, default=0)
    mnist.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="require the MNIST archives to exist locally; never touch the network",
    )
    synthetic = p.add_argument_group("synthetic_2d")
    synthetic.add_argument("--data-seed", type=int, default=0)
    synthetic.add_argument("--train-per-class", type=int, default=160)
    synthetic.add_argument("--test-per-class", type=int, default=240)
    synthetic.add_argument("--separation", type=float, default=2.5)
    synthetic.add_argument("--noise", type=float, default=0.45)


def load_dataset(args):
    """Arrays (train_x, train_y, test_x, test_y) plus a JSON-able description of the data."""
    if args.dataset == "synthetic_2d":
        train_x, train_y, test_x, test_y, means = make_synthetic_2d(
            args.data_seed,
            args.train_per_class,
            args.test_per_class,
            args.separation,
            args.noise,
        )
        settings = {
            "data_seed": args.data_seed,
            "train_per_class": args.train_per_class,
            "test_per_class": args.test_per_class,
            "class_separation": args.separation,
            "class_noise_std": args.noise,
            "class_means": means.tolist(),
        }
    else:
        train_x, train_y, test_x, test_y = load_projected_mnist(
            args.data_dir, args.projection_seed, download=args.download
        )
        settings = {
            "projection_seed": args.projection_seed,
            "projection_dimensions": N_COMPONENTS,
            "projection": "fixed orthonormal random projection; no data fitting",
        }
    return train_x, train_y, test_x, test_y, settings
