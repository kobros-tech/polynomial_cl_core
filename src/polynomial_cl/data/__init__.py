# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Data generation and loading, independent of how a experiment cuts data into experiences.

* ``synthetic``: target functions (sinusoid, circle), reproducible sampling and labelling, and
  the ten-class ring used by the class-incremental protocol check.
* ``mnist``: MNIST loading and the fixed random projection.
"""

from .mnist import N_COMPONENTS, load_mnist, load_projected_mnist
from .synthetic import (
    DOMAIN,
    boundary_curve,
    circular_target,
    make_circular_target,
    make_synthetic_2d,
    make_synthetic_dataset,
    sinusoidal_target,
    true_label,
)

__all__ = [
    "DOMAIN",
    "N_COMPONENTS",
    "boundary_curve",
    "circular_target",
    "load_mnist",
    "load_projected_mnist",
    "make_circular_target",
    "make_synthetic_2d",
    "make_synthetic_dataset",
    "sinusoidal_target",
    "true_label",
]
