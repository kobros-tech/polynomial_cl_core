# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""polynomial_cl: a small NumPy research instrument for polynomial expansion in continual learning.

* ``polynomial_cl.continual``: the primary setting, class-incremental learning with ten classes
  arriving in five two-class experiences and one softmax head.
* ``polynomial_cl.binary``: sequential *binary* learning under spatial distribution shift.
  This is not class-incremental learning.
* ``polynomial_cl.features`` and ``polynomial_cl.metrics``: the polynomial and metric
  primitives both settings share. ``models`` holds the binary model and the multiclass
  (softmax-head) model, ``training`` their two separate optimisers.
"""

from . import binary, continual, data, features, metrics, models, training
from .features import append_zero_coefficients, monomial_exponents

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "append_zero_coefficients",
    "binary",
    "continual",
    "data",
    "features",
    "metrics",
    "models",
    "monomial_exponents",
    "training",
]
