# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Polynomial feature primitives shared by the binary and class-incremental settings.

Only what both settings use lives here:

* ``monomial_exponents``: the monomials of total degree <= d in any number of variables,
  in a prefix-stable order;
* ``append_zero_coefficients``: zero-initialised expansion of a coefficient array;
* ``validate_degree``: the common degree check.

Ordering: by total degree k = 0..d, and within a degree lexicographic over the variable
indices, i.e. by decreasing power of the first variable. For two variables:
    degree 0: 1
    degree 1: x1, x2
    degree 2: x1^2, x1*x2, x2^2
    degree 3: x1^3, x1^2*x2, x1*x2^2, x2^3 ...
Consequence: the features of degree d are a *prefix* of those of degree d+1, so a model
can be expanded by appending zero coefficients without touching existing ones.

What is *not* shared is the feature map built on top of the monomials: the binary setting
uses raw 2-D monomials (``polynomial_cl.models.binary``), the class-incremental setting
degree-balanced scaled monomials of 16 variables (``polynomial_cl.continual.features``).

Note: this is explicit feature expansion (related to polynomial kernels). It is NOT an
implicit-kernel SVM.
"""

from __future__ import annotations

import itertools

import numpy as np


def validate_degree(degree: int) -> int:
    """Require a non-negative integer degree; booleans are not valid degrees."""
    if isinstance(degree, (bool, np.bool_)) or not isinstance(
        degree, (int, np.integer)
    ):
        raise TypeError("degree must be a non-negative integer")
    if degree < 0:
        raise ValueError("degree must be >= 0")
    return int(degree)


def monomial_exponents(n_variables: int, degree: int) -> np.ndarray:
    """Exponent rows of all monomials of total degree <= degree, in prefix-stable order.

    Shared by the binary (2-variable) and class-incremental (n-variable) experiments.
    Order: by total degree; within a degree, lexicographic over variable indices, i.e.
    decreasing power of the first variable. For two variables this is exactly the order
    documented in the module docstring, so degree-d features are a prefix of degree-(d+1).
    """
    degree = validate_degree(degree)
    if n_variables < 1:
        raise ValueError("n_variables must be positive")
    rows = [np.zeros(n_variables, dtype=np.int16)]
    for total_degree in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(
            range(n_variables), total_degree
        ):
            rows.append(np.bincount(indices, minlength=n_variables).astype(np.int16))
    return np.stack(rows)


def append_zero_coefficients(weights: np.ndarray, new_feature_count: int) -> np.ndarray:
    """Copy of ``weights`` with zero rows appended up to ``new_feature_count`` rows.

    Every existing coefficient is copied unchanged and the new ones are zero, so a model
    built on a prefix-compatible feature map computes exactly the same outputs. Works for a
    coefficient vector (binary model) and a coefficient matrix (softmax head) alike. Always
    returns a new array.
    """
    weights = np.asarray(weights)
    if new_feature_count < weights.shape[0]:
        raise ValueError("feature expansion cannot shrink the weight array")
    expanded = np.zeros((new_feature_count, *weights.shape[1:]), dtype=weights.dtype)
    expanded[: weights.shape[0]] = weights
    return expanded
