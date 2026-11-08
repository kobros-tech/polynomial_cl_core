# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Degree-balanced polynomial features in any number of variables.

The monomial enumeration and the zero-coefficient expansion are shared with the binary
experiments (``polynomial_cl.features``). What is specific here is the *scaling*: every
positive-degree group is divided by the square root of its size, so that higher-degree
groups do not dominate merely because they contain more monomials, whereas the binary
experiments use raw monomials. Monomials are ordered by total degree, so the features of
degree d are a prefix of those of degree d + 1; a head can grow by appending zero
coefficients (``polynomial_cl.models.multiclass.expand_weights``) without changing its
outputs.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from ..features import monomial_exponents

MAX_DEGREE = 3


def n_polynomial_features(n_components: int, degree: int) -> int:
    """Number of monomials of total degree at most degree, including the constant."""
    if n_components < 1 or degree < 0:
        raise ValueError("n_components must be positive and degree non-negative")
    return math.comb(n_components + degree, degree)


@functools.lru_cache(maxsize=16)
def _monomial_plan(n_variables: int, degree: int):
    """Read-only layout of the degree-balanced design matrix.

    Returns ``(scales, factor_index, group_bounds)``:

    * ``scales`` (n_features,) float32: ``1`` for the constant, ``1/sqrt(group size)`` for
      every positive-degree monomial;
    * ``factor_index`` (n_features, max(degree, 1)) intp: for the monomial ``prod_v x_v^e_v``
      the entries ``1 + (e - 1) * n_variables + v`` of its non-zero factors in increasing
      variable order, padded with ``0`` (the constant-one column of the factor table);
    * ``group_bounds``: the ``degree + 2`` offsets delimiting the degree groups.
    """
    exponents = monomial_exponents(n_variables, degree)
    degree_of_feature = exponents.sum(axis=1)
    scales = np.ones(len(exponents), dtype=np.float32)
    for total_degree in range(1, degree + 1):
        group_size = math.comb(n_variables + total_degree - 1, total_degree)
        scales[degree_of_feature == total_degree] = 1.0 / math.sqrt(group_size)
    factor_index = np.zeros((len(exponents), max(degree, 1)), dtype=np.intp)
    for row, exponent_row in enumerate(exponents):
        for slot, variable in enumerate(np.flatnonzero(exponent_row)):
            power = int(exponent_row[variable])
            factor_index[row, slot] = 1 + (power - 1) * n_variables + int(variable)
    counts = np.bincount(degree_of_feature, minlength=degree + 1)
    group_bounds = np.concatenate(([0], np.cumsum(counts)))
    for array in (scales, factor_index, group_bounds):
        array.setflags(write=False)
    return scales, factor_index, group_bounds


def scaled_polynomial_features(
    x: np.ndarray, degree: int, chunk_size: int = 256
) -> np.ndarray:
    """Degree-balanced polynomial design matrix, built in chunks to bound peak memory.

    Each positive-degree group is divided by sqrt(number of monomials in that degree); the
    constant feature stays 1.

    Every monomial is a product of at most ``degree`` per-variable powers, so the powers
    ``x_v ** e`` are computed once per chunk (a table of ``degree * n_variables`` columns)
    and each monomial is then a product of gathered table columns, taken in increasing
    variable order. This never builds the ``(chunk, n_features, n_variables)`` array of
    the direct formula, which made construction the dominant cost of a run, and the result
    is bit-for-bit the same as that formula: the powers come from the same ``np.power``
    float32 loop, the factors are multiplied in the same order, and the padding factor
    ``1.0`` is exact.

    The work is memory-bound, not arithmetic-bound (``np.power`` is a small part of it), so the
    default chunk is small enough for the gathered temporaries to stay in cache: on Split MNIST
    256 rows is about 1.5x faster than 4096, and the result does not depend on the chunk size.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("x must have shape (n_samples, n_components)")
    if degree < 0 or chunk_size < 1:
        raise ValueError("degree must be non-negative and chunk_size positive")

    n_variables = x.shape[1]
    scales, factor_index, bounds = _monomial_plan(n_variables, degree)
    out = np.empty((len(x), len(scales)), dtype=np.float32)
    for start in range(0, len(x), chunk_size):
        block = x[start : start + chunk_size]
        table = np.empty((len(block), 1 + degree * n_variables), dtype=np.float32)
        table[:, 0] = 1.0
        for power in range(1, degree + 1):
            exponent = np.full(block.shape, power, dtype=np.float32)
            first = 1 + (power - 1) * n_variables
            table[:, first : first + n_variables] = np.power(block, exponent)
        target = out[start : start + len(block)]
        for total_degree in range(degree + 1):
            lo, hi = bounds[total_degree], bounds[total_degree + 1]
            product = table[:, factor_index[lo:hi, 0]]
            for slot in range(1, total_degree):
                product *= table[:, factor_index[lo:hi, slot]]
            target[:, lo:hi] = product * scales[lo:hi]
    return out
