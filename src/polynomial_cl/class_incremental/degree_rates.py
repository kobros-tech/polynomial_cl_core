# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Per-degree learning rates: one step size for each polynomial degree group.

A *degree group* is the set of monomials of one total degree. For ``y = a*x^2 + b*x + c``
the groups are degree 0 (``c``), degree 1 (``b``) and degree 2 (``a``). A rate vector ``r``
has one entry per group and **position k is degree k**: ``r[0]`` is the constant feature,
``r[1]`` the linear monomials, ``r[2]`` the quadratic ones, ``r[3]`` the cubic ones. Its
length is always ``N_DEGREE_GROUPS`` (``MAX_DEGREE + 1``); a model whose active degree is
smaller simply never uses the trailing entries, and a degree group that appears through
expansion starts training with its own entry from the first update.

This is *not* ``feature_power``. ``feature_power`` derives one step size **per feature**
from the measured feature scale (a diagonal preconditioner); a rate vector is chosen by hand
and is constant within a degree group. The two mechanisms are different arguments of the
trainer and are never combined silently (``run_condition`` rejects the combination).

Zero-rate semantics: a rate of 0 is allowed and freezes that whole group (its coefficients
are never updated). A vector whose active entries are all zero would train nothing and is
rejected, as are negative, non-finite and boolean entries.
"""

from __future__ import annotations

import numpy as np

from ..features import monomial_exponents
from .features import MAX_DEGREE

N_DEGREE_GROUPS = MAX_DEGREE + 1


def as_degree_rates(rates, n_groups: int = N_DEGREE_GROUPS) -> np.ndarray:
    """Validated float64 vector with one rate per degree group (a scalar is broadcast)."""
    values = np.asarray(rates)
    if values.dtype == bool or not np.issubdtype(values.dtype, np.number):
        raise TypeError("learning rates must be numbers")
    values = values.astype(np.float64)
    if values.ndim == 0:
        values = np.full(n_groups, float(values))
    elif values.shape != (n_groups,):
        raise ValueError(
            f"learning rate must be a scalar or have shape ({n_groups},) "
            f"(degrees 0..{n_groups - 1}), got {values.shape}"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("learning rates must be finite and non-negative")
    if not np.any(values > 0):
        raise ValueError("at least one learning rate must be positive")
    return values.copy()


def degree_of_feature(n_components: int, degree: int) -> np.ndarray:
    """Total degree of every feature column (prefix-stable, length n_polynomial_features)."""
    return monomial_exponents(n_components, degree).sum(axis=1).astype(np.int64)


def group_sizes(group_index: np.ndarray, n_groups: int = N_DEGREE_GROUPS) -> np.ndarray:
    """Number of feature rows in every degree group (0 for groups that are not active)."""
    return np.bincount(group_index, minlength=n_groups)


def step_scale_for(
    rates: np.ndarray, group_index: np.ndarray
) -> tuple[float, np.ndarray]:
    """Split a rate vector into (base rate, per-feature multiplier) for the trainer.

    The effective rate of feature i is ``base * multiplier[i] = rates[group_index[i]]``; the
    base is the largest rate among the active groups, so the multipliers lie in [0, 1].
    """
    active = rates[: int(group_index.max()) + 1]
    if not np.any(active > 0):
        raise ValueError(
            "every active degree group has rate 0; nothing would be trained"
        )
    base = float(active.max())
    return base, (rates[group_index] / base).astype(np.float32)
