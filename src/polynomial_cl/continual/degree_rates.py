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


def _numeric_array(values, what: str) -> np.ndarray:
    """``values`` as a float64 array, after rejecting anything that is not a real number.

    The dtype is checked *before* the cast: ``astype(float64)`` would turn booleans into 0/1 and
    silently drop the imaginary part of complex values, so a check made afterwards sees neither.
    """
    try:
        flat = np.ravel(np.asarray(values, dtype=object))
        array = np.asarray(values)
    except ValueError as error:  # e.g. ragged nested lists
        raise ValueError(f"{what} must be a regular array of numbers") from error
    if any(isinstance(v, (bool, np.bool_)) for v in flat):
        raise TypeError(f"{what} must be numbers, not booleans")
    if array.dtype == bool or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{what} must be numbers")
    if np.iscomplexobj(array):
        raise TypeError(f"{what} must be real numbers, not complex")
    return array.astype(np.float64)


def as_degree_rates(rates, n_groups: int = N_DEGREE_GROUPS) -> np.ndarray:
    """Validated float64 vector with one rate per degree group (a scalar is broadcast)."""
    values = _numeric_array(rates, "learning rates")
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


def feature_rates(effective_rates, group_index: np.ndarray) -> np.ndarray:
    """The diagonal ``d`` of the learning-rate matrix ``D``: one float64 rate per feature.

    ``d_j = effective_rates[deg(j)]`` where ``effective_rates`` already holds one rate per
    degree group (``eta_d * c_d``, see ``apply_degree_control``) and ``group_index[j] = deg(j)``.
    ``D = diag(d)`` is never materialised: every consumer needs only ``d`` (SGD step ``d * g``)
    or ``sqrt(d)`` (null-space step ``sqrt(d) * [P + beta (I - P)] * sqrt(d) * g``).

    This is the single definition of the effective per-feature rate. It is prefix-stable, so
    after an expansion the rates of the new features are the entries of ``group_index`` that
    appear for the first time and the old features keep theirs. A rate of 0 is valid (that
    feature never moves); an all-zero vector is valid here too (a no-op); non-finite, negative
    or boolean or complex values, a vector longer than the number of degree groups, or a degree
    beyond the vector are rejected.
    """
    values = _numeric_array(effective_rates, "learning rates")
    if values.ndim != 1:
        raise ValueError("effective rates must be a vector with one entry per degree")
    if len(values) > N_DEGREE_GROUPS:
        raise ValueError(
            f"effective rates have {len(values)} entries but there are only "
            f"{N_DEGREE_GROUPS} degree groups (degrees 0..{N_DEGREE_GROUPS - 1})"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("learning rates must be finite and non-negative")
    group_index = np.asarray(group_index)
    if not np.issubdtype(group_index.dtype, np.integer) or group_index.ndim != 1:
        raise ValueError("group_index must be a 1-D integer array of degrees")
    if group_index.size and (group_index.min() < 0 or group_index.max() >= len(values)):
        raise ValueError(
            f"feature degrees must lie in 0..{len(values) - 1} to index the rate vector"
        )
    return values[group_index]


def diagonal_learning_rates(rates, group_index: np.ndarray, control=None) -> np.ndarray:
    """``d_j = eta_{deg(j)} * c_{deg(j)}`` from base rates and optional degree control.

    ``control=None`` (or all ones) gives exactly ``rates[group_index]``: degree control is the
    identity unless a factor is below 1.
    """
    effective = _numeric_array(rates, "learning rates")
    if control is not None:
        effective = apply_degree_control(effective, control)
    return feature_rates(effective, group_index)


def step_scale_for(
    rates: np.ndarray, group_index: np.ndarray
) -> tuple[float, np.ndarray]:
    """Split a rate vector into (base rate, per-feature multiplier) for the trainer.

    The effective rate of feature i is ``base * multiplier[i] = feature_rates(...)[i]``; the
    base is the largest rate among the active groups, so the multipliers lie in [0, 1].
    """
    active = rates[: int(group_index.max()) + 1]
    if not np.any(active > 0):
        raise ValueError(
            "every active degree group has rate 0; nothing would be trained"
        )
    base = float(active.max())
    return base, (feature_rates(rates, group_index) / base).astype(np.float32)


DEFAULT_DEGREE_CONTROL = (1.0,) * N_DEGREE_GROUPS


def as_degree_control(factors, n_groups: int = N_DEGREE_GROUPS) -> np.ndarray:
    """Validated float64 degree-control vector ``m`` (one factor in [0, 1] per degree group).

    ``M = diag(m_0 I_0, ..., m_D I_D)`` multiplies the degree-wise rate matrix ``H``: the
    effective rate of group d is ``rates[d] * m[d]``. 0 freezes the group, 1 keeps the
    original update. Unlike ``as_degree_rates`` the factors are never broadcast and an
    all-zero vector is allowed (it simply trains nothing).
    """
    values = _numeric_array(factors, "degree-control factors")
    if values.shape != (n_groups,):
        raise ValueError(
            f"degree-control factors must have shape ({n_groups},) "
            f"(degrees 0..{n_groups - 1}), got {values.shape}"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("degree-control factors must be finite and within [0, 1]")
    return values.copy()


def is_identity_control(factors) -> bool:
    """True if every factor is exactly 1 (the control then changes nothing)."""
    return bool(np.all(np.asarray(factors) == 1.0))


def apply_degree_control(rates: np.ndarray, factors) -> np.ndarray:
    """Effective per-degree rates ``H M``: elementwise product, may contain zeros."""
    return _numeric_array(rates, "learning rates") * as_degree_control(factors)
