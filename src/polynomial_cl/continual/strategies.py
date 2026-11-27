# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Update methods of the class-incremental experiments (NumPy only).

A *method* decides how the gradient of one experience is turned into a step:

* ``ordinary``: the rate-weighted step ``D G`` (``stable_cl.ordinary_step``); nothing is
  restricted;
* ``relaxed``: the relaxed null-space step ``D^(1/2) [P + beta (I - P)] D^(1/2) G``
  (``stable_cl.relaxed_step``), where ``P`` projects onto the null space of the design matrix
  of a few stored reference inputs per old class. ``beta = 0`` (the default) keeps those
  logits unchanged to first order and nowhere else is anything guaranteed; ``beta = 1`` is
  the ordinary step.

Whether replay examples are added to the training set is **not** a method: it is the
protocol parameter ``replay`` of ``run_condition``.

``update_restrictions`` builds the constraints of one experience. The null-space basis does
not depend on ``beta``, so ``beta`` is applied by ``fit_task`` only. The training loop, the
step-size scaling and the indicators stay in ``run_condition``.

The pre-C4 strategies ``frozen_old``, ``small_old_lr``, ``strategy_b_new`` and
``ordinary_no_replay`` were removed; the ``strategy_b`` of earlier results is
``method="relaxed", beta=0.0``. ``fit_task`` keeps its low-level ``update_scale`` and
``null_groups`` arguments (covered by the C2 fixtures); no method produces them any more.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import stable_cl

from .replay import stored_indices

METHODS = ("ordinary", "relaxed")
N_REF_PER_CLASS = (
    3  # relaxed: reference inputs per old class (first stored replay examples)
)

# Names of earlier API generations. They are never accepted by the current API; the legacy
# study loader reads them (see degree_study.load_legacy_study_cells).
LEGACY_STRATEGY_NAMES = (
    "ordinary",
    "ordinary_no_replay",
    "small_old_lr",
    "frozen_old",
    "strategy_b",
    "strategy_b_new",
)


def check_method(method) -> str:
    """Return ``method`` if it is a current method, else raise ``ValueError``."""
    if method not in METHODS:
        hint = (
            " (an earlier API generation: relaxed with beta=0 replaces strategy_b, "
            "replay=False replaces ordinary_no_replay; old results load through "
            "load_legacy_study_cells)"
            if method in LEGACY_STRATEGY_NAMES
            else ""
        )
        raise ValueError(f"unknown method: {method!r}; expected one of {METHODS}{hint}")
    return method


def check_beta(beta, method="relaxed") -> float:
    """Validate ``beta`` in [0, 1]; ``method='ordinary'`` has no beta and accepts only 0."""
    if isinstance(beta, (bool, np.bool_)) or not isinstance(
        beta, (int, float, np.integer, np.floating)
    ):
        raise TypeError("beta must be a real number")
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:  # also rejects NaN
        raise ValueError("beta must be in [0, 1]")
    if method == "ordinary" and beta != 0.0:
        raise ValueError(
            "beta applies to method='relaxed' only (relaxed with beta=1 is the ordinary "
            "update)"
        )
    return beta


class UpdateRestrictions(NamedTuple):
    """What ``fit_task`` receives: any of the four fields may be None (unrestricted)."""

    scale: (
        np.ndarray | None
    )  # per-coefficient step multiplier (no current method sets it)
    null_basis: np.ndarray | None  # one null-space basis shared by every output column
    refs: np.ndarray | None  # indices of the reference inputs behind the null space
    null_groups: list | None  # [(column slice, basis)] (no current method sets it)


def update_restrictions(
    method,
    *,
    old_classes,
    count,
    task,
    train_x,
    memory_indices,
    n_ref_per_class,
    step_scale,
    base_rate=1.0,
) -> UpdateRestrictions:
    """Constraints of ``method`` for experience ``task`` (0-based).

    ``old_classes`` are the class columns that existed before this experience; ``step_scale``
    (or None) is the per-feature step multiplier ``m`` and ``base_rate`` the learning-rate
    factor ``eta``. The null space is that of the reference design scaled by the effective
    rates ``d = eta * m`` (float64), as in ``stable_cl``; it does not depend on the positive
    factor ``eta``, which is passed for completeness, nor on ``beta``.
    """
    check_method(method)
    if method == "ordinary" or not task:
        return UpdateRestrictions(None, None, None, None)
    rates = np.full(count, float(base_rate))
    if step_scale is not None:
        rates = rates * np.asarray(step_scale, dtype=np.float64)
    refs = stored_indices(memory_indices, old_classes, n_ref_per_class)
    ref_design = train_x[refs, :count].astype(np.float64)
    null_basis = stable_cl.null_basis(ref_design, rates).astype(np.float32)
    return UpdateRestrictions(None, null_basis, refs, None)
