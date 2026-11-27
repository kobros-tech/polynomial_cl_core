# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""FROZEN copy of the pre-C4 ``update_restrictions`` (all six legacy strategies).

Test support only. C4 removed ``frozen_old``, ``small_old_lr``, ``strategy_b_new`` and
``ordinary_no_replay`` from the public API. The C2 golden fixtures
(``tests/golden/c2_multiclass_*.json``) were generated with these exact constructions, and
``fit_task`` still accepts ``update_scale`` and ``null_groups``, so the equivalence tests keep
building the same restrictions here instead of weakening the fixtures. The code below is the
former ``polynomial_cl/continual/strategies.py`` unchanged apart from this docstring and the
import of ``stored_indices``; do not edit it. The same code is recoverable from the
``legacy-strategies`` tag.

Original docstring: update restrictions of the six class-incremental strategies (NumPy only).

A strategy decides which coefficients may move during one experience and how far:

* ``ordinary`` / ``ordinary_no_replay``: nothing is restricted (the latter also has no replay,
  which is decided by the training loop in ``experiment.py``);
* ``small_old_lr`` / ``frozen_old``: coefficients whose feature row *and* class column already
  existed take a step scaled by ``OLD_LR_FACTOR`` / 0;
* ``strategy_b``: every coefficient may move, but only inside the null space of the design
  matrix of a few stored reference inputs per old class (their logits stay unchanged to first
  order, nowhere else is anything guaranteed);
* ``strategy_b_new``: old coefficients are frozen and the new ones move inside that null space,
  with a separate basis for the old-class columns and for the new-class columns.

``update_restrictions`` builds these constraints for one experience. The training loop, the
step-size scaling and the indicators stay in ``run_condition``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import stable_cl

from polynomial_cl.continual.replay import stored_indices

# Update rules compared on the ``expanding`` schedule (same definitions as the binary
# experiments in polynomial_cl.binary.experiment, but the code is independent). A coefficient
# is "old" if it existed before the current experience: its feature row was already present
# AND its class column was already introduced. New feature rows and the new classes'
# columns always train freely.
# ``strategy_b`` is the binary ``strategy_b_all`` (every coefficient may move inside the null
# space of the reference inputs); ``strategy_b_new`` is the binary ``strategy_b_new``: old
# coefficients are frozen and only new ones move, also inside that null space.
STRATEGIES = (
    "ordinary",
    "ordinary_no_replay",
    "small_old_lr",
    "frozen_old",
    "strategy_b",
    "strategy_b_new",
)
OLD_LR_FACTOR = 0.1
N_REF_PER_CLASS = (
    3  # strategy_b: reference inputs per old class (first stored replay examples)
)


class UpdateRestrictions(NamedTuple):
    """What ``fit_task`` receives: any of the four fields may be None (unrestricted)."""

    scale: np.ndarray | None  # per-coefficient step multiplier, shape of the weights
    null_basis: np.ndarray | None  # one null-space basis shared by every output column
    refs: np.ndarray | None  # indices of the reference inputs behind the null space(s)
    null_groups: (
        list | None
    )  # [(column slice, basis)] with a separate basis per column group


def update_restrictions(
    strategy,
    *,
    weights_shape,
    old_rows,
    old_classes,
    n_seen,
    count,
    task,
    train_x,
    memory_indices,
    n_ref_per_class,
    step_scale,
    base_rate=1.0,
) -> UpdateRestrictions:
    """Constraints of ``strategy`` for experience ``task`` (0-based).

    ``old_rows`` / ``old_classes`` are the feature rows and class columns that existed before
    this experience; ``step_scale`` (or None) is the per-feature step multiplier ``m`` and
    ``base_rate`` the learning-rate factor ``eta``. The null space is that of the reference
    design scaled by the effective rates ``d = eta * m`` (float64), as in ``stable_cl``; it does
    not depend on the positive factor ``eta``, which is passed for completeness.
    """
    scale = null_basis = refs = null_groups = None
    rates = np.full(count, float(base_rate))
    if step_scale is not None:
        rates = rates * np.asarray(step_scale, dtype=np.float64)

    def basis_of(ref_design, columns=slice(None)):
        """Null basis of the reference design with effective rates ``rates``, as float32."""
        basis = stable_cl.null_basis(ref_design[:, columns], rates[columns])
        return basis.astype(np.float32)

    if strategy in ("frozen_old", "small_old_lr"):
        scale = np.ones(weights_shape, np.float32)
        scale[:old_rows, :old_classes] = (
            0.0 if strategy == "frozen_old" else OLD_LR_FACTOR
        )
    elif strategy == "strategy_b_new" and task:
        scale = np.ones(weights_shape, np.float32)
        scale[:old_rows, :old_classes] = 0.0
        refs = stored_indices(memory_indices, old_classes, n_ref_per_class)
        ref_design = train_x[refs, :count].astype(np.float64)
        # Old-class columns may only move through the NEW feature rows; the null space is
        # taken inside those rows. New-class columns see every row (nothing there is old),
        # as in strategy_b, so reference inputs also keep their new-class logits.
        rows = np.arange(old_rows, count)
        old_basis = np.zeros((count, 0), np.float32)
        if len(rows):
            rows_basis = basis_of(ref_design, rows)
            old_basis = np.zeros((count, rows_basis.shape[1]), np.float32)
            old_basis[rows] = rows_basis
        null_groups = [
            (slice(0, old_classes), old_basis),
            (slice(old_classes, n_seen), basis_of(ref_design)),
        ]
    elif strategy == "strategy_b" and task:
        refs = stored_indices(memory_indices, old_classes, n_ref_per_class)
        ref_design = train_x[refs, :count].astype(np.float64)
        null_basis = basis_of(ref_design)
    return UpdateRestrictions(scale, null_basis, refs, null_groups)
