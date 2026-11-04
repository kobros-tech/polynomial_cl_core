# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Update restrictions of the six class-incremental strategies (NumPy only).

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

from ..training.multiclass import null_space_basis
from .replay import stored_indices

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
) -> UpdateRestrictions:
    """Constraints of ``strategy`` for experience ``task`` (0-based).

    ``old_rows`` / ``old_classes`` are the feature rows and class columns that existed before
    this experience; ``step_scale`` (or None) is the per-feature step multiplier that is also
    applied to the reference design, so the null space is that of the effective update.
    """
    scale = null_basis = refs = null_groups = None
    if strategy in ("frozen_old", "small_old_lr"):
        scale = np.ones(weights_shape, np.float32)
        scale[:old_rows, :old_classes] = (
            0.0 if strategy == "frozen_old" else OLD_LR_FACTOR
        )
    elif strategy == "strategy_b_new" and task:
        scale = np.ones(weights_shape, np.float32)
        scale[:old_rows, :old_classes] = 0.0
        refs = stored_indices(memory_indices, old_classes, n_ref_per_class)
        ref_design = train_x[refs, :count]
        if step_scale is not None:
            ref_design = ref_design * np.sqrt(step_scale)
        # Old-class columns may only move through the NEW feature rows; the null space is
        # taken inside those rows. New-class columns see every row (nothing there is old),
        # as in strategy_b, so reference inputs also keep their new-class logits.
        rows = np.arange(old_rows, count)
        old_basis = np.zeros((count, 0), np.float32)
        if len(rows):
            old_basis = np.zeros(
                (count, null_space_basis(ref_design[:, rows]).shape[1]), np.float32
            )
            old_basis[rows] = null_space_basis(ref_design[:, rows])
        null_groups = [
            (slice(0, old_classes), old_basis),
            (slice(old_classes, n_seen), null_space_basis(ref_design)),
        ]
    elif strategy == "strategy_b" and task:
        refs = stored_indices(memory_indices, old_classes, n_ref_per_class)
        ref_design = train_x[refs, :count]
        if step_scale is not None:
            ref_design = ref_design * np.sqrt(step_scale)
        null_basis = null_space_basis(ref_design)
    return UpdateRestrictions(scale, null_basis, refs, null_groups)
