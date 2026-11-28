# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Test support: restrictions under the pre-C4 strategy names, built by the right code.

The C2/C3 fixtures and several integration tests are written in the pre-C4 strategy
vocabulary. ``ordinary`` and ``strategy_b`` are produced by the production code (as the
methods ``ordinary`` and ``relaxed``, beta = 0); ``frozen_old``, ``small_old_lr`` and
``strategy_b_new`` no longer exist as methods and come from the frozen helper
``legacy_restrictions.py``, so ``fit_task``'s ``update_scale`` and ``null_groups`` paths keep
their coverage. Call it with the *legacy* keyword arguments.
"""

from legacy_restrictions import update_restrictions as legacy_restrictions

from polynomial_cl.continual.strategies import (
    update_restrictions as current_restrictions,
)

CURRENT_METHOD = {"ordinary": "ordinary", "strategy_b": "relaxed"}
CURRENT_KEYS = (
    "old_classes",
    "count",
    "task",
    "train_x",
    "memory_indices",
    "n_ref_per_class",
    "step_scale",
    "base_rate",
)


def build_restrictions(strategy, **kwargs):
    if strategy in CURRENT_METHOD:
        kept = {k: v for k, v in kwargs.items() if k in CURRENT_KEYS}
        return current_restrictions(CURRENT_METHOD[strategy], **kept)
    return legacy_restrictions(strategy, **kwargs)
