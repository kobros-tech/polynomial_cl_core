# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Mini-batch sampling policies of the class-incremental experiments (NumPy only).

Two policies decide which rows of the current experience plus replay form each update:

* ``shuffled`` (the default): all rows are shuffled every epoch and read once each. This is
  the trainer's own epoch loop in ``polynomial_cl.training.multiclass.fit_task``, so its
  sampler is ``None``.
* ``class_balanced``: every batch gives each class (near-)equal slots, repeating the examples
  of small classes (``balanced_batches``). The number of updates per epoch is the same as for
  ``shuffled``; the number of rows drawn can be slightly larger.

The trainer never imports this module: ``epoch_sampler`` turns a policy name into the callable
that ``fit_task`` accepts, so the dependency runs from ``continual`` to ``training`` only.
"""

from __future__ import annotations

import numpy as np

SAMPLING_POLICIES = ("shuffled", "class_balanced")


def balanced_batches(labels, n_updates, batch_size, rng):
    """Yield ``n_updates`` index batches with (near-)equal representation of every class.

    Every class that has examples gets ``batch_size // n_classes`` slots; the remaining slots
    go to randomly chosen classes. A class is read through its own shuffled cycle: its
    examples are used once each, in random order, before any is repeated (the cycle is then
    reshuffled), so a class with fewer examples than its slots is repeated, never padded
    from another class. All batches are full, so the number of updates matches the shuffled
    policy while the total of drawn rows can be slightly larger.
    """
    classes = np.unique(labels)
    pools = {int(c): np.flatnonzero(labels == c) for c in classes}
    cycle, cursor = {}, {}
    for c, pool in pools.items():
        cycle[c], cursor[c] = pool[rng.permutation(len(pool))], 0
    base, extra = divmod(batch_size, len(classes))
    for _ in range(n_updates):
        bonus = np.zeros(len(classes), dtype=np.int64)
        bonus[rng.choice(len(classes), extra, replace=False)] = 1
        parts = []
        for c, more in zip(classes, bonus):
            c, need = int(c), base + int(more)
            while need:
                if cursor[c] == len(cycle[c]):
                    cycle[c] = pools[c][rng.permutation(len(pools[c]))]
                    cursor[c] = 0
                take = min(need, len(cycle[c]) - cursor[c])
                parts.append(cycle[c][cursor[c] : cursor[c] + take])
                cursor[c] += take
                need -= take
        yield np.concatenate(parts)


def epoch_sampler(policy):
    """The ``epoch_sampler`` argument of ``fit_task`` for a sampling policy name."""
    if policy not in SAMPLING_POLICIES:
        raise ValueError(f"unknown sampling policy: {policy}")
    return None if policy == "shuffled" else balanced_batches
