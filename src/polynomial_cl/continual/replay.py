# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Replay memory of the class-incremental experiments (NumPy only).

The memory is a fixed per-class selection of training-set indices, drawn once per seed (see
``make_task_plan`` in ``experiment.py``, which draws it from the same random stream as the
experience shuffles, class by class). A stored example is never modified; every use of the
memory reads the same indices:

* replay rows added to every experience after the first (``stored_indices`` with all of them),
* the first few stored examples per old class that Strategy B protects (``stored_indices``
  with ``per_class``).
"""

from __future__ import annotations

import numpy as np


def select_memory(labels, class_id, memory_per_class, rng):
    """Draw ``min(memory_per_class, n_examples)`` training indices of one class, no repeats."""
    candidates = np.flatnonzero(labels == class_id)
    if len(candidates) == 0:
        raise ValueError(f"training data is missing class {class_id}")
    return rng.choice(candidates, min(memory_per_class, len(candidates)), replace=False)


def stored_indices(memory, n_old_classes, per_class=None):
    """Stored indices of classes ``0 .. n_old_classes - 1``, in class order.

    ``per_class`` keeps only the first that many stored examples of each class (the order in
    which they were drawn); ``None`` keeps all of them.
    """
    return np.concatenate([memory[c][:per_class] for c in range(n_old_classes)])
