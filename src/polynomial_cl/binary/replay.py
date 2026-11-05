# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Replay buffer storing *training* samples only, and balanced batch construction.

Sampling policy
---------------
* Each experience keeps at most `capacity_per_experience` samples, chosen once with a
  seeded RNG without replacement from its TRAIN split.
* `sample(m)` draws m rows. Without replacement, if fewer than m are stored, ALL stored
  rows are returned (never padded, never taken from elsewhere) and the actual count is
  reported. With replacement, exactly m rows are drawn.
* `add_experience` rejects anything that is not a 'train' Split, so evaluation data cannot
  enter the buffer through the API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.synthetic import validate_integer
from .data import Split


class ReplayBuffer:
    def __init__(self, capacity_per_experience: int = 100, seed: int = 0):
        self.capacity = validate_integer(
            capacity_per_experience, "capacity_per_experience", minimum=1
        )
        self.seed = seed
        self._x, self._y = {}, {}

    def add_experience(self, split: Split):
        if not isinstance(split, Split) or split.kind != "train":
            raise ValueError("replay buffer accepts only 'train' splits")
        rng = np.random.default_rng([self.seed, split.exp_id, 99])
        k = min(self.capacity, len(split))
        idx = np.sort(rng.choice(len(split), size=k, replace=False))
        self._x[split.exp_id] = split.x[idx].copy()
        self._y[split.exp_id] = split.y[idx].copy()

    @property
    def experience_ids(self):
        return sorted(self._x)

    def size(self, exp_id):
        return len(self._y[exp_id])

    def stored(self, exp_id):
        return self._x[exp_id], self._y[exp_id]

    def sample(self, exp_id, m, rng, replace=False):
        x, y = self._x[exp_id], self._y[exp_id]
        return _draw(x, y, m, rng, replace)

    def reference_inputs(self, n_per_experience: int):
        """First n stored inputs of every buffered experience (deterministic, inspectable)."""
        n_per_experience = validate_integer(
            n_per_experience, "n_per_experience", minimum=0
        )
        parts = [self._x[k][:n_per_experience] for k in self.experience_ids]
        return np.concatenate(parts) if parts else np.zeros((0, 2))


def _draw(x, y, m, rng, replace):
    n = len(y)
    if m <= 0 or n == 0:
        return x[:0], y[:0]
    if replace:
        idx = rng.integers(0, n, size=m)
    else:
        idx = rng.choice(n, size=min(m, n), replace=False)
    return x[idx], y[idx]


@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    source: np.ndarray  # experience id of every row
    counts: dict  # actual number of rows drawn from each experience


def build_batch(
    current: Split,
    buffer: ReplayBuffer,
    m_current,
    m_replay,
    replace=False,
    seed=0,
    step=0,
) -> Batch:
    """Batch = m_current rows of the current train split + m_replay rows from EACH buffered
    experience (m_replay may be an int or a {exp_id: m} dict). Deterministic in (seed, step)."""
    if current.kind != "train":
        raise ValueError("current split must be a train split")

    def validate_count(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    validate_count(m_current, "m_current")
    if isinstance(m_replay, dict):
        for exp_id, count in m_replay.items():
            validate_count(count, f"m_replay[{exp_id}]")
    else:
        validate_count(m_replay, "m_replay")

    xs, ys, src, counts = [], [], [], {}
    rng = np.random.default_rng([seed, step, current.exp_id])
    x, y = _draw(current.x, current.y, m_current, rng, replace)
    xs.append(x)
    ys.append(y)
    src.append(np.full(len(y), current.exp_id))
    counts[current.exp_id] = len(y)
    for k in buffer.experience_ids:
        if k == current.exp_id:
            continue
        m = m_replay.get(k, 0) if isinstance(m_replay, dict) else m_replay
        x, y = buffer.sample(k, m, rng, replace)
        xs.append(x)
        ys.append(y)
        src.append(np.full(len(y), k))
        counts[k] = len(y)
    return Batch(np.concatenate(xs), np.concatenate(ys), np.concatenate(src), counts)
