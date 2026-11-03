# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Experience construction for the binary sequential-learning experiment.

Experience k (1-based) of K draws x1 only from the k-th vertical strip of [-1, 1] (x2 is
uniform on [-1, 1]), so every experience sees a different part of the decision boundary and
remembering old strips and fitting a new one are different things. Labels come from a target
function of ``polynomial_cl.data.synthetic`` (default: the sinusoid). Train and test samples of
an experience come from independent random streams; the test split is fixed by
(seed, experience) and never used for training or replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.synthetic import (
    DOMAIN,
    SPLIT_STREAM,
    label_inputs,
    sample_inputs,
    sinusoidal_target,
    validate_integer,
    validate_noise,
)


# ---------------------------------------------------------------------- data
@dataclass(frozen=True)
class Split:
    """Samples of one experience. `kind` is 'train' or 'test' and is checked by the replay buffer."""

    exp_id: int
    kind: str
    x: np.ndarray
    y: np.ndarray

    def __len__(self):
        return len(self.y)


@dataclass(frozen=True)
class Experience:
    exp_id: int  # 1-based
    window: tuple  # x1 range of this experience
    train: Split
    test: Split


def experience_window(exp_id: int, n_experiences: int = 10):
    n_experiences = validate_integer(n_experiences, "n_experiences", minimum=1)
    exp_id = validate_integer(exp_id, "exp_id", minimum=1)
    if exp_id > n_experiences:
        raise ValueError("exp_id must be <= n_experiences")
    lo, hi = DOMAIN
    width = (hi - lo) / n_experiences
    return (lo + (exp_id - 1) * width, lo + exp_id * width)


def generate_split(
    exp_id,
    n,
    seed,
    kind="train",
    noise=0.0,
    n_experiences=10,
    target_function=sinusoidal_target,
) -> Split:
    """`noise` = probability of flipping each label (labels only; inputs are exact)."""
    if kind not in SPLIT_STREAM:
        raise ValueError("kind must be 'train' or 'test'")
    n = validate_integer(n, "n", minimum=0)
    n_experiences = validate_integer(n_experiences, "n_experiences", minimum=1)
    exp_id = validate_integer(exp_id, "exp_id", minimum=1)
    if exp_id > n_experiences:
        raise ValueError("exp_id must be <= n_experiences")
    validate_noise(noise)
    rng = np.random.default_rng([seed, exp_id, SPLIT_STREAM[kind]])
    x = sample_inputs(rng, n, x1_range=experience_window(exp_id, n_experiences))
    y = label_inputs(x, rng, target_function, noise)
    x.setflags(write=False)
    y.setflags(write=False)
    return Split(exp_id, kind, x, y)


def make_experiences(
    n_experiences=10, n_train=200, n_test=500, seed=0, noise=0.0, test_noise=0.0,
    target_function=sinusoidal_target,
):  # fmt: skip
    """Build all experiences. The test split depends only on (seed, exp, n_test, test_noise)."""
    return [
        Experience(
            k,
            experience_window(k, n_experiences),
            generate_split(
                k, n_train, seed, "train", noise, n_experiences, target_function
            ),
            generate_split(
                k, n_test, seed, "test", test_noise, n_experiences, target_function
            ),
        )
        for k in range(1, n_experiences + 1)
    ]


def combine_splits(splits):
    """Concatenate splits -> (x, y, source_experience_id)."""
    x = np.concatenate([s.x for s in splits])
    y = np.concatenate([s.y for s in splits])
    src = np.concatenate([np.full(len(s), s.exp_id) for s in splits])
    return x, y, src
