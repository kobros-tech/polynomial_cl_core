# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Reproducible synthetic data: target functions, labelling, and sampling (NumPy only).

A *target function* maps inputs of shape (n, 2) to a signed score; the label is ``score > 0``
(so a score of exactly 0 is class 0). Two targets are provided on ``DOMAIN`` x ``DOMAIN``:

* ``sinusoidal_target``: ``y = 1 iff x2 > AMPLITUDE * sin(FREQUENCY * x1)`` (the original task);
* ``circular_target``: ``y = 1 iff`` the point lies inside a circle (default centre (0, 0),
  radius 0.7), a different decision geometry for the same polynomial machinery.

The generator separates (1) sampling input coordinates, (2) evaluating the target,
(3) assigning labels (with optional label-flip noise) and (4) the train/test random streams.
How *experiences* are cut out of such data (vertical strips) is the business of
``polynomial_cl.binary``; the ten-class ring used by the class-incremental protocol check is at
the end of this module.
"""

from __future__ import annotations

import numpy as np

DOMAIN = (-1.0, 1.0)
AMPLITUDE = 0.5
FREQUENCY = 3.0
SPLIT_STREAM = {"train": 0, "test": 1}  # random-stream id of each split


def boundary_curve(x1):
    """x2-coordinate of the true decision boundary at x1."""
    return AMPLITUDE * np.sin(FREQUENCY * np.asarray(x1, dtype=float))


def sinusoidal_target(x) -> np.ndarray:
    """Signed margin x2 - boundary(x1); the true class is its sign."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return x[:, 1] - boundary_curve(x[:, 0])


def make_circular_target(center=(0.0, 0.0), radius=0.7):
    """Target function with class 1 inside the circle: score = radius^2 - |x - center|^2."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center must be two finite coordinates")

    def circular(x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return radius**2 - np.sum((x - center) ** 2, axis=1)

    return circular


circular_target = make_circular_target()


def true_label(x, target_function=sinusoidal_target) -> np.ndarray:
    """Ground-truth labels of ``target_function`` (default: the sinusoid); used to draw true regions."""
    return (target_function(x) > 0).astype(int)


def validate_integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_noise(noise):
    if not np.isfinite(noise) or not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be finite and between 0 and 1")


# ------------------------------------------------------------ sampling and labelling
def sample_inputs(rng, n, x1_range=DOMAIN, x2_range=DOMAIN) -> np.ndarray:
    """``n`` points uniform on the given ranges; x1 is drawn first, then x2."""
    return np.column_stack([rng.uniform(*x1_range, n), rng.uniform(*x2_range, n)])


def label_inputs(x, rng, target_function=sinusoidal_target, noise=0.0) -> np.ndarray:
    """Labels of ``x``; ``noise`` is the probability of flipping each label (inputs stay exact)."""
    y = true_label(x, target_function)
    if noise > 0:
        y = np.where(rng.random(len(y)) < noise, 1 - y, y)
    return y


def make_synthetic_dataset(
    target_function=sinusoidal_target, n_samples=200, seed=0, kind="train", noise=0.0
):
    """One reproducible dataset ``(x, y)`` over the whole domain.

    ``kind`` selects an independent random stream, so the ``"train"`` and ``"test"`` sets of one
    seed never share samples.
    """
    if kind not in SPLIT_STREAM:
        raise ValueError("kind must be 'train' or 'test'")
    n_samples = validate_integer(n_samples, "n_samples", minimum=0)
    validate_noise(noise)
    rng = np.random.default_rng([seed, SPLIT_STREAM[kind]])
    x = sample_inputs(rng, n_samples)
    return x, label_inputs(x, rng, target_function, noise)


# ---------------------------------------- ten-class ring (class-incremental check)
def make_synthetic_2d(
    seed, n_train_per_class=160, n_test_per_class=240, separation=2.5, noise=0.45
):
    """Ten ring-arranged Gaussian classes with independent train and test samples.

    Returns (train_x, train_y, test_x, test_y, class_means).
    """
    if n_train_per_class < 1 or n_test_per_class < 1:
        raise ValueError("sample counts per class must be positive")
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("separation must be finite and positive")
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("noise must be finite and positive")
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(10) / 10
    means = separation * np.column_stack((np.cos(angles), np.sin(angles)))
    train_x = np.concatenate(
        [rng.normal(means[c], noise, size=(n_train_per_class, 2)) for c in range(10)]
    ).astype(np.float32)
    test_x = np.concatenate(
        [rng.normal(means[c], noise, size=(n_test_per_class, 2)) for c in range(10)]
    ).astype(np.float32)
    return (
        train_x,
        np.repeat(np.arange(10), n_train_per_class),
        test_x,
        np.repeat(np.arange(10), n_test_per_class),
        means.astype(np.float32),
    )
