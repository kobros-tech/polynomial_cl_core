# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import hashlib

import numpy as np
import pytest

from polynomial_cl.binary.data import make_experiences
from polynomial_cl.binary.experiment import make_config, run_continual
from polynomial_cl.data.synthetic import (
    boundary_curve,
    circular_target,
    make_circular_target,
    make_synthetic_dataset,
    sinusoidal_target,
    true_label,
)
from polynomial_cl.models.binary import PolynomialLogisticModel
from polynomial_cl.training.binary import train


def _digest(**kwargs):
    h = hashlib.sha256()
    for e in make_experiences(**kwargs):
        for s in (e.train, e.test):
            h.update(np.ascontiguousarray(s.x).tobytes())
            h.update(np.ascontiguousarray(s.y).astype(np.int64).tobytes())
    return h.hexdigest()[:16]


# Digests of the pre-refactor implementation (commit 9e15732); the sinusoidal protocol,
# including its random-stream layout and label noise, must not change.
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "994748f7fe36c4f5"),
        ({"seed": 3, "n_train": 50, "n_test": 60}, "f81e2412bcf3385e"),
        ({"seed": 1, "n_train": 80, "n_test": 40, "noise": 0.2, "test_noise": 0.1}, "c094eae16a8430ff"),
        ({"n_experiences": 5, "seed": 7}, "d3b29edf26e5313a"),
    ],
)  # fmt: skip
def test_sinusoidal_experiences_are_bit_identical_to_the_pre_refactor_generator(
    kwargs, expected
):
    assert _digest(**kwargs) == expected


def test_sinusoid_rule_and_default_target():
    x = np.array([[0.0, 0.1], [0.0, -0.1], [0.5, 0.4]])
    expected = (x[:, 1] > 0.5 * np.sin(3 * x[:, 0])).astype(int)
    np.testing.assert_array_equal(true_label(x), expected)
    np.testing.assert_allclose(sinusoidal_target(x), x[:, 1] - boundary_curve(x[:, 0]))
    default = make_synthetic_dataset(n_samples=30, seed=1)
    explicit = make_synthetic_dataset(sinusoidal_target, 30, 1)
    np.testing.assert_array_equal(default[1], explicit[1])


def test_circular_target_geometry_and_label_convention():
    x = np.array([[0.0, 0.0], [0.69, 0.0], [0.71, 0.0], [0.0, -0.71], [0.9, 0.9]])
    np.testing.assert_array_equal(true_label(x, circular_target), [1, 1, 0, 0, 0])
    shifted = make_circular_target(center=(0.5, -0.5), radius=0.2)
    np.testing.assert_array_equal(
        true_label([[0.5, -0.5], [0.0, 0.0]], shifted), [1, 0]
    )
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            make_circular_target(radius=bad)
    with pytest.raises(ValueError):
        make_circular_target(center=(0.0, 0.0, 0.0))


def test_generator_is_reproducible_and_train_test_streams_are_independent():
    a = make_synthetic_dataset(circular_target, 100, seed=4)
    b = make_synthetic_dataset(circular_target, 100, seed=4)
    other = make_synthetic_dataset(circular_target, 100, seed=5)
    test = make_synthetic_dataset(circular_target, 100, seed=4, kind="test")
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[0], other[0]) and not np.array_equal(a[0], test[0])
    assert np.all(np.abs(a[0]) <= 1)
    with pytest.raises(ValueError):
        make_synthetic_dataset(kind="validation")
    with pytest.raises(ValueError):
        make_synthetic_dataset(noise=1.5)


def test_label_noise_flips_labels_but_not_inputs():
    clean = make_synthetic_dataset(circular_target, 400, seed=0)
    noisy = make_synthetic_dataset(circular_target, 400, seed=0, noise=0.3)
    np.testing.assert_array_equal(clean[0], noisy[0])
    assert 0.2 < np.mean(clean[1] != noisy[1]) < 0.4


def test_experiences_can_use_the_circular_target_without_changing_the_default():
    default = make_experiences(n_train=40, n_test=40, seed=2)
    circle = make_experiences(
        n_train=40, n_test=40, seed=2, target_function=circular_target
    )
    for d, c in zip(default, circle):
        np.testing.assert_array_equal(d.train.x, c.train.x)  # same sampling protocol
        np.testing.assert_array_equal(c.train.y, true_label(c.train.x, circular_target))
    assert any(
        not np.array_equal(d.train.y, c.train.y) for d, c in zip(default, circle)
    )


def test_degree_two_polynomial_can_learn_the_circle_but_degree_one_cannot():
    x, y = make_synthetic_dataset(circular_target, 1500, seed=0)
    xt, yt = make_synthetic_dataset(circular_target, 1500, seed=0, kind="test")
    accuracy = {}
    for degree in (1, 2):
        model = train(PolynomialLogisticModel(degree), x, y, lr=1.0, epochs=2000).model
        accuracy[degree] = np.mean(model.predict(xt) == yt)
    assert accuracy[2] > 0.97 and accuracy[1] < 0.8


def test_sequential_binary_run_works_on_the_circle():
    exps = make_experiences(
        n_experiences=3, n_train=60, n_test=60, target_function=circular_target
    )
    result = run_continual(
        exps, make_config("ordinary", epochs=30, capacity=40, m_current=40, m_replay=40)
    )
    assert len(result.records) == 3 and result.records[-1].acc_current > 0.5
