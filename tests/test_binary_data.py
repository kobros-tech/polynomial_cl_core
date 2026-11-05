# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.binary.data import combine_splits, generate_split, make_experiences
from polynomial_cl.data.synthetic import sinusoidal_target, true_label


def test_reproducible_and_seed_dependent():
    a, b, c = (make_experiences(seed=s, n_train=50, n_test=50) for s in (1, 1, 2))
    assert np.array_equal(a[3].train.x, b[3].train.x) and np.array_equal(
        a[3].test.y, b[3].test.y
    )
    assert not np.array_equal(a[3].train.x, c[3].train.x)


def test_evaluation_set_is_fixed_and_disjoint_from_training():
    small = make_experiences(seed=0, n_train=50, n_test=100)
    large = make_experiences(seed=0, n_train=400, n_test=100)
    for s, big in zip(small, large):
        assert np.array_equal(s.test.x, big.test.x)  # unaffected by training size
        tr = {tuple(r) for r in big.train.x}
        assert not any(tuple(r) in tr for r in big.test.x)


def test_noise_free_labels_follow_ground_truth_and_noise_flips():
    clean = generate_split(3, 4000, 0, "train", noise=0.0)
    assert np.array_equal(clean.y, true_label(clean.x))
    noisy = generate_split(3, 4000, 0, "train", noise=0.1)
    assert 0.07 < np.mean(noisy.y != true_label(noisy.x)) < 0.13
    assert np.array_equal(noisy.x, clean.x)


def test_experiences_have_different_distributions_and_balanced_classes():
    exps = make_experiences(seed=0)
    mids = [e.train.x[:, 0].mean() for e in exps]
    assert np.all(np.diff(mids) > 0)
    for e in exps:
        assert 0.2 < e.train.y.mean() < 0.8
        a, b = e.window
        assert e.train.x[:, 0].min() >= a and e.train.x[:, 0].max() <= b
    _, y, src = combine_splits([e.train for e in exps])
    assert len(y) == 2000 and set(src) == set(range(1, 11))


def test_boundary_is_nonlinear():
    from polynomial_cl.data.synthetic import boundary_curve

    xs = np.linspace(-1, 1, 9)
    second_diff = np.diff(boundary_curve(xs), 2)
    assert np.max(np.abs(second_diff)) > 0.05  # a straight line would give ~0
    # the sign of the score really is the label
    pts = np.array([[0.0, 0.1], [0.0, -0.1]])
    assert (
        list(true_label(pts)) == [1, 0]
        and sinusoidal_target(pts)[0] > 0 > sinusoidal_target(pts)[1]
    )


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"kind": "validation"}, ValueError, "kind must be 'train' or 'test'"),
        ({"n": -1}, ValueError, "n must be an integer"),
        ({"n": 1.5}, TypeError, "n must be an integer"),
        ({"exp_id": 0}, ValueError, "exp_id must be an integer"),
        ({"exp_id": 11}, ValueError, "exp_id must be <= n_experiences"),
        ({"n_experiences": 0}, ValueError, "n_experiences must be an integer"),
        ({"noise": -0.1}, ValueError, "noise must be finite"),
        ({"noise": 1.1}, ValueError, "noise must be finite"),
    ],
)
def test_generate_split_rejects_invalid_arguments(kwargs, error, message):
    args = {
        "exp_id": 1,
        "n": 10,
        "seed": 0,
        "kind": "train",
        "noise": 0.0,
        "n_experiences": 10,
    }
    args.update(kwargs)
    with pytest.raises(error, match=message):
        generate_split(**args)


@pytest.mark.parametrize(
    "exp_id, n_experiences, error",
    [
        (0, 10, ValueError),
        (11, 10, ValueError),
        (1, 0, ValueError),
        (1, 1.5, TypeError),
    ],
)
def test_experience_window_rejects_invalid_arguments(exp_id, n_experiences, error):
    from polynomial_cl.binary.data import experience_window

    with pytest.raises(error):
        experience_window(exp_id, n_experiences)
