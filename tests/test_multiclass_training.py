# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np

from polynomial_cl.training.multiclass import fit_task, softmax


def test_softmax_rows_are_distributions_and_stable_for_large_logits():
    p = softmax(np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]]))
    assert p.shape == (2, 3) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(p[1], 1 / 3)


def test_fit_task_updates_the_coefficient_array_in_place_and_skips_future_classes():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 5)).astype(np.float32)
    labels = np.repeat([0, 1], 20)
    weights = np.zeros(
        (5, 10), np.float32
    )  # ten output columns, only 0 and 1 introduced
    out = fit_task(
        weights, features, labels, np.empty((0, 5), np.float32), np.empty(0, np.int64),
        epochs=2, learning_rate=0.1, batch_size=16, seed=0,
        eligible_classes=np.arange(2),
    )  # fmt: skip
    assert out is weights and weights.shape == (5, 10)
    assert np.abs(weights[:, :2]).sum() > 0  # eligible classes learned
    assert not weights[:, 2:].any()  # future classes never touched
