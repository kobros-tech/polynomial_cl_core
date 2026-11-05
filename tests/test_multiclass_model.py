# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Multiclass model semantics: logits, class-eligible prediction, zero-initialised expansion."""

import numpy as np
import pytest

from polynomial_cl.continual.features import scaled_polynomial_features
from polynomial_cl.models.multiclass import expand_weights, logits, predict, softmax


def _head_and_inputs(seed=0, degree=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, 16)).astype(np.float32)
    features = scaled_polynomial_features(x, degree)
    head = rng.normal(size=(features.shape[1], 10)).astype(np.float32)
    return x, features, head


def test_logits_have_one_column_per_class_and_probabilities_are_distributions():
    _, features, head = _head_and_inputs()
    full = logits(head, features)
    assert full.shape == (len(features), 10)
    assert logits(head, features, 4).shape == (len(features), 4)
    probabilities = softmax(full)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(predict(full), probabilities.argmax(axis=1))


def test_prediction_only_considers_the_classes_that_are_in_play():
    _, features, head = _head_and_inputs()
    head[:, 6:] += 100.0 * np.sign(
        head[0, 6:]
    )  # make later classes dominate the logits
    head[0, 6:] = 50.0
    assert (predict(logits(head, features)) >= 6).all()  # all ten compete: they win
    for n_classes in (2, 4, 6):
        assert (predict(logits(head, features), n_classes) < n_classes).all()
        assert (predict(logits(head, features, n_classes)) < n_classes).all()


def test_selecting_columns_before_or_after_the_product_gives_the_same_prediction():
    _, features, head = _head_and_inputs(seed=3)
    for n_classes in (2, 6, 10):
        np.testing.assert_array_equal(
            predict(logits(head, features), n_classes),
            predict(logits(head, features, n_classes)),
        )


@pytest.mark.parametrize("n_classes", [None, 4])
def test_zero_initialised_expansion_leaves_logits_probabilities_and_predictions_unchanged(
    n_classes,
):
    x, small, head = _head_and_inputs(seed=1, degree=1)
    large = scaled_polynomial_features(x, 2)
    grown = expand_weights(head, large.shape[1])
    assert grown.shape == (large.shape[1], head.shape[1])
    assert not grown[head.shape[0] :].any()  # every new coefficient is zero
    before, after = logits(head, small, n_classes), logits(grown, large, n_classes)
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-5)
    np.testing.assert_allclose(softmax(after), softmax(before), rtol=0, atol=1e-5)
    np.testing.assert_array_equal(predict(after), predict(before))


def test_expansion_cannot_shrink_and_a_noop_returns_the_same_array():
    _, _, head = _head_and_inputs()
    assert expand_weights(head, head.shape[0]) is head
    with pytest.raises(ValueError, match="shrink"):
        expand_weights(head, head.shape[0] - 1)
