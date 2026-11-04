# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Multiclass (softmax-head) prediction semantics on polynomial features (NumPy only).

The head is a coefficient array ``weights`` of shape ``(n_features, n_classes)``; column ``c``
holds the coefficients of class ``c``. Its outputs are *logits* ``features @ weights``; the
prediction is the argmax over the classes that are in play, and the probabilities are the
softmax of those logits. There is no bias term: the constant monomial is the first feature.

Two properties of the class-incremental protocol live here because they are properties of the
model's outputs, not of how it is trained:

* **Class eligibility.** Only the first ``n_classes`` columns compete. Columns of classes that
  have not been introduced yet are excluded from the prediction, so a future class can never
  be predicted.
* **Zero-initialised expansion.** ``expand_weights`` appends zero rows for the new monomials.
  Every existing logit is unchanged, hence so is every prediction and probability.

This module is independent of the binary model (``polynomial_cl.models.binary``): outputs,
loss and prediction rule differ, only the polynomial foundation in ``polynomial_cl.features``
is shared. The loss and optimiser are in ``polynomial_cl.training.multiclass``.
"""

from __future__ import annotations

import numpy as np

from ..features import append_zero_coefficients


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable row-wise softmax probabilities."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def logits(
    weights: np.ndarray, features: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """Class logits ``features @ weights``; with ``n_classes`` only those first columns.

    ``n_classes`` selects the columns *before* the product (the eligible classes only). To
    keep the full product and cut afterwards, call ``logits(weights, features)`` and pass
    ``n_classes`` to ``predict``; the two orders can differ in the last floating-point bits.
    """
    return features @ (weights if n_classes is None else weights[:, :n_classes])


def predict(class_logits: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Predicted class: argmax over the first ``n_classes`` logit columns (default: all)."""
    columns = class_logits if n_classes is None else class_logits[:, :n_classes]
    return columns.argmax(axis=1)


def expand_weights(weights: np.ndarray, new_feature_count: int) -> np.ndarray:
    """Append zero coefficient rows; every existing logit is preserved exactly.

    Returns ``weights`` itself when no rows are added, otherwise a new array.
    """
    if new_feature_count == weights.shape[0]:
        return weights
    return append_zero_coefficients(weights, new_feature_count)
