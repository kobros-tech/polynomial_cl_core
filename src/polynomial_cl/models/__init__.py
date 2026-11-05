# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Prediction models on the shared polynomial foundation (``polynomial_cl.features``).

* ``binary``: the binary polynomial logistic model (one score, sigmoid, 0/1 labels).
* ``multiclass``: the softmax head used by class-incremental learning, a plain
  ``(n_features, n_classes)`` coefficient array with its logits, class-eligible prediction
  rule and zero-initialised expansion as functions.

The two are independent: neither imports the other, and class-incremental code uses only
``multiclass``. Their optimisers are in ``polynomial_cl.training``.
"""
