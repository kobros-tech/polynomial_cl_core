# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.continual.features import (
    expand_weights,
)
from polynomial_cl.continual.features import (
    monomial_exponents as continual_monomial_exponents,
)
from polynomial_cl.features import (
    append_zero_coefficients,
    monomial_exponents,
    validate_degree,
)
from polynomial_cl.models.binary import PolynomialLogisticModel


# ----------------------------------------- shared by binary and class-incremental code
def test_class_incremental_features_use_the_same_enumeration():
    assert continual_monomial_exponents is monomial_exponents
    for n_variables in (1, 2, 5):
        for degree in range(4):
            small = monomial_exponents(n_variables, degree)
            np.testing.assert_array_equal(
                monomial_exponents(n_variables, degree + 1)[: len(small)], small
            )


def test_monomial_enumeration_validates_arguments():
    with pytest.raises(ValueError):
        monomial_exponents(0, 2)
    with pytest.raises(ValueError):
        monomial_exponents(2, -1)
    with pytest.raises(TypeError):
        monomial_exponents(2, 1.5)


@pytest.mark.parametrize("shape", [(3,), (3, 4)])
def test_append_zero_coefficients_keeps_old_rows_and_returns_a_copy(shape):
    old = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
    new = append_zero_coefficients(old, 7)
    assert new.shape == (7, *shape[1:]) and new.dtype == old.dtype
    np.testing.assert_array_equal(new[:3], old)
    assert not new[3:].any()
    same_size = append_zero_coefficients(old, 3)
    np.testing.assert_array_equal(same_size, old)
    assert not np.shares_memory(same_size, old)
    with pytest.raises(ValueError):
        append_zero_coefficients(old, 2)


def test_binary_model_and_softmax_head_expand_through_the_same_helper():
    model = PolynomialLogisticModel(2, w=np.arange(1.0, 7.0))
    bigger = model.expanded(4)
    np.testing.assert_array_equal(bigger.w[:6], model.w)
    assert not bigger.w[6:].any() and bigger.w.dtype == model.w.dtype
    bigger.w[0] = -1.0  # the expanded model must not alias the original
    assert model.w[0] == 1.0

    head = np.ones((17, 10), dtype=np.float32)
    assert expand_weights(head, 17) is head
    np.testing.assert_array_equal(
        expand_weights(head, 153), append_zero_coefficients(head, 153)
    )


@pytest.mark.parametrize("degree", [0, 3, np.int64(2)])
def test_validate_degree_accepts_non_negative_integers(degree):
    assert validate_degree(degree) == int(degree)
    assert type(validate_degree(degree)) is int


@pytest.mark.parametrize(
    "degree, error",
    [
        (2.9, TypeError),
        (True, TypeError),
        (np.bool_(True), TypeError),
        (-1, ValueError),
    ],
)
def test_validate_degree_rejects_everything_else(degree, error):
    with pytest.raises(error, match="degree"):
        validate_degree(degree)


def test_zero_expansion_preserves_binary_and_multiclass_outputs():
    """Outputs, not shapes: the same helper keeps both model types' predictions unchanged."""
    from polynomial_cl.continual.features import scaled_polynomial_features
    from polynomial_cl.models.binary import polynomial_features

    rng = np.random.default_rng(0)
    x2 = rng.uniform(-1, 1, (40, 2))
    binary = PolynomialLogisticModel(2, w=rng.normal(size=6))
    wider = binary.expanded(4)
    # a wider matrix product sums in a different order, so equality holds to rounding only
    np.testing.assert_allclose(wider.score(x2), binary.score(x2), rtol=0, atol=1e-12)
    assert not wider.w[6:].any()

    x16 = rng.normal(size=(40, 16)).astype(np.float32)
    small = scaled_polynomial_features(x16, 1)
    large = scaled_polynomial_features(x16, 2)
    head = rng.normal(size=(small.shape[1], 10)).astype(np.float32)
    grown = expand_weights(head, large.shape[1])
    np.testing.assert_allclose(large @ grown, small @ head, rtol=0, atol=1e-5)
    assert not grown[small.shape[1] :].any()
    np.testing.assert_array_equal(large[:, : small.shape[1]], small)  # prefix property
    assert polynomial_features(x2, 2).shape[1] == 6
