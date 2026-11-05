# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.features import monomial_exponents
from polynomial_cl.models.binary import (
    PolynomialFeatures,
    feature_exponents,
    feature_names,
    n_features,
    polynomial_features,
)


def test_names_degree_1_2_3():
    assert feature_names(1) == ["1", "x1", "x2"]
    assert feature_names(2) == ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2"]
    assert feature_names(3)[6:] == ["x1^3", "x1^2*x2", "x1*x2^2", "x2^3"]


def test_ordering_is_deterministic_and_prefix_stable():
    assert feature_names(6) == feature_names(6)
    for d in range(9):
        assert feature_names(d + 1)[: n_features(d)] == feature_names(d)


@pytest.mark.parametrize("d", range(13))
def test_feature_count(d):
    assert n_features(d) == (d + 1) * (d + 2) // 2
    assert len(feature_names(d)) == n_features(d)
    assert polynomial_features(np.zeros((3, 2)), d).shape == (3, n_features(d))


def test_known_values():
    x = np.array([[2.0, 3.0]])
    np.testing.assert_array_equal(polynomial_features(x, 2)[0], [1, 2, 3, 4, 6, 9])
    np.testing.assert_array_equal(
        polynomial_features(x, 3)[0], [1, 2, 3, 4, 6, 9, 8, 12, 18, 27]
    )
    np.testing.assert_array_equal(polynomial_features([-1.0, 0.5], 1)[0], [1, -1, 0.5])


def test_wrapper_exposes_metadata():
    f = PolynomialFeatures(4)
    assert f.degree == 4 and f.n_features == 15 and f.names == feature_names(4)


@pytest.mark.parametrize(
    "degree, error",
    [
        (2.9, TypeError),
        (True, TypeError),
        (np.float64(3), TypeError),
        (np.bool_(True), TypeError),
        (-1, ValueError),
    ],
)
def test_invalid_degree_is_rejected_consistently(degree, error):
    with pytest.raises(error, match="degree"):
        PolynomialFeatures(degree)
    with pytest.raises(error, match="degree"):
        feature_names(degree)
    with pytest.raises(error, match="degree"):
        n_features(degree)
    with pytest.raises(error, match="degree"):
        polynomial_features([[0.0, 0.0]], degree)


def test_numpy_integer_degree_is_accepted():
    assert PolynomialFeatures(np.int64(2)).degree == 2


def test_binary_exponents_are_the_two_variable_case_of_the_shared_enumeration():
    assert feature_exponents(3) == [
        (0, 0),
        (1, 0), (0, 1),
        (2, 0), (1, 1), (0, 2),
        (3, 0), (2, 1), (1, 2), (0, 3),
    ]  # fmt: skip
    for degree in range(11):
        shared = monomial_exponents(2, degree)
        assert feature_exponents(degree) == [tuple(map(int, row)) for row in shared]
        assert len(shared) == n_features(degree)
