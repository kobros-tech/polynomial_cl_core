# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""The shared diagonal learning-rate definition ``d_j = eta_deg(j) * c_deg(j)``."""

import numpy as np
import pytest

from polynomial_cl.continual.degree_rates import (
    N_DEGREE_GROUPS,
    apply_degree_control,
    degree_of_feature,
    diagonal_learning_rates,
    feature_rates,
    step_scale_for,
)
from polynomial_cl.continual.features import (
    n_polynomial_features,
    scaled_polynomial_features,
)
from polynomial_cl.models.multiclass import expand_weights

RATES = [3.0, 2.0, 0.75, 1.5]


# ------------------------------------------------------- degree -> feature mapping
@pytest.mark.parametrize("n_components", [1, 2, 5])
def test_degree_of_feature_counts_and_order(n_components):
    degrees = degree_of_feature(n_components, 3)
    assert len(degrees) == n_polynomial_features(n_components, 3)
    assert degrees[0] == 0 and (np.diff(degrees) >= 0).all()  # grouped, ascending
    sizes = np.bincount(degrees)
    assert sizes[0] == 1 and sizes[1] == n_components


def test_degree_mapping_is_prefix_stable_across_active_degrees():
    full = degree_of_feature(4, 3)
    for degree in (1, 2):
        part = degree_of_feature(4, degree)
        assert (full[: len(part)] == part).all()


# ----------------------------------------------------------------- construction
def test_diagonal_is_the_degree_rate_times_the_degree_control_factor():
    degrees = degree_of_feature(3, 3)
    control = [1.0, 0.5, 0.0, 0.25]
    d = diagonal_learning_rates(RATES, degrees, control)
    assert d.shape == degrees.shape and d.dtype == np.float64  # a vector, not a matrix
    expected = np.array(RATES)[degrees] * np.array(control)[degrees]
    np.testing.assert_array_equal(d, expected)


@pytest.mark.parametrize("control", [None, [1, 1, 1, 1], [1.0] * N_DEGREE_GROUPS])
def test_identity_control_reproduces_the_plain_degree_rates_exactly(control):
    degrees = degree_of_feature(3, 3)
    np.testing.assert_array_equal(
        diagonal_learning_rates(RATES, degrees, control), np.array(RATES)[degrees]
    )


def test_zero_rate_or_zero_control_freezes_the_whole_degree_group():
    degrees = degree_of_feature(3, 3)
    d = diagonal_learning_rates([3.0, 0.0, 2.0, 2.0], degrees, [1, 1, 0, 1])
    assert (d[degrees == 1] == 0).all() and (d[degrees == 2] == 0).all()
    assert (d[degrees == 0] == 3).all() and (d[degrees == 3] == 2).all()
    # an all-zero effective vector is a legitimate no-op, not an error
    assert not feature_rates([0, 0, 0, 0], degrees).any()


def test_square_root_of_the_diagonal_squares_back_without_a_dense_matrix():
    d = diagonal_learning_rates(RATES, degree_of_feature(3, 3), [1, 0.5, 1, 0.1])
    np.testing.assert_allclose(np.sqrt(d) ** 2, d, rtol=1e-14)


def test_step_scale_for_is_consistent_with_the_diagonal():
    """One definition: base * multiplier is exactly the diagonal of feature_rates."""
    degrees = degree_of_feature(3, 3)
    effective = apply_degree_control(np.array(RATES), [1, 0.5, 1, 0.1])
    base, multiplier = step_scale_for(effective, degrees)
    np.testing.assert_allclose(
        base * multiplier.astype(np.float64),
        feature_rates(effective, degrees),
        rtol=1e-6,
    )
    assert multiplier.max() == pytest.approx(1.0) and multiplier.min() >= 0


# ------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "bad, error",
    [
        ([1, 1, 1, -0.1], ValueError),
        ([1, 1, 1, np.nan], ValueError),
        ([1, 1, 1, np.inf], ValueError),
        ([[1, 1, 1, 1]], ValueError),
        ([True, False, True, True], TypeError),
        (["a", "b", "c", "d"], TypeError),
        ([1, 1], ValueError),  # degree 3 cannot index a two-entry vector
    ],
)
def test_invalid_rate_vectors_are_rejected(bad, error):
    with pytest.raises(error):
        feature_rates(bad, degree_of_feature(3, 3))


def test_group_index_must_be_a_one_dimensional_integer_array_of_valid_degrees():
    with pytest.raises(ValueError, match="integer"):
        feature_rates(RATES, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="integer"):
        feature_rates(RATES, np.array([[0, 1]]))
    with pytest.raises(ValueError, match="lie in"):
        feature_rates(RATES, np.array([0, 1, 4]))
    with pytest.raises(ValueError, match="lie in"):
        feature_rates(RATES, np.array([-1, 0]))


def test_control_factors_outside_the_unit_interval_are_rejected():
    degrees = degree_of_feature(2, 3)
    for bad in ([1, 1, 1, 1.5], [1, 1, 1, -0.5], [1, 1, 1], [1, 1, 1, np.nan]):
        with pytest.raises(ValueError):
            diagonal_learning_rates(RATES, degrees, bad)


# -------------------------------------------------------------- expansion behaviour
def test_expansion_keeps_old_rates_and_gives_new_features_their_own_degree_rate():
    d_old = diagonal_learning_rates(RATES, degree_of_feature(4, 1), [1, 0.5, 1, 1])
    d_new = diagonal_learning_rates(RATES, degree_of_feature(4, 2), [1, 0.5, 1, 1])
    n_old = len(d_old)
    np.testing.assert_array_equal(d_new[:n_old], d_old)
    # the new features are the degree-2 monomials: rate 0.75, control 1
    assert len(d_new) == n_polynomial_features(4, 2)
    assert (d_new[n_old:] == 0.75).all()


def test_expansion_preserves_the_function_for_trained_nonzero_coefficients():
    """phi_new(x)^T T W = phi_old(x)^T W with T = [I; 0], on the real scaled features."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6)).astype(np.float32)
    old = scaled_polynomial_features(x, 1)
    for new_degree in (2, 3):
        new = scaled_polynomial_features(x, new_degree)
        # the old features are literally the leading columns (no re-scaling on expansion)
        np.testing.assert_array_equal(new[:, : old.shape[1]], old)
        weights = rng.normal(size=(old.shape[1], 5)).astype(np.float32)
        expanded = expand_weights(weights, new.shape[1])
        np.testing.assert_array_equal(expanded[: len(weights)], weights)  # copied
        assert not expanded[len(weights) :].any()  # new coefficients start at zero
        np.testing.assert_allclose(new @ expanded, old @ weights, atol=1e-5)
        old = new


# ---------------------------------------- types are rejected before any float conversion
@pytest.mark.parametrize(
    "bad",
    [
        [True, False, True, True],
        np.array([True, False, True, True]),
        [1 + 0j, 1, 1, 1],
        np.array([1 + 0j, 1, 1, 1]),
        [1 + 2j, 1, 1, 1],
    ],
)
def test_boolean_and_complex_rates_are_rejected_by_every_public_helper(bad):
    degrees = degree_of_feature(3, 3)
    with pytest.raises(TypeError):
        feature_rates(bad, degrees)
    with pytest.raises(TypeError):
        diagonal_learning_rates(bad, degrees)  # was cast to float64 before any check
    with pytest.raises(TypeError):
        diagonal_learning_rates(RATES, degrees, bad)  # as control factors
    with pytest.raises(TypeError):
        apply_degree_control(np.array(RATES), bad)
    with pytest.raises(TypeError):
        apply_degree_control(bad, [1, 1, 1, 1])


def test_complex_rates_do_not_lose_their_imaginary_part_silently():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a ComplexWarning would also fail this
        with pytest.raises(TypeError, match="complex"):
            feature_rates([1 + 2j, 1, 1, 1], degree_of_feature(3, 3))


def test_real_integer_and_float_rates_are_still_accepted():
    degrees = degree_of_feature(3, 3)
    np.testing.assert_array_equal(
        diagonal_learning_rates([3, 2, 1, 0], degrees),
        diagonal_learning_rates(np.array([3.0, 2.0, 1.0, 0.0], np.float32), degrees),
    )


# ------------------------------------- other invalid inputs are handled consistently
@pytest.mark.parametrize(
    "bad, error, message",
    [
        (None, TypeError, "must be numbers"),
        ("abcd", TypeError, "must be numbers"),
        (["1", "1", "1", "1"], TypeError, "must be numbers"),
        ([1, None, 1, 1], TypeError, "must be numbers"),
        ({"a": 1}, TypeError, "must be numbers"),
        ([1, [2, 3], 4, 5], ValueError, "regular array of numbers"),
        ([[1, 1, 1, 1]], ValueError, "one entry per degree"),
        (2.0, ValueError, "one entry per degree"),
        ([1, 1, 1, np.nan], ValueError, "finite and non-negative"),
        ([1, 1, 1, np.inf], ValueError, "finite and non-negative"),
        ([1, 1, 1, -1], ValueError, "finite and non-negative"),
        ([1] * 5, ValueError, "only 4 degree groups"),
    ],
)
def test_invalid_rates_fail_with_the_same_error_in_every_public_helper(
    bad, error, message
):
    degrees = degree_of_feature(3, 3)
    with pytest.raises(error, match=message):
        feature_rates(bad, degrees)
    with pytest.raises(error, match=message):
        diagonal_learning_rates(bad, degrees)


@pytest.mark.parametrize(
    "bad, error",
    [
        (None, TypeError),
        ("abcd", TypeError),
        ([1, [2, 3], 4, 5], ValueError),
        ([[1, 1, 1, 1]], ValueError),
        (0.5, ValueError),  # a control is never broadcast from a scalar
        ([1, 1, 1], ValueError),
        ([1] * 5, ValueError),
        ([1, 1, 1, 1.5], ValueError),
        ([1, 1, 1, -0.1], ValueError),
    ],
)
def test_invalid_control_factors_fail_in_every_public_helper(bad, error):
    from polynomial_cl.continual.degree_rates import as_degree_control

    degrees = degree_of_feature(3, 3)
    with pytest.raises(error):
        as_degree_control(bad)
    with pytest.raises(error):
        apply_degree_control(np.array(RATES), bad)
    if bad is None:
        # ``control=None`` is the documented "no control" value of diagonal_learning_rates
        np.testing.assert_array_equal(
            diagonal_learning_rates(RATES, degrees, bad),
            diagonal_learning_rates(RATES, degrees),
        )
    else:
        with pytest.raises(error):
            diagonal_learning_rates(RATES, degrees, bad)


def test_a_rate_vector_may_be_shorter_than_the_groups_only_when_degrees_stay_below_it():
    # degree-1 data has groups 0 and 1 only, so a two-entry vector indexes it fully
    np.testing.assert_array_equal(
        feature_rates([2.0, 3.0], degree_of_feature(2, 1)),
        [2.0, 3.0, 3.0],
    )
    with pytest.raises(ValueError, match="lie in"):
        feature_rates([2.0, 3.0], degree_of_feature(2, 2))
