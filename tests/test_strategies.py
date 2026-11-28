# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from legacy_restrictions import (
    OLD_LR_FACTOR,
)
from legacy_restrictions import (
    STRATEGIES as LEGACY_STRATEGIES,
)
from legacy_restrictions import (
    update_restrictions as legacy_update_restrictions,
)

from polynomial_cl.continual.strategies import (
    LEGACY_STRATEGY_NAMES,
    METHODS,
    check_beta,
    check_method,
    update_restrictions,
)

COUNT, OLD_ROWS, OLD_CLASSES, N_SEEN = 9, 4, 2, 4


@pytest.fixture
def setting():
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(60, COUNT)).astype(np.float32)
    memory = {c: np.arange(10 * c, 10 * c + 10) for c in range(N_SEEN)}
    return {
        "weights_shape": (COUNT, 10),
        "old_rows": OLD_ROWS,
        "old_classes": OLD_CLASSES,
        "n_seen": N_SEEN,
        "count": COUNT,
        "task": 1,
        "train_x": train_x,
        "memory_indices": memory,
        "n_ref_per_class": 3,
        "step_scale": None,
    }


@pytest.fixture
def current(setting):
    """The arguments of the current ``update_restrictions`` (no per-coefficient geometry)."""
    keep = ("old_classes", "count", "task", "train_x", "memory_indices")
    return {
        **{k: setting[k] for k in keep},
        "n_ref_per_class": setting["n_ref_per_class"],
        "step_scale": setting["step_scale"],
    }


# ------------------------------------------------------------------ current methods
def test_the_two_methods_are_exposed_in_order():
    assert METHODS == ("ordinary", "relaxed")


@pytest.mark.parametrize(
    "name",
    [
        "strategy_b",
        "strategy_b_new",
        "frozen_old",
        "small_old_lr",
        "ordinary_no_replay",
        "nope",
    ],
)
def test_removed_and_unknown_names_are_rejected(name, current):
    with pytest.raises(ValueError, match="unknown method"):
        check_method(name)
    with pytest.raises(ValueError, match="unknown method"):
        update_restrictions(name, **current)


def test_the_ordinary_method_restricts_nothing(current):
    assert update_restrictions("ordinary", **current) == (None, None, None, None)


def test_relaxed_moves_everything_inside_one_null_space_of_the_reference_inputs(
    current,
):
    r = update_restrictions("relaxed", **current)
    assert r.scale is None and r.null_groups is None
    assert len(r.refs) == 3 * OLD_CLASSES  # first stored examples of each old class
    design = current["train_x"][r.refs]
    np.testing.assert_allclose(design @ r.null_basis, 0, atol=1e-5)


def test_relaxed_has_no_constraint_in_the_first_experience(current):
    first = {**current, "task": 0, "old_classes": 0}
    assert update_restrictions("relaxed", **first) == (None, None, None, None)


def test_the_relaxed_basis_equals_the_one_of_the_former_strategy_b(setting, current):
    new = update_restrictions("relaxed", **current)
    old = legacy_update_restrictions("strategy_b", **setting)
    np.testing.assert_array_equal(new.refs, old.refs)
    np.testing.assert_array_equal(new.null_basis, old.null_basis)


def test_beta_is_validated_on_the_closed_unit_interval():
    for ok in (0, 0.0, 0.5, 1, 1.0, np.float32(0.25)):
        assert check_beta(ok) == float(ok)
    for bad in (-1e-9, 1.000001, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="beta"):
            check_beta(bad)
    for bad in (True, "0.5", None):
        with pytest.raises(TypeError, match="beta"):
            check_beta(bad)
    assert check_beta(0.0, "ordinary") == 0.0
    with pytest.raises(ValueError, match="relaxed"):
        check_beta(0.5, "ordinary")


def test_legacy_names_are_known_only_as_names_for_the_legacy_loader():
    assert set(LEGACY_STRATEGY_NAMES) == set(LEGACY_STRATEGIES)


# ------------------------------------------- frozen pre-C4 builders (test support only)
# tests/legacy_restrictions.py builds the restrictions behind the C2 golden fixtures of
# ``fit_task``'s update_scale / null_groups arguments. These tests keep that helper honest.
@pytest.mark.parametrize("strategy", ["ordinary", "ordinary_no_replay"])
def test_legacy_ordinary_strategies_restrict_nothing(strategy, setting):
    assert legacy_update_restrictions(strategy, **setting) == (None, None, None, None)


@pytest.mark.parametrize(
    "strategy, factor", [("frozen_old", 0.0), ("small_old_lr", OLD_LR_FACTOR)]
)
def test_legacy_old_coefficients_get_a_scaled_step_and_everything_else_trains_freely(
    strategy, factor, setting
):
    r = legacy_update_restrictions(strategy, **setting)
    assert r.null_basis is None and r.refs is None and r.null_groups is None
    old_block = r.scale[:OLD_ROWS, :OLD_CLASSES]
    assert (old_block == factor).all()
    assert (
        r.scale.sum() - old_block.sum()
    ) == r.scale.size - old_block.size  # all other 1.0


def test_legacy_strategy_b_new_freezes_old_coefficients_with_separate_column_bases(
    setting,
):
    r = legacy_update_restrictions("strategy_b_new", **setting)
    assert r.null_basis is None
    assert (r.scale[:OLD_ROWS, :OLD_CLASSES] == 0).all() and r.scale.sum() > 0
    (old_cols, old_basis), (new_cols, new_basis) = r.null_groups
    assert old_cols == slice(0, OLD_CLASSES) and new_cols == slice(OLD_CLASSES, N_SEEN)
    assert not old_basis[
        :OLD_ROWS
    ].any()  # old-class columns move through new rows only
    design = setting["train_x"][r.refs]
    np.testing.assert_allclose(design @ old_basis, 0, atol=1e-5)
    np.testing.assert_allclose(design @ new_basis, 0, atol=1e-5)
