# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.continual.strategies import (
    OLD_LR_FACTOR,
    STRATEGIES,
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


def test_the_six_strategies_are_exposed_in_order():
    assert STRATEGIES == (
        "ordinary",
        "ordinary_no_replay",
        "small_old_lr",
        "frozen_old",
        "strategy_b",
        "strategy_b_new",
    )


@pytest.mark.parametrize("strategy", ["ordinary", "ordinary_no_replay"])
def test_ordinary_strategies_restrict_nothing(strategy, setting):
    assert update_restrictions(strategy, **setting) == (None, None, None, None)


@pytest.mark.parametrize(
    "strategy, factor", [("frozen_old", 0.0), ("small_old_lr", OLD_LR_FACTOR)]
)
def test_old_coefficients_get_a_scaled_step_and_everything_else_trains_freely(
    strategy, factor, setting
):
    r = update_restrictions(strategy, **setting)
    assert r.null_basis is None and r.refs is None and r.null_groups is None
    old_block = r.scale[:OLD_ROWS, :OLD_CLASSES]
    assert (old_block == factor).all()
    assert (
        r.scale.sum() - old_block.sum()
    ) == r.scale.size - old_block.size  # all other 1.0


def test_strategy_b_moves_everything_inside_one_null_space_of_the_reference_inputs(
    setting,
):
    r = update_restrictions("strategy_b", **setting)
    assert r.scale is None and r.null_groups is None
    assert len(r.refs) == 3 * OLD_CLASSES  # first stored examples of each old class
    design = setting["train_x"][r.refs]
    np.testing.assert_allclose(design @ r.null_basis, 0, atol=1e-5)


def test_strategy_b_new_freezes_old_coefficients_with_separate_old_and_new_column_bases(
    setting,
):
    r = update_restrictions("strategy_b_new", **setting)
    assert r.null_basis is None
    assert (r.scale[:OLD_ROWS, :OLD_CLASSES] == 0).all() and r.scale.sum() > 0
    (old_cols, old_basis), (new_cols, new_basis) = r.null_groups
    assert old_cols == slice(0, OLD_CLASSES) and new_cols == slice(OLD_CLASSES, N_SEEN)
    assert not old_basis[
        :OLD_ROWS
    ].any()  # old-class columns move only through new feature rows
    design = setting["train_x"][r.refs]
    np.testing.assert_allclose(design @ old_basis, 0, atol=1e-5)
    np.testing.assert_allclose(
        design @ new_basis, 0, atol=1e-5
    )  # new classes see every row


@pytest.mark.parametrize("strategy", ["strategy_b", "strategy_b_new"])
def test_strategy_b_has_no_constraint_in_the_first_experience(strategy, setting):
    first = {**setting, "task": 0, "old_rows": 0, "old_classes": 0, "n_seen": 2}
    assert update_restrictions(strategy, **first) == (None, None, None, None)
