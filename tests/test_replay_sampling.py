# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT
"""Replay memory and sampling policies of the class-incremental package."""

import numpy as np
import pytest

from polynomial_cl.continual.replay import select_memory, stored_indices
from polynomial_cl.continual.sampling import (
    SAMPLING_POLICIES,
    balanced_batches,
    epoch_sampler,
)
from polynomial_cl.training.multiclass import fit_task


def test_memory_is_a_repeat_free_draw_of_one_class_capped_at_its_size():
    labels = np.repeat(np.arange(4), [10, 3, 8, 8])
    memory = select_memory(labels, 1, 5, np.random.default_rng(0))
    assert sorted(memory) == [
        10,
        11,
        12,
    ]  # only 3 examples exist: all of them, no repeats
    memory = select_memory(labels, 0, 4, np.random.default_rng(0))
    assert len(set(memory)) == 4 and (labels[memory] == 0).all()
    with pytest.raises(ValueError, match="missing class 9"):
        select_memory(labels, 9, 4, np.random.default_rng(0))


def test_stored_indices_concatenate_old_classes_in_order_and_cut_a_prefix_per_class():
    memory = {0: np.array([5, 1, 9]), 1: np.array([7, 3]), 2: np.array([4, 8, 2])}
    np.testing.assert_array_equal(stored_indices(memory, 2), [5, 1, 9, 7, 3])
    np.testing.assert_array_equal(stored_indices(memory, 3, 2), [5, 1, 7, 3, 4, 8])
    np.testing.assert_array_equal(  # the previous inline expressions
        stored_indices(memory, 3, 1), np.concatenate([memory[c][:1] for c in range(3)])
    )


def test_policy_names_map_to_samplers_and_unknown_names_are_rejected():
    assert SAMPLING_POLICIES == ("shuffled", "class_balanced")
    assert (
        epoch_sampler("shuffled") is None
    )  # the trainer's own shuffle-every-epoch loop
    assert epoch_sampler("class_balanced") is balanced_batches
    with pytest.raises(ValueError, match="unknown sampling policy"):
        epoch_sampler("oversample")


def _fit_with(sampler):
    rng = np.random.default_rng(0)
    x, y = (
        rng.normal(size=(60, 4)).astype(np.float32),
        np.repeat([0, 1, 2], [40, 15, 5]),
    )
    return fit_task(
        np.zeros((4, 3), np.float32),
        x,
        y,
        x[:0],
        y[:0],
        epochs=2,
        learning_rate=0.1,
        batch_size=16,
        seed=3,
        epoch_sampler=sampler,
    )


def test_the_trainer_uses_the_injected_sampler_and_the_default_matches_shuffled():
    np.testing.assert_array_equal(_fit_with(None), _fit_with(epoch_sampler("shuffled")))
    assert not np.array_equal(
        _fit_with(None), _fit_with(epoch_sampler("class_balanced"))
    )
