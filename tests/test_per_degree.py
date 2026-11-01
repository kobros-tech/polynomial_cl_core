# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Per-degree learning rates, class-balanced sampling, and the study protocol."""

import numpy as np
import pytest

from polynomial_cl.class_incremental import (
    STRATEGIES,
    candidate_vectors,
    make_feature_data,
    make_synthetic_2d,
    make_task_plan,
    run_condition,
    run_joint_reference,
    run_per_degree_study,
)
from polynomial_cl.class_incremental.degree_rates import (
    N_DEGREE_GROUPS,
    as_degree_rates,
    degree_of_feature,
    group_sizes,
    step_scale_for,
)
from polynomial_cl.class_incremental.degree_study import (
    combine_best_directions,
    mean_degree_diagnostic,
    vector_summary,
)
from polynomial_cl.class_incremental.features import (
    MAX_DEGREE,
    n_polynomial_features,
)
from polynomial_cl.class_incremental.training import (
    _balanced_batches,
    fit_task,
    null_space_basis,
)

EMPTY_X, EMPTY_Y = np.empty((0, 10), np.float32), np.empty(0, np.int64)


@pytest.fixture(scope="module")
def tiny_data():
    x_train, y_train, x_test, y_test, _ = make_synthetic_2d(
        9, n_train_per_class=12, n_test_per_class=6, separation=3.0, noise=0.2
    )
    return make_feature_data(x_train, y_train, x_test, y_test)


def _run(data, strategy="ordinary", seed=2, **kwargs):
    tasks, memory = make_task_plan(data.train_y, seed, 4)
    settings = {
        "epochs": 2,
        "learning_rate": 0.05,
        "batch_size": 8,
        "n_ref_per_class": 1,
    }
    return run_condition(
        "expanding",
        data,
        tasks,
        memory,
        seed=seed,
        strategy=strategy,
        **{**settings, **kwargs},
    )


# ------------------------------------------------------------ rate vector
def test_scalar_is_broadcast_to_every_degree_group():
    np.testing.assert_array_equal(as_degree_rates(0.2), np.full(N_DEGREE_GROUPS, 0.2))
    assert N_DEGREE_GROUPS == MAX_DEGREE + 1


def test_vector_is_copied_and_keeps_degree_order():
    source = [0.1, 0.2, 0.3, 0.4]
    rates = as_degree_rates(source)
    source[0] = 99.0
    np.testing.assert_array_equal(rates, [0.1, 0.2, 0.3, 0.4])  # position k = degree k


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        ([0.1, 0.2], ValueError),  # wrong length
        ([0.1] * 5, ValueError),
        ([[0.1] * 4], ValueError),  # wrong rank
        ([0.1, -0.1, 0.1, 0.1], ValueError),  # negative
        ([0.1, np.nan, 0.1, 0.1], ValueError),
        ([0.1, np.inf, 0.1, 0.1], ValueError),
        (np.nan, ValueError),
        ([0.0] * 4, ValueError),  # nothing would train
        (0.0, ValueError),
        (True, TypeError),  # booleans are not rates
        ("fast", TypeError),
    ],
)
def test_invalid_rate_vectors_are_rejected(bad, error):
    with pytest.raises(error):
        as_degree_rates(bad)


def test_zero_rate_for_some_group_is_allowed():
    assert as_degree_rates([0.0, 0.1, 0.0, 0.1])[0] == 0.0


def test_degree_groups_map_to_the_documented_feature_columns():
    # y = a*x^2 + b*x + c in one variable: columns are c, b, a -> degrees 0, 1, 2.
    np.testing.assert_array_equal(degree_of_feature(1, 2), [0, 1, 2])
    index = degree_of_feature(16, MAX_DEGREE)
    assert len(index) == n_polynomial_features(16, MAX_DEGREE) == 969
    np.testing.assert_array_equal(group_sizes(index), [1, 16, 136, 816])
    # prefix stability: a smaller degree is a prefix of the larger group index
    np.testing.assert_array_equal(degree_of_feature(16, 2), index[:153])


def test_step_scale_reconstructs_the_rate_of_every_feature():
    index = degree_of_feature(2, 3)
    rates = np.array([0.4, 0.1, 0.0, 0.2])
    base, scale = step_scale_for(rates, index)
    np.testing.assert_allclose(base * scale, rates[index], rtol=1e-6)
    assert base == 0.4 and scale.max() == 1.0
    # only active groups count: with degree 1 the cubic rate is irrelevant
    base1, _ = step_scale_for(np.array([0.0, 0.3, 9.0, 9.0]), degree_of_feature(2, 1))
    assert base1 == 0.3
    with pytest.raises(ValueError, match="nothing would be trained"):
        step_scale_for(np.array([0.0, 0.0, 0.5, 0.5]), degree_of_feature(2, 1))


# -------------------------------------------- actual parameter updates
def _toy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 10)).astype(np.float32)
    x[:, 0] = 1.0
    return x, rng.integers(0, 2, size=40)


def _one_step(x, y, group_scale, base, **kwargs):
    weights = np.zeros((10, 10), np.float32)
    fit_task(
        weights, x, y, EMPTY_X, EMPTY_Y, epochs=1, learning_rate=base, batch_size=len(y),
        seed=0, eligible_classes=np.arange(2), group_scale=group_scale, **kwargs,
    )  # fmt: skip
    return weights[:, :2]


def test_each_degree_group_moves_by_exactly_its_own_rate():
    x, y = _toy()
    index = degree_of_feature(2, 3)
    unit = _one_step(x, y, None, 1.0)  # one full-batch step with rate 1 = -gradient
    rates = np.array([0.7, 0.05, 0.3, 0.11])
    base, scale = step_scale_for(rates, index)
    stepped = _one_step(x, y, scale, base)
    for group, rate in enumerate(rates):
        rows = index == group
        np.testing.assert_allclose(
            stepped[rows], rate * unit[rows], rtol=1e-4, atol=1e-7
        )


def test_changing_one_rate_changes_only_that_group():
    x, y = _toy()
    index = degree_of_feature(2, 3)
    rates = np.array([0.2, 0.2, 0.2, 0.2])
    reference = _one_step(x, y, *step_scale_for(rates, index)[::-1])
    for group in range(N_DEGREE_GROUPS):
        varied = rates.copy()
        varied[group] = 0.4
        base, scale = step_scale_for(varied, index)
        moved = _one_step(x, y, scale, base)
        same = index != group
        np.testing.assert_allclose(moved[same], reference[same], atol=1e-7)
        assert np.abs(moved[~same]).sum() > 1.5 * np.abs(reference[~same]).sum()


def test_zero_rate_freezes_a_group_exactly_over_many_updates():
    x, y = _toy()
    index = degree_of_feature(2, 3)
    base, scale = step_scale_for(np.array([0.5, 0.0, 0.5, 0.0]), index)
    weights = np.zeros((10, 10), np.float32)
    fit_task(
        weights, x, y, EMPTY_X, EMPTY_Y, epochs=5, learning_rate=base, batch_size=7,
        seed=1, eligible_classes=np.arange(2), group_scale=scale,
    )  # fmt: skip
    for frozen in (1, 3):
        assert not weights[index == frozen].any()
    assert weights[index == 0].any() and weights[index == 2].any()


def test_group_scale_is_validated():
    x, y = _toy()
    kwargs = {"epochs": 1, "learning_rate": 0.1, "batch_size": 8, "seed": 0}
    w = lambda: np.zeros((10, 10), np.float32)
    for bad in (np.ones(9), -np.ones(10), np.full(10, np.nan), np.zeros(10)):
        with pytest.raises(ValueError, match="group_scale"):
            fit_task(w(), x, y, EMPTY_X, EMPTY_Y, group_scale=bad, **kwargs)
    with pytest.raises(ValueError, match="group_index"):
        fit_task(
            w(), x, y, EMPTY_X, EMPTY_Y, group_index=np.zeros(3, np.int64), **kwargs
        )


def test_group_scale_of_ones_is_the_scalar_rate_and_feature_scale_stays_separate():
    x, y = _toy()
    np.testing.assert_allclose(
        _one_step(x, y, np.ones(10, np.float32), 0.3),
        _one_step(x, y, None, 0.3),
        atol=1e-7,
    )
    # feature_scale (a diagonal preconditioner) and group_scale multiply into one step size
    both = _one_step(
        x,
        y,
        np.full(10, 0.5, np.float32),
        1.0,
        feature_scale=np.full(10, 0.5, np.float32),
    )
    np.testing.assert_allclose(both, _one_step(x, y, None, 0.25), atol=1e-7)


def test_update_masks_and_group_scale_compose():
    x, y = _toy()
    index = degree_of_feature(2, 3)
    base, scale = step_scale_for(np.array([0.5, 0.25, 0.5, 0.125]), index)
    start = np.random.default_rng(3).normal(size=(10, 10)).astype(np.float32)
    mask = np.ones_like(start)
    mask[:3, :2] = 0.0  # frozen_old-like: old rows (degree 0 and 1) of old classes
    mask[3:6, :2] = 0.1  # small_old_lr-like
    out = start.copy()
    fit_task(
        out, x, y, EMPTY_X, EMPTY_Y, epochs=3, learning_rate=base, batch_size=8, seed=0,
        eligible_classes=np.arange(2), update_scale=mask, group_scale=scale,
    )  # fmt: skip
    np.testing.assert_array_equal(
        out[:3, :2], start[:3, :2]
    )  # frozen stays bit-identical
    assert np.any(out[3:, :2] != start[3:, :2])


def test_null_projection_keeps_reference_logits_with_group_rates_including_zero():
    x, y = _toy()
    index = degree_of_feature(2, 3)
    base, scale = step_scale_for(np.array([0.5, 0.0, 0.3, 0.1]), index)
    refs = x[:3]
    basis = null_space_basis(refs * np.sqrt(scale))
    start = np.random.default_rng(4).normal(size=(10, 10)).astype(np.float32)
    out = start.copy()
    fit_task(
        out, x, y, EMPTY_X, EMPTY_Y, epochs=5, learning_rate=base, batch_size=8, seed=0,
        eligible_classes=np.arange(2), null_basis=basis, group_scale=scale,
    )  # fmt: skip
    np.testing.assert_allclose(refs @ out[:, :2], refs @ start[:, :2], atol=1e-4)
    assert np.abs(x[3:] @ out[:, :2] - x[3:] @ start[:, :2]).max() > 1e-3
    np.testing.assert_array_equal(
        out[index == 1], start[index == 1]
    )  # zero rate: untouched


# -------------------------------------------------- run-level behaviour
def test_equal_rate_vector_reproduces_the_scalar_run_for_every_strategy(tiny_data):
    for strategy in STRATEGIES:
        scalar = _run(tiny_data, strategy, learning_rate=0.05)
        vector = _run(tiny_data, strategy, learning_rate=[0.05] * 4)
        assert vector["accuracy_matrix"] == scalar["accuracy_matrix"]
        assert vector["train_loss_by_task"] == scalar["train_loss_by_task"]
        assert vector["per_degree_rates"] and not scalar["per_degree_rates"]


def test_run_records_the_vector_and_applies_each_group_rate_after_expansion(tiny_data):
    rates = [0.2, 0.05, 0.4, 0.1]
    run = _run(tiny_data, learning_rate=rates)
    assert run["learning_rate"] == run["degree_learning_rates"] == rates
    lr = run["degree_diagnostics"]["lr"]
    # degrees 1, 2, 3, 3, 3: a new group appears with its own rate and not before
    assert lr[0] == [0.2, 0.05, None, None]
    assert lr[1] == [0.2, 0.05, 0.4, None]
    assert lr[2] == lr[4] == rates
    assert (
        run["expansion_max_logit_changes"]
        and max(run["expansion_max_logit_changes"]) <= 1e-5
    )


def test_different_rates_produce_independent_coefficient_changes(tiny_data):
    run = _run(tiny_data, learning_rate=[0.2, 0.0, 0.2, 0.2])
    change = run["degree_diagnostics"]["coef_change_l1"]
    assert all(row[1] == 0.0 for row in change)  # frozen linear group never moves
    assert all(row[0] > 0 for row in change)
    assert run["degree_diagnostics"]["n_trainable"][1][1] == 0
    base = _run(tiny_data, learning_rate=0.2)
    assert run["accuracy_matrix"] != base["accuracy_matrix"]


def test_group_sizes_in_the_diagnostics_count_coefficients_per_degree(tiny_data):
    run = _run(tiny_data, learning_rate=0.1)
    n_coefficients = run["degree_diagnostics"]["n_coefficients"]
    assert n_coefficients[0] == [1 * 2, 2 * 2, None, None]  # 2 classes seen
    assert n_coefficients[4] == [1 * 10, 2 * 10, 3 * 10, 4 * 10]  # 10 classes seen


def test_old_coefficient_definition_with_rate_vectors(tiny_data):
    run = _run(tiny_data, "frozen_old", learning_rate=[0.2, 0.2, 0.2, 0.2])
    trainable = run["degree_diagnostics"]["n_trainable"]
    # task 1: classes 0-1 are old, so the old rows of both existing groups are frozen
    assert trainable[1][0] == 1 * 4 - 1 * 2 and trainable[1][1] == 2 * 4 - 2 * 2


def test_strategy_b_keeps_reference_logits_with_a_rate_vector(tiny_data):
    run = _run(tiny_data, "strategy_b", learning_rate=[0.3, 0.0, 0.1, 0.2])
    assert max(c for c in run["ref_logit_change_by_task"] if c is not None) <= 1e-4
    assert all(row[1] == 0.0 for row in run["degree_diagnostics"]["coef_change_l1"])


def test_rate_vector_and_feature_power_are_different_mechanisms(tiny_data):
    with pytest.raises(ValueError, match="feature_power"):
        _run(tiny_data, learning_rate=[0.1] * 4, feature_power=1.0)
    power = _run(tiny_data, learning_rate=0.05, feature_power=1.0)
    assert not power["per_degree_rates"]


def test_all_zero_active_groups_in_a_run_is_rejected(tiny_data):
    with pytest.raises(ValueError, match="nothing would be trained"):
        _run(tiny_data, learning_rate=[0.0, 0.0, 0.3, 0.3])  # experience 1 has degree 1


def test_runs_are_reproducible_for_fixed_seeds(tiny_data):
    for sampling in ("shuffled", "class_balanced"):
        first = _run(tiny_data, learning_rate=[0.1, 0.2, 0.05, 0.3], sampling=sampling)
        again = _run(tiny_data, learning_rate=[0.1, 0.2, 0.05, 0.3], sampling=sampling)
        assert first == again
    assert _run(tiny_data, seed=3) != _run(tiny_data, seed=2)


def test_diagnostics_have_the_documented_shape_and_are_finite(tiny_data):
    run = _run(tiny_data, learning_rate=[0.1, 0.1, 0.1, 0.1])
    for key, rows in run["degree_diagnostics"].items():
        assert len(rows) == 5 and all(len(r) == N_DEGREE_GROUPS for r in rows), key
        assert all(v is None or np.isfinite(v) for r in rows for v in r), key
    assert all(v is None for v in run["degree_diagnostics"]["logit_change"][0])
    assert all(v is not None for v in run["degree_diagnostics"]["logit_change"][1][:2])
    mean = mean_degree_diagnostic([run, run], "mean_step_l1")
    assert mean.shape == (5, N_DEGREE_GROUPS) and np.isnan(mean[0, 3])


# ---------------------------------------------------------- sampling
def test_class_balanced_batches_have_equal_class_counts_and_repeat_small_pools():
    labels = np.repeat(np.arange(4), [3, 500, 500, 40])
    rng = np.random.default_rng(0)
    batches = list(_balanced_batches(labels, 20, 32, rng))
    assert len(batches) == 20 and all(len(b) == 32 for b in batches)
    for batch in batches:
        counts = np.bincount(labels[batch], minlength=4)
        assert counts.min() == 8 and counts.max() == 8  # 32 / 4 classes
    # the 3-example class is used without leaving its own class: every index maps back to it
    small = np.concatenate(batches)
    assert set(labels[small]) == {0, 1, 2, 3}
    assert set(small[labels[small] == 0]) <= {0, 1, 2}


def test_class_balanced_remainder_goes_to_distinct_classes():
    labels = np.repeat(np.arange(3), 50)
    for batch in _balanced_batches(labels, 10, 10, np.random.default_rng(1)):
        counts = np.bincount(labels[batch], minlength=3)
        assert sorted(counts) == [3, 3, 4]


def test_sampling_policies_use_the_same_number_of_updates_and_report_counts(tiny_data):
    shuffled = _run(tiny_data, sampling="shuffled")
    balanced = _run(tiny_data, sampling="class_balanced")
    for a, b in zip(shuffled["sampling_by_task"], balanced["sampling_by_task"]):
        assert a["n_updates"] == b["n_updates"]
        assert a["pool_sizes_by_class"] == b["pool_sizes_by_class"]
        assert a["policy"] == "shuffled" and b["policy"] == "class_balanced"
    last_a, last_b = shuffled["sampling_by_task"][-1], balanced["sampling_by_task"][-1]
    # shuffled: draws follow the pool (new classes dominate); balanced: equal per class
    assert last_a["pool_sizes_by_class"] == [4] * 8 + [12, 12]
    assert last_a["samples_drawn_by_class"] == [4 * 2] * 8 + [12 * 2] * 2
    drawn = np.array(last_b["samples_drawn_by_class"])
    assert drawn.max() - drawn.min() <= last_b["n_updates"] * 2  # remainder slots only
    assert np.allclose(
        last_b["mean_batch_composition"], 0.8, atol=0.3
    )  # 8 / 10 classes


def test_no_future_class_is_ever_drawn_and_replay_pools_come_from_old_classes(
    tiny_data,
):
    for sampling in ("shuffled", "class_balanced"):
        run = _run(tiny_data, sampling=sampling)
        for task, log in enumerate(run["sampling_by_task"]):
            seen = 2 * (task + 1)
            assert not any(log["pool_sizes_by_class"][seen:])
            assert not any(log["samples_drawn_by_class"][seen:])
            assert all(c > 0 for c in log["samples_drawn_by_class"][:seen])
            # old classes only through the fixed memory (<= 4 each), current classes in full
            assert max(log["pool_sizes_by_class"][: 2 * task], default=0) <= 4
            assert min(log["pool_sizes_by_class"][2 * task : seen]) == 12


def test_no_replay_pool_is_current_classes_only_under_both_policies(tiny_data):
    for sampling in ("shuffled", "class_balanced"):
        run = _run(tiny_data, "ordinary_no_replay", sampling=sampling)
        log = run["sampling_by_task"][3]
        assert not any(log["pool_sizes_by_class"][:6]) and not any(
            log["samples_drawn_by_class"][:6]
        )


def test_unknown_sampling_policy_is_rejected(tiny_data):
    with pytest.raises(ValueError, match="sampling"):
        _run(tiny_data, sampling="oversample")


def test_shuffled_sampling_is_the_unchanged_default(tiny_data):
    assert _run(tiny_data) == _run(tiny_data, sampling="shuffled")


# -------------------------------------------- metrics and the study
def test_accuracy_and_forgetting_follow_from_the_matrix(tiny_data):
    run = _run(tiny_data, learning_rate=[0.1, 0.2, 0.1, 0.3])
    matrix = np.array(
        [[np.nan if v is None else v for v in row] for row in run["accuracy_matrix"]]
    )
    assert run["current_accuracy_by_task"] == pytest.approx(list(np.diag(matrix)))
    assert run["final_current_accuracy"] == matrix[-1, -1]
    assert run["final_mean_accuracy"] == pytest.approx(np.mean(matrix[-1]))
    for t in range(1, 5):
        assert run["mean_old_accuracy_by_task"][t] == pytest.approx(
            np.mean(matrix[t, :t])
        )
    best_minus_final = [
        np.nanmax(matrix[:, i]) - matrix[-1, i] for i in range(5)
    ]  # lower triangle has no NaN below the diagonal
    assert run["forgetting_by_task"] == pytest.approx(best_minus_final)
    assert run["mean_forgetting"] == pytest.approx(np.mean(best_minus_final[:-1]))
    assert run["mean_forgetting"] >= 0.0


def test_candidate_vectors_vary_one_group_at_a_time():
    from polynomial_cl.class_incremental.degree_study import RATE_FACTORS

    vectors = candidate_vectors(0.1)
    assert RATE_FACTORS == (0.25, 0.5, 2.0, 4.0)
    assert len(vectors) == 1 + len(RATE_FACTORS) * N_DEGREE_GROUPS
    assert vectors["shared"] == [0.1] * 4
    for name, vector in vectors.items():
        if name != "shared":
            assert sum(v != 0.1 for v in vector) == 1
    assert vectors["d2x4"] == pytest.approx([0.1, 0.1, 0.4, 0.1])
    assert vectors["d0x0.25"] == pytest.approx([0.025, 0.1, 0.1, 0.1])


def test_combined_vector_takes_each_groups_best_direction_only_if_it_helps():
    factors = (0.25, 0.5, 2.0, 4.0)
    scores = {"shared": 0.5}
    for g in range(4):
        for f in factors:
            scores[f"d{g}x{f:g}"] = 0.4
    scores["d1x4"], scores["d1x2"], scores["d2x0.25"], scores["d3x4"] = (
        0.6,
        0.62,
        0.55,
        0.7,
    )
    # group 1 takes its best factor (2, not 4); group 0 keeps the base rate (nothing beats it)
    assert combine_best_directions(0.1, scores) == pytest.approx([0.1, 0.2, 0.025, 0.4])


def test_study_rejects_overlapping_seeds_and_runs_end_to_end(tiny_data):
    with pytest.raises(ValueError, match="overlap"):
        run_per_degree_study(tiny_data, tiny_data, [0, 1], [1, 2])
    study = run_per_degree_study(
        tiny_data, tiny_data, [0], [1, 2],
        strategies=("ordinary",), samplings=("shuffled", "class_balanced"),
        epochs=1, batch_size=16, memory_per_class=4, n_ref_per_class=1,
    )  # fmt: skip
    assert len(study["cells"]) == 2
    for cell in study["cells"]:
        assert cell["selected"] in cell["validation_scores"]
        assert set(cell["runs"]) == set(cell["vectors"])
        assert [r["seed"] for r in cell["runs"]["shared"]] == [1, 2]
        summary = vector_summary(cell["runs"]["shared"])
        assert 0 <= summary["final_mean_accuracy"]["mean"] <= 1


def test_joint_reference_has_no_forgetting_and_is_reproducible(tiny_data):
    first = run_joint_reference(
        tiny_data, degree=3, epochs=2, learning_rate=0.1, batch_size=16
    )
    assert first == run_joint_reference(
        tiny_data, degree=3, epochs=2, learning_rate=0.1, batch_size=16
    )
    assert "mean_forgetting" not in first and "accuracy_matrix" not in first
    assert (
        0.0 <= first["test_accuracy"] <= 1.0 and len(first["per_class_accuracy"]) == 10
    )
    assert first["train_loss"][1] < first["train_loss"][0]
    with pytest.raises(ValueError, match="degree"):
        run_joint_reference(tiny_data, degree=4)


def test_cell_files_are_merged_and_inconsistent_seeds_rejected(tmp_path, tiny_data):
    import json

    from polynomial_cl.class_incremental.degree_study import load_study_cells

    assert load_study_cells(tmp_path) is None
    for strategy in ("frozen_old", "ordinary"):
        study = run_per_degree_study(
            tiny_data, tiny_data, [0], [1], strategies=(strategy,), samplings=("shuffled",),
            epochs=1, batch_size=16, memory_per_class=4, n_ref_per_class=1,
        )  # fmt: skip
        (tmp_path / f"cell_{strategy}.json").write_text(json.dumps(study))
    merged = load_study_cells(tmp_path)
    assert [c["strategy"] for c in merged["cells"]] == ["ordinary", "frozen_old"]
    other = json.loads((tmp_path / "cell_ordinary.json").read_text())
    other["test_seeds"] = [5]
    (tmp_path / "cell_zz.json").write_text(json.dumps(other))
    with pytest.raises(ValueError, match="different seed sets"):
        load_study_cells(tmp_path)


def test_strategy_b_trainable_dimension_excludes_zero_rate_directions(tiny_data):
    rates = [0.3, 0.0, 0.1, 0.2]  # the linear group never moves
    run = _run(tiny_data, "strategy_b", learning_rate=rates)
    index = degree_of_feature(2, 3)
    counts = [3, 6, 10, 10, 10]  # features per experience (degrees 1, 2, 3, 3, 3)
    naive_overcounts = False
    for task, (trainable, null_dim) in enumerate(
        zip(run["n_trainable_by_task"], run["null_dim_by_task"])
    ):
        n_seen = 2 * (task + 1)
        active = int((np.array(rates)[index[: counts[task]]] > 0).sum())
        assert trainable % n_seen == 0
        assert trainable // n_seen <= active  # free directions live on active rows
        if null_dim is None:  # experience 1: no constraint yet, only the rates restrict
            assert trainable == active * n_seen
        else:
            assert trainable // n_seen <= null_dim
            naive_overcounts |= trainable < null_dim * n_seen
    # without zero rates nothing changes: the effective dimension is the null dimension
    full = _run(tiny_data, "strategy_b", learning_rate=0.05)
    assert full["n_trainable_by_task"][1:] == [
        d * 2 * (t + 1) for t, d in enumerate(full["null_dim_by_task"]) if t
    ]


def test_sampling_budget_reports_updates_and_examples_processed(tiny_data):
    shuffled = _run(tiny_data, sampling="shuffled")["sampling_by_task"]
    balanced = _run(tiny_data, sampling="class_balanced")["sampling_by_task"]
    for a, b in zip(shuffled, balanced):
        assert a["n_updates"] == b["n_updates"]
        assert (
            a["examples_processed"] == a["examples_available"]
        )  # nothing dropped or added
        assert sum(a["samples_drawn_by_class"]) == a["examples_processed"]
        assert b["examples_processed"] == sum(b["samples_drawn_by_class"])
        # full batches: at most one batch worth of extra presentations per epoch (2 epochs, 8)
        assert 0 <= b["examples_processed"] - b["examples_available"] < 2 * 8


def _m(acc, forg, cur):
    return {
        "final_mean_accuracy": acc,
        "mean_forgetting": forg,
        "final_current_accuracy": cur,
    }


def test_pareto_front_and_selection_consider_forgetting_not_only_accuracy():
    from polynomial_cl.class_incremental.degree_study import (
        choose_candidate,
        pareto_front,
        tradeoff_score,
    )

    metrics = {
        "high_acc_forgets": _m(0.60, 0.50, 0.90),  # best accuracy, heavy forgetting
        "balanced": _m(0.58, 0.10, 0.60),
        "dominated": _m(0.50, 0.30, 0.50),  # worse than "balanced" on both
        "rigid": _m(0.40, 0.02, 0.10),  # least forgetting, hardly learns
    }
    assert pareto_front(metrics) == ["high_acc_forgets", "balanced", "rigid"]
    assert (
        max(metrics, key=lambda n: metrics[n]["final_mean_accuracy"])
        == "high_acc_forgets"
    )
    assert (
        choose_candidate(metrics) == "balanced"
    )  # accuracy alone would pick the other
    assert tradeoff_score(metrics["balanced"]) == pytest.approx(0.48)
    # a floor on newest-experience accuracy removes the rigid candidate and can change the choice
    metrics["rigid"] = _m(0.58, 0.01, 0.10)
    assert choose_candidate(metrics) == "rigid"
    assert choose_candidate(metrics, min_current_accuracy=0.5) == "balanced"
    # an unreachable floor is ignored rather than leaving nothing to choose
    assert choose_candidate(metrics, min_current_accuracy=0.99) == "rigid"


def test_study_cells_report_validation_metrics_and_the_pareto_front(tiny_data):
    study = run_per_degree_study(
        tiny_data, tiny_data, [0], [1], strategies=("ordinary",), samplings=("shuffled",),
        epochs=1, batch_size=16, memory_per_class=4, n_ref_per_class=1,
        min_current_accuracy=0.0,
    )  # fmt: skip
    cell = study["cells"][0]
    assert set(cell["validation_metrics"]) == set(cell["vectors"])
    assert (
        set(cell["pareto"]) <= set(cell["vectors"])
        and cell["selected"] in cell["pareto"]
    )
    assert "forgetting" in study["selection_rule"]


# ----------------------------------------------------------- strategy_b_new
def test_strategy_b_new_freezes_old_coefficients_and_keeps_reference_logits(tiny_data):
    run = _run(tiny_data, "strategy_b_new", n_ref_per_class=1)
    # features per experience are 3, 6, 10, 10, 10: experiences 4 and 5 add no feature rows,
    # so no old-class output can move at all (old coefficients frozen, no new rows to use)
    assert run["old_logit_change_by_task"][3] == 0.0
    assert run["old_logit_change_by_task"][4] == 0.0
    assert max(c for c in run["ref_logit_change_by_task"] if c is not None) <= 1e-4
    assert run["null_dim_by_task"][0] is None  # nothing to preserve in experience 1
    unconstrained = _run(tiny_data, "strategy_b", n_ref_per_class=1)
    # strategy_b may still move old coefficients (inside the null space); b_new may not
    assert unconstrained["old_logit_change_by_task"][3] > 0.0


def test_strategy_b_new_updates_match_the_documented_subspaces():
    rng = np.random.default_rng(5)
    count, old_rows, old_classes, n_seen = 10, 6, 2, 4
    x = rng.normal(size=(30, count)).astype(np.float32)
    y = rng.integers(old_classes, n_seen, size=30)
    refs = rng.normal(size=(3, count)).astype(np.float32)
    rows = np.arange(old_rows, count)
    rows_basis = null_space_basis(refs[:, rows])
    old_basis = np.zeros((count, rows_basis.shape[1]), np.float32)
    old_basis[rows] = rows_basis
    groups = [
        (slice(0, old_classes), old_basis),
        (slice(old_classes, n_seen), null_space_basis(refs)),
    ]
    mask = np.ones((count, 10), np.float32)
    mask[:old_rows, :old_classes] = 0.0
    start = rng.normal(size=(count, 10)).astype(np.float32)
    out = start.copy()
    fit_task(
        out, x, y, EMPTY_X, EMPTY_Y, epochs=4, learning_rate=0.1, batch_size=8, seed=0,
        eligible_classes=np.arange(n_seen), update_scale=mask, null_groups=groups,
    )  # fmt: skip
    moved = out - start
    assert not moved[:old_rows, :old_classes].any()  # old coefficients bit-identical
    assert np.abs(moved[:, old_classes:n_seen]).max() > 1e-3  # new classes do learn
    np.testing.assert_allclose(
        refs @ moved[:, :n_seen], 0, atol=1e-4
    )  # references unchanged
    assert not moved[:, n_seen:].any()  # classes not introduced yet are untouched


def test_null_groups_are_validated_and_exclusive_with_null_basis():
    x, y = _toy()
    w = np.zeros((10, 10), np.float32)
    kwargs = {"epochs": 1, "learning_rate": 0.1, "batch_size": 8, "seed": 0}
    basis = np.eye(10, 3, dtype=np.float32)
    with pytest.raises(ValueError, match="cannot be combined"):
        fit_task(
            w,
            x,
            y,
            EMPTY_X,
            EMPTY_Y,
            null_basis=basis,
            null_groups=[(slice(0, 2), basis)],
            **kwargs,
        )
    with pytest.raises(ValueError, match="null_groups need"):
        fit_task(w, x, y, EMPTY_X, EMPTY_Y, null_groups=[((0, 2), basis)], **kwargs)
    with pytest.raises(ValueError, match="null_groups need"):
        fit_task(
            w, x, y, EMPTY_X, EMPTY_Y, null_groups=[(slice(0, 2), basis[:5])], **kwargs
        )


def test_strategy_b_new_with_rate_vector_keeps_references_and_zero_group_frozen(
    tiny_data,
):
    run = _run(
        tiny_data,
        "strategy_b_new",
        n_ref_per_class=1,
        learning_rate=[0.3, 0.0, 0.1, 0.2],
    )
    assert max(c for c in run["ref_logit_change_by_task"] if c is not None) <= 1e-4
    assert all(row[1] == 0.0 for row in run["degree_diagnostics"]["coef_change_l1"])


def test_epoch_robustness_reuses_selected_vectors_without_reselecting(tiny_data):
    from polynomial_cl.class_incremental import run_epoch_robustness

    study = run_per_degree_study(
        tiny_data, tiny_data, [0], [1, 2], strategies=("ordinary",), samplings=("shuffled",),
        epochs=1, batch_size=16, memory_per_class=4, n_ref_per_class=1,
    )  # fmt: skip
    longer = run_epoch_robustness(
        tiny_data, study, 2, batch_size=16, memory_per_class=4, n_ref_per_class=1
    )
    cell, source = longer["cells"][0], study["cells"][0]
    assert longer["epochs"] == 2 and longer["test_seeds"] == [1, 2]
    assert cell["selected"] == source["selected"]
    assert set(cell["runs"]) <= {"shared", source["selected"]}
    assert cell["vectors"]["shared"] == source["vectors"]["shared"]
    assert all(r["seed"] in (1, 2) for runs in cell["runs"].values() for r in runs)
    with pytest.raises(ValueError, match="selection seeds"):
        run_epoch_robustness(tiny_data, study, 2, test_seeds=[0])


def test_expansion_tolerance_scales_with_logit_size_but_still_rejects_nonzero_growth(
    tiny_data, monkeypatch
):
    from polynomial_cl.class_incremental import experiment as experiment_module
    from polynomial_cl.class_incremental.experiment import (
        EXPANSION_TOLERANCE,
        expansion_tolerance,
    )

    assert expansion_tolerance(np.full((2, 2), 0.3)) == EXPANSION_TOLERANCE
    assert expansion_tolerance(np.array([[-400.0, 3.0]])) == pytest.approx(
        400 * EXPANSION_TOLERANCE
    )
    # float32 rounding of large logits (> 1e-5 absolute) is accepted ...
    assert 1.2e-5 <= expansion_tolerance(np.array([[100.0]]))
    # ... but a genuinely non-zero new coefficient is not
    real_expand = experiment_module.expand_weights

    def bad_expand(weights, count):
        out = real_expand(weights, count)
        out[weights.shape[0] :] = 0.5
        return out

    monkeypatch.setattr(experiment_module, "expand_weights", bad_expand)
    with pytest.raises(RuntimeError, match="zero-initialized expansion"):
        _run(tiny_data, learning_rate=0.5)


def test_progress_log_reports_stages_and_every_run(tiny_data):
    lines = []
    run_per_degree_study(
        tiny_data, tiny_data, [0], [1], strategies=("ordinary",), samplings=("shuffled",),
        epochs=1, batch_size=16, memory_per_class=4, n_ref_per_class=1, log=lines.append,
    )  # fmt: skip
    text = "\n".join(lines)
    for expected in (
        "stage 1",
        "stage 2",
        "validation:",
        "test 1/",
        "seed 0:",
        "seed 1:",
    ):
        assert expected in text


def test_strategy_b_variants_report_both_null_space_dimensions(tiny_data):
    shared = _run(tiny_data, "strategy_b", n_ref_per_class=1)
    assert shared["null_dim_by_task"][0] is None
    assert shared["null_dim_new_class_columns_by_task"][0] is None
    # strategy_b uses one basis for every column, so the two reported dimensions agree
    assert (
        shared["null_dim_by_task"][1:]
        == shared["null_dim_new_class_columns_by_task"][1:]
    )
    new = _run(tiny_data, "strategy_b_new", n_ref_per_class=1)
    old_dim, new_dim = (
        new["null_dim_by_task"],
        new["null_dim_new_class_columns_by_task"],
    )
    assert old_dim[0] is None and new_dim[0] is None
    # experiences 4 and 5 add no feature rows: nothing is left for the old-class outputs,
    # while the new classes' columns keep a free subspace
    assert old_dim[3] == old_dim[4] == 0
    assert new_dim[3] > 0 and new_dim[4] > 0
    for other in ("ordinary", "frozen_old"):
        run = _run(tiny_data, other)
        assert all(v is None for v in run["null_dim_new_class_columns_by_task"])
