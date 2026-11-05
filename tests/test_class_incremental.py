# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.continual import (
    CONDITIONS,
    STRATEGIES,
    TASK_ORDER,
    holdout_split,
    make_feature_data,
    make_task_plan,
    run_comparison,
    run_condition,
    run_strategy_comparison,
    summarize_runs,
)
from polynomial_cl.continual import experiment as experiment_module
from polynomial_cl.continual.experiment import accuracy_by_task
from polynomial_cl.continual.features import (
    monomial_exponents,
    n_polynomial_features,
    scaled_polynomial_features,
)
from polynomial_cl.data import make_synthetic_2d
from polynomial_cl.data.mnist import make_projection, project_images
from polynomial_cl.models.multiclass import expand_weights
from polynomial_cl.training.multiclass import (
    cross_entropy,
    fit_task,
    null_space_basis,
)


# ---------------------------------------------------------------- data
def test_synthetic_data_is_reproducible_with_independent_splits():
    first = make_synthetic_2d(4, n_train_per_class=5, n_test_per_class=7)
    second = make_synthetic_2d(4, n_train_per_class=5, n_test_per_class=7)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    x_train, y_train, x_test, y_test, means = first
    assert (
        x_train.shape == (50, 2) and x_test.shape == (70, 2) and means.shape == (10, 2)
    )
    assert set(y_train) == set(y_test) == set(range(10))
    assert not np.array_equal(x_train[:5], x_test[:5])


@pytest.mark.parametrize(
    "kwargs", [{"n_train_per_class": 0}, {"separation": 0.0}, {"noise": float("nan")}]
)
def test_synthetic_data_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        make_synthetic_2d(0, **kwargs)


def test_tasks_introduce_every_class_exactly_once():
    assert [c for pair in TASK_ORDER for c in pair] == list(range(10))


def test_projection_is_deterministic_and_orthonormal():
    first = make_projection(seed=9, input_dim=12, n_components=4)
    np.testing.assert_array_equal(
        first, make_projection(seed=9, input_dim=12, n_components=4)
    )
    np.testing.assert_allclose(first.T @ first, np.eye(4), atol=1e-6)
    assert project_images(np.zeros((3, 12)), first).shape == (3, 4)
    with pytest.raises(ValueError):
        project_images(np.zeros((3, 11)), first)


# ------------------------------------------------------------ features
def test_feature_counts_match_the_documented_sizes():
    assert [n_polynomial_features(16, d) for d in (1, 2, 3)] == [17, 153, 969]
    for degree in range(4):
        assert len(monomial_exponents(4, degree)) == n_polynomial_features(4, degree)


def test_features_are_prefix_stable_and_finite():
    x = np.random.default_rng(4).normal(size=(7, 4)).astype(np.float32)
    linear, cubic = scaled_polynomial_features(x, 1), scaled_polynomial_features(x, 3)
    assert np.isfinite(cubic).all()
    np.testing.assert_array_equal(cubic[:, : linear.shape[1]], linear)


def test_features_known_values_and_degree_group_scaling():
    x = np.array([[2.0, 3.0]], dtype=np.float32)
    got = scaled_polynomial_features(x, 2)[0]
    # constant stays 1; degree-1 group has 2 monomials, degree-2 group has 3.
    expected = [1, 2 / 2**0.5, 3 / 2**0.5, 4 / 3**0.5, 6 / 3**0.5, 9 / 3**0.5]
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_feature_chunking_does_not_change_the_result():
    x = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    np.testing.assert_array_equal(
        scaled_polynomial_features(x, 3, chunk_size=3), scaled_polynomial_features(x, 3)
    )


def test_zero_initialised_expansion_preserves_logits_exactly():
    rng = np.random.default_rng(5)
    old_features = rng.normal(size=(11, 17)).astype(np.float32)
    old_weights = rng.normal(size=(17, 10)).astype(np.float32)
    expanded = expand_weights(old_weights, 153)
    new_features = rng.normal(size=(11, 153)).astype(np.float32)
    new_features[:, :17] = old_features
    np.testing.assert_array_equal(expanded[:17], old_weights)
    np.testing.assert_array_equal(expanded[17:], 0.0)
    np.testing.assert_allclose(
        new_features @ expanded, old_features @ old_weights, atol=1e-5
    )
    assert expand_weights(old_weights, 17) is old_weights
    with pytest.raises(ValueError, match="cannot shrink"):
        expand_weights(old_weights, 16)


# ------------------------------------------------------------ training
def _fit(weights, x, y, eligible, replay=None, epochs=2, **restrictions):
    rx, ry = (
        replay
        if replay
        else (np.empty((0, x.shape[1]), np.float32), np.empty(0, np.int64))
    )
    return fit_task(
        weights, x, y, rx, ry, epochs=epochs, learning_rate=0.1, batch_size=3, seed=5,
        eligible_classes=np.array(eligible), **restrictions,
    )  # fmt: skip


def test_training_never_updates_future_class_weights():
    x = np.array([[1.0, 0.5], [-0.5, 1.0], [0.2, -1.0]], dtype=np.float32)
    trained = _fit(np.zeros((2, 10), np.float32), x, np.array([0, 1, 0]), [0, 1])
    assert np.any(trained[:, :2] != 0)
    np.testing.assert_array_equal(trained[:, 2:], 0.0)


def test_training_rejects_labels_outside_eligible_classes():
    with pytest.raises(ValueError, match="belong to eligible_classes"):
        _fit(
            np.zeros((2, 4), np.float32),
            np.ones((2, 2), np.float32),
            np.array([0, 2]),
            [0, 1],
        )


def test_training_separates_two_easy_classes_and_uses_replay_rows():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 0.3, (20, 1)), rng.normal(2, 0.3, (20, 1))])
    x = np.column_stack([np.ones(40), x]).astype(np.float32)
    y = np.repeat([0, 1], 20)
    w = _fit(np.zeros((2, 10), np.float32), x, y, [0, 1], epochs=30)
    assert np.mean((x @ w[:, :2]).argmax(axis=1) == y) == 1.0
    # Replay rows of an earlier class enter the update: the class-0 column differs.
    with_replay = _fit(
        np.zeros((2, 10), np.float32), x[20:], y[20:], [0, 1], replay=(x[:5], y[:5])
    )
    without = _fit(np.zeros((2, 10), np.float32), x[20:], y[20:], [0, 1])
    assert not np.array_equal(with_replay, without)


# ---------------------------------------------------------- task plan
def test_task_plan_is_deterministic_and_respects_classes_and_memory():
    labels = np.repeat(np.arange(10), 6)
    tasks, memory = make_task_plan(labels, seed=12, memory_per_class=2)
    again_tasks, again_memory = make_task_plan(labels, seed=12, memory_per_class=2)
    for task, classes, repeat in zip(tasks, TASK_ORDER, again_tasks):
        assert set(labels[task]) == set(classes)
        np.testing.assert_array_equal(task, repeat)
    for class_id in range(10):
        np.testing.assert_array_equal(memory[class_id], again_memory[class_id])
        assert len(memory[class_id]) == 2 and np.all(
            labels[memory[class_id]] == class_id
        )
    _, capped = make_task_plan(labels, seed=0, memory_per_class=100)
    assert all(len(m) == 6 for m in capped.values())  # never more than are available
    with pytest.raises(ValueError):
        make_task_plan(labels, seed=0, memory_per_class=0)
    with pytest.raises(ValueError, match="missing"):
        make_task_plan(labels[labels < 8], seed=0, memory_per_class=1)


def test_evaluation_only_lets_introduced_classes_compete():
    weights = np.zeros((1, 10), dtype=np.float32)
    weights[0, 9] = 100.0  # class 9 would win everything if it were allowed to compete
    features = np.ones((4, 1), dtype=np.float32)
    labels = np.array([0, 1, 0, 1])
    accuracy = accuracy_by_task(weights, features, labels, n_tasks_seen=1)
    assert accuracy[0] == pytest.approx(0.5)  # ties between 0 and 1 resolve to class 0
    assert np.isnan(accuracy[1:]).all()


# --------------------------------------------------------- experiment
@pytest.fixture(scope="module")
def tiny_data():
    x_train, y_train, x_test, y_test, _ = make_synthetic_2d(
        9, n_train_per_class=6, n_test_per_class=5, separation=3.0, noise=0.2
    )
    return make_feature_data(x_train, y_train, x_test, y_test)


def _run_all(data, seed=2):
    tasks, memory = make_task_plan(data.train_y, seed, memory_per_class=2)
    return {
        c: run_condition(c, data, tasks, memory, seed=seed, epochs=1, learning_rate=0.05,
                         batch_size=8)
        for c in CONDITIONS
    }  # fmt: skip


def test_conditions_differ_only_in_degree_schedule(tiny_data):
    results = _run_all(tiny_data)
    assert results["fixed_linear"]["feature_count_by_task"] == [3] * 5
    assert results["expanding"]["feature_count_by_task"] == [3, 6, 10, 10, 10]
    assert results["fixed_capacity"]["feature_count_by_task"] == [10] * 5


def test_expansion_invariant_is_checked_and_recorded(tiny_data):
    results = _run_all(tiny_data)
    assert len(results["expanding"]["expansion_max_logit_changes"]) == 2
    assert max(results["expanding"]["expansion_max_logit_changes"]) <= 1e-5
    assert results["fixed_linear"]["expansion_max_logit_changes"] == []
    assert results["fixed_capacity"]["expansion_max_logit_changes"] == []


def test_a_broken_expansion_is_rejected(tiny_data, monkeypatch):
    def corrupt(weights, new_count):
        out = np.ones((new_count, weights.shape[1]), dtype=weights.dtype)
        out[: weights.shape[0]] = weights
        return out

    monkeypatch.setattr(experiment_module, "expand_weights", corrupt)
    tasks, memory = make_task_plan(tiny_data.train_y, 0, 2)
    with pytest.raises(RuntimeError, match="changed logits"):
        run_condition("expanding", tiny_data, tasks, memory, seed=0, epochs=1)


def test_accuracy_matrix_is_lower_triangular_and_metrics_follow_from_it(tiny_data):
    for result in _run_all(tiny_data).values():
        matrix = np.array(
            [[np.nan if v is None else v for v in r] for r in result["accuracy_matrix"]]
        )
        assert matrix.shape == (5, 5)
        assert np.isnan(matrix[np.triu_indices(5, k=1)]).all()
        assert np.isfinite(matrix[np.tril_indices(5)]).all()
        # Metrics recomputed independently from the matrix.
        forgetting = [max(0.0, np.max(matrix[t:, t]) - matrix[-1, t]) for t in range(5)]
        np.testing.assert_allclose(result["forgetting_by_task"], forgetting)
        assert result["mean_forgetting"] == pytest.approx(
            np.mean(forgetting[:-1])
        )  # last excluded
        assert result["final_mean_accuracy"] == pytest.approx(matrix[-1].mean())
        assert result["final_current_accuracy"] == pytest.approx(matrix[-1, -1])


def test_unknown_condition_is_rejected(tiny_data):
    tasks, memory = make_task_plan(tiny_data.train_y, 0, 2)
    with pytest.raises(ValueError, match="unknown condition"):
        run_condition("nope", tiny_data, tasks, memory, seed=0)


def test_comparison_is_deterministic_and_summarises_over_seeds(tiny_data):
    kwargs = {
        "epochs": 1,
        "learning_rate": 0.05,
        "batch_size": 8,
        "memory_per_class": 2,
    }
    first = run_comparison(tiny_data, [0, 1], **kwargs)
    second = run_comparison(tiny_data, [0, 1], **kwargs)
    assert first == second
    assert len(first["runs"]) == 2 * len(CONDITIONS)
    assert set(first["summary"]) == set(CONDITIONS)
    values = [
        r["final_mean_accuracy"] for r in first["runs"] if r["condition"] == "expanding"
    ]
    stats = first["summary"]["expanding"]["final_mean_accuracy"]
    assert stats["mean"] == pytest.approx(np.mean(values))
    assert stats["std"] == pytest.approx(np.std(values, ddof=1))
    with pytest.raises(ValueError, match="seed"):
        run_comparison(tiny_data, [], **kwargs)


def test_summary_uses_sample_std_and_zero_for_a_single_seed():
    rows = [
        {"condition": c, "final_mean_accuracy": a, "final_current_accuracy": a, "mean_forgetting": f}
        for c, a, f in (("expanding", 0.7, 0.1), ("expanding", 0.9, 0.3), ("fixed_linear", 0.5, 0.2))
    ]  # fmt: skip
    summary = summarize_runs(rows)
    assert summary["expanding"]["final_mean_accuracy"]["mean"] == pytest.approx(0.8)
    assert summary["expanding"]["final_mean_accuracy"]["std"] == pytest.approx(
        np.std([0.7, 0.9], ddof=1)
    )
    assert summary["fixed_linear"]["mean_forgetting"] == {"mean": 0.2, "std": 0.0}


# ------------------------------------------------------------ strategies
def test_null_space_basis_annihilates_the_design_and_is_orthonormal():
    design = np.random.default_rng(1).normal(size=(5, 12))
    basis = null_space_basis(design)
    assert basis.shape == (12, 7)
    np.testing.assert_allclose(design @ basis, 0.0, atol=1e-5)
    np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=1e-5)
    assert null_space_basis(np.empty((0, 4))).shape == (
        4,
        4,
    )  # no constraint -> everything free
    assert null_space_basis(np.random.default_rng(2).normal(size=(6, 6))).shape == (
        6,
        0,
    )


def _toy_problem():
    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(30), rng.normal(size=(30, 3))]).astype(np.float32)
    return x, rng.integers(0, 2, size=30)


def test_frozen_and_slowed_coefficients_follow_update_scale():
    x, y = _toy_problem()
    start = np.random.default_rng(4).normal(size=(4, 10)).astype(np.float32)
    scale = np.ones_like(start)
    scale[:2, :2] = 0.0
    frozen = _fit(start.copy(), x, y, [0, 1], update_scale=scale)
    np.testing.assert_array_equal(frozen[:2, :2], start[:2, :2])
    assert np.any(frozen[2:, :2] != start[2:, :2]) and np.any(
        frozen[:2, :2] == start[:2, :2]
    )
    scale[:2, :2] = 0.1
    slowed = _fit(start.copy(), x, y, [0, 1], update_scale=scale)
    free = _fit(start.copy(), x, y, [0, 1])
    moved = lambda w: np.abs(w[:2, :2] - start[:2, :2]).sum()
    assert 0 < moved(slowed) < moved(free)


def test_null_basis_keeps_reference_logits_fixed_while_other_inputs_move():
    x, y = _toy_problem()
    refs = x[:3]
    basis = null_space_basis(refs)
    start = np.random.default_rng(5).normal(size=(4, 10)).astype(np.float32)
    trained = _fit(start.copy(), x, y, [0, 1], null_basis=basis, epochs=5)
    np.testing.assert_allclose(refs @ trained[:, :2], refs @ start[:, :2], atol=1e-5)
    assert np.abs(x[3:] @ trained[:, :2] - x[3:] @ start[:, :2]).max() > 1e-3


def test_update_restrictions_validate_their_shapes():
    x, y = _toy_problem()
    with pytest.raises(ValueError, match="update_scale"):
        _fit(np.zeros((4, 10), np.float32), x, y, [0, 1], update_scale=np.ones((4, 9)))
    with pytest.raises(ValueError, match="null_basis"):
        _fit(np.zeros((4, 10), np.float32), x, y, [0, 1], null_basis=np.ones((3, 2)))


def _strategy_runs(data, **kwargs):
    tasks, memory = make_task_plan(data.train_y, 2, 2)
    return {
        s: run_condition("expanding", data, tasks, memory, seed=2, epochs=1, learning_rate=0.05,
                         batch_size=8, strategy=s, **kwargs)
        for s in STRATEGIES
    }  # fmt: skip


def test_default_strategy_is_ordinary_and_unknown_strategies_are_rejected(tiny_data):
    tasks, memory = make_task_plan(tiny_data.train_y, 2, 2)
    kwargs = {"seed": 2, "epochs": 1, "learning_rate": 0.05, "batch_size": 8}
    default = run_condition("expanding", tiny_data, tasks, memory, **kwargs)
    explicit = run_condition(
        "expanding", tiny_data, tasks, memory, strategy="ordinary", **kwargs
    )
    assert default == explicit
    with pytest.raises(ValueError, match="unknown strategy"):
        run_condition("expanding", tiny_data, tasks, memory, strategy="nope", **kwargs)


def test_old_coefficient_definition_gives_the_expected_trainable_counts(tiny_data):
    # 2-D features at degrees 1,2,3,3,3 -> 3,6,10,10,10 rows; classes introduced 2 per experience.
    # Old coefficients = (rows present before) x (classes introduced before):
    #   exp2: 3x2, exp3: 6x4, exp4: 10x6, exp5: 10x8, exp1: none.
    runs = _strategy_runs(tiny_data)
    total = [3 * 2, 6 * 4, 10 * 6, 10 * 8, 10 * 10]
    assert runs["ordinary"]["n_trainable_by_task"] == total
    assert runs["frozen_old"]["n_trainable_by_task"] == [
        6,
        24 - 6,
        60 - 24,
        80 - 60,
        100 - 80,
    ]
    assert runs["small_old_lr"]["n_trainable_by_task"] == total  # slowed, not frozen


def test_strategy_b_preserves_reference_logits_and_reports_the_free_dimension(
    tiny_data,
):
    runs = _strategy_runs(tiny_data, n_ref_per_class=1)
    b = runs["strategy_b"]
    # one reference per old class -> 2, 4, 6, 8 constraints on 6, 10, 10, 10 features
    assert b["null_dim_by_task"] == [None, 4, 6, 4, 2]
    assert max(c for c in b["ref_logit_change_by_task"] if c is not None) <= 1e-4
    assert b["n_trainable_by_task"][1:] == [4 * 4, 6 * 6, 4 * 8, 2 * 10]
    for name in set(STRATEGIES) - {"strategy_b", "strategy_b_new"}:
        assert all(v is None for v in runs[name]["null_dim_by_task"])
        assert all(v is None for v in runs[name]["ref_logit_change_by_task"])


def test_indicators_are_recorded_for_every_strategy(tiny_data):
    for name, run in _strategy_runs(tiny_data).items():
        assert len(run["train_loss_by_task"]) == 5, name
        losses = run["train_loss_by_task"]
        assert all(before >= after >= 0 for before, after in losses), name
        if name not in (
            "strategy_b",
            "strategy_b_new",
        ):  # B may have no free dimension left, i.e. nothing to learn
            assert all(before > after for before, after in losses), name
            assert all(v > 0 for v in run["l1_update_by_task"]), name
        assert run["old_logit_change_by_task"][0] is None
        assert all(v >= 0 for v in run["old_logit_change_by_task"][1:])
        assert 0 <= run["mean_old_flip_fraction"] <= 1
        assert run["old_takeover_fraction_by_task"][0] is None
        assert all(0 <= v <= 1 for v in run["old_takeover_fraction_by_task"][1:])
        assert 0 <= run["mean_old_takeover_fraction"] <= 1


def test_new_class_takeover_is_separate_from_flips_among_old_classes(
    tiny_data, monkeypatch
):
    """A new class can steal old examples although the old logits do not move at all."""

    def boost_new_classes(weights, *args, eligible_classes, **kwargs):
        # Bias of the two newest classes ends up far above every existing class's bias.
        weights[0, eligible_classes[-2:]] = weights[0, eligible_classes].max() + 1000.0
        return weights

    monkeypatch.setattr(experiment_module, "fit_task", boost_new_classes)
    tasks, memory = make_task_plan(tiny_data.train_y, 0, 2)
    run = run_condition("fixed_linear", tiny_data, tasks, memory, seed=0, epochs=1)
    assert run["old_takeover_fraction_by_task"] == [None, 1.0, 1.0, 1.0, 1.0]
    assert run["old_flip_fraction_by_task"] == [None, 0.0, 0.0, 0.0, 0.0]
    assert run["old_logit_change_by_task"] == [None, 0.0, 0.0, 0.0, 0.0]
    assert run["mean_old_takeover_fraction"] == 1.0
    # The accuracy matrix shows the consequence: old experiences collapse.
    final = run["accuracy_matrix"][-1]
    assert final[0] == 0.0 and final[-1] > 0.0


def test_takeover_bounds_the_accuracy_on_old_experiences(tiny_data):
    """An old example predicted as a new class is wrong, so accuracy <= 1 - takeover.

    Every class has the same number of test samples, so the mean over the old experiences'
    accuracies is the accuracy over all old test inputs.
    """
    tasks, memory = make_task_plan(tiny_data.train_y, 0, 2)
    for condition in CONDITIONS:
        run = run_condition(condition, tiny_data, tasks, memory, seed=0, epochs=2)
        for task in range(1, 5):
            old_accuracy = np.mean(run["accuracy_matrix"][task][:task])
            assert (
                old_accuracy <= 1.0 - run["old_takeover_fraction_by_task"][task] + 1e-9
            )


def test_frozen_old_changes_old_logits_less_than_ordinary_training(tiny_data):
    runs = _strategy_runs(tiny_data)
    assert (
        runs["frozen_old"]["mean_old_logit_change"]
        < runs["ordinary"]["mean_old_logit_change"]
    )


def test_strategy_comparison_pairs_strategies_within_a_seed(tiny_data):
    kwargs = {
        "epochs": 1,
        "learning_rate": 0.05,
        "batch_size": 8,
        "memory_per_class": 2,
    }
    result = run_strategy_comparison(tiny_data, [0, 1], **kwargs)
    assert result == run_strategy_comparison(tiny_data, [0, 1], **kwargs)
    assert len(result["runs"]) == 2 * len(STRATEGIES)
    assert set(result["summary"]) == set(STRATEGIES)
    assert "mean_old_flip_fraction" in result["summary"]["strategy_b"]
    with pytest.raises(ValueError, match="seed"):
        run_strategy_comparison(tiny_data, [], **kwargs)


# ------------------------------------------------- per-feature step sizes
def test_holdout_split_is_deterministic_disjoint_and_validates():
    x = np.arange(200).reshape(-1, 1).astype(np.float32)
    y = np.repeat(np.arange(10), 20)
    a = holdout_split(x, y, 0.25, seed=3)
    b = holdout_split(x, y, 0.25, seed=3)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    fit_x, fit_y, held_x, held_y = a
    assert (
        len(fit_x) == 150
        and len(held_x) == 50
        and len(fit_y) == 150
        and len(held_y) == 50
    )
    assert not set(fit_x.ravel()) & set(held_x.ravel())
    for fraction in (0.0, 1.0):
        with pytest.raises(ValueError, match="fraction"):
            holdout_split(x, y, fraction)
    with pytest.raises(ValueError, match="missing a class"):
        holdout_split(x[:12], np.arange(12) % 10, 0.1)


def test_feature_scale_of_ones_changes_nothing_and_other_scales_do():
    x, y = _toy_problem()
    start = np.random.default_rng(6).normal(size=(4, 10)).astype(np.float32)
    plain = _fit(start.copy(), x, y, [0, 1])
    ones = _fit(start.copy(), x, y, [0, 1], feature_scale=np.ones(4, np.float32))
    np.testing.assert_allclose(ones, plain, atol=1e-6)
    scaled = _fit(
        start.copy(), x, y, [0, 1], feature_scale=np.array([1, 1, 1, 8], np.float32)
    )
    assert np.abs(scaled[3] - start[3]).sum() > np.abs(plain[3] - start[3]).sum()
    with pytest.raises(ValueError, match="feature_scale"):
        _fit(
            start.copy(), x, y, [0, 1], feature_scale=np.array([1, 1, 1, 0], np.float32)
        )


def test_preconditioned_null_projection_keeps_constraint_and_descends():
    x, y = _toy_problem()
    scale = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    refs = x[:2]
    basis = null_space_basis(refs * np.sqrt(scale))
    start = np.random.default_rng(7).normal(size=(4, 10)).astype(np.float32)
    trained = _fit(
        start.copy(), x, y, [0, 1], null_basis=basis, feature_scale=scale, epochs=5
    )
    np.testing.assert_allclose(refs @ trained[:, :2], refs @ start[:, :2], atol=1e-4)
    before = cross_entropy(start, x, y, 2)
    assert cross_entropy(trained, x, y, 2) < before


def test_feature_power_runs_for_every_strategy_and_zero_is_the_default(tiny_data):
    tasks, memory = make_task_plan(tiny_data.train_y, 2, 2)
    kwargs = {"seed": 2, "epochs": 1, "learning_rate": 0.05, "batch_size": 8}
    default = run_condition("expanding", tiny_data, tasks, memory, **kwargs)
    assert default == run_condition(
        "expanding", tiny_data, tasks, memory, feature_power=0.0, **kwargs
    )
    for strategy in STRATEGIES:
        run = run_condition(
            "expanding", tiny_data, tasks, memory, strategy=strategy, feature_power=1.0,
            n_ref_per_class=1, **kwargs,
        )  # fmt: skip
        assert run["accuracy_matrix"] != default["accuracy_matrix"] or strategy in (
            "strategy_b",
            "strategy_b_new",
        )
        assert np.isfinite(run["final_mean_accuracy"]) and np.isfinite(
            run["mean_loss_reduction"]
        )
        if strategy == "strategy_b":
            assert (
                max(c for c in run["ref_logit_change_by_task"] if c is not None) <= 1e-4
            )


def test_per_class_accuracy_comes_from_the_same_predictions_as_the_task_accuracy():
    x_train, y_train, x_test, y_test, _ = make_synthetic_2d(
        0, n_train_per_class=30, n_test_per_class=20
    )
    data = make_feature_data(x_train, y_train, x_test, y_test)
    run = run_condition(
        "expanding", data, *make_task_plan(y_train, 0, 10), seed=0, epochs=2
    )
    by_class = np.array(
        [
            [np.nan if v is None else v for v in row]
            for row in run["accuracy_by_class_matrix"]
        ]
    )
    matrix = np.array(
        [[np.nan if v is None else v for v in row] for row in run["accuracy_matrix"]]
    )
    assert by_class.shape == (5, 10)
    for t in range(5):  # classes appear in their own experience and never earlier
        assert np.isfinite(by_class[t, : 2 * (t + 1)]).all()
        assert np.isnan(by_class[t, 2 * (t + 1) :]).all()
        # both classes of an experience have equal test counts, so their mean is the task accuracy
        for task, classes in enumerate(TASK_ORDER[: t + 1]):
            assert np.mean(by_class[t, list(classes)]) == pytest.approx(matrix[t, task])
