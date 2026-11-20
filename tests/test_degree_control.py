# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Degree-control matrix M: effective step ``H M`` for every strategy."""

import numpy as np
import pytest

from polynomial_cl.continual import make_feature_data, make_task_plan, run_condition
from polynomial_cl.continual.degree_rates import (
    apply_degree_control,
    as_degree_control,
    degree_of_feature,
    step_scale_for,
)
from polynomial_cl.continual.experiment import MAX_DEGREE
from polynomial_cl.continual.strategies import STRATEGIES
from polynomial_cl.training.multiclass import fit_task, null_space_basis

RATES = [3.0, 3.0, 0.75, 3.0]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(10, 16)) * 1.5
    y = np.repeat(np.arange(10), 50)
    x = (centres[y] + rng.normal(size=(500, 16))).astype(np.float32)
    ty = np.repeat(np.arange(10), 20)
    tx = (centres[ty] + rng.normal(size=(200, 16))).astype(np.float32)
    return make_feature_data(x, y, tx, ty)


def run(data, strategy="ordinary", factors=None, rates=RATES, seed=3, **kw):
    tasks, memory = make_task_plan(data.train_y, seed, 20)
    return run_condition(
        "expanding",
        data,
        tasks,
        memory,
        seed=seed,
        epochs=2,
        batch_size=32,
        strategy=strategy,
        learning_rate=rates,
        sampling="class_balanced",
        degree_control_factors=factors,
        diagnostics=True,  # the comparisons below read indicators and per-degree statistics
        **kw,
    )


def same(a, b):
    keys = (
        "accuracy_matrix",
        "forgetting_by_task",
        "train_loss_by_task",
        "l1_update_by_task",
    )
    return all(a[k] == b[k] for k in keys)


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    "bad",
    [
        [1, 1, 1],
        [1] * 5,
        1.0,
        [[1, 1, 1, 1]],
        [1, 1, 1, -0.1],
        [1, 1, 1, 1.01],
        [1, 1, 1, np.nan],
        [1, 1, 1, np.inf],
        [True, 1, 1, 1],
        ["a", "b", "c", "d"],
    ],
)
def test_invalid_factors_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        as_degree_control(bad)


def test_run_condition_rejects_invalid_factors(data):
    with pytest.raises(ValueError):
        run(data, factors=[1, 1, 1])
    with pytest.raises(ValueError):
        run(data, factors=[1, 1, 2, 1])


def test_boundary_values_are_valid():
    assert as_degree_control([0, 1, 0.05, 0.5]).tolist() == [0, 1, 0.05, 0.5]
    assert as_degree_control([0, 0, 0, 0]).tolist() == [0, 0, 0, 0]


def test_effective_rate_is_the_product():
    out = apply_degree_control(np.array(RATES), [1, 0, 0.05, 0.5])
    assert out.tolist() == [3.0, 0.0, 0.75 * 0.05, 1.5]


# ------------------------------------------------------------------ equivalence
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_ones_reproduces_the_original_run_exactly(data, strategy):
    assert same(run(data, strategy), run(data, strategy, [1.0] * 4))
    assert same(run(data, strategy), run(data, strategy, [1, 1, 1, 1]))


@pytest.mark.parametrize("strategy", ["ordinary", "frozen_old", "strategy_b"])
@pytest.mark.parametrize("factors", [[1, 0, 1, 1], [1, 1, 0.05, 1], [0.25, 1, 1, 0]])
def test_control_equals_the_product_as_learning_rate(data, strategy, factors):
    """Zero (or 0.05, ...) control == the same effective learning rate, bit for bit."""
    effective = list(np.array(RATES) * np.array(factors))
    assert same(
        run(data, strategy, factors), run(data, strategy, None, rates=effective)
    )


def test_zero_factor_equals_zero_learning_rate_for_ordinary_sgd(data):
    a = run(data, "ordinary", [1, 1, 0, 1])
    b = run(data, "ordinary", None, rates=[3.0, 3.0, 0.0, 3.0])
    assert same(a, b)


# ------------------------------------------------------------------ freezing / scaling
@pytest.mark.parametrize("strategy", ["ordinary", "frozen_old", "strategy_b"])
def test_zero_factor_freezes_that_group_and_others_still_move(data, strategy):
    r = run(data, strategy, [1, 0, 1, 1])
    change = r["degree_diagnostics"]["coef_change_l1"]
    for task_row in change:
        assert task_row[1] == 0.0
    assert any(row[0] > 0 for row in change)
    assert any(row[2] is not None and row[2] > 0 for row in change)


def test_all_zero_factors_train_nothing(data):
    r = run(data, "ordinary", [0, 0, 0, 0])
    assert all(v == 0.0 for v in r["l1_update_by_task"])
    assert all(v == 0 for v in r["n_trainable_by_task"])
    assert r["train_loss_by_task"][0][0] == r["train_loss_by_task"][0][1]
    assert r["effective_degree_learning_rates"] == [0.0] * 4


def test_all_zero_factors_also_work_for_strategy_b_and_frozen(data):
    for strategy in ("strategy_b", "frozen_old"):
        r = run(data, strategy, [0, 0, 0, 0])
        assert all(v == 0.0 for v in r["l1_update_by_task"])


@pytest.mark.parametrize("strategy", ["ordinary", "frozen_old", "strategy_b"])
def test_zero_active_degrees_skip_training_until_expansion(data, strategy):
    # Degrees 0 and 1 are the only active groups in the first experience and both are
    # controlled to zero, although the full control vector is not all zero.
    r = run(data, strategy, [0, 0, 1, 1])

    assert r["l1_update_by_task"][0] == 0.0
    assert r["n_trainable_by_task"][0] == 0
    assert r["train_loss_by_task"][0][0] == r["train_loss_by_task"][0][1]

    # A later experience activates a degree with a positive rate, so training resumes.
    assert r["l1_update_by_task"][1] > 0
    assert r["n_trainable_by_task"][1] > 0


def test_factor_005_scales_one_update_by_5_percent_and_leaves_others_alone():
    rng = np.random.default_rng(1)
    n_features = 969  # degree 3 on 16 components
    group = degree_of_feature(16, MAX_DEGREE)
    x = rng.normal(size=(64, n_features)).astype(np.float32) * 0.1
    y = rng.integers(0, 4, 64)
    empty = (np.empty((0, n_features), np.float32), np.empty(0, np.int64))
    base = np.array(RATES)

    def step(factors):
        w = np.zeros((n_features, 10), np.float32)
        lr, scale = step_scale_for(apply_degree_control(base, factors), group)
        fit_task(
            w,
            x,
            y,
            *empty,
            epochs=1,
            learning_rate=lr,
            batch_size=64,
            seed=0,
            eligible_classes=np.arange(4),
            group_scale=scale,
            group_index=group,
        )
        return w

    ref, ctrl = step([1, 1, 1, 1]), step([1, 1, 0.05, 1])
    g2 = group == 2
    assert np.allclose(ctrl[g2], 0.05 * ref[g2], rtol=1e-5, atol=1e-9)
    assert np.abs(ref[g2]).max() > 0
    assert np.array_equal(ctrl[~g2], ref[~g2])  # uncontrolled groups bit-identical


def test_frozen_old_keeps_its_freezing_and_adds_the_control(data):
    a = run(data, "frozen_old", [1, 1, 1, 1])
    b = run(data, "frozen_old", [1, 0.25, 1, 1])
    assert a["n_trainable_by_task"] == b["n_trainable_by_task"]
    assert a["accuracy_matrix"] != b["accuracy_matrix"]
    da, db = a["degree_diagnostics"], b["degree_diagnostics"]
    assert (
        da["n_trainable"] == db["n_trainable"]
    )  # the control never changes what is frozen
    for task in range(1, 5):  # old rows stay frozen: fewer trainable than coefficients
        assert da["n_trainable"][task][0] < da["n_coefficients"][task][0]
    assert all(row[1] == 3.0 * 0.25 for row in db["lr"])  # effective rate of degree 1
    assert all(row[1] == 3.0 for row in da["lr"])


# ------------------------------------------------------------------ Strategy B
@pytest.mark.parametrize(
    "factors", [[1, 1, 1, 1], [1, 0.05, 1, 1], [0, 1, 0.25, 1], [1, 0, 0.05, 1]]
)
def test_strategy_b_preserves_reference_logits_under_control(data, factors):
    r = run(data, "strategy_b", factors)
    changes = [v for v in r["ref_logit_change_by_task"] if v is not None]
    assert changes and max(changes) < 1e-3


def test_strategy_b_projection_uses_the_controlled_step_scale():
    """sqrt(D) N N^T sqrt(D) g leaves the reference design unchanged for D = H M."""
    rng = np.random.default_rng(2)
    group = degree_of_feature(16, MAX_DEGREE)[:153]  # degree 2
    design = rng.normal(size=(6, 153))
    d = apply_degree_control(np.array(RATES), [1, 0.05, 0.25, 1])[group]
    root = np.sqrt(d)
    basis = null_space_basis(design * root).astype(np.float64)
    g = rng.normal(size=(153, 4))
    step = root[:, None] * (basis @ (basis.T @ (root[:, None] * g)))
    assert np.abs(design @ step).max() < 1e-4
    # a mask applied AFTER the projection of the uncontrolled step breaks preservation
    plain = null_space_basis(design).astype(np.float64)
    masked = np.sqrt(np.array(RATES)[group] * 0 + 1)[:, None] * (plain @ (plain.T @ g))
    masked[group == 1] *= 0.0
    assert np.abs(design @ masked).max() > 1e-3


# ------------------------------------------------------------------ study module
def test_individual_candidates_change_one_degree_at_a_time():
    from polynomial_cl.continual.degree_control import FACTORS, individual_candidates

    cands = individual_candidates()
    assert len(cands) == 1 + 4 * (len(FACTORS) - 1)
    for vec in cands.values():
        assert sum(v != 1.0 for v in vec) <= 1


def test_combinations_are_limited_and_need_two_changed_degrees():
    from polynomial_cl.continual.degree_control import (
        combination_candidates,
        individual_candidates,
        name_of,
    )

    scores = dict.fromkeys(individual_candidates(), 0.0)
    scores["m[0,1,1,1]"] = 0.1
    scores["m[1,1,0.25,1]"] = 0.2
    multi = combination_candidates(scores)
    assert "m[0,1,0.25,1]" in multi
    assert all(sum(v != 1.0 for v in vec) >= 2 for vec in multi.values())
    assert name_of((1, 1, 1, 1)) not in multi
    assert len(multi) < 12


def test_study_requires_disjoint_seeds(data):
    from polynomial_cl.continual.degree_control import run_degree_control_study

    with pytest.raises(ValueError):
        run_degree_control_study(data, data, [0, 1], [1, 2])


# ------------------------------------------------- acquisition constraint and selection
def fake_run(diagonal):
    """A run record with the given accuracy right after each experience (others 0.5)."""
    n = len(diagonal)
    matrix = [[0.5 if j <= i else None for j in range(n)] for i in range(n)]
    for t, v in enumerate(diagonal):
        matrix[t][t] = v
    return {
        "accuracy_matrix": matrix,
        "final_mean_accuracy": 0.8,
        "mean_forgetting": 0.05,
        "final_current_accuracy": diagonal[-1],
    }


def test_immediate_accuracy_is_the_diagonal_and_mean_metrics_uses_seed_means():
    from polynomial_cl.continual.degree_control import (
        immediate_accuracy,
        mean_metrics,
    )

    a, b = fake_run([0.9, 0.2, 0.8]), fake_run([0.7, 0.4, 0.6])
    assert immediate_accuracy(a) == [0.9, 0.2, 0.8]
    m = mean_metrics([a, b])
    assert m["immediate_accuracy_by_experience"] == pytest.approx([0.8, 0.3, 0.7])
    # the constraint sees the worst experience of the seed means, not the worst single run
    assert m["min_mean_immediate_accuracy"] == pytest.approx(0.3)
    assert "min_immediate_accuracy" not in m  # the old, ambiguous name is gone


def test_a_candidate_that_defers_learning_is_infeasible_even_with_the_best_tradeoff():
    from polynomial_cl.continual.degree_control import choose_feasible, is_feasible

    ordinary = {
        "final_mean_accuracy": 0.80,
        "mean_forgetting": 0.12,
        "final_current_accuracy": 0.7,
        "min_mean_immediate_accuracy": 0.7,
    }
    # low forgetting only because experience 2 was never learned on arrival (0.000)
    deferred = {
        "final_mean_accuracy": 0.82,
        "mean_forgetting": 0.03,
        "final_current_accuracy": 0.68,
        "min_mean_immediate_accuracy": 0.0,
    }
    assert not is_feasible(deferred, 0.5) and is_feasible(ordinary, 0.5)
    both = {"m[1,1,1,1]": ordinary, "m[0.05,0,0,1]": deferred}
    assert choose_feasible(both, 0.0) == "m[0.05,0,0,1]"  # trade-off alone prefers it
    assert choose_feasible(both, 0.5) == "m[1,1,1,1]"  # the constraint comes first


def test_nothing_is_selected_when_no_candidate_is_feasible():
    from polynomial_cl.continual.degree_control import choose_feasible

    m = {
        "final_mean_accuracy": 0.8,
        "mean_forgetting": 0.1,
        "final_current_accuracy": 0.7,
        "min_mean_immediate_accuracy": 0.4,
    }
    assert choose_feasible({"a": m, "b": dict(m)}, 0.5) is None  # no least-bad fallback
    assert choose_feasible({}, 0.5) is None
    assert choose_feasible({"a": m}, 0.4) == "a"  # the floor is inclusive


def test_infeasible_single_degree_changes_do_not_enter_combinations():
    from polynomial_cl.continual.degree_control import (
        combination_candidates,
        individual_candidates,
    )

    scores = dict.fromkeys(individual_candidates(), 0.0)
    scores["m[0,1,1,1]"] = 0.1
    scores["m[1,1,0.25,1]"] = 0.2
    both = combination_candidates(scores, examples=())
    assert "m[0,1,0.25,1]" in both
    only_second = combination_candidates(
        scores, examples=(), feasible={"m[1,1,1,1]", "m[1,1,0.25,1]"}
    )
    assert "m[0,1,0.25,1]" not in only_second  # one feasible degree cannot combine


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), True, "0.5"])
def test_study_rejects_an_invalid_acquisition_floor(data, bad):
    from polynomial_cl.continual.degree_control import run_degree_control_study

    with pytest.raises(ValueError, match="min_acquisition_accuracy"):
        run_degree_control_study(
            data, data, [0], [1], min_acquisition_accuracy=bad, strategies=("ordinary",)
        )


@pytest.fixture(scope="module")
def study(data):
    from polynomial_cl.continual.degree_control import run_degree_control_study

    return run_degree_control_study(
        data,
        data,
        [0],
        [1],
        strategies=("ordinary",),
        epochs=1,
        batch_size=64,
        memory_per_class=5,
        n_ref_per_class=2,
        min_acquisition_accuracy=0.0,
    )


def test_study_records_immediate_accuracy_and_the_floor(study):
    cell = study["cells"][0]
    sel = cell["selection"]
    assert study["min_acquisition_accuracy"] == 0.0 and sel["acquisition_floor"] == 0.0
    assert sel["no_feasible_candidate"] == [] and sel["baseline_feasible"]
    assert sel["best_overall"] in sel["validation_metrics"]
    first = next(iter(sel["validation_metrics"].values()))
    assert len(first["immediate_accuracy_by_experience"]) == 5
    row = cell["final"]["baseline"]
    assert len(row["immediate_accuracy_by_experience"]) == 5
    assert "worst_immediate_accuracy" in row["summary"]
    assert "min_mean_immediate_accuracy" not in row["summary"]  # not a per-run metric
    assert 0 <= row["min_mean_immediate_accuracy"] <= 1
    assert "min_immediate_accuracy" not in row["summary"] and (
        "min_immediate_accuracy" not in first
    )


def stub_runs(monkeypatch, immediate):
    """Replace the training loop: ``immediate(factors)`` gives the diagonal of every run."""
    from polynomial_cl.continual import degree_control as dc

    def fake(data, seeds, strategy, rates, factors, *, settings):
        return [
            {
                **dc.with_immediate_accuracy(fake_run(immediate(tuple(factors)))),
                "seed": seed,
                "runtime_seconds": 0.0,
                "effective_degree_learning_rates": [1.0] * 4,
            }
            for seed in seeds
        ]

    monkeypatch.setattr(dc, "run_seeds", fake)
    return dc


def test_study_reports_that_nothing_was_selected_when_no_candidate_is_feasible(
    monkeypatch,
):
    dc = stub_runs(monkeypatch, lambda factors: [0.3] * 5)  # nothing learns on arrival
    result = dc.run_degree_control_study(
        None, None, [0], [1], strategies=("ordinary",), min_acquisition_accuracy=0.5
    )
    sel = result["cells"][0]["selection"]
    assert sel["feasible"] == [] and not sel["baseline_feasible"]
    assert sel["best_individual"] is None and sel["best_overall"] is None
    assert sel["best_multiple"] is None
    assert set(sel["no_feasible_candidate"]) == {
        "best_individual",
        "best_multiple",
        "best_overall",
    }
    final = result["cells"][0]["final"]
    assert final["best_individual"] is None and final["best_overall"] is None
    assert final["baseline"] is not None  # the baseline is always reported


def test_study_never_selects_a_candidate_that_defers_learning(monkeypatch):
    def immediate(factors):
        return [0.9] * 5 if factors == (1.0,) * 4 else [0.9, 0.0, 0.9, 0.9, 0.9]

    dc = stub_runs(monkeypatch, immediate)
    result = dc.run_degree_control_study(
        None, None, [0], [1], strategies=("ordinary",), min_acquisition_accuracy=0.5
    )
    sel = result["cells"][0]["selection"]
    assert sel["feasible"] == ["m[1,1,1,1]"] and sel["baseline_feasible"]
    assert sel["best_individual"] == sel["best_overall"] == "m[1,1,1,1]"
    assert (
        sel["best_multiple"] is None
    )  # no feasible change, so no combination was built
    assert sel["multiple"] == [n for n in sel["multiple"] if n.count("1") < 4]


def test_feasible_pareto_front_never_contains_an_infeasible_candidate(monkeypatch):
    """A candidate that defers learning can dominate on accuracy and forgetting.

    It is nevertheless ineligible. The unrestricted front is allowed to show it; the feasible front must not.
    """
    from polynomial_cl.continual import degree_control as dc

    def deferred(factors):
        # degree 2 switched off: best-looking metrics, but experience 2 is never learned
        return factors[1] == 0.0

    def fake(data, seeds, strategy, rates, factors, *, settings):
        bad = deferred(tuple(factors))
        run = fake_run([0.9, 0.0, 0.9, 0.9, 0.9] if bad else [0.9] * 5)
        run["final_mean_accuracy"] = 0.95 if bad else 0.80
        run["mean_forgetting"] = 0.01 if bad else 0.10
        return [
            {
                **dc.with_immediate_accuracy(run),
                "seed": seed,
                "runtime_seconds": 0.0,
                "effective_degree_learning_rates": [1.0] * 4,
            }
            for seed in seeds
        ]

    monkeypatch.setattr(dc, "run_seeds", fake)
    result = dc.run_degree_control_study(
        None, None, [0], [1], strategies=("ordinary",), min_acquisition_accuracy=0.5
    )
    sel = result["cells"][0]["selection"]
    feasible = set(sel["feasible"])
    infeasible = set(sel["validation_metrics"]) - feasible
    assert infeasible  # the scenario really contains ineligible candidates

    # the unrestricted fronts keep their old meaning: the dominating, infeasible candidates
    assert sel["pareto_unrestricted"]
    assert set(sel["pareto_unrestricted"]) <= infeasible
    assert set(sel["pareto_individual_unrestricted"]) <= infeasible

    # the feasible fronts contain only eligible candidates, and are not empty
    assert sel["pareto_feasible"] and set(sel["pareto_feasible"]) <= feasible
    assert sel["pareto_individual_feasible"]
    assert set(sel["pareto_individual_feasible"]) <= feasible & set(sel["individual"])

    # the selection rule works on the feasible front, so its choices lie on it
    assert sel["best_overall"] in sel["pareto_feasible"]
    assert sel["best_individual"] in sel["pareto_individual_feasible"]
    # the ambiguous names are gone
    assert "pareto_all" not in sel and "pareto_individual" not in sel


def test_feasible_pareto_fronts_are_empty_when_nothing_is_feasible(monkeypatch):
    dc = stub_runs(monkeypatch, lambda factors: [0.3] * 5)  # nothing learns on arrival
    result = dc.run_degree_control_study(
        None, None, [0], [1], strategies=("ordinary",), min_acquisition_accuracy=0.5
    )
    sel = result["cells"][0]["selection"]
    assert sel["pareto_feasible"] == [] and sel["pareto_individual_feasible"] == []
    assert sel["pareto_unrestricted"]  # still reported, for context


def test_seed_mean_minimum_and_mean_of_per_run_minima_are_different_quantities():
    from polynomial_cl.continual.degree_control import (
        is_feasible,
        mean_metrics,
        summarize,
        with_immediate_accuracy,
    )

    # the seeds disagree about which experience is worst
    runs = [fake_run([0.9, 0.2, 0.8]), fake_run([0.2, 0.9, 0.8])]
    base = {"runtime_seconds": 0.0}
    runs = [{**with_immediate_accuracy(r), **base} for r in runs]
    m = mean_metrics(runs)
    assert m["immediate_accuracy_by_experience"] == pytest.approx([0.55, 0.55, 0.8])
    assert m["min_mean_immediate_accuracy"] == pytest.approx(0.55)
    summary = summarize(runs)["worst_immediate_accuracy"]
    assert summary["mean"] == pytest.approx(0.2)  # mean of the per-run minima
    assert summary["mean"] < m["min_mean_immediate_accuracy"]
    # feasibility uses the seed-mean minimum: 0.55 passes a 0.5 floor, 0.2 would not
    assert is_feasible(m, 0.5)
    assert not is_feasible(m, 0.6)
    assert not any("min_mean" in key for key in summarize(runs))


def test_final_rows_report_both_quantities_under_their_own_names(monkeypatch):
    dc = stub_runs(monkeypatch, lambda factors: [0.9, 0.2, 0.8, 0.8, 0.8])
    result = dc.run_degree_control_study(
        None,
        None,
        [0, 1],
        [2, 3],
        strategies=("ordinary",),
        min_acquisition_accuracy=0.1,
    )
    row = result["cells"][0]["final"]["baseline"]
    by_experience = row["immediate_accuracy_by_experience"]
    assert (
        row["min_mean_immediate_accuracy"] == min(by_experience) == pytest.approx(0.2)
    )
    assert row["summary"]["worst_immediate_accuracy"]["mean"] == pytest.approx(0.2)
    assert set(row["paired_vs_baseline"]) >= {"worst_immediate_accuracy"}
    assert "min_mean_immediate_accuracy" not in row["paired_vs_baseline"]


# ------------------------------------------------------------------- seed validation
@pytest.mark.parametrize(
    "select, test, message",
    [
        ([], [1], "select_seeds must not be empty"),
        ([0], [], "test_seeds must not be empty"),
        ([0, 0, 1], [2], r"select_seeds contains duplicate seeds: \[0\]"),
        ([0], [2, 3, 2], r"test_seeds contains duplicate seeds: \[2\]"),
        ([0, 1], [1, 2], r"overlap on \[1\]"),
        ([0, 1, 2], [2, 1, 5], r"overlap on \[1, 2\]"),
        ([0.5], [1], "non-negative integers"),
        ([0], [True], "non-negative integers"),
        ([-1], [1], "non-negative integers"),
        (["0"], [1], "non-negative integers"),
        ("01", [1], "list of integer seeds"),
        (3, [1], "list of integer seeds"),
    ],
)
def test_invalid_seed_lists_are_rejected_with_a_clear_message(select, test, message):
    from polynomial_cl.continual.degree_control import (
        run_degree_control_study,
        validate_seed_lists,
    )

    with pytest.raises(ValueError, match=message):
        validate_seed_lists(select, test)
    # the study validates before it touches any data or trains anything
    with pytest.raises(ValueError, match=message):
        run_degree_control_study(None, None, select, test)


def test_valid_seed_lists_are_returned_as_plain_integer_lists():
    from polynomial_cl.continual.degree_control import validate_seed_lists

    select, test = validate_seed_lists((0, np.int64(1)), range(5, 8))
    assert select == [0, 1] and test == [5, 6, 7]
    assert all(type(x) is int for x in select + test)


@pytest.mark.parametrize("strategies", [(), ("no_such_strategy",), ("ordinary", "b")])
def test_study_rejects_empty_or_unknown_strategies_before_running(strategies):
    from polynomial_cl.continual.degree_control import run_degree_control_study

    with pytest.raises(ValueError, match="strategies must be a non-empty subset"):
        run_degree_control_study(None, None, [0], [1], strategies=strategies)


# ------------------------------------------------------------ example vectors in B
def test_example_vectors_do_not_depend_on_their_single_degree_parts():
    from polynomial_cl.continual.degree_control import (
        EXAMPLE_VECTORS,
        combination_candidates,
        individual_candidates,
        name_of,
    )

    expected = {name_of(v) for v in EXAMPLE_VECTORS if sum(x != 1 for x in v) >= 2}
    assert expected  # the examples are multi-degree vectors
    scores = dict.fromkeys(individual_candidates(), 0.0)  # nothing beats the baseline
    assert set(combination_candidates(scores)) == expected
    # and none of the single-degree candidates is feasible either
    assert set(combination_candidates(scores, feasible=set())) == expected
    assert combination_candidates(scores, examples=(), feasible=set()) == {}


def test_example_vectors_are_evaluated_but_selected_only_on_their_own_metrics(
    monkeypatch,
):
    # every change is infeasible, so no combination can be built from singles; the example
    # vectors are still measured, and each is judged on its own immediate accuracy
    def immediate(factors):
        if factors == (1.0,) * 4:
            return [0.9] * 5
        return [0.9] * 5 if factors == (1, 0.05, 0.1, 1) else [0.9, 0.0, 0.9, 0.9, 0.9]

    dc = stub_runs(monkeypatch, immediate)
    result = dc.run_degree_control_study(
        None, None, [0], [1], strategies=("ordinary",), min_acquisition_accuracy=0.5
    )
    sel = result["cells"][0]["selection"]
    examples = {dc.name_of(v) for v in dc.EXAMPLE_VECTORS}
    assert set(sel["multiple"]) == examples  # no combination of feasible singles exists
    assert "m[1,0.05,0.1,1]" in sel["feasible"]  # a feasible example is eligible
    assert not {"m[1,0,0.05,1]", "m[0,1,0.25,1]"} & set(
        sel["feasible"]
    )  # others are not


# ------------------------------------------------------ feature_power compatibility
def test_identity_control_is_accepted_with_feature_power_and_changes_nothing(data):
    plain = run(data, feature_power=1.0, rates=0.5)
    identity = run(data, feature_power=1.0, rates=0.5, factors=[1, 1, 1, 1])
    assert same(plain, identity)
    assert identity["degree_control_factors"] == [1.0] * 4


@pytest.mark.parametrize("factors", [[0.5, 1, 1, 1], [1, 1, 1, 0], [0.9] * 4])
def test_a_control_that_changes_something_cannot_be_combined_with_feature_power(
    data, factors
):
    with pytest.raises(ValueError, match="feature_power"):
        run(data, feature_power=1.0, rates=0.5, factors=factors)


def test_a_rate_vector_cannot_be_combined_with_feature_power_even_with_identity_control(
    data,
):
    with pytest.raises(ValueError, match="feature_power"):
        run(data, feature_power=1.0, rates=RATES, factors=[1, 1, 1, 1])


# ------------------------------------------------------------- protocol guarantees
def test_selection_does_not_depend_on_the_test_data(data, study):
    from polynomial_cl.continual.degree_control import run_degree_control_study

    rng = np.random.default_rng(99)
    other_test_x = rng.normal(size=data.test_x.shape).astype(data.test_x.dtype)
    other_test = data._replace(test_x=other_test_x)
    changed = run_degree_control_study(
        data,
        other_test,
        [0],
        [1],
        strategies=("ordinary",),
        epochs=1,
        batch_size=64,
        memory_per_class=5,
        n_ref_per_class=2,
        min_acquisition_accuracy=0.0,
    )
    assert changed["cells"][0]["selection"] == study["cells"][0]["selection"]
    # the test data is used, though: only the final comparison changes
    assert changed["cells"][0]["final"] != study["cells"][0]["final"]


def test_final_comparison_pairs_every_row_with_the_baseline_on_the_same_test_seeds(
    monkeypatch,
):
    dc = stub_runs(monkeypatch, lambda factors: [0.9] * 5)
    result = dc.run_degree_control_study(
        None,
        None,
        [0],
        [4, 2, 7],
        strategies=("ordinary",),
        min_acquisition_accuracy=0.5,
    )
    final = result["cells"][0]["final"]
    for label, row in final.items():
        assert row["seeds"] == [4, 2, 7], label  # the test seeds, in the given order
        for metric, diff in row["paired_vs_baseline"].items():
            assert diff["n"] == 3 and len(diff["by_seed"]) == 3, (label, metric)
    # a row that is the baseline vector is paired with itself: zero difference
    assert final["baseline"]["paired_vs_baseline"]["final_mean_accuracy"][
        "by_seed"
    ] == [
        0.0,
        0.0,
        0.0,
    ]


def test_paired_difference_refuses_runs_from_different_seeds():
    from polynomial_cl.continual.degree_control import paired_difference

    a = [
        {"seed": 1, "final_mean_accuracy": 0.5},
        {"seed": 2, "final_mean_accuracy": 0.6},
    ]
    b = [
        {"seed": 2, "final_mean_accuracy": 0.5},
        {"seed": 1, "final_mean_accuracy": 0.6},
    ]
    with pytest.raises(ValueError, match="identical seeds"):
        paired_difference(a, b, "final_mean_accuracy")
    d = paired_difference(a, a, "final_mean_accuracy")
    assert d["mean_diff"] == 0.0 and d["n"] == 2
