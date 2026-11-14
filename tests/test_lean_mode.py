# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Execution efficiency must never change what is learned.

Guarded here against the behaviour it replaced:

* the factor-table construction of ``scaled_polynomial_features`` must reproduce the direct
  ``(chunk, features, variables)`` formula bit for bit and must not allocate that array;
* diagnostics are opt-in everywhere (API and command line) and are pure observation: a run with
  ``diagnostics=True`` learns, expands and evaluates exactly like the default run, and the
  default run does none of the observation work and records none of its fields;
* timings are reported only when asked for.
"""

import inspect
import json
import math
import tracemalloc

import numpy as np
import pytest
from script_runner import run_script

from polynomial_cl import continual
from polynomial_cl.continual import (
    CONDITIONS,
    STRATEGIES,
    degree_group_rows,
    make_feature_data,
    make_task_plan,
    old_class_rows,
    require_diagnostics,
    run_comparison,
    run_condition,
    strategy_diagnostic_rows,
)
from polynomial_cl.continual.degree_study import study_rows, vector_summary
from polynomial_cl.continual.experiment import INDICATORS
from polynomial_cl.continual.features import scaled_polynomial_features
from polynomial_cl.data import make_synthetic_2d
from polynomial_cl.features import monomial_exponents


# The previous implementation, kept verbatim as the oracle for bit-for-bit equivalence.
def reference_features(x, degree, chunk_size=1024):
    x = np.asarray(x, dtype=np.float32)
    exponents = monomial_exponents(x.shape[1], degree)
    degree_of_feature = exponents.sum(axis=1)
    scales = np.ones(len(exponents), dtype=np.float32)
    for total_degree in range(1, degree + 1):
        group_size = math.comb(x.shape[1] + total_degree - 1, total_degree)
        scales[degree_of_feature == total_degree] = 1.0 / math.sqrt(group_size)
    out = np.empty((len(x), len(exponents)), dtype=np.float32)
    for start in range(0, len(x), chunk_size):
        block = x[start : start + chunk_size]
        powers = np.power(block[:, None, :], exponents[None, :, :])
        out[start : start + len(block)] = np.prod(powers, axis=2) * scales
    return out


# ------------------------------------------------------------------ feature construction
@pytest.mark.parametrize("n_variables", [1, 2, 5, 16])
@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_features_are_bit_identical_to_the_direct_formula(n_variables, degree):
    rng = np.random.default_rng(10 * n_variables + degree)
    x = rng.normal(size=(41, n_variables)).astype(np.float32)
    x[0] = 0.0  # zeros and sign changes exercise power(0, e) and odd powers
    x[1] = -np.abs(x[1])
    x[2] *= 50.0
    x[3] *= 1e-3
    expected = reference_features(x, degree)
    assert np.array_equal(
        scaled_polynomial_features(x, degree), expected
    )  # default chunk
    for chunk_size in (1, 7, 4096):
        got = scaled_polynomial_features(x, degree, chunk_size=chunk_size)
        assert got.dtype == np.float32
        assert np.array_equal(got, expected), (n_variables, degree, chunk_size)


@pytest.mark.parametrize("degree", [1, 3])
def test_features_reproduce_the_power_loop_on_many_rows(degree):
    # numpy's float32 power loop returns x ** 1 that differs from x by one unit in the last
    # place for a few values in thousands, so "x ** 1 is x" is not a safe shortcut: the
    # published results were produced with the power loop and must stay reproducible.
    x = np.random.default_rng(7).normal(size=(20000, 16)).astype(np.float32)
    got = scaled_polynomial_features(x, degree)
    assert np.array_equal(got, reference_features(x, degree))


@pytest.mark.parametrize("n_variables, degree", [(1, 3), (2, 3), (3, 4), (16, 3)])
def test_features_match_an_independent_float64_evaluation(n_variables, degree):
    # The oracle above is the previous implementation; this one shares no code with either:
    # explicit monomials in float64, scaled by the size of their degree group.
    x = np.random.default_rng(n_variables).uniform(-1.5, 1.5, (23, n_variables))
    x32 = x.astype(np.float32)
    got = scaled_polynomial_features(x32, degree)
    exponents = monomial_exponents(n_variables, degree)
    assert got.shape == (23, math.comb(n_variables + degree, degree))
    assert got.shape[1] == len(exponents) and got.dtype == np.float32
    total = exponents.sum(axis=1)
    assert np.all(np.diff(total) >= 0)  # constant, then degree 1, 2, ...
    expected = np.prod(x32.astype(np.float64)[:, None, :] ** exponents[None], axis=2)
    for d in range(degree + 1):
        group = total == d
        assert group.sum() == math.comb(n_variables + d - 1, d)  # monomials of degree d
        scale = 1.0 / math.sqrt(group.sum()) if d else 1.0
        np.testing.assert_allclose(
            got[:, group], expected[:, group] * scale, rtol=2e-6, atol=2e-7
        )
    assert np.all(got[:, 0] == 1.0)  # the constant feature is unscaled


@pytest.mark.parametrize("rows", [1, 3, 4, 5, 8, 9, 12])
def test_chunk_boundaries_do_not_change_the_result(rows):
    x = np.random.default_rng(rows).normal(size=(rows, 5)).astype(np.float32)
    whole = scaled_polynomial_features(x, 3, chunk_size=rows + 1)
    for chunk_size in (1, 2, 4):  # rows below, at and across a multiple of the chunk
        chunked = scaled_polynomial_features(x, 3, chunk_size=chunk_size)
        assert np.array_equal(chunked, whole), (rows, chunk_size)


@pytest.mark.parametrize(
    "bad",
    [
        {"x": np.zeros(5), "degree": 3},  # not two-dimensional
        {"x": np.zeros((3, 0)), "degree": 3},  # no variables
        {"x": np.zeros((3, 2)), "degree": -1},
        {"x": np.zeros((3, 2)), "degree": 3, "chunk_size": 0},
    ],
)
def test_invalid_feature_arguments_are_rejected(bad):
    with pytest.raises(ValueError):
        scaled_polynomial_features(**bad)


def test_features_of_no_rows_is_an_empty_matrix_of_the_right_width():
    out = scaled_polynomial_features(np.empty((0, 4), np.float32), 3)
    assert out.shape == (0, math.comb(4 + 3, 3))


def test_features_do_not_allocate_the_three_dimensional_intermediate():
    # The direct formula needs rows x 969 x 16 float32 values (about 64 MB for 1024 rows);
    # the factor table needs a few times the 4 MB output.
    x = np.random.default_rng(0).normal(size=(1024, 16)).astype(np.float32)
    scaled_polynomial_features(
        x[:8], 3
    )  # warm the cached layout outside the measurement
    tracemalloc.start()
    try:
        scaled_polynomial_features(x, 3, chunk_size=1024)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 24 * 2**20, f"peak {peak / 2**20:.1f} MiB"


def test_the_cached_layout_is_read_only_so_callers_cannot_corrupt_it():
    from polynomial_cl.continual.features import _monomial_plan

    for array in _monomial_plan(4, 3):
        assert not array.flags.writeable


# ------------------------------------------------------------------ opt-in diagnostics
def _functions_with_a_diagnostics_parameter():
    modules = (continual, continual.degree_study, continual.experiment)
    found = {}
    for module in modules:
        for name, function in vars(module).items():
            if inspect.isfunction(function):
                if "diagnostics" in inspect.signature(function).parameters:
                    found[name] = function
    return found


def test_diagnostics_default_to_off_in_every_function_that_takes_them():
    found = _functions_with_a_diagnostics_parameter()
    # Every entry point that can run a polynomial experiment must be here, so that a new
    # one cannot quietly default the other way.
    expected = {
        "run_condition",
        "run_comparison",
        "run_strategy_comparison",
        "run_sampling_comparison",
        "select_rate_vectors",
        "evaluate_vectors",
        "run_epoch_robustness",
        "_settings",
    }
    assert expected <= set(found), expected - set(found)
    for name, function in found.items():
        default = inspect.signature(function).parameters["diagnostics"].default
        assert default is False, f"{name} defaults diagnostics to {default!r}"


def test_the_study_runner_passes_diagnostics_on_and_defaults_to_off():
    # run_per_degree_study forwards **settings to the selection and test runs; its own
    # signature has no explicit parameter, so the default comes from those functions.
    parameters = inspect.signature(continual.run_per_degree_study).parameters
    assert "diagnostics" not in parameters and "settings" in parameters


@pytest.fixture(scope="module")
def tiny_data():
    x_train, y_train, x_test, y_test, _ = make_synthetic_2d(
        9, n_train_per_class=6, n_test_per_class=5, separation=3.0, noise=0.2
    )
    return make_feature_data(x_train, y_train, x_test, y_test)


KWARGS = {"epochs": 2, "learning_rate": 0.05, "batch_size": 8}

# Everything that describes the learning run itself; none of it may depend on diagnostics.
LEARNING_FIELDS = (
    "condition",
    "strategy",
    "seed",
    "active_degree_by_task",
    "feature_count_by_task",
    "expansion_max_logit_changes",
    "accuracy_matrix",
    "accuracy_by_class_matrix",
    "final_mean_accuracy",
    "final_current_accuracy",
    "forgetting_by_task",
    "mean_forgetting",
    "current_accuracy_by_task",
    "mean_old_accuracy_by_task",
    "mean_accuracy_by_task",
    "learning_rate",
    "degree_learning_rates",
    "degree_control_factors",
    "effective_degree_learning_rates",
    "per_degree_rates",
    "sampling",
    "sampling_by_task",
)
# The fields that exist only because a run was observed.
OBSERVATION_FIELDS = {
    "degree_diagnostics",
    *INDICATORS,
    "mean_old_logit_change",
    "mean_old_flip_fraction",
    "mean_old_takeover_fraction",
    "mean_loss_reduction",
}


def _plan(data, seed=3, memory=2):
    return make_task_plan(data.train_y, seed, memory)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_diagnostics_do_not_change_what_is_learned(tiny_data, condition, strategy):
    tasks, memory = _plan(tiny_data)
    kwargs = {"seed": 3, "strategy": strategy, **KWARGS}
    default = run_condition(condition, tiny_data, tasks, memory, **kwargs)
    observed = run_condition(
        condition, tiny_data, tasks, memory, diagnostics=True, **kwargs
    )
    for field in LEARNING_FIELDS:
        assert observed[field] == default[field], field


@pytest.mark.parametrize("sampling", ["shuffled", "class_balanced"])
def test_diagnostics_do_not_change_the_sampling_accounting(tiny_data, sampling):
    tasks, memory = _plan(tiny_data, 4)
    kwargs = {"seed": 4, "sampling": sampling, **KWARGS}
    default = run_condition("expanding", tiny_data, tasks, memory, **kwargs)
    observed = run_condition(
        "expanding", tiny_data, tasks, memory, diagnostics=True, **kwargs
    )
    assert observed["sampling_by_task"] == default["sampling_by_task"]
    assert observed["accuracy_matrix"] == default["accuracy_matrix"]


@pytest.mark.parametrize(
    "extra", [{"feature_power": 1.0}, {"learning_rate": [0.05, 0.02, 0.01, 0.01]}]
)
def test_diagnostics_do_not_change_feature_power_or_per_degree_runs(tiny_data, extra):
    tasks, memory = _plan(tiny_data, 5)
    kwargs = {**KWARGS, "seed": 5, **extra}
    default = run_condition("expanding", tiny_data, tasks, memory, **kwargs)
    observed = run_condition(
        "expanding", tiny_data, tasks, memory, diagnostics=True, **kwargs
    )
    for field in LEARNING_FIELDS:
        assert observed[field] == default[field], field


def test_the_default_record_has_no_observation_fields(tiny_data):
    tasks, memory = _plan(tiny_data)
    default = run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS)
    observed = run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, diagnostics=True, **KWARGS
    )
    assert default["diagnostics"] is False and observed["diagnostics"] is True
    assert not OBSERVATION_FIELDS & set(default)
    assert OBSERVATION_FIELDS <= set(observed)
    # Nothing but the observation fields (and the flag's value) differs between the two.
    assert set(observed) - set(default) == OBSERVATION_FIELDS
    assert all(v is not None for v in observed["train_loss_by_task"])
    json.dumps(default, allow_nan=False)  # plain JSON, no None placeholders to filter


def test_reading_an_unmeasured_field_fails_loudly(tiny_data):
    tasks, memory = _plan(tiny_data)
    default = run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS)
    with pytest.raises(KeyError):
        default["mean_old_flip_fraction"]


def test_the_default_run_calls_no_diagnostic_helper(tiny_data, monkeypatch):
    from polynomial_cl.continual import experiment as module

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic work ran without diagnostics=True")

    for name in ("_degree_diagnostics", "_record_indicators", "cross_entropy"):
        monkeypatch.setattr(module, name, forbidden)
    tasks, memory = _plan(tiny_data)
    run = run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS)
    assert run["diagnostics"] is False


class _CountingNumpy:
    """Stand-in for the ``np`` of the experiment module that counts ``concatenate``."""

    def __init__(self):
        self.concatenations = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def concatenate(self, *args, **kwargs):
        self.concatenations += 1
        return np.concatenate(*args, **kwargs)


@pytest.mark.parametrize(
    "extra, stacks",
    [
        ({}, False),  # the stacked training set is not needed
        ({"diagnostics": True}, True),  # the loss passes read it
        ({"feature_power": 1.0}, True),  # the feature rescaling reads it
    ],
)
def test_the_stacked_training_set_is_built_only_when_something_reads_it(
    tiny_data, monkeypatch, extra, stacks
):
    from polynomial_cl.continual import experiment as module

    counting = _CountingNumpy()
    monkeypatch.setattr(module, "np", counting)
    tasks, memory = _plan(tiny_data)
    run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS, **extra)
    assert (counting.concatenations > 0) == stacks


def test_the_default_run_does_not_ask_the_trainer_for_group_statistics(
    tiny_data, monkeypatch
):
    from polynomial_cl.continual import experiment as module

    seen = []
    real = module.fit_task

    def spy(*args, **kwargs):
        seen.append(kwargs.get("group_index"))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "fit_task", spy)
    tasks, memory = _plan(tiny_data)
    run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS)
    assert seen and all(g is None for g in seen)
    seen.clear()
    run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, diagnostics=True, **KWARGS
    )
    assert seen and all(g is not None for g in seen)


def test_diagnostics_can_be_enabled_through_every_comparison_function(tiny_data):
    kwargs = {"epochs": 1, "learning_rate": 0.05, "batch_size": 8}
    conditions = continual.run_comparison(
        tiny_data, [0], memory_per_class=2, diagnostics=True, **kwargs
    )
    strategies = continual.run_strategy_comparison(
        tiny_data, [0], memory_per_class=2, diagnostics=True, **kwargs
    )
    assert all(r["diagnostics"] is True for r in conditions["runs"])
    assert all(r["diagnostics"] is True for r in strategies["runs"])
    for flag in (False, True):  # the default (off) and the explicit opt-in
        options = {"diagnostics": flag} if flag else {}
        sampling = continual.run_sampling_comparison(
            tiny_data,
            [0],
            strategies=("ordinary",),
            memory_per_class=2,
            **options,
            **kwargs,
        )
        runs = [r for p in sampling.values() for r in p["runs"]]
        assert runs and all(r["diagnostics"] is flag for r in runs)


def test_comparison_summary_is_identical_with_and_without_diagnostics(tiny_data):
    kwargs = {"epochs": 1, "learning_rate": 0.05, "batch_size": 8}
    kwargs["memory_per_class"] = 2
    default = run_comparison(tiny_data, [0, 1], **kwargs)
    observed = run_comparison(tiny_data, [0, 1], diagnostics=True, **kwargs)
    assert observed["summary"] == default["summary"]
    assert all(r["diagnostics"] is False for r in default["runs"])


def test_strategy_summary_covers_the_diagnostic_metrics_only_when_recorded(tiny_data):
    from polynomial_cl.continual.experiment import STRATEGY_METRICS, SUMMARY_METRICS

    kwargs = {"epochs": 1, "learning_rate": 0.05, "batch_size": 8}
    kwargs["memory_per_class"] = 2
    default = continual.run_strategy_comparison(tiny_data, [0, 1], **kwargs)
    observed = continual.run_strategy_comparison(
        tiny_data, [0, 1], diagnostics=True, **kwargs
    )
    for stats in default["summary"].values():
        assert tuple(stats) == SUMMARY_METRICS  # no NaN-filled diagnostic columns
    for stats in observed["summary"].values():
        assert tuple(stats) == STRATEGY_METRICS
    for name, stats in default["summary"].items():
        for metric in SUMMARY_METRICS:
            assert observed["summary"][name][metric] == stats[metric]


# ------------------------------------------------------------------ degree studies
STUDY = {
    "strategies": ("ordinary",),
    "samplings": ("shuffled",),
    "epochs": 1,
    "batch_size": 8,
    "memory_per_class": 2,
    "n_ref_per_class": 1,
}


@pytest.fixture(scope="module")
def studies(tiny_data):
    run = continual.run_per_degree_study
    return (
        run(tiny_data, tiny_data, [0], [1, 2], **STUDY),
        run(tiny_data, tiny_data, [0], [1, 2], diagnostics=True, **STUDY),
    )


def test_a_study_selects_and_reports_the_same_with_and_without_diagnostics(studies):
    default, observed = studies
    (a,), (b,) = default["cells"], observed["cells"]
    selection = ("vectors", "validation_metrics", "validation_scores", "pareto")
    for key in (*selection, "selected", "base_rate", "shared_grid"):
        assert a[key] == b[key], key
    for name in a["runs"]:
        for lean, full in zip(a["runs"][name], b["runs"][name]):
            for field in LEARNING_FIELDS:
                assert lean[field] == full[field], (name, field)


def test_study_runs_record_diagnostics_only_when_asked(studies):
    default, observed = studies
    for study, expected in ((default, False), (observed, True)):
        runs = [r for c in study["cells"] for rs in c["runs"].values() for r in rs]
        assert runs and all(r["diagnostics"] is expected for r in runs)


def test_study_tables_work_for_both_kinds_of_run(studies):
    default, observed = studies
    lean_rows, full_rows = study_rows(default), study_rows(observed)
    assert lean_rows and "takeover" not in lean_rows[0] and "flips" not in lean_rows[0]
    assert "takeover" in full_rows[0] and "flips" in full_rows[0]
    for lean, full in zip(lean_rows, full_rows):  # shared columns agree
        assert {k: full[k] for k in lean} == lean


def test_vector_summary_follows_what_the_runs_recorded(studies):
    from polynomial_cl.continual.degree_study import TRADEOFF_METRICS
    from polynomial_cl.continual.experiment import SUMMARY_METRICS

    default, observed = studies
    lean_runs = default["cells"][0]["runs"]["shared"]
    full_runs = observed["cells"][0]["runs"]["shared"]
    assert set(vector_summary(lean_runs)) == set(SUMMARY_METRICS)
    assert set(vector_summary(full_runs)) == set(TRADEOFF_METRICS)


def test_epoch_robustness_passes_diagnostics_through(tiny_data, studies):
    default, _ = studies
    settings = {
        "batch_size": STUDY["batch_size"],
        "memory_per_class": STUDY["memory_per_class"],
        "n_ref_per_class": STUDY["n_ref_per_class"],
    }
    for flag in (False, True):
        again = continual.run_epoch_robustness(
            tiny_data, default, 1, test_seeds=[1], diagnostics=flag, **settings
        )
        runs = [r for c in again["cells"] for rs in c["runs"].values() for r in rs]
        assert runs and all(r["diagnostics"] is flag for r in runs)


# ------------------------------------------------------------------ timings
def test_timings_are_not_part_of_a_record_unless_requested(tiny_data):
    tasks, memory = _plan(tiny_data)
    record = run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS)
    assert not [k for k in record if "second" in k or "timing" in k]
    timings = {}
    timed = run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, timings=timings, **KWARGS
    )
    assert timed == record  # asking for timings leaves the record untouched


def test_timings_split_the_run(tiny_data):
    tasks, memory = _plan(tiny_data)
    timings = {}
    run_condition(
        "expanding",
        tiny_data,
        tasks,
        memory,
        seed=3,
        diagnostics=True,
        timings=timings,
        **KWARGS,
    )
    expected = {
        "training_seconds",
        "evaluation_seconds",
        "diagnostics_seconds",
        "other_seconds",
        "total_seconds",
    }
    assert set(timings) == expected
    assert all(v >= 0 for v in timings.values())
    assert timings["training_seconds"] > 0 and timings["diagnostics_seconds"] > 0
    parts = sum(v for k, v in timings.items() if k != "total_seconds")
    assert parts == pytest.approx(timings["total_seconds"])


def test_timings_attribute_time_to_the_operation_they_name(tiny_data, monkeypatch):
    # Slow down one operation at a time: only its own bucket may take the delay.
    import time

    from polynomial_cl.continual import experiment as module

    delay = 0.05

    def slowed(name):
        real = getattr(module, name)

        def wrapper(*args, **kwargs):
            time.sleep(delay)
            return real(*args, **kwargs)

        return wrapper

    tasks, memory = _plan(tiny_data)
    for name, bucket in (
        ("fit_task", "training_seconds"),
        ("accuracy_by_task", "evaluation_seconds"),
        ("_record_indicators", "diagnostics_seconds"),
    ):
        monkeypatch.setattr(module, name, slowed(name))
        timings = {}
        run_condition(
            "expanding",
            tiny_data,
            tasks,
            memory,
            seed=3,
            diagnostics=True,
            timings=timings,
            **KWARGS,
        )
        monkeypatch.undo()
        assert timings[bucket] >= 5 * delay * 0.9, (name, timings)
        others = {"training_seconds", "evaluation_seconds", "diagnostics_seconds"}
        for other in others - {bucket}:
            assert timings[other] < 5 * delay * 0.9, (name, other, timings)


def _delayed_total(tiny_data, monkeypatch, name, delay=0.1, **options):
    """Run once with ``name`` of the experiment module slowed down; return the timings."""
    import time

    from polynomial_cl.continual import experiment as module

    real = getattr(module, name)

    def slowed(*args, **kwargs):
        time.sleep(delay)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, slowed)
    tasks, memory = _plan(tiny_data)
    timings = {}
    run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, timings=timings, **KWARGS,
        **options,
    )  # fmt: skip
    monkeypatch.undo()
    return timings


@pytest.mark.parametrize(
    "name",
    [
        "as_degree_rates",  # first work after the argument validation
        "degree_of_feature",  # run initialisation, before the experience loop
        "forgetting_from_matrix",  # after the loop, while the record is built
    ],
)
def test_total_covers_the_whole_call_and_the_delay_lands_in_other(
    tiny_data, monkeypatch, name
):
    delay = 0.1
    timings = _delayed_total(tiny_data, monkeypatch, name, delay)
    assert timings["total_seconds"] >= delay * 0.9, (name, timings)
    assert timings["other_seconds"] >= delay * 0.9, (name, timings)
    for bucket in ("training_seconds", "evaluation_seconds", "diagnostics_seconds"):
        assert timings[bucket] < delay * 0.9, (name, bucket, timings)
    parts = sum(v for k, v in timings.items() if k != "total_seconds")
    assert parts == pytest.approx(timings["total_seconds"])


class _SlowConcatenate(_CountingNumpy):
    """Counting stand-in whose ``concatenate`` also takes ``delay`` seconds."""

    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def concatenate(self, *args, **kwargs):
        import time

        time.sleep(self.delay)
        return super().concatenate(*args, **kwargs)


@pytest.mark.parametrize(
    "options, bucket, other",
    [
        # stacked only for the loss passes: its cost is part of the diagnostics
        ({"diagnostics": True}, "diagnostics_seconds", "other_seconds"),
        # stacked for the feature rescaling: ordinary work, not diagnostics
        ({"feature_power": 1.0}, "other_seconds", "diagnostics_seconds"),
    ],
)
def test_the_stacked_training_set_is_timed_as_what_needs_it(
    tiny_data, monkeypatch, options, bucket, other
):
    from polynomial_cl.continual import experiment as module

    slow = _SlowConcatenate(0.05)
    monkeypatch.setattr(module, "np", slow)
    tasks, memory = _plan(tiny_data)
    timings = {}
    run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, timings=timings, **KWARGS,
        **options,
    )  # fmt: skip
    cost = slow.concatenations * slow.delay
    assert slow.concatenations > 0
    assert timings[bucket] >= cost * 0.9, timings
    assert timings[other] < cost * 0.5, timings


def test_the_default_run_reports_exactly_zero_diagnostics_time(tiny_data):
    tasks, memory = _plan(tiny_data)
    timings = {}
    run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, timings=timings, **KWARGS
    )
    assert timings["diagnostics_seconds"] == 0.0
    assert timings["training_seconds"] > 0 and timings["evaluation_seconds"] > 0


# ------------------------------------------------------------------ helpers refuse runs without diagnostics
def test_diagnostic_helpers_explain_why_a_default_run_cannot_be_used(tiny_data):
    tasks, memory = _plan(tiny_data)
    default = run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS)
    for helper in (old_class_rows, strategy_diagnostic_rows, degree_group_rows):
        with pytest.raises(ValueError, match="diagnostics=True"):
            helper(default)
    with pytest.raises(ValueError, match="my_report needs"):
        require_diagnostics(default, "my_report")


def test_plots_of_diagnostics_refuse_a_default_run(tiny_data):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from polynomial_cl.plotting.continual import plot_cil_takeover_flips

    tasks, memory = _plan(tiny_data)
    default = run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS)
    with pytest.raises(ValueError, match="diagnostics=True"):
        plot_cil_takeover_flips({"default": [default]})


def test_records_written_before_the_flag_existed_count_as_full_records():
    require_diagnostics({"final_mean_accuracy": 0.5})  # no "diagnostics" key: no error


# ------------------------------------------------------------------ command line
COMMON = [
    "--dataset", "synthetic_2d", "--seeds", 0, "--epochs", 1,
    "--memory-per-class", 2, "--train-per-class", 6, "--test-per-class", 4,
]  # fmt: skip


def _read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("experiment", ["conditions", "strategies"])
def test_cli_is_lean_by_default_and_observes_only_with_the_flag(tmp_path, experiment):
    default, observed = tmp_path / "default.json", tmp_path / "observed.json"
    run_script(
        "class_incremental.py", *COMMON, "--experiment", experiment,
        "--output", default,
    )  # fmt: skip
    run_script(
        "class_incremental.py", *COMMON, "--experiment", experiment,
        "--diagnostics", "--output", observed,
    )  # fmt: skip
    lean, full = _read(default), _read(observed)
    assert lean["protocol"]["diagnostics"] is False
    assert full["protocol"]["diagnostics"] is True
    assert all(r["diagnostics"] is False for r in lean["runs"])
    assert all(r["diagnostics"] is True for r in full["runs"])
    assert not any(OBSERVATION_FIELDS & set(r) for r in lean["runs"])
    assert all(OBSERVATION_FIELDS <= set(r) for r in full["runs"])
    for lean_run, full_run in zip(lean["runs"], full["runs"]):
        for field in LEARNING_FIELDS:
            assert lean_run[field] == full_run[field], field


def test_cli_summary_of_the_conditions_does_not_depend_on_the_flag(tmp_path):
    default, observed = tmp_path / "default.json", tmp_path / "observed.json"
    run_script("class_incremental.py", *COMMON, "--output", default)
    run_script("class_incremental.py", *COMMON, "--diagnostics", "--output", observed)
    assert _read(default)["summary"] == _read(observed)["summary"]


def test_cli_strategy_summary_has_the_diagnostic_metrics_only_with_the_flag(tmp_path):
    from polynomial_cl.continual.experiment import STRATEGY_METRICS, SUMMARY_METRICS

    default, observed = tmp_path / "default.json", tmp_path / "observed.json"
    args = (*COMMON, "--experiment", "strategies")
    run_script("class_incremental.py", *args, "--output", default)
    run_script("class_incremental.py", *args, "--diagnostics", "--output", observed)
    for stats in _read(default)["summary"].values():
        assert tuple(stats) == SUMMARY_METRICS
    for stats in _read(observed)["summary"].values():
        assert tuple(stats) == STRATEGY_METRICS


def test_the_old_negative_flag_no_longer_exists(tmp_path):
    result = run_script(
        "class_incremental.py", *COMMON, "--no-diagnostics",
        "--output", tmp_path / "x.json", check=False,
    )  # fmt: skip
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr


STUDY_ARGS = (
    "--dataset", "synthetic_2d", "--train-per-class", 12, "--test-per-class", 6,
    "--memory-per-class", 4, "--n-ref-per-class", 1, "--batch-size", 16,
    "--strategies", "ordinary", "--samplings", "shuffled",
    "--validation-fraction", 0.5, "--epochs", 1,
)  # fmt: skip


def _study_runs(study):
    return [r for c in study["cells"] for rs in c["runs"].values() for r in rs]


def test_per_degree_cli_is_lean_by_default_and_observes_only_with_the_flag(tmp_path):
    default, observed = tmp_path / "default.json", tmp_path / "observed.json"
    args = (*STUDY_ARGS, "--experiment", "per-degree", "--select-seeds", 0)
    args += ("--test-seeds", 1)
    run_script("per_degree_lr.py", *args, "--output", default)
    run_script("per_degree_lr.py", *args, "--diagnostics", "--output", observed)
    lean, full = _read(default), _read(observed)
    assert lean["protocol"]["diagnostics"] is False
    assert full["protocol"]["diagnostics"] is True
    assert all(r["diagnostics"] is False for r in _study_runs(lean))
    assert all(r["diagnostics"] is True for r in _study_runs(full))
    assert lean["cells"][0]["selected"] == full["cells"][0]["selected"]


def test_per_degree_cli_passes_the_flag_to_the_epoch_and_sampling_experiments(tmp_path):
    cells = tmp_path / "cells"
    cells.mkdir()
    run_script(
        "per_degree_lr.py", *STUDY_ARGS, "--experiment", "per-degree",
        "--select-seeds", 0, "--test-seeds", 1, "--output", cells / "cell_a.json",
    )  # fmt: skip
    for flag in ((), ("--diagnostics",)):
        for experiment in ("epochs", "sampling"):
            out = tmp_path / f"{experiment}{len(flag)}.json"
            extra = ("--cells-dir", cells) if experiment == "epochs" else ("--seeds", 0)
            run_script(
                "per_degree_lr.py", *STUDY_ARGS, "--experiment", experiment,
                *extra, *flag, "--output", out,
            )  # fmt: skip
            payload = _read(out)
            runs = (
                _study_runs(payload)
                if experiment == "epochs"
                else [
                    r
                    for p in payload.values()
                    if isinstance(p, dict) and "runs" in p
                    for r in p["runs"]
                ]
            )
            assert runs and all(r["diagnostics"] is bool(flag) for r in runs)
