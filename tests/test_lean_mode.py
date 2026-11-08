# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Execution efficiency must never change what is learned.

Two speed-ups are guarded here against the behaviour they replaced:

* the factor-table construction of ``scaled_polynomial_features`` must reproduce the direct
  ``(chunk, features, variables)`` formula bit for bit and must not allocate that array;
* ``run_condition(..., diagnostics=False)`` skips only observation-only work, so everything
  that describes the learning run is identical to a ``diagnostics=True`` run.
"""

import json
import math
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

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
from polynomial_cl.continual.experiment import (
    DEGREE_DIAGNOSTICS,
    INDICATORS,
)
from polynomial_cl.continual.features import scaled_polynomial_features
from polynomial_cl.data import make_synthetic_2d
from polynomial_cl.features import monomial_exponents

ROOT = Path(__file__).parents[1]


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


# ------------------------------------------------------------------ lean run mode
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
    "per_degree_rates",
    "sampling",
    "sampling_by_task",
)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_lean_run_learns_exactly_what_the_full_run_learns(
    tiny_data, condition, strategy
):
    tasks, memory = make_task_plan(tiny_data.train_y, 3, 2)
    full = run_condition(
        condition, tiny_data, tasks, memory, seed=3, strategy=strategy, **KWARGS
    )
    lean = run_condition(
        condition,
        tiny_data,
        tasks,
        memory,
        seed=3,
        strategy=strategy,
        diagnostics=False,
        **KWARGS,
    )
    for field in LEARNING_FIELDS:
        assert lean[field] == full[field], field


@pytest.mark.parametrize("sampling", ["shuffled", "class_balanced"])
def test_lean_run_keeps_the_sampling_accounting(tiny_data, sampling):
    tasks, memory = make_task_plan(tiny_data.train_y, 4, 2)
    kwargs = {"seed": 4, "sampling": sampling, **KWARGS}
    full = run_condition("expanding", tiny_data, tasks, memory, **kwargs)
    lean = run_condition(
        "expanding", tiny_data, tasks, memory, diagnostics=False, **kwargs
    )
    assert lean["sampling_by_task"] == full["sampling_by_task"]
    assert lean["accuracy_matrix"] == full["accuracy_matrix"]


def test_lean_run_works_with_feature_power_and_per_degree_rates(tiny_data):
    tasks, memory = make_task_plan(tiny_data.train_y, 5, 2)
    for extra in ({"feature_power": 1.0}, {"learning_rate": [0.05, 0.02, 0.01, 0.01]}):
        kwargs = {**KWARGS, "seed": 5, **extra}
        full = run_condition("expanding", tiny_data, tasks, memory, **kwargs)
        lean = run_condition(
            "expanding", tiny_data, tasks, memory, diagnostics=False, **kwargs
        )
        for field in LEARNING_FIELDS:
            assert lean[field] == full[field], (field, extra)


def test_lean_record_marks_itself_and_leaves_every_diagnostic_empty(tiny_data):
    tasks, memory = make_task_plan(tiny_data.train_y, 3, 2)
    lean = run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, diagnostics=False, **KWARGS
    )
    full = run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS)
    assert lean["diagnostics"] is False and full["diagnostics"] is True
    assert set(lean) == set(full)  # same schema, only the values are missing
    for key in INDICATORS:
        assert lean[key] == [None] * 5, key
    for key in DEGREE_DIAGNOSTICS:
        assert lean["degree_diagnostics"][key] == [[None] * 4] * 5, key
    for key in (
        "mean_old_logit_change",
        "mean_old_flip_fraction",
        "mean_old_takeover_fraction",
        "mean_loss_reduction",
    ):
        assert lean[key] is None and full[key] is not None, key
    json.dumps(lean, allow_nan=False)  # still a plain JSON record


def test_diagnostics_are_on_by_default_and_the_default_record_is_unchanged(tiny_data):
    tasks, memory = make_task_plan(tiny_data.train_y, 3, 2)
    default = run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS)
    explicit = run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, diagnostics=True, **KWARGS
    )
    assert default == explicit
    assert default["diagnostics"] is True
    assert all(v is not None for v in default["train_loss_by_task"])


def test_lean_run_calls_no_diagnostic_helper(tiny_data, monkeypatch):
    from polynomial_cl.continual import experiment as module

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic work ran in lean mode")

    for name in ("_degree_diagnostics", "_record_indicators", "cross_entropy"):
        monkeypatch.setattr(module, name, forbidden)
    tasks, memory = make_task_plan(tiny_data.train_y, 3, 2)
    run = run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, diagnostics=False, **KWARGS
    )
    assert run["diagnostics"] is False


def test_lean_run_does_not_ask_the_trainer_for_group_statistics(tiny_data, monkeypatch):
    from polynomial_cl.continual import experiment as module

    seen = []
    real = module.fit_task

    def spy(*args, **kwargs):
        seen.append(kwargs.get("group_index"))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "fit_task", spy)
    tasks, memory = make_task_plan(tiny_data.train_y, 3, 2)
    run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, diagnostics=False, **KWARGS
    )
    assert seen and all(g is None for g in seen)
    seen.clear()
    run_condition("expanding", tiny_data, tasks, memory, seed=3, **KWARGS)
    assert seen and all(g is not None for g in seen)


def test_comparison_summary_is_identical_in_lean_mode(tiny_data):
    kwargs = {
        "epochs": 1,
        "learning_rate": 0.05,
        "batch_size": 8,
        "memory_per_class": 2,
    }
    full = run_comparison(tiny_data, [0, 1], **kwargs)
    lean = run_comparison(tiny_data, [0, 1], diagnostics=False, **kwargs)
    assert lean["summary"] == full["summary"]
    assert all(r["diagnostics"] is False for r in lean["runs"])


# ------------------------------------------------------------------ timings
def test_timings_split_the_run_and_do_not_change_the_record(tiny_data):
    tasks, memory = make_task_plan(tiny_data.train_y, 3, 2)
    timings = {}
    timed = run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, timings=timings, **KWARGS
    )
    assert timed == run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, **KWARGS
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


def test_lean_timings_report_exactly_zero_diagnostics(tiny_data):
    tasks, memory = make_task_plan(tiny_data.train_y, 3, 2)
    timings = {}
    run_condition(
        "expanding",
        tiny_data,
        tasks,
        memory,
        seed=3,
        diagnostics=False,
        timings=timings,
        **KWARGS,
    )
    assert timings["diagnostics_seconds"] == 0.0
    assert timings["training_seconds"] > 0 and timings["evaluation_seconds"] > 0


# ------------------------------------------------------------------ helpers refuse lean runs
def test_diagnostic_helpers_explain_why_a_lean_run_cannot_be_used(tiny_data):
    tasks, memory = make_task_plan(tiny_data.train_y, 3, 2)
    lean = run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, diagnostics=False, **KWARGS
    )
    for helper in (old_class_rows, strategy_diagnostic_rows, degree_group_rows):
        with pytest.raises(ValueError, match="diagnostics=False"):
            helper(lean)
    with pytest.raises(ValueError, match="my_report needs"):
        require_diagnostics(lean, "my_report")


def test_plots_of_diagnostics_refuse_a_lean_run(tiny_data):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from polynomial_cl.plotting.continual import plot_cil_takeover_flips

    tasks, memory = make_task_plan(tiny_data.train_y, 3, 2)
    lean = run_condition(
        "expanding", tiny_data, tasks, memory, seed=3, diagnostics=False, **KWARGS
    )
    with pytest.raises(ValueError, match="diagnostics=False"):
        plot_cil_takeover_flips({"lean": [lean]})


def test_records_written_before_the_flag_existed_count_as_full_records():
    require_diagnostics({"final_mean_accuracy": 0.5})  # no "diagnostics" key: no error


# ------------------------------------------------------------------ command line
def _run_cli(*args, check=True):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "class_incremental.py"),
            *map(str, args),
        ],
        capture_output=True,
        text=True,
        check=check,
    )


COMMON = [
    "--dataset", "synthetic_2d", "--seeds", 0, "--epochs", 1,
    "--memory-per-class", 2, "--train-per-class", 6, "--test-per-class", 4,
]  # fmt: skip


def test_cli_no_diagnostics_gives_the_same_summary_with_lean_runs(tmp_path):
    full_out, lean_out = tmp_path / "full.json", tmp_path / "lean.json"
    _run_cli(*COMMON, "--output", full_out)
    _run_cli(*COMMON, "--output", lean_out, "--no-diagnostics")
    full, lean = (json.loads(p.read_text()) for p in (full_out, lean_out))
    assert lean["summary"] == full["summary"]
    assert all(r["diagnostics"] is False for r in lean["runs"])
    assert all(r["diagnostics"] is True for r in full["runs"])


def test_cli_rejects_no_diagnostics_for_the_strategy_study(tmp_path):
    result = _run_cli(
        *COMMON,
        "--experiment", "strategies",
        "--no-diagnostics",
        "--output", tmp_path / "x.json",
        check=False,
    )  # fmt: skip
    assert result.returncode != 0
    assert "--no-diagnostics applies to --experiment conditions only" in result.stderr
