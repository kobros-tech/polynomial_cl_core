# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.continual import (
    make_feature_data,
    old_class_rows,
    run_comparison,
    run_sampling_comparison,
    run_strategy_comparison,
    sampling_rows,
    strategy_diagnostic_rows,
    summary_rows,
)
from polynomial_cl.data import make_synthetic_2d


@pytest.fixture(scope="module")
def data():
    return make_feature_data(
        *make_synthetic_2d(0, n_train_per_class=40, n_test_per_class=20)[:4]
    )


def test_summary_rows_format_mean_and_std_for_every_condition(data):
    result = run_comparison(data, [0, 1], epochs=1, memory_per_class=10)
    rows = summary_rows(result)
    assert [r["condition"] for r in rows] == list(result["summary"])
    stats = result["summary"]["expanding"]["final_mean_accuracy"]
    assert next(r for r in rows if r["condition"] == "expanding")[
        "final_mean_accuracy"
    ] == (f"{stats['mean']:.3f}+/-{stats['std']:.3f}")
    only = summary_rows(result, key="method", metric_names=("mean_forgetting",))
    assert set(only[0]) == {"method", "mean_forgetting"}


def test_old_class_and_strategy_rows_cover_the_right_experiences(data):
    result = run_strategy_comparison(
        data, [0], epochs=1, memory_per_class=10, diagnostics=True
    )
    run = next(r for r in result["runs"] if r["strategy"] == "strategy_b_new")
    assert [r["after_experience"] for r in old_class_rows(run)] == [2, 3, 4, 5]
    rows = strategy_diagnostic_rows(run)
    assert [r["experience"] for r in rows] == [1, 2, 3, 4, 5]
    assert (
        rows[0]["null_dim"] == "-"
        and rows[-1]["n_trainable"] == run["n_trainable_by_task"][-1]
    )
    ordinary = next(r for r in result["runs"] if r["strategy"] == "ordinary")
    assert all(r["null_dim"] == "-" for r in strategy_diagnostic_rows(ordinary))


def test_sampling_rows_report_processed_examples_and_repetition(data):
    sampling = run_sampling_comparison(
        data, [0], learning_rate=0.1, epochs=2, batch_size=32
    )
    rows = sampling_rows(sampling)
    assert {r["sampling"] for r in rows} == {"shuffled", "class_balanced"}
    shuffled = next(
        r for r in rows if r["sampling"] == "shuffled" and r["strategy"] == "ordinary"
    )
    assert (
        shuffled["examples_processed"] == shuffled["examples_available"]
    )  # one pass per epoch
    assert float(shuffled["draws_per_example_min"]) == pytest.approx(2.0)  # epochs = 2
    balanced = next(
        r
        for r in rows
        if r["sampling"] == "class_balanced" and r["strategy"] == "ordinary"
    )
    assert balanced["updates"] == shuffled["updates"]
    assert np.isfinite(float(balanced["draws_per_example_max"]))


def test_per_seed_rows_list_every_run_with_its_seed(data):
    result = run_comparison(data, [3, 4], epochs=1, memory_per_class=10)
    runs = {
        c: [r for r in result["runs"] if r["condition"] == c] for c in ("expanding",)
    }
    from polynomial_cl.continual import per_seed_rows

    rows = per_seed_rows(runs, key="condition")
    assert [(r["condition"], r["seed"]) for r in rows] == [
        ("expanding", 3),
        ("expanding", 4),
    ]
    assert (
        rows[0]["final_mean_accuracy"]
        == f"{runs['expanding'][0]['final_mean_accuracy']:.3f}"
    )


def test_study_rows_and_pairs_cover_shared_and_selected_vectors(data):
    from polynomial_cl.continual import run_per_degree_study
    from polynomial_cl.continual.degree_study import study_pairs, study_rows

    study = run_per_degree_study(
        data,
        data,
        [0],
        [1],
        strategies=("ordinary",),
        samplings=("shuffled",),
        epochs=1,
        batch_size=64,
        memory_per_class=10,
    )
    rows = study_rows(study)
    cell = study["cells"][0]
    assert rows[0]["vector"].rstrip("*^") == "shared"
    assert any(
        r["vector"].startswith(cell["selected"]) and "*" in r["vector"] for r in rows
    ) or (cell["selected"] == "shared")
    pairs = study_pairs(study)
    assert (
        "ordinary/shuff shared" in pairs
        and pairs["ordinary/shuff shared"][1] == "shuffled"
    )
