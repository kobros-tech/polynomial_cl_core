# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np

from polynomial_cl.binary import make_experiences, run_joint_study
from polynomial_cl.continual import (
    make_feature_data,
    run_joint_comparison,
    run_joint_reference,
)
from polynomial_cl.data import circular_target, make_synthetic_2d


def test_joint_study_pools_training_data_and_reports_every_degree_and_budget():
    exps = make_experiences(
        n_experiences=4, n_train=40, n_test=40, target_function=circular_target
    )
    rows = run_joint_study(exps, degrees=(1, 2), epochs_list=(20, 40), lr=0.5)
    assert [(r["degree"], r["epochs"]) for r in rows] == [
        (1, 20),
        (1, 40),
        (2, 20),
        (2, 40),
    ]
    for r in rows:
        assert len(r["accuracy_by_experience"]) == 4
        assert r["mean_accuracy"] == np.mean(r["accuracy_by_experience"])
        assert len(r["loss_history"]) == r["epochs"] + 1
    assert (
        rows[1]["final_loss"] < rows[0]["final_loss"]
    )  # a longer budget keeps descending


def test_joint_study_is_reproducible():
    exps = make_experiences(n_experiences=3, n_train=30, n_test=30)
    a = run_joint_study(exps, (2,), (30,), lr=0.5, seed=1)
    b = run_joint_study(exps, (2,), (30,), lr=0.5, seed=1)
    assert a[0]["mean_accuracy"] == b[0]["mean_accuracy"]
    np.testing.assert_array_equal(a[0]["loss_history"], b[0]["loss_history"])


def test_ten_class_joint_comparison_matches_single_reference_runs_and_has_no_forgetting():
    data = make_feature_data(
        *make_synthetic_2d(0, n_train_per_class=20, n_test_per_class=10)[:4]
    )
    rows = run_joint_comparison(data, degrees=(1,), epochs_list=(2,), seeds=(0, 1))
    singles = [
        run_joint_reference(data, degree=1, seed=s, epochs=2)["test_accuracy"]
        for s in (0, 1)
    ]
    assert rows[0]["test_accuracy_by_seed"] == singles
    assert rows[0]["test_accuracy_mean"] == np.mean(singles)
    assert not any("forget" in key for key in rows[0])
    assert (
        run_joint_comparison(data, (1,), (1,), seeds=(0,))[0]["test_accuracy_std"]
        is None
    )
