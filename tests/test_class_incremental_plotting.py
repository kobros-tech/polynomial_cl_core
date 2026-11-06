# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from polynomial_cl.continual import (
    make_feature_data,
    make_task_plan,
    run_condition,
)
from polynomial_cl.data import make_synthetic_2d
from polynomial_cl.plotting import continual as P


def test_class_incremental_plots_run_and_do_not_change_the_runs():
    import copy

    x, y, tx, ty, _ = make_synthetic_2d(
        1, n_train_per_class=10, n_test_per_class=5, separation=3.0, noise=0.2
    )
    data = make_feature_data(x, y, tx, ty)
    tasks, memory = make_task_plan(data.train_y, 0, 3)
    runs = {
        policy: [
            run_condition(
                "expanding",
                data,
                tasks,
                memory,
                seed=s,
                epochs=1,
                batch_size=8,
                learning_rate=[0.1, 0.2, 0.05, 0.1],
                sampling=policy,
            )
            for s in (0, 1)
        ]
        for policy in ("shuffled", "class_balanced")
    }
    snapshot = copy.deepcopy(runs)
    P.plot_cil_accuracy_over_experiences(runs)
    P.plot_cil_forgetting(runs)
    P.plot_cil_takeover_flips(runs)
    P.plot_cil_tradeoff(
        [{"label": k, "runs": v, "sampling": k} for k, v in runs.items()]
    )
    P.plot_cil_accuracy_matrix(runs["shuffled"])
    for key in ("coef_change_l1", "mean_step_l1", "logit_contribution", "logit_change"):
        P.plot_degree_diagnostic(runs["shuffled"], key)
    vectors = {"shared": [0.1] * 4, "d1x4": [0.1, 0.4, 0.1, 0.1]}
    P.plot_rate_vectors_with_results(
        vectors, dict.fromkeys(vectors, runs["shuffled"]), "shared"
    )
    P.plot_sampling_counts(
        {k: [r["sampling_by_task"][0] for r in v][:1] for k, v in runs.items()}
    )
    plt.close("all")
    assert runs == snapshot


def test_per_class_accuracy_plot_leaves_unseen_classes_empty_and_does_not_change_runs():
    import copy

    import numpy as np

    x, y, tx, ty, _ = make_synthetic_2d(0, n_train_per_class=20, n_test_per_class=10)
    data = make_feature_data(x, y, tx, ty)
    runs = [
        run_condition("expanding", data, *make_task_plan(y, s, 5), seed=s, epochs=1)
        for s in (0, 1)
    ]
    before = copy.deepcopy(runs)
    ax = P.plot_cil_per_class_accuracy(
        {"expanding": runs}, task=1
    )  # after 2 of 5 experiences
    heights = [bar.get_height() for bar in ax.patches]
    assert (
        len(heights) == 10
        and np.isfinite(heights[:4]).all()
        and np.isnan(heights[4:]).all()
    )
    assert runs == before
    plt.close(ax.figure)
