# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polynomial_cl.binary.data import make_experiences
from polynomial_cl.binary.experiment import make_config, run_continual
from polynomial_cl.models.binary import PolynomialLogisticModel
from polynomial_cl.plotting import synthetic as P


def test_plot_functions_run_and_do_not_change_model():
    exps = make_experiences(n_train=60, n_test=80, seed=0)
    res = run_continual(
        exps,
        make_config(
            "ordinary", epochs=20, m_current=40, m_replay=40, capacity=40, stop_after=3
        ),
    )
    rec = res.records[-1]
    model = rec.model_after
    before = model.w.copy()
    samples = (exps[0].test.x, exps[0].test.y)

    P.plot_ground_truth(samples=samples, res=40)
    P.plot_predictions(model, samples=samples, res=40)
    P.plot_correctness(model, samples=samples, res=40)
    P.plot_prediction_changes(rec.model_before, rec.model_after, res=40)
    P.plot_step(rec, exps, res=40)
    P.plot_coefficients(rec)
    P.plot_coefficients(model=model)
    P.plot_loss_curve(rec.loss_history)
    P.plot_learning_history(res)
    P.plot_accuracy_matrix(res)
    assert "FROZEN" in P.format_coefficient_table(
        rec
    ) or "trainable" in P.format_coefficient_table(rec)
    plt.close("all")
    assert np.array_equal(model.w, before)


def test_correctness_grid_categories_with_known_models():
    zero = PolynomialLogisticModel(1)  # score 0 everywhere -> predicts class 0
    _, _, cat = P.correctness_grid(zero, 50)
    assert set(np.unique(cat)) <= {0, 3}  # only TN or FN possible
    all_one = PolynomialLogisticModel(1, [1.0, 0.0, 0.0])
    _, _, cat = P.correctness_grid(all_one, 50)
    assert set(np.unique(cat)) <= {1, 2}  # only TP or FP possible


def test_default_truth_is_the_sinusoid_curve_and_other_targets_are_contours():
    from polynomial_cl.data.synthetic import boundary_curve, circular_target

    ax = P.plot_ground_truth()
    (curve,) = ax.get_lines()
    np.testing.assert_allclose(curve.get_ydata(), boundary_curve(curve.get_xdata()))
    plt.close(ax.figure)

    ax = P.plot_ground_truth(target=circular_target, title="circle")
    assert not ax.get_lines()  # no sinusoid curve is drawn for the circle
    assert len(ax.collections) >= 2  # filled regions + the zero-level contour
    plt.close(ax.figure)


def test_plots_accept_a_target_and_the_degree_comparison_runs():
    from polynomial_cl.data.synthetic import circular_target, make_synthetic_dataset
    from polynomial_cl.training.binary import train

    x, y = make_synthetic_dataset(circular_target, 200, seed=0)
    models = [
        train(PolynomialLogisticModel(d), x, y, lr=0.5, epochs=20).model for d in (1, 2)
    ]
    fig = P.plot_degree_comparison(
        models, samples=(x, y), res=40, target=circular_target
    )
    assert len(fig.axes) == 2 and fig.axes[1].get_title() == "degree 2"
    plt.close(fig)
    ax = P.plot_correctness(models[1], res=40, samples=(x, y), target=circular_target)
    ax2 = P.plot_prediction_changes(
        models[0], models[1], res=40, target=circular_target
    )
    plt.close(ax.figure)
    plt.close(ax2.figure)


def test_accuracy_vs_degree_plots_one_line_per_budget():
    rows = [
        {"degree": d, "epochs": e, "mean_accuracy": 0.5 + 0.1 * d + e / 1000}
        for d in (1, 2, 3)
        for e in (100, 200)
    ]
    ax = P.plot_accuracy_vs_degree(rows)
    assert len(ax.get_lines()) == 2 and list(ax.get_lines()[0].get_xdata()) == [1, 2, 3]
    plt.close(ax.figure)


def test_accuracy_vs_budget_has_one_line_per_degree_and_loss_curves_can_overlay():
    rows = [
        {"degree": d, "epochs": e, "mean_accuracy": 0.6 + 0.05 * d + e / 5000}
        for d in (1, 2)
        for e in (10, 100, 1000)
    ]
    ax = P.plot_accuracy_vs_budget(rows)
    assert len(ax.get_lines()) == 2 and ax.get_xscale() == "log"
    plt.close(ax.figure)
    ax = P.plot_loss_curve([1.0, 0.5], label="degree 1")
    P.plot_loss_curve([1.0, 0.2], ax=ax, label="degree 2")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "degree 1",
        "degree 2",
    ]
    plt.close(ax.figure)
