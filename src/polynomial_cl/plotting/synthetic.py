# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Plots for the 2-D binary experiments. All functions take a model (never modify it) and return the Axes/Figure.

The true boundary is the sinusoid unless a ``target`` function (e.g. ``circular_target``) is given.

Conventions
-----------
* Regions (filled background, soft colours) come from a dense grid evaluated with either the
  ground-truth rule or the model.  Samples (dots with a white edge) are drawn on top and are
  never used to colour regions, so samples and regions cannot be confused.
* All input-space plots share the same limits (DOMAIN) so that boundary changes are comparable.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ..data.synthetic import DOMAIN, boundary_curve, sinusoidal_target, true_label
from ..metrics import format_table

C0, C1 = "#6baed6", "#fdae6b"  # class 0 / class 1 region colours
CLASS_CMAP = ListedColormap([C0, C1])
# correctness categories: 0=TN, 1=TP, 2=FP, 3=FN
CORRECT_COLORS = ["#9ecae1", "#fdd0a2", "#d62728", "#7b3294"]
CORRECT_LABELS = [
    "correct: class 0 (TN)",
    "correct: class 1 (TP)",
    "false positive (pred 1, true 0)",
    "false negative (pred 0, true 1)",
]


def make_grid(res=200):
    g = np.linspace(*DOMAIN, res)
    xx, yy = np.meshgrid(g, g)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def _ax(ax, size=(4.2, 4.0)):
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    ax.set_xlim(*DOMAIN)
    ax.set_ylim(*DOMAIN)
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return ax


def _truth_line(ax, target=None, **kw):
    """True decision boundary: the sinusoid curve by default, else the zero level of ``target``."""
    style = {"color": "k", "lw": 1.2, "ls": "--", **kw}
    if target is None or target is sinusoidal_target:
        xs = np.linspace(*DOMAIN, 400)
        ax.plot(xs, boundary_curve(xs), **style)
    else:
        xx, yy, pts = make_grid(400)
        ax.contour(
            xx,
            yy,
            np.asarray(target(pts)).reshape(xx.shape),
            levels=[0],
            colors=style["color"],
            linewidths=style["lw"],
            linestyles=style["ls"],
        )


def plot_samples(ax, x, y, correct=None, s=14, alpha=0.9):
    """Dots coloured by TRUE label; if `correct` given, misclassified samples get a black 'x'."""
    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=np.where(np.asarray(y) == 1, "#e6550d", "#3182bd"),
        s=s,
        edgecolors="white",
        linewidths=0.4,
        alpha=alpha,
        zorder=3,
    )
    if correct is not None:
        bad = ~np.asarray(correct, bool)
        ax.scatter(
            x[bad, 0], x[bad, 1], marker="x", c="k", s=s * 2.5, linewidths=1.0, zorder=4
        )
    return ax


# ------------------------------------------------------------ A. ground truth
def plot_ground_truth(
    ax=None, res=200, samples=None, title="Ground truth", target=None
):
    xx, yy, pts = make_grid(res)
    ax = _ax(ax)
    ax.contourf(
        xx,
        yy,
        true_label(pts, target or sinusoidal_target).reshape(xx.shape),
        levels=[-0.5, 0.5, 1.5],
        cmap=CLASS_CMAP,
        alpha=0.55,
    )
    _truth_line(ax, target)
    if samples is not None:
        plot_samples(ax, *samples)
    ax.set_title(title)
    return ax


# ------------------------------------------------------------ B. predictions
def plot_predictions(
    model, ax=None, res=200, samples=None, title=None, show_truth=True, target=None
):
    xx, yy, pts = make_grid(res)
    ax = _ax(ax)
    s = model.score(pts).reshape(xx.shape)
    ax.contourf(
        xx,
        yy,
        (s > 0).astype(int),
        levels=[-0.5, 0.5, 1.5],
        cmap=CLASS_CMAP,
        alpha=0.55,
    )
    if (s > 0).any() and (s < 0).any():
        ax.contour(xx, yy, s, levels=[0], colors="k", linewidths=1.8)  # f(x1,x2)=0
    if show_truth:
        _truth_line(ax, target, color="0.35", lw=1.0)
    if samples is not None:
        plot_samples(ax, *samples)
    ax.set_title(title or f"Predicted regions (degree {model.degree})")
    return ax


# ------------------------------------------------------------ C. correctness
def correctness_grid(model, res=200, target=None):
    xx, yy, pts = make_grid(res)
    t, p = true_label(pts, target or sinusoidal_target), model.predict(pts)
    cat = np.where(t == p, p, np.where(p == 1, 2, 3))  # TN=0, TP=1, FP=2, FN=3
    return xx, yy, cat.reshape(xx.shape)


def plot_correctness(
    model,
    ax=None,
    res=200,
    samples=None,
    title="Correct vs incorrect regions",
    legend=True,
    target=None,
):
    xx, yy, cat = correctness_grid(model, res, target)
    ax = _ax(ax)
    ax.contourf(
        xx,
        yy,
        cat,
        levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
        colors=CORRECT_COLORS,
        alpha=0.85,
    )
    _truth_line(ax, target)
    if samples is not None:  # samples = (x, y); misclassified ones are crossed
        x, y = samples
        plot_samples(ax, x, y, correct=model.predict(x) == y)
    if legend:
        ax.legend(
            handles=[
                Patch(color=c, label=lab)
                for c, lab in zip(CORRECT_COLORS, CORRECT_LABELS)
            ],
            fontsize=6,
            loc="lower left",
            framealpha=0.9,
        )
    ax.set_title(title)
    return ax


def plot_prediction_changes(
    model_a, model_b, ax=None, res=200, title="Changed predictions", target=None
):
    xx, yy, pts = make_grid(res)
    a, b = model_a.predict(pts), model_b.predict(pts)
    cat = np.where(a == b, 0, np.where(b == 1, 1, 2)).reshape(
        xx.shape
    )  # 0 same, 1: 0->1, 2: 1->0
    ax = _ax(ax)
    ax.contourf(
        xx,
        yy,
        cat,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=["#f0f0f0", "#31a354", "#de2d26"],
    )
    _truth_line(ax, target)
    ax.set_title(f"{title} ({np.mean(a != b):.1%} of area)")
    ax.legend(
        handles=[
            Patch(color="#31a354", label="0 -> 1"),
            Patch(color="#de2d26", label="1 -> 0"),
        ],
        fontsize=6,
        loc="lower left",
        framealpha=0.9,
    )
    return ax


# ------------------------------------------------------------ D. before / after
def plot_step(record, experiences=None, res=160, figsize=(21, 4.2), target=None):
    """before expansion | right after expansion | after training | ground truth | prediction changes.

    ``target`` is the function that generated the labels (default: the sinusoid); it is drawn in
    every panel, so pass the one the experiences were made with, e.g. ``circular_target``.
    """
    fig, axs = plt.subplots(1, 5, figsize=figsize)
    cur = experiences[record.t].test if experiences is not None else None
    samples = (cur.x, cur.y) if cur is not None else None
    plot_predictions(
        record.model_before,
        axs[0],
        res,
        title=f"Exp {record.t + 1}: before expansion (deg {record.degree_before})",
        target=target,
    )
    plot_predictions(
        record.model_expanded,
        axs[1],
        res,
        title=f"after expansion (deg {record.degree_after}), untrained",
        target=target,
    )
    plot_predictions(
        record.model_after,
        axs[2],
        res,
        samples=samples,
        title="after training (current test samples)",
        target=target,
    )
    plot_ground_truth(axs[3], res, target=target)
    plot_prediction_changes(
        record.model_before,
        record.model_after,
        axs[4],
        res,
        "before -> after training",
        target=target,
    )
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ E. coefficients
def coefficient_table(record):
    rows = []
    for k, name in enumerate(record.feature_names):
        prev = record.model_before.w[k] if k < record.model_before.n_features else 0.0
        rows.append(
            {
                "feature": name,
                "previous": float(prev),
                "new": float(record.w_after[k]),
                "abs_change": float(abs(record.w_after[k] - record.w_expanded[k])),
                "lr": float(record.lr_vector[k]),
                "status": "trainable" if record.trainable[k] else "FROZEN",
                "introduced": "new" if k >= record.model_before.n_features else "old",
            }
        )
    return rows


def format_coefficient_table(record):
    return format_table(coefficient_table(record), "{:.4f}")


def plot_coefficients(record=None, ax=None, model=None, title=None):
    """Bar plot: previous vs new coefficient per feature (hatched = frozen). With `model=...`
    only the model's final coefficients are drawn."""
    if ax is None:
        _, ax = plt.subplots(
            figsize=(
                max(
                    6,
                    0.35 * (len(record.feature_names) if record else model.n_features),
                ),
                3.5,
            )
        )
    if model is not None:
        names, vals = model.feature_names, model.w
        ax.bar(range(len(vals)), vals, color="#3182bd")
    else:
        rows = coefficient_table(record)
        names = [r["feature"] for r in rows]
        idx = np.arange(len(rows))
        ax.bar(
            idx - 0.2,
            [r["previous"] for r in rows],
            0.4,
            label="previous",
            color="#bdbdbd",
        )
        bars = ax.bar(
            idx + 0.2, [r["new"] for r in rows], 0.4, label="new", color="#3182bd"
        )
        for b, r in zip(bars, rows):
            if r["status"] == "FROZEN":
                b.set_hatch("//")
        ax.legend(fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("coefficient")
    ax.set_title(
        title
        or (
            f"Coefficients after experience {record.t + 1}"
            if record
            else "Coefficients"
        )
    )
    return ax


def plot_loss_curve(loss_history, ax=None, title="Training loss", label=None):
    """Loss against epoch; pass the same ``ax`` and a ``label`` per call to overlay curves."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(loss_history, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE")
    ax.set_title(title)
    if label is not None:
        ax.legend(fontsize=7)
    return ax


# ------------------------------------------------------------ F. learning history
def plot_learning_history(result, figsize=(16, 8)):
    recs = result.records
    T = np.arange(1, len(recs) + 1)
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    ax = axs[0, 0]
    for r in recs:
        ax.plot(
            np.arange(len(r.loss_history)) / max(len(r.loss_history) - 1, 1) + r.t,
            r.loss_history,
        )
    ax.set_title("Training loss (each experience: one unit on x)")
    ax.set_xlabel("experience")
    ax.set_ylabel("BCE")
    ax = axs[0, 1]
    A = result.acc_matrix[: len(recs), : len(recs)]
    for i in range(len(recs)):
        ax.plot(T[i:], A[i:, i], marker="o", ms=3, label=f"E{i + 1}")
    ax.set_title("Accuracy of each experience over time")
    ax.set_xlabel("after experience")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=6, ncol=2)
    ax = axs[0, 2]
    ax.plot(T, [r.acc_current for r in recs], marker="o", label="current experience")
    ax.plot(T, [r.mean_acc for r in recs], marker="s", label="mean over observed")
    ax.set_title("Current vs mean accuracy")
    ax.set_xlabel("experience")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=7)
    ax = axs[1, 0]
    ax.plot(T, [r.mean_forgetting for r in recs], marker="o", label="mean")
    ax.plot(T, [r.max_forgetting for r in recs], marker="s", label="max")
    ax.set_title("Forgetting (best-so-far minus current accuracy)")
    ax.set_xlabel("experience")
    ax.legend(fontsize=7)
    ax = axs[1, 1]
    T2, R2 = T[1:], recs[1:]  # experience 1 has no "old" function
    ax.semilogy(
        T2,
        [max(r.max_old_score_change, 1e-18) for r in R2],
        marker="o",
        label="max |Δ score| (old test sets)",
    )
    ref = [r.ref_score_change if r.ref_score_change is not None else np.nan for r in R2]
    ax.semilogy(
        T2,
        [max(v, 1e-18) if not np.isnan(v) else np.nan for v in ref],
        marker="x",
        label="max |Δ score| (reference inputs)",
    )
    ax.set_title("Change in old scores")
    ax.set_xlabel("experience")
    ax.legend(fontsize=7)
    ax = axs[1, 2]
    ax.plot(
        T, [r.n_trainable for r in recs], marker="o", label="trainable coefficients"
    )
    if any(r.null_dim is not None for r in recs):
        ax.plot(
            T,
            [r.null_dim if r.null_dim is not None else np.nan for r in recs],
            marker="s",
            label="free dims after constraint",
        )
    ax.set_title("Degrees of freedom")
    ax.set_xlabel("experience")
    ax.legend(fontsize=7)
    fig.suptitle(f"Learning history: {result.config.name}")
    fig.tight_layout()
    return fig


def plot_accuracy_matrix(result, ax=None, title=None):
    """Compact summary: A[t, i] = accuracy on experience i after learning experience t."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    K = len(result.records)
    A = result.acc_matrix[:K, :K]
    ax.imshow(A, vmin=0.5, vmax=1.0, cmap="viridis")
    for t in range(K):
        for i in range(t + 1):
            ax.text(
                i,
                t,
                f"{A[t, i] * 100:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="w" if A[t, i] < 0.8 else "k",
            )
    ax.set_xticks(range(K))
    ax.set_xticklabels(range(1, K + 1))
    ax.set_yticks(range(K))
    ax.set_yticklabels(range(1, K + 1))
    ax.set_xlabel("evaluated experience i")
    ax.set_ylabel("after learning experience t")
    ax.set_title(title or f"Accuracy matrix (%): {result.config.name}")
    return ax


# ------------------------------------------------------------ G. joint-training studies
def plot_degree_comparison(models, samples=None, res=160, target=None, figsize=None):
    """Predicted regions of several models (e.g. one per polynomial degree), side by side."""
    fig, axs = plt.subplots(1, len(models), figsize=figsize or (3.4 * len(models), 3.6))
    for ax, model in zip(np.atleast_1d(axs), models):
        plot_predictions(
            model,
            ax,
            res,
            samples=samples,
            target=target,
            title=f"degree {model.degree}",
        )
    fig.tight_layout()
    return fig


def plot_accuracy_vs_degree(rows, ax=None, title="Joint-training accuracy by degree"):
    """Mean evaluation accuracy against polynomial degree, one line per epoch budget.

    ``rows`` are the dicts of ``polynomial_cl.binary.run_joint_study``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.6))
    for epochs in sorted({r["epochs"] for r in rows}):
        chosen = sorted(
            (r for r in rows if r["epochs"] == epochs), key=lambda r: r["degree"]
        )
        ax.plot(
            [r["degree"] for r in chosen],
            [r["mean_accuracy"] for r in chosen],
            marker="o",
            label=f"{epochs} epochs",
        )
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("mean evaluation accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(title)
    return ax


def plot_accuracy_vs_budget(
    rows, ax=None, title="Joint-training accuracy by training budget"
):
    """Mean evaluation accuracy against the number of epochs, one line per polynomial degree."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.6))
    for degree in sorted({r["degree"] for r in rows}):
        chosen = sorted(
            (r for r in rows if r["degree"] == degree), key=lambda r: r["epochs"]
        )
        ax.plot(
            [r["epochs"] for r in chosen],
            [r["mean_accuracy"] for r in chosen],
            marker="o",
            label=f"degree {degree}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("epochs (full-batch gradient steps)")
    ax.set_ylabel("mean evaluation accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(title)
    return ax


def plot_sequential_curves(results_by_label, axs=None):
    """Mean accuracy and mean forgetting after every experience (mean over seeds, +/- 1 std).

    ``results_by_label`` maps a label to the ``ContinualResult`` of each seed (one is enough).
    """
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, results in results_by_label.items():
        for ax, field in zip(axs, ("mean_acc", "mean_forgetting")):
            values = np.array([[getattr(x, field) for x in r.records] for r in results])
            x = np.arange(1, values.shape[1] + 1)
            mean = values.mean(axis=0)
            ax.plot(x, mean, marker="o", ms=3, label=label)
            if len(values) > 1:
                std = values.std(axis=0, ddof=1)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    axs[0].set(
        title="Mean accuracy on the experiences seen",
        xlabel="experience",
        ylabel="accuracy",
    )
    axs[1].set(title="Mean forgetting", xlabel="experience", ylabel="forgetting")
    axs[0].set_ylim(0, 1.02)
    axs[0].legend(fontsize=7)
    return axs
