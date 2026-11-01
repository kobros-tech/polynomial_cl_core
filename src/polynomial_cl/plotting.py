# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Reusable plotting helpers. All functions take a model (never modify it) and return the Axes/Figure.

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

from .data import DOMAIN, boundary_curve, true_label
from .metrics import format_table

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


def _truth_line(ax, **kw):
    xs = np.linspace(*DOMAIN, 400)
    ax.plot(xs, boundary_curve(xs), **{"color": "k", "lw": 1.2, "ls": "--", **kw})


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
def plot_ground_truth(ax=None, res=200, samples=None, title="Ground truth"):
    xx, yy, pts = make_grid(res)
    ax = _ax(ax)
    ax.contourf(
        xx,
        yy,
        true_label(pts).reshape(xx.shape),
        levels=[-0.5, 0.5, 1.5],
        cmap=CLASS_CMAP,
        alpha=0.55,
    )
    _truth_line(ax)
    if samples is not None:
        plot_samples(ax, *samples)
    ax.set_title(title)
    return ax


# ------------------------------------------------------------ B. predictions
def plot_predictions(
    model, ax=None, res=200, samples=None, title=None, show_truth=True
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
        _truth_line(ax, color="0.35", lw=1.0)
    if samples is not None:
        plot_samples(ax, *samples)
    ax.set_title(title or f"Predicted regions (degree {model.degree})")
    return ax


# ------------------------------------------------------------ C. correctness
def correctness_grid(model, res=200):
    xx, yy, pts = make_grid(res)
    t, p = true_label(pts), model.predict(pts)
    cat = np.where(t == p, p, np.where(p == 1, 2, 3))  # TN=0, TP=1, FP=2, FN=3
    return xx, yy, cat.reshape(xx.shape)


def plot_correctness(
    model,
    ax=None,
    res=200,
    samples=None,
    title="Correct vs incorrect regions",
    legend=True,
):
    xx, yy, cat = correctness_grid(model, res)
    ax = _ax(ax)
    ax.contourf(
        xx,
        yy,
        cat,
        levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
        colors=CORRECT_COLORS,
        alpha=0.85,
    )
    _truth_line(ax)
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
    model_a, model_b, ax=None, res=200, title="Changed predictions"
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
    _truth_line(ax)
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
def plot_step(record, experiences=None, res=160, figsize=(21, 4.2)):
    """before expansion | right after expansion | after training | ground truth | prediction changes."""
    fig, axs = plt.subplots(1, 5, figsize=figsize)
    cur = experiences[record.t].test if experiences is not None else None
    samples = (cur.x, cur.y) if cur is not None else None
    plot_predictions(
        record.model_before,
        axs[0],
        res,
        title=f"Exp {record.t + 1}: before expansion (deg {record.degree_before})",
    )
    plot_predictions(
        record.model_expanded,
        axs[1],
        res,
        title=f"after expansion (deg {record.degree_after}), untrained",
    )
    plot_predictions(
        record.model_after,
        axs[2],
        res,
        samples=samples,
        title="after training (current test samples)",
    )
    plot_ground_truth(axs[3], res)
    plot_prediction_changes(
        record.model_before, record.model_after, axs[4], res, "before -> after training"
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


def plot_loss_curve(loss_history, ax=None, title="Training loss"):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(loss_history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE")
    ax.set_title(title)
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


# ===================================================== class-incremental plots
# These take lists of run records from ``polynomial_cl.class_incremental`` (plain dicts) and are
# shared by every CIL comparison: accuracy and forgetting always use the 0-1 axis so methods
# can be compared across figures.
_SAMPLING_STYLE = {"shuffled": ("o", "-"), "class_balanced": ("s", "--")}
_DEGREE_COLORS = ("tab:gray", "tab:blue", "tab:orange", "tab:green")


def _cil_ax(ax, size):
    """Plain axes (the binary helper ``_ax`` fixes the x1/x2 domain and equal aspect)."""
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    return ax


def _stack(runs, key):
    """(n_runs, n_tasks) array of a per-experience list; None becomes NaN."""
    return np.array(
        [[np.nan if v is None else v for v in r[key]] for r in runs], dtype=float
    )


def _mean_std(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    return mean, std


def plot_cil_accuracy_over_experiences(runs_by_label, ax=None, title=None):
    """Mean accuracy over the experiences seen so far (solid) and on the newest one (dashed).

    ``runs_by_label`` maps a label to the list of runs (seeds) of that method.
    """
    ax = _cil_ax(ax, (6.5, 4.4))
    x = np.arange(1, 6)
    for i, (label, runs) in enumerate(runs_by_label.items()):
        color = f"C{i % 10}"
        mean, std = _mean_std(_stack(runs, "mean_accuracy_by_task"))
        ax.plot(x, mean, "-o", color=color, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        cur, _ = _mean_std(_stack(runs, "current_accuracy_by_task"))
        ax.plot(x, cur, "--", color=color, alpha=0.6)
    ax.set(
        xlabel="experience", ylabel="accuracy", ylim=(0, 1.02), xticks=x,
        title=title or "Mean accuracy over seen experiences (solid) / newest (dashed)",
    )  # fmt: skip
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return ax


def plot_cil_forgetting(runs_by_label, ax=None, title=None):
    """Mean forgetting per method: bars are the mean over seeds, whiskers the sample std."""
    ax = _cil_ax(ax, (6.5, 4.2))
    labels = list(runs_by_label)
    values = [[r["mean_forgetting"] for r in runs] for runs in runs_by_label.values()]
    means = [np.mean(v) for v in values]
    stds = [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in values]
    ax.bar(
        range(len(labels)),
        means,
        yerr=stds,
        color=[f"C{i % 10}" for i in range(len(labels))],
        capsize=3,
    )
    ax.set(xticks=range(len(labels)), ylim=(0, 1), ylabel="mean forgetting (lower is better)",
           title=title or "Forgetting")  # fmt: skip
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_cil_tradeoff(points, ax=None, title=None, annotate=True):
    """Final mean accuracy against mean forgetting; the ideal corner is top left.

    ``points`` is a list of dicts with ``label``, ``runs`` and optional ``sampling`` and
    ``color``; the marker shows the sampling policy and the marker size the final accuracy on
    the newest experience (plasticity), so a method that only forgets less because it hardly
    learns the new classes is visible.
    """
    ax = _cil_ax(ax, (6.5, 5.0))
    for i, p in enumerate(points):
        runs = p["runs"]
        forgetting = np.mean([r["mean_forgetting"] for r in runs])
        accuracy_ = np.mean([r["final_mean_accuracy"] for r in runs])
        current = np.mean([r["final_current_accuracy"] for r in runs])
        marker = _SAMPLING_STYLE.get(p.get("sampling", "shuffled"), ("o", "-"))[0]
        ax.scatter(forgetting, accuracy_, s=30 + 220 * current, marker=marker,
                   color=p.get("color", f"C{i % 10}"), alpha=0.8, edgecolor="k", linewidth=0.6)  # fmt: skip
        if annotate:
            ax.annotate(p["label"], (forgetting, accuracy_), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")  # fmt: skip
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean forgetting (lower is better)",
           ylabel="final mean accuracy (higher is better)",
           title=title or "Accuracy / forgetting trade-off (marker size = newest-experience accuracy)")  # fmt: skip
    ax.grid(alpha=0.3)
    return ax


def plot_cil_takeover_flips(runs_by_label, ax=None, title=None):
    """Old-class test inputs predicted as a NEW class (takeover) and moved among OLD classes."""
    ax = _cil_ax(ax, (7.0, 4.2))
    labels = list(runs_by_label)
    x = np.arange(len(labels))
    for k, (key, name) in enumerate(
        (("mean_old_takeover_fraction", "takeover by new classes"),
         ("mean_old_flip_fraction", "flips among old classes"))
    ):  # fmt: skip
        values = [[r[key] for r in runs] for runs in runs_by_label.values()]
        ax.bar(x + (k - 0.5) * 0.38, [np.mean(v) for v in values], 0.38,
               yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in values], capsize=2, label=name)  # fmt: skip
    ax.set(xticks=x, ylim=(0, 1), ylabel="fraction of old-class test inputs",
           title=title or "Where old-class accuracy goes")  # fmt: skip
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_cil_accuracy_matrix(runs, ax=None, title=None):
    """Seed-mean accuracy matrix (row = after experience t, column = evaluated experience)."""
    ax = _cil_ax(ax, (4.6, 4.0))
    stacked = np.array(
        [[[np.nan if v is None else v for v in row] for row in r["accuracy_matrix"]] for r in runs],
        dtype=float,
    )  # fmt: skip
    mean = np.mean(stacked, axis=0)  # NaN in the unreached upper triangle stays NaN
    image = ax.imshow(mean, vmin=0, vmax=1, cmap="viridis")
    for t in range(5):
        for i in range(t + 1):
            ax.text(i, t, f"{mean[t, i]:.2f}", ha="center", va="center", fontsize=8,
                    color="w" if mean[t, i] < 0.6 else "k")  # fmt: skip
    ax.set(xticks=range(5), yticks=range(5), xticklabels=range(1, 6), yticklabels=range(1, 6),
           xlabel="evaluated experience", ylabel="after training experience",
           title=title or "Accuracy matrix")  # fmt: skip
    plt.colorbar(image, ax=ax, fraction=0.046)
    return ax


def plot_degree_diagnostic(runs, key, ax=None, title=None, logy=True, ylabel=None):
    """Seed-mean per-degree diagnostic: for every experience one bar per active degree group."""
    from .class_incremental.degree_study import mean_degree_diagnostic

    ax = _cil_ax(ax, (7.0, 4.2))
    values = mean_degree_diagnostic(runs, key)
    n_groups = values.shape[1]
    for g in range(n_groups):
        ax.bar(np.arange(5) + (g - (n_groups - 1) / 2) * 0.2, np.nan_to_num(values[:, g]), 0.2,
               color=_DEGREE_COLORS[g], label=f"degree {g}")  # fmt: skip
    if logy:
        ax.set_yscale("log")
    ax.set(xticks=range(5), xticklabels=range(1, 6), xlabel="experience",
           ylabel=ylabel or key, title=title or key)  # fmt: skip
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_rate_vectors_with_results(vectors, runs_by_vector, selected=None, ax_rates=None,
                                   ax_results=None, title=None):  # fmt: skip
    """Left: the rate vector of every candidate (one column per degree). Right: its results.

    Rates are drawn on a log colour scale; accuracy and forgetting are seed means with whiskers
    (sample std), so each vector can be read next to what it achieved.
    """
    if ax_rates is None or ax_results is None:
        _, (ax_rates, ax_results) = plt.subplots(1, 2, figsize=(12, 0.45 * len(vectors) + 2),
                                                 gridspec_kw={"width_ratios": [1, 2]})  # fmt: skip
    names = list(vectors)
    rates = np.array([vectors[n] for n in names], dtype=float)
    image = ax_rates.imshow(
        np.log10(np.maximum(rates, 1e-4)), cmap="magma_r", aspect="auto"
    )
    for r, row in enumerate(rates):
        for c, value in enumerate(row):
            ax_rates.text(c, r, f"{value:.3g}", ha="center", va="center", fontsize=7,
                          color="w" if value > np.median(rates) else "k")  # fmt: skip
    ax_rates.set(xticks=range(rates.shape[1]), xticklabels=[f"d{g}" for g in range(rates.shape[1])],
                 yticks=range(len(names)), title="learning rate per degree")  # fmt: skip
    ax_rates.set_yticklabels(
        [("* " if n == selected else "") + n for n in names], fontsize=8
    )
    plt.colorbar(image, ax=ax_rates, fraction=0.046, label="log10 rate")
    y = np.arange(len(names))
    for k, (key, label) in enumerate((("final_mean_accuracy", "final mean accuracy"),
                                      ("mean_forgetting", "mean forgetting"),
                                      ("final_current_accuracy", "newest-experience accuracy"))):  # fmt: skip
        vals = [[r[key] for r in runs_by_vector[n]] for n in names]
        ax_results.barh(y + (k - 1) * 0.27, [np.mean(v) for v in vals], 0.27,
                        xerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals], label=label)  # fmt: skip
    ax_results.set(xlim=(0, 1), yticks=y, yticklabels=[], xlabel="0-1 scale",
                   title=title or "results (* = selected on validation)")  # fmt: skip
    ax_results.invert_yaxis()
    ax_results.legend(fontsize=7, loc="lower right")
    ax_results.grid(axis="x", alpha=0.3)
    return ax_rates, ax_results


def plot_sampling_counts(logs_by_policy, task=-1, ax=None, title=None):
    """Examples per class available (pool) and actually drawn in one experience, per policy."""
    ax = _cil_ax(ax, (7.0, 4.0))
    first = next(iter(logs_by_policy.values()))[task]
    pool = np.array(first["pool_sizes_by_class"], dtype=float)
    x = np.arange(10)
    ax.bar(x - 0.3, pool, 0.2, color="lightgray", label="pool (stored or new examples)")
    for k, (policy, logs) in enumerate(logs_by_policy.items()):
        ax.bar(
            x - 0.1 + 0.2 * k,
            logs[task]["samples_drawn_by_class"],
            0.2,
            label=f"drawn, {policy}",
        )
    ax.set(xticks=x, xlabel="class", ylabel="examples (symlog scale)",
           title=title or "Class composition of one experience's training")  # fmt: skip
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    return ax
