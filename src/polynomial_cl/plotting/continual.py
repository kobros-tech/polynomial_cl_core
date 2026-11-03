# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Plots for the class-incremental experiments.

These take lists of run records from ``polynomial_cl.continual`` (plain dicts) and are
shared by every CIL comparison: accuracy and forgetting always use the 0-1 axis so methods
can be compared across figures. The binary experiment has its own plotting module
(``polynomial_cl.plotting.synthetic``); the two share no helper because their axes, inputs
and conventions differ.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..continual.degree_study import mean_degree_diagnostic

_SAMPLING_STYLE = {"shuffled": ("o", "-"), "class_balanced": ("s", "--")}
_DEGREE_COLORS = ("tab:gray", "tab:blue", "tab:orange", "tab:green")


def _cil_ax(ax, size):
    """The given axes, or a new figure's axes of the given size."""
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    return ax


def _stack(runs, key):
    """(n_runs, n_tasks) array of a per-experience list; None becomes NaN."""
    return np.array(
        [[np.nan if v is None else v for v in r[key]] for r in runs], dtype=float
    )


def _stack_nan(runs, key, row):
    """Row ``row`` of a per-run matrix field with unseen entries (None) as NaN: (runs, columns)."""
    return np.array(
        [[np.nan if v is None else v for v in run[key][row]] for run in runs],
        dtype=float,
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


def plot_cil_per_class_accuracy(runs_by_label, task=-1, ax=None, title=None):
    """Mean test accuracy of each of the ten classes after experience ``task`` (default: last).

    Bars are the mean over seeds, whiskers one standard deviation; classes not yet introduced
    at ``task`` are left empty. ``runs_by_label`` maps a label to a list of run records.
    """
    ax = _cil_ax(ax, (7.0, 3.6))
    labels = list(runs_by_label)
    width = 0.8 / max(len(labels), 1)
    for i, label in enumerate(labels):
        values = _stack_nan(runs_by_label[label], "accuracy_by_class_matrix", task)
        seen = np.isfinite(values)
        n = seen.sum(axis=0)
        mean = np.divide(
            np.where(seen, values, 0.0).sum(axis=0),
            n,
            out=np.full(10, np.nan),
            where=n > 0,
        )
        squares = np.where(seen, (values - mean) ** 2, 0.0).sum(axis=0)
        std = np.sqrt(np.divide(squares, n - 1, out=np.zeros(10), where=n > 1))
        ax.bar(np.arange(10) + (i - (len(labels) - 1) / 2) * width, mean, width, yerr=std, label=label, capsize=2)  # fmt: skip
    ax.set_xticks(range(10))
    ax.set_xlabel("class")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    when = "the last experience" if task == -1 else f"experience {task + 1}"
    ax.set_title(title or f"Per-class accuracy after {when}")
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
