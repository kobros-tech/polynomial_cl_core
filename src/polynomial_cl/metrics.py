# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Metrics for measuring learning and preservation (all computed identically for every method).

``forgetting`` and ``mean_std_by_condition`` are used by both the binary and the
class-incremental experiments; ``format_table`` prints result rows in the notebooks. The
binary experiment additionally records ``accuracy``, ``old_score_change``,
``prediction_flip_fraction`` and ``parameter_changes`` at every step. All of them are
array-level measures that do not depend on a result or record type.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def old_score_change(score_before, score_after) -> float:
    """max_x |f_after(x) - f_before(x)| over a fixed set of old inputs."""
    return float(np.max(np.abs(np.asarray(score_after) - np.asarray(score_before))))


def prediction_flip_fraction(score_before, score_after) -> float:
    """Fraction of inputs whose predicted label changed."""
    return float(
        np.mean((np.asarray(score_before) > 0) != (np.asarray(score_after) > 0))
    )


def forgetting(acc_matrix, t):
    """For experience i <= t: best accuracy achieved so far (rows i..t) minus current accuracy."""
    return np.array(
        [np.max(acc_matrix[i : t + 1, i]) - acc_matrix[t, i] for i in range(t + 1)]
    )


def parameter_changes(w_before, w_after, trainable_mask):
    """Return max frozen-coefficient change and total trainable-coefficient L1 change.

    Returns None when no coefficients are frozen; zero means frozen coefficients exist
    and all of them were unchanged.
    """
    d = np.abs(np.asarray(w_after) - np.asarray(w_before))
    m = np.asarray(trainable_mask, dtype=bool)
    frozen_max = float(d[~m].max()) if (~m).any() else None
    return frozen_max, float(d[m].sum())


def mean_std_by_condition(rows, metric_names, key="condition"):
    """Aggregate per-seed result rows: {condition: {metric: {"mean", "std"}}}.

    Each row is a dict with a ``key`` entry (default "condition") plus the named metrics. "std" is the sample
    standard deviation (ddof=1), or 0.0 when a condition has a single seed.
    """
    summary = {}
    for condition in dict.fromkeys(row[key] for row in rows):
        subset = [row for row in rows if row[key] == condition]
        summary[condition] = {}
        for name in metric_names:
            values = np.asarray([row[name] for row in subset], dtype=float)
            summary[condition][name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return summary


def format_table(rows, floatfmt="{:.3f}"):
    if not rows:
        return ""
    cols = list(rows[0])

    def cell(v):
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)

    widths = [max(len(c), *(len(cell(r[c])) for r in rows)) for c in cols]

    def line(vals):
        return "  ".join(v.rjust(w) for v, w in zip(vals, widths))

    return "\n".join([line(cols)] + [line([cell(r[c]) for c in cols]) for r in rows])
