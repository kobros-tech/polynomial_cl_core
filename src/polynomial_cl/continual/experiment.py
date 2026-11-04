# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Matched class-incremental comparison of polynomial feature expansion.

Ten classes arrive in five two-class experiences. A single softmax head with ten outputs is
trained on the current experience plus a fixed per-class replay memory; training and
evaluation only let classes introduced so far compete, so the output weights of future
classes are never trained. Three conditions are compared on the *same* task order, replay
indices, seed, optimiser and number of epochs; they differ only in polynomial degree:

* ``fixed_linear``:   degree 1 throughout (no expansion).
* ``expanding``:      degree 1, 2, then 3 from experience 3 on; new coefficients start at zero.
* ``fixed_capacity``: degree 3 from experience 1 (same final capacity, no gradual growth).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..metrics import forgetting as forgetting_from_matrix
from ..metrics import mean_std_by_condition
from ..models.multiclass import expand_weights, logits, predict
from ..training.multiclass import (
    cross_entropy,
    fit_task,
)
from .degree_rates import (
    N_DEGREE_GROUPS,
    as_degree_rates,
    degree_of_feature,
    step_scale_for,
)
from .features import (
    MAX_DEGREE,
    n_polynomial_features,
    scaled_polynomial_features,
)
from .replay import select_memory, stored_indices
from .sampling import epoch_sampler
from .strategies import (
    N_REF_PER_CLASS,
    STRATEGIES,
    update_restrictions,
)

# Ten classes arrive in five two-class experiences, in this order.
TASK_ORDER = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
CONDITIONS = ("fixed_linear", "expanding", "fixed_capacity")
DEGREES = {
    "fixed_linear": (1, 1, 1, 1, 1),
    "expanding": (1, 2, 3, 3, 3),
    "fixed_capacity": (MAX_DEGREE,) * 5,
}
SUMMARY_METRICS = ("final_mean_accuracy", "final_current_accuracy", "mean_forgetting")

STRATEGY_METRICS = (
    *SUMMARY_METRICS,
    "mean_old_logit_change",
    "mean_old_flip_fraction",
    "mean_old_takeover_fraction",
    "mean_loss_reduction",
)

# Defaults shared by the command line tool and the notebook.
EPOCHS = 3
LEARNING_RATE = 0.1
BATCH_SIZE = 128
MEMORY_PER_CLASS = 200

# Largest allowed change of any logit when zero coefficients are appended (float32 rounding).
EXPANSION_TOLERANCE = 1e-5


def expansion_tolerance(class_logits):
    """Allowed logit change when zero coefficients are appended.

    Appending zeros is exact in real arithmetic; in float32 the same logits are summed with a
    different matrix shape, so rounding grows with their magnitude (large learning rates give
    large logits). The bound is EXPANSION_TOLERANCE for logits of order 1 and relative above
    that, which still rejects any non-zero new coefficient.
    """
    return EXPANSION_TOLERANCE * max(1.0, float(np.max(np.abs(class_logits))))


class FeatureData(NamedTuple):
    """Degree-``MAX_DEGREE`` design matrices; lower degrees are column prefixes of these."""

    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    n_components: int


def make_feature_data(train_x, train_y, test_x, test_y) -> FeatureData:
    """Compute the polynomial features once; every condition and seed reuses them."""
    return FeatureData(
        scaled_polynomial_features(train_x, MAX_DEGREE),
        np.asarray(train_y),
        scaled_polynomial_features(test_x, MAX_DEGREE),
        np.asarray(test_y),
        np.shape(train_x)[1],
    )


def holdout_split(train_x, train_y, fraction, seed=123):
    """Random split of the training data into (train, validation) arrays.

    Use the validation part, never the test set, to choose hyper-parameters.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    order = np.random.default_rng(seed).permutation(len(train_x))
    cut = len(train_x) - round(fraction * len(train_x))
    fit, held = order[:cut], order[cut:]
    for part, name in ((fit, "training"), (held, "validation")):
        if len(np.unique(train_y[part])) != len(np.unique(train_y)):
            raise ValueError(
                f"the {name} part is missing a class; use more data or another fraction"
            )
    return train_x[fit], train_y[fit], train_x[held], train_y[held]


def make_task_plan(labels, seed, memory_per_class):
    """Shuffled training indices per experience, and one fixed replay memory per class."""
    if memory_per_class < 1:
        raise ValueError("memory_per_class must be positive")
    rng = np.random.default_rng(seed)
    tasks, memory = [], {}
    for classes in TASK_ORDER:
        indices = np.flatnonzero(np.isin(labels, classes))
        if len(indices) == 0:
            raise ValueError(f"training data is missing task classes {classes}")
        tasks.append(indices[rng.permutation(len(indices))])
        for class_id in classes:
            memory[class_id] = select_memory(labels, class_id, memory_per_class, rng)
    return tasks, memory


def accuracy_by_task(weights, test_x, test_y, n_tasks_seen, by_class=None):
    """Accuracy on every experience seen so far (NaN for later ones).

    Predictions are the argmax over the classes introduced so far only. If ``by_class`` (an
    array of 10 NaNs) is given, it is filled with the per-class accuracy from the same
    predictions (no extra forward pass); classes not introduced yet stay NaN.
    """
    accuracy = np.full(len(TASK_ORDER), np.nan)
    for task, classes in enumerate(TASK_ORDER[:n_tasks_seen]):
        rows = np.flatnonzero(np.isin(test_y, classes))
        predictions = predict(logits(weights, test_x[rows]), 2 * n_tasks_seen)
        accuracy[task] = np.mean(predictions == test_y[rows])
        if by_class is not None:
            for c in classes:
                of_class = test_y[rows] == c
                by_class[c] = np.mean(predictions[of_class] == c)
    return accuracy


INDICATORS = (
    "train_loss_by_task",  # [loss before, loss after] on this experience's training set (+ replay)
    "l1_update_by_task",  # sum |w_after - w_before| over the coefficients used this experience
    "n_trainable_by_task",  # coefficients that may change (B: rank of sqrt(D) N, x classes)
    # Strategy B variants only. Dimension of the null-space basis that constrains the OLD-class
    # output columns (for strategy_b one shared basis for every column; for strategy_b_new the
    # basis restricted to the newly added feature rows) and the NEW-class columns. None in
    # experience 1 (nothing to preserve) and for every other rule.
    "null_dim_by_task",
    "null_dim_new_class_columns_by_task",
    "old_logit_change_by_task",  # max change of old-class logits on old-class test inputs
    "old_flip_fraction_by_task",  # old-class test inputs whose prediction among old classes changed
    "old_takeover_fraction_by_task",  # old-class test inputs now predicted as a NEW class
    "ref_logit_change_by_task",  # Strategy B variants: max logit change at the reference inputs
)


def _mean_after_first(values):
    """Mean over experiences 2..5 (experience 1 has no old function to preserve)."""
    return float(np.mean([v for v in values[1:] if v is not None]))


def _record_indicators(
    out, weights, before, data, count, task, old_test, refs, scale, null_basis,
    loss_before, loss_after, step_scale=None, null_groups=None,
):  # fmt: skip
    n_seen, old_classes = 2 * (task + 1), 2 * task
    used = (slice(None, count), slice(None, n_seen))
    out["train_loss_by_task"].append([loss_before, loss_after])
    out["l1_update_by_task"].append(float(np.abs(weights - before)[used].sum()))
    if null_groups is not None:
        root = (
            None if step_scale is None else np.sqrt(np.asarray(step_scale, np.float64))
        )
        trainable = 0
        for columns, basis in null_groups:
            free = basis if root is None else root[:, None] * basis
            rank = int(np.linalg.matrix_rank(free)) if free.size else 0
            trainable += rank * len(range(*columns.indices(n_seen)))
    elif null_basis is not None:
        # The step is sqrt(D) N N^T sqrt(D) g, so the directions that can actually move are the
        # column space of sqrt(D) N (rank), not all columns of N: a zero-rate feature group
        # contributes null-space directions that are multiplied by 0.
        free = null_basis
        if step_scale is not None:
            free = np.sqrt(np.asarray(step_scale, np.float64))[:, None] * null_basis
        effective = int(np.linalg.matrix_rank(free)) if free.size else 0
        trainable = effective * n_seen
    else:
        # coefficient is trainable if neither its update mask nor its rate (row) is zero
        allowed = np.ones((count, n_seen), bool)
        if scale is not None:
            allowed &= scale[used] > 0
        if step_scale is not None:
            allowed &= (np.asarray(step_scale)[:count] > 0)[:, None]
        trainable = int(allowed.sum())
    out["n_trainable_by_task"].append(int(trainable))
    if null_groups is not None:  # strategy_b_new: separate bases per column group
        old_dim, new_dim = (int(basis.shape[1]) for _, basis in null_groups)
    elif null_basis is not None:  # strategy_b: one basis shared by every column
        old_dim = new_dim = int(null_basis.shape[1])
    else:
        old_dim = new_dim = None
    out["null_dim_by_task"].append(old_dim)
    out["null_dim_new_class_columns_by_task"].append(new_dim)
    if task:
        x = data.test_x[old_test, :count]
        old_before, old_after = (
            logits(before, x, old_classes),
            logits(weights, x, old_classes),
        )
        out["old_logit_change_by_task"].append(
            float(np.max(np.abs(old_after - old_before)))
        )
        flips = predict(old_before) != predict(old_after)
        out["old_flip_fraction_by_task"].append(float(flips.mean()))
        # The flip diagnostic above only lets old classes compete, so it cannot see a new
        # class taking over an old example while the old logits barely move. Here every
        # class seen so far competes, exactly as in the accuracy matrix.
        predicted = predict(logits(weights, x, n_seen))
        out["old_takeover_fraction_by_task"].append(
            float((predicted >= old_classes).mean())
        )
    else:
        out["old_logit_change_by_task"].append(None)
        out["old_flip_fraction_by_task"].append(None)
        out["old_takeover_fraction_by_task"].append(None)
    if refs is None:
        out["ref_logit_change_by_task"].append(None)
    else:
        ref_x = data.train_x[refs, :count]
        change = np.abs(ref_x @ weights[:, :n_seen] - ref_x @ before[:, :n_seen])
        out["ref_logit_change_by_task"].append(float(change.max()))


DEGREE_DIAGNOSTICS = (
    "lr",  # rate applied to the group in this experience (0 = frozen)
    "n_coefficients",  # rows in the group x classes seen so far
    "n_trainable",  # of those, positive rate and update scale (Strategy B: upper bound)
    "mean_gradient_l1",  # mean over updates of the raw gradient L1 (before any scaling)
    "mean_step_l1",  # mean over updates of the applied step L1 (after rates and masks)
    "coef_change_l1",  # sum |w_after - w_before| over the group's seen-class coefficients
    "relative_change",  # ||w_after - w_before||_2 / ||w_after||_2
    "logit_contribution",  # mean |x_g . w_g| over seen-class test inputs, seen classes
    "logit_change",  # mean |x_g . dw_g| over old-class test inputs, old classes (task > 0)
)


def _stat(stats, key, group):
    """Trainer statistic of one group, or None if the trainer did not report it."""
    return stats[key][group] if key in stats else None


def _degree_diagnostics(
    weights, before, data, count, task, group_index, n_active, rates, scale, stats
):
    """Per-degree-group diagnostics for one experience (None for inactive degree groups)."""
    n_seen, old_classes = 2 * (task + 1), 2 * task
    seen_test = np.flatnonzero(data.test_y < n_seen)
    old_test = np.flatnonzero(data.test_y < old_classes)
    out = {k: [None] * N_DEGREE_GROUPS for k in DEGREE_DIAGNOSTICS}
    for g in range(n_active):
        rows = np.flatnonzero(group_index == g)
        delta = (weights - before)[rows][:, :n_seen].astype(np.float64)
        after = weights[rows][:, :n_seen].astype(np.float64)
        rate = float(rates[g])
        if scale is None:
            trainable = len(rows) * n_seen if rate > 0 else 0
        else:
            trainable = int((scale[rows][:, :n_seen] > 0).sum()) if rate > 0 else 0
        out["lr"][g] = rate
        out["n_coefficients"][g] = len(rows) * n_seen
        out["n_trainable"][g] = int(trainable)
        out["mean_gradient_l1"][g] = _stat(stats, "mean_gradient_l1_by_group", g)
        out["mean_step_l1"][g] = _stat(stats, "mean_step_l1_by_group", g)
        out["coef_change_l1"][g] = float(np.abs(delta).sum())
        out["relative_change"][g] = float(
            np.linalg.norm(delta) / (np.linalg.norm(after) + 1e-12)
        )
        x_seen = data.test_x[seen_test][:, rows].astype(np.float64)
        out["logit_contribution"][g] = float(np.abs(x_seen @ after).mean())
        if task:
            x_old = data.test_x[old_test][:, rows].astype(np.float64)
            out["logit_change"][g] = float(
                np.abs(x_old @ delta[:, :old_classes]).mean()
            )
    return out


def run_condition(
    condition,
    data: FeatureData,
    task_indices,
    memory_indices,
    *,
    seed,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    strategy="ordinary",
    n_ref_per_class=N_REF_PER_CLASS,
    feature_power=0.0,
    sampling="shuffled",
):
    """Run one condition; returns a JSON-serialisable record with the accuracy matrix.

    ``strategy`` selects the update rule (see ``STRATEGIES``); the default is plain training with
    replay. Besides accuracy, every record carries indicators of whether the model is learning
    (loss before/after, size of the update, trainable coefficients) and how much the old
    function moved (change of old-class logits on old-class test inputs).

    ``feature_power`` p > 0 gives every feature its own step size
    ``learning_rate * (max_rms / rms_i) ** p`` (rms of feature i on the experience's training
    set): p = 0 is one shared rate, p = 1 diagonal preconditioning.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    sampler = epoch_sampler(sampling)
    per_degree = np.ndim(learning_rate) > 0
    if per_degree and feature_power:
        raise ValueError(
            "a per-degree learning-rate vector cannot be combined with feature_power"
        )
    rate_vector = as_degree_rates(learning_rate)
    degrees = DEGREES[condition]
    n_tasks = len(degrees)
    weights = np.zeros(
        (n_polynomial_features(data.n_components, degrees[0]), 10), np.float32
    )
    matrix = np.full((n_tasks, n_tasks), np.nan)
    class_matrix = np.full(
        (n_tasks, 10), np.nan
    )  # accuracy of each class after each experience
    feature_counts, expansion_changes = [], []
    indicators = {k: [] for k in INDICATORS}
    degree_diag = {k: [] for k in DEGREE_DIAGNOSTICS}
    sampling_log = []
    group_index_full = degree_of_feature(data.n_components, MAX_DEGREE)

    for task, indices in enumerate(task_indices):
        old_rows = weights.shape[0] if task else 0
        count = n_polynomial_features(data.n_components, degrees[task])
        if count > weights.shape[0]:
            # Expansion invariant: appending zero coefficients must not change any logit.
            probe = data.test_x[:256, :count]
            before = probe[:, : weights.shape[0]] @ weights
            weights = expand_weights(weights, count)
            change = float(np.max(np.abs(probe @ weights - before)))
            if change > expansion_tolerance(before):
                raise RuntimeError(
                    f"zero-initialized expansion changed logits by {change}"
                )
            expansion_changes.append(change)
        feature_counts.append(count)

        old_classes = 2 * task
        n_seen = 2 * (task + 1)
        if task and strategy != "ordinary_no_replay":
            replay = stored_indices(memory_indices, old_classes)
        else:
            replay = np.empty(0, dtype=np.int64)
        train_x = np.concatenate(
            [data.train_x[indices, :count], data.train_x[replay, :count]]
        )
        train_y = data.train_y[np.concatenate([indices, replay])].astype(
            np.int64, copy=False
        )
        feature_scale = None
        if feature_power:
            rms = np.sqrt(np.mean(train_x.astype(np.float64) ** 2, axis=0))
            feature_scale = (
                (rms.max() / np.maximum(rms, 1e-12)) ** feature_power
            ).astype(np.float32)
        group_index = group_index_full[:count]
        n_active = int(group_index.max()) + 1
        base_rate, group_scale = learning_rate, None
        if per_degree:
            base_rate, group_scale = step_scale_for(rate_vector, group_index)
        step_scale = feature_scale
        if group_scale is not None:
            step_scale = (
                group_scale if feature_scale is None else feature_scale * group_scale
            )
        restrictions = update_restrictions(
            strategy,
            weights_shape=weights.shape,
            old_rows=old_rows,
            old_classes=old_classes,
            n_seen=n_seen,
            count=count,
            task=task,
            train_x=data.train_x,
            memory_indices=memory_indices,
            n_ref_per_class=n_ref_per_class,
            step_scale=step_scale,
        )
        scale, null_basis, refs, null_groups = restrictions
        old_test = np.flatnonzero(data.test_y < old_classes)
        before = weights.copy()
        loss_before = cross_entropy(weights, train_x, train_y, n_seen)
        fit_stats = {}
        weights = fit_task(
            weights,
            data.train_x[indices, :count],
            data.train_y[indices].astype(np.int64, copy=False),
            data.train_x[replay, :count],
            data.train_y[replay].astype(np.int64, copy=False),
            epochs=epochs,
            learning_rate=base_rate,
            batch_size=batch_size,
            seed=seed + task,
            eligible_classes=np.arange(n_seen),
            update_scale=scale,
            null_basis=null_basis,
            null_groups=null_groups,
            feature_scale=feature_scale,
            group_scale=group_scale,
            group_index=group_index,
            epoch_sampler=sampler,
            stats=fit_stats,
        )
        effective = (
            rate_vector
            if per_degree
            else np.full(N_DEGREE_GROUPS, float(learning_rate))
        )
        if feature_scale is not None:  # report the mean step size of each group
            sums = np.bincount(group_index, feature_scale, N_DEGREE_GROUPS)
            sizes = np.maximum(np.bincount(group_index, minlength=N_DEGREE_GROUPS), 1)
            effective = effective * sums / sizes
        diag = _degree_diagnostics(
            weights, before, data, count, task, group_index, n_active,
            effective, scale, fit_stats,
        )  # fmt: skip
        for key in DEGREE_DIAGNOSTICS:
            degree_diag[key].append(diag[key])
        sampling_log.append(
            {"policy": sampling}
            | {
                k: fit_stats.get(k)
                for k in (
                    "pool_sizes_by_class",
                    "samples_drawn_by_class",
                    "n_updates",
                    "examples_processed",
                    "examples_available",
                    "mean_batch_composition",
                )
            }
        )
        _record_indicators(
            indicators, weights, before, data, count, task, old_test, refs, scale, null_basis,
            loss_before, cross_entropy(weights, train_x, train_y, n_seen),
            step_scale=step_scale, null_groups=null_groups,
        )  # fmt: skip
        matrix[task, : task + 1] = accuracy_by_task(
            weights,
            data.test_x[:, :count],
            data.test_y,
            task + 1,
            by_class=class_matrix[task],
        )[: task + 1]

    # Forgetting of an experience: best accuracy it ever had minus its final accuracy
    # (shared definition from polynomial_cl.metrics; the lower triangle has no NaN).
    forgetting = [float(v) for v in forgetting_from_matrix(matrix, n_tasks - 1)]
    return {
        "condition": condition,
        "strategy": strategy,
        "seed": seed,
        "active_degree_by_task": list(degrees),
        "feature_count_by_task": feature_counts,
        "expansion_max_logit_changes": expansion_changes,
        "accuracy_matrix": [
            [float(v) if np.isfinite(v) else None for v in row] for row in matrix
        ],
        "accuracy_by_class_matrix": [
            [float(v) if np.isfinite(v) else None for v in row] for row in class_matrix
        ],
        "final_mean_accuracy": float(np.nanmean(matrix[-1])),
        "final_current_accuracy": float(matrix[-1, -1]),
        "forgetting_by_task": forgetting,
        "learning_rate": [float(v) for v in rate_vector]
        if per_degree
        else float(learning_rate),
        "degree_learning_rates": [float(v) for v in rate_vector],
        "per_degree_rates": bool(per_degree),
        "sampling": sampling,
        "sampling_by_task": sampling_log,
        "degree_diagnostics": degree_diag,
        "current_accuracy_by_task": [float(matrix[t, t]) for t in range(n_tasks)],
        "mean_old_accuracy_by_task": [None]
        + [float(np.mean(matrix[t, :t])) for t in range(1, n_tasks)],
        "mean_accuracy_by_task": [
            float(np.mean(matrix[t, : t + 1])) for t in range(n_tasks)
        ],
        # The last experience has no later evaluation, so it is excluded from the mean.
        "mean_forgetting": float(np.mean(forgetting[:-1])),
        **indicators,
        "mean_old_logit_change": _mean_after_first(
            indicators["old_logit_change_by_task"]
        ),
        "mean_old_flip_fraction": _mean_after_first(
            indicators["old_flip_fraction_by_task"]
        ),
        "mean_old_takeover_fraction": _mean_after_first(
            indicators["old_takeover_fraction_by_task"]
        ),
        "mean_loss_reduction": float(
            np.mean(
                [before - after for before, after in indicators["train_loss_by_task"]]
            )
        ),
    }


def summarize_runs(runs):
    """Mean and sample standard deviation over seeds, per condition."""
    return mean_std_by_condition(runs, SUMMARY_METRICS)


def run_comparison(
    data: FeatureData,
    seeds,
    *,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    memory_per_class=MEMORY_PER_CLASS,
    feature_power=0.0,
    sampling="shuffled",
):
    """Run all three conditions for every seed; within a seed they share task order and replay.

    ``data`` comes from ``make_feature_data`` (computed once, reusable across calls).
    """
    if not len(seeds):
        raise ValueError("at least one seed is required")
    runs = []
    for seed in seeds:
        tasks, memory = make_task_plan(data.train_y, seed, memory_per_class)
        for condition in CONDITIONS:
            runs.append(
                run_condition(
                    condition,
                    data,
                    tasks,
                    memory,
                    seed=seed,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    feature_power=feature_power,
                    sampling=sampling,
                )
            )
    return {"runs": runs, "summary": summarize_runs(runs)}


def run_strategy_comparison(
    data: FeatureData,
    seeds,
    *,
    strategies=STRATEGIES,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    memory_per_class=MEMORY_PER_CLASS,
    n_ref_per_class=N_REF_PER_CLASS,
    feature_power=0.0,
    sampling="shuffled",
):
    """Compare update rules on the ``expanding`` schedule; same task order and replay per seed."""
    if not len(seeds):
        raise ValueError("at least one seed is required")
    runs = []
    for seed in seeds:
        tasks, memory = make_task_plan(data.train_y, seed, memory_per_class)
        for strategy in strategies:
            runs.append(
                run_condition(
                    "expanding", data, tasks, memory, seed=seed, epochs=epochs,
                    learning_rate=learning_rate, batch_size=batch_size,
                    strategy=strategy, n_ref_per_class=n_ref_per_class, feature_power=feature_power,
                    sampling=sampling,
                )
            )  # fmt: skip
    return {
        "runs": runs,
        "summary": mean_std_by_condition(runs, STRATEGY_METRICS, key="strategy"),
    }


def run_joint_reference(
    data: FeatureData,
    *,
    degree=MAX_DEGREE,
    seed=0,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
):
    """Conventional shuffled joint training on all ten classes (not a continual-learning run).

    One softmax head sees every training example of every class at once, with the same
    features, optimiser, rate and batch size as the continual runs; ``epochs`` here counts
    passes over the *whole* training set. Accuracy is argmax over all ten classes on the
    independent test set. There is no sequence of experiences, so no forgetting is reported.
    """
    if not 1 <= degree <= MAX_DEGREE:
        raise ValueError(f"degree must be between 1 and {MAX_DEGREE}")
    count = n_polynomial_features(data.n_components, degree)
    weights = np.zeros((count, 10), np.float32)
    labels = data.train_y.astype(np.int64, copy=False)
    empty = np.empty((0, count), np.float32)
    loss_before = cross_entropy(weights, data.train_x[:, :count], labels, 10)
    fit_task(
        weights,
        data.train_x[:, :count],
        labels,
        empty,
        np.empty(0, np.int64),
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )
    predictions = predict(logits(weights, data.test_x[:, :count]))
    per_class = [float(np.mean(predictions[data.test_y == c] == c)) for c in range(10)]
    return {
        "degree": degree,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": float(learning_rate),
        "n_features": count,
        "test_accuracy": float(np.mean(predictions == data.test_y)),
        "per_class_accuracy": per_class,
        "train_loss": [
            loss_before,
            cross_entropy(weights, data.train_x[:, :count], labels, 10),
        ],
    }


def run_joint_comparison(
    data: FeatureData, degrees, epochs_list, seeds, learning_rate=LEARNING_RATE
):
    """``run_joint_reference`` over degrees x epochs x seeds, summarised over the seeds.

    A reference condition, not a continual-learning strategy: no forgetting is computed.
    """
    rows = []
    for degree in degrees:
        for epochs in epochs_list:
            runs = [
                run_joint_reference(
                    data,
                    degree=degree,
                    seed=s,
                    epochs=epochs,
                    learning_rate=learning_rate,
                )
                for s in seeds
            ]
            accs = [r["test_accuracy"] for r in runs]
            rows.append(
                {
                    "degree": degree,
                    "epochs": epochs,
                    "learning_rate": float(learning_rate),
                    "n_features": runs[0]["n_features"],
                    "seeds": list(seeds),
                    "test_accuracy_by_seed": accs,
                    "test_accuracy_mean": float(np.mean(accs)),
                    "test_accuracy_std": float(np.std(accs, ddof=1))
                    if len(accs) > 1
                    else None,
                    "per_class_accuracy_mean": [
                        float(v)
                        for v in np.mean(
                            [r["per_class_accuracy"] for r in runs], axis=0
                        )
                    ],
                }
            )
    return rows


# ---------------------------------------------------------------- reporting helpers
def summary_rows(result, key="condition", metric_names=None):
    """Table rows (mean +/- std over seeds) from the ``summary`` of a comparison result."""
    rows = []
    for name, stats in result["summary"].items():
        chosen = metric_names or tuple(stats)
        rows.append(
            {
                key: name,
                **{
                    m: f"{stats[m]['mean']:.3f}+/-{stats[m]['std']:.3f}" for m in chosen
                },
            }
        )
    return rows


def old_class_rows(run):
    """Per experience (from the 2nd on): how training affected the old classes' test inputs."""
    return [
        {
            "after_experience": t + 1,
            "taken_by_new_class": f"{run['old_takeover_fraction_by_task'][t]:.3f}",
            "flipped_among_old": f"{run['old_flip_fraction_by_task'][t]:.3f}",
            "max_old_logit_change": f"{run['old_logit_change_by_task'][t]:.2f}",
        }
        for t in range(1, len(run["accuracy_matrix"]))
    ]


def _dash(value):
    return "-" if value is None else value


def strategy_diagnostic_rows(run):
    """Per experience: loss, trainable coefficients and Strategy B null-space sizes of one run.

    ``null_dim`` is the dimension of the allowed update space for the old-class columns and
    ``null_dim_new`` for the new-class columns (``-`` where the strategy has no null-space constraint).
    """
    return [
        {
            "experience": t + 1,
            "loss_before": f"{before:.2f}",
            "loss_after": f"{after:.2f}",
            "n_trainable": run["n_trainable_by_task"][t],
            "null_dim": _dash(run["null_dim_by_task"][t]),
            "null_dim_new": _dash(run["null_dim_new_class_columns_by_task"][t]),
            "l1_update": f"{run['l1_update_by_task'][t]:.2f}",
        }
        for t, (before, after) in enumerate(run["train_loss_by_task"])
    ]


def sampling_rows(sampling):
    """Examples processed and repetition per sampling policy and strategy (mean over seeds).

    ``sampling`` is the output of ``run_sampling_comparison``. ``draws_per_example`` is, for
    the last experience, how often each stored example of the least- and most-drawn class was
    used (1 per epoch for ``shuffled``); classes with an empty pool are ignored.
    """
    rows = []
    for policy, result in sampling.items():
        for strategy in result["summary"]:
            runs = [r for r in result["runs"] if r["strategy"] == strategy]
            last = [r["sampling_by_task"][-1] for r in runs]
            per_example = []
            for log in last:
                pool = np.array(log["pool_sizes_by_class"], dtype=float)
                drawn = np.array(log["samples_drawn_by_class"], dtype=float)
                per_example.append(drawn[pool > 0] / pool[pool > 0])
            rows.append(
                {
                    "sampling": policy,
                    "strategy": strategy,
                    "examples_processed": int(
                        np.mean(
                            [
                                sum(
                                    t["examples_processed"]
                                    for t in r["sampling_by_task"]
                                )
                                for r in runs
                            ]
                        )
                    ),
                    "examples_available": int(
                        np.mean(
                            [
                                sum(
                                    t["examples_available"]
                                    for t in r["sampling_by_task"]
                                )
                                for r in runs
                            ]
                        )
                    ),
                    "updates": int(
                        np.mean(
                            [
                                sum(t["n_updates"] for t in r["sampling_by_task"])
                                for r in runs
                            ]
                        )
                    ),
                    "draws_per_example_min": f"{np.mean([v.min() for v in per_example]):.2f}",
                    "draws_per_example_max": f"{np.mean([v.max() for v in per_example]):.2f}",
                }
            )
    return rows


def per_seed_rows(runs_by_label, key="method"):
    """One row per (label, seed): final mean accuracy, newest-experience accuracy, forgetting."""
    return [
        {
            key: label,
            "seed": run["seed"],
            "final_mean_accuracy": f"{run['final_mean_accuracy']:.3f}",
            "final_current_accuracy": f"{run['final_current_accuracy']:.3f}",
            "mean_forgetting": f"{run['mean_forgetting']:.3f}",
        }
        for label, runs in runs_by_label.items()
        for run in runs
    ]
