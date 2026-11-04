# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Softmax-head training for class-incremental experiments (NumPy only)."""

from __future__ import annotations

import numpy as np

from ..models.multiclass import softmax

SAMPLING_POLICIES = ("shuffled", "class_balanced")


def _balanced_batches(labels, n_updates, batch_size, rng):
    """Yield ``n_updates`` index batches with (near-)equal representation of every class.

    Every class that has examples gets ``batch_size // n_classes`` slots; the remaining slots
    go to randomly chosen classes. A class is read through its own shuffled cycle: its
    examples are used once each, in random order, before any is repeated (the cycle is then
    reshuffled), so a class with fewer examples than its slots is repeated, never padded
    from another class. All batches are full, so the number of updates matches the shuffled
    policy while the total of drawn rows can be slightly larger.
    """
    classes = np.unique(labels)
    pools = {int(c): np.flatnonzero(labels == c) for c in classes}
    cycle, cursor = {}, {}
    for c, pool in pools.items():
        cycle[c], cursor[c] = pool[rng.permutation(len(pool))], 0
    base, extra = divmod(batch_size, len(classes))
    for _ in range(n_updates):
        bonus = np.zeros(len(classes), dtype=np.int64)
        bonus[rng.choice(len(classes), extra, replace=False)] = 1
        parts = []
        for c, more in zip(classes, bonus):
            c, need = int(c), base + int(more)
            while need:
                if cursor[c] == len(cycle[c]):
                    cycle[c] = pools[c][rng.permutation(len(pools[c]))]
                    cursor[c] = 0
                take = min(need, len(cycle[c]) - cursor[c])
                parts.append(cycle[c][cursor[c] : cursor[c] + take])
                cursor[c] += take
                need -= take
        yield np.concatenate(parts)


def fit_task(
    weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    replay_features: np.ndarray,
    replay_labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    eligible_classes: np.ndarray | None = None,
    update_scale: np.ndarray | None = None,
    null_basis: np.ndarray | None = None,
    null_groups: list | None = None,
    feature_scale: np.ndarray | None = None,
    group_scale: np.ndarray | None = None,
    group_index: np.ndarray | None = None,
    sampling: str = "shuffled",
    stats: dict | None = None,
) -> np.ndarray:
    """Train a softmax head on the current task's examples plus replay examples.

    If eligible_classes is given, the softmax competition and the gradient are restricted to
    those output columns, so output weights of classes that have not been introduced yet are
    never touched. The weight array is updated in place and returned. Replay examples are
    appended to the current examples for this call only.

    Optional restrictions on the update (used by the strategy comparison):
    ``update_scale`` (n_features, n_classes) multiplies the gradient elementwise (0 freezes a
    coefficient, 0.1 slows it); ``null_basis`` (n_features, k) projects every gradient column
    onto its span; if the basis spans the null space of a design matrix, the logits of those
    design rows are unchanged by every update. ``feature_scale`` (n_features,) gives each
    feature its own step size (diagonal preconditioner D): the step is D g. Combined with
    ``null_basis`` it is D^(1/2) N N^T D^(1/2) g, so ``null_basis`` must then span the null
    space of the design scaled by D^(1/2); this stays a descent direction and keeps the
    constraint exactly.

    ``null_groups`` generalises ``null_basis`` to column-dependent constraints: a list of
    ``(columns, basis)`` where ``columns`` is a slice over the eligible output columns and
    ``basis`` (n_features, k) spans the directions allowed for those columns (rows outside the
    trainable rows of that column group are zero in the basis, so they cannot move). Columns
    outside every group are updated unprojected. It cannot be combined with ``null_basis``.

    ``group_scale`` (n_features,) is the per-degree-group multiplier (see ``degree_rates``):
    non-negative, 0 freezes a feature row. It multiplies into the same diagonal D as
    ``feature_scale`` (D = feature_scale * group_scale), so everything above, including the
    null-space projection, holds for D; the two arguments stay separate because they are
    different mechanisms. The effective rate of feature i is ``learning_rate * D[i]``.

    ``sampling`` selects how mini-batches are formed: ``"shuffled"`` (default) shuffles the
    combined rows every epoch; ``"class_balanced"`` draws every batch with equal class
    representation (see ``_balanced_batches``). Both make ceil(rows / batch_size) updates per
    epoch. If ``stats`` is a dict it is filled with the rows drawn per class and, when
    ``group_index`` (degree of every feature row) is given, the mean raw-gradient L1 and
    the mean applied-step L1 of every group.
    """
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must be a non-empty 2D array")
    if labels.shape != (len(features),):
        raise ValueError("labels must have one entry per feature row")
    if weights.ndim != 2 or weights.shape[0] != features.shape[1]:
        raise ValueError("weights must have shape (n_features, n_classes)")
    if len(replay_labels):
        if replay_features.ndim != 2 or replay_features.shape[1] != features.shape[1]:
            raise ValueError("replay_features must match the current feature dimension")
        if replay_labels.shape != (len(replay_features),):
            raise ValueError("replay labels must match replay feature rows")
        features = np.concatenate((features, replay_features), axis=0)
        labels = np.concatenate((labels, replay_labels), axis=0)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if sampling not in SAMPLING_POLICIES:
        raise ValueError(f"sampling must be one of {SAMPLING_POLICIES}")
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)
    if np.any(labels < 0) or np.any(labels >= weights.shape[1]):
        raise ValueError("labels must index an output class")
    if update_scale is not None and update_scale.shape != weights.shape:
        raise ValueError("update_scale must have the same shape as weights")
    if null_basis is not None and null_basis.shape[0] != weights.shape[0]:
        raise ValueError("null_basis must have one row per feature")
    if null_groups is not None:
        if null_basis is not None:
            raise ValueError("null_groups and null_basis cannot be combined")
        for columns, basis in null_groups:
            if not isinstance(columns, slice) or basis.shape[0] != weights.shape[0]:
                raise ValueError(
                    "null_groups need (slice, basis with one row per feature)"
                )
    if feature_scale is not None:
        feature_scale = np.asarray(feature_scale, dtype=weights.dtype)
        if feature_scale.shape != (weights.shape[0],) or not np.all(feature_scale > 0):
            raise ValueError(
                "feature_scale must be positive with one entry per feature"
            )
    if group_scale is not None:
        group_scale = np.asarray(group_scale, dtype=weights.dtype)
        if (
            group_scale.shape != (weights.shape[0],)
            or not np.all(np.isfinite(group_scale))
            or np.any(group_scale < 0)
        ):
            raise ValueError(
                "group_scale must be finite and non-negative with one entry per feature"
            )
        if not np.any(group_scale > 0):
            raise ValueError("group_scale must have at least one positive entry")
    if group_index is not None and group_index.shape != (weights.shape[0],):
        raise ValueError("group_index must have one entry per feature")
    step_scale = None
    if feature_scale is not None or group_scale is not None:
        step_scale = np.ones(weights.shape[0], dtype=weights.dtype)
        if feature_scale is not None:
            step_scale = step_scale * feature_scale
        if group_scale is not None:
            step_scale = step_scale * group_scale
        scale_root = np.sqrt(step_scale)[:, None]
    if eligible_classes is None:
        eligible_classes = np.arange(weights.shape[1], dtype=np.int64)
    else:
        eligible_classes = np.asarray(eligible_classes, dtype=np.int64)
        if (
            eligible_classes.ndim != 1
            or len(eligible_classes) == 0
            or np.any(eligible_classes < 0)
            or np.any(eligible_classes >= weights.shape[1])
            or np.any(np.diff(eligible_classes) <= 0)
        ):
            raise ValueError("eligible_classes must be sorted, unique, and in range")
        if not np.isin(labels, eligible_classes).all():
            raise ValueError("labels must belong to eligible_classes")

    label_columns = np.searchsorted(eligible_classes, labels)
    rng = np.random.default_rng(seed)
    n_updates_per_epoch = -(-len(labels) // batch_size)
    drawn = np.zeros(weights.shape[1], dtype=np.int64)
    n_groups = 0 if group_index is None else int(group_index.max()) + 1
    grad_l1, step_l1 = np.zeros(n_groups), np.zeros(n_groups)
    n_updates = 0

    def batches():
        for _ in range(epochs):
            if sampling == "shuffled":
                order = rng.permutation(len(labels))
                for start in range(0, len(labels), batch_size):
                    yield order[start : start + batch_size]
            else:
                yield from _balanced_batches(
                    label_columns, n_updates_per_epoch, batch_size, rng
                )

    for batch in batches():
        xb, yb = features[batch], label_columns[batch]
        drawn += np.bincount(eligible_classes[yb], minlength=weights.shape[1])
        probabilities = softmax(xb @ weights[:, eligible_classes])
        probabilities[np.arange(len(yb)), yb] -= 1.0
        gradient = xb.T @ probabilities / len(yb)
        raw_gradient = gradient
        if update_scale is not None:
            gradient = gradient * update_scale[:, eligible_classes]
        if null_groups is not None:
            masked = gradient
            gradient = (
                masked.copy() if step_scale is None else masked * step_scale[:, None]
            )
            for columns, basis in null_groups:
                if step_scale is None:
                    gradient[:, columns] = basis @ (basis.T @ masked[:, columns])
                else:
                    gradient[:, columns] = scale_root * (
                        basis @ (basis.T @ (scale_root * masked[:, columns]))
                    )
        elif step_scale is None:
            if null_basis is not None:
                gradient = null_basis @ (null_basis.T @ gradient)
        elif null_basis is None:
            gradient = gradient * step_scale[:, None]
        else:
            gradient = scale_root * (
                null_basis @ (null_basis.T @ (scale_root * gradient))
            )
        step = learning_rate * gradient
        weights[:, eligible_classes] -= step
        if group_index is not None:
            grad_l1 += np.bincount(
                group_index, np.abs(raw_gradient).sum(axis=1), n_groups
            )
            step_l1 += np.bincount(group_index, np.abs(step).sum(axis=1), n_groups)
        n_updates += 1
    if stats is not None:
        stats["samples_drawn_by_class"] = drawn.tolist()
        stats["n_updates"] = n_updates
        # budget accounting: shuffled processes exactly epochs x rows (its last batch can be
        # short); class_balanced always fills every batch, so it can present slightly more
        stats["examples_processed"] = int(drawn.sum())
        stats["examples_available"] = int(epochs * len(labels))
        stats["mean_batch_composition"] = (drawn / n_updates).tolist()
        stats["pool_sizes_by_class"] = np.bincount(
            eligible_classes[label_columns], minlength=weights.shape[1]
        ).tolist()
        if group_index is not None:
            stats["mean_gradient_l1_by_group"] = (grad_l1 / n_updates).tolist()
            stats["mean_step_l1_by_group"] = (step_l1 / n_updates).tolist()
    return weights


def null_space_basis(design: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis (n_features, k) of {v : design @ v = 0}; k = 0 if nothing is free."""
    design = np.asarray(design, dtype=np.float64)
    if design.shape[0] == 0:
        return np.eye(design.shape[1], dtype=np.float32)
    _, singular, vt = np.linalg.svd(design, full_matrices=True)
    rank = int(np.sum(singular > rtol * max(singular[0], 1.0)))
    return vt[rank:].T.astype(np.float32)


def cross_entropy(weights, features, labels, n_classes):
    """Mean softmax cross-entropy when only the first ``n_classes`` outputs compete."""
    probabilities = softmax(features @ weights[:, :n_classes])
    return float(
        -np.mean(np.log(probabilities[np.arange(len(labels)), labels] + 1e-12))
    )
