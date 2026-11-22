# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Generate ``c2_multiclass_fit_task.json`` and ``c2_multiclass_run_condition.json``.

These fixtures are the *former* behaviour of the multiclass trainer: they must be produced by
a core checkout from before the stable integration (C1), where ``fit_task`` and
``update_restrictions`` still contain the inline null-space mathematics. The script refuses to
run on code that already delegates to ``stable_cl``. Usage, from a checkout of
``eba751ee4a28a9ff4946b0f243da048a4cee104b`` (core ``main`` before C1)::

    PYTHONPATH=<checkout>/src python tests/golden/generate_c2_multiclass.py \\
        --legacy-commit eba751ee4a28a9ff4946b0f243da048a4cee104b --out tests/golden

Every input (features, labels, mini-batch index lists, weights, rate vectors, task plan) is
stored in the file, so the tests do not depend on any random number stream; the outputs are
the legacy results. Nothing here trains on real data or runs a benchmark.
"""

import argparse
import inspect
import json
import platform
import sys
from pathlib import Path

import numpy as np

NUM_CLASSES = 10
# COUNT < 10, the number of features of a degree-3 polynomial of two inputs (expansion)
COUNT, OLD_ROWS, OLD_CLASSES, N_SEEN, TASK = 6, 3, 2, 4, 1
N_REF_PER_CLASS = 1
STRATEGIES = ["ordinary", "frozen_old", "small_old_lr", "strategy_b", "strategy_b_new"]
FAMILIES = {  # name -> (kind, learning rate / per-degree rate vector)
    "shared": ("shared", 0.3),
    "feature_power": ("feature_power", 0.3),
    "per_degree": ("per_degree", [0.4, 0.2, 0.1, 0.05]),
    # degree 1 is frozen by its rate 0
    "per_degree_zero": ("per_degree", [0.4, 0.0, 0.1, 0.05]),
}
FIT_TASK_WHAT = "fit_task update tensors and update_restrictions, legacy values"
RUN_CONDITION_WHAT = "run_condition records on a tiny synthetic problem, legacy values"


def refuse_current_code():
    import polynomial_cl.training.multiclass as multiclass

    if "stable_cl" in inspect.getsource(multiclass):
        sys.exit(
            "this checkout already delegates to stable_cl; the multiclass fixtures must be "
            "generated from the inline implementation that preceded it"
        )


def listed(array):
    return None if array is None else np.asarray(array).tolist()


def projector(basis):
    basis = np.asarray(basis, dtype=np.float64)
    return (basis @ basis.T).tolist()


def projected(basis):
    return None if basis is None else projector(basis)


def group_records(null_groups, with_shape):
    if null_groups is None:
        return None
    records = []
    for columns, basis in null_groups:
        record = {
            "start": columns.start,
            "stop": columns.stop,
            "projector": projector(basis),
        }
        if with_shape:
            record["shape"] = list(basis.shape)
        records.append(record)
    return records


def memory_record(memory):
    return {str(k): np.asarray(v).tolist() for k, v in memory.items()}


class ReplayBatches:
    """epoch_sampler that replays stored index batches, one list per epoch."""

    def __init__(self, epochs):
        self.epochs, self.calls = epochs, 0

    def __call__(self, labels, n_updates, batch_size, rng):
        batches = self.epochs[self.calls]
        self.calls += 1
        yield from (np.asarray(b, dtype=np.int64) for b in batches)


def build_problem():
    from polynomial_cl.continual import make_feature_data, make_task_plan
    from polynomial_cl.continual.replay import stored_indices
    from polynomial_cl.data import make_synthetic_2d

    x_train, y_train, x_test, y_test, _ = make_synthetic_2d(
        9, n_train_per_class=12, n_test_per_class=6, separation=3.0, noise=0.2
    )
    data = make_feature_data(x_train, y_train, x_test, y_test)
    tasks, memory = make_task_plan(data.train_y, 0, 5)
    return data, tasks, memory, stored_indices(memory, OLD_CLASSES)


def effective_rates(kind, value, group_index, stacked):
    """feature_scale, group_scale, base rate - the experiment's rate construction."""
    from polynomial_cl.continual.degree_rates import step_scale_for

    feature_scale = group_scale = None
    base_rate = value
    if kind == "feature_power":
        rms = np.sqrt(np.mean(stacked.astype(np.float64) ** 2, axis=0))
        feature_scale = ((rms.max() / np.maximum(rms, 1e-12)) ** 0.5).astype(np.float32)
    elif kind == "per_degree":
        base_rate, group_scale = step_scale_for(np.array(value), group_index)
    return feature_scale, group_scale, float(base_rate)


def step_scale_of(feature_scale, group_scale):
    if group_scale is None:
        return feature_scale
    return group_scale if feature_scale is None else feature_scale * group_scale


def fit_case_fixture(strategy, family, problem, batch_plans):
    from polynomial_cl.continual.degree_rates import degree_of_feature
    from polynomial_cl.continual.strategies import update_restrictions
    from polynomial_cl.training.multiclass import fit_task

    data, tasks, memory, replay, weights0 = problem
    indices = tasks[TASK]
    features = data.train_x[indices, :COUNT]
    labels = data.train_y[indices].astype(np.int64)
    replay_x = data.train_x[replay, :COUNT]
    replay_y = data.train_y[replay].astype(np.int64)
    stacked = np.concatenate([features, replay_x])
    group_index = degree_of_feature(2, 3)[:COUNT]
    kind, value = FAMILIES[family]
    feature_scale, group_scale, base_rate = effective_rates(
        kind, value, group_index, stacked
    )
    step_scale = step_scale_of(feature_scale, group_scale)
    restrictions = update_restrictions(
        strategy,
        weights_shape=weights0.shape,
        old_rows=OLD_ROWS,
        old_classes=OLD_CLASSES,
        n_seen=N_SEEN,
        count=COUNT,
        task=TASK,
        train_x=data.train_x,
        memory_indices=memory,
        n_ref_per_class=N_REF_PER_CLASS,
        step_scale=step_scale,
    )
    scale, null_basis, refs, null_groups = restrictions
    runs = {}
    for label, epochs in batch_plans.items():
        weights = weights0.copy()
        out = fit_task(
            weights,
            features,
            labels,
            replay_x,
            replay_y,
            epochs=len(epochs),
            learning_rate=base_rate,
            batch_size=12,
            seed=0,
            eligible_classes=np.arange(N_SEEN),
            update_scale=scale,
            null_basis=null_basis,
            null_groups=null_groups,
            feature_scale=feature_scale,
            group_scale=group_scale,
            epoch_sampler=ReplayBatches(epochs),
        )
        runs[label] = {
            "batches": [[b.tolist() for b in epoch] for epoch in epochs],
            "weights_after": out.tolist(),
        }
    return {
        "name": f"{strategy}/{family}",
        "strategy": strategy,
        "family": family,
        "base_rate": base_rate,
        "feature_scale": listed(feature_scale),
        "group_scale": listed(group_scale),
        "restrictions": {
            "scale": listed(scale),
            "refs": listed(refs),
            "null_basis_projector": projected(null_basis),
            "null_basis_shape": None if null_basis is None else list(null_basis.shape),
            "null_groups": group_records(null_groups, with_shape=True),
        },
        "runs": runs,
    }


def noop_fixture(strategy, problem):
    """update_restrictions for an experience whose active groups all have rate 0."""
    from polynomial_cl.continual.strategies import update_restrictions

    data, tasks, memory, replay, weights0 = problem
    zeros = np.zeros(COUNT, np.float32)  # what run_condition passes: base_rate = 1.0
    scale, null_basis, refs, null_groups = update_restrictions(
        strategy,
        weights_shape=weights0.shape,
        old_rows=OLD_ROWS,
        old_classes=OLD_CLASSES,
        n_seen=N_SEEN,
        count=COUNT,
        task=TASK,
        train_x=data.train_x,
        memory_indices=memory,
        n_ref_per_class=N_REF_PER_CLASS,
        step_scale=zeros,
    )
    return {
        "strategy": strategy,
        "scale": listed(scale),
        "refs": listed(refs),
        "null_basis_projector": projected(null_basis),
        "null_groups": group_records(null_groups, with_shape=False),
    }


def sigma_min_over_cases(problem):
    """Smallest sigma_1 of the scaled reference design over every strategy_b case."""
    from polynomial_cl.continual.degree_rates import degree_of_feature
    from polynomial_cl.continual.replay import stored_indices

    data, tasks, memory, replay, _ = problem
    group_index = degree_of_feature(2, 3)[:COUNT]
    stacked = data.train_x[np.concatenate([tasks[TASK], replay]), :COUNT]
    refs = stored_indices(memory, OLD_CLASSES, N_REF_PER_CLASS)
    sigmas = []
    for kind, value in FAMILIES.values():
        feature_scale, group_scale, _ = effective_rates(
            kind, value, group_index, stacked
        )
        scale = step_scale_of(feature_scale, group_scale)
        design = data.train_x[refs, :COUNT].astype(np.float64)
        if scale is not None:
            design = design * np.sqrt(scale.astype(np.float64))
        sigmas.append(float(np.linalg.svd(design, compute_uv=False)[0]))
    return min(sigmas)


def run_condition_fixture(problem_raw):
    from polynomial_cl.continual import make_feature_data, run_condition

    x_train, y_train, x_test, y_test, tasks, memory = problem_raw
    data = make_feature_data(x_train, y_train, x_test, y_test)
    configs = [
        ("ordinary", "ordinary", {"learning_rate": 0.3}),
        ("ordinary", "feature_power", {"learning_rate": 0.3, "feature_power": 0.5}),
        ("ordinary", "per_degree", {"learning_rate": [0.4, 0.2, 0.1, 0.05]}),
        (
            "ordinary",
            "per_degree_zero_group",
            {
                "learning_rate": [0.4, 0.2, 0.1, 0.05],
                "degree_control_factors": [1, 0, 1, 1],
            },
        ),
        (
            "ordinary",
            "noop_first_experience",
            {
                "learning_rate": [0.4, 0.2, 0.1, 0.05],
                "degree_control_factors": [0, 0, 1, 1],
            },
        ),
        ("strategy_b", "shared", {"learning_rate": 0.3}),
        ("strategy_b", "feature_power", {"learning_rate": 0.3, "feature_power": 0.5}),
        ("strategy_b", "per_degree", {"learning_rate": [0.4, 0.2, 0.1, 0.05]}),
        (
            "strategy_b",
            "noop_first_experience",
            {
                "learning_rate": [0.4, 0.2, 0.1, 0.05],
                "degree_control_factors": [0, 0, 1, 1],
            },
        ),
        ("strategy_b_new", "shared", {"learning_rate": 0.3}),
        ("strategy_b_new", "per_degree", {"learning_rate": [0.4, 0.2, 0.1, 0.05]}),
        ("frozen_old", "per_degree", {"learning_rate": [0.4, 0.2, 0.1, 0.05]}),
        ("small_old_lr", "shared", {"learning_rate": 0.3}),
    ]
    runs = []
    for strategy, label, kwargs in configs:
        record = run_condition(
            "expanding",
            data,
            tasks,
            memory,
            seed=2,
            epochs=2,
            # at least the rows of every experience: one full-batch update per epoch
            batch_size=64,
            strategy=strategy,
            n_ref_per_class=1,
            diagnostics=True,
            **kwargs,
        )
        runs.append(
            {"strategy": strategy, "label": label, "kwargs": kwargs, "record": record}
        )
    return runs


def meta(args, what):
    return {
        "schema": 1,
        "what": what,
        "generated_from": {
            "repository": "kobros-tech/polynomial_cl_core",
            "commit": args.legacy_commit,
            "state": "before the stable integration (C1): inline null-space mathematics",
        },
        "generator": "tests/golden/generate_c2_multiclass.py",
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def dump(path, payload):
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--legacy-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    refuse_current_code()

    data, tasks, memory, replay = build_problem()
    rng = np.random.default_rng(11)
    weights0 = np.zeros((COUNT, NUM_CLASSES), np.float32)
    weights0[:OLD_ROWS, :OLD_CLASSES] = (0.3 * rng.normal(size=(OLD_ROWS, 2))).astype(
        np.float32
    )  # coefficients learned earlier; the rows added by expansion are zero
    problem = (data, tasks, memory, replay, weights0)
    n_rows = len(tasks[TASK]) + len(replay)
    perms = [rng.permutation(n_rows) for _ in range(2)]
    batch_plans = {
        "single": [[np.arange(n_rows)]],
        # 6 updates; the last batch of each epoch is short
        "multi": [[p[:12], p[12:24], p[24:]] for p in perms],
    }
    sigma_min = sigma_min_over_cases(problem)
    if sigma_min < 1.0:
        sys.exit(
            f"sigma_1 {sigma_min} < 1: the legacy and relative tolerances would differ"
        )
    cases = [
        fit_case_fixture(strategy, family, problem, batch_plans)
        for strategy in STRATEGIES
        for family in FAMILIES
    ]
    payload = {
        "meta": meta(args, FIT_TASK_WHAT)
        | {"min_sigma1_of_reference_design": sigma_min},
        "problem": {
            "count": COUNT,
            "old_rows": OLD_ROWS,
            "old_classes": OLD_CLASSES,
            "n_seen": N_SEEN,
            "task": TASK,
            "n_ref_per_class": N_REF_PER_CLASS,
            "num_classes": NUM_CLASSES,
            "train_x": data.train_x.tolist(),
            "train_y": data.train_y.tolist(),
            "task_indices": [t.tolist() for t in tasks],
            "memory_indices": memory_record(memory),
            "replay_indices": replay.tolist(),
            "weights0": weights0.tolist(),
        },
        "cases": cases,
        "noop": [noop_fixture(s, problem) for s in STRATEGIES],
    }
    dump(args.out / "c2_multiclass_fit_task.json", payload)

    from polynomial_cl.data import make_synthetic_2d

    x_train, y_train, x_test, y_test, _ = make_synthetic_2d(
        9, n_train_per_class=12, n_test_per_class=6, separation=3.0, noise=0.2
    )
    runs = run_condition_fixture((x_train, y_train, x_test, y_test, tasks, memory))
    payload = {
        "meta": meta(args, RUN_CONDITION_WHAT),
        "problem": {
            "x_train": x_train.tolist(),
            "y_train": y_train.tolist(),
            "x_test": x_test.tolist(),
            "y_test": y_test.tolist(),
            "task_indices": [np.asarray(t).tolist() for t in tasks],
            "memory_indices": memory_record(memory),
            "settings": {
                "condition": "expanding",
                "seed": 2,
                "epochs": 2,
                "batch_size": 64,
                "n_ref_per_class": 1,
            },
        },
        "runs": runs,
    }
    dump(args.out / "c2_multiclass_run_condition.json", payload)


if __name__ == "__main__":
    main()
