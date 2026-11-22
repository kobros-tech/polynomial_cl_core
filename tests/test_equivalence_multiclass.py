# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""C2: the integrated multiclass trainer reproduces the former inline implementation.

The golden files in ``tests/golden`` hold what core computed *before* it delegated to
``stable_cl`` (see ``generate_c2_multiclass.py``): update tensors of ``fit_task`` after one and
after six mini-batch updates, the restrictions of ``update_restrictions`` and complete
``run_condition`` records. Their inputs are stored, so no random stream is involved: mini-batch
order is replayed through ``epoch_sampler``.

Every case uses a learning rate different from 1 and non-trivial rate multipliers, so a rate
vector without the learning rate, or with it applied twice, changes the result; the negative
controls below show that this is detected. Tolerances are float32 round-off (the trainer
computes the step in float64, the former code in float32); the measured maxima are noted at
each constant and are far below the size of any real defect.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import stable_cl

from polynomial_cl.continual import make_feature_data, run_condition
from polynomial_cl.continual.strategies import update_restrictions
from polynomial_cl.models.multiclass import softmax
from polynomial_cl.training.multiclass import fit_task

GOLDEN = Path(__file__).parent / "golden"
LEGACY_COMMIT = "eba751ee4a28a9ff4946b0f243da048a4cee104b"

# Each tolerance is about ten times the largest deviation measured over the fixtures.
TOL_WEIGHTS = 1e-6  # |W_current - W_legacy| after 1 and 6 updates; measured 8.9e-08
TOL_PROJECTOR = 1e-6  # float32 projector entries; measured 7.3e-08
TOL_REFERENCE = 1e-6  # float64 stable reference vs float32 legacy; measured 3.1e-08
TOL_RECORD = 3e-5  # numbers in run_condition records; measured 2.9e-06
# A wrong learning rate must deviate by at least this many tolerances; the smallest deviation
# measured over the fixtures is 4.3e-02, i.e. more than 4000 tolerances.
CONTROL_FACTOR = 100


def load(name):
    return json.loads((GOLDEN / name).read_text())


FIT = load("c2_multiclass_fit_task.json")
RUN = load("c2_multiclass_run_condition.json")
PROBLEM = FIT["problem"]
COUNT = PROBLEM["count"]
OLD_ROWS, OLD_CLASSES = PROBLEM["old_rows"], PROBLEM["old_classes"]
N_SEEN, TASK = PROBLEM["n_seen"], PROBLEM["task"]
TRAIN_X = np.asarray(PROBLEM["train_x"], dtype=np.float32)
TRAIN_Y = np.asarray(PROBLEM["train_y"], dtype=np.int64)
TASKS = [np.asarray(t, dtype=np.int64) for t in PROBLEM["task_indices"]]


def memory_from(record):
    return {int(k): np.asarray(v, dtype=np.int64) for k, v in record.items()}


MEMORY = memory_from(PROBLEM["memory_indices"])
REPLAY = np.asarray(PROBLEM["replay_indices"], dtype=np.int64)
WEIGHTS0 = np.asarray(PROBLEM["weights0"], dtype=np.float32)
FEATURES = TRAIN_X[TASKS[TASK], :COUNT]
LABELS = TRAIN_Y[TASKS[TASK]]
REPLAY_X, REPLAY_Y = TRAIN_X[REPLAY, :COUNT], TRAIN_Y[REPLAY]
CASES = {case["name"]: case for case in FIT["cases"]}
PLANS = ["single", "multi"]


class ReplayBatches:
    """epoch_sampler replaying the stored index batches (one list per epoch)."""

    def __init__(self, epochs):
        self.epochs, self.calls = epochs, 0

    def __call__(self, labels, n_updates, batch_size, rng):
        batches = self.epochs[self.calls]
        self.calls += 1
        yield from (np.asarray(b, dtype=np.int64) for b in batches)


def arrays_of(case):
    def as32(values):
        return None if values is None else np.asarray(values, dtype=np.float32)

    return as32(case["feature_scale"]), as32(case["group_scale"]), case["base_rate"]


def step_scale_of(feature_scale, group_scale):
    """The experiment's combination of the two multipliers."""
    if group_scale is None:
        return feature_scale
    return group_scale if feature_scale is None else feature_scale * group_scale


def restrictions_of(case, step_scale=None, base_rate=None):
    feature_scale, group_scale, rate = arrays_of(case)
    if step_scale is None:
        step_scale = step_scale_of(feature_scale, group_scale)
    return update_restrictions(
        case["strategy"],
        weights_shape=WEIGHTS0.shape,
        old_rows=OLD_ROWS,
        old_classes=OLD_CLASSES,
        n_seen=N_SEEN,
        count=COUNT,
        task=TASK,
        train_x=TRAIN_X,
        memory_indices=MEMORY,
        n_ref_per_class=PROBLEM["n_ref_per_class"],
        step_scale=step_scale,
        base_rate=rate if base_rate is None else base_rate,
    )


def train(case, plan, learning_rate=None):
    """Production path: update_restrictions then fit_task on the stored batches."""
    feature_scale, group_scale, rate = arrays_of(case)
    scale, null_basis, _, null_groups = restrictions_of(case)
    epochs = case["runs"][plan]["batches"]
    return fit_task(
        WEIGHTS0.copy(),
        FEATURES,
        LABELS,
        REPLAY_X,
        REPLAY_Y,
        epochs=len(epochs),
        learning_rate=rate if learning_rate is None else learning_rate,
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


def legacy_after(case, plan):
    return np.asarray(case["runs"][plan]["weights_after"], dtype=np.float32)


def update_tensor(after):
    return WEIGHTS0.astype(np.float64) - after.astype(np.float64)


def deviation(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


ALL = [(name, plan) for name in CASES for plan in PLANS]


# ------------------------------------------------------------------ fixture provenance
def test_golden_files_come_from_the_inline_implementation():
    for payload in (FIT, RUN):
        assert payload["meta"]["schema"] == 1
        assert payload["meta"]["generated_from"]["commit"] == LEGACY_COMMIT
        assert "inline" in payload["meta"]["generated_from"]["state"]
    # equal thresholds: the legacy max(sigma_1, 1) rule equals the relative rule only if sigma_1 >= 1
    assert FIT["meta"]["min_sigma1_of_reference_design"] >= 1.0


def test_fixture_cases_cover_every_strategy_rate_family_and_plan():
    strategies = {c["strategy"] for c in CASES.values()}
    families = {c["family"] for c in CASES.values()}
    assert strategies == {
        "ordinary",
        "frozen_old",
        "small_old_lr",
        "strategy_b",
        "strategy_b_new",
    }
    assert families == {"shared", "feature_power", "per_degree", "per_degree_zero"}
    assert COUNT < 10 and OLD_ROWS < COUNT  # expansion: fewer active rows than features
    for case in CASES.values():
        # a missing or doubled rate must show
        assert case["base_rate"] not in (0.0, 1.0)
        assert deviation(update_tensor(legacy_after(case, "single")), 0.0) > 1e-3


# ------------------------------------------------------------------ update tensors
@pytest.mark.parametrize("name, plan", ALL)
def test_update_tensor_equals_the_former_implementation(name, plan):
    case = CASES[name]
    expected = update_tensor(legacy_after(case, plan))
    actual = update_tensor(train(case, plan))
    assert deviation(actual, expected) <= TOL_WEIGHTS


@pytest.mark.parametrize("name", [n for n in CASES if "per_degree_zero" in n])
def test_rows_of_a_frozen_degree_do_not_move(name):
    case = CASES[name]
    zero_rows = np.flatnonzero(np.asarray(case["group_scale"]) == 0)
    assert len(zero_rows) and np.any(np.asarray(case["group_scale"]) > 0)
    for plan in PLANS:
        np.testing.assert_array_equal(train(case, plan)[zero_rows], WEIGHTS0[zero_rows])


@pytest.mark.parametrize(
    "name", [n for n in CASES if n.startswith(("frozen_old", "strategy_b_new"))]
)
def test_old_by_old_block_stays_exactly_unchanged(name):
    after = train(CASES[name], "multi")
    np.testing.assert_array_equal(
        after[:OLD_ROWS, :OLD_CLASSES], WEIGHTS0[:OLD_ROWS, :OLD_CLASSES]
    )


# ------------------------------------------------------------------ restrictions and bases
def projector(basis):
    basis = np.asarray(basis, dtype=np.float64)
    return basis @ basis.T


@pytest.mark.parametrize("name", list(CASES))
def test_update_restrictions_equal_the_former_ones(name):
    case = CASES[name]
    expected = case["restrictions"]
    scale, null_basis, refs, null_groups = restrictions_of(case)
    for got, want in ((scale, expected["scale"]), (refs, expected["refs"])):
        assert (got is None) == (want is None)
        if got is not None:
            np.testing.assert_array_equal(got, np.asarray(want))
    if expected["null_basis_projector"] is None:
        assert null_basis is None
    else:
        assert list(null_basis.shape) == expected["null_basis_shape"]
        assert null_basis.dtype == np.float32
        wanted = np.asarray(expected["null_basis_projector"])
        assert deviation(projector(null_basis), wanted) <= TOL_PROJECTOR
    if expected["null_groups"] is None:
        assert null_groups is None
    else:
        assert len(null_groups) == len(expected["null_groups"])
        for (columns, basis), want in zip(null_groups, expected["null_groups"]):
            assert (columns.start, columns.stop) == (want["start"], want["stop"])
            assert list(basis.shape) == want["shape"]
            wanted = np.asarray(want["projector"])
            assert deviation(projector(basis), wanted) <= TOL_PROJECTOR


@pytest.mark.parametrize("entry", FIT["noop"], ids=lambda e: e["strategy"])
def test_all_zero_experience_restrictions_equal_the_former_ones(entry):
    # run_condition passes base_rate = 1.0 and step_scale = 0 when degree control zeroes every
    # active group; the experience is not trained, but its restrictions are still built.
    scale, null_basis, refs, null_groups = update_restrictions(
        entry["strategy"],
        weights_shape=WEIGHTS0.shape,
        old_rows=OLD_ROWS,
        old_classes=OLD_CLASSES,
        n_seen=N_SEEN,
        count=COUNT,
        task=TASK,
        train_x=TRAIN_X,
        memory_indices=MEMORY,
        n_ref_per_class=PROBLEM["n_ref_per_class"],
        step_scale=np.zeros(COUNT, np.float32),
        base_rate=1.0,
    )
    assert (scale is None) == (entry["scale"] is None)
    if entry["null_basis_projector"] is not None:
        wanted = np.asarray(entry["null_basis_projector"])
        assert deviation(projector(null_basis), wanted) <= TOL_PROJECTOR
        # nothing is constrained
        assert deviation(projector(null_basis), np.eye(COUNT)) <= TOL_PROJECTOR
    if entry["null_groups"] is not None:
        for (_columns, basis), want in zip(null_groups, entry["null_groups"]):
            wanted = np.asarray(want["projector"])
            assert deviation(projector(basis), wanted) <= TOL_PROJECTOR


def test_fit_task_refuses_a_group_scale_that_is_zero_everywhere():
    with pytest.raises(ValueError, match="positive"):
        fit_task(
            WEIGHTS0.copy(),
            FEATURES,
            LABELS,
            REPLAY_X,
            REPLAY_Y,
            epochs=1,
            learning_rate=1.0,
            batch_size=12,
            seed=0,
            eligible_classes=np.arange(N_SEEN),
            group_scale=np.zeros(COUNT, np.float32),
        )


# ------------------------------------------------------------------ independent reference
def one_step_reference(case, rates):
    """One update from ``stable_cl`` alone, for the stored single batch and effective ``rates``.

    Only the stored inputs and the stored legacy restrictions' *indices* are used; the step is
    ``ordinary_step`` / ``relaxed_step`` with the diagonal ``rates`` (float64).
    """
    batch = np.asarray(case["runs"]["single"]["batches"][0][0])
    stacked = np.concatenate([FEATURES, REPLAY_X])[batch]
    labels = np.concatenate([LABELS, REPLAY_Y])[batch]
    eligible = np.arange(N_SEEN)
    probabilities = softmax(stacked @ WEIGHTS0[:, eligible])
    probabilities[np.arange(len(labels)), labels] -= 1.0
    gradient = (stacked.T @ probabilities / len(labels)).astype(np.float64)
    expected = case["restrictions"]
    if expected["scale"] is not None:
        gradient = gradient * np.asarray(expected["scale"], np.float64)[:, eligible]
    strategy = case["strategy"]
    if strategy in ("strategy_b", "strategy_b_new"):
        refs = np.asarray(expected["refs"])
        design = TRAIN_X[refs, :COUNT].astype(np.float64)
        if strategy == "strategy_b":
            basis = stable_cl.null_basis(design, rates)
            step = stable_cl.relaxed_step(gradient, basis, rates, 0.0)
        else:
            rows = np.arange(OLD_ROWS, COUNT)
            rows_basis = stable_cl.null_basis(design[:, rows], rates[rows])
            old = np.zeros((COUNT, rows_basis.shape[1]))
            old[rows] = rows_basis
            full = stable_cl.null_basis(design, rates)
            step = np.empty_like(gradient)
            step[:, :OLD_CLASSES] = stable_cl.relaxed_step(
                gradient[:, :OLD_CLASSES], old, rates, 0.0
            )
            step[:, OLD_CLASSES:] = stable_cl.relaxed_step(
                gradient[:, OLD_CLASSES:], full, rates, 0.0
            )
    else:
        step = stable_cl.ordinary_step(gradient, rates)
    return step


def effective_rates(case):
    feature_scale, group_scale, rate = arrays_of(case)
    scale = step_scale_of(feature_scale, group_scale)
    ones = np.ones(COUNT) if scale is None else scale.astype(np.float64)
    # d = learning rate * (feature rate multiplier * degree multiplier)
    return rate * ones


@pytest.mark.parametrize("name", list(CASES))
def test_stable_reference_with_effective_rates_equals_the_former_update(name):
    case = CASES[name]
    step = one_step_reference(case, effective_rates(case))
    legacy = update_tensor(legacy_after(case, "single"))
    eligible = np.arange(N_SEEN)
    assert deviation(step, legacy[:, eligible]) <= TOL_REFERENCE


@pytest.mark.parametrize("mode", ["omit", "double"])
@pytest.mark.parametrize("name", list(CASES))
def test_negative_control_wrong_learning_rate_in_the_reference_is_detected(name, mode):
    case = CASES[name]
    rate = case["base_rate"]
    rates = effective_rates(case)
    # "omit": no learning rate; "double": the learning rate applied twice
    wrong = rates / rate if mode == "omit" else rates * rate
    step = one_step_reference(case, wrong)
    legacy = update_tensor(legacy_after(case, "single"))[:, :N_SEEN]
    assert deviation(step, legacy) > CONTROL_FACTOR * TOL_REFERENCE


@pytest.mark.parametrize("mode", ["omit", "double"])
@pytest.mark.parametrize("name, plan", ALL)
def test_negative_control_wrong_learning_rate_at_the_call_site_is_detected(
    name, plan, mode
):
    case = CASES[name]
    rate = case["base_rate"]
    wrong = 1.0 if mode == "omit" else rate * rate
    expected = update_tensor(legacy_after(case, plan))
    actual = update_tensor(train(case, plan, learning_rate=wrong))
    assert deviation(actual, expected) > CONTROL_FACTOR * TOL_WEIGHTS


@pytest.mark.parametrize("mode", ["omit", "double"])
@pytest.mark.parametrize("name", [n for n in CASES if n.startswith("strategy_b")])
def test_negative_control_wrong_rates_inside_the_stable_call_are_detected(
    name, mode, monkeypatch
):
    case = CASES[name]
    rate = case["base_rate"]
    real = stable_cl.relaxed_step

    def mutated(gradient, basis, rates, beta):
        wrong = rates / rate if mode == "omit" else rates * rate
        return real(gradient, basis, wrong, beta)

    monkeypatch.setattr(stable_cl, "relaxed_step", mutated)
    expected = update_tensor(legacy_after(case, "single"))
    actual = update_tensor(train(case, "single"))
    assert deviation(actual, expected) > CONTROL_FACTOR * TOL_WEIGHTS


# ------------------------------------------------------------------ run_condition records
def rebuild_run_inputs():
    problem = RUN["problem"]
    data = make_feature_data(
        np.asarray(problem["x_train"]),
        np.asarray(problem["y_train"]),
        np.asarray(problem["x_test"]),
        np.asarray(problem["y_test"]),
    )
    tasks = [np.asarray(t, dtype=np.int64) for t in problem["task_indices"]]
    memory = memory_from(problem["memory_indices"])
    return data, tasks, memory, problem["settings"]


RUN_INPUTS = rebuild_run_inputs()


def run_current(entry):
    data, tasks, memory, settings = RUN_INPUTS
    return run_condition(
        settings["condition"],
        data,
        tasks,
        memory,
        seed=settings["seed"],
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        strategy=entry["strategy"],
        n_ref_per_class=settings["n_ref_per_class"],
        diagnostics=True,
        **entry["kwargs"],
    )


def compare_records(actual, expected, path="record"):
    """Largest numeric deviation; structure, strings, integers and None must match exactly."""
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        worst = [
            compare_records(actual[key], value, f"{path}.{key}")
            for key, value in expected.items()
        ]
        return max(worst, default=0.0)
    if isinstance(expected, list):
        assert len(actual) == len(expected), path
        worst = [
            compare_records(a, e, f"{path}[{i}]")
            for i, (a, e) in enumerate(zip(actual, expected))
        ]
        return max(worst, default=0.0)
    if isinstance(expected, float):
        assert isinstance(actual, float) and math.isfinite(actual), path
        return abs(actual - expected)
    assert actual == expected, path
    return 0.0


RUNS = {f"{r['strategy']}/{r['label']}": r for r in RUN["runs"]}


@pytest.mark.parametrize("name", list(RUNS))
def test_run_condition_record_equals_the_former_implementation(name):
    entry = RUNS[name]
    assert compare_records(run_current(entry), entry["record"]) <= TOL_RECORD


@pytest.mark.parametrize("name", [n for n in RUNS if "noop" in n])
def test_all_zero_first_experience_leaves_the_weights_untouched(name):
    record = run_current(RUNS[name])
    loss_before, loss_after = record["train_loss_by_task"][0]
    assert loss_before == loss_after  # nothing was trained in experience 0
    assert record["effective_degree_learning_rates"][:2] == [0.0, 0.0]
    later = record["train_loss_by_task"][2]
    assert later[0] != later[1]  # the groups with a positive rate train once active


def test_run_condition_fixture_covers_the_rate_families_and_strategies():
    labels = {r["label"] for r in RUN["runs"]}
    assert {"shared", "feature_power", "per_degree", "per_degree_zero_group"} <= labels
    assert {"noop_first_experience"} <= labels
    assert {r["strategy"] for r in RUN["runs"]} == {
        "ordinary",
        "strategy_b",
        "strategy_b_new",
        "frozen_old",
        "small_old_lr",
    }
