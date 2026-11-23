# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""C1: the softmax trainer takes its mathematics from the ``stable_cl`` package.

What is checked here is the *integration*: which functions are called, with which effective
rates, in which order the update mask and the projection are applied, and that the result
equals the former inline implementation on small deterministic problems (``legacy_step`` below
is a copy of that implementation, kept only as a reference). It is not the numerical
equivalence fixture set; that is a separate stage.

Round-off: the former code computed in float32, the integrated code computes the step in
float64, so trained weights agree to float32 round-off, not bitwise (``ATOL``).
"""

import ast
import json
import re
import tomllib
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import stable_cl

from polynomial_cl.continual import strategies
from polynomial_cl.continual.degree_rates import (
    diagonal_learning_rates,
    step_scale_for,
)
from polynomial_cl.continual.strategies import update_restrictions
from polynomial_cl.models.multiclass import softmax
from polynomial_cl.training import multiclass
from polynomial_cl.training.multiclass import fit_task

ROOT = Path(__file__).parents[1]
STABLE_SHA = "b6dbd86c3fc76f888c5c69167ed09ac03cc34a5d"
ATOL = 1e-6  # float32 round-off of three full-batch steps; the largest measured is 3e-8

P, K_ALL, K_ELIGIBLE, N_ROWS, EPOCHS, LR = 8, 5, 4, 30, 3, 0.3


# ---------------------------------------------------------------- reference (former code)
def legacy_step(
    gradient,
    *,
    learning_rate,
    update_scale,
    eligible,
    null_basis,
    null_groups,
    step_scale,
    mask_first=True,
):
    """The former inline step of ``fit_task`` (float32), copied for comparison."""
    scale_root = None if step_scale is None else np.sqrt(step_scale)[:, None]
    if update_scale is not None and mask_first:
        gradient = gradient * update_scale[:, eligible]
    if null_groups is not None:
        masked = gradient
        gradient = masked.copy() if step_scale is None else masked * step_scale[:, None]
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
        gradient = scale_root * (null_basis @ (null_basis.T @ (scale_root * gradient)))
    if update_scale is not None and not mask_first:
        gradient = gradient * update_scale[:, eligible]
    return learning_rate * gradient


def legacy_fit(weights, features, labels, eligible, **restrictions):
    """Full-batch loop with the former step; ``weights`` is updated in place."""
    for _ in range(EPOCHS):
        probabilities = softmax(features @ weights[:, eligible])
        probabilities[np.arange(len(labels)), labels] -= 1.0
        gradient = features.T @ probabilities / len(labels)
        step = legacy_step(
            gradient, learning_rate=LR, eligible=eligible, **restrictions
        )
        weights[:, eligible] -= step
    return weights


# ---------------------------------------------------------------- problem and cases
@pytest.fixture
def problem():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(N_ROWS, P)).astype(np.float32)
    labels = rng.integers(0, K_ELIGIBLE, size=N_ROWS)
    weights = (0.1 * rng.normal(size=(P, K_ALL))).astype(np.float32)
    refs = rng.normal(size=(3, P))
    return features, labels, weights, refs


def scales():
    feature_scale = np.linspace(0.5, 2.0, P).astype(np.float32)
    # the last two feature rows have rate 0
    group_scale = np.array([1, 1, 0.5, 0.5, 0.25, 0.25, 0, 0], np.float32)
    return feature_scale, group_scale


def update_mask():
    mask = np.ones((P, K_ALL), np.float32)
    mask[:4, :2] = 0.0  # old x old frozen
    mask[5:, 2:4] = 0.1  # slowed
    return mask


def make_case(name, refs):
    """fit_task keyword arguments of one restriction combination."""
    feature_scale, group_scale = scales()
    step_scale = feature_scale * group_scale
    # exact tokens: "group" (a scale) is not "groups" (null groups)
    tokens = set(name.split("_"))
    case = {}
    if "feature" in tokens:
        case["feature_scale"] = feature_scale
    if "group" in tokens:
        case["group_scale"] = group_scale
    if "mask" in tokens:
        case["update_scale"] = update_mask()
    effective = None
    if {"feature", "group"} <= tokens:
        effective = step_scale
    elif "feature" in tokens:
        effective = feature_scale
    elif "group" in tokens:
        effective = group_scale
    rates = np.ones(P) if effective is None else effective.astype(np.float64)
    if "basis" in tokens:
        case["null_basis"] = stable_cl.null_basis(refs, rates).astype(np.float32)
    if "groups" in tokens:
        rows = np.arange(4, P)
        rows_basis = stable_cl.null_basis(refs[:, rows], rates[rows])
        padded = np.zeros((P, rows_basis.shape[1]))
        padded[rows] = rows_basis
        case["null_groups"] = [
            # old-class columns move through the new rows only
            (slice(0, 2), padded.astype(np.float32)),
            (slice(2, 3), stable_cl.null_basis(refs, rates).astype(np.float32)),
            # eligible column 3 lies outside every group: ordinary rate-weighted step
        ]
    return case, effective


CASES = [
    "ordinary",
    "feature",
    "group",
    "feature_group",
    "mask",
    "feature_group_mask",
    "basis",
    "basis_group",
    "basis_feature_group_mask",
    "basis_mask",
    "groups",
    "groups_feature_group",
    "groups_feature_group_mask",
]


@pytest.mark.parametrize("name", CASES)
def test_trained_weights_equal_the_former_inline_implementation(name, problem):
    features, labels, weights, refs = problem
    case, effective = make_case(name, refs)
    eligible = np.arange(K_ELIGIBLE)
    expected = legacy_fit(
        weights.copy(),
        features,
        labels,
        eligible,
        update_scale=case.get("update_scale"),
        null_basis=case.get("null_basis"),
        null_groups=case.get("null_groups"),
        step_scale=None if effective is None else effective.astype(np.float32),
    )
    actual = fit_task(
        weights.copy(),
        features,
        labels,
        np.empty((0, P), np.float32),
        np.empty(0, np.int64),
        epochs=EPOCHS,
        learning_rate=LR,
        batch_size=N_ROWS,
        seed=0,
        eligible_classes=eligible,
        **case,
    )
    assert not np.allclose(actual, weights)  # something was trained
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)


def test_the_update_mask_is_applied_before_the_projection(problem):
    # The tests above would not notice a swapped order if both orders agreed on the case.
    features, labels, weights, refs = problem
    case, _ = make_case("basis_mask", refs)
    eligible = np.arange(K_ELIGIBLE)
    restrictions = {
        "update_scale": case["update_scale"],
        "null_basis": case["null_basis"],
        "null_groups": None,
        "step_scale": None,
    }
    mask_first = legacy_fit(weights.copy(), features, labels, eligible, **restrictions)
    mask_last = legacy_fit(
        weights.copy(), features, labels, eligible, mask_first=False, **restrictions
    )
    assert np.abs(mask_first - mask_last).max() > 1e-3  # the orders differ here
    actual = fit_task(
        weights.copy(),
        features,
        labels,
        np.empty((0, P), np.float32),
        np.empty(0, np.int64),
        epochs=EPOCHS,
        learning_rate=LR,
        batch_size=N_ROWS,
        seed=0,
        eligible_classes=eligible,
        update_scale=case["update_scale"],
        null_basis=case["null_basis"],
    )
    np.testing.assert_allclose(actual, mask_first, atol=ATOL, rtol=0)


# ---------------------------------------------------------------- calls into stable_cl
def spy(monkeypatch, name):
    calls = []
    real = getattr(stable_cl, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(stable_cl, name, wrapper)
    return calls


def run_fit(problem, **case):
    features, labels, weights, _ = problem
    return fit_task(
        weights.copy(),
        features,
        labels,
        np.empty((0, P), np.float32),
        np.empty(0, np.int64),
        epochs=EPOCHS,
        learning_rate=LR,
        batch_size=N_ROWS,
        seed=0,
        eligible_classes=np.arange(K_ELIGIBLE),
        **case,
    )


def test_core_uses_the_stable_package_not_a_copy():
    assert multiclass.stable_cl is stable_cl and strategies.stable_cl is stable_cl
    assert not Path(stable_cl.__file__).resolve().is_relative_to(ROOT / "src")


def test_null_space_steps_call_relaxed_step_with_beta_zero_and_the_full_effective_rates(
    monkeypatch, problem
):
    calls = spy(monkeypatch, "relaxed_step")
    case, effective = make_case("basis_feature_group", problem[3])
    run_fit(problem, **case)
    assert len(calls) == EPOCHS  # one call per mini-batch
    expected = LR * effective.astype(np.float64)
    for (gradient, basis, rates, beta), _ in calls:
        assert beta == 0.0
        assert rates.dtype == np.float64 and gradient.dtype == np.float64
        # eta * m, learning rate included
        np.testing.assert_array_equal(rates, expected)
        assert basis is case["null_basis"]


def test_null_groups_call_relaxed_step_once_per_group_on_that_groups_columns(
    monkeypatch, problem
):
    calls = spy(monkeypatch, "relaxed_step")
    case, effective = make_case("groups_feature_group", problem[3])
    run_fit(problem, **case)
    assert len(calls) == EPOCHS * len(case["null_groups"])
    widths = [columns.stop - columns.start for columns, _ in case["null_groups"]]
    for i, ((gradient, basis, rates, beta), _) in enumerate(calls):
        columns, group_basis = case["null_groups"][i % len(widths)]
        assert gradient.shape == (P, widths[i % len(widths)]) and beta == 0.0
        assert basis is group_basis
        np.testing.assert_array_equal(rates, LR * effective.astype(np.float64))


@pytest.mark.parametrize("name", ["ordinary", "feature_group", "mask"])
def test_ordinary_updates_run_inline_without_calling_stable_per_minibatch(
    monkeypatch, problem, name
):
    # The inline `d[:, None] * G` is stable_cl.ordinary_step written without its per-call
    # validation; neither it nor relaxed_step(beta=1) is used on the ordinary path.
    relaxed = spy(monkeypatch, "relaxed_step")
    ordinary = spy(monkeypatch, "ordinary_step")
    case, _ = make_case(name, problem[3])
    run_fit(problem, **case)
    assert relaxed == [] and ordinary == []


@pytest.mark.parametrize("name", ["ordinary", "feature_group", "mask"])
def test_one_ordinary_step_equals_stable_ordinary_step_with_effective_rates(
    problem, name
):
    features, labels, weights, refs = problem
    case, effective = make_case(name, refs)
    eligible = np.arange(K_ELIGIBLE)
    probabilities = softmax(features @ weights[:, eligible])
    probabilities[np.arange(N_ROWS), labels] -= 1.0
    gradient = (features.T @ probabilities / N_ROWS).astype(np.float64)
    if "update_scale" in case:
        gradient = gradient * case["update_scale"][:, eligible]
    d = LR * (np.ones(P) if effective is None else effective.astype(np.float64))
    expected = weights.astype(np.float64).copy()
    expected[:, eligible] -= stable_cl.ordinary_step(gradient, d)
    actual = fit_task(
        weights.copy(),
        features,
        labels,
        np.empty((0, P), np.float32),
        np.empty(0, np.int64),
        epochs=1,
        learning_rate=LR,
        batch_size=N_ROWS,
        seed=0,
        eligible_classes=eligible,
        **case,
    )
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)


def test_one_relaxed_step_equals_stable_relaxed_step_with_effective_rates(problem):
    features, labels, weights, refs = problem
    case, effective = make_case("basis_feature_group", refs)
    eligible = np.arange(K_ELIGIBLE)
    probabilities = softmax(features @ weights[:, eligible])
    probabilities[np.arange(N_ROWS), labels] -= 1.0
    gradient = (features.T @ probabilities / N_ROWS).astype(np.float64)
    d = LR * effective.astype(np.float64)
    expected = weights.astype(np.float64).copy()
    expected[:, eligible] -= stable_cl.relaxed_step(
        gradient, case["null_basis"].astype(np.float64), d, 0.0
    )
    actual = fit_task(
        weights.copy(),
        features,
        labels,
        np.empty((0, P), np.float32),
        np.empty(0, np.int64),
        epochs=1,
        learning_rate=LR,
        batch_size=N_ROWS,
        seed=0,
        eligible_classes=eligible,
        **case,
    )
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)


# ---------------------------------------------------------------- zero rates and masks
@pytest.mark.parametrize(
    "name", ["group", "feature_group", "basis_group", "groups_feature_group"]
)
def test_zero_rate_feature_rows_do_not_move(name, problem):
    _, _, weights, refs = problem
    case, _ = make_case(name, refs)
    after = run_fit(problem, **case)
    frozen = [6, 7]  # group_scale is 0 on these rows
    np.testing.assert_array_equal(after[frozen], weights[frozen])
    assert not np.array_equal(after[:6], weights[:6])


@pytest.mark.parametrize(
    "name, new_rows_move", [("groups", True), ("groups_feature_group_mask", False)]
)
def test_old_by_old_block_is_frozen_exactly_under_grouped_null_space(
    name, new_rows_move, problem
):
    # With the zero-rate rows of "groups_feature_group_mask" the three reference constraints
    # leave no free direction in the new rows of the old columns, so nothing moves there.
    _, _, weights, refs = problem
    case, _ = make_case(name, refs)
    case["update_scale"] = update_mask()  # old x old block is 0
    after = run_fit(problem, **case)
    np.testing.assert_array_equal(after[:4, :2], weights[:4, :2])
    assert np.array_equal(after[4:, :2], weights[4:, :2]) != new_rows_move
    assert not np.array_equal(after[:, 2:K_ELIGIBLE], weights[:, 2:K_ELIGIBLE])


def test_non_eligible_output_columns_are_never_touched(problem):
    _, _, weights, refs = problem
    case, _ = make_case("groups_feature_group_mask", refs)
    after = run_fit(problem, **case)
    np.testing.assert_array_equal(after[:, K_ELIGIBLE:], weights[:, K_ELIGIBLE:])


def test_a_non_finite_gradient_is_rejected_by_stable_in_null_space_steps(problem):
    features, labels, weights, refs = problem
    case, _ = make_case("basis", refs)
    bad = features.copy()
    bad[0, 0] = np.inf
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="finite"):
        fit_task(
            weights.copy(),
            bad,
            labels,
            np.empty((0, P), np.float32),
            np.empty(0, np.int64),
            epochs=1,
            learning_rate=LR,
            batch_size=N_ROWS,
            seed=0,
            **case,
        )


# ---------------------------------------------------------------- effective-rate construction
@pytest.mark.parametrize("control", [None, [1, 1, 1, 1], [1, 0.5, 0, 1], [0, 0, 0, 0]])
def test_core_rate_vector_equals_stable_diagonal_rates(control):
    rates = np.array([3.0, 2.0, 0.75, 1.5])
    group_index = np.array([0, 1, 1, 2, 2, 2, 3, 3, 0])
    ours = diagonal_learning_rates(rates, group_index, control)
    theirs = stable_cl.diagonal_rates(rates, group_index, control)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("control", [None, [1, 0.5, 0, 1]])
def test_step_scale_split_recovers_the_effective_rates(control):
    rates = np.array([3.0, 2.0, 0.75, 1.5])
    group_index = np.array([0, 1, 1, 2, 2, 2, 3, 3, 0])
    effective = stable_cl.diagonal_rates(rates, group_index, control)
    levels = rates if control is None else rates * np.array(control)
    base, multiplier = step_scale_for(levels, group_index)
    assert multiplier.dtype == np.float32 and multiplier.max() <= 1.0
    recovered = base * multiplier.astype(np.float64)
    np.testing.assert_allclose(recovered, effective, rtol=1e-6)


# ---------------------------------------------------------------- restrictions / bases
COUNT, OLD_ROWS, OLD_CLASSES, N_SEEN = 9, 4, 2, 4


@pytest.fixture
def setting():
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(60, COUNT)).astype(np.float32)
    memory = {c: np.arange(10 * c, 10 * c + 10) for c in range(N_SEEN)}
    return {
        "weights_shape": (COUNT, 10),
        "old_rows": OLD_ROWS,
        "old_classes": OLD_CLASSES,
        "n_seen": N_SEEN,
        "count": COUNT,
        "task": 1,
        "train_x": train_x,
        "memory_indices": memory,
        "n_ref_per_class": 3,
        "step_scale": None,
    }


def test_strategy_bases_come_from_stable_null_basis_with_float64_effective_rates(
    monkeypatch, setting
):
    calls = spy(monkeypatch, "null_basis")
    setting["step_scale"] = np.linspace(0.2, 1.0, COUNT).astype(np.float32)
    update_restrictions("strategy_b", base_rate=0.4, **setting)
    [((design, rates), _)] = calls
    assert design.dtype == np.float64 and rates.dtype == np.float64
    np.testing.assert_array_equal(rates, 0.4 * setting["step_scale"].astype(np.float64))


@pytest.mark.parametrize("strategy", ["strategy_b", "strategy_b_new"])
def test_strategy_bases_do_not_depend_on_the_learning_rate_factor(strategy, setting):
    # With the former max(sigma_1, 1) rule a tiny rate made every direction null.
    setting["step_scale"] = np.linspace(0.2, 1.0, COUNT).astype(np.float32)
    unit = update_restrictions(strategy, base_rate=1.0, **setting)
    tiny = update_restrictions(strategy, base_rate=1e-22, **setting)
    pairs = (
        [(unit.null_basis, tiny.null_basis)]
        if strategy == "strategy_b"
        else [(a[1], b[1]) for a, b in zip(unit.null_groups, tiny.null_groups)]
    )
    for a, b in pairs:
        assert a.dtype == np.float32 and a.shape == b.shape and a.shape[1] < COUNT
        np.testing.assert_allclose(a @ a.T, b @ b.T, atol=1e-5)


@pytest.mark.parametrize("strategy", ["strategy_b", "strategy_b_new"])
def test_reference_logits_are_preserved_through_the_integrated_training_path(
    strategy, setting
):
    rng = np.random.default_rng(3)
    step_scale = np.array([1, 1, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25], np.float32)
    restrictions = update_restrictions(
        strategy, base_rate=0.5, **{**setting, "step_scale": step_scale}
    )
    train_x = setting["train_x"]
    weights = (0.1 * rng.normal(size=(COUNT, 10))).astype(np.float32)
    labels = rng.integers(0, N_SEEN, size=len(train_x))

    def train(**restrict):
        return fit_task(
            weights.copy(),
            train_x,
            labels,
            train_x[:0],
            labels[:0],
            epochs=2,
            learning_rate=0.5,
            batch_size=len(train_x),
            seed=0,
            eligible_classes=np.arange(N_SEEN),
            group_scale=step_scale,
            **restrict,
        )

    ordinary = train()
    after = train(
        update_scale=restrictions.scale,
        null_basis=restrictions.null_basis,
        null_groups=restrictions.null_groups,
    )
    before = weights
    refs = train_x[restrictions.refs].astype(np.float64)
    constrained = np.abs(refs @ (after - before)[:, :N_SEEN].astype(np.float64)).max()
    free = np.abs(refs @ (ordinary - before)[:, :N_SEEN].astype(np.float64)).max()
    assert free > 1e-2 and constrained < 1e-4 * free


# ---------------------------------------------------------------- the changed tolerance
def legacy_null_dim(design, rtol):
    singular = np.linalg.svd(design, compute_uv=False)
    return design.shape[1] - int((singular > rtol * max(singular[0], 1.0)).sum())


def test_bases_use_the_relative_tolerance_of_stable():
    rates = np.ones(12)
    design = 0.01 * np.random.default_rng(0).normal(size=(4, 12))  # sigma_1 < 1
    assert np.linalg.svd(design, compute_uv=False)[0] < 1
    # the former rule: tol * max(sigma_1, 1) = 0.3 exceeds every singular value
    assert legacy_null_dim(design, 0.3) == 12
    assert stable_cl.null_basis(design, rates, 0.3).shape[1] < 12
    # with sigma_1 >= 1 and rtol = 1e-10 the two rules coincide
    design = np.random.default_rng(1).normal(size=(4, 12)) * 5
    assert stable_cl.null_basis(design, rates).shape[1] == legacy_null_dim(
        design, 1e-10
    )


# ---------------------------------------------------------------- architecture (multiclass path)
LINALG_DECOMPOSITIONS = {"svd", "qr", "eig", "eigh", "pinv", "lstsq", "matrix_rank"}


@pytest.mark.parametrize("path", ["training/multiclass.py", "continual/strategies.py"])
def test_multiclass_integration_path_has_no_local_decomposition(path):
    tree = ast.parse((ROOT / "src" / "polynomial_cl" / path).read_text())
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not used & LINALG_DECOMPOSITIONS
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "null_space_basis" not in imported


def test_the_float32_null_space_wrapper_is_gone_from_the_package():
    # core keeps no null-space API of its own; the bases come from stable_cl.null_basis
    assert not hasattr(multiclass, "null_space_basis")


# ---------------------------------------------------------------- dependency pin
def test_stable_dependency_is_pinned_to_one_immutable_commit():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    pins = [
        d for d in project["dependencies"] if d.startswith("stable-continual-learning")
    ]
    assert len(pins) == 1
    match = re.fullmatch(
        r"stable-continual-learning @ git\+https://github\.com/"
        r"kobros-tech/stable_continual_learning_package\.git@([0-9a-f]{40})",
        pins[0],
    )
    assert match, pins[0]
    assert match.group(1) == STABLE_SHA


def test_python_floor_matches_the_stable_package_and_ruff_target():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["project"]["requires-python"] == ">=3.11"
    assert config["tool"]["ruff"]["target-version"] == "py311"


def test_installed_stable_distribution_is_the_pinned_commit_when_installed_from_git():
    try:
        distribution = metadata.distribution("stable-continual-learning")
    except metadata.PackageNotFoundError:
        pytest.skip("stable-continual-learning is not installed as a distribution")
    raw = distribution.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    if "vcs_info" not in info:
        pytest.skip("not installed from a git URL (e.g. a local development copy)")
    assert info["vcs_info"]["commit_id"] == STABLE_SHA
