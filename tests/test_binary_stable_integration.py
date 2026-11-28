# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""C3b: the binary trainer and its preservation basis take their mathematics from ``stable_cl``.

``tests/test_equivalence_binary.py`` (unchanged from the C2 stage) holds the numbers of the
former implementation and still passes against the delegating one. This module checks the
integration itself: which functions are called with which arguments, how the stored former
bases compare with the new ones, the one case where the relative tolerance deliberately
differs, and that no null-space or relaxed-update implementation is left in the package.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import stable_cl

from polynomial_cl.models.binary import PolynomialLogisticModel, n_features
from polynomial_cl.training import binary
from polynomial_cl.training.binary import preservation_null_basis, train

ROOT = Path(__file__).parents[1]
GOLDEN = ROOT / "tests" / "golden"
FIXTURE = json.loads((GOLDEN / "c2_binary_train.json").read_text())
PROBLEM = FIXTURE["problem"]
CASES = {case["name"]: case for case in FIXTURE["cases"]}
PRESERVE = [name for name, case in CASES.items() if case["preserve"]]
X, Y = np.asarray(PROBLEM["x"]), np.asarray(PROBLEM["y"])
X_REF = np.asarray(PROBLEM["x_ref"])
OFFSET = np.asarray(PROBLEM["offset"])
DESIGN_REF = np.asarray(PROBLEM["design_ref"])
DEGREE, SEED, BASE_LR = PROBLEM["new_degree"], PROBLEM["seed"], PROBLEM["base_lr"]
TOL_BASIS = 1e-12  # projector of the new basis vs the stored former one; measured 0
TOL_GOLDEN = 1e-10  # trained weights vs the stored former results; measured 1.1e-16
CONTROL_FACTOR = 100


def spy(monkeypatch, name):
    calls, real = [], getattr(stable_cl, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(stable_cl, name, wrapper)
    return calls


def run_case(case, variant="multi", **overrides):
    setting = case["runs"][variant]
    basis = case["null_basis"]
    arguments = {
        "lr": np.asarray(case["lr"]),
        "epochs": setting["epochs"],
        "batch_size": setting["batch_size"],
        "seed": SEED,
        "trainable": np.asarray(case["trainable"], bool),
        "offset": OFFSET if case["composed"] else None,
        "null_basis": None if basis is None else np.asarray(basis),
    }
    arguments.update(overrides)
    model = PolynomialLogisticModel(DEGREE, np.asarray(case["w_before"]))
    return train(model, X, Y, **arguments)


def deviation(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


# ------------------------------------------------------------------ the preservation basis
@pytest.mark.parametrize("name", PRESERVE)
def test_preservation_basis_equals_the_stored_former_basis(name):
    case = CASES[name]
    mask = np.asarray(case["trainable"], bool)
    basis = preservation_null_basis(DEGREE, X_REF, mask)
    former = np.asarray(case["null_basis"])
    assert basis.shape == former.shape and basis.shape[1] > 0
    assert deviation(basis @ basis.T, former @ former.T) <= TOL_BASIS
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)
    assert np.all(basis[~mask] == 0.0)  # exactly zero rows for frozen coefficients
    assert np.abs(DESIGN_REF @ basis).max() <= 1e-12  # g(x_ref) = 0


def test_one_of_the_stored_cases_already_has_sigma1_below_one():
    # composed_new_preserve: sigma_1 of the trainable reference design is about 0.49, so the
    # former and the relative threshold differ in principle; here they select the same space.
    mask = np.asarray(CASES["composed_new_preserve"]["trainable"], bool)
    sigma1 = np.linalg.svd(DESIGN_REF[:, mask], compute_uv=False)[0]
    assert sigma1 < 1.0


def former_dimension(x_ref, mask):
    """Null-space dimension under the former rule rtol * max(sigma_1, 1)."""
    design = np.asarray(PolynomialLogisticModel(DEGREE).design(np.atleast_2d(x_ref)))[
        :, mask
    ]
    singular = np.linalg.svd(design, compute_uv=False)
    return design.shape[1] - int((singular > 1e-10 * max(singular[0], 1.0)).sum())


def test_references_near_the_origin_are_constraints_not_ignored():
    # Frozen bias and old coefficients, reference inputs of size 1e-4: the trainable columns are
    # cubic monomials of about 1e-12, so sigma_1 is far below the old absolute floor 1e-10 and
    # the former rule found no constraint at all (every direction "null").
    n_old = n_features(2)
    mask = np.ones(n_features(DEGREE), bool)
    mask[:n_old] = False
    x_ref = 1e-4 * np.array([[1.0, 0.5], [0.3, -0.8]])
    assert former_dimension(x_ref, mask) == mask.sum()  # the hazard of the former rule
    basis = preservation_null_basis(DEGREE, x_ref, mask)
    assert basis.shape[1] == mask.sum() - 2  # two reference inputs, two constraints
    design = PolynomialLogisticModel(DEGREE).design(x_ref)
    sigma1 = np.linalg.svd(design[:, mask], compute_uv=False)[0]
    assert np.abs(design @ basis).max() <= 1e-10 * sigma1  # the constraints hold


@pytest.mark.parametrize("rtol", [1.0, 2.0, -1e-3, float("nan")])
def test_preservation_basis_rejects_a_tolerance_outside_zero_one(rtol):
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        preservation_null_basis(2, X_REF, rtol=rtol)


def test_preservation_basis_edge_cases_keep_their_former_results():
    assert preservation_null_basis(2, np.empty((0, 2))).shape == (6, 6)
    assert np.array_equal(preservation_null_basis(2, np.empty((0, 2))), np.eye(6))
    none_trainable = np.zeros(6, bool)
    assert preservation_null_basis(2, X_REF, none_trainable).shape == (6, 0)
    three_points = np.array([[0.1, 0.2], [0.5, -0.3], [0.9, 0.4]])
    # three constraints use up the three coefficients of a degree-1 model
    assert preservation_null_basis(1, three_points).shape == (3, 0)


# ------------------------------------------------------------------ calls into stable_cl
@pytest.mark.parametrize("name", PRESERVE)
def test_projected_steps_are_relaxed_steps_with_beta_zero_and_the_effective_rates(
    monkeypatch, name
):
    calls = spy(monkeypatch, "relaxed_step")
    case = CASES[name]
    result = run_case(case, "multi")
    epochs = case["runs"]["multi"]["epochs"]
    assert len(calls) == epochs  # one call per update (full batch)
    expected_rates = np.asarray(case["lr"]) * np.asarray(case["trainable"], bool)
    for (gradient, basis, rates, beta), _ in calls:
        assert beta == 0.0
        assert gradient.shape == (len(expected_rates), 1)
        assert gradient.dtype == np.float64
        assert basis.shape == np.asarray(case["null_basis"]).shape
        np.testing.assert_array_equal(rates, expected_rates)  # lr * trainable
    np.testing.assert_array_equal(result.lr_vector, expected_rates)


def test_every_mini_batch_is_one_stable_call(monkeypatch):
    # three epochs of shuffled mini-batches: ceil(30 / 7) = 5 updates per epoch
    calls = spy(monkeypatch, "relaxed_step")
    run_case(CASES["composed_all_preserve"], "sampling")
    assert len(calls) == 3 * 5


@pytest.mark.parametrize("name", [n for n in CASES if not CASES[n]["preserve"]])
def test_ordinary_updates_run_inline_without_calling_stable(monkeypatch, name):
    relaxed = spy(monkeypatch, "relaxed_step")
    ordinary = spy(monkeypatch, "ordinary_step")
    run_case(CASES[name], "multi")
    assert relaxed == [] and ordinary == []


@pytest.mark.parametrize("mode", ["omit", "double"])
@pytest.mark.parametrize("name", PRESERVE)
def test_negative_control_wrong_rates_inside_the_stable_call_are_detected(
    monkeypatch, name, mode
):
    real = stable_cl.relaxed_step

    def mutated(gradient, basis, rates, beta):
        wrong = rates / BASE_LR if mode == "omit" else rates * BASE_LR
        return real(gradient, basis, wrong, beta)

    monkeypatch.setattr(stable_cl, "relaxed_step", mutated)
    case = CASES[name]
    stored = case["runs"]["multi"]["w_after"]
    assert deviation(run_case(case).w_after, stored) > CONTROL_FACTOR * TOL_GOLDEN


@pytest.mark.parametrize("name", PRESERVE)
def test_trained_weights_equal_the_stored_former_results(name):
    case = CASES[name]
    for variant in ("single", "multi"):
        stored = case["runs"][variant]["w_after"]
        assert deviation(run_case(case, variant).w_after, stored) <= TOL_GOLDEN


def test_a_non_finite_gradient_is_rejected_by_stable_in_projected_steps():
    # Behaviour change: the former projection propagated NaN silently; stable validates.
    case = CASES["composed_all_preserve"]
    bad = X.copy()
    bad[0, 0] = np.inf
    model = PolynomialLogisticModel(DEGREE, np.asarray(case["w_before"]))
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="finite"):
        train(
            model,
            bad,
            Y,
            np.asarray(case["lr"]),
            1,
            None,
            SEED,
            np.asarray(case["trainable"], bool),
            OFFSET,
            np.asarray(case["null_basis"]),
        )


def test_shared_rate_requirement_of_the_projected_step_is_kept():
    case = CASES["composed_all_preserve"]
    uneven = np.asarray(case["lr"]).copy()
    uneven[0] *= 2
    with pytest.raises(ValueError, match="shared learning rate"):
        run_case(case, lr=uneven)


# ------------------------------------------------------------------ architecture (whole package)
LINALG_DECOMPOSITIONS = {"svd", "qr", "eig", "eigh", "pinv", "lstsq", "matrix_rank"}
# The only decompositions left in the package, none of which is an update or a null space:
#   data/mnist.py        qr: orthonormalises the random projection of MNIST pixels
#   continual/experiment matrix_rank: reports how many directions can move (a diagnostic)
ALLOWED = {("data/mnist.py", "qr"), ("continual/experiment.py", "matrix_rank")}


def decompositions_used():
    found = set()
    package = ROOT / "src" / "polynomial_cl"
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in LINALG_DECOMPOSITIONS:
                found.add((path.relative_to(package).as_posix(), node.attr))
    return found


def test_package_has_no_null_space_or_relaxed_update_implementation():
    assert decompositions_used() == ALLOWED


MODULES = ["training/multiclass.py", "training/binary.py", "continual/strategies.py"]


def test_both_trainers_import_the_stable_package():
    assert binary.stable_cl is stable_cl
    for module in MODULES:
        source = (ROOT / "src" / "polynomial_cl" / module).read_text()
        assert "import stable_cl" in source
