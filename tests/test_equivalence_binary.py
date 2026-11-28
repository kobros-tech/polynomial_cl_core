# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""C2: the binary trainer's projected step is ``stable_cl``'s update with the effective rates.

``training.binary.train`` still contains its own update: a gradient step with per-coefficient
rates ``lr * trainable``, projected by ``N N^T`` when a preservation basis is given. This module
pins that behaviour, so that replacing it by calls to ``stable_cl`` can be checked against it:

* ``c2_binary_train.json`` stores the results of the current ``train`` (scopes ``all`` and
  ``new``, composed ``f = f_old + g`` with an offset, plain training, single and multiple
  updates, full-batch and shuffled mini-batches) together with all of its inputs;
* ``reference_loop`` is the same loop with the step written as
  ``stable_cl.ordinary_step`` / ``stable_cl.relaxed_step(beta=0)`` and ``d = lr * trainable``;
* the whole ``run_continual`` is exercised through a spy on ``train``, so that the arguments the
  experiment really passes (scope masks, offsets, bases, sampling) are checked, not only
  hand-made ones.

The preservation basis of the binary code lives in the trainable coordinates only, while
``stable_cl.null_basis`` also contains the frozen coordinates (they have rate 0); the two give
the same *update*, which is what the basis test compares. The mini-batch order comes from
``numpy.random.default_rng(seed).permutation``; the stored golden values of those cases are
skipped, with a reason, if a NumPy release changes that stream (the reference tests replay the
same stream and are not affected).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import stable_cl

from polynomial_cl.binary import experiment
from polynomial_cl.binary.data import make_experiences
from polynomial_cl.binary.experiment import make_config, run_continual
from polynomial_cl.models.binary import PolynomialLogisticModel, sigmoid
from polynomial_cl.training.binary import train

GOLDEN = Path(__file__).parent / "golden"
FIXTURE = json.loads((GOLDEN / "c2_binary_train.json").read_text())
PROBLEM = FIXTURE["problem"]
CASES = {case["name"]: case for case in FIXTURE["cases"]}
VARIANTS = ["single", "multi", "sampling"]
ALL = [(name, variant) for name in CASES for variant in VARIANTS]
X, Y = np.asarray(PROBLEM["x"]), np.asarray(PROBLEM["y"])
OFFSET = np.asarray(PROBLEM["offset"])
DESIGN_REF = np.asarray(PROBLEM["design_ref"])
SEED, BASE_LR = PROBLEM["seed"], PROBLEM["base_lr"]
N_OLD = len(PROBLEM["old_w"])

# Both sides are float64 NumPy, so the tolerances only have to absorb BLAS differences between
# machines. Largest deviations measured over the cases (on the machine that made the fixture):
TOL_GOLDEN = 1e-10  # current train vs stored results; measured 0 (bit-identical)
TOL_REFERENCE = 1e-10  # stable-based loop vs stored results / train; measured 3.3e-16
TOL_STEP = 1e-12  # one projected step; measured 2.5e-16
# A wrong learning rate must deviate by at least this many tolerances; the smallest deviation
# measured is 1.3e-02 (fixtures) and 4.1e-01 (run_continual), far above 100 * 1e-10.
CONTROL_FACTOR = 100


def stream_unchanged():
    canary = FIXTURE["meta"]["permutation_canary"]
    drawn = np.random.default_rng(canary["seed"]).permutation(canary["n"]).tolist()
    return drawn == canary["first_permutation"]


def skip_if_stream_changed(variant):
    if variant == "sampling" and not stream_unchanged():
        pytest.skip("numpy's permutation stream differs from the one the fixture used")


def current_train(case, variant):
    setting = case["runs"][variant]
    return train(
        PolynomialLogisticModel(PROBLEM["new_degree"], np.asarray(case["w_before"])),
        X,
        Y,
        np.asarray(case["lr"]),
        setting["epochs"],
        setting["batch_size"],
        SEED,
        np.asarray(case["trainable"], bool),
        OFFSET if case["composed"] else None,
        None if case["null_basis"] is None else np.asarray(case["null_basis"]),
    )


def reference_loop(w0, design, y, offset, rates, basis, epochs, batch_size, seed):
    """``train`` with the step written by ``stable_cl`` (``rates`` = lr * trainable)."""
    w = np.asarray(w0, dtype=float).copy()
    offset = np.zeros(len(y)) if offset is None else offset
    rng = np.random.default_rng(seed)
    n = len(y)
    size = n if batch_size is None else batch_size
    for _ in range(epochs):
        order = rng.permutation(n) if size < n else np.arange(n)
        for start in range(0, n, size):
            rows = order[start : start + size]
            error = sigmoid(offset[rows] + design[rows] @ w) - y[rows]
            gradient = (design[rows].T @ error / len(rows))[:, None]
            if basis is None:
                step = stable_cl.ordinary_step(gradient, rates)
            else:
                step = stable_cl.relaxed_step(gradient, basis, rates, 0.0)
            w -= step[:, 0]
    return w


def reference_for(case, variant, rates=None):
    setting = case["runs"][variant]
    mask = np.asarray(case["trainable"], bool)
    rates = np.asarray(case["lr"]) * mask if rates is None else rates
    basis = stable_cl.null_basis(DESIGN_REF, rates) if case["preserve"] else None
    design = PolynomialLogisticModel(PROBLEM["new_degree"]).design(X)
    return reference_loop(
        case["w_before"],
        design,
        Y,
        OFFSET if case["composed"] else None,
        rates,
        basis,
        setting["epochs"],
        setting["batch_size"],
        SEED,
    )


def deviation(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


# ------------------------------------------------------------------ provenance and coverage
def test_fixture_provenance_and_coverage():
    meta = FIXTURE["meta"]
    assert meta["schema"] == 1
    assert len(meta["generated_from"]["commit"]) == 40
    assert {c["scope"] for c in CASES.values()} == {None, "all", "new"}
    assert any(c["composed"] and c["preserve"] for c in CASES.values())
    assert any(not c["composed"] for c in CASES.values())
    assert PROBLEM["base_lr"] not in (0.0, 1.0)  # a missing or doubled rate must show
    for case in CASES.values():
        if case["preserve"]:
            # a non-trivial basis: neither empty nor the whole space
            assert 0 < len(case["null_basis"][0]) < len(case["w_before"])


# ------------------------------------------------------------------ golden values
@pytest.mark.parametrize("name, variant", ALL)
def test_train_reproduces_the_stored_results(name, variant):
    skip_if_stream_changed(variant)
    case = CASES[name]
    expected = case["runs"][variant]
    result = current_train(case, variant)
    assert deviation(result.w_after, expected["w_after"]) <= TOL_GOLDEN
    assert deviation(result.loss_history, expected["loss_history"]) <= TOL_GOLDEN
    np.testing.assert_array_equal(result.lr_vector, expected["lr_vector"])
    moved = np.abs(result.w_after - np.asarray(case["w_before"])).max()
    assert moved > 1e-3  # the stored update is not trivial


@pytest.mark.parametrize("name, variant", ALL)
def test_stable_based_loop_equals_the_binary_trainer(name, variant):
    skip_if_stream_changed(variant)
    case = CASES[name]
    stored = case["runs"][variant]["w_after"]
    assert deviation(reference_for(case, variant), stored) <= TOL_REFERENCE


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", [n for n, c in CASES.items() if c["preserve"]])
def test_strategy_b_scopes_are_relaxed_step_with_beta_zero(name, variant):
    # strategy_b_all / strategy_b_new of the binary code = relaxed_step(beta = 0) with
    # d = lr * trainable and the null space of the reference design under those rates.
    skip_if_stream_changed(variant)
    case = CASES[name]
    rates = np.asarray(case["lr"]) * np.asarray(case["trainable"], bool)
    basis = stable_cl.null_basis(DESIGN_REF, rates)
    result = current_train(case, variant)
    reference = reference_loop(
        case["w_before"],
        PolynomialLogisticModel(PROBLEM["new_degree"]).design(X),
        Y,
        OFFSET,
        rates,
        basis,
        case["runs"][variant]["epochs"],
        case["runs"][variant]["batch_size"],
        SEED,
    )
    assert deviation(result.w_after, reference) <= TOL_REFERENCE
    assert np.abs(DESIGN_REF @ result.w_after).max() <= 1e-8  # references are preserved


@pytest.mark.parametrize("mode", ["omit", "double"])
@pytest.mark.parametrize("name, variant", ALL)
def test_negative_control_wrong_learning_rate_is_detected(name, variant, mode):
    skip_if_stream_changed(variant)
    case = CASES[name]
    rates = np.asarray(case["lr"]) * np.asarray(case["trainable"], bool)
    wrong = rates / BASE_LR if mode == "omit" else rates * BASE_LR
    stored = case["runs"][variant]["w_after"]
    assert deviation(reference_for(case, variant, wrong), stored) > (
        CONTROL_FACTOR * TOL_REFERENCE
    )


# ------------------------------------------------------------------ scopes, masks, offset
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "name", [n for n, c in CASES.items() if not all(c["trainable"])]
)
def test_frozen_coefficients_are_unchanged_bit_for_bit(name, variant):
    case = CASES[name]
    mask = np.asarray(case["trainable"], bool)
    result = current_train(case, variant)
    np.testing.assert_array_equal(
        result.w_after[~mask], np.asarray(case["w_before"])[~mask]
    )
    assert np.any(result.w_after[mask] != np.asarray(case["w_before"])[mask])


@pytest.mark.parametrize("variant", VARIANTS)
def test_composed_all_and_new_scope_train_the_stated_coefficients(variant):
    for preserve in ("composed_all_preserve", "composed_all_free"):
        g = current_train(CASES[preserve], variant).w_after
        # g spans old and new rows
        assert np.any(g[:N_OLD] != 0) and np.any(g[N_OLD:] != 0)
    for name in ("composed_new_preserve", "composed_new_free"):
        g = current_train(CASES[name], variant).w_after
        assert not np.any(g[:N_OLD])  # scope "new": the old rows of g stay 0
        assert np.any(g[N_OLD:] != 0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_offset_composition_equals_training_the_expanded_model(variant):
    # f = f_old + g with g trained from 0 and offset f_old is the expanded model trained in place.
    n_total = len(CASES["ordinary_shared"]["w_before"])
    old = np.concatenate([PROBLEM["old_w"], np.zeros(n_total - N_OLD)])
    plain = current_train(CASES["ordinary_shared"], variant).w_after
    composed = current_train(CASES["composed_all_free"], variant).w_after
    np.testing.assert_allclose(plain, old + composed, atol=TOL_REFERENCE, rtol=0)
    plain = current_train(CASES["ordinary_frozen_old"], variant).w_after
    composed = current_train(CASES["composed_new_free"], variant).w_after
    np.testing.assert_allclose(plain, old + composed, atol=TOL_REFERENCE, rtol=0)


# ------------------------------------------------------------------ the preservation basis
@pytest.mark.parametrize("name", [n for n, c in CASES.items() if c["preserve"]])
def test_binary_basis_and_stable_basis_give_the_same_update(name):
    case = CASES[name]
    mask = np.asarray(case["trainable"], bool)
    rates = np.asarray(case["lr"]) * mask
    own = np.asarray(case["null_basis"])
    theirs = stable_cl.null_basis(DESIGN_REF, rates)
    assert np.all(own[~mask] == 0.0)  # exact zero rows for frozen coefficients
    # the binary basis omits the frozen (zero-rate) coordinates that stable's basis contains
    assert own.shape[1] == theirs.shape[1] - int((~mask).sum())
    rng = np.random.default_rng(4)
    for _ in range(5):
        gradient = rng.normal(size=(len(mask), 1))
        # the binary trainer's step
        projected = own @ (own.T @ (rates[:, None] * gradient))
        shared = stable_cl.relaxed_step(gradient, theirs, rates, 0.0)
        assert deviation(projected, shared) <= TOL_STEP
        # the step preserves the reference inputs
        assert np.abs(DESIGN_REF @ shared).max() <= 1e-10


# ------------------------------------------------------------------ run_continual through a spy
EXPERIENCES = make_experiences(n_experiences=3, n_train=30, n_test=20, seed=5)
CONTINUAL = {
    "strategy_b_all": {},
    "strategy_b_new": {},
    "ordinary": {},
    "ordinary+feature_power": {"lr_feature_power": 0.5},
    "frozen_old": {},
    "small_old_lr": {},
}


def recorded_calls(label, monkeypatch):
    strategy = label.split("+")[0]
    config = make_config(
        strategy,
        degrees=[1, 2, 3],
        lr=BASE_LR,
        epochs=3,
        batch_size=8,
        capacity=15,
        m_current=20,
        m_replay=10,
        n_ref_per_exp=1,
        seed=1,
        **CONTINUAL[label],
    )
    calls, real = [], experiment.train

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append((args, result))
        return result

    monkeypatch.setattr(experiment, "train", spy)
    run_continual(EXPERIENCES, config)
    return calls


def call_reference(args, rates_of=lambda rates: rates):
    model, x, y, lr_vec, epochs, batch_size, seed, mask = args[:8]
    offset = args[8] if len(args) > 8 else None
    basis = args[9] if len(args) > 9 else None
    rates = rates_of(np.asarray(lr_vec) * np.asarray(mask, bool))
    return reference_loop(
        model.w, model.design(x), y, offset, rates, basis, epochs, batch_size, seed
    )


@pytest.mark.parametrize("label", list(CONTINUAL))
def test_every_training_call_of_run_continual_equals_the_stable_based_loop(
    label, monkeypatch
):
    calls = recorded_calls(label, monkeypatch)
    assert len(calls) == 3  # one training call per experience
    composed = label.startswith("strategy_b")
    for t, (args, result) in enumerate(calls):
        x, batch_size, mask = args[1], args[5], np.asarray(args[7], bool)
        assert len(x) > batch_size  # shuffled mini-batches, i.e. sampling is exercised
        assert deviation(call_reference(args), result.w_after) <= TOL_REFERENCE
        offset = args[8] if len(args) > 8 else None
        assert (offset is not None) == composed  # the offset f_old only when composed
        if composed and t > 0:
            assert args[9] is not None and args[9].shape[1] > 0  # preservation basis
        if label == "strategy_b_all":
            assert mask.all()
        if label == "strategy_b_new" and t > 0:
            n_old = calls[t - 1][0][0].n_features  # coefficients of the previous degree
            assert not mask[:n_old].any() and mask[n_old:].all()


@pytest.mark.parametrize("mode", ["omit", "double"])
@pytest.mark.parametrize("label", list(CONTINUAL))
def test_negative_control_wrong_rates_in_run_continual_are_detected(
    label, mode, monkeypatch
):
    calls = recorded_calls(label, monkeypatch)
    wrong = (lambda r: r / BASE_LR) if mode == "omit" else (lambda r: r * BASE_LR)
    worst = max(
        deviation(call_reference(args, wrong), result.w_after) for args, result in calls
    )
    assert worst > CONTROL_FACTOR * TOL_REFERENCE
