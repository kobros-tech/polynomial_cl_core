# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.binary.data import make_experiences
from polynomial_cl.binary.experiment import make_config, run_continual
from polynomial_cl.models.binary import PolynomialLogisticModel


@pytest.fixture(scope="module")
def exps():
    return make_experiences(n_train=60, n_test=80, seed=3)


def _random_model(d, seed=0):
    return PolynomialLogisticModel(
        d, np.random.default_rng(seed).normal(size=(d + 1) * (d + 2) // 2)
    )


def test_old_coefficients_copied_exactly_and_new_are_zero():
    m = _random_model(3)
    e = m.expanded(4)
    assert np.array_equal(e.w[: m.n_features], m.w)
    assert np.all(e.w[m.n_features :] == 0.0)
    assert e.feature_names[: m.n_features] == m.feature_names
    assert e.bias == m.bias  # bias preserved, stored once (constant feature)


def test_expansion_does_not_mutate_original():
    m = _random_model(2)
    w = m.w.copy()
    m.expanded(5)
    assert np.array_equal(m.w, w) and m.degree == 2


def test_cannot_shrink():
    with pytest.raises(ValueError):
        _random_model(3).expanded(2)


def test_scores_and_probabilities_identical(exps):
    rng = np.random.default_rng(1)
    sets = {
        "train": np.concatenate([e.train.x for e in exps]),
        "eval": np.concatenate([e.test.x for e in exps]),
        "random": rng.uniform(-1, 1, (300, 2)),
        "outside": rng.uniform(-4, 4, (300, 2)),
    }
    for d in (1, 2, 5, 9):
        m = _random_model(d, seed=d)
        for k in (1, 2):
            e = m.expanded(d + k)
            for name, x in sets.items():
                np.testing.assert_allclose(
                    e.score(x), m.score(x), rtol=1e-10, atol=1e-10, err_msg=name
                )
                np.testing.assert_allclose(
                    e.proba(x), m.proba(x), rtol=1e-10, atol=1e-12, err_msg=name
                )
                np.testing.assert_array_equal(e.predict(x), m.predict(x))


def test_continual_loop_verifies_invariant_every_step(exps):
    res = run_continual(
        exps, make_config("ordinary", epochs=20, m_current=40, m_replay=40, capacity=40)
    )
    assert len(res.records) == 10
    for r in res.records:
        assert r.invariant_score_diff < 1e-9 and r.invariant_proba_diff < 1e-9
        assert r.degree_after == r.t + 1
        np.testing.assert_allclose(
            r.model_expanded.score(exps[0].test.x),
            r.model_before.score(exps[0].test.x),
            atol=1e-10,
        )


def test_seed_summary_rows_step_rows_and_sequential_curves():
    import matplotlib.pyplot as plt

    from polynomial_cl.binary import run_seeds, seed_summary_rows, step_rows
    from polynomial_cl.plotting import synthetic as P

    data = {"n_train": 40, "n_test": 40, "n_experiences": 4}
    runs = {
        name: run_seeds(
            [0, 1],
            make_config(name, epochs=20, m_current=20, m_replay=20, capacity=30),
            data,
        )
        for name in ("ordinary", "strategy_b_new")
    }
    rows = seed_summary_rows(runs, key="strategy")
    assert [r["strategy"] for r in rows] == ["ordinary", "strategy_b_new"]
    assert rows[0]["null_dim_last"] == "n/a" and rows[1]["null_dim_last"] != "n/a"
    last = [r.records[-1].mean_acc for r in runs["ordinary"]]
    assert (
        rows[0]["final_mean_acc"] == f"{np.mean(last):.3f}+/-{np.std(last, ddof=1):.3f}"
    )
    steps = step_rows(runs["strategy_b_new"][0])
    assert [s["exp"] for s in steps] == [1, 2, 3, 4]
    assert (
        abs(steps[-1]["ref_score_change"]) < 1e-9
    )  # the constraint's defining property
    axs = P.plot_sequential_curves(runs)
    assert len(axs[0].get_lines()) == 2
    plt.close(axs[0].figure)


def test_config_with_defaults_lets_a_preset_keep_its_defining_fields():
    from polynomial_cl.binary import config_with_defaults

    shared = {"lr": 0.3, "epochs": 25, "m_replay": 50, "capacity": 80}
    no_replay = config_with_defaults("ordinary_no_replay", shared)
    assert no_replay.m_replay == 0 and no_replay.lr == 0.3 and no_replay.epochs == 25
    ordinary = config_with_defaults("ordinary", shared, seed=4)
    assert ordinary.m_replay == 50 and ordinary.seed == 4
    with pytest.raises(ValueError, match="defines"):
        config_with_defaults("ordinary_no_replay", shared, m_replay=10)
