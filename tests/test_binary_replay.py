# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.binary.data import make_experiences
from polynomial_cl.binary.experiment import make_config, run_continual
from polynomial_cl.binary.replay import ReplayBuffer, build_batch


@pytest.fixture(scope="module")
def exps():
    return make_experiences(n_train=120, n_test=150, seed=4)


def _rowset(a):
    return {tuple(r) for r in np.round(a, 12)}


def test_buffer_rejects_evaluation_data(exps):
    buf = ReplayBuffer(50)
    with pytest.raises(ValueError):
        buf.add_experience(exps[0].test)
    with pytest.raises(ValueError):
        buf.add_experience((exps[0].test.x, exps[0].test.y))
    assert buf.experience_ids == []


def test_no_evaluation_sample_in_buffer_or_batches(exps):
    buf = ReplayBuffer(60, seed=1)
    for e in exps[:5]:
        buf.add_experience(e.train)
    test_rows = set().union(*[_rowset(e.test.x) for e in exps])
    for k in buf.experience_ids:
        assert not (_rowset(buf.stored(k)[0]) & test_rows)
    b = build_batch(exps[5].train, buf, 50, 50, seed=1, step=5)
    assert not (_rowset(b.x) & test_rows)


def test_labels_valid_and_match_originals(exps):
    buf = ReplayBuffer(60, seed=1)
    buf.add_experience(exps[2].train)
    x, y = buf.stored(3)
    assert set(np.unique(y)) <= {0, 1}
    lookup = {
        tuple(r): lab for r, lab in zip(np.round(exps[2].train.x, 12), exps[2].train.y)
    }
    assert all(lookup[tuple(r)] == lab for r, lab in zip(np.round(x, 12), y))


def test_sampling_reproducible(exps):
    def make(seed):
        buf = ReplayBuffer(60, seed=seed)
        for e in exps[:3]:
            buf.add_experience(e.train)
        return build_batch(exps[3].train, buf, 30, 30, seed=seed, step=3)

    a, b, c = make(7), make(7), make(8)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.x, c.x)


def test_balanced_batch_composition(exps):
    buf = ReplayBuffer(60)
    for e in exps[:3]:
        buf.add_experience(e.train)
    b = build_batch(exps[3].train, buf, 25, 25, seed=0, step=3)
    assert b.counts == {4: 25, 1: 25, 2: 25, 3: 25}
    assert len(b.y) == 100
    for k, n in b.counts.items():
        assert int(np.sum(b.source == k)) == n
    d = build_batch(
        exps[3].train, buf, 10, {1: 5, 2: 0, 3: 20}, seed=0, step=3
    )  # per-experience control
    assert d.counts == {4: 10, 1: 5, 2: 0, 3: 20}


@pytest.mark.parametrize(
    "m_current, m_replay, error, message",
    [
        (-1, 0, ValueError, "m_current"),
        (1.5, 0, TypeError, "m_current"),
        (True, 0, TypeError, "m_current"),
        (1, -1, ValueError, "m_replay"),
        (1, 1.5, TypeError, "m_replay"),
        (1, {1: -1}, ValueError, "m_replay\\[1\\]"),
        (1, {1: 0.5}, TypeError, "m_replay\\[1\\]"),
    ],
)
def test_batch_rejects_invalid_sample_counts(exps, m_current, m_replay, error, message):
    buf = ReplayBuffer(20)
    buf.add_experience(exps[0].train)

    with pytest.raises(error, match=message):
        build_batch(exps[1].train, buf, m_current, m_replay)


def test_batch_allows_zero_sample_counts(exps):
    buf = ReplayBuffer(20)
    buf.add_experience(exps[0].train)

    batch = build_batch(exps[1].train, buf, 0, {1: 0})

    assert len(batch.y) == 0
    assert batch.counts == {2: 0, 1: 0}


def test_insufficient_samples_policy(exps):
    buf = ReplayBuffer(20)
    buf.add_experience(exps[0].train)
    no_rep = build_batch(exps[1].train, buf, 10, 50, replace=False, seed=0)
    assert no_rep.counts[1] == 20  # all stored rows, never more
    assert len(set(map(tuple, no_rep.x[no_rep.source == 1]))) == 20
    rep = build_batch(exps[1].train, buf, 10, 50, replace=True, seed=0)
    assert rep.counts[1] == 50


@pytest.mark.parametrize(
    "capacity, error",
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_buffer_rejects_invalid_capacity(capacity, error):
    with pytest.raises(error, match="capacity_per_experience"):
        ReplayBuffer(capacity)


def test_capacity_respected(exps):
    buf = ReplayBuffer(33)
    buf.add_experience(exps[0].train)
    assert buf.size(1) == 33


@pytest.mark.parametrize(
    "count, error",
    [(-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_reference_inputs_rejects_invalid_count(exps, count, error):
    buf = ReplayBuffer(20)
    buf.add_experience(exps[0].train)
    with pytest.raises(error, match="n_per_experience"):
        buf.reference_inputs(count)


def test_reference_inputs_allows_zero(exps):
    buf = ReplayBuffer(20)
    buf.add_experience(exps[0].train)
    assert buf.reference_inputs(0).shape == (0, 2)


def test_loop_records_actual_counts_and_uses_only_past(exps):
    res = run_continual(
        exps,
        make_config(
            "ordinary", epochs=5, m_current=30, m_replay=30, capacity=30, stop_after=5
        ),
    )
    for r in res.records:
        assert set(r.batch_counts) == set(
            range(1, r.t + 2)
        )  # only experiences seen so far
        assert all(v == 30 for v in r.batch_counts.values())
