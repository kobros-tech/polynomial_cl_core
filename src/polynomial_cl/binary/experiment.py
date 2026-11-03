# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Expansion-aware sequential binary learning loop and its update strategies (NumPy only).

Strategy B represents f_t = f_old + g_t.  f_old is the (expanded) previous model and is
never modified.  g_t starts at zero.  Two variants are distinguished explicitly:
    A. frozen old component  : g is unconstrained (the old *function* is untouched but the
                               combined predictions may change anywhere).
    B. preserved predictions : g(x_i) = 0 is enforced exactly on reference inputs x_i by
                               restricting g's coefficients to the null space of the
                               reference design matrix.  This preserves f_old at those
                               finitely many inputs only; nothing is guaranteed elsewhere.

The optimiser itself is ``polynomial_cl.training.binary.train``. This loop belongs to the
binary setting (fixed label set, spatially shifting inputs) and is not class-incremental.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..data.synthetic import DOMAIN
from ..metrics import (
    accuracy,
    forgetting,
    old_score_change,
    parameter_changes,
    prediction_flip_fraction,
)
from ..models.binary import PolynomialLogisticModel, check_expansion_invariant
from ..training.binary import positive_int, preservation_null_basis, train
from .data import Experience, combine_splits, make_experiences
from .replay import ReplayBuffer, build_batch


# ============================================================ continual loop
@dataclass
class ContinualConfig:
    degrees: Sequence[int] | None = None  # per experience; default 1,2,...,K
    lr: float = 0.5
    lr_old_factor: float = 1.0  # old coefficients use lr * factor
    # Per-coefficient conditioning: rate_i = lr * (max_j rms_j / rms_i) ** lr_feature_power, with rms
    # the root-mean-square of feature i on the training batch. 0 = one shared rate; 1 = diagonal
    # (Jacobi) preconditioning, which gives small high-degree features larger steps.
    lr_feature_power: float = 0.0
    freeze_old: bool = False  # old coefficients get mask False
    epochs: int = 300
    batch_size: int | None = None
    capacity: int = 100  # replay buffer capacity per experience
    m_current: int = 100  # samples from the current experience per update
    m_replay: int = 100  # samples from EACH old experience (0 = no replay)
    replace: bool = False  # sampling with replacement?
    composed: bool = False  # Strategy B representation f = f_old + g
    preserve: bool = False  # enforce g(x_ref) = 0 (needs composed=True)
    scope: str = "all"  # composed: g over "all" coefficients or only "new"
    n_ref_per_exp: int = 3  # reference inputs per old experience
    seed: int = 0
    stop_after: int | None = None  # process only the first `stop_after` experiences
    name: str = "ordinary"


def degree_schedule(n, start=1, step=1, max_degree=None):
    d = [start + step * i for i in range(n)]
    return [min(v, max_degree) for v in d] if max_degree else d


STRATEGIES = {
    "ordinary_no_replay": {"m_replay": 0},
    "ordinary": {},
    "frozen_old": {"freeze_old": True},
    "small_old_lr": {"lr_old_factor": 0.1},
    "strategy_b_all": {"composed": True, "preserve": True, "scope": "all"},
    "strategy_b_new": {"composed": True, "preserve": True, "scope": "new"},
}


def make_config(strategy="ordinary", **overrides) -> ContinualConfig:
    """Preset + shared hyper-parameters. A preset's defining fields (e.g. m_replay=0 for
    'ordinary_no_replay') cannot be overridden, because doing so would silently turn one
    strategy into another."""
    clash = set(STRATEGIES[strategy]) & set(overrides)
    if clash:
        raise ValueError(
            f"strategy '{strategy}' defines {sorted(clash)}; do not override them"
        )
    return ContinualConfig(name=strategy, **{**STRATEGIES[strategy], **overrides})


def config_with_defaults(strategy, defaults, **overrides) -> ContinualConfig:
    """``make_config`` where a preset's defining fields win over the shared ``defaults``.

    The shared settings only fill what the preset does not define (e.g. ``m_replay`` is not
    taken from ``defaults`` for ``ordinary_no_replay``); explicit ``overrides`` still may not
    touch a defining field.
    """
    shared = {k: v for k, v in defaults.items() if k not in STRATEGIES[strategy]}
    return make_config(strategy, **shared, **overrides)


@dataclass
class StepRecord:
    t: int  # 0-based index of the experience
    degree_before: int
    degree_after: int
    feature_names: list
    model_before: PolynomialLogisticModel  # after previous experience (pre-expansion)
    model_expanded: (
        PolynomialLogisticModel  # immediately after expansion, before training
    )
    model_after: PolynomialLogisticModel
    g_w: np.ndarray | None  # trainable component g (composed mode)
    w_expanded: np.ndarray
    w_after: np.ndarray
    lr_vector: np.ndarray
    trainable: np.ndarray
    n_trainable: int
    null_dim: int | None  # degrees of freedom left by the constraint
    loss_history: np.ndarray
    batch_counts: dict
    invariant_score_diff: float
    invariant_proba_diff: float
    acc_row: np.ndarray  # accuracy on experiences 0..t
    acc_current: float
    mean_acc: float
    forgetting: np.ndarray
    mean_forgetting: float
    max_forgetting: float
    max_old_score_change: float
    old_flip_fraction: float
    ref_score_change: float | None  # max |f_after - f_before| on reference inputs
    ref_flip_fraction: float | None
    frozen_max_change: (
        float | None
    )  # None when the trainable mask has no frozen coefficients
    preserved_component_max_change: (
        float | None
    )  # 0 by construction for composed strategies
    combined_max_coefficient_change: float
    trainable_l1_update: float


@dataclass
class ContinualResult:
    config: ContinualConfig
    experiences: list
    records: list
    acc_matrix: np.ndarray  # [t, i], NaN where i > t


def _probes(seen, seed):
    rng = np.random.default_rng([seed, 12345])
    inside = rng.uniform(*DOMAIN, (500, 2))
    outside = rng.uniform(-3, 3, (500, 2))
    return [e.train.x for e in seen] + [e.test.x for e in seen] + [inside, outside]


def run_continual(
    experiences: Sequence[Experience], cfg: ContinualConfig
) -> ContinualResult:
    """Process experiences one at a time. Only the current train split, the replay buffer
    and the current parameters are used for learning; test splits are used for evaluation
    only, and future experiences are never touched."""
    if not experiences:
        raise ValueError("experiences must not be empty")
    if cfg.scope not in {"all", "new"}:
        raise ValueError("scope must be 'all' or 'new'")
    if not np.isfinite(cfg.lr) or cfg.lr < 0:
        raise ValueError("lr must be finite and non-negative")
    if not np.isfinite(cfg.lr_old_factor) or cfg.lr_old_factor < 0:
        raise ValueError("lr_old_factor must be finite and non-negative")
    if not np.isfinite(cfg.lr_feature_power) or cfg.lr_feature_power < 0:
        raise ValueError("lr_feature_power must be finite and non-negative")
    if cfg.lr_feature_power and cfg.preserve:
        raise ValueError(
            "lr_feature_power needs one shared rate; incompatible with preserve"
        )
    positive_int(cfg.epochs, "epochs", allow_zero=True)
    positive_int(cfg.capacity, "capacity")
    positive_int(cfg.m_current, "m_current")
    positive_int(cfg.m_replay, "m_replay", allow_zero=True)
    positive_int(cfg.n_ref_per_exp, "n_ref_per_exp", allow_zero=True)
    if cfg.batch_size is not None:
        positive_int(cfg.batch_size, "batch_size")
    if cfg.stop_after is not None:
        positive_int(cfg.stop_after, "stop_after")
    K = (
        len(experiences)
        if cfg.stop_after is None
        else min(cfg.stop_after, len(experiences))
    )
    degrees = (
        list(cfg.degrees)
        if cfg.degrees is not None
        else degree_schedule(len(experiences))
    )
    if len(degrees) < K:
        raise ValueError(
            f"degrees must contain at least {K} entries for the processed experiences"
        )
    if any(
        isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer))
        for d in degrees[:K]
    ):
        raise TypeError("each processed degree must be an integer")
    if any(d < 1 for d in degrees[:K]):
        raise ValueError("each processed degree must be a positive integer")
    if any(later < earlier for earlier, later in zip(degrees[:K], degrees[1:K])):
        raise ValueError("processed degrees must be non-decreasing")
    if cfg.preserve and not cfg.composed:
        raise ValueError("preserve=True requires composed=True")
    if cfg.preserve and cfg.n_ref_per_exp == 0:
        raise ValueError("n_ref_per_exp must be >= 1 when preserve=True")
    buffer = ReplayBuffer(cfg.capacity, cfg.seed)
    acc = np.full((len(experiences), len(experiences)), np.nan)
    records, model = [], None

    for t in range(K):
        exp = experiences[t]
        d = degrees[t]
        seen = experiences[: t + 1]
        if model is None:
            model_before = PolynomialLogisticModel(d)
            n_old = 0
        else:
            model_before = model
            n_old = model.n_features
        expanded = model_before.expanded(d)
        ds, dp = check_expansion_invariant(
            model_before, expanded, _probes(seen, cfg.seed)
        )
        assert ds < 1e-9 and dp < 1e-9, f"expansion invariant violated: {ds}, {dp}"

        batch = build_batch(
            exp.train, buffer, cfg.m_current, cfg.m_replay, cfg.replace, cfg.seed, t
        )
        n_new_total = expanded.n_features
        mask = np.ones(n_new_total, bool)
        lr_vec = np.full(n_new_total, cfg.lr, dtype=float)
        if cfg.lr_feature_power:
            rms = np.sqrt(np.mean(expanded.design(batch.x) ** 2, axis=0))
            lr_vec *= (rms.max() / np.maximum(rms, 1e-12)) ** cfg.lr_feature_power
        if cfg.freeze_old:
            mask[:n_old] = False
        if cfg.composed:
            if cfg.scope == "new":
                mask[:n_old] = False
        else:
            lr_vec[:n_old] *= cfg.lr_old_factor
        seed_t = [cfg.seed, t]

        ref_x, null_basis, null_dim, g_w = None, None, None, None
        if cfg.composed:
            if cfg.preserve and t > 0:
                ref_x = buffer.reference_inputs(cfg.n_ref_per_exp)
                null_basis = preservation_null_basis(d, ref_x, mask)
                null_dim = null_basis.shape[1]
            res = train(
                PolynomialLogisticModel(d),
                batch.x,
                batch.y,
                lr_vec,
                cfg.epochs,
                cfg.batch_size,
                seed_t,
                mask,
                expanded.score(batch.x),
                null_basis,
            )
            after = expanded.with_added(res.model)  # f_old + g
            g_w = res.model.w.copy()
        else:
            res = train(
                expanded,
                batch.x,
                batch.y,
                lr_vec,
                cfg.epochs,
                cfg.batch_size,
                seed_t,
                mask,
            )
            after = res.model

        # ---- evaluation on every seen experience's fixed test split
        for i, e in enumerate(seen):
            acc[t, i] = accuracy(e.test.y, after.predict(e.test.x))
        fg = forgetting(acc, t)
        if t > 0:
            xo = np.concatenate([e.test.x for e in seen[:t]])
            sb, sa = model_before.score(xo), after.score(xo)
            mos, flip = old_score_change(sb, sa), prediction_flip_fraction(sb, sa)
            mean_fg, max_fg = float(fg[:t].mean()), float(fg[:t].max())
        else:
            mos = flip = mean_fg = max_fg = 0.0
        ref_chg = ref_flip = None
        if ref_x is not None and len(ref_x):
            sb, sa = expanded.score(ref_x), after.score(ref_x)
            ref_chg, ref_flip = (
                old_score_change(sb, sa),
                prediction_flip_fraction(sb, sa),
            )
        # A frozen-coefficient diagnostic is undefined when every coefficient is
        # trainable. Do not report 0.0 for an empty frozen set.
        fmax = None if np.all(mask) else parameter_changes(expanded.w, after.w, mask)[0]
        _, l1 = parameter_changes(expanded.w, after.w, mask)
        preserved_change = 0.0 if cfg.composed else None
        combined_change = float(np.max(np.abs(after.w - expanded.w)))

        records.append(
            StepRecord(
                t,
                model_before.degree,
                d,
                expanded.feature_names,
                model_before,
                expanded,
                after,
                g_w,
                expanded.w.copy(),
                after.w.copy(),
                res.lr_vector,
                mask,
                int(mask.sum()),
                null_dim,
                res.loss_history,
                batch.counts,
                ds,
                dp,
                acc[t, : t + 1].copy(),
                float(acc[t, t]),
                float(np.mean(acc[t, : t + 1])),
                fg,
                mean_fg,
                max_fg,
                mos,
                flip,
                ref_chg,
                ref_flip,
                fmax,
                preserved_change,
                combined_change,
                l1,
            )
        )

        buffer.add_experience(
            exp.train
        )  # only permitted (training) samples enter replay
        model = after
    return ContinualResult(cfg, list(experiences), records, acc)


def run_seeds(seeds, cfg: ContinualConfig, data_kwargs=None):
    """Repeat one configuration over several seeds (data + replay + shuffling all depend on the seed)."""
    out = []
    for s in seeds:
        exps = make_experiences(seed=s, **(data_kwargs or {}))
        out.append(run_continual(exps, replace(cfg, seed=s)))
    return out


def summary_table(result: ContinualResult):
    """One row (dict) per processed experience of a binary continual run."""
    rows = []
    for r in result.records:
        rows.append(
            {
                "exp": r.t + 1,
                "degree": r.degree_after,
                "trainable": r.n_trainable,
                "final_loss": r.loss_history[-1],
                "acc_current": r.acc_current,
                "mean_acc": r.mean_acc,
                "mean_forgetting": r.mean_forgetting,
                "max_forgetting": r.max_forgetting,
                "max_old_score_change": r.max_old_score_change,
                "old_flip_frac": r.old_flip_fraction,
                "frozen_max_change": r.frozen_max_change,
                "preserved_component_max_change": r.preserved_component_max_change,
                "combined_max_coefficient_change": r.combined_max_coefficient_change,
                "trainable_L1_update": r.trainable_l1_update,
            }
        )
    return rows


def run_joint_study(experiences, degrees, epochs_list, lr, seed=0):
    """Joint (non-continual) training of the binary model: one row per (degree, epochs).

    The training samples of all experiences are pooled and each model is trained from scratch
    on the same pool, with the same learning rate and seed; only the degree and the number of
    full-batch epochs differ. Accuracy is measured on every experience's fixed test split.
    There is no sequence of experiences here, so no forgetting is defined or reported.
    """
    x, y, _ = combine_splits([e.train for e in experiences])
    rows = []
    for degree in degrees:
        for epochs in epochs_list:
            result = train(
                PolynomialLogisticModel(degree), x, y, lr=lr, epochs=epochs, seed=seed
            )
            accs = [
                accuracy(e.test.y, result.model.predict(e.test.x)) for e in experiences
            ]
            rows.append(
                {
                    "degree": int(degree),
                    "epochs": int(epochs),
                    "n_features": result.model.n_features,
                    "final_loss": float(result.loss_history[-1]),
                    "mean_accuracy": float(np.mean(accs)),
                    "min_accuracy": float(np.min(accs)),
                    "accuracy_by_experience": accs,
                    "loss_history": result.loss_history,
                    "model": result.model,
                }
            )
    return rows


def _mean_std_text(values):
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.3f}+/-{std:.3f}"


def seed_summary_rows(results_by_label, key="variant"):
    """One table row per label from lists of runs over seeds (mean +/- sample std, ddof=1).

    ``results_by_label`` maps a label to the ``ContinualResult`` of every seed. The columns are
    the final mean accuracy, final accuracy on the newest and on the first experience, final
    forgetting, the mean over experiences of the old-evaluation-set score change, and the
    constrained dimension of the last experience (``n/a`` without a constraint).
    """
    rows = []
    for label, results in results_by_label.items():
        last = [r.records[-1] for r in results]
        null_dim = last[0].null_dim
        old_change = np.mean(
            [[x.max_old_score_change for x in r.records[1:]] for r in results]
        )
        rows.append(
            {
                key: label,
                "final_mean_acc": _mean_std_text([x.mean_acc for x in last]),
                "final_acc_current": _mean_std_text([x.acc_current for x in last]),
                "final_acc_E1": _mean_std_text([r.acc_matrix[-1, 0] for r in results]),
                "final_mean_forgetting": _mean_std_text(
                    [x.mean_forgetting for x in last]
                ),
                "mean_old_score_change": f"{old_change:.3f}",
                "null_dim_last": "n/a" if null_dim is None else null_dim,
            }
        )
    return rows


def step_rows(result):
    """Per-experience diagnostics of one run (loss reduction, trainable dimensions, constraint).

    ``ref_score_change`` is the change at the reference inputs (about 1e-15 for Strategy B by
    construction); ``old_score_change`` is measured on the whole old evaluation sets, which are
    not the reference inputs. ``null_dim`` is only defined for a constrained update.
    """
    nan = float("nan")
    return [
        {
            "exp": r.t + 1,
            "degree": r.degree_after,
            "null_dim": "n/a (unconstrained)" if r.null_dim is None else r.null_dim,
            "trainable_dims": r.n_trainable,
            "loss_reduction": float(r.loss_history[0] - r.loss_history[-1]),
            "trainable_l1_update": r.trainable_l1_update,
            "acc_current": r.acc_current,
            "mean_acc": r.mean_acc,
            "ref_score_change": nan
            if r.ref_score_change is None
            else r.ref_score_change,
            "old_score_change(eval sets)": r.max_old_score_change,
            "old_flip_frac": r.old_flip_fraction,
            "frozen_change": nan
            if r.frozen_max_change is None
            else r.frozen_max_change,
        }
        for r in result.records
    ]
