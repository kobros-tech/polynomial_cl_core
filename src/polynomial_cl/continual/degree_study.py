# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Sampling-policy comparison and the compact per-degree learning-rate study.

Protocol (all selection on validation data, never on the test set):

1. ``run_sampling_comparison``: the same methods, rates, seeds and replay memory under the
   existing ``shuffled`` sampling and under ``class_balanced`` sampling.
2. ``select_rate_vectors`` for one (method, sampling) cell, on a validation split carved out
   of the training set with *selection* seeds:

   * stage 1, shared reference: the best scalar rate from ``SHARED_RATE_GRID``;
   * stage 2, independent vectors: from that reference, one degree group at a time is made
     ``RATE_FACTORS`` times larger or smaller (``candidate_vectors``). Groups are not tied to
     each other and no ordering between degrees is imposed; the vector for one run is applied
     to all groups simultaneously;
   * stage 3, limited follow-up: every degree group takes the factor that was best for it on
     validation (``combine_best_directions``); this is one additional vector, not a grid.

3. ``evaluate_vectors``: the candidate vectors are then run on the test data with *test* seeds
   that are disjoint from the selection seeds. The others are reported too, so the rate vectors
   can be read alongside their results.

Selection rule (validation data only; accuracy alone is never the criterion): every candidate
is scored on final mean accuracy (higher is better), mean forgetting (lower is better) and
final accuracy on the newest experience. ``pareto_front`` lists the candidates that no other
candidate beats on both accuracy and forgetting. Where a single vector is needed (the shared
reference of stage 1, the direction of each group in stage 3, the vector marked ``selected``)
the rule is: keep the candidates whose newest-experience accuracy is at least
``min_current_accuracy`` (default: no floor), restrict to the Pareto front, and take the one
with the largest ``accuracy - forgetting`` (``tradeoff_score``). The whole Pareto front is
reported; ``selected`` is a convenience for plots, not a claim of optimality.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from ..metrics import mean_std_by_condition
from .degree_rates import N_DEGREE_GROUPS
from .experiment import (
    BATCH_SIZE,
    EPOCHS,
    MEMORY_PER_CLASS,
    METHODS,
    N_REF_PER_CLASS,
    STRATEGY_METRICS,
    make_task_plan,
    recorded_metrics,
    require_diagnostics,
    run_condition,
)
from .strategies import LEGACY_STRATEGY_NAMES, check_method

SHARED_RATE_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
RATE_FACTORS = (0.25, 0.5, 2.0, 4.0)
SELECTION_METRICS = ("final_mean_accuracy", "mean_forgetting", "final_current_accuracy")
TRADEOFF_METRICS = (*STRATEGY_METRICS, "final_current_accuracy")


def _run_seeds(data, seeds, method, rates, *, sampling, settings, log=None, label=""):
    """Run one method with one learning-rate setting for every seed (paired plans).

    ``log`` (optional callable taking a string) receives one line per finished run.
    """
    if not len(seeds):
        raise ValueError("at least one seed is required")
    runs = []
    for seed in seeds:
        started = time.perf_counter()
        tasks, memory = make_task_plan(data.train_y, seed, settings["memory_per_class"])
        runs.append(
            run_condition(
                "expanding",
                data,
                tasks,
                memory,
                seed=seed,
                epochs=settings["epochs"],
                learning_rate=rates,
                batch_size=settings["batch_size"],
                method=method,
                beta=settings["beta"] if method == "relaxed" else 0.0,
                replay=settings["replay"],
                n_ref_per_class=settings["n_ref_per_class"],
                sampling=sampling,
                diagnostics=settings["diagnostics"],
            )
        )
        if log:
            run = runs[-1]
            log(
                f"    {label} seed {seed}: acc {run['final_mean_accuracy']:.3f} "
                f"forg {run['mean_forgetting']:.3f} "
                f"({time.perf_counter() - started:.1f}s)"
            )
    return runs


def _settings(
    epochs,
    batch_size,
    memory_per_class,
    n_ref_per_class,
    diagnostics=False,
    beta=0.0,
    replay=True,
):
    return {
        "diagnostics": diagnostics,
        "beta": beta,
        "replay": replay,
        "epochs": epochs,
        "batch_size": batch_size,
        "memory_per_class": memory_per_class,
        "n_ref_per_class": n_ref_per_class,
    }


def mean_metric(runs, metric):
    """Mean of one scalar metric over runs."""
    return float(np.mean([r[metric] for r in runs]))


def tradeoff_score(metrics):
    """Accuracy minus forgetting: rewards learning and penalises losing old classes."""
    return metrics["final_mean_accuracy"] - metrics["mean_forgetting"]


def pareto_front(metrics_by_name):
    """Names whose (accuracy, forgetting) no other candidate beats on both, in input order."""
    front = []
    for name, m in metrics_by_name.items():
        dominated = any(
            o["final_mean_accuracy"] >= m["final_mean_accuracy"]
            and o["mean_forgetting"] <= m["mean_forgetting"]
            and (
                o["final_mean_accuracy"] > m["final_mean_accuracy"]
                or o["mean_forgetting"] < m["mean_forgetting"]
            )
            for other, o in metrics_by_name.items()
            if other != name
        )
        if not dominated:
            front.append(name)
    return front


def choose_candidate(metrics_by_name, min_current_accuracy=None):
    """Selection rule of the module docstring; returns the chosen candidate name.

    If no candidate reaches ``min_current_accuracy`` the floor is ignored for that choice
    (a floor must not leave nothing to choose from).
    """
    names = list(metrics_by_name)
    if min_current_accuracy is not None:
        passing = [
            n
            for n in names
            if metrics_by_name[n]["final_current_accuracy"] >= min_current_accuracy
        ]
        names = passing or names
    front = pareto_front({n: metrics_by_name[n] for n in names})
    return max(front, key=lambda n: tradeoff_score(metrics_by_name[n]))


def run_sampling_comparison(
    data,
    seeds,
    *,
    policies=("shuffled", "class_balanced"),
    methods=METHODS,
    epochs=EPOCHS,
    learning_rate=0.1,
    batch_size=BATCH_SIZE,
    memory_per_class=MEMORY_PER_CLASS,
    n_ref_per_class=N_REF_PER_CLASS,
    diagnostics=False,
    beta=0.0,
    replay=True,
    log=None,
):
    """Same methods, rate, seeds and replay memory under each sampling policy."""
    settings = _settings(
        epochs, batch_size, memory_per_class, n_ref_per_class, diagnostics, beta, replay
    )
    out = {}
    for policy in policies:
        runs = [
            run
            for method in methods
            for run in _run_seeds(
                data,
                seeds,
                method,
                learning_rate,
                sampling=policy,
                settings=settings,
                log=log,
                label=f"[{policy}/{method}]",
            )
        ]
        out[policy] = {
            "runs": runs,
            "summary": mean_std_by_condition(
                runs, recorded_metrics(runs, TRADEOFF_METRICS), key="method"
            ),
        }
    return out


def candidate_vectors(base, factors=RATE_FACTORS, n_groups=N_DEGREE_GROUPS):
    """Reference vector plus one-group-at-a-time variants; name -> rate vector.

    Names are ``shared`` and ``d<k>x<factor>`` (degree group k scaled by factor).
    """
    vectors = {"shared": [float(base)] * n_groups}
    for group in range(n_groups):
        for factor in factors:
            vector = [float(base)] * n_groups
            vector[group] = float(base) * factor
            vectors[f"d{group}x{factor:g}"] = vector
    return vectors


def combine_best_directions(
    base, scores, factors=RATE_FACTORS, n_groups=N_DEGREE_GROUPS
):
    """Follow-up vector: each group takes the factor (or 1) that scored best for it.

    ``scores`` maps a candidate name from ``candidate_vectors`` to its validation score
    (``tradeoff_score``, higher is better); a
    group keeps the base rate unless one of its variants beats the shared reference.
    """
    vector = [float(base)] * n_groups
    for group in range(n_groups):
        best_name, best_score = "shared", scores["shared"]
        for factor in factors:
            name = f"d{group}x{factor:g}"
            if scores[name] > best_score:
                best_name, best_score = name, scores[name]
        if best_name != "shared":
            vector[group] = float(base) * float(best_name.split("x")[1])
    return vector


def select_rate_vectors(
    validation_data,
    selection_seeds,
    method,
    sampling,
    *,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    memory_per_class=MEMORY_PER_CLASS,
    n_ref_per_class=N_REF_PER_CLASS,
    shared_grid=SHARED_RATE_GRID,
    factors=RATE_FACTORS,
    min_current_accuracy=None,
    diagnostics=False,
    beta=0.0,
    replay=True,
    log=None,
):
    """Stages 1-3 on validation data for one (method, sampling) cell.

    Returns the candidate vectors (name -> vector), their validation metrics (mean over the
    selection seeds), the Pareto front, and the name chosen by the selection rule. The
    selection reads only accuracy and forgetting, so ``diagnostics`` (off by default) only
    decides what the validation runs record.
    """
    settings = _settings(
        epochs, batch_size, memory_per_class, n_ref_per_class, diagnostics, beta, replay
    )

    tag = f"[{method}/{sampling}]"

    def say(message):
        if log:
            log(message)

    def measure(rates, what):
        say(f"  {tag} validation: {what}")
        runs = _run_seeds(
            validation_data,
            selection_seeds,
            method,
            rates,
            sampling=sampling,
            settings=settings,
            log=log,
            label=what,
        )
        result = {m: mean_metric(runs, m) for m in SELECTION_METRICS}
        say(
            f"    -> acc {result['final_mean_accuracy']:.3f} "
            f"forg {result['mean_forgetting']:.3f} "
            f"newest {result['final_current_accuracy']:.3f}"
        )
        return result

    say(f"{tag} stage 1: shared-rate grid {list(shared_grid)}")
    grid = {float(lr): measure(float(lr), f"shared lr={lr:g}") for lr in shared_grid}
    base = choose_candidate({str(k): v for k, v in grid.items()}, min_current_accuracy)
    base = float(base)
    vectors = candidate_vectors(base, factors)
    metrics = {"shared": grid[base]}
    say(f"{tag} stage 2: base rate {base:g}, {len(vectors) - 1} one-group vectors")
    for name, vector in vectors.items():
        if name != "shared":
            metrics[name] = measure(vector, f"{name} {vector}")
    scores = {name: tradeoff_score(m) for name, m in metrics.items()}
    combined = combine_best_directions(base, scores, factors)
    if combined != vectors["shared"] and combined not in vectors.values():
        vectors["combined"] = combined
        say(f"{tag} stage 3: follow-up vector")
        metrics["combined"] = measure(combined, f"combined {combined}")
        scores["combined"] = tradeoff_score(metrics["combined"])
    return {
        "shared_grid": {str(k): v for k, v in grid.items()},
        "base_rate": base,
        "vectors": vectors,
        "validation_metrics": metrics,
        "validation_scores": scores,
        "pareto": pareto_front(metrics),
        "selected": choose_candidate(metrics, min_current_accuracy),
        "min_current_accuracy": min_current_accuracy,
    }


def evaluate_vectors(
    test_data,
    test_seeds,
    method,
    sampling,
    vectors,
    *,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    memory_per_class=MEMORY_PER_CLASS,
    n_ref_per_class=N_REF_PER_CLASS,
    diagnostics=False,
    beta=0.0,
    replay=True,
    log=None,
):
    """Run every named vector on the test data; returns name -> list of run records."""
    settings = _settings(
        epochs, batch_size, memory_per_class, n_ref_per_class, diagnostics, beta, replay
    )
    out = {}
    for i, (name, vector) in enumerate(vectors.items(), 1):
        if log:
            log(f"  [{method}/{sampling}] test {i}/{len(vectors)}: {name} {vector}")
        out[name] = _run_seeds(
            test_data,
            test_seeds,
            method,
            vector,
            sampling=sampling,
            settings=settings,
            log=log,
            label=name,
        )
    return out


def run_per_degree_study(
    validation_data,
    test_data,
    selection_seeds,
    test_seeds,
    *,
    methods=METHODS,
    samplings=("shuffled", "class_balanced"),
    progress=None,
    min_current_accuracy=None,
    log=None,
    **settings,
):
    """Selection on validation data, then test evaluation, for every (method, sampling).

    The selection and test seeds must be disjoint. Returns a JSON-serialisable dict. Settings
    (``epochs``, ``batch_size``, ``memory_per_class``, ``n_ref_per_class``, ``diagnostics``
    (off by default), ``beta`` (relaxed method only, default 0.0) and ``replay`` (default
    True)) are passed on to the selection and test runs. A cell is identified by
    ``(method, sampling)``: one beta per method. Before several betas are compared, beta must
    become part of that identity (``load_study_cells`` refuses duplicate identities).
    """
    shared = sorted(set(selection_seeds) & set(test_seeds))
    if shared:
        raise ValueError(
            f"selection and test seeds overlap on {shared}: the same stochastic runs "
            "would choose the rates and then be reported as the test estimate"
        )
    cells = []
    for sampling in samplings:
        for method in methods:
            selection = select_rate_vectors(
                validation_data,
                selection_seeds,
                method,
                sampling,
                min_current_accuracy=min_current_accuracy,
                log=log,
                **settings,
            )
            runs = evaluate_vectors(
                test_data,
                test_seeds,
                method,
                sampling,
                selection["vectors"],
                log=log,
                **settings,
            )
            cell = {
                "api_generation": "current",
                "method": method,
                "beta": float(settings.get("beta", 0.0))
                if method == "relaxed"
                else None,
                "replay": bool(settings.get("replay", True)),
                "sampling": sampling,
                **selection,
                "runs": runs,
            }
            cells.append(cell)
            if progress:
                progress(cell)
    return {
        "api_generation": "current",
        "selection_seeds": list(selection_seeds),
        "test_seeds": list(test_seeds),
        "selection_rule": "pareto(accuracy, forgetting), max accuracy - forgetting",
        "cells": cells,
    }


def vector_summary(runs, metrics=None):
    """Mean and sample std over seeds of the trade-off metrics for one list of runs.

    Default ``metrics``: ``TRADEOFF_METRICS`` for runs made with diagnostics, otherwise the
    accuracy and forgetting summary metrics (see ``recorded_metrics``).
    """
    out = {}
    if metrics is None:
        metrics = recorded_metrics(runs, TRADEOFF_METRICS)
    for metric in metrics:
        values = np.array([r[metric] for r in runs], dtype=float)
        out[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return out


def mean_degree_diagnostic(runs, key):
    """(n_experiences, n_degree_groups) mean over runs of a degree diagnostic (NaN if absent)."""
    for r in runs:
        require_diagnostics(r, "mean_degree_diagnostic")
    stacked = np.array(
        [
            [
                [np.nan if v is None else v for v in row]
                for row in r["degree_diagnostics"][key]
            ]
            for r in runs
        ],
        dtype=float,
    )
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # all-NaN column = inactive group
        return np.nanmean(stacked, axis=0)


def _read_cell_payloads(directory):
    """Parsed ``cell_*.json`` payloads of ``directory`` (None if there are none)."""
    import json
    from pathlib import Path

    files = sorted(Path(directory).glob("cell_*.json"))
    if not files:
        return None
    payloads = [json.loads(f.read_text()) for f in files]
    seeds = {(tuple(p["selection_seeds"]), tuple(p["test_seeds"])) for p in payloads}
    if len(seeds) != 1:
        raise ValueError("cell files were produced with different seed sets")
    return payloads


def _merge_cells(payloads, cells, sort_key, name):
    identities = [(c[name], c["sampling"]) for c in cells]
    duplicates = sorted({i for i in identities if identities.count(i) > 1})
    if duplicates:
        raise ValueError(
            f"several cells share the identity ({name}, sampling) {duplicates}: a study "
            "cell is identified by it, so configurations that differ in anything else "
            "(for example beta) cannot be merged"
        )
    first = payloads[0]
    cells.sort(key=sort_key)
    return {
        "selection_seeds": first["selection_seeds"],
        "test_seeds": first["test_seeds"],
        "selection_rule": first["selection_rule"],
        "cells": cells,
    }


def load_study_cells(directory):
    """Merge the ``cell_*.json`` files written by ``experiments/per_degree_lr.py``.

    Returns the same structure as ``run_per_degree_study`` (``cells`` sorted by sampling and
    method), or None if the directory has no cell files. Seed sets must agree across files.

    Only current-generation files are accepted: every cell needs a ``method`` in ``METHODS``
    and no ``strategy`` field. A file written under the earlier strategy vocabulary is
    rejected, never reinterpreted as a current configuration; read it with
    ``load_legacy_study_cells``.
    """
    payloads = _read_cell_payloads(directory)
    if payloads is None:
        return None
    cells = [c for p in payloads for c in p["cells"]]
    for cell in cells:
        if "strategy" in cell or cell.get("api_generation") == "legacy":
            raise ValueError(
                f"cell {cell.get('strategy')!r}/{cell.get('sampling')!r} is from an earlier "
                "API generation (strategy vocabulary); load it with load_legacy_study_cells"
            )
        if "method" not in cell:
            raise ValueError(
                "cell has no 'method': not a current-generation study cell"
            )
        check_method(cell["method"])
    for payload in payloads:
        if payload.get("api_generation", "current") != "current":
            raise ValueError("cell file is not a current-generation study file")
    merged = _merge_cells(
        payloads, cells, lambda c: (c["sampling"], METHODS.index(c["method"])), "method"
    )
    return {"api_generation": "current", **merged}


# What a historical strategy means in the current vocabulary (None: no current equivalent).
# strategy_b -> relaxed beta=0 and ordinary_no_replay -> ordinary replay=False are the
# relations established by the C2 equivalence tests; the other three strategies were removed.
_LEGACY_EQUIVALENTS = {
    "ordinary": {"method": "ordinary", "beta": None, "replay": True},
    "ordinary_no_replay": {"method": "ordinary", "beta": None, "replay": False},
    "strategy_b": {"method": "relaxed", "beta": 0.0, "replay": True},
    "small_old_lr": None,
    "frozen_old": None,
    "strategy_b_new": None,
}


def load_legacy_study_cells(directory):
    """Read ``cell_*.json`` files written under the earlier strategy vocabulary.

    Historical records are preserved exactly: every cell keeps its ``strategy`` name and its
    runs. Each returned cell is marked ``api_generation = "legacy"``; so is the study. A cell
    additionally gets ``equivalent_current_configuration`` (a dict with ``method``, ``beta``
    and ``replay``, or None when the strategy was removed and has no current equivalent).
    That field is information only: the cell has no ``method`` key, so the analysis
    functions of this module (``study_rows``, ``run_epoch_robustness`` ...) refuse it rather
    than mix it with current results. Current-generation files are rejected here.
    Returns None if the directory has no cell files.
    """
    payloads = _read_cell_payloads(directory)
    if payloads is None:
        return None
    cells = []
    for payload in payloads:
        if payload.get("api_generation", "legacy") != "legacy":
            raise ValueError("cell file is not a legacy study file")
        for cell in payload["cells"]:
            if "method" in cell or cell.get("api_generation") == "current":
                raise ValueError(
                    f"cell {cell.get('method')!r}/{cell.get('sampling')!r} is current-"
                    "generation; load it with load_study_cells"
                )
            if cell.get("strategy") not in LEGACY_STRATEGY_NAMES:
                raise ValueError(f"unknown legacy strategy: {cell.get('strategy')!r}")
            cells.append(
                {
                    **cell,
                    "api_generation": "legacy",
                    "equivalent_current_configuration": _LEGACY_EQUIVALENTS[
                        cell["strategy"]
                    ],
                }
            )
    merged = _merge_cells(
        payloads,
        cells,
        lambda c: (c["sampling"], LEGACY_STRATEGY_NAMES.index(c["strategy"])),
        "strategy",
    )
    return {"api_generation": "legacy", **merged}


def require_current_study(study, what="this function"):
    """Raise unless ``study`` holds only current-generation cells (never mix generations)."""
    if study.get("api_generation") == "legacy" or any(
        c.get("api_generation") == "legacy" or "method" not in c for c in study["cells"]
    ):
        raise ValueError(
            f"{what} takes current-generation studies only; a legacy study (strategy "
            "vocabulary, from load_legacy_study_cells) is not mixed with current results"
        )


def run_epoch_robustness(
    test_data,
    study,
    epochs,
    *,
    test_seeds=None,
    batch_size=BATCH_SIZE,
    memory_per_class=MEMORY_PER_CLASS,
    n_ref_per_class=N_REF_PER_CLASS,
    methods=None,
    samplings=None,
    diagnostics=False,
    log=None,
):
    """Re-run the shared reference and the selected vector of every cell at another epoch count.

    The vectors are the ones chosen at the study's own epoch count (nothing is re-selected),
    so this asks whether the conclusion survives longer training, not what the best rates for
    that length are. ``study`` is the structure from ``run_per_degree_study`` or
    ``load_study_cells``; ``test_seeds`` defaults to the study's test seeds.
    """
    require_current_study(study, "run_epoch_robustness")
    seeds = list(study["test_seeds"] if test_seeds is None else test_seeds)
    if set(seeds) & set(study["selection_seeds"]):
        raise ValueError("robustness seeds must not include the selection seeds")
    cells = []
    for cell in study["cells"]:
        if methods is not None and cell["method"] not in methods:
            continue
        if samplings is not None and cell["sampling"] not in samplings:
            continue
        names = list(dict.fromkeys(["shared", cell["selected"]]))
        runs = evaluate_vectors(
            test_data,
            seeds,
            cell["method"],
            cell["sampling"],
            {n: cell["vectors"][n] for n in names},
            epochs=epochs,
            batch_size=batch_size,
            memory_per_class=memory_per_class,
            n_ref_per_class=n_ref_per_class,
            diagnostics=diagnostics,
            beta=cell.get("beta") or 0.0,
            replay=cell.get("replay", True),
            log=log,
        )
        cells.append(
            {
                "method": cell["method"],
                "beta": cell.get("beta"),
                "replay": cell.get("replay", True),
                "sampling": cell["sampling"],
                "epochs": epochs,
                "selected": cell["selected"],
                "vectors": {n: cell["vectors"][n] for n in names},
                "runs": runs,
            }
        )
    return {"epochs": epochs, "test_seeds": seeds, "cells": cells}


def degree_group_rows(run):
    """One row per (experience, active degree group) of one run: rate, size and actual change.

    ``trainable`` counts the coefficients that can move; a group whose rate is zero has none,
    and its measured ``coef_change_l1`` is exactly 0.
    """
    require_diagnostics(run, "degree_group_rows")
    d = run["degree_diagnostics"]
    rows = []
    for task, groups in enumerate(d["lr"]):
        for degree, rate in enumerate(groups):
            if rate is None:
                continue
            rows.append(
                {
                    "experience": task + 1,
                    "degree": degree,
                    "rate": rate,
                    "coefficients": d["n_coefficients"][task][degree],
                    "trainable": d["n_trainable"][task][degree],
                    "coef_change_l1": d["coef_change_l1"][task][degree],
                }
            )
    return rows


def study_rows(study):
    """Rows for the shared-rate and the selected vector of every cell of a per-degree study.

    ``*`` marks the vector selected on validation data and ``^`` a vector on the validation
    Pareto front of accuracy and forgetting.
    """
    require_current_study(study, "study_rows")
    rows = []
    for cell in study["cells"]:
        for name, runs in cell["runs"].items():
            if name not in ("shared", cell["selected"]):
                continue
            s = vector_summary(runs)
            row = {
                "sampling": cell["sampling"][:5],
                "method": cell["method"],
                "vector": name
                + ("*" if name == cell["selected"] else "")
                + ("^" if name in cell["pareto"] else ""),
                "rates d0..d3": " ".join(f"{v:g}" for v in cell["vectors"][name]),
                "acc": f"{s['final_mean_accuracy']['mean']:.3f}+/-{s['final_mean_accuracy']['std']:.3f}",
                "newest": f"{s['final_current_accuracy']['mean']:.3f}",
                "forget": f"{s['mean_forgetting']['mean']:.3f}",
            }
            if "mean_old_takeover_fraction" in s:  # runs made with diagnostics
                row["takeover"] = f"{s['mean_old_takeover_fraction']['mean']:.3f}"
                row["flips"] = f"{s['mean_old_flip_fraction']['mean']:.3f}"
            rows.append(row)
    return rows


def study_pairs(study):
    """``{label: (runs, sampling)}`` for the shared and selected vector of every study cell."""
    require_current_study(study, "study_pairs")
    pairs = {}
    for cell in study["cells"]:
        tag = f"{cell['method']}/{cell['sampling'][:5]}"
        pairs[f"{tag} shared"] = (cell["runs"]["shared"], cell["sampling"])
        pairs[f"{tag} {cell['selected']}"] = (
            cell["runs"][cell["selected"]],
            cell["sampling"],
        )
    return pairs
