# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Degree-control experiment: does reducing single degrees' plasticity help?

Protocol (candidate selection on validation data only; the test set is touched once, for the
final configurations):

* ``BASELINES``: the three fixed configurations (rates, strategy, sampling) -- never changed.
* Experiment A (``individual_candidates``): per strategy, the all-one baseline, then one
  degree at a time with every factor of ``FACTORS`` below 1, the others at 1.
* Experiment B (``combination_candidates``): a *limited* set built from A, with no grid search.
  It has two parts. (i) Combinations: the best factor of each degree whose single-degree
  candidate beat the baseline's accuracy - forgetting *and* is feasible, then every combination
  of two or more of those degrees. (ii) The fixed ``EXAMPLE_VECTORS``, which are *always*
  evaluated: they do not depend on how their single-degree parts scored or whether those were
  feasible. Every candidate of B is measured on validation data like any other and is then
  eligible only on its own measured metrics (see the selection rule), so an example vector is
  never selected merely because it is an example.
* Experiment C (``final_comparison``): per strategy, baseline / best individual / best multiple /
  best overall, run on the test data with the test seeds, paired by seed. Two Pareto fronts of
  the validation candidates are reported under distinct names: ``pareto_feasible`` (only
  candidates that satisfy the acquisition constraint, i.e. the fronts the selection rule works
  on) and ``pareto_unrestricted`` (every measured candidate, feasible or not, for context only:
  it can contain candidates the rule would never select). Each also has an ``_individual``
  variant restricted to the single-degree candidates of Experiment A.

Selection rule (``choose_feasible``), applied on validation data only:

1. *Acquisition constraint.* A candidate is feasible only if, for **every** experience ``t``, its
   seed-mean accuracy on experience ``t`` measured right after ``t`` was learned
   (``accuracy_matrix[t][t]``) is at least ``ACQUISITION_FLOOR``. The floor is a constant of this
   module (and a command-line option), fixed before any candidate is evaluated and never derived
   from validation or test results. It is applied to seed means, not to each seed, and 0.5 is an
   experimental constraint set with earlier poor first-experience results in view, not an
   independently established threshold. Without it, accuracy minus forgetting rewards a configuration
   that does not learn an experience on arrival (so there is little to forget) and recovers it
   later through replay.
2. Among the feasible candidates: Pareto front of final accuracy and forgetting, then the largest
   accuracy - forgetting.
3. If no candidate is feasible, nothing is selected: the choice is ``None``, the result records
   ``"no_feasible_candidate"`` and the run says so. The least-bad candidate is never substituted.

The baseline (all factors 1) is always run on the test data and reported, feasible or not.

Immediate-accuracy quantities (the names say which aggregation is meant; they are different
numbers and are never reported under one name):

* ``immediate_accuracy_by_experience``: seed mean of ``accuracy_matrix[t][t]`` for every ``t``.
* ``min_mean_immediate_accuracy``: the minimum of that list. This is the quantity the acquisition
  floor is applied to (on validation data, for selection). It is also reported for the test runs,
  where it is informative only and never feeds back into selection.
* ``worst_immediate_accuracy``: a per-run value, the worst experience *within one run*; its
  ``summary`` entry is the mean and standard deviation of that value over seeds. The mean of the
  per-run minima is never larger than ``min_mean_immediate_accuracy`` (and is strictly smaller
  whenever the seeds disagree about which experience is worst), so the two are different
  numbers; the floor is never applied to this one.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from .degree_rates import N_DEGREE_GROUPS
from .degree_study import choose_candidate, pareto_front, tradeoff_score
from .experiment import (
    BATCH_SIZE,
    EPOCHS,
    MEMORY_PER_CLASS,
    N_REF_PER_CLASS,
    make_task_plan,
    run_condition,
)

FACTORS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
BASELINES = {
    "ordinary": {"strategy": "ordinary", "rates": [3.0, 3.0, 3.0, 3.0]},
    "frozen_old": {"strategy": "frozen_old", "rates": [2.5, 2.5, 10.0, 10.0]},
    "strategy_b": {"strategy": "strategy_b", "rates": [3.0, 3.0, 0.75, 3.0]},
}
SAMPLING = "class_balanced"
EXAMPLE_VECTORS = (
    (1, 0, 0.05, 1),
    (0, 1, 0.25, 1),
    (1, 0.05, 0.1, 1),
    (0.05, 0, 0.25, 1),
)
METRICS = ("final_mean_accuracy", "mean_forgetting", "final_current_accuracy")
# Per-run metrics that are summarised over seeds and compared pairwise with the baseline.
RUN_METRICS = (*METRICS, "worst_immediate_accuracy")
# Declared before selection, not tuned on validation or test data: every experience must be
# learned to at least this accuracy on arrival for a candidate to be eligible.
ACQUISITION_FLOOR = 0.5
ONES = (1.0,) * N_DEGREE_GROUPS


def name_of(factors):
    return "m[" + ",".join(f"{v:g}" for v in factors) + "]"


def individual_candidates(factors=FACTORS, n_groups=N_DEGREE_GROUPS):
    """``{name: factor vector}``: the all-one baseline and one degree changed at a time."""
    out = {name_of(ONES): ONES}
    for degree in range(n_groups):
        for f in factors:
            if f == 1.0:
                continue
            vec = [1.0] * n_groups
            vec[degree] = float(f)
            out[name_of(vec)] = tuple(vec)
    return out


def combination_candidates(individual_scores, examples=EXAMPLE_VECTORS, feasible=None):
    """Limited multi-degree vectors from Experiment A validation scores.

    ``individual_scores``: name -> tradeoff score for every vector of ``individual_candidates``.

    Combinations: a degree contributes only if its best single-degree factor beats the baseline
    score and, when ``feasible`` (a collection of candidate names) is given, is feasible; every
    combination of two or more contributing degrees is returned.

    ``examples`` are added unconditionally (when they change at least two degrees): they do not
    depend on the scores or the feasibility of their single-degree parts. That is intended, since
    the caller measures every returned vector and ``choose_feasible`` then decides on the
    vector's own metrics.
    """
    base = individual_scores[name_of(ONES)]
    best = {}
    for name, vec in individual_candidates().items():
        changed = [d for d, v in enumerate(vec) if v != 1.0]
        if feasible is not None and name not in feasible:
            continue
        if len(changed) == 1 and individual_scores[name] > base:
            d = changed[0]
            if d not in best or individual_scores[name] > best[d][0]:
                best[d] = (individual_scores[name], vec[d])
    out = {}

    def add(vec):
        vec = tuple(float(v) for v in vec)
        if vec != ONES and sum(v != 1.0 for v in vec) >= 2:
            out[name_of(vec)] = vec

    degrees = sorted(best)
    for size in range(2, len(degrees) + 1):
        for subset in itertools.combinations(degrees, size):
            vec = [1.0] * N_DEGREE_GROUPS
            for d in subset:
                vec[d] = best[d][1]
            add(vec)
    for vec in examples:
        add(vec)
    return out


def run_seeds(data, seeds, strategy, rates, factors, *, settings):
    """One strategy / rate vector / control vector for every seed; records get wall time."""
    runs = []
    for seed in seeds:
        tasks, memory = make_task_plan(data.train_y, seed, settings["memory_per_class"])
        started = time.perf_counter()
        run = run_condition(
            "expanding",
            data,
            tasks,
            memory,
            seed=seed,
            epochs=settings["epochs"],
            learning_rate=rates,
            batch_size=settings["batch_size"],
            strategy=strategy,
            n_ref_per_class=settings["n_ref_per_class"],
            sampling=SAMPLING,
            degree_control_factors=factors,
        )
        run["runtime_seconds"] = time.perf_counter() - started
        runs.append(with_immediate_accuracy(run))
    return runs


def immediate_accuracy(run):
    """Accuracy on every experience measured right after that experience was learned."""
    matrix = run["accuracy_matrix"]
    return [float(matrix[t][t]) for t in range(len(matrix))]


def immediate_accuracy_by_experience(runs):
    """Seed-mean immediate accuracy of every experience."""
    return np.mean([immediate_accuracy(r) for r in runs], axis=0).tolist()


def with_immediate_accuracy(run):
    """A shallow copy of ``run`` with the per-run ``worst_immediate_accuracy`` added.

    That is the worst experience of this one run. It is summarised over seeds by ``summarize``
    (a mean of per-run minima); it is not the quantity the acquisition floor uses.
    """
    return {**run, "worst_immediate_accuracy": min(immediate_accuracy(run))}


def min_mean_immediate_accuracy(runs):
    """Minimum over experiences of the seed-mean immediate accuracy: what the floor is applied to."""
    return float(min(immediate_accuracy_by_experience(runs)))


def mean_metrics(runs):
    """Seed means of ``METRICS`` plus the immediate-accuracy quantities the floor is applied to.

    ``min_mean_immediate_accuracy`` is the minimum over experiences of the *seed-mean* immediate
    accuracy (``immediate_accuracy_by_experience`` is the list it is the minimum of).
    """
    out = {m: float(np.mean([r[m] for r in runs])) for m in METRICS}
    out["min_mean_immediate_accuracy"] = min_mean_immediate_accuracy(runs)
    out["immediate_accuracy_by_experience"] = immediate_accuracy_by_experience(runs)
    return out


def is_feasible(metrics, floor):
    """True if every experience is learned to at least ``floor`` on arrival (seed means)."""
    return bool(metrics["min_mean_immediate_accuracy"] >= floor)


def choose_feasible(metrics_by_name, floor):
    """The selection rule: acquisition constraint first, then Pareto front, then trade-off.

    Returns ``None`` when no candidate satisfies the constraint; it never falls back to an
    infeasible candidate.
    """
    feasible = {n: m for n, m in metrics_by_name.items() if is_feasible(m, floor)}
    return choose_candidate(feasible) if feasible else None


def summarize(runs, metrics=(*RUN_METRICS, "runtime_seconds")):
    out = {}
    for m in metrics:
        v = np.array([r[m] for r in runs], dtype=float)
        out[m] = {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        }
    return out


def paired_difference(runs, baseline_runs, metric):
    """Per-seed difference ``runs - baseline`` (same seeds): mean, std, wins and n."""
    if [r["seed"] for r in runs] != [r["seed"] for r in baseline_runs]:
        raise ValueError("paired comparison needs identical seeds in identical order")
    d = np.array([a[metric] - b[metric] for a, b in zip(runs, baseline_runs)])
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean_diff": float(d.mean()),
        "std_diff": sd,
        "t": float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else None,
        "n": n,
        "by_seed": d.tolist(),
    }


def select_for_strategy(
    validation_data, select_seeds, key, *, settings, floor=ACQUISITION_FLOOR, log=None
):
    """Experiments A and B for one strategy on validation data (no test data involved).

    ``floor`` is the acquisition constraint of the module docstring. A ``best_*`` entry is
    ``None`` when no candidate of its pool is feasible.
    """
    base = BASELINES[key]
    strategy, rates = base["strategy"], base["rates"]
    metrics, vectors = {}, {}

    def measure(group, candidates):
        for name, factors in candidates.items():
            if name in metrics:
                continue
            runs = run_seeds(
                validation_data,
                select_seeds,
                strategy,
                rates,
                factors,
                settings=settings,
            )
            metrics[name] = mean_metrics(runs)
            vectors[name] = list(factors)
            if log:
                m = metrics[name]
                log(
                    f"  [{key}/{group}] {name}: acc {m['final_mean_accuracy']:.3f} "
                    f"forg {m['mean_forgetting']:.3f} newest {m['final_current_accuracy']:.3f} "
                    f"min-mean-on-arrival {m['min_mean_immediate_accuracy']:.3f} "
                    f"{'ok' if is_feasible(m, floor) else 'INFEASIBLE'}"
                )

    individual = individual_candidates()
    measure("A", individual)
    scores = {n: tradeoff_score(metrics[n]) for n in individual}
    feasible_individual = {n for n in individual if is_feasible(metrics[n], floor)}
    multiple = combination_candidates(scores, feasible=feasible_individual)
    measure("B", multiple)
    feasible = [n for n in metrics if is_feasible(metrics[n], floor)]
    best = {
        "best_individual": choose_feasible({n: metrics[n] for n in individual}, floor),
        "best_multiple": choose_feasible({n: metrics[n] for n in multiple}, floor)
        if multiple
        else None,
        "best_overall": choose_feasible(metrics, floor),
    }
    if log:
        for label, name in best.items():
            if name is None:
                log(f"  [{key}] {label}: NO FEASIBLE CANDIDATE (floor {floor:g})")
    return {
        "strategy": strategy,
        "rates": rates,
        "acquisition_floor": floor,
        "individual": list(individual),
        "multiple": list(multiple),
        "vectors": vectors,
        "validation_metrics": metrics,
        "feasible": feasible,
        "baseline_feasible": name_of(ONES) in feasible,
        "no_feasible_candidate": [
            label for label, name in best.items() if name is None
        ],
        # Fronts of the candidates the selection rule may choose from ...
        "pareto_feasible": pareto_front({n: metrics[n] for n in feasible}),
        "pareto_individual_feasible": pareto_front(
            {n: metrics[n] for n in individual if n in feasible_individual}
        ),
        # ... and, for context only, of every measured candidate (may contain infeasible ones).
        "pareto_unrestricted": pareto_front(metrics),
        "pareto_individual_unrestricted": pareto_front(
            {n: metrics[n] for n in individual}
        ),
        **best,
    }


def final_comparison(test_data, test_seeds, key, selection, *, settings, log=None):
    """Experiment C for one strategy: the (up to) four named configurations on the test set."""
    base = BASELINES[key]
    names = {
        "baseline": name_of(ONES),
        "best_individual": selection["best_individual"],
        "best_multiple": selection["best_multiple"],
        "best_overall": selection["best_overall"],
    }
    runs_by_vector = {}
    for label, name in names.items():
        if name is None or name in runs_by_vector:
            continue
        runs_by_vector[name] = run_seeds(
            test_data,
            test_seeds,
            base["strategy"],
            base["rates"],
            selection["vectors"][name],
            settings=settings,
        )
        if log:
            s = summarize(runs_by_vector[name])
            log(
                f"  [{key}/test] {label} {name}: acc {s['final_mean_accuracy']['mean']:.4f} "
                f"forg {s['mean_forgetting']['mean']:.4f}"
            )
    baseline_runs = runs_by_vector[names["baseline"]]
    rows = {}
    for label, name in names.items():
        if name is None:
            rows[label] = None
            continue
        runs = runs_by_vector[name]
        rows[label] = {
            "vector_name": name,
            "degree_control_factors": selection["vectors"][name],
            "degree_learning_rates": base["rates"],
            "effective_degree_learning_rates": runs[0][
                "effective_degree_learning_rates"
            ],
            "strategy": base["strategy"],
            "sampling": SAMPLING,
            "summary": summarize(runs),
            "immediate_accuracy_by_experience": immediate_accuracy_by_experience(runs),
            "min_mean_immediate_accuracy": min_mean_immediate_accuracy(runs),
            "paired_vs_baseline": {
                m: paired_difference(runs, baseline_runs, m)
                for m in (*RUN_METRICS, "runtime_seconds")
            },
            "seeds": [r["seed"] for r in runs],
        }
    return rows


def validate_seed_lists(select_seeds, test_seeds):
    """Return both seed lists as lists of ints, or raise ``ValueError`` saying what is wrong.

    Each list must be non-empty, hold non-negative integers (not bool) and have no duplicates, and
    the two lists must not share a seed: test seeds must never be used for selection.
    """
    checked = []
    for name, seeds in (("select_seeds", select_seeds), ("test_seeds", test_seeds)):
        if isinstance(seeds, (str, bytes)) or not hasattr(seeds, "__iter__"):
            raise ValueError(f"{name} must be a list of integer seeds, got {seeds!r}")
        seeds = list(seeds)
        if not seeds:
            raise ValueError(f"{name} must not be empty")
        for seed in seeds:
            if (
                isinstance(seed, (bool, np.bool_))
                or not isinstance(seed, (int, np.integer))
                or seed < 0
            ):
                raise ValueError(
                    f"{name} must hold non-negative integers, got {seed!r}"
                )
        duplicates = sorted({int(x) for x in seeds if seeds.count(x) > 1})
        if duplicates:
            raise ValueError(f"{name} contains duplicate seeds: {duplicates}")
        checked.append([int(x) for x in seeds])
    shared = sorted(set(checked[0]) & set(checked[1]))
    if shared:
        raise ValueError(
            f"select_seeds and test_seeds overlap on {shared}; they must be disjoint"
        )
    return checked[0], checked[1]


def run_degree_control_study(
    validation_data,
    test_data,
    select_seeds,
    test_seeds,
    *,
    strategies=tuple(BASELINES),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    memory_per_class=MEMORY_PER_CLASS,
    n_ref_per_class=N_REF_PER_CLASS,
    min_acquisition_accuracy=ACQUISITION_FLOOR,
    log=None,
):
    """Experiments A, B (validation) and C (test); selection and test seeds must be disjoint.

    ``min_acquisition_accuracy`` is the floor of the acquisition constraint; it must be fixed by the
    caller before any result is seen.
    """
    select_seeds, test_seeds = validate_seed_lists(select_seeds, test_seeds)
    strategies = tuple(strategies)
    unknown = [k for k in strategies if k not in BASELINES]
    if not strategies or unknown:
        raise ValueError(
            f"strategies must be a non-empty subset of {list(BASELINES)}"
            + (f"; unknown: {unknown}" if unknown else "")
        )
    if not (
        isinstance(min_acquisition_accuracy, (int, float))
        and not isinstance(min_acquisition_accuracy, bool)
        and 0.0 <= min_acquisition_accuracy <= 1.0
    ):
        raise ValueError("min_acquisition_accuracy must be a number in [0, 1]")
    settings = {
        "epochs": epochs,
        "batch_size": batch_size,
        "memory_per_class": memory_per_class,
        "n_ref_per_class": n_ref_per_class,
    }
    cells = []
    for key in strategies:
        if log:
            log(f"=== {key}: selection on validation data, seeds {list(select_seeds)}")
        selection = select_for_strategy(
            validation_data,
            select_seeds,
            key,
            settings=settings,
            floor=min_acquisition_accuracy,
            log=log,
        )
        if log:
            log(f"=== {key}: final comparison on test data, seeds {list(test_seeds)}")
        final = final_comparison(
            test_data, test_seeds, key, selection, settings=settings, log=log
        )
        cells.append({"key": key, "selection": selection, "final": final})
    return {
        "select_seeds": list(select_seeds),
        "test_seeds": list(test_seeds),
        "settings": settings,
        "sampling": SAMPLING,
        "factors": list(FACTORS),
        "min_acquisition_accuracy": min_acquisition_accuracy,
        "selection_rule": (
            "validation only: (1) every experience's seed-mean accuracy right after it is learned "
            ">= min_acquisition_accuracy; (2) among feasible candidates, Pareto front of accuracy "
            "and forgetting, then max accuracy - forgetting; (3) none feasible -> no selection"
        ),
        "cells": cells,
    }
