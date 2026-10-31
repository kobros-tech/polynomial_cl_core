# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Class-incremental comparison: fixed linear vs expanding vs fixed-capacity polynomial heads.

Entry points: ``make_synthetic_2d`` / ``load_projected_mnist`` for data and ``run_comparison``
for the experiment (see ``experiment.py`` for the protocol).
"""

from .data import TASK_ORDER, holdout_split, load_projected_mnist, make_synthetic_2d
from .degree_rates import N_DEGREE_GROUPS, as_degree_rates, degree_of_feature
from .degree_study import (
    candidate_vectors,
    load_study_cells,
    run_epoch_robustness,
    run_per_degree_study,
    run_sampling_comparison,
)
from .experiment import (
    CONDITIONS,
    STRATEGIES,
    make_feature_data,
    make_task_plan,
    run_comparison,
    run_condition,
    run_joint_reference,
    run_strategy_comparison,
    summarize_runs,
)

__all__ = [
    "CONDITIONS",
    "N_DEGREE_GROUPS",
    "STRATEGIES",
    "TASK_ORDER",
    "as_degree_rates",
    "candidate_vectors",
    "degree_of_feature",
    "holdout_split",
    "load_projected_mnist",
    "load_study_cells",
    "make_feature_data",
    "make_synthetic_2d",
    "make_task_plan",
    "run_comparison",
    "run_condition",
    "run_epoch_robustness",
    "run_joint_reference",
    "run_per_degree_study",
    "run_sampling_comparison",
    "run_strategy_comparison",
    "summarize_runs",
]
