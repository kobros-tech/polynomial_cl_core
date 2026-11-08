# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Class-incremental comparison: fixed linear vs expanding vs fixed-capacity polynomial heads.

Entry point: ``run_comparison`` (see ``experiment.py`` for the protocol). The data it consumes comes
from ``polynomial_cl.data`` (``make_synthetic_2d``, ``load_projected_mnist``); the experience
sequence (``TASK_ORDER``) and the validation hold-out are defined here.
"""

from .degree_rates import N_DEGREE_GROUPS, as_degree_rates, degree_of_feature
from .degree_study import (
    candidate_vectors,
    degree_group_rows,
    load_study_cells,
    run_epoch_robustness,
    run_per_degree_study,
    run_sampling_comparison,
)
from .experiment import (
    CONDITIONS,
    TASK_ORDER,
    holdout_split,
    make_feature_data,
    make_task_plan,
    old_class_rows,
    per_seed_rows,
    require_diagnostics,
    run_comparison,
    run_condition,
    run_joint_comparison,
    run_joint_reference,
    run_strategy_comparison,
    sampling_rows,
    strategy_diagnostic_rows,
    summarize_runs,
    summary_rows,
)
from .strategies import STRATEGIES, update_restrictions

__all__ = [
    "CONDITIONS",
    "N_DEGREE_GROUPS",
    "STRATEGIES",
    "TASK_ORDER",
    "as_degree_rates",
    "candidate_vectors",
    "degree_group_rows",
    "degree_of_feature",
    "holdout_split",
    "load_study_cells",
    "make_feature_data",
    "make_task_plan",
    "old_class_rows",
    "per_seed_rows",
    "require_diagnostics",
    "run_comparison",
    "run_condition",
    "run_epoch_robustness",
    "run_joint_comparison",
    "run_joint_reference",
    "run_per_degree_study",
    "run_sampling_comparison",
    "run_strategy_comparison",
    "sampling_rows",
    "strategy_diagnostic_rows",
    "summarize_runs",
    "summary_rows",
    "update_restrictions",
]
