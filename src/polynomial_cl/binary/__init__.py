# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Binary sequential learning under spatial distribution shift (NOT class-incremental learning).

Two classes, a fixed label set, and a nonlinear ground-truth boundary
``y = 1 iff x2 > 0.5 sin(3 x1)``. Experience k only shows a vertical strip of the plane, so what
changes between experiences is *where* the samples come from, never which classes exist. A
binary logistic model (``polynomial_cl.models.binary``) grows in degree between experiences
(zero-initialised expansion) and is trained with replay and several update rules
(``STRATEGIES``) by ``polynomial_cl.training.binary``. This package holds the binary-specific
experience construction, replay buffer and sequential loop. The ten-class setting lives in
``polynomial_cl.continual``.
"""

from .data import (
    Experience,
    Split,
    combine_splits,
    generate_split,
    make_experiences,
)
from .experiment import (
    STRATEGIES,
    ContinualConfig,
    ContinualResult,
    StepRecord,
    config_with_defaults,
    degree_schedule,
    make_config,
    run_continual,
    run_joint_study,
    run_seeds,
    seed_summary_rows,
    step_rows,
    summary_table,
)
from .replay import ReplayBuffer, build_batch

__all__ = [
    "STRATEGIES",
    "ContinualConfig",
    "ContinualResult",
    "Experience",
    "ReplayBuffer",
    "Split",
    "StepRecord",
    "build_batch",
    "combine_splits",
    "config_with_defaults",
    "degree_schedule",
    "generate_split",
    "make_config",
    "make_experiences",
    "run_continual",
    "run_joint_study",
    "run_seeds",
    "seed_summary_rows",
    "step_rows",
    "summary_table",
]
