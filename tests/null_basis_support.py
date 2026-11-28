# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Test helper: a float32 null-space basis built with ``stable_cl``.

The package itself has no null-space code; the tests need a basis of ``{v : design v = 0}`` to
hand to ``fit_task``. This helper is ``stable_cl.null_basis`` with unit rates, cast to float32
as the strategies do, and exists only to keep those tests short.
"""

import numpy as np
import stable_cl


def null_space_basis(design, rtol=1e-10):
    design = np.asarray(design, dtype=np.float64)
    basis = stable_cl.null_basis(design, np.ones(design.shape[1]), rtol)
    return basis.astype(np.float32)
