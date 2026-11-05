# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Matplotlib plots, imported on demand so that Matplotlib is only loaded when it is used.

* ``plotting.synthetic``: the 2-D binary experiments (true and learned decision regions, samples,
  coefficients, training curves, sequential-run histories).
* ``plotting.continual``: the class-incremental experiments (accuracy, forgetting, per-class
  accuracy, strategy and degree diagnostics).

The two modules share no helper because their axes, inputs and conventions differ.
"""
