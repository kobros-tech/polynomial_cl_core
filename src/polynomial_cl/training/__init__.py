# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Optimisers. ``training.binary`` trains the binary model; ``training.multiclass`` trains the
softmax coefficient array of the class-incremental experiments. They share no trainer because
their loss, outputs and update semantics differ.
"""
