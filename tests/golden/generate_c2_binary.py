# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Generate ``c2_binary_train.json``: the current behaviour of ``training.binary``.

The binary trainer still carries its own update (a projected step with one shared rate) and
its own preservation basis. This file freezes their results *before* they are replaced by calls
to the ``stable_cl`` package, so the replacement can be checked against the numbers produced
by the code that is being replaced. Usage, from the repository root::

    PYTHONPATH=src python tests/golden/generate_c2_binary.py \\
        --commit <core commit the numbers come from> --out tests/golden

Inputs (samples, offsets, masks, rate vectors, reference inputs, bases) are stored in the file.
The mini-batch cases draw their order from ``numpy.random.default_rng(seed).permutation``; the
first permutation is stored as a canary so that a NumPy release that changes the stream makes
the test skip those cases explicitly instead of failing for an unrelated reason.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from polynomial_cl.models.binary import (
    PolynomialLogisticModel,
    n_features,
    polynomial_features,
)
from polynomial_cl.training.binary import preservation_null_basis, train

N_SAMPLES = 30
OLD_DEGREE, NEW_DEGREE = 2, 3
SEED = [3, 1]  # as run_continual passes [cfg.seed, t]
BASE_LR = 0.4  # the learning rate every rate vector below is built from
VARIANTS = {
    "single": {"epochs": 1, "batch_size": None},
    "multi": {"epochs": 4, "batch_size": None},
    "sampling": {"epochs": 3, "batch_size": 7},
}


def listed(array):
    return None if array is None else np.asarray(array).tolist()


def build_inputs():
    rng = np.random.default_rng(21)
    x = rng.uniform(-1, 1, size=(N_SAMPLES, 2))
    y = (np.sin(3 * x[:, 0]) + 0.5 * x[:, 1] > 0).astype(float)
    x_ref = rng.uniform(-1, 1, size=(2, 2))  # reference inputs of old experiences
    n_old, n_new = n_features(OLD_DEGREE), n_features(NEW_DEGREE)
    w_old = 0.5 * rng.normal(size=n_old)
    old_model = PolynomialLogisticModel(OLD_DEGREE, w_old)
    # the model before the experience (the same function, with zero coefficients appended)
    expanded = old_model.expanded(NEW_DEGREE)
    return x, y, x_ref, n_old, n_new, old_model, expanded


def feature_rates(expanded, x, lr, power):
    """lr * (max rms / rms) ** power per coefficient, as run_continual builds it."""
    rms = np.sqrt(np.mean(expanded.design(x) ** 2, axis=0))
    return lr * (rms.max() / np.maximum(rms, 1e-12)) ** power


def cases(x, x_ref, n_old, n_new, expanded):
    lr = BASE_LR
    all_trainable = np.ones(n_new, bool)
    new_only = np.ones(n_new, bool)
    new_only[:n_old] = False
    frozen_old = new_only.copy()
    old_slow = np.full(n_new, lr)
    old_slow[:n_old] *= 0.25
    zero_degree_three = np.full(n_new, lr)
    zero_degree_three[n_old:] = 0.0  # a coefficient group with rate 0
    return [
        # composed f = f_old + g: g starts at 0, offset carries f_old; shared rate lr
        ("composed_all_preserve", "all", True, True, np.full(n_new, lr), all_trainable),
        ("composed_new_preserve", "new", True, True, np.full(n_new, lr), new_only),
        ("composed_all_free", "all", True, False, np.full(n_new, lr), all_trainable),
        ("composed_new_free", "new", True, False, np.full(n_new, lr), new_only),
        # plain training of the expanded model (no offset, no basis)
        ("ordinary_shared", None, False, False, np.full(n_new, lr), all_trainable),
        ("ordinary_old_factor", None, False, False, old_slow, all_trainable),
        ("ordinary_frozen_old", None, False, False, np.full(n_new, lr), frozen_old),
        (
            "ordinary_zero_rate_group",
            None,
            False,
            False,
            zero_degree_three,
            all_trainable,
        ),
        (
            "ordinary_feature_power",
            None,
            False,
            False,
            feature_rates(expanded, x, lr, 0.5),
            all_trainable,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    x, y, x_ref, n_old, n_new, old_model, expanded = build_inputs()
    offset = expanded.score(x)
    fixtures = []
    for name, scope, composed, preserve, lr_vec, mask in cases(
        x, x_ref, n_old, n_new, expanded
    ):
        basis = preservation_null_basis(NEW_DEGREE, x_ref, mask) if preserve else None
        base = (
            PolynomialLogisticModel(NEW_DEGREE) if composed else expanded
        )  # composed: g starts at zero
        runs = {}
        for label, setting in VARIANTS.items():
            result = train(
                base,
                x,
                y,
                lr_vec,
                setting["epochs"],
                setting["batch_size"],
                SEED,
                mask,
                offset if composed else None,
                basis,
            )
            runs[label] = {
                **setting,
                "w_after": result.w_after.tolist(),
                "lr_vector": result.lr_vector.tolist(),
                "loss_history": result.loss_history.tolist(),
            }
        fixtures.append(
            {
                "name": name,
                "scope": scope,
                "composed": composed,
                "preserve": preserve,
                "lr": lr_vec.tolist(),
                "trainable": mask.tolist(),
                "w_before": base.w.tolist(),
                "null_basis": listed(basis),
                "runs": runs,
            }
        )
    payload = {
        "meta": {
            "schema": 1,
            "what": "training.binary.train results of the current implementation",
            "generated_from": {
                "repository": "kobros-tech/polynomial_cl_core",
                "commit": args.commit,
                "state": "binary trainer with its own projected step (not yet delegating)",
            },
            "generator": "tests/golden/generate_c2_binary.py",
            "numpy": np.__version__,
            "python": platform.python_version(),
            "permutation_canary": {
                "seed": SEED,
                "n": N_SAMPLES,
                "first_permutation": np.random.default_rng(SEED)
                .permutation(N_SAMPLES)
                .tolist(),
            },
        },
        "problem": {
            "old_degree": OLD_DEGREE,
            "new_degree": NEW_DEGREE,
            "seed": SEED,
            "base_lr": BASE_LR,
            "x": x.tolist(),
            "y": y.tolist(),
            "x_ref": x_ref.tolist(),
            "offset": offset.tolist(),
            "old_w": old_model.w.tolist(),
            "design_ref": polynomial_features(x_ref, NEW_DEGREE).tolist(),
        },
        "cases": fixtures,
    }
    path = args.out / "c2_binary_train.json"
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
