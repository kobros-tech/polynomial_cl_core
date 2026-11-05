# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Gradient-descent training of the binary polynomial model (NumPy only).

`train` supports a global or per-coefficient learning rate, a boolean trainable mask (frozen
coefficients are never touched; this is asserted), shuffled mini-batches with an explicit seed,
an optional fixed `offset` score (used to represent f = f_old + g), and an optional `null_basis`
that restricts every update to a subspace (Strategy B, see ``polynomial_cl.binary.experiment``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.binary import (
    PolynomialLogisticModel,
    bce_with_logits,
    n_features,
    polynomial_features,
    sigmoid,
)


# ============================================================ basic training
@dataclass
class TrainResult:
    model: PolynomialLogisticModel
    loss_history: np.ndarray  # loss before training, then after every epoch
    w_before: np.ndarray
    w_after: np.ndarray
    lr_vector: np.ndarray  # effective per-coefficient learning rate (0 if frozen)
    trainable: np.ndarray  # boolean mask


def _as_vector(lr, n):
    values = np.asarray(lr, dtype=float)
    if values.ndim == 0:
        values = np.full(n, float(values))
    elif values.shape != (n,):
        raise ValueError(
            f"lr must be a scalar or have shape ({n},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("learning rates must be finite and non-negative")
    return values.copy()


def positive_int(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return int(value)


def train(
    model,
    x,
    y,
    lr=0.5,
    epochs=100,
    batch_size=None,
    seed=0,
    trainable=None,
    offset=None,
    null_basis=None,
) -> TrainResult:
    """Gradient descent on mean BCE. The input model is not modified; a trained copy is returned.

    lr         scalar or per-coefficient vector (eta_i). Frozen coefficients get eta_i = 0.
    batch_size None = full batch; otherwise shuffled mini-batches (reshuffled every epoch,
               RNG = default_rng(seed)).
    offset     fixed per-sample score added to the model score (f = offset + w.phi(x)).
    null_basis (n_coef, k) matrix; each update step is projected onto its column space.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("x must be a non-empty 2D array")
    if y.shape != (len(x),) or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite 1D array with one label per row of x")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y must contain only binary labels 0 and 1")
    epochs = positive_int(epochs, "epochs", allow_zero=True)
    n_coef = model.n_features
    mask = (
        np.ones(n_coef, bool)
        if trainable is None
        else np.asarray(trainable, bool).copy()
    )
    if mask.shape != (n_coef,):
        raise ValueError(f"trainable must have shape ({n_coef},)")
    raw_lr = _as_vector(lr, n_coef)
    if null_basis is not None:
        null_basis = np.asarray(null_basis, dtype=float)
        if null_basis.ndim != 2 or null_basis.shape[0] != n_coef:
            raise ValueError(f"null_basis must have shape ({n_coef}, k)")
        if not np.all(np.isfinite(null_basis)):
            raise ValueError("null_basis must contain only finite values")
        if not np.allclose(
            null_basis.T @ null_basis, np.eye(null_basis.shape[1]), atol=1e-8
        ):
            raise ValueError("columns of null_basis must be orthonormal")
        # The trainer guarantees bit-for-bit preservation of frozen coefficients.
        # Therefore the projection basis must be exactly zero in those coordinates;
        # even tiny nonzero entries could otherwise move a frozen coefficient.
        if np.any(null_basis[~mask] != 0.0):
            raise ValueError(
                "null_basis must have exactly zero rows for frozen coefficients"
            )
        active = mask & np.any(np.abs(null_basis) > 1e-12, axis=1)
        active_rates = raw_lr[active]
        if active_rates.size and not np.allclose(active_rates, active_rates[0]):
            raise ValueError(
                "null_basis constraints require one shared learning rate across "
                "all trainable coefficients represented in the null space"
            )
    lr_vec = raw_lr * mask
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    if off.shape != (len(y),) or not np.all(np.isfinite(off)):
        raise ValueError("offset must be a finite 1D array with one value per sample")
    phi = model.design(x)
    w0 = model.w.copy()
    w = w0.copy()
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = n if batch_size is None else positive_int(batch_size, "batch_size")

    def full_loss():
        return bce_with_logits(off + phi @ w, y)

    hist = [full_loss()]
    for _ in range(int(epochs)):
        order = rng.permutation(n) if bs < n else np.arange(n)
        for s in range(0, n, bs):
            idx = order[s : s + bs]
            err = sigmoid(off[idx] + phi[idx] @ w) - y[idx]
            grad = phi[idx].T @ err / len(idx)
            step = lr_vec * grad
            if null_basis is not None:
                step = null_basis @ (null_basis.T @ step)
            w -= step
        hist.append(full_loss())
    assert np.array_equal(w[~mask], w0[~mask]), "a frozen coefficient changed"
    return TrainResult(
        PolynomialLogisticModel(model.degree, w), np.array(hist), w0, w, lr_vec, mask
    )


# ================================================== preservation constraint (B)
def preservation_null_basis(degree, x_ref, trainable=None, rtol=1e-10):
    """Orthonormal basis N (n_coef, k) of {v : Phi(x_ref) v = 0, v_i = 0 for frozen i}.

    Any g with coefficients v = N z satisfies g(x_ref) = 0 exactly (to rounding). If the
    constraints use up all degrees of freedom, k = 0 and g cannot learn anything - that is
    a result to report, not something to repair.
    """
    n = n_features(degree)
    mask = np.ones(n, bool) if trainable is None else np.asarray(trainable, bool)
    cols = np.flatnonzero(mask)
    x_ref = np.atleast_2d(x_ref)
    if len(x_ref) == 0 or len(cols) == 0:
        basis_t = np.eye(len(cols))
    else:
        a = polynomial_features(x_ref, degree)[:, cols]
        _, s, vt = np.linalg.svd(a, full_matrices=True)
        rank = int(np.sum(s > rtol * max(s[0], 1.0))) if len(s) else 0
        basis_t = vt[rank:].T
    basis = np.zeros((n, basis_t.shape[1]))
    basis[cols, :] = basis_t
    return basis
