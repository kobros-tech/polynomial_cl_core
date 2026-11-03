# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Binary logistic regression on explicit polynomial features (NumPy only).

f_d(x) = w . phi_d(x).  The bias is the coefficient of the constant feature (w[0]); it is
not stored separately, so it can never be counted twice.

The feature map is the *raw* 2-D monomial map in float64 (x1^i * x2^j). It shares only the
monomial enumeration and the zero-coefficient expansion with the class-incremental code
(``polynomial_cl.features``); the class-incremental setting uses its own degree-scaled map
(``polynomial_cl.continual.features``) and a softmax head instead of this model.
"""

from __future__ import annotations

import numpy as np

from ..features import append_zero_coefficients, monomial_exponents, validate_degree


# ------------------------------------------------------------ 2-D raw feature map
def feature_exponents(degree: int):
    """List of (i, j) meaning x1^i * x2^j, in the canonical order."""
    degree = validate_degree(degree)
    return [(int(i), int(j)) for i, j in monomial_exponents(2, degree)]


def _name(i: int, j: int) -> str:
    parts = []
    for var, p in (("x1", i), ("x2", j)):
        if p == 1:
            parts.append(var)
        elif p > 1:
            parts.append(f"{var}^{p}")
    return "*".join(parts) if parts else "1"


def feature_names(degree: int):
    return [_name(i, j) for i, j in feature_exponents(degree)]


def n_features(degree: int) -> int:
    degree = validate_degree(degree)
    return (degree + 1) * (degree + 2) // 2


def polynomial_features(x, degree: int) -> np.ndarray:
    """Design matrix Phi_d(x) of shape (n, n_features(degree))."""
    degree = validate_degree(degree)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 2:
        raise ValueError("expected inputs of shape (n, 2)")
    p = np.arange(degree + 1)
    p1 = x[:, :1] ** p  # (n, d+1): x1^0..x1^d
    p2 = x[:, 1:] ** p
    return np.column_stack([p1[:, i] * p2[:, j] for i, j in feature_exponents(degree)])


class PolynomialFeatures:
    """Small convenience wrapper exposing degree, names, and feature count."""

    def __init__(self, degree: int):
        self.degree = validate_degree(degree)
        self.exponents = feature_exponents(self.degree)
        self.names = feature_names(self.degree)
        self.n_features = len(self.names)

    def __call__(self, x):
        return polynomial_features(x, self.degree)


# ------------------------------------------------------------ the model
def sigmoid(z):
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def bce_with_logits(z, y):
    """Mean binary cross-entropy computed from logits without log(sigmoid(.)) underflow:
    max(z,0) - z*y + log(1 + exp(-|z|))."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


class PolynomialLogisticModel:
    def __init__(self, degree: int, w=None):
        self.features = PolynomialFeatures(degree)
        n = self.features.n_features
        if w is None:
            self.w = np.zeros(n)
        else:
            w = np.array(w, dtype=float)
            if w.shape != (n,):
                raise ValueError(f"expected {n} coefficients for degree {degree}")
            self.w = w

    # introspection -------------------------------------------------------
    @property
    def degree(self):
        return self.features.degree

    @property
    def feature_names(self):
        return self.features.names

    @property
    def n_features(self):
        return self.features.n_features

    @property
    def bias(self):
        return self.w[0]

    def coefficients(self):
        return dict(zip(self.feature_names, self.w))

    # inference -----------------------------------------------------------
    def design(self, x):
        return self.features(x)

    def score(self, x):
        return self.design(x) @ self.w

    def proba(self, x):
        return sigmoid(self.score(x))

    def predict(self, x):
        return (self.score(x) > 0).astype(int)

    # structure -----------------------------------------------------------
    def copy(self):
        return PolynomialLogisticModel(self.degree, self.w.copy())

    def expanded(self, new_degree: int):
        """Expansion invariant: keep every existing coefficient exactly, append zeros.
        No training happens here; f_{d+1}(x) == f_d(x) for all x."""
        if new_degree < self.degree:
            raise ValueError("can only expand to an equal or larger degree")
        new = PolynomialLogisticModel(new_degree)
        assert (
            new.feature_names[: self.n_features] == self.feature_names
        )  # prefix property
        new.w = append_zero_coefficients(self.w, new.n_features)
        return new

    def with_added(self, g: PolynomialLogisticModel):
        """Combined model f_old + g (g is padded with zeros if it has lower degree)."""
        d = max(self.degree, g.degree)
        out = self.expanded(d)
        out.w = out.w + g.expanded(d).w
        return out


def check_expansion_invariant(old, new, point_sets):
    """Max |score diff| and max |proba diff| between two models over the given inputs."""
    x = np.concatenate([np.atleast_2d(p) for p in point_sets])
    ds = float(np.max(np.abs(new.score(x) - old.score(x))))
    dp = float(np.max(np.abs(new.proba(x) - old.proba(x))))
    return ds, dp
