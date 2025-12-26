"""
ctfc.core.chrono_derivative
==========================

Chrono-derivative operators for Chrono-Tensorial Flux Calculus (CTFC).

The chrono-derivative generalizes the classical time derivative by
incorporating *flux-weighted temporal memory*. It measures how fast
the covariance geometry deforms under persistent spatial interaction.

In CTFC:
- Classical derivative → instantaneous change
- Chrono-derivative → flux-amplified geometric response

This operator is central to:
- detecting regime transitions,
- quantifying structural deformation,
- coupling memory with dynamics.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from ctfc.core.covariance import covariance_difference
from ctfc.core.flux import flux_magnitude

# ============================================================
# Chrono-derivative (continuous-time motivation)
# ============================================================
#
# The theoretical operator is:
#
#   ∇_chrono C(t) = dC/dt
#                  + Σ_{k=1}^r ( ||Ψ(t)||^k / k! ) d^kC/dt^k
#
# In practice, only low-order terms are stable.
# CTFC≈ uses a first-order discrete realization.
#
# ============================================================


def chrono_derivative(
    C_t: np.ndarray,
    C_prev: np.ndarray,
    psi_t: np.ndarray,
    method: Literal["fro", "trace"] = "fro",
    scale: Literal["linear", "quadratic"] = "linear",
) -> float:
    """
    Compute the chrono-derivative magnitude at time t.

    Parameters
    ----------
    C_t : np.ndarray, shape (n, n)
        Covariance matrix at time t.

    C_prev : np.ndarray, shape (n, n)
        Covariance matrix at time t-1.

    psi_t : np.ndarray, shape (n,)
        Flux vector Ψ(t).

    method : {"fro", "trace"}, default="fro"
        Base geometric difference metric.

    scale : {"linear", "quadratic"}, default="linear"
        Flux scaling rule:
        - "linear":   (1 + ||Ψ||)
        - "quadratic": (1 + ||Ψ||^2)

    Returns
    -------
    phi_cd : float
        Chrono-derivative magnitude Φ_cd(t).

    Mathematical definition (CTFC≈)
    --------------------------------
        ΔC(t) = C(t) - C(t-1)

        Φ_cd(t) = ||ΔC(t)|| · g(||Ψ(t)||)

    where:
        g(x) = 1 + x        (linear)
        g(x) = 1 + x^2      (quadratic)

    Interpretation
    --------------
    - Measures *how fast correlation geometry deforms*
    - Flux amplifies deformation when spatial interaction is strong
    - Large values indicate regime transitions

    Properties
    ----------
    - Non-negative
    - Lipschitz in C and Ψ
    - Stable under bounded flux
    """

    C_t = np.asarray(C_t, dtype=float)
    C_prev = np.asarray(C_prev, dtype=float)
    psi_t = np.asarray(psi_t, dtype=float)

    if C_t.shape != C_prev.shape:
        raise ValueError("C_t and C_prev must have the same shape")

    if psi_t.ndim != 1:
        raise ValueError("psi_t must be a one-dimensional array")

    # --------------------------------------------------------
    # Base covariance deformation
    # --------------------------------------------------------
    delta_C = covariance_difference(C_t, C_prev, norm=method)

    # --------------------------------------------------------
    # Flux magnitude
    # --------------------------------------------------------
    psi_norm = flux_magnitude(psi_t)

    # --------------------------------------------------------
    # Flux scaling
    # --------------------------------------------------------
    if scale == "linear":
        amplification = 1.0 + psi_norm

    elif scale == "quadratic":
        amplification = 1.0 + psi_norm**2

    else:
        raise ValueError(f"Unknown scale: {scale}")

    return float(delta_C * amplification)


# ============================================================
# Batched chrono-derivative (time series)
# ============================================================


def chrono_derivative_series(
    C_seq: np.ndarray,
    Psi_seq: np.ndarray,
    method: Literal["fro", "trace"] = "fro",
    scale: Literal["linear", "quadratic"] = "linear",
) -> np.ndarray:
    """
    Compute chrono-derivative over a time series.

    Parameters
    ----------
    C_seq : np.ndarray, shape (T, n, n)
        Sequence of covariance matrices.

    Psi_seq : np.ndarray, shape (T, n)
        Sequence of flux vectors.

    method : {"fro", "trace"}, default="fro"
        Covariance difference metric.

    scale : {"linear", "quadratic"}, default="linear"
        Flux amplification rule.

    Returns
    -------
    Phi : np.ndarray, shape (T,)
        Chrono-derivative magnitudes.
        Phi[0] = 0 by definition.

    Notes
    -----
    This is primarily used in:
    - CTFC embedding pipelines
    - diagnostics & visualization
    """

    C_seq = np.asarray(C_seq, dtype=float)
    Psi_seq = np.asarray(Psi_seq, dtype=float)

    if C_seq.ndim != 3:
        raise ValueError("C_seq must have shape (T, n, n)")

    if Psi_seq.ndim != 2:
        raise ValueError("Psi_seq must have shape (T, n)")

    if C_seq.shape[0] != Psi_seq.shape[0]:
        raise ValueError("C_seq and Psi_seq must have same length")

    T = C_seq.shape[0]
    Phi = np.zeros(T, dtype=float)

    for t in range(1, T):
        Phi[t] = chrono_derivative(
            C_t=C_seq[t],
            C_prev=C_seq[t - 1],
            psi_t=Psi_seq[t],
            method=method,
            scale=scale,
        )

    return Phi

