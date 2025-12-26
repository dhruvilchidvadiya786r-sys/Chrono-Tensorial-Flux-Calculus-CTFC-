"""
ctfc.core.curvature
===================

Curvature surrogate operators for Chrono-Tensorial Flux Calculus (CTFC).

This module defines a *discrete curvature surrogate* that captures the
joint effect of:
- covariance geometry,
- graph-based spatial structure,
- temporal memory accumulation.

In CTFC, curvature is not classical Riemannian curvature.
It is an *information-geometric deformation energy* induced by
persistent flux on the covariance manifold.

This operator completes the CTFC invariant layer.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ctfc.core.covariance import safe_covariance
from ctfc.core.memory import integral_memory

# ============================================================
# Similarity & distance (internal utilities)
# ============================================================


def correlation_similarity(
    C: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute normalized correlation-based similarity matrix from covariance.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Covariance matrix.

    eps : float, default=1e-12
        Numerical stability constant.

    Returns
    -------
    S : np.ndarray, shape (n, n)
        Correlation similarity matrix with entries in [-1, 1].

    Mathematical definition
    -----------------------
        S_ij = C_ij / sqrt(C_ii * C_jj + eps)

    Notes
    -----
    - Diagonal entries are exactly 1
    - Robust to small variances
    """

    C = np.asarray(C, dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")

    diag = np.diag(C)
    denom = np.sqrt(np.outer(diag, diag)) + eps

    S = C / denom

    # Numerical safety
    S = np.clip(S, -1.0, 1.0)

    return S


def correlation_distance(
    S: np.ndarray,
) -> np.ndarray:
    """
    Convert correlation similarity to distance.

    Parameters
    ----------
    S : np.ndarray, shape (n, n)
        Similarity matrix.

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Distance matrix.

    Definition
    ----------
        D_ij = 1 - S_ij
    """

    S = np.asarray(S, dtype=float)

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square matrix")

    return 1.0 - S


# ============================================================
# Graph Laplacian
# ============================================================


def graph_laplacian(
    D: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Construct graph Laplacian from distance matrix.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Distance matrix.

    sigma : float, default=1.0
        Kernel bandwidth.

    Returns
    -------
    L : np.ndarray, shape (n, n)
        Unnormalized graph Laplacian.

    Mathematical definition
    -----------------------
        W_ij = exp( -D_ij^2 / (2 sigma^2) )
        L = diag(sum_j W_ij) - W

    Notes
    -----
    - Fully connected weighted graph
    - Always symmetric
    - Positive semi-definite
    """

    D = np.asarray(D, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Affinity kernel
    W = np.exp(-(D**2) / (2.0 * sigma**2))

    # Degree matrix
    degree = np.sum(W, axis=1)
    L = np.diag(degree) - W

    # Enforce symmetry
    L = 0.5 * (L + L.T)

    return L


# ============================================================
# Curvature surrogate (CTFC core operator)
# ============================================================


def curvature_surrogate(
    C: np.ndarray,
    I_t: float,
    r: int = 1,
    sigma: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """
    Compute CTFC curvature surrogate R_approx(t).

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Covariance matrix at time t.

    I_t : float
        Temporal memory value at time t.

    r : int, default=1
        Effective temporal rank.

    sigma : float, default=1.0
        Kernel bandwidth for graph Laplacian.

    eps : float, default=1e-8
        Regularization strength for covariance safety.

    Returns
    -------
    R : float
        Curvature surrogate value.

    Mathematical definition
    -----------------------
        S = corr(C)
        D = 1 - S
        L = Laplacian(D)

        R_approx(t) = tr(C L) * (1 + α I(t))
        α = r / (r + 1)

    Interpretation
    --------------
    - tr(C L) measures alignment between covariance geometry
      and graph curvature
    - Memory term amplifies persistent deformation
    - Captures structural stress of the system

    Properties
    ----------
    - Scalar invariant
    - Bounded for bounded C and I
    - Monotone in memory
    """

    if r < 0:
        raise ValueError("r must be non-negative")

    if I_t < 0:
        raise ValueError("I_t must be non-negative")

    # Ensure safe covariance
    C = safe_covariance(C, eps=eps)

    # Similarity and distance
    S = correlation_similarity(C)
    D = correlation_distance(S)

    # Graph Laplacian
    L = graph_laplacian(D, sigma=sigma)

    # Base curvature energy
    base_curvature = float(np.trace(C @ L))

    # Memory amplification
    alpha = r / (r + 1.0)
    amplification = 1.0 + alpha * I_t

    return base_curvature * amplification


# ============================================================
# Batched curvature (time series)
# ============================================================


def curvature_series(
    C_seq: np.ndarray,
    I: np.ndarray,
    r: int = 1,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Compute curvature surrogate over time.

    Parameters
    ----------
    C_seq : np.ndarray, shape (T, n, n)
        Covariance sequence.

    I : np.ndarray, shape (T,)
        Memory sequence.

    r : int, default=1
        Effective temporal rank.

    sigma : float, default=1.0
        Kernel bandwidth.

    Returns
    -------
    R_seq : np.ndarray, shape (T,)
        Curvature surrogate time series.
    """

    C_seq = np.asarray(C_seq, dtype=float)
    I = np.asarray(I, dtype=float)

    if C_seq.ndim != 3:
        raise ValueError("C_seq must have shape (T, n, n)")

    if I.ndim != 1 or I.shape[0] != C_seq.shape[0]:
        raise ValueError("I must match time dimension of C_seq")

    T = C_seq.shape[0]
    R = np.zeros(T, dtype=float)

    for t in range(T):
        R[t] = curvature_surrogate(
            C=C_seq[t],
            I_t=I[t],
            r=r,
            sigma=sigma,
        )

    return R
