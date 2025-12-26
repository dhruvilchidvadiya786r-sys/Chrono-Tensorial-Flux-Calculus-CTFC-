"""
ctfc.core.covariance
===================

Covariance operators for Chrono-Tensorial Flux Calculus (CTFC).

This module defines robust covariance estimation utilities for
spatiotemporal systems where correlations evolve over time.

In CTFC, covariance matrices are not static statistics.
They are interpreted as:
- local metric tensors,
- points on the SPD manifold,
- objects that evolve under flux-driven deformation.

Numerical stability, symmetry, and positive semi-definiteness
are treated as *hard constraints*.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# ============================================================
# Safe covariance estimation
# ============================================================


def safe_covariance(
    X: np.ndarray,
    bias: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute a numerically safe covariance matrix.

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
        Data matrix with T observations and n variables.

    bias : bool, default=False
        If True, normalize by T instead of (T - 1).

    eps : float, default=1e-8
        Diagonal regularization strength to ensure
        positive semi-definiteness.

    Returns
    -------
    C : np.ndarray, shape (n, n)
        Symmetric, positive semi-definite covariance matrix.

    Mathematical definition
    -----------------------
        C = (1 / (T - 1)) (X - μ)^T (X - μ) + ε I

    where μ is the sample mean.

    Properties
    ----------
    - Symmetric by construction
    - Strictly positive definite if eps > 0
    - Stable under degenerate samples

    Notes
    -----
    This function *never* returns NaNs or Infs.
    It is safe to use inside iterative pipelines.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must have shape (T, n)")

    T, n = X.shape

    if T < 2:
        # Degenerate case: return small diagonal matrix
        return eps * np.eye(n)

    # Center data
    mean = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean

    # Normalization factor
    denom = T if bias else (T - 1)

    # Raw covariance
    C = (Xc.T @ Xc) / denom

    # Regularize diagonal
    C += eps * np.eye(n)

    # Enforce symmetry explicitly
    C = 0.5 * (C + C.T)

    return C


# ============================================================
# Rolling covariance (temporal geometry)
# ============================================================


def rolling_covariance(
    X: np.ndarray,
    window: int,
    bias: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute rolling covariance matrices over time.

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
        Time series data.

    window : int
        Window length (w ≥ 2).

    bias : bool, default=False
        Use biased estimator if True.

    eps : float, default=1e-8
        Diagonal regularization.

    Returns
    -------
    C_seq : np.ndarray, shape (T, n, n)
        Sequence of covariance matrices.
        Entries before t < window are zero matrices.

    Interpretation in CTFC
    ----------------------
    Each C(t) is a *local metric tensor* on the chrono-manifold,
    encoding instantaneous correlation geometry.

    Notes
    -----
    - Uses safe_covariance internally
    - Guaranteed finite and SPD
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must have shape (T, n)")

    T, n = X.shape

    if window < 2:
        raise ValueError("window must be >= 2")

    C_seq = np.zeros((T, n, n), dtype=float)

    for t in range(window - 1, T):
        W = X[t - window + 1 : t + 1]
        C_seq[t] = safe_covariance(W, bias=bias, eps=eps)

    return C_seq


# ============================================================
# Covariance diagnostics
# ============================================================


def covariance_trace(C: np.ndarray) -> float:
    """
    Compute trace of a covariance matrix.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)

    Returns
    -------
    tr : float
        Trace of C.

    Interpretation
    --------------
    Total variance / energy in the system.
    """

    C = np.asarray(C, dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")

    return float(np.trace(C))


def covariance_frobenius_norm(C: np.ndarray) -> float:
    """
    Compute Frobenius norm of covariance matrix.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)

    Returns
    -------
    norm : float

    Interpretation
    --------------
    Measures overall correlation energy.
    Used in chrono-derivative calculations.
    """

    C = np.asarray(C, dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")

    return float(np.linalg.norm(C, ord="fro"))


# ============================================================
# Eigenvalue utilities (safe spectral geometry)
# ============================================================


def top_eigenvalues(
    C: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """
    Compute top-k eigenvalues of a covariance matrix.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Symmetric covariance matrix.

    k : int, default=1
        Number of leading eigenvalues.

    Returns
    -------
    eigvals : np.ndarray, shape (k,)
        Sorted in descending order.

    Notes
    -----
    - Uses eigh (safe for symmetric matrices)
    - Clips small negative eigenvalues caused by numerics
    """

    C = np.asarray(C, dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")

    if k < 1:
        raise ValueError("k must be >= 1")

    # Eigen decomposition
    eigvals = np.linalg.eigvalsh(C)

    # Numerical safety: clip negatives
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)

    # Sort descending
    eigvals = eigvals[::-1]

    # Pad if necessary
    if k > eigvals.shape[0]:
        eigvals = np.pad(eigvals, (0, k - eigvals.shape[0]))

    return eigvals[:k]


# ============================================================
# Covariance differences (temporal deformation)
# ============================================================


def covariance_difference(
    C_t: np.ndarray,
    C_prev: np.ndarray,
    norm: str = "fro",
) -> float:
    """
    Compute difference between consecutive covariance matrices.

    Parameters
    ----------
    C_t : np.ndarray, shape (n, n)
        Current covariance.

    C_prev : np.ndarray, shape (n, n)
        Previous covariance.

    norm : {"fro", "trace"}, default="fro"
        Difference metric.

    Returns
    -------
    diff : float

    Interpretation
    --------------
    Measures instantaneous deformation of correlation geometry.
    This is the base signal for chrono-derivative operators.
    """

    C_t = np.asarray(C_t, dtype=float)
    C_prev = np.asarray(C_prev, dtype=float)

    if C_t.shape != C_prev.shape:
        raise ValueError("C_t and C_prev must have the same shape")

    if norm == "fro":
        return float(np.linalg.norm(C_t - C_prev, ord="fro"))

    elif norm == "trace":
        return float(abs(np.trace(C_t - C_prev)))

    else:
        raise ValueError(f"Unknown norm: {norm}")

