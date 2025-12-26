"""
ctfc.geometry.similarity
=======================

Similarity operators for Chrono-Tensorial Flux Calculus (CTFC).

This module defines similarity measures derived from covariance
structure. Similarity in CTFC is not an ad hoc distance; it is a
*geometric normalization* that turns covariance tensors into
well-conditioned relational objects.

These similarities are used to:
- build graph Laplacians,
- approximate curvature,
- encode local manifold structure.

Design principle
----------------
Similarity must be:
- symmetric,
- bounded,
- invariant to scale,
- numerically stable.

This module enforces those properties explicitly.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

# ============================================================
# Correlation-based similarity (primary)
# ============================================================


def correlation_similarity(
    C: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute correlation-based similarity from a covariance matrix.

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

    Properties
    ----------
    - S_ii = 1
    - |S_ij| ≤ 1
    - invariant to marginal scaling
    - symmetric

    Interpretation
    --------------
    Correlation similarity measures *relative co-movement* rather than
    absolute covariance. This makes it ideal for geometric constructions
    such as graph Laplacians.

    Notes
    -----
    This is the canonical similarity used throughout CTFC.
    """

    C = np.asarray(C, dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")

    diag = np.diag(C)

    if np.any(diag < 0):
        raise ValueError("Covariance matrix must be positive semi-definite")

    denom = np.sqrt(np.outer(diag, diag)) + eps
    S = C / denom

    # Enforce numerical bounds
    S = np.clip(S, -1.0, 1.0)

    # Force exact ones on diagonal
    np.fill_diagonal(S, 1.0)

    return S


# ============================================================
# Distance conversion
# ============================================================


def similarity_to_distance(
    S: np.ndarray,
    method: Literal["linear", "angular"] = "linear",
) -> np.ndarray:
    """
    Convert similarity matrix to a distance matrix.

    Parameters
    ----------
    S : np.ndarray, shape (n, n)
        Similarity matrix.

    method : {"linear", "angular"}, default="linear"
        Conversion method.

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Distance matrix.

    Definitions
    -----------
    Linear:
        D_ij = 1 - S_ij

    Angular:
        D_ij = arccos(S_ij) / π

    Properties
    ----------
    - D_ij ≥ 0
    - D_ii = 0
    - symmetric
    """

    S = np.asarray(S, dtype=float)

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square matrix")

    if method == "linear":
        D = 1.0 - S

    elif method == "angular":
        S_clip = np.clip(S, -1.0, 1.0)
        D = np.arccos(S_clip) / np.pi

    else:
        raise ValueError(f"Unknown method: {method}")

    # Numerical safety
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)

    return D


# ============================================================
# Kernel affinity (optional but useful)
# ============================================================


def affinity_kernel(
    D: np.ndarray,
    sigma: float = 1.0,
    kernel: Literal["gaussian"] = "gaussian",
) -> np.ndarray:
    """
    Convert distance matrix to affinity (weight) matrix.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Distance matrix.

    sigma : float, default=1.0
        Kernel bandwidth.

    kernel : {"gaussian"}, default="gaussian"
        Kernel type.

    Returns
    -------
    W : np.ndarray, shape (n, n)
        Affinity matrix.

    Mathematical definition
    -----------------------
        W_ij = exp( -D_ij^2 / (2 σ^2) )

    Notes
    -----
    This function is a convenience wrapper.
    The Laplacian construction lives in laplacian.py.
    """

    D = np.asarray(D, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if kernel != "gaussian":
        raise ValueError("Only Gaussian kernel is supported")

    W = np.exp(-(D**2) / (2.0 * sigma**2))

    # Enforce symmetry
    W = 0.5 * (W + W.T)

    return W


# ============================================================
# Diagnostics
# ============================================================


def similarity_statistics(S: np.ndarray) -> dict[str, float]:
    """
    Compute basic statistics of a similarity matrix.

    Parameters
    ----------
    S : np.ndarray, shape (n, n)

    Returns
    -------
    stats : dict
        {
            "min": min similarity,
            "max": max similarity,
            "mean": mean off-diagonal similarity
        }
    """

    S = np.asarray(S, dtype=float)

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square matrix")

    n = S.shape[0]
    mask = ~np.eye(n, dtype=bool)

    return {
        "min": float(np.min(S[mask])),
        "max": float(np.max(S[mask])),
        "mean": float(np.mean(S[mask])),
    }

