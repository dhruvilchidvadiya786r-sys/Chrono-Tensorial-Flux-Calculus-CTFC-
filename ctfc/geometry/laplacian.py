"""
ctfc.geometry.laplacian
======================

Graph Laplacian operators for Chrono-Tensorial Flux Calculus (CTFC).

This module constructs Laplacians from affinity or distance matrices
to approximate geometric structure induced by covariance similarity.

In CTFC, Laplacians are used to:
- approximate curvature via tr(C L),
- encode local manifold geometry,
- remain invariant to global scaling.

Design principles
-----------------
- Symmetry is enforced.
- Positive semidefiniteness is preserved.
- Normalization choices are explicit and documented.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Tuple

from ctfc.geometry.similarity import affinity_kernel

# ============================================================
# Degree utilities
# ============================================================


def degree_matrix(
    W: np.ndarray,
) -> np.ndarray:
    """
    Compute the degree matrix from an affinity matrix.

    Parameters
    ----------
    W : np.ndarray, shape (n, n)
        Affinity (weight) matrix.

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Diagonal degree matrix with D_ii = sum_j W_ij.
    """

    W = np.asarray(W, dtype=float)

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")

    degrees = np.sum(W, axis=1)
    return np.diag(degrees)


# ============================================================
# Core Laplacian constructions
# ============================================================


def unnormalized_laplacian(
    W: np.ndarray,
) -> np.ndarray:
    """
    Construct the unnormalized graph Laplacian.

    Parameters
    ----------
    W : np.ndarray, shape (n, n)
        Affinity matrix.

    Returns
    -------
    L : np.ndarray, shape (n, n)
        Unnormalized Laplacian: L = D - W.

    Properties
    ----------
    - Symmetric if W is symmetric
    - Positive semidefinite
    """

    W = np.asarray(W, dtype=float)

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")

    D = degree_matrix(W)
    L = D - W

    # Enforce symmetry
    return 0.5 * (L + L.T)


def normalized_laplacian(
    W: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Construct the symmetric normalized graph Laplacian.

    Parameters
    ----------
    W : np.ndarray, shape (n, n)
        Affinity matrix.

    eps : float, default=1e-12
        Numerical stability constant.

    Returns
    -------
    L_norm : np.ndarray, shape (n, n)
        Normalized Laplacian:
            L = I - D^{-1/2} W D^{-1/2}

    Properties
    ----------
    - Eigenvalues in [0, 2]
    - Scale invariant
    - Preferred for spectral analysis
    """

    W = np.asarray(W, dtype=float)

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")

    degrees = np.sum(W, axis=1)
    inv_sqrt_deg = 1.0 / np.sqrt(degrees + eps)

    D_inv_sqrt = np.diag(inv_sqrt_deg)
    I = np.eye(W.shape[0])

    L = I - D_inv_sqrt @ W @ D_inv_sqrt

    # Enforce symmetry
    return 0.5 * (L + L.T)


# ============================================================
# Distance → Laplacian pipeline (CTFC standard)
# ============================================================


def laplacian_from_distance(
    D: np.ndarray,
    *,
    sigma: float = 1.0,
    normalization: Literal["unnormalized", "normalized"] = "unnormalized",
) -> np.ndarray:
    """
    Construct a graph Laplacian directly from a distance matrix.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Distance matrix.

    sigma : float, default=1.0
        Kernel bandwidth for Gaussian affinity.

    normalization : {"unnormalized", "normalized"}, default="unnormalized"
        Type of Laplacian.

    Returns
    -------
    L : np.ndarray, shape (n, n)
        Graph Laplacian.

    Notes
    -----
    This is the *canonical CTFC geometry pipeline*:

        distance → affinity → Laplacian
    """

    D = np.asarray(D, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix")

    # Affinity
    W = affinity_kernel(D, sigma=sigma)

    if normalization == "unnormalized":
        return unnormalized_laplacian(W)

    elif normalization == "normalized":
        return normalized_laplacian(W)

    else:
        raise ValueError(f"Unknown normalization: {normalization}")


# ============================================================
# Spectral diagnostics
# ============================================================


def laplacian_spectrum(
    L: np.ndarray,
    k: int | None = None,
) -> np.ndarray:
    """
    Compute eigenvalues of a Laplacian matrix.

    Parameters
    ----------
    L : np.ndarray, shape (n, n)
        Laplacian matrix.

    k : int or None
        Number of smallest eigenvalues to return.
        If None, return full spectrum.

    Returns
    -------
    eigvals : np.ndarray
        Sorted eigenvalues (ascending).

    Interpretation
    --------------
    - Small eigenvalues encode connectivity
    - Zero eigenvalues correspond to connected components
    """

    L = np.asarray(L, dtype=float)

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix")

    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.clip(eigvals, 0.0, None)  # numerical safety

    if k is not None:
        if k < 1:
            raise ValueError("k must be >= 1")
        eigvals = eigvals[:k]

    return eigvals


# ============================================================
# Energy forms (used in curvature interpretation)
# ============================================================


def laplacian_quadratic_form(
    x: np.ndarray,
    L: np.ndarray,
) -> float:
    """
    Compute the Laplacian quadratic form xᵀ L x.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Vector on graph nodes.

    L : np.ndarray, shape (n, n)
        Laplacian matrix.

    Returns
    -------
    energy : float

    Interpretation
    --------------
    Measures smoothness of x with respect to graph geometry.
    """

    x = np.asarray(x, dtype=float)
    L = np.asarray(L, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix")

    if L.shape[0] != x.shape[0]:
        raise ValueError("Dimensions of x and L do not match")

    return float(x.T @ L @ x)

