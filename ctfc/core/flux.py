"""
ctfc.core.flux
==============

Flux operators for Chrono-Tensorial Flux Calculus (CTFC).

This module defines the spatial flux field Ψ(t), which encodes instantaneous
spatial distortion, interaction, or information flow across dimensions of a
multivariate signal.

Mathematical role
-----------------
Given a multivariate signal X(t) ∈ R^n, the flux field Ψ(t) ∈ R^n captures
local spatial gradients or deviations from neighborhood structure.

In CTFC, flux is the *generator* of:
- temporal memory (via integral memory operators),
- chrono-derivatives (via flux-weighted temporal responses),
- curvature amplification.

This module is deliberately simple, bounded, and interpretable.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# ============================================================
# Flux estimation
# ============================================================


def compute_flux(
    x_t: np.ndarray,
    adjacency: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the spatial flux field Ψ(t) at a single time step.

    Parameters
    ----------
    x_t : np.ndarray, shape (n,)
        Multivariate observation at time t.

    adjacency : np.ndarray, shape (n, n), optional
        Adjacency or neighborhood matrix defining spatial coupling.
        If None, a fully connected neighborhood (mean-field) is assumed.

    normalize : bool, default=True
        If True, normalize flux by neighborhood degree to ensure
        boundedness and scale invariance.

    Returns
    -------
    psi_t : np.ndarray, shape (n,)
        Flux vector Ψ(t).

    Mathematical definition
    -----------------------
    For each component i:

        Ψ_i(t) = (1 / |N_i|) * Σ_{j ∈ N_i} (x_j(t) - x_i(t))

    If adjacency is fully connected, this reduces to:

        Ψ(t) = x_t - mean(x_t)

    Interpretation
    --------------
    - Ψ_i(t) > 0 : node i receives net inflow
    - Ψ_i(t) < 0 : node i exports net outflow
    - Ψ(t) = 0   : spatial equilibrium

    Notes
    -----
    - This operator is linear in x_t.
    - Lipschitz continuous with respect to x_t.
    - Guaranteed to be bounded for bounded x_t.
    """

    x_t = np.asarray(x_t, dtype=float)

    if x_t.ndim != 1:
        raise ValueError("x_t must be a one-dimensional array")

    n = x_t.shape[0]

    # --------------------------------------------------------
    # Mean-field (fully connected) flux
    # --------------------------------------------------------
    if adjacency is None:
        mean_x = np.mean(x_t)
        psi_t = x_t - mean_x
        return psi_t

    # --------------------------------------------------------
    # Graph-based flux
    # --------------------------------------------------------
    adjacency = np.asarray(adjacency, dtype=float)

    if adjacency.shape != (n, n):
        raise ValueError("adjacency must have shape (n, n)")

    # Remove self-loops
    A = adjacency.copy()
    np.fill_diagonal(A, 0.0)

    # Degree (neighborhood size)
    degree = A.sum(axis=1)

    # Avoid division by zero
    degree_safe = np.where(degree > 0, degree, 1.0)

    # Flux computation
    # Ψ_i = Σ_j A_ij (x_j - x_i)
    psi_t = A @ x_t - degree * x_t

    if normalize:
        psi_t = psi_t / degree_safe

    return psi_t


# ============================================================
# Flux magnitude
# ============================================================


def flux_magnitude(
    psi_t: np.ndarray,
    ord: int = 2,
) -> float:
    """
    Compute the magnitude of the flux field.

    Parameters
    ----------
    psi_t : np.ndarray, shape (n,)
        Flux vector Ψ(t).

    ord : int, default=2
        Norm order. By default, Euclidean norm (L2).

    Returns
    -------
    magnitude : float
        Scalar flux magnitude ||Ψ(t)||.

    Mathematical definition
    -----------------------
        ||Ψ(t)|| = ( Σ_i |Ψ_i(t)|^ord )^(1/ord)

    Role in CTFC
    ------------
    - Drives integral memory accumulation
    - Scales chrono-derivative terms
    - Modulates curvature amplification

    Notes
    -----
    - Always non-negative
    - Lipschitz continuous
    - Bounded if Ψ(t) is bounded
    """

    psi_t = np.asarray(psi_t, dtype=float)

    if psi_t.ndim != 1:
        raise ValueError("psi_t must be a one-dimensional array")

    return float(np.linalg.norm(psi_t, ord=ord))


# ============================================================
# Batched utilities (optional, internal use)
# ============================================================


def batch_flux(
    X: np.ndarray,
    adjacency: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute flux Ψ(t) for a sequence of observations.

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
        Time series of observations.

    adjacency : np.ndarray, shape (n, n), optional
        Spatial adjacency matrix.

    normalize : bool, default=True
        Normalize flux by neighborhood degree.

    Returns
    -------
    Psi : np.ndarray, shape (T, n)
        Flux field over time.

    Notes
    -----
    This function is not part of the public API but is useful
    for internal pipelines and experiments.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must have shape (T, n)")

    T, _ = X.shape
    Psi = np.zeros_like(X)

    for t in range(T):
        Psi[t] = compute_flux(X[t], adjacency=adjacency, normalize=normalize)

    return Psi

