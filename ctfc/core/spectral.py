"""
ctfc.core.spectral
==================

Flux-scaled spectral operators for Chrono-Tensorial Flux Calculus (CTFC).

This module defines spectral invariants derived from covariance matrices
whose eigenstructure is *modulated by flux intensity*.

In CTFC:
- Eigenvalues are not static variances
- They are *geometry-sensitive energy modes*
- Flux scales spectral dominance to reflect interaction strength

This is the final step before embedding construction.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from ctfc.core.flux import flux_magnitude
from ctfc.core.covariance import safe_covariance, top_eigenvalues

# ============================================================
# Flux-scaled eigenvalues (core operator)
# ============================================================


def flux_scaled_eigenvalues(
    C: np.ndarray,
    psi_t: np.ndarray,
    k: int = 1,
    scale: Literal["linear", "quadratic"] = "linear",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute flux-scaled leading eigenvalues of a covariance matrix.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Covariance matrix at time t.

    psi_t : np.ndarray, shape (n,)
        Flux vector Ψ(t).

    k : int, default=1
        Number of leading eigenvalues to return.

    scale : {"linear", "quadratic"}, default="linear"
        Flux scaling rule.

    eps : float, default=1e-8
        Regularization for covariance safety.

    Returns
    -------
    lambda_scaled : np.ndarray, shape (k,)
        Flux-scaled eigenvalues λ'_1(t), …, λ'_k(t).

    Mathematical definition
    -----------------------
    Let:
        λ_i = i-th eigenvalue of C
        ψ = ||Ψ(t)||

    Then:
        λ'_i(t) = λ_i · g(ψ)

    where:
        g(ψ) = 1 + ψ        (linear)
        g(ψ) = 1 + ψ^2      (quadratic)

    Interpretation
    --------------
    - Eigenvalues represent variance along principal modes
    - Flux amplifies spectral dominance when interaction is strong
    - Preserves ordering and positivity

    Properties
    ----------
    - Non-negative
    - Bounded for bounded ψ
    - Stable under small perturbations of C
    """

    C = np.asarray(C, dtype=float)
    psi_t = np.asarray(psi_t, dtype=float)

    if psi_t.ndim != 1:
        raise ValueError("psi_t must be a one-dimensional array")

    if k < 1:
        raise ValueError("k must be >= 1")

    # Ensure SPD covariance
    C = safe_covariance(C, eps=eps)

    # Leading eigenvalues
    eigvals = top_eigenvalues(C, k=k)

    # Flux magnitude
    psi_norm = flux_magnitude(psi_t)

    # Scaling rule
    if scale == "linear":
        factor = 1.0 + psi_norm
    elif scale == "quadratic":
        factor = 1.0 + psi_norm**2
    else:
        raise ValueError(f"Unknown scale: {scale}")

    return eigvals * factor


# ============================================================
# Spectral energy ratios (diagnostics)
# ============================================================


def spectral_energy_ratio(
    C: np.ndarray,
    psi_t: np.ndarray,
    k: int = 1,
    scale: Literal["linear", "quadratic"] = "linear",
) -> float:
    """
    Compute fraction of total spectral energy carried by top-k modes.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Covariance matrix.

    psi_t : np.ndarray, shape (n,)
        Flux vector.

    k : int, default=1
        Number of leading modes.

    scale : {"linear", "quadratic"}, default="linear"
        Flux scaling rule.

    Returns
    -------
    ratio : float
        Spectral energy ratio in [0, 1].

    Interpretation
    --------------
    - Measures effective dimensionality
    - High ratio → dominant low-rank structure
    - Flux increases dominance under interaction
    """

    C = np.asarray(C, dtype=float)
    psi_t = np.asarray(psi_t, dtype=float)

    # All eigenvalues
    all_eigs = top_eigenvalues(C, k=C.shape[0])

    # Scaled eigenvalues
    scaled_top = flux_scaled_eigenvalues(C, psi_t, k=k, scale=scale)
    scaled_all = flux_scaled_eigenvalues(C, psi_t, k=C.shape[0], scale=scale)

    total_energy = np.sum(scaled_all)
    top_energy = np.sum(scaled_top)

    if total_energy <= 0:
        return 0.0

    return float(top_energy / total_energy)


# ============================================================
# Batched spectral operators
# ============================================================


def spectral_series(
    C_seq: np.ndarray,
    Psi_seq: np.ndarray,
    k: int = 1,
    scale: Literal["linear", "quadratic"] = "linear",
) -> np.ndarray:
    """
    Compute flux-scaled spectral modes over time.

    Parameters
    ----------
    C_seq : np.ndarray, shape (T, n, n)
        Covariance sequence.

    Psi_seq : np.ndarray, shape (T, n)
        Flux sequence.

    k : int, default=1
        Number of leading modes.

    scale : {"linear", "quadratic"}, default="linear"
        Flux scaling rule.

    Returns
    -------
    Lambda_seq : np.ndarray, shape (T, k)
        Time series of flux-scaled eigenvalues.

    Notes
    -----
    Used directly in CTFC embedding construction.
    """

    C_seq = np.asarray(C_seq, dtype=float)
    Psi_seq = np.asarray(Psi_seq, dtype=float)

    if C_seq.ndim != 3:
        raise ValueError("C_seq must have shape (T, n, n)")

    if Psi_seq.ndim != 2:
        raise ValueError("Psi_seq must have shape (T, n)")

    if C_seq.shape[0] != Psi_seq.shape[0]:
        raise ValueError("Time dimensions must match")

    T = C_seq.shape[0]
    Lambda = np.zeros((T, k), dtype=float)

    for t in range(T):
        Lambda[t] = flux_scaled_eigenvalues(
            C=C_seq[t],
            psi_t=Psi_seq[t],
            k=k,
            scale=scale,
        )

    return Lambda

