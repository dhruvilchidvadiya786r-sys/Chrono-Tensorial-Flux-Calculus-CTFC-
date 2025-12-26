"""
ctfc.core.memory
================

Temporal memory operators for Chrono-Tensorial Flux Calculus (CTFC).

This module defines operators that convert instantaneous flux activity
into *non-Markovian temporal memory*. Memory in CTFC is not heuristic;
it is a geometric path integral over flux magnitude.

Memory is the backbone of:
- non-Markovian behavior,
- curvature amplification,
- regime persistence.

This module provides both:
- exact integral memory (continuous-time interpretation),
- stable discrete approximations (EWMA / Volterra surrogates).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# ============================================================
# Continuous-time integral memory (conceptual reference)
# ============================================================


def integral_memory(
    psi_norms: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Compute cumulative integral memory I(t) from flux magnitudes.

    Parameters
    ----------
    psi_norms : np.ndarray, shape (T,)
        Sequence of flux magnitudes ||Ψ(t)||.

    dt : float, default=1.0
        Time step size.

    Returns
    -------
    I : np.ndarray, shape (T,)
        Integral memory sequence.

    Mathematical definition
    -----------------------
        I(t_k) = ∑_{i=1}^k ||Ψ(t_i)|| · dt
              ≈ ∫_0^t ||Ψ(s)|| ds

    Interpretation
    --------------
    - I(t) measures *accumulated geometric distortion*
    - Encodes persistence of flux activity
    - Non-decreasing by construction

    Properties
    ----------
    - Monotone increasing
    - Lipschitz continuous
    - O(dt) convergence to continuous integral

    Notes
    -----
    This function is conceptually exact but assumes:
    - uniform sampling
    - no forgetting

    In practice, EWMA memory is preferred for stability.
    """

    psi_norms = np.asarray(psi_norms, dtype=float)

    if psi_norms.ndim != 1:
        raise ValueError("psi_norms must be a one-dimensional array")

    if dt <= 0:
        raise ValueError("dt must be positive")

    # Riemann sum
    return np.cumsum(psi_norms) * dt


# ============================================================
# EWMA / Volterra memory (stable discrete realization)
# ============================================================


def ewma_memory(
    psi_norms: np.ndarray,
    kappa: float = 0.9,
    I0: float = 0.0,
) -> np.ndarray:
    """
    Compute exponentially weighted memory (CTFC≈ default).

    Parameters
    ----------
    psi_norms : np.ndarray, shape (T,)
        Flux magnitude sequence ||Ψ(t)||.

    kappa : float, default=0.9
        Memory retention coefficient.
        Must satisfy 0 < kappa < 1.

    I0 : float, default=0.0
        Initial memory value.

    Returns
    -------
    I : np.ndarray, shape (T,)
        EWMA memory sequence.

    Mathematical definition
    -----------------------
        I_t = κ I_{t-1} + ||Ψ(t)||

    This is a discrete Volterra operator with kernel:
        K(s) = κ^s

    Interpretation
    --------------
    - Recent flux has higher weight
    - Older flux decays exponentially
    - Prevents unbounded growth

    Properties
    ----------
    - Stable for all bounded ||Ψ||
    - Lipschitz in psi_norms
    - Converges to integral memory as κ → 1

    Theoretical guarantee
    ---------------------
    If κ = exp(-λΔt), then:
        I_t → ∫_0^t e^{-λ(t-s)} ||Ψ(s)|| ds

    Notes
    -----
    This is the **default memory operator** used in CTFC≈.
    """

    psi_norms = np.asarray(psi_norms, dtype=float)

    if psi_norms.ndim != 1:
        raise ValueError("psi_norms must be a one-dimensional array")

    if not (0.0 < kappa < 1.0):
        raise ValueError("kappa must satisfy 0 < kappa < 1")

    T = psi_norms.shape[0]
    I = np.zeros(T, dtype=float)

    I_prev = float(I0)

    for t in range(T):
        I_t = kappa * I_prev + psi_norms[t]
        I[t] = I_t
        I_prev = I_t

    return I


# ============================================================
# Memory normalization utilities
# ============================================================


def normalized_memory(
    I: np.ndarray,
    method: str = "max",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize memory sequence for scale-invariant usage.

    Parameters
    ----------
    I : np.ndarray, shape (T,)
        Memory sequence.

    method : {"max", "zscore", "log"}, default="max"
        Normalization strategy.

    eps : float, default=1e-12
        Numerical stability constant.

    Returns
    -------
    I_norm : np.ndarray, shape (T,)
        Normalized memory.

    Use cases
    ---------
    - Visualization
    - Comparative studies
    - Downstream ML models

    Notes
    -----
    Normalization is **not** part of core CTFC theory.
    Use only for post-processing.
    """

    I = np.asarray(I, dtype=float)

    if I.ndim != 1:
        raise ValueError("I must be a one-dimensional array")

    if method == "max":
        return I / (np.max(I) + eps)

    elif method == "zscore":
        return (I - np.mean(I)) / (np.std(I) + eps)

    elif method == "log":
        return np.log1p(I)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============================================================
# Memory diagnostics (research & testing)
# ============================================================


def memory_increment(
    I: np.ndarray,
) -> np.ndarray:
    """
    Compute incremental memory changes ΔI(t).

    Parameters
    ----------
    I : np.ndarray, shape (T,)
        Memory sequence.

    Returns
    -------
    dI : np.ndarray, shape (T,)
        Memory increments.

    Notes
    -----
    Useful for:
    - regime change detection
    - curvature amplification diagnostics
    """

    I = np.asarray(I, dtype=float)

    if I.ndim != 1:
        raise ValueError("I must be a one-dimensional array")

    dI = np.zeros_like(I)
    dI[1:] = I[1:] - I[:-1]

    return dI

