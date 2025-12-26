"""
ctfc.core.trace
===============

Chrono-trace and chrono-contraction operators for
Chrono-Tensorial Flux Calculus (CTFC).

These operators generalize the classical trace of a covariance matrix
by incorporating *flux-induced temporal memory*.

In CTFC:
- The classical trace measures instantaneous variance.
- Chrono-trace and chrono-contraction measure *effective variance*
  under spatial flux and temporal rank amplification.

Both operators are:
- bounded,
- monotone in flux magnitude,
- interpretable,
- numerically stable.

They form the invariant backbone of CTFC embeddings.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from ctfc.core.flux import flux_magnitude

# ============================================================
# Chrono-trace
# ============================================================


def chrono_trace(
    C: np.ndarray,
    psi_t: np.ndarray,
    r: int = 1,
    scale: Literal["quadratic"] = "quadratic",
) -> float:
    """
    Compute the chrono-trace of a covariance matrix.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Covariance matrix at time t.

    psi_t : np.ndarray, shape (n,)
        Flux vector Ψ(t).

    r : int, default=1
        Effective temporal rank parameter.
        Controls sensitivity to flux amplification.

    scale : {"quadratic"}, default="quadratic"
        Flux scaling rule.
        Currently only quadratic scaling is supported,
        consistent with CTFC theory.

    Returns
    -------
    tr_chrono : float
        Chrono-trace value.

    Mathematical definition
    -----------------------
    Let:
        tr(C) = Σ_i C_ii
        ψ = ||Ψ(t)||

    Then:
        Trace_chrono(C)(t)
        = tr(C) * (1 + r / (r + 2) * ψ^2)

    Properties
    ----------
    - Trace_chrono ≥ tr(C)
    - Bounded by tr(C) * (1 + ψ^2)
    - Monotone in ψ
    - Stable for all r ≥ 0

    Interpretation
    --------------
    - Measures total variance under flux distortion
    - Larger r → stronger sensitivity to flux
    - Captures nonlinear variance inflation
    """

    C = np.asarray(C, dtype=float)
    psi_t = np.asarray(psi_t, dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")

    if psi_t.ndim != 1:
        raise ValueError("psi_t must be a one-dimensional array")

    if r < 0:
        raise ValueError("r must be non-negative")

    # Classical trace
    tr_C = float(np.trace(C))

    # Flux magnitude
    psi_norm = flux_magnitude(psi_t)

    if scale != "quadratic":
        raise ValueError("Only quadratic scaling is supported")

    # Flux amplification factor
    amplification = 1.0 + (r / (r + 2.0)) * (psi_norm**2)

    return tr_C * amplification


# ============================================================
# Chrono-contraction
# ============================================================


def chrono_contraction(
    C: np.ndarray,
    psi_t: np.ndarray,
    r: int = 1,
    scale: Literal["quadratic"] = "quadratic",
) -> float:
    """
    Compute the chrono-contraction of a covariance matrix.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Covariance matrix at time t.

    psi_t : np.ndarray, shape (n,)
        Flux vector Ψ(t).

    r : int, default=1
        Effective temporal rank parameter.

    scale : {"quadratic"}, default="quadratic"
        Flux scaling rule.

    Returns
    -------
    contract_chrono : float
        Chrono-contraction value.

    Mathematical definition
    -----------------------
    Let:
        tr(C) = Σ_i C_ii
        ψ = ||Ψ(t)||

    Then:
        Contract_chrono(C)(t)
        = tr(C) * (1 + r / (r + 1) * ψ^2)

    Properties
    ----------
    - Contract_chrono ≥ tr(C)
    - Stronger amplification than chrono-trace
    - Bounded by tr(C) * (1 + ψ^2)
    - Monotone in ψ

    Interpretation
    --------------
    - Measures directional energy contraction
    - More sensitive to flux than chrono-trace
    - Emphasizes dissipative alignment
    """

    C = np.asarray(C, dtype=float)
    psi_t = np.asarray(psi_t, dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")

    if psi_t.ndim != 1:
        raise ValueError("psi_t must be a one-dimensional array")

    if r < 0:
        raise ValueError("r must be non-negative")

    # Classical trace
    tr_C = float(np.trace(C))

    # Flux magnitude
    psi_norm = flux_magnitude(psi_t)

    if scale != "quadratic":
        raise ValueError("Only quadratic scaling is supported")

    # Flux amplification factor
    amplification = 1.0 + (r / (r + 1.0)) * (psi_norm**2)

    return tr_C * amplification


# ============================================================
# Diagnostics & comparison utilities
# ============================================================


def trace_amplification_factors(
    psi_t: np.ndarray,
    r: int = 1,
) -> dict[str, float]:
    """
    Compute amplification factors for chrono-trace and chrono-contraction.

    Parameters
    ----------
    psi_t : np.ndarray, shape (n,)
        Flux vector Ψ(t).

    r : int, default=1
        Effective temporal rank.

    Returns
    -------
    factors : dict
        {
            "trace": amplification for chrono-trace,
            "contraction": amplification for chrono-contraction
        }

    Notes
    -----
    Useful for:
    - diagnostics
    - theoretical validation
    - plotting amplification curves
    """

    psi_t = np.asarray(psi_t, dtype=float)

    if psi_t.ndim != 1:
        raise ValueError("psi_t must be a one-dimensional array")

    if r < 0:
        raise ValueError("r must be non-negative")

    psi_norm = flux_magnitude(psi_t)

    return {
        "trace": 1.0 + (r / (r + 2.0)) * (psi_norm**2),
        "contraction": 1.0 + (r / (r + 1.0)) * (psi_norm**2),
    }

