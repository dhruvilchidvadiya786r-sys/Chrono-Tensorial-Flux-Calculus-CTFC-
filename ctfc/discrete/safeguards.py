"""
ctfc.discrete.safeguards
=======================

Numerical and theoretical safeguards for CTFC≈.

This module contains *defensive mechanisms* that protect CTFC
from numerical instability, degenerate data, and pathological inputs.

Safeguards are:
- explicit,
- conservative,
- mathematically justified.

They do NOT change the theory — they enforce its assumptions.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

# ============================================================
# Constants (centralized safety thresholds)
# ============================================================

EPS_COV = 1e-8        # covariance regularization
EPS_DIV = 1e-12       # division safety
MAX_FLUX = 1e6        # hard flux cap (diagnostic)
MAX_MEMORY = 1e8      # memory blow-up prevention


# ============================================================
# Input sanitation
# ============================================================


def sanitize_time_series(
    X: np.ndarray,
    *,
    allow_nan: bool = False,
) -> np.ndarray:
    """
    Sanitize raw input time series.

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
        Input data.

    allow_nan : bool, default=False
        If False, NaNs/Infs are replaced by column means.

    Returns
    -------
    X_clean : np.ndarray
        Sanitized data.

    Notes
    -----
    - CTFC assumes finite observations.
    - This function enforces that assumption explicitly.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must have shape (T, n)")

    if allow_nan:
        return X

    X_clean = X.copy()

    for j in range(X.shape[1]):
        col = X_clean[:, j]
        mask = ~np.isfinite(col)

        if np.any(mask):
            mean_val = np.nanmean(col)
            if not np.isfinite(mean_val):
                mean_val = 0.0
            col[mask] = mean_val
            X_clean[:, j] = col

    return X_clean


# ============================================================
# Flux safeguards
# ============================================================


def clip_flux(
    psi_t: np.ndarray,
    max_norm: float = MAX_FLUX,
) -> np.ndarray:
    """
    Clip flux vector if its magnitude exceeds a hard threshold.

    Parameters
    ----------
    psi_t : np.ndarray, shape (n,)
        Flux vector.

    max_norm : float
        Maximum allowed L2 norm.

    Returns
    -------
    psi_safe : np.ndarray

    Notes
    -----
    This is a *last-resort* safeguard.
    Under correct assumptions, this should never trigger.
    """

    psi_t = np.asarray(psi_t, dtype=float)

    norm = np.linalg.norm(psi_t)

    if norm > max_norm:
        psi_t = psi_t * (max_norm / (norm + EPS_DIV))

    return psi_t


# ============================================================
# Memory safeguards
# ============================================================


def cap_memory(
    I_t: float,
    max_value: float = MAX_MEMORY,
) -> float:
    """
    Cap memory value to prevent runaway accumulation.

    Parameters
    ----------
    I_t : float
        Memory value.

    max_value : float
        Maximum allowed memory.

    Returns
    -------
    I_safe : float

    Interpretation
    --------------
    Memory blow-up indicates:
    - violated bounded-flux assumption
    - invalid kappa choice
    - corrupted data
    """

    if not np.isfinite(I_t):
        return max_value

    return float(min(I_t, max_value))


# ============================================================
# Covariance safeguards
# ============================================================


def enforce_spd(
    C: np.ndarray,
    eps: float = EPS_COV,
) -> np.ndarray:
    """
    Enforce symmetric positive definiteness (SPD).

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Candidate covariance matrix.

    eps : float
        Diagonal regularization.

    Returns
    -------
    C_spd : np.ndarray
        Guaranteed SPD matrix.

    Notes
    -----
    - Clips negative eigenvalues
    - Reconstructs matrix safely
    """

    C = np.asarray(C, dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square")

    # Symmetrize
    C = 0.5 * (C + C.T)

    # Eigenvalue correction
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, eps, None)

    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ============================================================
# Spectral safeguards
# ============================================================


def clip_eigenvalues(
    eigvals: np.ndarray,
    *,
    min_value: float = 0.0,
    max_value: float | None = None,
) -> np.ndarray:
    """
    Clip eigenvalues to a safe range.

    Parameters
    ----------
    eigvals : np.ndarray
        Eigenvalues.

    min_value : float, default=0.0
        Lower bound.

    max_value : float or None
        Upper bound.

    Returns
    -------
    eigvals_safe : np.ndarray
    """

    eigvals = np.asarray(eigvals, dtype=float)

    if max_value is None:
        return np.clip(eigvals, min_value, None)

    return np.clip(eigvals, min_value, max_value)


# ============================================================
# Diagnostic checks (used in tests & experiments)
# ============================================================


def assert_bounded(
    value: float,
    *,
    upper: float,
    name: str = "quantity",
) -> None:
    """
    Assert that a scalar value is finite and bounded.

    Raises
    ------
    AssertionError if violated.
    """

    if not np.isfinite(value):
        raise AssertionError(f"{name} is not finite")

    if value > upper:
        raise AssertionError(
            f"{name} exceeded bound: {value} > {upper}"
        )


def check_monotone(
    seq: np.ndarray,
    name: str = "sequence",
) -> None:
    """
    Assert that a sequence is non-decreasing.

    Used for memory validation.
    """

    seq = np.asarray(seq, dtype=float)

    if np.any(seq[1:] < seq[:-1]):
        raise AssertionError(f"{name} is not monotone")


# ============================================================
# Composite safeguard (pipeline-level)
# ============================================================


def apply_safeguards(
    *,
    X: np.ndarray,
) -> Tuple[np.ndarray]:
    """
    Apply all pipeline-level safeguards to input data.

    Parameters
    ----------
    X : np.ndarray, shape (T, n)

    Returns
    -------
    X_safe : np.ndarray

    Notes
    -----
    This function is intentionally minimal.
    It is meant to be called once at pipeline entry.
    """

    X_safe = sanitize_time_series(X)
    return (X_safe,)

