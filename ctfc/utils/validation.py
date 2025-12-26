"""
ctfc.utils.validation
====================

Input and configuration validation for Chrono-Tensorial Flux Calculus (CTFC).

This module enforces *all non-negotiable preconditions* required for the
mathematical validity and numerical stability of CTFC and CTFC≈.

Design principles
-----------------
- Fail fast, fail loudly.
- No silent coercion of invalid parameters.
- Every check corresponds to an assumption or guarantee.

If validation passes, CTFC guarantees apply.
If validation fails, CTFC is *not defined*.
"""

from __future__ import annotations

import numpy as np
from typing import Literal


# ============================================================
# Core input validation
# ============================================================


def validate_time_series(
    X: np.ndarray,
) -> None:
    """
    Validate the raw input time series.

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
        Multivariate time series.

    Raises
    ------
    ValueError
        If input violates CTFC assumptions.
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array")

    if X.ndim != 2:
        raise ValueError("X must have shape (T, n)")

    T, n = X.shape

    if T < 2:
        raise ValueError(
            "Time series must contain at least 2 time steps"
        )

    if n < 1:
        raise ValueError(
            "Time series must have at least one dimension"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Time series contains NaN or infinite values"
        )


# ============================================================
# Parameter validation
# ============================================================


def validate_window(
    window: int,
    T: int,
) -> None:
    """
    Validate rolling window size.

    Parameters
    ----------
    window : int
        Rolling covariance window.

    T : int
        Length of time series.
    """

    if not isinstance(window, int):
        raise TypeError("window must be an integer")

    if window < 2:
        raise ValueError("window must be >= 2")

    if window > T:
        raise ValueError(
            "window cannot exceed length of time series"
        )


def validate_kappa(
    kappa: float,
) -> None:
    """
    Validate memory retention coefficient κ.
    """

    if not isinstance(kappa, (int, float)):
        raise TypeError("kappa must be a real number")

    if not (0.0 < kappa < 1.0):
        raise ValueError(
            "kappa must satisfy 0 < kappa < 1 "
            "(required for bounded memory)"
        )


def validate_r(
    r: int,
) -> None:
    """
    Validate effective temporal rank parameter r.
    """

    if not isinstance(r, int):
        raise TypeError("r must be an integer")

    if r < 0:
        raise ValueError(
            "r must be non-negative"
        )


def validate_k_eig(
    k_eig: int,
    n: int,
) -> None:
    """
    Validate number of spectral modes.
    """

    if not isinstance(k_eig, int):
        raise TypeError("k_eig must be an integer")

    if k_eig < 1:
        raise ValueError(
            "k_eig must be >= 1"
        )

    if k_eig > n:
        raise ValueError(
            "k_eig cannot exceed data dimension"
        )


def validate_scale(
    scale: str,
) -> None:
    """
    Validate flux scaling rule.
    """

    if scale not in ("linear", "quadratic"):
        raise ValueError(
            "scale must be either 'linear' or 'quadratic'"
        )


# ============================================================
# Composite validation (used by CTFC≈)
# ============================================================


def validate_inputs(
    *,
    X: np.ndarray,
    window: int,
    kappa: float,
    r: int,
    k_eig: int,
    scale: str,
) -> None:
    """
    Validate all CTFC≈ inputs and configuration parameters.

    This function is the **single validation entry point**
    used by `compute_ctfc_approx`.

    Parameters
    ----------
    X : np.ndarray
        Input time series.

    window : int
        Rolling covariance window.

    kappa : float
        Memory retention coefficient.

    r : int
        Effective temporal rank.

    k_eig : int
        Number of spectral modes.

    scale : str
        Flux scaling rule.

    Raises
    ------
    ValueError
        If any assumption is violated.
    """

    # Validate data
    validate_time_series(X)

    T, n = X.shape

    # Validate parameters
    validate_window(window, T)
    validate_kappa(kappa)
    validate_r(r)
    validate_k_eig(k_eig, n)
    validate_scale(scale)


# ============================================================
# Optional soft warnings (non-fatal)
# ============================================================


def warn_if_suspicious(
    *,
    window: int,
    kappa: float,
    T: int,
) -> None:
    """
    Emit warnings for *potentially problematic but valid* settings.

    This function does NOT raise errors.
    It exists to guide expert users.

    Examples
    --------
    - window very small
    - kappa extremely close to 1
    """

    if window < 5:
        print(
            "[CTFC WARNING] window is very small; "
            "covariance may be noisy."
        )

    if kappa > 0.98:
        print(
            "[CTFC WARNING] kappa is very close to 1; "
            "memory may decay very slowly."
        )

    if window > T // 2:
        print(
            "[CTFC WARNING] window is large relative to data length; "
            "temporal resolution may be reduced."
        )

