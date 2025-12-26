"""
ctfc.core.embedding
===================

CTFC embedding construction.

This module assembles all Chrono-Tensorial Flux Calculus (CTFC)
operators into a single, structured embedding vector E(t).

The embedding is:
- interpretable (each coordinate has mathematical meaning),
- invariant-driven (no arbitrary features),
- stable under noise,
- suitable for downstream analysis or learning.

This is the canonical representation produced by CTFC.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from ctfc.core.flux import compute_flux, flux_magnitude
from ctfc.core.memory import ewma_memory
from ctfc.core.covariance import rolling_covariance
from ctfc.core.chrono_derivative import chrono_derivative_series
from ctfc.core.trace import chrono_trace, chrono_contraction
from ctfc.core.curvature import curvature_series
from ctfc.core.spectral import spectral_series

# ============================================================
# Single-time embedding (conceptual reference)
# ============================================================


def ctfc_embedding_single(
    C_t: np.ndarray,
    psi_t: np.ndarray,
    I_t: float,
    phi_cd_t: float,
    r: int = 1,
    k_eig: int = 1,
    scale: str = "linear",
) -> Dict[str, float | np.ndarray]:
    """
    Construct CTFC embedding at a single time step.

    Parameters
    ----------
    C_t : np.ndarray, shape (n, n)
        Covariance matrix at time t.

    psi_t : np.ndarray, shape (n,)
        Flux vector at time t.

    I_t : float
        Memory value at time t.

    phi_cd_t : float
        Chrono-derivative magnitude at time t.

    r : int, default=1
        Effective temporal rank.

    k_eig : int, default=1
        Number of spectral modes.

    scale : {"linear", "quadratic"}, default="linear"
        Flux scaling rule for spectral modes.

    Returns
    -------
    embedding : dict
        Dictionary of named CTFC invariants.

    Notes
    -----
    This function is mostly conceptual; in practice,
    the batched embedding below is used.
    """

    emb: Dict[str, float | np.ndarray] = {}

    # Flux invariants
    psi_norm = flux_magnitude(psi_t)
    emb["flux_norm"] = psi_norm

    # Memory
    emb["memory"] = I_t

    # Chrono-derivative
    emb["chrono_derivative"] = phi_cd_t

    # Trace-based invariants
    emb["chrono_trace"] = chrono_trace(C_t, psi_t, r=r)
    emb["chrono_contraction"] = chrono_contraction(C_t, psi_t, r=r)

    return emb


# ============================================================
# Full CTFC embedding over time (MAIN API)
# ============================================================


def ctfc_embedding(
    X: np.ndarray,
    window: int = 20,
    kappa: float = 0.9,
    r: int = 1,
    k_eig: int = 1,
    scale: str = "linear",
) -> Dict[str, np.ndarray]:
    """
    Compute the full CTFC embedding for a multivariate time series.

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
        Multivariate time series.

    window : int, default=20
        Rolling window length for covariance estimation.

    kappa : float, default=0.9
        EWMA memory retention coefficient.

    r : int, default=1
        Effective temporal rank parameter.

    k_eig : int, default=1
        Number of spectral modes.

    scale : {"linear", "quadratic"}, default="linear"
        Flux scaling rule for spectral invariants.

    Returns
    -------
    embedding : dict[str, np.ndarray]
        Dictionary containing CTFC embedding components:

        - "flux_norm": ||Ψ(t)||
        - "memory": I(t)
        - "chrono_derivative": Φ_cd(t)
        - "chrono_trace": Trace_chrono(t)
        - "chrono_contraction": Contract_chrono(t)
        - "curvature": R_approx(t)
        - "spectral": λ'_i(t), i = 1..k_eig

    Interpretation
    --------------
    Each key corresponds to a mathematically defined invariant.
    Together they form a low-dimensional, geometry-aware
    representation of spatiotemporal dynamics.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must have shape (T, n)")

    T, n = X.shape

    # --------------------------------------------------------
    # Step 1: Flux computation
    # --------------------------------------------------------
    Psi = np.zeros((T, n), dtype=float)
    for t in range(T):
        Psi[t] = compute_flux(X[t])

    flux_norms = np.linalg.norm(Psi, axis=1)

    # --------------------------------------------------------
    # Step 2: Memory (EWMA)
    # --------------------------------------------------------
    memory = ewma_memory(flux_norms, kappa=kappa)

    # --------------------------------------------------------
    # Step 3: Rolling covariance
    # --------------------------------------------------------
    C_seq = rolling_covariance(X, window=window)

    # --------------------------------------------------------
    # Step 4: Chrono-derivative
    # --------------------------------------------------------
    phi_cd = chrono_derivative_series(C_seq, Psi, scale="linear")

    # --------------------------------------------------------
    # Step 5: Trace-based invariants
    # --------------------------------------------------------
    chrono_tr = np.zeros(T, dtype=float)
    chrono_ctr = np.zeros(T, dtype=float)

    for t in range(T):
        chrono_tr[t] = chrono_trace(C_seq[t], Psi[t], r=r)
        chrono_ctr[t] = chrono_contraction(C_seq[t], Psi[t], r=r)

    # --------------------------------------------------------
    # Step 6: Curvature surrogate
    # --------------------------------------------------------
    curvature = curvature_series(C_seq, memory, r=r)

    # --------------------------------------------------------
    # Step 7: Spectral invariants
    # --------------------------------------------------------
    spectral = spectral_series(C_seq, Psi, k=k_eig, scale=scale)

    # --------------------------------------------------------
    # Assemble embedding
    # --------------------------------------------------------
    embedding: Dict[str, np.ndarray] = {
        "flux_norm": flux_norms,
        "memory": memory,
        "chrono_derivative": phi_cd,
        "chrono_trace": chrono_tr,
        "chrono_contraction": chrono_ctr,
        "curvature": curvature,
        "spectral": spectral,
    }

    return embedding

