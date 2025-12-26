"""
ctfc.discrete.ctfc_approx
========================

CTFC≈ : Stable discrete realization of Chrono-Tensorial Flux Calculus.

This module provides the *canonical, production-grade entry point*
for computing CTFC embeddings from real-world data.

CTFC≈ is not a heuristic approximation.
It is a mathematically controlled discretization of the continuous
CTFC operators, designed to:

- preserve geometric meaning,
- guarantee boundedness,
- remain stable under noise,
- be reproducible on finite data.

This file intentionally freezes defaults and exposes a minimal API.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from ctfc.core.embedding import ctfc_embedding
from ctfc.utils.validation import validate_inputs
from ctfc.utils.reproducibility import set_global_seed

# ============================================================
# Default configuration (theory-backed)
# ============================================================

DEFAULT_CONFIG = {
    "window": 20,       # rolling covariance window
    "kappa": 0.9,       # EWMA memory retention
    "r": 1,             # effective temporal rank
    "k_eig": 1,         # number of spectral modes
    "scale": "linear",  # flux scaling
}

# ============================================================
# Main CTFC≈ API (THIS IS WHAT USERS CALL)
# ============================================================


def compute_ctfc_approx(
    X: np.ndarray,
    *,
    window: int = DEFAULT_CONFIG["window"],
    kappa: float = DEFAULT_CONFIG["kappa"],
    r: int = DEFAULT_CONFIG["r"],
    k_eig: int = DEFAULT_CONFIG["k_eig"],
    scale: str = DEFAULT_CONFIG["scale"],
    seed: Optional[int] = None,
    validate: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute the CTFC≈ embedding for a multivariate time series.

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
        Multivariate time series data.

    window : int, default=20
        Rolling window size for covariance estimation.

    kappa : float, default=0.9
        Memory retention coefficient (0 < kappa < 1).

    r : int, default=1
        Effective temporal rank parameter.

    k_eig : int, default=1
        Number of flux-scaled spectral modes.

    scale : {"linear", "quadratic"}, default="linear"
        Flux scaling rule.

    seed : int, optional
        Random seed for reproducibility (only affects downstream randomness).

    validate : bool, default=True
        If True, validate inputs and configuration.

    Returns
    -------
    embedding : dict[str, np.ndarray]
        Dictionary containing CTFC≈ embedding components:

        - flux_norm
        - memory
        - chrono_derivative
        - chrono_trace
        - chrono_contraction
        - curvature
        - spectral

    Theoretical guarantees
    ----------------------
    Under bounded X and valid parameters, CTFC≈ guarantees:

    - bounded memory,
    - bounded curvature surrogate,
    - Lipschitz continuity of embedding components,
    - numerical stability for finite samples.

    This function is safe for:
    - research use,
    - empirical studies,
    - downstream ML pipelines.
    """

    # --------------------------------------------------------
    # Optional reproducibility control
    # --------------------------------------------------------
    if seed is not None:
        set_global_seed(seed)

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------
    if validate:
        validate_inputs(
            X=X,
            window=window,
            kappa=kappa,
            r=r,
            k_eig=k_eig,
            scale=scale,
        )

    # --------------------------------------------------------
    # Core CTFC embedding
    # --------------------------------------------------------
    embedding = ctfc_embedding(
        X=X,
        window=window,
        kappa=kappa,
        r=r,
        k_eig=k_eig,
        scale=scale,
    )

    return embedding


# ============================================================
# Lightweight wrapper (vectorized embedding)
# ============================================================


def ctfc_feature_matrix(
    X: np.ndarray,
    *,
    window: int = DEFAULT_CONFIG["window"],
    kappa: float = DEFAULT_CONFIG["kappa"],
    r: int = DEFAULT_CONFIG["r"],
    k_eig: int = DEFAULT_CONFIG["k_eig"],
    scale: str = DEFAULT_CONFIG["scale"],
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convert CTFC≈ embedding dictionary into a feature matrix.

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
        Input time series.

    Returns
    -------
    F : np.ndarray, shape (T, d)
        Feature matrix suitable for ML models.

    Notes
    -----
    Feature order:
        [flux_norm,
         memory,
         chrono_derivative,
         chrono_trace,
         chrono_contraction,
         curvature,
         spectral_1, ..., spectral_k]
    """

    emb = compute_ctfc_approx(
        X,
        window=window,
        kappa=kappa,
        r=r,
        k_eig=k_eig,
        scale=scale,
        seed=seed,
    )

    features = [
        emb["flux_norm"],
        emb["memory"],
        emb["chrono_derivative"],
        emb["chrono_trace"],
        emb["chrono_contraction"],
        emb["curvature"],
    ]

    spectral = emb["spectral"]
    if spectral.ndim == 1:
        features.append(spectral)
    else:
        for i in range(spectral.shape[1]):
            features.append(spectral[:, i])

    return np.column_stack(features)


# ============================================================
# Configuration helper (explicit + frozen)
# ============================================================


def get_default_config() -> Dict[str, object]:
    """
    Return a copy of the default CTFC≈ configuration.

    This function exists to:
    - make defaults explicit,
    - avoid hidden magic numbers,
    - support reproducible experiment logging.
    """

    return DEFAULT_CONFIG.copy()

