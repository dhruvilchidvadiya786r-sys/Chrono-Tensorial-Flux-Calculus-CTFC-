"""
ctfc.utils.reproducibility
=========================

Reproducibility utilities for Chrono-Tensorial Flux Calculus (CTFC).

This module centralizes all reproducibility-related controls:
- random seeding,
- deterministic behavior,
- experiment metadata capture.

CTFC is deterministic by design.
This module exists to ensure *external dependencies* remain controlled.
"""

from __future__ import annotations

import os
import random
import hashlib
from typing import Dict, Any, Optional

import numpy as np


# ============================================================
# Global seeding
# ============================================================


def set_global_seed(
    seed: int,
    *,
    deterministic_numpy: bool = True,
) -> None:
    """
    Set global random seed across all relevant libraries.

    Parameters
    ----------
    seed : int
        Seed value.

    deterministic_numpy : bool, default=True
        If True, enforce deterministic NumPy behavior
        where possible.

    Notes
    -----
    - CTFC core operators are deterministic.
    - This function controls randomness from:
        • Python's random
        • NumPy
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # Optional deterministic flags
    if deterministic_numpy:
        os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# Configuration hashing (experiment fingerprinting)
# ============================================================


def hash_config(
    config: Dict[str, Any],
) -> str:
    """
    Generate a stable hash from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration parameters.

    Returns
    -------
    hash : str
        SHA256 hash string.

    Notes
    -----
    - Order-invariant
    - Stable across runs
    - Safe for logging and experiment IDs
    """

    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    # Sort keys for stability
    items = sorted(config.items())

    # Convert to canonical string
    canonical = repr(items).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


# ============================================================
# Data fingerprinting
# ============================================================


def fingerprint_data(
    X: np.ndarray,
) -> str:
    """
    Compute a fingerprint for input data.

    Parameters
    ----------
    X : np.ndarray
        Input time series.

    Returns
    -------
    fingerprint : str
        SHA256 hash of the raw data buffer.

    Notes
    -----
    - Sensitive to data ordering and values
    - Useful for detecting accidental data drift
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array")

    return hashlib.sha256(X.tobytes()).hexdigest()


# ============================================================
# Experiment metadata bundle
# ============================================================


def experiment_signature(
    *,
    X: np.ndarray,
    config: Dict[str, Any],
    seed: Optional[int] = None,
) -> Dict[str, str]:
    """
    Generate a complete experiment signature.

    Parameters
    ----------
    X : np.ndarray
        Input data.

    config : dict
        CTFC configuration.

    seed : int or None
        Random seed.

    Returns
    -------
    signature : dict
        {
            "data_fingerprint": ...,
            "config_hash": ...,
            "seed": ...
        }

    Purpose
    -------
    This signature uniquely identifies an experiment.
    It should be logged with every result.
    """

    sig = {
        "data_fingerprint": fingerprint_data(X),
        "config_hash": hash_config(config),
        "seed": str(seed) if seed is not None else "None",
    }

    return sig

