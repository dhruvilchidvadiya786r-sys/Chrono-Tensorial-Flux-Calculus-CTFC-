"""
ctfc.analysis.metrics
=====================

Evaluation and diagnostic metrics for Chrono-Tensorial Flux Calculus (CTFC).

This module defines *post-embedding metrics* that quantify:
- stability,
- regime change,
- geometric dominance,
- memory influence,
- spectral concentration.

These are NOT training losses.
They are scientific diagnostics designed to interpret CTFC outputs.

All metrics are:
- scale-aware,
- bounded where possible,
- invariant-consistent,
- numerically stable.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

# ============================================================
# Utility checks
# ============================================================


def _check_1d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return x


def _finite_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


# ============================================================
# Stability metrics
# ============================================================


def temporal_variation(
    x: np.ndarray,
) -> float:
    """
    Compute mean absolute temporal variation.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Time series.

    Returns
    -------
    tv : float

    Definition
    ----------
        TV(x) = mean_t |x_t - x_{t-1}|

    Interpretation
    --------------
    - Low value → stable quantity
    - High value → volatile dynamics

    Used for:
    - chrono-derivative stability
    - curvature smoothness
    """

    x = _check_1d(x, "x")

    if x.size < 2:
        return 0.0

    return _finite_mean(np.abs(np.diff(x)))


def normalized_variation(
    x: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute scale-normalized temporal variation.

    Definition
    ----------
        NTV(x) = mean |Δx| / (mean |x| + eps)

    Interpretation
    --------------
    - Comparable across invariants
    - Used to rank CTFC components by stability
    """

    x = _check_1d(x, "x")
    return temporal_variation(x) / (_finite_mean(np.abs(x)) + eps)


# ============================================================
# Regime-change metrics
# ============================================================


def regime_change_score(
    chrono_derivative: np.ndarray,
    threshold: float | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Detect regime changes using chrono-derivative.

    Parameters
    ----------
    chrono_derivative : np.ndarray, shape (T,)
        Φ_cd(t) sequence.

    threshold : float or None
        If None, uses mean + std heuristic.

    Returns
    -------
    flags : np.ndarray, shape (T,)
        Boolean array marking regime changes.

    score : float
        Fraction of time points flagged.

    Interpretation
    --------------
    - Spikes in Φ_cd indicate structural transitions
    """

    phi = _check_1d(chrono_derivative, "chrono_derivative")

    if threshold is None:
        mu = np.mean(phi)
        sigma = np.std(phi)
        threshold = mu + sigma

    flags = phi > threshold
    score = float(np.mean(flags))

    return flags, score


# ============================================================
# Memory dominance metrics
# ============================================================


def memory_influence_ratio(
    memory: np.ndarray,
    invariant: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Measure how strongly memory correlates with an invariant.

    Parameters
    ----------
    memory : np.ndarray, shape (T,)
        Memory sequence I(t).

    invariant : np.ndarray, shape (T,)
        Any CTFC invariant.

    Returns
    -------
    ratio : float in [0, 1]

    Definition
    ----------
        MIR = |corr(I, invariant)|

    Interpretation
    --------------
    - Near 0 → memory-independent behavior
    - Near 1 → memory-dominated dynamics
    """

    I = _check_1d(memory, "memory")
    x = _check_1d(invariant, "invariant")

    if I.size != x.size:
        raise ValueError("memory and invariant must have same length")

    if np.std(I) < eps or np.std(x) < eps:
        return 0.0

    corr = np.corrcoef(I, x)[0, 1]
    return float(abs(corr))


# ============================================================
# Spectral concentration metrics
# ============================================================


def spectral_concentration(
    spectral: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Measure concentration of spectral energy.

    Parameters
    ----------
    spectral : np.ndarray, shape (T, k)
        Flux-scaled eigenvalues.

    Returns
    -------
    concentration : float in [0, 1]

    Definition
    ----------
        SC = mean_t ( λ₁ / sum_i λ_i )

    Interpretation
    --------------
    - Near 1 → low-rank / dominant mode
    - Near 0 → diffuse geometry
    """

    spectral = np.asarray(spectral, dtype=float)

    if spectral.ndim != 2:
        raise ValueError("spectral must have shape (T, k)")

    total = np.sum(spectral, axis=1) + eps
    ratio = spectral[:, 0] / total

    return float(_finite_mean(ratio))


# ============================================================
# Geometry dominance metrics
# ============================================================


def curvature_to_trace_ratio(
    curvature: np.ndarray,
    chrono_trace: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compare curvature magnitude to variance magnitude.

    Definition
    ----------
        CTR = mean |curvature| / (mean |trace| + eps)

    Interpretation
    --------------
    - High → geometry-driven system
    - Low → variance-dominated system
    """

    R = _check_1d(curvature, "curvature")
    T = _check_1d(chrono_trace, "chrono_trace")

    if R.size != T.size:
        raise ValueError("Inputs must have same length")

    return _finite_mean(np.abs(R)) / (_finite_mean(np.abs(T)) + eps)


# ============================================================
# Aggregate diagnostic report
# ============================================================


def ctfc_diagnostic_report(
    embedding: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Generate a compact diagnostic report from CTFC embedding.

    Parameters
    ----------
    embedding : dict
        Output of compute_ctfc_approx.

    Returns
    -------
    report : dict[str, float]

    Reported metrics
    ----------------
    - chrono_stability
    - curvature_stability
    - regime_change_rate
    - memory_curvature_coupling
    - spectral_concentration
    - geometry_dominance
    """

    report: Dict[str, float] = {}

    # Stability
    report["chrono_stability"] = normalized_variation(
        embedding["chrono_derivative"]
    )
    report["curvature_stability"] = normalized_variation(
        embedding["curvature"]
    )

    # Regime changes
    _, rc_score = regime_change_score(
        embedding["chrono_derivative"]
    )
    report["regime_change_rate"] = rc_score

    # Memory influence
    report["memory_curvature_coupling"] = memory_influence_ratio(
        embedding["memory"],
        embedding["curvature"],
    )

    # Spectral
    report["spectral_concentration"] = spectral_concentration(
        embedding["spectral"]
    )

    # Geometry vs variance
    report["geometry_dominance"] = curvature_to_trace_ratio(
        embedding["curvature"],
        embedding["chrono_trace"],
    )

    return report

