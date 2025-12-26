"""
ctfc.discrete.defaults
======================

Default configuration and theory-backed constants for CTFC≈.

This module centralizes all default hyperparameters used in the
stable discrete realization of Chrono-Tensorial Flux Calculus (CTFC≈).

Design principles
-----------------
- No magic numbers scattered across the codebase
- Defaults must be:
    • theoretically motivated
    • numerically stable
    • empirically reasonable
- All defaults must be explicitly overridable by the user

This file acts as the *configuration spine* of CTFC≈.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ============================================================
# Core CTFC≈ defaults (theory-backed)
# ============================================================

DEFAULT_WINDOW: int = 20
"""
Rolling covariance window length.

Justification:
- Must be >= 2 for covariance estimation
- Large enough to stabilize covariance
- Small enough to capture regime changes

Empirically:
- 15–30 works well for financial and synthetic systems
"""

DEFAULT_KAPPA: float = 0.9
"""
EWMA memory retention coefficient (0 < κ < 1).

Interpretation:
- κ ≈ exp(-λΔt)
- Controls decay rate of temporal memory

Theoretical role:
- Ensures bounded memory
- Approximates Volterra integral as κ → 1
"""

DEFAULT_R: int = 1
"""
Effective temporal rank parameter.

Interpretation:
- Controls strength of flux amplification
- Appears in:
    • chrono-trace
    • chrono-contraction
    • curvature amplification

r = 1 is the minimal nontrivial setting.
"""

DEFAULT_K_EIG: int = 1
"""
Number of flux-scaled spectral modes.

Interpretation:
- k = 1 captures dominant geometric mode
- Higher k captures richer latent structure

CTFC is not PCA:
- Even k = 1 is highly informative
"""

DEFAULT_SCALE: Literal["linear", "quadratic"] = "linear"
"""
Flux scaling rule.

Options:
- "linear": 1 + ||Ψ||
- "quadratic": 1 + ||Ψ||²

Default is linear for:
- numerical stability
- interpretability
"""

# ============================================================
# Numerical safety defaults
# ============================================================

DEFAULT_EPS_COV: float = 1e-8
"""
Diagonal regularization for covariance matrices.

Purpose:
- Enforce SPD
- Prevent numerical singularities
"""

DEFAULT_EPS_DIV: float = 1e-12
"""
Division safety constant.

Used to avoid division by zero in normalization.
"""

# ============================================================
# Curvature defaults
# ============================================================

DEFAULT_SIGMA: float = 1.0
"""
Kernel bandwidth for graph Laplacian construction.

Interpretation:
- Controls locality of correlation geometry
- σ = 1.0 corresponds to normalized correlation distance
"""

# ============================================================
# Dataclass wrapper (recommended usage)
# ============================================================

@dataclass(frozen=True)
class CTFCConfig:
    """
    Immutable configuration object for CTFC≈.

    This is the recommended way to pass configuration
    through experiments and pipelines.
    """

    window: int = DEFAULT_WINDOW
    kappa: float = DEFAULT_KAPPA
    r: int = DEFAULT_R
    k_eig: int = DEFAULT_K_EIG
    scale: Literal["linear", "quadratic"] = DEFAULT_SCALE
    sigma: float = DEFAULT_SIGMA
    eps_cov: float = DEFAULT_EPS_COV
    eps_div: float = DEFAULT_EPS_DIV

    def as_dict(self) -> dict:
        """Return configuration as a plain dictionary."""
        return {
            "window": self.window,
            "kappa": self.kappa,
            "r": self.r,
            "k_eig": self.k_eig,
            "scale": self.scale,
            "sigma": self.sigma,
            "eps_cov": self.eps_cov,
            "eps_div": self.eps_div,
        }


# ============================================================
# Public helper
# ============================================================

def get_default_config() -> CTFCConfig:
    """
    Return the default CTFC≈ configuration.

    This function exists to:
    - make defaults explicit
    - avoid accidental mutation
    - support experiment logging
    """

    return CTFCConfig()

