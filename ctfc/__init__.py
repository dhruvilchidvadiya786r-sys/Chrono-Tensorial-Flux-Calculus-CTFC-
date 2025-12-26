"""
Chrono-Tensorial Flux Calculus (CTFC)
===================================

CTFC is a geometric framework for modeling spatiotemporal dynamics
with non-Markovian memory, spatial flux interactions, and evolving
covariance geometry.

This package provides:
- Chrono-operators (flux, memory, derivatives, curvature)
- Stable discrete realization (CTFC≈)
- Geometry-aware embeddings
- Reproducible experimental tools

Philosophy
----------
CTFC treats covariance matrices as trajectories on a flux-deformed
manifold rather than static statistical objects. Temporal structure
is encoded via Volterra-type memory, flux-weighted derivatives, and
curvature surrogates.

This package is designed for:
- Applied mathematics research
- Scientific computing
- Interpretable spatiotemporal modeling
- Reproducible experimentation

The public API is intentionally minimal and explicit.
"""

# ============================================================
# Versioning
# ============================================================

__version__ = "0.1.0"

# ============================================================
# Core Mathematical Operators (Public API)
# ============================================================

from ctfc.core.flux import (
    compute_flux,
    flux_magnitude,
)

from ctfc.core.memory import (
    integral_memory,
    ewma_memory,
)

from ctfc.core.covariance import (
    rolling_covariance,
    safe_covariance,
)

from ctfc.core.chrono_derivative import (
    chrono_derivative,
)

from ctfc.core.trace import (
    chrono_trace,
    chrono_contraction,
)

from ctfc.core.curvature import (
    curvature_surrogate,
)

from ctfc.core.spectral import (
    flux_scaled_eigenvalues,
)

from ctfc.core.embedding import (
    ctfc_embedding,
)

# ============================================================
# Discrete Stable Realization (CTFC≈)
# ============================================================

from ctfc.discrete.ctfc_approx import (
    compute_ctfc_approx,
)

# ============================================================
# Geometry Utilities (Advanced Users)
# ============================================================

from ctfc.geometry.similarity import (
    correlation_similarity,
)

from ctfc.geometry.laplacian import (
    graph_laplacian,
)

# ============================================================
# Validation & Reproducibility Utilities
# ============================================================

from ctfc.utils.validation import (
    validate_inputs,
)

from ctfc.utils.reproducibility import (
    set_global_seed,
)

# ============================================================
# Explicit Public Interface
# ============================================================

__all__ = [
    # version
    "__version__",

    # flux
    "compute_flux",
    "flux_magnitude",

    # memory
    "integral_memory",
    "ewma_memory",

    # covariance
    "rolling_covariance",
    "safe_covariance",

    # chrono operators
    "chrono_derivative",
    "chrono_trace",
    "chrono_contraction",

    # curvature
    "curvature_surrogate",

    # spectral
    "flux_scaled_eigenvalues",

    # embedding
    "ctfc_embedding",

    # discrete CTFC≈
    "compute_ctfc_approx",

    # geometry (advanced)
    "correlation_similarity",
    "graph_laplacian",

    # utils
    "validate_inputs",
    "set_global_seed",
]

