
# Operator Map — Chrono-Tensorial Flux Calculus (CTFC)

This document is the **formal operator-level specification** of the
Chrono-Tensorial Flux Calculus (CTFC).

It provides a **one-to-one mapping** between:

- mathematical operators,
- their theoretical definitions,
- their concrete code implementations,
- and their geometric / physical interpretations.

This file is the **authoritative contract** of the CTFC framework.

---

## 0. Global Pipeline Overview

CTFC transforms a multivariate time series into a sequence of
**geometry-aware, memory-sensitive invariants**.



Raw data X(t)
↓
Flux Ψ(t)
↓
Temporal Memory I(t)
↓
Covariance Geometry C(t)
↓
Chrono-Derivative Φ_cd(t)
↓
Invariant Operators
├─ Trace / Contraction
├─ Curvature Surrogate
├─ Spectral Modes
↓
CTFC Embedding E(t)


Each arrow corresponds to a **mathematically defined operator**.



## 1. Flux Operators

### 1.1 Flux Vector

**Mathematical definition**
```

Ψ(t) = F(X(t))

```

**Code**
- `ctfc/core/flux.py`
  - `compute_flux(X_t)`
  - `flux_magnitude(Ψ_t)`

**Output**
- Flux vector `Ψ(t)`
- Flux norm `||Ψ(t)||`

**Interpretation**
- Encodes instantaneous interaction / activity
- Drives all non-Markovian effects
- Zero flux ⇒ static geometry

---

## 2. Temporal Memory Operators

### 2.1 Integral Memory (Theoretical Reference)

**Mathematical definition**
```

I(t) = ∫₀ᵗ ||Ψ(s)|| ds

```

**Code**
- `ctfc/core/memory.py`
  - `integral_memory(...)`

**Interpretation**
- Accumulated geometric deformation
- Monotone by construction
- Reference operator (not used directly in CTFC≈)

---

### 2.2 EWMA / Volterra Memory (CTFC≈)

**Mathematical definition**
```

Iₜ = κ Iₜ₋₁ + ||Ψ(t)||

```

**Code**
- `ctfc/core/memory.py`
  - `ewma_memory(...)`

**Interpretation**
- Stable discrete Volterra operator
- Encodes persistence
- Prevents unbounded growth

---

## 3. Covariance Geometry

### 3.1 Covariance as a Metric Tensor

**Mathematical definition**
```

C(t) = Cov(Xₜ₋w : Xₜ)

```

**Code**
- `ctfc/core/covariance.py`
  - `safe_covariance(...)`
  - `rolling_covariance(...)`

**Properties**
- Symmetric
- Positive semi-definite
- Regularized

**Interpretation**
- Local geometry of the system
- Point on the SPD manifold
- Base object for all geometric operators

---

## 4. Chrono-Derivative Operator

### 4.1 Chrono-Derivative (Truncated)

**Theoretical operator**
```

∇chrono C(t)
= dC/dt + Σₖ ( ||Ψ(t)||ᵏ / k! ) dᵏC/dtᵏ

```

**CTFC≈ realization**
```

Φ_cd(t) = ||C(t) − C(t−1)|| · g(||Ψ(t)||)

```

**Code**
- `ctfc/core/chrono_derivative.py`
  - `chrono_derivative(...)`
  - `chrono_derivative_series(...)`

**Interpretation**
- Measures speed of geometric deformation
- Detects regime transitions
- Flux-amplified temporal response

---

## 5. Trace-Based Invariants

### 5.1 Chrono-Trace

**Mathematical definition**
```

Tr_chrono(C)
= tr(C) · (1 + r/(r+2) · ||Ψ||²)

```

**Code**
- `ctfc/core/trace.py`
  - `chrono_trace(...)`

**Interpretation**
- Flux-inflated total variance
- Global energy measure

---

### 5.2 Chrono-Contraction

**Mathematical definition**
```

Con_chrono(C)
= tr(C) · (1 + r/(r+1) · ||Ψ||²)

```

**Code**
- `ctfc/core/trace.py`
  - `chrono_contraction(...)`

**Interpretation**
- Directional / dissipative energy
- Stronger flux sensitivity than chrono-trace

---

## 6. Similarity & Geometry Operators

### 6.1 Correlation Similarity

**Mathematical definition**
```

Sᵢⱼ = Cᵢⱼ / sqrt(Cᵢᵢ Cⱼⱼ)

```

**Code**
- `ctfc/geometry/similarity.py`
  - `correlation_similarity(...)`

**Interpretation**
- Scale-invariant relational geometry
- Basis for graph construction

---

### 6.2 Distance & Affinity

**Definitions**
```

Dᵢⱼ = 1 − Sᵢⱼ
Wᵢⱼ = exp( −Dᵢⱼ² / (2σ²) )

```

**Code**
- `similarity_to_distance(...)`
- `affinity_kernel(...)`

---

## 7. Laplacian Operators

### 7.1 Graph Laplacian

**Unnormalized**
```

L = D − W

```

**Normalized**
```

L = I − D⁻¹ᐟ² W D⁻¹ᐟ²

```

**Code**
- `ctfc/geometry/laplacian.py`
  - `unnormalized_laplacian(...)`
  - `normalized_laplacian(...)`
  - `laplacian_from_distance(...)`

**Interpretation**
- Discrete geometry operator
- Encodes curvature and connectivity

---

## 8. Curvature Surrogate

### 8.1 CTFC Curvature

**Mathematical definition**
```

R_approx(t)
= tr(C(t) L(t)) · (1 + α I(t)),
α = r/(r+1)

```

**Code**
- `ctfc/core/curvature.py`
  - `curvature_surrogate(...)`
  - `curvature_series(...)`

**Interpretation**
- Structural stress of geometry
- Memory-amplified deformation energy

---

## 9. Spectral Operators

### 9.1 Flux-Scaled Eigenvalues

**Definition**
```

λᵢ′(t) = λᵢ(C(t)) · g(||Ψ(t)||)

```

**Code**
- `ctfc/core/spectral.py`
  - `flux_scaled_eigenvalues(...)`
  - `spectral_series(...)`

**Interpretation**
- Interaction-aware latent modes
- Geometry-preserving spectral structure

---

## 10. CTFC Embedding Operator

### 10.1 Embedding Vector

**Definition**
```

E(t) =
[
||Ψ||,
I,
Φ_cd,
Tr_chrono,
Con_chrono,
R_approx,
λ₁′, …, λₖ′
]

```

**Code**
- `ctfc/core/embedding.py`
  - `ctfc_embedding(...)`

**Interpretation**
- Compact, interpretable system state
- Geometry-aware time representation

---

## 11. Discrete CTFC≈ Operator

### 11.1 Canonical Entry Point

**Code**
- `ctfc/discrete/ctfc_approx.py`
  - `compute_ctfc_approx(...)`

**Role**
- Freezes safe defaults
- Enforces validation and safeguards
- Provides reproducible API

---

## 12. Safeguards & Defaults

### Safeguards
- `ctfc/discrete/safeguards.py`
  - SPD enforcement
  - Flux clipping
  - Memory capping

### Defaults
- `ctfc/discrete/defaults.py`
  - Theory-backed parameters
  - Single source of truth

---

## 13. Analysis Metrics

**Code**
- `ctfc/analysis/metrics.py`

**Purpose**
- Stability analysis
- Regime detection
- Memory dominance
- Geometry vs variance dominance

---

## 14. Summary Table

| Concept | Operator | File |
|------|--------|------|
| Flux | Ψ(t) | flux.py |
| Memory | I(t) | memory.py |
| Geometry | C(t) | covariance.py |
| Deformation | Φ_cd | chrono_derivative.py |
| Variance | Tr_chrono | trace.py |
| Curvature | R_approx | curvature.py |
| Spectrum | λ′ | spectral.py |
| Embedding | E(t) | embedding.py |
| Discretization | CTFC≈ | ctfc_approx.py |

---

## Final Note

This operator map is **not optional documentation**.

It is the **formal definition of CTFC as a computational theory**.
If something is not listed here, it is **not part of CTFC**.

This document ensures:
- mathematical clarity,
- implementation correctness,
- long-term maintainability,
- reviewer confidence.
```

