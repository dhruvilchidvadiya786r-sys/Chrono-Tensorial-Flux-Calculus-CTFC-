




# Assumptions — Chrono-Tensorial Flux Calculus (CTFC)

This document lists **all explicit assumptions** underlying the
Chrono-Tensorial Flux Calculus (CTFC) framework.

These assumptions are **not hidden** and **not optional**.
They define the **domain of validity** of the theory and its discrete
realization (CTFC≈).

If an assumption is violated, CTFC behavior is **undefined unless a
safeguard is triggered**.

---

## 0. Philosophy of Assumptions

CTFC is designed to be:

- mathematically explicit,
- geometrically interpretable,
- numerically stable.

To achieve this, CTFC **does not attempt to model arbitrary chaos**.
It models **structured, persistent, geometry-bearing dynamics**.

All assumptions exist to:
- guarantee boundedness,
- preserve interpretability,
- maintain geometric meaning.

---

## 1. Data-Level Assumptions

### 1.1 Finite Observations

**Assumption**
```

X(t) ∈ ℝⁿ  is finite for all t

```

**Explanation**
- No NaNs or ±∞ values are allowed in theory.
- Discrete CTFC≈ replaces invalid entries via sanitization.

**Violation impact**
- Covariance becomes undefined.
- Flux magnitudes may blow up.

**Safeguard**
- `sanitize_time_series` in `safeguards.py`

---

### 1.2 Fixed Dimensionality

**Assumption**
```

dim(X(t)) = n  is constant over time

```

**Explanation**
- CTFC geometry lives in a fixed SPD manifold.
- Variable dimensionality breaks manifold consistency.

**Violation impact**
- Covariance geometry is undefined.
- Laplacian construction fails.

**Safeguard**
- Hard validation failure.

---

### 1.3 Sufficient Sample Size

**Assumption**
```

Window size w ≥ 2

```

**Explanation**
- Covariance estimation requires at least two samples.
- Larger windows stabilize geometry.

**Violation impact**
- Covariance degeneracy.
- Chrono-derivative becomes meaningless.

**Safeguard**
- Minimum window enforcement.

---

## 2. Flux Assumptions

### 2.1 Bounded Flux Magnitude

**Assumption**
```

||Ψ(t)|| ≤ M < ∞

```

**Explanation**
- Flux represents interaction strength.
- Unbounded flux implies physical inconsistency.

**Violation impact**
- Memory diverges.
- Chrono-derivative explodes.

**Safeguard**
- Flux clipping (`clip_flux`)
- Memory capping (`cap_memory`)

---

### 2.2 Flux Encodes Interaction, Not Noise

**Assumption**
- Flux captures *structured interaction*, not white noise.

**Explanation**
- CTFC is not a denoising framework.
- Random flux destroys geometric meaning.

**Violation impact**
- Geometry becomes unstable.
- Curvature loses interpretability.

**Safeguard**
- None (theoretical assumption)

---

## 3. Memory Assumptions

### 3.1 Non-Negative Memory

**Assumption**
```

I(t) ≥ 0

```

**Explanation**
- Memory measures accumulated deformation.
- Negative memory is meaningless.

**Violation impact**
- Curvature amplification breaks.

**Safeguard**
- Memory construction guarantees non-negativity.

---

### 3.2 Exponential Forgetting

**Assumption**
```

0 < κ < 1

```

**Explanation**
- Ensures bounded memory.
- Approximates Volterra integrals.

**Violation impact**
- κ ≥ 1 → divergence
- κ ≤ 0 → memory collapse

**Safeguard**
- Parameter validation.

---

## 4. Covariance Geometry Assumptions

### 4.1 Symmetry

**Assumption**
```

C(t) = C(t)ᵀ

```

**Explanation**
- Covariance is a symmetric bilinear form.
- Required for spectral geometry.

**Violation impact**
- Eigenvalues become complex.

**Safeguard**
- Explicit symmetrization.

---

### 4.2 Positive Semi-Definiteness

**Assumption**
```

xᵀ C(t) x ≥ 0  for all x

```

**Explanation**
- Covariance defines a valid metric tensor.
- Negative eigenvalues are non-physical.

**Violation impact**
- Geometry invalid.
- Curvature meaningless.

**Safeguard**
- Eigenvalue clipping
- SPD enforcement

---

## 5. Chrono-Derivative Assumptions

### 5.1 Finite Temporal Variation

**Assumption**
```

||C(t) − C(t−1)|| < ∞

```

**Explanation**
- Chrono-derivative measures deformation speed.
- Infinite jumps are excluded.

**Violation impact**
- Regime detection meaningless.

**Safeguard**
- Implicit via finite data + safeguards.

---

### 5.2 Truncation Validity

**Assumption**
- Higher-order temporal derivatives are negligible.

**Explanation**
- CTFC≈ truncates the infinite chrono-series.
- Valid when geometry evolves smoothly.

**Violation impact**
- Loss of theoretical fidelity.

**Safeguard**
- Conservative scaling choices.

---

## 6. Similarity & Graph Assumptions

### 6.1 Correlation as Geometry

**Assumption**
- Correlation captures relational structure.

**Explanation**
- Scale-invariant geometry is required.
- Absolute covariance is insufficient.

**Violation impact**
- Laplacian becomes scale-dependent.

**Safeguard**
- Correlation normalization.

---

### 6.2 Fully Connected Graph

**Assumption**
- Graph is fully connected (dense affinity).

**Explanation**
- Ensures Laplacian well-posedness.
- Avoids disconnected components.

**Violation impact**
- Zero eigenvalues proliferate.

**Safeguard**
- Gaussian affinity kernel.

---

## 7. Curvature Assumptions

### 7.1 Curvature Is Surrogate, Not Riemannian

**Assumption**
- `R_approx` is **not** classical curvature.

**Explanation**
- It measures deformation energy, not sectional curvature.

**Violation impact**
- Conceptual misinterpretation.

**Safeguard**
- Explicit naming and documentation.

---

### 7.2 Memory Amplifies, Not Creates Curvature

**Assumption**
- Memory scales curvature; it does not generate it.

**Explanation**
- No deformation ⇒ zero curvature regardless of memory.

**Violation impact**
- Artificial curvature.

**Safeguard**
- Multiplicative structure.

---

## 8. Spectral Assumptions

### 8.1 Eigenvalue Ordering Stability

**Assumption**
- Leading eigenvalues correspond to dominant geometry.

**Explanation**
- Small perturbations do not reorder spectrum.

**Violation impact**
- Spectral modes lose meaning.

**Safeguard**
- Regularization and clipping.

---

## 9. Embedding Assumptions

### 9.1 Low-Dimensional Sufficiency

**Assumption**
- A small number of invariants captures system state.

**Explanation**
- CTFC is not exhaustive.
- It captures *dominant geometry*.

**Violation impact**
- Information loss.

**Safeguard**
- Increase spectral modes (k).

---

## 10. What CTFC Does NOT Assume

CTFC explicitly **does not assume**:

- stationarity,
- Gaussianity,
- linear dynamics,
- equilibrium,
- ergodicity,
- Markovianity.

This is intentional.

---

## 11. Consequences of Assumption Violations

| Violation | Result |
|--------|------|
| Unbounded flux | Memory blow-up |
| Non-SPD covariance | Invalid geometry |
| Random flux | Noisy curvature |
| Variable dimension | Manifold failure |
| κ ≥ 1 | Divergence |

Safeguards attempt recovery **only when mathematically justified**.

---

## 12. Final Statement

CTFC is **not a black-box method**.

It is a **geometric theory implemented as code**.
These assumptions define its **mathematical contract**.

If you violate them:
- the code may still run,
- but the theory does not apply.

This document exists to prevent that confusion.


