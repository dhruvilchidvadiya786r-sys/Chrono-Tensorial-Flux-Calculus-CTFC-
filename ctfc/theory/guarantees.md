
# Guarantees — Chrono-Tensorial Flux Calculus (CTFC)

This document states the **formal guarantees** provided by the
Chrono-Tensorial Flux Calculus (CTFC) framework and its stable discrete
realization (CTFC≈).

These guarantees are **conditional**:  
they hold **if and only if** the assumptions listed in `assumptions.md`
are satisfied.

This file defines **what CTFC promises**, mathematically and
computationally.



## 0. Philosophy of Guarantees

CTFC is designed to guarantee:

- boundedness,
- geometric consistency,
- numerical stability,
- interpretability of invariants,
- reproducibility of results.

CTFC **does not guarantee optimality, universality, or completeness**.
It guarantees **controlled, interpretable behavior**.

---

## 1. Well-Posedness Guarantees

### 1.1 Existence of All Operators

**Guarantee**

For any input time series \( X(t) \) satisfying the assumptions:

- all CTFC operators are well-defined,
- no operator is undefined or singular.

This includes:
- flux,
- memory,
- covariance,
- chrono-derivative,
- trace / contraction,
- curvature surrogate,
- spectral invariants.

**Why this holds**
- Explicit regularization
- No division by unbounded quantities
- All operators map finite inputs to finite outputs

---

### 1.2 Determinism

**Guarantee**

CTFC is a **deterministic mapping**:
```

X(t)  →  E(t)

```

Given:
- identical input data,
- identical configuration,

the output embedding is **identical**.

**Why this holds**
- No stochastic operators in the core
- Optional randomness is explicitly controlled via seeds

---

## 2. Boundedness Guarantees

### 2.1 Bounded Flux Propagation

**Guarantee**
```

||Ψ(t)|| ≤ M  ⇒  all downstream invariants are bounded

```

**Implications**
- Memory remains finite
- Chrono-derivative does not explode
- Curvature remains controlled

**Mechanism**
- Flux enters only through bounded amplification functions
- Explicit clipping exists as a safeguard

---

### 2.2 Bounded Memory

**Guarantee**
```

0 < κ < 1  ⇒  I(t) is bounded for bounded flux

```

**Formal statement**
\[
I_t = \kappa I_{t-1} + ||\Psi(t)|| \;\Rightarrow\;
\sup_t I(t) < \infty
\]

**Why this matters**
- Prevents long-term divergence
- Ensures curvature amplification remains finite

---

### 2.3 Bounded Curvature Surrogate

**Guarantee**
\[
|R_{\text{approx}}(t)| < \infty
\]

for all valid inputs.

**Reason**
- \( \operatorname{tr}(C L) \) is finite for SPD \( C \)
- Memory enters multiplicatively with bounded coefficient

This is a **non-trivial guarantee**: curvature cannot blow up silently.

---

## 3. Geometric Guarantees

### 3.1 Symmetry Preservation

**Guarantee**
```

C(t) is symmetric for all t
L(t) is symmetric for all t

```

**Why this holds**
- Explicit symmetrization
- Eigenvalue-based reconstruction

**Consequence**
- All spectral operators are real-valued
- No complex geometry artifacts

---

### 3.2 Positive Semi-Definiteness

**Guarantee**
```

C(t) ∈ SPD(n)  for all t

```

**Why this holds**
- Diagonal regularization
- Eigenvalue clipping

**Consequence**
- Geometry is always valid
- Distances, similarities, and spectra are meaningful

---

### 3.3 Scale Invariance of Geometry

**Guarantee**
```

X(t) → a·X(t)   (a ≠ 0)

```
does **not change relational geometry**.

**Why**
- Correlation normalization removes scale
- Laplacian constructed from correlation distances

**Consequence**
- CTFC focuses on structure, not magnitude

---

## 4. Temporal Guarantees

### 4.1 Causality

**Guarantee**

All CTFC operators are **causal**:
```

E(t) depends only on X(s) for s ≤ t

```

**Why this matters**
- Valid for online / streaming use
- No future leakage

---

### 4.2 Temporal Consistency

**Guarantee**

Small changes in input data produce **small changes in embedding**.

Formally:
```

CTFC is Lipschitz-continuous in X(t)

```

**Implication**
- Robust to noise
- No chaotic amplification

---

## 5. Chrono-Derivative Guarantees

### 5.1 Non-Negativity

**Guarantee**
```

Φ_cd(t) ≥ 0

```

**Reason**
- Defined as a norm times a positive amplification

**Interpretation**
- Chrono-derivative measures magnitude, not direction
- Zero indicates geometric stasis

---

### 5.2 Regime Sensitivity

**Guarantee**

If the covariance geometry changes abruptly,
```

Φ_cd(t) increases sharply

```

**Consequence**
- Regime transitions are detectable
- No smoothing hides structural change

---

## 6. Trace & Contraction Guarantees

### 6.1 Monotone Flux Amplification

**Guarantee**
```

||Ψ₁|| ≤ ||Ψ₂||  ⇒
Tr_chrono(Ψ₁) ≤ Tr_chrono(Ψ₂)

```

Same for chrono-contraction.

**Why**
- Rational amplification functions
- No oscillatory scaling

---

### 6.2 Bounded Amplification

**Guarantee**
```

Tr_chrono ≤ tr(C) · (1 + ||Ψ||²)

```

**Consequence**
- No runaway variance inflation
- Physical interpretability preserved

---

## 7. Curvature Guarantees

### 7.1 Zero Deformation ⇒ Zero Curvature

**Guarantee**
```

C(t) = constant  ⇒  R_approx(t) = 0

```

**Why**
- Curvature depends on alignment between C and L
- No deformation ⇒ no curvature energy

---

### 7.2 Memory Does Not Create Curvature

**Guarantee**
```

I(t) > 0  and  tr(C L) = 0  ⇒  R_approx(t) = 0

```

**Interpretation**
- Memory amplifies existing deformation
- Memory alone cannot invent structure

---

## 8. Spectral Guarantees

### 8.1 Non-Negative Eigenvalues

**Guarantee**
```

λᵢ′(t) ≥ 0  for all i, t

```

**Why**
- Based on SPD covariance
- Explicit clipping

---

### 8.2 Spectral Ordering Stability

**Guarantee**

Flux scaling preserves eigenvalue ordering.

**Consequence**
- Dominant modes remain dominant
- Spectral interpretation is stable

---

## 9. Embedding Guarantees

### 9.1 Interpretability

**Guarantee**

Every coordinate of the CTFC embedding corresponds to a **named operator**
with a precise mathematical meaning.

There are:
- no learned weights,
- no hidden features,
- no opaque transformations.

---

### 9.2 Finite Dimensionality

**Guarantee**
```

dim(E(t)) is finite and controlled

```

**Consequence**
- Embeddings are usable
- No curse-of-dimensionality explosion

---

## 10. Computational Guarantees

### 10.1 Numerical Stability

**Guarantee**
- No division by zero
- No NaNs or Infs propagate
- All matrix operations are conditioned

**Mechanism**
- Regularization
- Safeguards
- Explicit validation

---

### 10.2 Reproducibility

**Guarantee**
- Same inputs + same config ⇒ same outputs

**Why**
- Centralized defaults
- Immutable configuration
- Explicit seeding

---

## 11. What CTFC Does NOT Guarantee

CTFC explicitly does **not** guarantee:

- optimal prediction,
- minimal embeddings,
- recovery of true physical laws,
- universality across all systems,
- invariance to arbitrary noise.

These are **outside the scope** of CTFC.

---

## 12. Relationship to Assumptions

All guarantees in this document are **conditional**.

If an assumption in `assumptions.md` is violated:
- the code may still run,
- but **no guarantee applies**.

This separation is intentional and explicit.

---

## 13. Final Statement

CTFC provides **strong guarantees**, but only within its declared domain.

These guarantees make CTFC:
- mathematically defensible,
- computationally reliable,
- scientifically interpretable.

They are the reason CTFC is a **theory**, not a heuristic.
