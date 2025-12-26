# Chrono-Tensorial Flux Calculus (CTFC)

**A geometric framework for spatiotemporal dynamics with memory, flux, and curvature**

---

## Overview

**Chrono-Tensorial Flux Calculus (CTFC)** is a general mathematical and computational framework for representing **non-Markovian spatiotemporal systems**.  
It treats evolving covariance structures not as static objects, but as **trajectories on a flux-deformed geometric manifold**, where **temporal memory, spatial interaction, and curvature** are intrinsic.

CTFC unifies:
- temporal memory (Volterra-type integrals),
- spatial flux fields (graph-based gradients),
- covariance geometry (SPD manifolds),
- curvature surrogates (Laplacian–covariance contractions),

into a **single coherent calculus**.

This repository provides:
- 📘 **full theory documentation**
- 🧮 **exact operator implementations**
- 🧪 **reproducible experiments**
- 🧠 **a stable discrete realization (CTFC≈)** suitable for real data

---

## Why CTFC?

### The core problem

Most existing methods fail to model **temporal structure correctly**:

| Method | Limitation |
|-----|-----------|
| PCA / ICA | Static, no temporal memory |
| Kernel methods | Geometry without dynamics |
| DMD / Koopman | Linearized, weak nonlinearity |
| GARCH | Scalar variance only |
| Deep RNNs | Opaque, non-geometric |

Real systems (markets, brains, climate, physics) exhibit:
- long-range memory,
- evolving correlations,
- regime-dependent geometry,
- structural curvature.

CTFC is built **specifically** to model these phenomena.

---

## Core Idea (One Sentence)

> **Spatiotemporal systems evolve on a flux-deformed manifold of covariance tensors, not in Euclidean feature space.**

---

## Mathematical Foundations

### Objects

- Multivariate signal:  
  \[
  X(t) \in \mathbb{R}^n
  \]

- Time-dependent covariance:  
  \[
  C(t) \in \mathcal{S}_{+}^n
  \]

- Flux field (spatial distortion):  
  \[
  \Psi(t) \in \mathbb{R}^n
  \]

- Integral memory:  
  \[
  I(t) = \int_0^t \|\Psi(s)\| ds
  \]

CTFC treats \( C(t) \) as a **metric tensor** on a chrono-manifold whose geometry evolves under flux.

---

## Chrono-Operators

CTFC introduces **chrono-operators**, which generalize classical statistics:

| Operator | Meaning |
|-------|--------|
| Flux \( \Psi(t) \) | Spatial interaction / information flow |
| Memory \( I(t) \) | Non-Markovian persistence |
| Chrono-Derivative | Flux-weighted temporal response |
| Chrono-Trace | Flux-amplified variance |
| Chrono-Contraction | Energy contraction |
| Curvature Surrogate | Geometry × memory coupling |
| Flux-Scaled Eigenmodes | Spectral deformation |

All operators are:
- bounded,
- interpretable,
- numerically stable,
- discretely realizable.

---

## CTFC Embedding

The **CTFC embedding** aggregates invariant quantities:

\[
E(t) =
\big[
I(t),
\|\Psi(t)\|,
\mathrm{Trace}_{\text{chrono}}(C),
\mathrm{Contract}_{\text{chrono}}(C),
\Phi_{\text{cd}}(t),
R_{\text{approx}}(t),
\lambda'_1(t),\dots
\big]
\]

This yields a **low-dimensional manifold embedding** that preserves:
- temporal continuity,
- geometric structure,
- regime transitions.

---

## CTFC≈ : Stable Discrete Realization

The continuous CTFC equations are defined on function spaces.  
Real data is **finite, noisy, and discrete**.

**CTFC≈** is a carefully designed discrete surrogate that:
- preserves algebraic structure,
- controls variance,
- guarantees boundedness,
- is reproducible on real datasets.

It replaces:
- Volterra integrals → EWMA memory
- infinitesimal derivatives → finite differences
- continuous Laplacians → graph kernels

without breaking the geometry.

---

## Repository Structure

```text
chrono-tensorial-flux/
├── ctfc/               # Core library (operators & embeddings)
├── docs/               # Full theory documentation
├── experiments/        # Synthetic + real-world validation
├── tests/              # Theorem-level tests
├── notebooks/          # Tutorials & intuition
├── paper/              # Reference paper
└── reproducibility/    # Deterministic experiments

