


# Chrono-Tensorial Flux Calculus (CTFC)

**Chrono-Tensorial Flux Calculus (CTFC)** is a geometric framework for
analyzing multivariate time series with **memory, structure, and
non-Markovian dynamics**.

CTFC treats time-evolving covariance not as a statistic, but as a
**geometric object** that deforms under persistent interaction (flux).
From this viewpoint, CTFC derives a family of **interpretable,
geometry-aware invariants** that summarize system dynamics.

This documentation describes the **theory, operators, guarantees, and
usage** of CTFC and its stable discrete realization **CTFC≈**.

---

## Why CTFC Exists

Most time-series methods assume at least one of the following:

- stationarity,
- Markovian dynamics,
- linear response,
- equilibrium behavior.

CTFC assumes **none of these**.

Instead, CTFC is built on three principles:

1. **Geometry**  
   Correlations define a geometry, not just a matrix.

2. **Memory**  
   Persistent interaction accumulates deformation over time.

3. **Flux**  
   Structure changes are driven by interaction strength, not noise.

CTFC was created to formalize these ideas **mathematically and
computationally**.

---

## Core Idea (One Sentence)

> **CTFC studies how covariance geometry evolves under flux-driven,
memory-amplified deformation, and encodes this evolution into
interpretable invariants.**

---

## High-Level Pipeline

CTFC processes data through a fixed sequence of operators:



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



Every step is:
- mathematically defined,
- explicitly implemented,
- independently testable.



## What CTFC Produces

For each time step `t`, CTFC produces an **embedding vector**:



E(t) =
[
||Ψ(t)||,          # flux magnitude
I(t),              # temporal memory
Φ_cd(t),           # chrono-derivative
Tr_chrono(t),      # flux-inflated variance
Con_chrono(t),     # directional contraction
R_approx(t),       # curvature surrogate
λ₁′(t)…λₖ′(t)      # flux-scaled spectral modes
]



Each coordinate has:
- a precise mathematical definition,
- a geometric interpretation,
- explicit guarantees.

There are **no learned weights** and **no hidden features**.



## CTFC≈ (Discrete Realization)

CTFC≈ is the **stable, finite-sample realization** of the continuous
CTFC theory.

It provides:

- bounded memory via EWMA (Volterra kernel),
- regularized covariance geometry,
- truncated chrono-derivatives,
- safeguarded numerical behavior.

CTFC≈ is what you actually run in practice.



## Design Goals

CTFC is designed to be:

- **Interpretable**  
  Every output has a name and meaning.

- **Geometrically consistent**  
  All operators respect symmetry and SPD structure.

- **Numerically stable**  
  No silent divergence or undefined behavior.

- **Reproducible**  
  Deterministic core, explicit configuration, auditable runs.

- **Theory-first**  
  Code follows mathematics, not the other way around.



## What CTFC Is *Not*

CTFC is **not**:

- a black-box ML model,
- a forecasting algorithm,
- a denoising method,
- a universal dynamical system solver.

CTFC is a **representation framework**.
What you do with the representation is up to you.


## Documentation Structure

This documentation is organized as follows:

###  Theory & Contracts
- `operator_map.md` — complete operator dictionary
- `assumptions.md` — explicit domain of validity
- `guarantees.md` — formal behavioral guarantees

###  Core Implementation
- `flux.md`
- `memory.md`
- `covariance.md`
- `chrono_derivative.md`
- `trace.md`
- `curvature.md`
- `spectral.md`
- `embedding.md`

###  Geometry Layer
- `similarity.md`
- `laplacian.md`

###  Discrete System (CTFC≈)
- `defaults.md`
- `safeguards.md`
- `validation.md`
- `reproducibility.md`
- `logging.md`
- `ctfc_approx.md`

###  Analysis
- `metrics.md`

Each document corresponds **directly** to code in the repository.

---

## Quick Start (Minimal)

```python
from ctfc.discrete.ctfc_approx import compute_ctfc_approx

E = compute_ctfc_approx(X)
````

This single call:

* validates inputs,
* applies safeguards,
* computes all CTFC invariants,
* returns a structured embedding.

---

## Reproducibility Example

```python
from ctfc.discrete.defaults import CTFCConfig
from ctfc.utils.reproducibility import experiment_signature

config = CTFCConfig()
embedding = compute_ctfc_approx(X, **config.as_dict())

signature = experiment_signature(
    X=X,
    config=config.as_dict(),
    seed=42,
)
```

This uniquely fingerprints the experiment.

---

## Intended Audience

CTFC is intended for:

* researchers in applied mathematics,
* quantitative finance and econometrics,
* complex systems analysis,
* signal processing with memory,
* scientific ML practitioners who require interpretability.

If you are looking for a plug-and-play predictor, CTFC is not the right tool.
If you want **structural understanding**, CTFC is.

---

## Status of the Framework

CTFC is:

* mathematically specified,
* fully implemented,
* safeguarded and validated,
* reproducible by design.

It is suitable for:

* research use,
* methodological papers,
* grant submissions,
* long-term extension.

---

## Final Remark

CTFC treats **time as geometry with memory**.

If that idea resonates with your problem, you are in the right place.



