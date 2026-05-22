# Emergent Entanglement and SU(2) Bound on Tripartite Correlations

**Central result:** The exact GHZ state (τ₃ = 1) is impossible for any
SU(2)-invariant Hamiltonian, because no rank-3 symmetric invariant tensor
d_ijk exists in SU(2). The Holst-Immirzi action reaches τ₃ ~ 0.96–0.98
with optimal initial states but never τ₃ = 1. This gap is algebraically exact.

> Preprint: *Emergent Entanglement and an SU(2) Symmetry Bound on Tripartite
> Correlations in Minimal Spin Networks* — Schroyle AD (2026)

---

## What this repository contains

Numerical verification of a relational toy framework in which:

- Quantum entanglement emerges from interaction with a relational clock sector **C**
- **C** is identified with edges of an LQG spin network (j = 1/2)
- The Holst-Immirzi action discretized on the minimal graph generates
  `H_int = H_EH + (g/γ) ε_ijk σ_A^i σ_B^j σ_C^k`
- The Tsirelson bound B = 2√2 emerges as the exceptional point of a
  PT-symmetric Hamiltonian when `g·τ = ħ/2`
- Dark states are classified by a three-condition criterion (C1 ∧ C2 ∧ C3)
  verified at 100% accuracy over 18 test Hamiltonians

---

## Quick start

```bash
git clone https://github.com/<your-username>/emergent-entanglement-su2
cd emergent-entanglement-su2
pip install -r requirements.txt

# Run the main result (Holst action + SSB sweep)
python experiments/02_holst_ssb.py

# Dark state criterion verification
python experiments/01_dark_states.py

# C self-interaction experiment
python experiments/03_C_self_interaction.py
```

Each script prints numerical results to stdout and saves a figure to `figures/`.

---

## Repository structure

```
emergent-entanglement-su2/
├── README.md
├── requirements.txt
├── core/
│   └── operators.py        # All reusable functions — import from here
├── experiments/
│   ├── 01_dark_states.py   # Dark state criterion, 18 Hamiltonians, 100% accuracy
│   ├── 02_holst_ssb.py     # Main result: Holst action, SSB sweep, gamma sweep
│   └── 03_C_self_interaction.py  # A-C1-C2: C has independent internal dynamics
├── figures/                # Generated PNG figures (git-ignored, regenerate with scripts)
└── paper/
    └── preprint_v7.docx    # Full preprint
```

---

## Key results at a glance

| Hamiltonian | Symmetry | τ₃ range | GHZ exact |
|---|---|---|---|
| H_EH (bilinear) | SU(2) | 0.00 – 0.25 * | Impossible |
| H_Holst (full) | SU(2) | 0.16 – 0.98 * | Impossible |
| H_Holst + SSB | U(1) | 0.80 – 0.95 * | Impossible |
| Any SU(2)-invariant H | SU(2) | < 1.000 | **Algebraically impossible** |

\* state-dependent — range over initial states tested

The value τ₃ ≈ 0.165 appears specifically with initial state |00⟩|+⟩ and
γ = 0.2375 (Immirzi). It is a reproducible numerical signature of the
Holst action with that initial condition, not a universal algebraic bound.

---

## C self-interaction (A-C1-C2)

When C interacts with itself (system A × C1 × C2):

1. **Only CC active:** S_A = 0 exactly for all t.
   C has fully independent internal dynamics — it self-entangles without
   affecting observable systems A. This is consistent with C as a vacuum
   degree of freedom.

2. **Holst A-C1 + self-interaction C1-C2:**
   I(A:C1)_max = 0.941 — higher than A-C-B = 0.633.
   C self-interaction *amplifies* entanglement transfer to A.

3. **A-C-C does NOT collapse to A-C.** It enriches the structure.

---

## Core module

```python
from core.operators import (
    make_H_Holst, make_H_SSB, make_H_clock_qubit,
    state_std, state_opt,
    evolve, evolve_max,
    tau3, concurrence, von_neumann, mutual_information,
    check_dark_state_criteria,
)

# Reproduce the main result in 10 lines:
H     = make_H_Holst(g=0.5, gamma=0.2375)
clock = make_H_clock_qubit()
psi0  = state_opt()                         # |++>|theta=1.35, phi=2.69>

times, res = evolve(H, clock, psi0, N=400, T=12*3.14159)

print(f"tau3_max  = {res['tau3'].max():.4f}")   # ~ 0.96
print(f"F_GHZ_max = {res['fghz'].max():.4f}")   # < 1.00 (never exact)
```

---

## Falsifiable prediction

Prepare three qubits with a qubit clock, implement the Holst Hamiltonian
(ε_ijk coupling), measure τ₃ via state tomography:

- **τ₃ < 1.000** (approaching but never reaching): consistent with
  SU(2)-symmetric vacuum.
- **τ₃ = 1.000 exactly**: would require SSB (SU(2)→U(1)) or SU(3) extension.

The experimentally meaningful question is: *can τ₃ = 1 be reached?*
Our algebraic result says NO with SU(2).

---

## Limitations

This is an explicit toy framework:

- No Lorentz covariance
- No field-theoretic derivation of H_int from first principles
- Weak-field approximation in the Holst discretization
- Finite-dimensional Hilbert spaces (N_Fock = 15–30)
- Approximate canonical relation [T_C, H_C] = iħ (exact only in coherent states)

No claims of a complete theory of quantum gravity are made.

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
```

Python 3.9+ recommended.

---

## Citation

If you use this code, please cite:

```
Schroyle AD (2026). Emergent Entanglement and an SU(2) Symmetry Bound
on Tripartite Correlations in Minimal Spin Networks. Preprint v7.0.
```

---

## License

MIT License. See LICENSE file.
