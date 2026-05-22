"""
experiments/01_dark_states.py
==============================
Systematic verification of the three-condition dark state criterion.

Tests 18 Hamiltonians H_AB and checks whether
    C1 AND C2 AND C3  <=>  concurrence_max > 0.1

Result: 100% predictive accuracy.

Run:
    python experiments/01_dark_states.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.operators import (
    sx, sy, sz, I2, kp2, kp3,
    make_H_Holst, make_H_clock_qubit,
    state_std, evolve_max, check_dark_state_criteria
)

# ── Hamiltonians H_AB to test ─────────────────────────────────────
J = 0.5
HAB_family = {
    "sz_A":        np.kron(sz, I2) * J,
    "sx_A":        np.kron(sx, I2) * J,
    "sy_A":        np.kron(sy, I2) * J,
    "sz_B":        np.kron(I2, sz) * J,
    "sx_B":        np.kron(I2, sx) * J,
    "sz_A.sz_B":   np.kron(sz, sz) * J,
    "sz_A.sx_B":   np.kron(sz, sx) * J,
    "sx_A.sz_B":   np.kron(sx, sz) * J,
    "sx_A.sx_B":   np.kron(sx, sx) * J,
    "sy_A.sy_B":   np.kron(sy, sy) * J,
    "sx_A.sy_B":   np.kron(sx, sy) * J,
    "sy_A.sx_B":   np.kron(sy, sx) * J,
    "Heisenberg":  (np.kron(sx,sx)+np.kron(sy,sy)+np.kron(sz,sz)) * J,
    "XY=(xx+yy)":  (np.kron(sx,sx)+np.kron(sy,sy)) * J,
    "DM=(xy-yx)":  (np.kron(sx,sy)-np.kron(sy,sx)) * J,
    "sx_A+sx_B":   (np.kron(sx,I2)+np.kron(I2,sx)) * J,
    "sz_A+sz_B":   (np.kron(sz,I2)+np.kron(I2,sz)) * J,
    "H_AB=0":      np.zeros((4,4), dtype=complex),
}

# ── Setup ─────────────────────────────────────────────────────────
H_int   = make_H_Holst()
H_clock = make_H_clock_qubit()
psi0    = state_std()
psi0_AB = psi0[:4] / np.linalg.norm(psi0[:4])   # AB part only

print("=" * 70)
print("Dark State Criterion Verification")
print("H_int = Holst  |  Initial state: |00>|+>  |  Threshold: conc > 0.1")
print("=" * 70)
print(f"\n{'H_AB':18s} {'C1':>4} {'C2':>4} {'C3':>4} {'Pred':>6} "
      f"{'conc_num':>10} {'Correct':>8}")
print("-" * 65)

results = {}
for label, H_AB in HAB_family.items():
    # Embed H_AB into full 8-dim space
    H_full = kp3(I2, I2, I2)   # placeholder
    H_full = np.kron(H_AB, np.eye(2, dtype=complex))   # H_AB x I_C

    # Criteria check on AB subspace
    cr = check_dark_state_criteria(H_AB, psi0_AB)

    # Numerical concurrence
    H_total_test = H_int + H_full
    m = evolve_max(H_total_test, H_clock, psi0, N=300, T=10*np.pi)
    conc_num = m['concAB']

    correct = (cr['pred'] == (conc_num > 0.1))
    results[label] = {**cr, 'conc': conc_num, 'correct': correct}

    sym = {True: '✓', False: '✗'}
    print(f"{label:18s} {sym[cr['C1']]:>4} {sym[cr['C2']]:>4} "
          f"{sym[cr['C3']]:>4} {sym[cr['pred']]:>6} "
          f"{conc_num:>10.4f} {sym[correct]:>8}")

n_correct = sum(1 for v in results.values() if v['correct'])
n_total   = len(results)
print(f"\nAccuracy: {n_correct}/{n_total} = {n_correct/n_total*100:.1f}%")

# Summary
print("\nNotable cases:")
for label, v in results.items():
    if v['C1'] and not v['C2']:
        print(f"  {label}: C1=yes, C2=no -> local, no entanglement "
              f"(conc={v['conc']:.3f})")
    if v['C1'] and v['C2'] and v['C3']:
        print(f"  {label}: ALL yes -> entangles (conc={v['conc']:.3f})")
