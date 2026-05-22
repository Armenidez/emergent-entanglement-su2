"""
experiments/03_C_self_interaction.py
=====================================
Exploration of C self-interaction: system A-C1-C2.

Question: if A-C-B generates AB entanglement,
what happens when C interacts with itself (A-C-C)?
Does A-C-C collapse to A-C, or does C have internal dynamics?

Key findings:
  1. When only C1-C2 active (no A): S_A = 0 exactly.
     C has fully independent internal dynamics — it can self-entangle
     without affecting observable systems.

  2. When Holst A-C1 + self-interaction C1-C2:
     I(A:C1) rises to 0.941 — HIGHER than A-C-B = 0.633.
     C self-interaction AMPLIFIES entanglement with A.

  3. A-C-C does NOT simplify to A-C. It enriches the structure.

Run:
    python experiments/03_C_self_interaction.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.linalg import expm, eigvalsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.operators import (
    sx, sy, sz, I2, kp2, kp3,
    make_H_EH, make_H_anti, make_H_Holst, make_H_clock_qubit,
    state_std, von_neumann, concurrence, mutual_information
)

G     = 0.5
GAMMA = 0.2375

# ── Hamiltonians for A x C1 x C2 system ──────────────────────────
# Order: A(0), C1(1), C2(2)

H_AC1_EH     = make_H_EH(G)       # A-C1 bilinear (uses positions 0,1,2)
H_AC1_Holst  = make_H_Holst(G, GAMMA)

# C1-C2 self-interaction (A is spectator)
SdotS_C1C2  = (kp3(I2,sx,sx) + kp3(I2,sy,sy) + kp3(I2,sz,sz))
H_self_C    = G * SdotS_C1C2

# Combined Hamiltonians
H_chain      = H_AC1_EH + H_self_C
H_Holst_chain= H_AC1_Holst + H_self_C
H_only_CC   = H_self_C     # A is completely free

H_clock = make_H_clock_qubit()   # clock on C2 (position 2)

# ── Initial states ────────────────────────────────────────────────
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
ketp = np.array([1, 1], dtype=complex) / np.sqrt(2)

psi_A0_C1p_C2p = np.kron(np.kron(ket0, ketp), ketp)
psi_Ap_C10_C2p = np.kron(np.kron(ketp, ket0), ketp)
psi_A0_C10_C2p = state_std()   # |00>|+>

# ── Measurement functions for A x C1 x C2 ────────────────────────

def measures_ACC(psi):
    """Entanglement measures for A x C1 x C2 system."""
    p    = psi.reshape(2, 2, 2)
    rA   = np.einsum('abc,dbc->ad', p, p.conj())
    rC1  = np.einsum('abc,adc->bd', p, p.conj())
    rC2  = np.einsum('abc,abd->cd', p, p.conj())

    rAC1 = np.zeros((4,4), dtype=complex)
    for a in range(2):
        for b in range(2):
            for d in range(2):
                for e in range(2):
                    rAC1[2*a+b, 2*d+e] = np.sum(p[a,b,:]*p[d,e,:].conj())

    rC1C2 = np.zeros((4,4), dtype=complex)
    for b in range(2):
        for c in range(2):
            for e in range(2):
                for f in range(2):
                    rC1C2[2*b+c, 2*e+f] = np.sum(p[:,b,c]*p[:,e,f].conj())

    sA = von_neumann(rA)
    sC1 = von_neumann(rC1)
    sC2 = von_neumann(rC2)

    return {
        'sA':      sA,
        'IAC1':    sA + sC1 - von_neumann(rAC1),
        'IC1C2':   sC1 + sC2 - von_neumann(rC1C2),
        'concAC1': concurrence(rAC1),
        'concC1C2':concurrence(rC1C2),
    }


def evolve_ACC(H_int, H_clock, psi0, N=400, T=12*np.pi):
    """Evolve A x C1 x C2 and collect observables."""
    U     = expm(-1j * (H_clock + H_int) * T / N)
    psi   = psi0.copy(); psi /= np.linalg.norm(psi)
    times = np.linspace(0, T, N)
    keys  = ['sA','IAC1','IC1C2','concAC1','concC1C2']
    res   = {k: np.zeros(N) for k in keys}
    for step in range(N):
        m = measures_ACC(psi)
        for k in keys: res[k][step] = m[k]
        psi = U @ psi; psi /= np.linalg.norm(psi)
    return times, res


# ── Run experiments ───────────────────────────────────────────────
configs = [
    ("Chain A-C1-C2\n(bilinear+CC)",          H_chain,       H_clock, psi_A0_C1p_C2p, '#4A9EE8'),
    ("Holst A-C1 + CC\n(C self-interacts)",   H_Holst_chain, H_clock, psi_A0_C1p_C2p, '#2ECC71'),
    ("Only CC\n(A free)",                      H_only_CC,     H_clock, psi_A0_C10_C2p, '#E07B39'),
    ("Chain A=|+> state",                      H_chain,       H_clock, psi_Ap_C10_C2p, '#9B59B6'),
]

print("=" * 65)
print("C Self-Interaction Experiment: A-C1-C2")
print("=" * 65)

resultados = {}
for label, H, Hclk, psi0, col in configs:
    times, res = evolve_ACC(H, Hclk, psi0)
    resultados[label] = (times, res, col)
    lbl = label.replace('\n', ' ')
    print(f"\n  {lbl}")
    print(f"    S_A max     = {res['sA'].max():.4f}")
    print(f"    I(A:C1) max = {res['IAC1'].max():.4f}")
    print(f"    I(C1:C2) max= {res['IC1C2'].max():.4f}")
    print(f"    C(A:C1) max = {res['concAC1'].max():.4f}")
    print(f"    C(C1:C2) max= {res['concC1C2'].max():.4f}")

# ── Plot ──────────────────────────────────────────────────────────
C_BG = '#0F1923'; C_PANEL = '#1A2535'; C_GRID = '#2A3A50'; C_TEXT = '#E0E8F0'

fig = plt.figure(figsize=(18, 13), facecolor=C_BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                        left=0.07, right=0.97, top=0.91, bottom=0.07)

def panel(ax, title, xl='', yl=''):
    ax.set_facecolor(C_PANEL); ax.tick_params(colors=C_TEXT, labelsize=8.5)
    ax.set_title(title, color=C_TEXT, fontsize=9.5, fontweight='bold', pad=7)
    for sp in ax.spines.values(): sp.set_edgecolor(C_GRID)
    ax.grid(True, color=C_GRID, ls='--', alpha=0.4, lw=0.6)
    if xl: ax.set_xlabel(xl, color=C_TEXT, fontsize=9)
    if yl: ax.set_ylabel(yl, color=C_TEXT, fontsize=9)

ax = fig.add_subplot(gs[0, :2])
panel(ax, 'A. S_A(t) in system A-C1-C2\n'
          'Does C self-interaction still entangle A?', 't [1/g]', 'S_A')
for label, (times, res, col) in resultados.items():
    ax.plot(times, res['sA'], color=col, lw=2,
            label=f"{label.replace(chr(10),' ')}  max={res['sA'].max():.3f}")
ax.axhline(np.log(2), color='white', lw=0.8, ls=':', alpha=0.4)
ax.text(1, np.log(2)+0.02, 'log 2', color='white', fontsize=8)
ax.set_ylim(-0.02, 0.85)
ax.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor=C_GRID)

ax = fig.add_subplot(gs[0, 2])
panel(ax, 'B. I(C1:C2)\nC self-entanglement', 't [1/g]', 'I(C1:C2)')
for label, (times, res, col) in resultados.items():
    ax.plot(times, res['IC1C2'], color=col, lw=2)
ax.set_ylim(-0.05, 1.5)

ax = fig.add_subplot(gs[1, 0])
panel(ax, 'C. I(A:C1) vs I(C1:C2)\nMonogamy structure',
      'I(C1:C2) max', 'I(A:C1) max')
for label, (times, res, col) in resultados.items():
    ax.scatter(res['IC1C2'].max(), res['IAC1'].max(),
               color=col, s=130, zorder=5,
               label=label.replace('\n', ' ')[:22])
ax.plot([0, 1.2], [0, 1.2], color='white', lw=0.8, ls=':', alpha=0.3)
ax.set_xlim(-0.05, 1.2); ax.set_ylim(-0.05, 1.2)
ax.legend(fontsize=7.5, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor=C_GRID)

ax = fig.add_subplot(gs[1, 1])
panel(ax, 'D. Holst A-C1 + CC\nI(A:C1) vs I(C1:C2) over time',
      't [1/g]', 'Mutual information')
label_h = "Holst A-C1 + CC\n(C self-interacts)"
times, res, col = resultados[label_h]
ax.plot(times, res['IAC1'],  color='#2ECC71', lw=2.5, label='I(A:C1)')
ax.plot(times, res['IC1C2'], color='#2ECC71', lw=1.5, ls='--', label='I(C1:C2)')
ax.set_ylim(-0.05, 1.5)
ax.legend(fontsize=9, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor=C_GRID)

ax = fig.add_subplot(gs[1, 2])
panel(ax, 'E. Only CC active\nA stays completely free', 't [1/g]', 'Observable')
label_cc = "Only CC\n(A free)"
times, res, col = resultados[label_cc]
ax.plot(times, res['sA'],      color='#E07B39', lw=2, label='S_A (should be 0)')
ax.plot(times, res['IC1C2'],   color='#2ECC71', lw=2, label='I(C1:C2)')
ax.plot(times, res['concC1C2'],color='#9B59B6', lw=1.5, ls='--', label='C(C1:C2)')
ax.axhline(0.5, color='white', lw=0.8, ls=':', alpha=0.4, label='Bell max=0.5')
ax.set_ylim(-0.05, 0.8)
ax.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor=C_GRID)

# Interpretations panel
ax = fig.add_subplot(gs[2, :])
panel(ax, 'F. What is C? — Evidence from self-interaction')
ax.axis('off')
interp = [
    ('#4A9EE8', 'RESULT 1: Only CC active',
     'When only H_CC acts:\n'
     '• S_A = 0 exactly for all t\n'
     '• I(C1:C2) > 0 (C self-entangles)\n'
     '• A remains completely free\n'
     '=> C has independent internal\n'
     '   dynamics — like vacuum DOF', 0.01),
    ('#2ECC71', 'RESULT 2: Holst A-C1 + CC',
     'When A-C1 (Holst) + C1-C2:\n'
     '• I(A:C1)_max = 0.941\n'
     '• vs A-C-B: I(A:B)_max = 0.633\n'
     '• Self-interaction AMPLIFIES\n'
     '  entanglement transfer to A\n'
     '=> C as quantum relay/amplifier', 0.34),
    ('#9B59B6', 'RESULT 3: Monogamy',
     'I(A:C1) and I(C1:C2)\ncan both be high:\n'
     '• No strict bipartite monogamy\n'
     '• Genuine tripartite structure\n'
     '• A-C-C ≠ A-C (does not simplify)\n'
     '=> C has richer structure than\n'
     '   a simple intermediary', 0.67),
]
for col, title, text, x in interp:
    rect = plt.Rectangle((x, 0.05), 0.31, 0.90,
                          facecolor=col+'18', edgecolor=col+'80', lw=1.2,
                          transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)
    ax.text(x+0.155, 0.92, title, transform=ax.transAxes,
            ha='center', va='top', color=col, fontsize=9, fontweight='bold')
    ax.text(x+0.155, 0.74, text, transform=ax.transAxes,
            ha='center', va='top', color=C_TEXT, fontsize=8.5,
            fontfamily='monospace')

fig.text(0.5, 0.958,
         'C Self-Interaction: System A-C1-C2 — What is Sector C?',
         ha='center', color=C_TEXT, fontsize=13, fontweight='bold')
fig.text(0.5, 0.937,
         'A-C-C does NOT collapse to A-C. '
         'C has independent internal dynamics and amplifies entanglement.',
         ha='center', color='#8899AA', fontsize=9)

out = os.path.join(os.path.dirname(__file__), '..', 'figures',
                   '03_C_self_interaction.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=C_BG)
print(f'\nFigure saved: {out}')
