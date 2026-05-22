"""
experiments/02_holst_ssb.py
============================
Main result: SU(2) symmetry structure and tripartite entanglement.

Reproduces Figure 3 from the preprint:
- tau3(t) for EH, Holst, Holst+SSB
- tau3_max vs Immirzi parameter gamma (gamma-independent bound)
- tau3_max vs SSB angle theta (monotonic increase)
- Complete action -> entanglement map

Key result:
    tau3 = 1 (exact GHZ) is IMPOSSIBLE with any SU(2)-invariant H.
    The Holst action reaches tau3 ~ 0.96-0.98 with optimal initial states.
    This gap is algebraically exact: no d_ijk symmetric tensor in SU(2).

Run:
    python experiments/02_holst_ssb.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.operators import (
    sx, sy, sz, I2, kp3,
    make_H_EH, make_H_anti, make_H_sym, make_H_Holst,
    make_H_SSB, make_H_clock_qubit,
    state_std, state_opt, GHZ,
    evolve, evolve_max, tau3, reduced_states
)

# ── Parameters ────────────────────────────────────────────────────
G     = 0.5
GAMMA = 0.2375
N_STEPS = 400
T_TOTAL = 12 * np.pi

H_EH     = make_H_EH(G)
H_anti   = make_H_anti()
H_sym    = make_H_sym()
H_Holst  = make_H_Holst(G, GAMMA)
H_SSB    = H_Holst + G * H_sym
H_clock  = make_H_clock_qubit()

psi_std = state_std()
psi_opt = state_opt()

# ── 1. Time evolution for main cases ─────────────────────────────
print("Computing time evolutions...")
cases = {
    "EH bilineal  |00>|+>":  (H_EH,    psi_std, '#4A9EE8', '--'),
    "Holst  |00>|+>":        (H_Holst, psi_std, '#E07B39', '--'),
    "Holst  |opt>":          (H_Holst, psi_opt, '#2ECC71', '-'),
    "Holst+SSB  |opt>":      (H_SSB,   psi_opt, '#9B59B6', '-'),
}

evolutions = {}
for label, (H, psi0, col, ls) in cases.items():
    times, res = evolve(H, H_clock, psi0, N=N_STEPS, T=T_TOTAL)
    evolutions[label] = {'times': times, 'res': res, 'col': col, 'ls': ls}
    print(f"  {label:35s}: tau3_max={res['tau3'].max():.4f}  "
          f"F_GHZ_max={res['fghz'].max():.4f}")

# ── 2. Gamma sweep (optimal state) ────────────────────────────────
print("\nGamma sweep...")
gammas = np.array([0.05, 0.1, 0.15, 0.2, 0.2375, 0.3, 0.5, 1.0, 2.0, 5.0])
tau3_gam = []
for gam in gammas:
    H_g = make_H_EH(G) + (G / gam) * make_H_anti()
    m   = evolve_max(H_g, H_clock, psi_opt, N=300, T=T_TOTAL)
    tau3_gam.append(m['tau3'])
    print(f"  gamma={gam:.4f}: tau3={tau3_gam[-1]:.4f}")

# ── 3. SSB theta sweep ────────────────────────────────────────────
print("\nSSB theta sweep...")
thetas = np.linspace(0, np.pi/2, 14)
tau3_ssb = []; fghz_ssb = []
for theta in thetas:
    n_vec  = [np.sin(theta), 0, np.cos(theta)]
    H_ssb  = H_Holst + make_H_SSB(n_vec, G)
    m      = evolve_max(H_ssb, H_clock, psi_std, N=300, T=T_TOTAL)
    tau3_ssb.append(m['tau3'])
    fghz_ssb.append(m['fghz'])
    print(f"  theta={theta:.3f}: tau3={tau3_ssb[-1]:.4f}")

# ── Plot ──────────────────────────────────────────────────────────
C_BG = '#0F1923'; C_PANEL = '#1A2535'; C_GRID = '#2A3A50'; C_TEXT = '#E0E8F0'

fig = plt.figure(figsize=(18, 13), facecolor=C_BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.36,
                        left=0.07, right=0.97, top=0.91, bottom=0.07)

def panel(ax, title, xl='', yl=''):
    ax.set_facecolor(C_PANEL); ax.tick_params(colors=C_TEXT, labelsize=8.5)
    ax.set_title(title, color=C_TEXT, fontsize=9.5, fontweight='bold', pad=7)
    for sp in ax.spines.values(): sp.set_edgecolor(C_GRID)
    ax.grid(True, color=C_GRID, ls='--', alpha=0.4, lw=0.6)
    if xl: ax.set_xlabel(xl, color=C_TEXT, fontsize=9)
    if yl: ax.set_ylabel(yl, color=C_TEXT, fontsize=9)

# A: tau3(t)
ax = fig.add_subplot(gs[0, :2])
panel(ax, 'A. tau3(t) — SU(2) cannot reach tau3=1 exactly\n'
          '(algebraic result: no d_ijk in SU(2))', 't [1/g]', 'tau3')
for label, d in evolutions.items():
    ax.plot(d['times'], d['res']['tau3'], color=d['col'], lw=2.2,
            ls=d['ls'], label=f"{label}  [max={d['res']['tau3'].max():.3f}]")
ax.axhline(1.0, color='white', lw=1.2, ls=':', alpha=0.4,
           label='GHZ exact = 1  (IMPOSSIBLE with SU(2))')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=7.5, facecolor=C_PANEL, labelcolor=C_TEXT,
          edgecolor=C_GRID, ncol=2)

# B: F_GHZ(t)
ax = fig.add_subplot(gs[0, 2])
panel(ax, 'B. GHZ fidelity F(t)', 't [1/g]', 'F_GHZ')
for label, d in evolutions.items():
    ax.plot(d['times'], d['res']['fghz'], color=d['col'],
            lw=2, ls=d['ls'])
ax.axhline(1.0, color='white', lw=0.8, ls=':', alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# C: tau3_max vs gamma
ax = fig.add_subplot(gs[1, 0])
panel(ax, 'C. tau3_max vs gamma\n(optimal state — SU(2) intact)',
      'gamma', 'tau3_max')
ax.plot(gammas, tau3_gam, 'o-', color='#2ECC71', lw=2.5, ms=9)
ax.axvline(GAMMA, color='#F39C12', lw=1.5, ls='--', alpha=0.8,
           label=f'Immirzi={GAMMA}')
idx = np.argmin(np.abs(gammas - GAMMA))
ax.annotate(f'gamma={GAMMA}\ntau3={tau3_gam[idx]:.3f}',
            (gammas[idx], tau3_gam[idx]),
            textcoords='offset points', xytext=(10, 8),
            color='#F39C12', fontsize=8.5)
ax.axhline(1.0, color='white', lw=0.8, ls=':', alpha=0.3,
           label='GHZ exact (unreachable)')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor=C_GRID)

# D: tau3_max vs theta SSB
ax = fig.add_subplot(gs[1, 1])
panel(ax, 'D. tau3_max vs SSB angle theta\n'
          'Holst + n(theta) term  |  std initial state',
      'theta (rad)', 'tau3_max')
ax.plot(thetas, tau3_ssb, 's-', color='#9B59B6', lw=2.5, ms=8,
        label='tau3_max')
ax.plot(thetas, fghz_ssb, '^--', color='#E07B39', lw=2, ms=7,
        label='F_GHZ_max')
ax.axhline(0.165, color='#E74C3C', lw=1, ls='--', alpha=0.7,
           label='tau3 with |00>|+> (state-dependent)')
ax.set_ylim(-0.02, 0.85)
ax.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor=C_GRID)

# E: Summary table
ax = fig.add_subplot(gs[1, 2])
panel(ax, 'E. True limits — Corrected result')
ax.axis('off')
txt = (
    "ALGEBRAIC RESULT (SU(2)):\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "No d_ijk symmetric rank-3\n"
    "invariant tensor in SU(2).\n"
    "=> tau3 = 1 IMPOSSIBLE.\n\n"
    "NUMERICAL BOUNDS:\n"
    "H_EH:     tau3 in [0, 0.25]\n"
    "H_Holst:  tau3 in [0.165,\n"
    "           0.98]  (state dep)\n"
    "H_SSB:    tau3 -> ~0.95\n"
    "GHZ exact (tau3=1): NEVER\n\n"
    "The 0.165 value appears with\n"
    "|00>|+> and gamma=0.2375.\n"
    "NOT a universal bound."
)
ax.text(0.05, 0.97, txt, transform=ax.transAxes,
        color=C_TEXT, fontsize=8.5, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0A1020',
                  edgecolor='#2ECC71', alpha=0.95))

# F: Complete map
ax = fig.add_subplot(gs[2, :])
panel(ax, 'F. Complete Map: Gravitational Action -> Entanglement Structure')
ax.axis('off')
rows = [
    ('#4A9EE8', 'EH bilinear',    'SU(2)', '0.00-0.25 *', 'Bell/W, no GHZ'),
    ('#E07B39', 'Holst complete', 'SU(2)', '0.16-0.98 *', 'Near-GHZ, not exact'),
    ('#9B59B6', 'Holst + SSB',    'U(1)',  '0.80-0.95 *', 'GHZ high'),
    ('#F39C12', 'Any SU(2) H',    'SU(2)', '< 1.000 (exact)', 'GHZ=1 IMPOSSIBLE'),
    ('#E74C3C', 'SU(3) with d_abc','SU(3)','-> 1.000',    'GHZ exact possible'),
]
xs = [0.01, 0.20, 0.36, 0.52, 0.68]
headers = ['Hamiltonian', 'Symmetry', 'tau3 range', 'Note']
for ci, hdr in enumerate(headers):
    ax.text(xs[ci]+0.08, 0.95, hdr, transform=ax.transAxes,
            ha='center', va='top', color='#AABBCC',
            fontsize=9, fontweight='bold')
for ri, row in enumerate(rows):
    col = row[0]; vals = row[1:]
    y = 0.80 - ri * 0.155
    rect = plt.Rectangle((0.005, y-0.08), 0.99, 0.135,
                          facecolor=col+'20', edgecolor=col+'70', lw=0.8,
                          transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)
    for ci, val in enumerate(vals):
        ax.text(xs[ci]+0.08, y, val, transform=ax.transAxes,
                ha='center', va='center', color=C_TEXT, fontsize=8.5)

fig.text(0.5, 0.958,
         'SU(2) Symmetry Bound on Tripartite Entanglement: '
         'Holst Action, SSB, and the GHZ Impossibility Theorem',
         ha='center', color=C_TEXT, fontsize=13, fontweight='bold')
fig.text(0.5, 0.937,
         f'g={G}  |  gamma={GAMMA} (Immirzi)  |  '
         f'Algebraic result: tau3=1 impossible with SU(2)',
         ha='center', color='#8899AA', fontsize=9)

out = os.path.join(os.path.dirname(__file__), '..', 'figures',
                   '02_holst_ssb_main.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=C_BG)
print(f'\nFigure saved: {out}')
