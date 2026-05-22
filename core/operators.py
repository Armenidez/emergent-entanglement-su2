"""
core/operators.py
=================
Core operators, states, and entanglement measures for the
Emergent Entanglement / SU(2) Bound project.

All functions are pure (no side effects, no plotting).
Import this module from any experiment script.

Authors: Schroyle AD (2026)
"""

import numpy as np
from scipy.linalg import expm, eigvalsh

# ── Pauli matrices ────────────────────────────────────────────────
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

PAULIS = {'x': sx, 'y': sy, 'z': sz}
LEVI_CIVITA = {'xyz': 1, 'yzx': 1, 'zxy': 1,
               'xzy': -1, 'zyx': -1, 'yxz': -1}

# ── Tensor products ───────────────────────────────────────────────

def kp2(A, B):
    return np.kron(A, B)

def kp3(A, B, C):
    return np.kron(np.kron(A, B), C)

def kp4(A, B, C, D):
    return np.kron(np.kron(np.kron(A, B), C), D)


# ── Standard Hamiltonians ─────────────────────────────────────────

def make_H_EH(g=0.5):
    """
    Einstein-Hilbert bilinear: H = g (sigma_A . sigma_B) x sigma_z^C
    Discretization of the Thiemann Hamiltonian on the minimal graph.
    System order: A, B, C  (dim = 8)
    """
    SdotS = kp3(sx,sx,I2) + kp3(sy,sy,I2) + kp3(sz,sz,I2)
    return g * SdotS @ kp3(I2, I2, sz)


def make_H_anti():
    """
    Antisymmetric trilinear: H = epsilon_ijk sigma_A^i sigma_B^j sigma_C^k
    Arises from the Holst-Pontryagin term in the gravitational action.
    System order: A, B, C  (dim = 8)
    """
    return sum(
        LEVI_CIVITA[nm] * kp3(*[PAULIS[c] for c in nm])
        for nm in LEVI_CIVITA
    )


def make_H_sym():
    """
    Symmetric trilinear: H = sigma_x^3 + sigma_y^3 + sigma_z^3
    NOT SU(2)-invariant (no d_ijk in SU(2)).
    Requires spontaneous symmetry breaking to arise physically.
    System order: A, B, C  (dim = 8)
    """
    return kp3(sx,sx,sx) + kp3(sy,sy,sy) + kp3(sz,sz,sz)


def make_H_Holst(g=0.5, gamma=0.2375):
    """
    Full Holst-Immirzi Hamiltonian:
        H = H_EH + (g/gamma) * H_anti
    Parameters:
        g     : coupling constant (default 0.5)
        gamma : Immirzi parameter (default 0.2375, from BH entropy)
    System order: A, B, C  (dim = 8)
    """
    return make_H_EH(g) + (g / gamma) * make_H_anti()


def make_H_SSB(n, g=0.5):
    """
    Spontaneous symmetry breaking term:
        H_SSB = g (n.sigma_A)(n.sigma_B)(n.sigma_C)
    where n = (nx, ny, nz) is the vacuum direction.
    SU(2) -> U(1) breaking; theta = pi/2 gives maximum effect.
    System order: A, B, C  (dim = 8)
    """
    n = np.array(n, dtype=float)
    n = n / np.linalg.norm(n)
    sA_n = sum(n[i] * kp3([sx,sy,sz][i], I2,  I2)  for i in range(3))
    sB_n = sum(n[i] * kp3(I2, [sx,sy,sz][i],  I2)  for i in range(3))
    sC_n = sum(n[i] * kp3(I2,  I2, [sx,sy,sz][i])  for i in range(3))
    return g * sA_n @ sB_n @ sC_n


def make_H_clock_qubit():
    """
    Minimal qubit clock Hamiltonian for sector C:
        H_C = (hbar/2) sigma_x^C
    Acts on the full 8-dim space (A, B, C).
    """
    return 0.5 * kp3(I2, I2, sx)


# ── Initial states ────────────────────────────────────────────────

def state_std():
    """Standard initial state: |00>_AB x |+>_C"""
    ket0 = np.array([1, 0], dtype=complex)
    ketp = np.array([1, 1], dtype=complex) / np.sqrt(2)
    return np.kron(np.kron(ket0, ket0), ketp)


def state_opt(theta=1.35, phi=2.69):
    """
    Optimal initial state for maximum tau3 with Holst:
        |++>_AB x |theta, phi>_C
    """
    ketp = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ketC = np.array([np.cos(theta/2),
                     np.exp(1j*phi)*np.sin(theta/2)], dtype=complex)
    return np.kron(np.kron(ketp, ketp), ketC)


def state_product(ab_state, c_theta=0.0, c_phi=0.0):
    """
    General product state: |ab_state>_AB x |theta, phi>_C
    ab_state: array of shape (4,) — two-qubit state
    """
    ketC = np.array([np.cos(c_theta/2),
                     np.exp(1j*c_phi)*np.sin(c_theta/2)], dtype=complex)
    return np.kron(ab_state, ketC)


# ── Reference states ──────────────────────────────────────────────

GHZ = np.array([1,0,0,0,0,0,0,1], dtype=complex) / np.sqrt(2)
W   = np.array([0,1,1,0,1,0,0,0], dtype=complex) / np.sqrt(3)


# ── Reduced density matrices ──────────────────────────────────────

def reduced_states(psi):
    """
    Compute all reduced density matrices for a 3-qubit pure state.
    System order: A(0), B(1), C(2)

    Returns dict with keys:
        rA, rB, rC        : single-qubit (2x2)
        rAB, rAC, rBC     : two-qubit   (4x4)
    """
    p = psi.reshape(2, 2, 2)

    rA  = np.einsum('abc,dbc->ad', p, p.conj())
    rB  = np.einsum('abc,adc->bd', p, p.conj())
    rC  = np.einsum('abc,abd->cd', p, p.conj())

    def two_body(idx1, idx2):
        """Build 4x4 reduced density matrix for qubits idx1, idx2."""
        rho = np.zeros((4, 4), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        bra = [slice(None), slice(None), slice(None)]
                        ket = [slice(None), slice(None), slice(None)]
                        bra[idx1] = i; bra[idx2] = j
                        ket[idx1] = k; ket[idx2] = l
                        rho[2*i+j, 2*k+l] = np.sum(
                            p[tuple(bra)] * p[tuple(ket)].conj()
                        )
        return rho

    rAB = two_body(0, 1)
    rAC = two_body(0, 2)
    rBC = two_body(1, 2)

    return {'rA': rA, 'rB': rB, 'rC': rC,
            'rAB': rAB, 'rAC': rAC, 'rBC': rBC}


# ── Entanglement measures ─────────────────────────────────────────

def von_neumann(rho, tol=1e-12):
    """Von Neumann entropy S = -Tr[rho log rho]."""
    ev = np.maximum(np.real(eigvalsh(rho)), tol)
    ev = ev / ev.sum()
    return float(-np.sum(ev * np.log(ev)))


def concurrence(rho):
    """
    Wootters concurrence for a 2-qubit mixed state.
    C = max(0, lambda1 - lambda2 - lambda3 - lambda4)
    where lambdas are sqrt of eigenvalues of rho*(sysy rho* sysy).
    """
    sysy = np.kron(sy, sy)
    M    = rho @ (sysy @ rho.conj() @ sysy)
    ev   = np.sort(np.sqrt(np.maximum(np.real(np.linalg.eigvals(M)), 0)))[::-1]
    return float(max(0, ev[0] - ev[1] - ev[2] - ev[3]))


def tau3(psi):
    """
    CKW tripartite tangle (3-tangle) for a 3-qubit pure state.
        tau3 = 4 det(rho_A) - C^2(rho_AB) - C^2(rho_AC)
    Range [0, 1]. tau3 = 1 for GHZ, tau3 = 0 for W and product states.

    Key result: tau3 = 1 is IMPOSSIBLE for any SU(2)-invariant Hamiltonian
    because no rank-3 symmetric invariant tensor d_ijk exists in SU(2).
    """
    r = reduced_states(psi)
    ev_A = np.real(eigvalsh(r['rA']))
    C_sq = float(4 * max(0, np.prod(ev_A)))
    return float(C_sq - concurrence(r['rAB'])**2 - concurrence(r['rAC'])**2)


def mutual_information(psi):
    """
    All pairwise mutual informations for a 3-qubit pure state.
    I(X:Y) = S(X) + S(Y) - S(XY)
    Returns dict: IAB, IAC, IBC
    """
    r = reduced_states(psi)
    sA  = von_neumann(r['rA']);  sB  = von_neumann(r['rB'])
    sC  = von_neumann(r['rC'])
    sAB = von_neumann(r['rAB']); sAC = von_neumann(r['rAC'])
    sBC = von_neumann(r['rBC'])
    return {
        'IAB': sA + sB - sAB,
        'IAC': sA + sC - sAC,
        'IBC': sB + sC - sBC,
    }


def chsh(rho_AB):
    """
    CHSH parameter with optimal measurement angles.
    Optimal angles: a=0, a'=pi/2, b=pi/4, b'=-pi/4.
    Tsirelson bound: B <= 2*sqrt(2).
    """
    def E(ta, tb):
        oa = np.cos(ta)*sz + np.sin(ta)*sx
        ob = np.cos(tb)*sz + np.sin(tb)*sx
        return np.real(np.trace(rho_AB @ np.kron(oa, ob)))
    return abs(E(0, np.pi/4) + E(0, -np.pi/4)
             + E(np.pi/2, np.pi/4) - E(np.pi/2, -np.pi/4))


def ghz_fidelity(psi):
    """Fidelity with the GHZ state: F = |<GHZ|psi>|^2"""
    return float(abs(psi.conj() @ GHZ)**2)


# ── Time evolution ────────────────────────────────────────────────

def evolve(H_int, H_clock, psi0, N=400, T=12*np.pi, hbar=1.0):
    """
    Unitary time evolution under H_total = H_clock + H_int.
    Uses exact matrix exponentiation (scipy.linalg.expm).

    Parameters:
        H_int   : interaction Hamiltonian (8x8)
        H_clock : clock Hamiltonian (8x8), use make_H_clock_qubit()
        psi0    : initial state (8,)
        N       : number of time steps
        T       : total evolution time
        hbar    : reduced Planck constant (default 1.0)

    Returns:
        times   : array of shape (N,)
        results : dict with arrays of shape (N,):
                  tau3, fghz, SA, concAB, CHSH, IAB, IAC, IBC
    """
    H_total = H_clock + H_int
    U       = expm(-1j * H_total * T / N / hbar)

    psi    = psi0.copy()
    psi   /= np.linalg.norm(psi)
    times  = np.linspace(0, T, N)

    out = {k: np.zeros(N) for k in
           ['tau3','fghz','SA','concAB','CHSH','IAB','IAC','IBC']}

    for step in range(N):
        r = reduced_states(psi)
        mi = mutual_information(psi)

        out['tau3'][step]   = tau3(psi)
        out['fghz'][step]   = ghz_fidelity(psi)
        out['SA'][step]     = von_neumann(r['rA'])
        out['concAB'][step] = concurrence(r['rAB'])
        out['CHSH'][step]   = chsh(r['rAB'])
        out['IAB'][step]    = mi['IAB']
        out['IAC'][step]    = mi['IAC']
        out['IBC'][step]    = mi['IBC']

        psi  = U @ psi
        psi /= np.linalg.norm(psi)

    return times, out


def evolve_max(H_int, H_clock, psi0, N=400, T=12*np.pi, hbar=1.0):
    """
    Like evolve() but returns only the maximum values of each observable.
    Much faster when you only need tau3_max, fghz_max, etc.
    """
    H_total = H_clock + H_int
    U       = expm(-1j * H_total * T / N / hbar)
    psi     = psi0.copy()
    psi    /= np.linalg.norm(psi)

    maxvals = {k: 0.0 for k in ['tau3','fghz','SA','concAB','CHSH']}

    for _ in range(N):
        r = reduced_states(psi)
        vals = {
            'tau3':   tau3(psi),
            'fghz':   ghz_fidelity(psi),
            'SA':     von_neumann(r['rA']),
            'concAB': concurrence(r['rAB']),
            'CHSH':   chsh(r['rAB']),
        }
        for k in maxvals:
            if vals[k] > maxvals[k]:
                maxvals[k] = vals[k]
        psi  = U @ psi
        psi /= np.linalg.norm(psi)

    return maxvals


# ── Dark state criterion ──────────────────────────────────────────

def check_dark_state_criteria(H_AB, psi0_AB, tol=1e-8):
    """
    Check the three-condition criterion for entanglement emergence.
    Given H_AB (4x4) and initial AB state psi0_AB (4,):

    C1: psi0 is NOT an eigenstate of H_AB
    C2: H_AB|psi0> is not a product state
    C3: projection inside symmetry sector mixes states

    Returns:
        dict with keys C1, C2, C3 (bool) and residuals r1, r2, r3 (float)
        pred: C1 AND C2 AND C3  (predicted entanglement > 0.1)
    """
    Hp = H_AB @ psi0_AB
    norm_Hp = np.linalg.norm(Hp)

    # C1: not an eigenstate
    if norm_Hp < tol:
        c1, r1 = False, 0.0
    else:
        mu = (psi0_AB.conj() @ Hp) / (psi0_AB.conj() @ psi0_AB)
        r1 = float(np.linalg.norm(Hp - mu * psi0_AB))
        c1 = r1 > tol

    # C2: H|psi> is not a product state (Schmidt rank > 1)
    if norm_Hp < tol:
        c2, r2 = False, 0.0
    else:
        mat  = (Hp / norm_Hp).reshape(2, 2)
        svs  = np.linalg.svd(mat, compute_uv=False)
        svs  = svs[svs > tol]
        c2   = len(svs) > 1
        r2   = float(svs[1]) if len(svs) > 1 else 0.0

    # C3: intra-subspace mixing
    Pplus  = np.diag([1, 0, 0, 1]).astype(complex)
    Pminus = np.diag([0, 1, 1, 0]).astype(complex)
    c3, r3 = False, 0.0
    for P in [Pplus, Pminus]:
        Pp = P @ H_AB @ P @ psi0_AB
        np_Pp = np.linalg.norm(Pp)
        if np_Pp > tol:
            overlap = abs(psi0_AB.conj() @ Pp) / np_Pp
            if overlap < 1.0 - tol:
                c3, r3 = True, float(np_Pp)
                break

    return {'C1': c1, 'C2': c2, 'C3': c3,
            'r1': r1, 'r2': r2, 'r3': r3,
            'pred': c1 and c2 and c3}
