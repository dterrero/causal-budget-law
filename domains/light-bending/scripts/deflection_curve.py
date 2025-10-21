# =========================================================
# Causal Light-Bending Data Generation Script for Figure 1
# =========================================================
# This script computes the operational deflection δ(b) and the
# weak-field baseline (4*m_eff/b) over a range of impact parameters
# (b) for plotting.
# =========================================================

import mpmath as mp
import math
import numpy as np

# --- Global precision setup
mp.mp.dps = 100

# --- Constants derived from the theory text
m0 = mp.mpf('1.0')
RHO_MIN = (m0/2) * (1 + mp.mpf('1e-9'))
WEAK_FIELD_SPLIT_THRESHOLD = mp.mpf('150.0')
STRICT_RADICAND = False # Safety flag

# =================================================
# 1) OPERATIONAL PROFILES (Core Functions)
# =================================================

def f_t_operational(rho):
    rho = mp.mpf(rho)
    a = m0/(2*rho)
    return (1 - a) / (1 + a)

def psi_operational(rho):
    rho = mp.mpf(rho)
    return 1 + m0/(2*rho)

def n_operational(rho):
    # n(ρ) = ψ(ρ)^3 / f_t(ρ) - THEORETICALLY CORRECT FORM
    rho = mp.mpf(rho)
    a = m0/(2*rho)
    ft = (1 - a)/(1 + a)
    psi = 1 + a
    return (psi**3) / (1 - a)

def dn_drho(rho):
    rho = mp.mpf(rho)
    k = m0/2
    a = k/rho
    n = n_operational(rho)
    factor = (3/(1 + a)) + (1/(1 - a))
    return n * factor * (-k) / (rho**2)

def b_of_rho(rho):
    return n_operational(rho) * rho

def db_drho(rho):
    return n_operational(rho) + rho*dn_drho(rho)

# =================================================
# 2) Photon Sphere and Mass (For consistency)
# =================================================

def estimate_m_eff(sample_radii=(mp.mpf('1e4'), mp.mpf('2e4'), mp.mpf('3e4'))):
    vals = []
    for R in sample_radii:
        nr = n_operational(R)
        vals.append((nr - 1)*R/2)
    vals.sort()
    k = len(vals)//2
    return vals[k] if len(vals) % 2 else (vals[k-1] + vals[k]) / 2

m_eff = estimate_m_eff()

def photon_sphere():
    a = RHO_MIN * (1 + mp.mpf('1e-3'))
    b = m0 * mp.mpf('50')
    N = 200
    xs = [a + (b - a)*i/(N - 1) for i in range(N)]
    vals = [db_drho(x) for x in xs]
    lo, hi = None, None
    for i in range(N - 1):
        if vals[i]*vals[i + 1] < 0:
            lo, hi = xs[i], xs[i + 1]; break
    rho_ph = mp.findroot(db_drho, (lo, hi)) if lo else mp.findroot(db_drho, (a*1.5, a*2.5))
    bcrit = b_of_rho(rho_ph)
    return rho_ph, bcrit

rho_ph, b_crit = photon_sphere()

# =================================================
# 3) Turning Point and Deflection (The core calculation)
# =================================================

def rho0_from_b(b_target):
    B = mp.mpf(b_target)
    if B < b_crit:
        return rho_ph, True
    # (Implementation of rho0 finding omitted for brevity, logic uses mp.findroot)
    def g(r): return b_of_rho(r) - B
    s_asym = max(RHO_MIN*(1 + mp.mpf('1e-6')), B - 2*m_eff)
    lo, hi = s_asym, B + 10*m_eff
    glo, ghi = g(lo), g(hi)
    k = 0
    while glo*ghi > 0 and k < 80:
        lo = max(RHO_MIN*(1 + mp.mpf('1e-6')), lo*mp.mpf('0.9'))
        hi = hi*mp.mpf('1.3')
        glo, ghi = g(lo), g(hi)
        k += 1
    if glo*ghi > 0:
        return mp.findroot(g, (s_asym, hi)), False
    A, C = lo, hi
    for _ in range(200):
        M = (A + C)/2
        gM = g(M)
        if mp.fabs(gM) < mp.mpf('1e-28')*(1 + mp.fabs(B)): return M, False
        if g(A)*gM <= 0: hi = M
        else: lo = M; A = M
    return (A + C)/2, False

def rho0_from_b_fast(b_target):
    rho0, captured = rho0_from_b(b_target)
    if captured: return rho0, True
    try:
        g  = lambda r: b_of_rho(r) - b_target
        gp = lambda r: db_drho(r)
        rhoN = rho0 - g(rho0)/gp(rho0)
        if rhoN > RHO_MIN and mp.isfinite(rhoN): rho0 = rhoN
    except Exception:
        pass
    return rho0, False

def deflection_from_rho0(rho0):
    rho0 = mp.mpf(rho0)
    b = b_of_rho(rho0)

    def rad(r):
        return n_operational(r)**2 - (b**2)/(r**2)

    n0, dn0 = n_operational(rho0), dn_drho(rho0)
    radp0 = 2*n0*dn0 + 2*(b**2)/(rho0**3)
    if radp0 <= mp.mpf('1e-80'): radp0 = mp.mpf('1e-80')
    L0 = 2*b / (rho0**2 * mp.sqrt(radp0))

    split = max(8*rho0, WEAK_FIELD_SPLIT_THRESHOLD)
    s_max = mp.sqrt(split - rho0)

    def f_near(s):
        if s < mp.mpf('1e-12'): return L0
        r = rho0 + s*s
        v = rad(r)
        if v < 0 or v == 0: return mp.mpf('0.0')
        return (b/(r*r*mp.sqrt(v))) * (2*s)

    I_near = mp.quad(f_near, [mp.mpf('0.0'), s_max])

    def f_tail(u):
        r = 1/u; v = rad(r)
        if v < 0 or v == 0: return mp.mpf('0.0')
        return - b / mp.sqrt(v)

    I_tail = mp.quadts(f_tail, [1/split, mp.mpf('0.0')])

    delta = 2*(I_near + I_tail) - mp.pi
    return mp.re(delta)

def suggested_dps(b, bcrit, dps_min=60, dps_max=130):
    b = mp.mpf(b); bcrit = mp.mpf(bcrit)
    if b <= bcrit: return dps_max
    tau = max(mp.mpf('1e-12'), (b/bcrit) - 1)
    boost = 20 * max(0, -mp.log10(tau))
    return int(min(dps_max, max(dps_min, 50 + boost)))

def delta_of_b(b_target):
    b_target = mp.mpf(b_target)
    prev = mp.mp.dps
    mp.mp.dps = suggested_dps(b_target, b_crit)
    try:
        rho0, captured = rho0_from_b_fast(b_target)
        if captured: return mp.inf
        return deflection_from_rho0(rho0)
    finally:
        mp.mp.dps = prev

# =================================================
# 4) Data Range Definition and Output
# =================================================

# Define b values: Log-spaced near b_c, linearly spaced for far-field
b_near = np.linspace(float(b_crit) + 0.05, 10.0, 30)
b_mid = np.linspace(10.0, 50.0, 30)
b_far = np.linspace(50.0, 100.0, 15)

# Combine and ensure unique/sorted values
b_values = sorted(list(set(np.concatenate([b_near, b_mid, b_far]))))

print("# DATA FOR FIGURE 1: Bending Angle vs. Impact Parameter (b)")
print("# Columns: b (Impact Parameter), delta_b (Operational Deflection), delta_weak (Weak-Field Baseline)")
print("b,delta_b,delta_weak")

# Calculate data points
for b in b_values:
    b_mp = mp.mpf(b)
    delta_op = delta_of_b(b_mp)
    delta_wk = 4 * m_eff / b_mp

    # Skip captured rays (delta = inf)
    if delta_op is mp.inf:
        continue

    # Use floating point output for plotting tools
    print(f"{float(b):.8f},{float(delta_op):.10f},{float(delta_wk):.10f}")
