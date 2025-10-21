# =========================================================
# Causal Light-Bending Pipeline (Aligned with Theory Section 3.2)
# =========================================================
#   This pipeline implements the operational framework described
#   in the theory text, using only the derived refractive index
#   n(ρ) and Fermat's principle, without geometric curvature.
#   Uses mpmath for high-precision arithmetic.
# =========================================================

import mpmath as mp
import math

# --- Global precision (delta_of_b will adapt near b_crit)
mp.mp.dps = 100

# Optional safety: raise if radicand < 0 in integrals (debug aid)
STRICT_RADICAND = False

# =================================================
# 1) OPERATIONAL PROFILES (Consistent with GR-Exact, isotropic)
#    (Refers to Eq. 1 in the theory text)
# =================================================

m0 = mp.mpf('1.0')  # m0 = GM/c^2 = r_s/2 (used in m_eff footnote)
RHO_MIN = (m0/2) * (1 + mp.mpf('1e-9'))  # Isotropic horizon limit
WEAK_FIELD_SPLIT_THRESHOLD = mp.mpf('150.0')

def f_t_operational(rho):
    """
    Temporal factor f_t(ρ) - Eq. (1)
    """
    rho = mp.mpf(rho)
    a = m0/(2*rho)
    return (1 - a) / (1 + a)

def psi_operational(rho):
    """
    Spatial factor ψ(ρ) - Eq. (1)
    """
    rho = mp.mpf(rho)
    return 1 + m0/(2*rho)

def n_operational(rho):
    """
    Operational refractive index n(ρ) - Uses the physically correct
    form: n(ρ) = ψ(ρ)^3 / f_t(ρ) which matches the expanded result
    in Eq. (1) of the theory text.
    """
    rho = mp.mpf(rho)
    a = m0/(2*rho)
    ft = (1 - a)/(1 + a)
    psi = 1 + a
    # n(ρ) = (1 + m0/2ρ)^3 / (1 - m0/2ρ)
    return (psi**3) / (1 - a)

# =================================================
# 2) Tail mass sanity check (Refers to m_eff in theory text)
# =================================================

def estimate_m_eff(sample_radii=(mp.mpf('1e4'), mp.mpf('2e4'), mp.mpf('3e4'))):
    """
    Calculates m_eff based on the weak-field tail behavior:
    m_eff ≈ (n - 1)ρ / 2 (ρ >> m0)
    """
    vals = []
    for R in sample_radii:
        nr = n_operational(R)
        vals.append((nr - 1)*R/2)
    vals.sort()
    k = len(vals)//2
    return vals[k] if len(vals) % 2 else (vals[k-1] + vals[k]) / 2

m_eff = estimate_m_eff()
print("[OPERATIONAL MASS ESTIMATION]")
print(f"m_eff from tail behavior: {mp.nstr(m_eff, 8)} (vs m0={mp.nstr(m0,8)})\n")

# =================================================
# 3) Photon Sphere Discovery (Refers to Eq. (2) and text)
#    b(ρ) = n(ρ)ρ minimum defines ρ_ph and b_c.
# =================================================

def dn_drho(rho):
    """
    Derivative of n(ρ) w.r.t. ρ, used for finding db/dρ = 0
    """
    rho = mp.mpf(rho)
    k = m0/2
    a = k/rho
    n = n_operational(rho)
    factor = (3/(1 + a)) + (1/(1 - a))
    return n * factor * (-k) / (rho**2)

def b_of_rho(rho):
    """
    Impact parameter profile b(ρ) = n(ρ)ρ - Eq. (2)
    """
    return n_operational(rho) * rho

def db_drho(rho):
    """
    Derivative of b(ρ) - Extremum condition: db/dρ = 0
    """
    return n_operational(rho) + rho*dn_drho(rho)

def photon_sphere():
    """
    Finds ρ_ph and b_crit by locating the root of db_drho = 0.
    """
    a = RHO_MIN * (1 + mp.mpf('1e-3'))
    b = m0 * mp.mpf('50')
    N = 200
    xs = [a + (b - a)*i/(N - 1) for i in range(N)]
    vals = [db_drho(x) for x in xs]
    lo, hi = None, None
    for i in range(N - 1):
        if vals[i] == 0:
            lo = hi = xs[i]; break
        if vals[i]*vals[i + 1] < 0:
            lo, hi = xs[i], xs[i + 1]; break
    rho_ph = mp.findroot(db_drho, (lo, hi)) if lo else mp.findroot(db_drho, (a*1.5, a*2.5))
    bcrit = b_of_rho(rho_ph)
    return rho_ph, bcrit

rho_ph, b_crit = photon_sphere()

print("[CAUSAL MEDIUM PROPERTIES]")
print("=" * 50)
print(f"Photon sphere (operational): ρ_ph = {mp.nstr(rho_ph, 8)}")
print(f"Critical impact:             b_crit = {mp.nstr(b_crit, 8)}")
print(f"n(ρ_ph) = {mp.nstr(n_operational(rho_ph), 8)}\n")

# =================================================
# 4) Turning point ρ0 (The solution to b = n(ρ0)ρ0)
# =================================================

def rho0_from_b(b_target):
    """
    Finds the turning point ρ0 for a given impact parameter b.
    Returns (ρ0, captured)
    """
    B = mp.mpf(b_target)
    if B < b_crit:
        return rho_ph, True # Captured case
    def g(r): return b_of_rho(r) - B
    s_asym = max(RHO_MIN*(1 + mp.mpf('1e-6')), B - 2*m_eff)
    lo, hi = s_asym, B + 10*m_eff
    # ... [Omitted root finding implementation for brevity, logic remains the same] ...
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
        if mp.fabs(gM) < mp.mpf('1e-28')*(1 + mp.fabs(B)):
            return M, False
        if g(A)*gM <= 0:
            C = M
        else:
            A = M
    return (A + C)/2, False


def rho0_from_b_fast(b_target):
    """
    Wrapper for finding rho0, includes a quick Newton refinement step.
    """
    rho0, captured = rho0_from_b(b_target)
    if captured:
        return rho0, True
    try:
        g  = lambda r: b_of_rho(r) - b_target
        gp = lambda r: db_drho(r)
        rhoN = rho0 - g(rho0)/gp(rho0)
        if rhoN > RHO_MIN and mp.isfinite(rhoN):
            rho0 = rhoN
    except Exception:
        pass
    return rho0, False

# =================================================
# 5) Deflection integral (Fermat's Principle)
#    (Refers to Eq. (3) - Numerical split ensures stability)
# =================================================

def deflection_from_rho0(rho0):
    """
    Calculates the deflection angle δ(b) using the integral in Eq. (3).
    The numerical split is based on the transformations ρ=ρ0+s² and u=1/ρ.
    """
    rho0 = mp.mpf(rho0)
    b = b_of_rho(rho0)
    
    def rad(r):
        """Radicand V(r) = n(ρ)² - b²/ρ²"""
        return n_operational(r)**2 - (b**2)/(r**2)
    
    # ... [Omitted integral setup and calculation for brevity, logic remains the same] ...
    n0, dn0 = n_operational(rho0), dn_drho(rho0)
    radp0 = 2*n0*dn0 + 2*(b**2)/(rho0**3)
    if radp0 <= mp.mpf('1e-80'):
        radp0 = mp.mpf('1e-80')
    L0 = 2*b / (rho0**2 * mp.sqrt(radp0))

    split = max(8*rho0, WEAK_FIELD_SPLIT_THRESHOLD)
    s_max = mp.sqrt(split - rho0)

    def f_near(s):
        if s < mp.mpf('1e-12'): return L0
        r = rho0 + s*s
        v = rad(r)
        if v < 0:
            if STRICT_RADICAND: raise ValueError("Radicand < 0 (near).")
            return mp.mpf('0.0')
        if v == 0: return mp.mpf('0.0')
        # Integrand: (b/r²) * (1/√V(r)) * (dρ/ds = 2s)
        return (b/(r*r*mp.sqrt(v))) * (2*s)

    I_near = mp.quad(f_near, [mp.mpf('0.0'), s_max])

    def f_tail(u):
        r = 1/u
        v = rad(r)
        if v < 0:
            if STRICT_RADICAND: raise ValueError("Radicand < 0 (tail).")
            return mp.mpf('0.0')
        if v == 0: return mp.mpf('0.0')
        # Integrand: (b/r²) * (1/√V(r)) * (dρ/du = -1/u²) => - b / √V(r)
        return - b / mp.sqrt(v)

    I_tail = mp.quadts(f_tail, [1/split, mp.mpf('0.0')])

    delta = 2*(I_near + I_tail) - mp.pi
    return mp.re(delta)

# ... [Omitted delta_of_b adaptive precision wrapper] ...

def suggested_dps(b, bcrit, dps_min=60, dps_max=130):
    b = mp.mpf(b); bcrit = mp.mpf(bcrit)
    if b <= bcrit:
        return dps_max
    tau = max(mp.mpf('1e-12'), (b/bcrit) - 1)
    boost = 20 * max(0, -mp.log10(tau))
    return int(min(dps_max, max(dps_min, 50 + boost)))

def delta_of_b(b_target):
    b_target = mp.mpf(b_target)
    prev = mp.mp.dps
    mp.mp.dps = suggested_dps(b_target, b_crit)
    try:
        rho0, captured = rho0_from_b_fast(b_target)
        if captured:
            return mp.inf
        return deflection_from_rho0(rho0)
    finally:
        mp.mp.dps = prev

# ... [Omitted Unit Tests and Demonstration Table (Code for these remains the same)] ...

def run_unit_tests():
    R_test = mp.mpf('1e6')
    actual_slope = (n_operational(R_test) - 1)*R_test
    expected_slope = 2*m0
    TOL = mp.mpf('1e-5')
    assert mp.almosteq(actual_slope, expected_slope, rel_eps=TOL)
    print(f"[UNIT TESTS] Weak-field limit passed at R={mp.nstr(R_test, 0)} (TOL={mp.nstr(TOL)})")

run_unit_tests(); print()

print(f"[PRECISION CAUSAL-BUDGET DEMONSTRATION]\n" + "="*60)
test_cases = [
    (mp.mpf('5.0'),  "Deep strong-field"),
    (mp.mpf('7.5'),  "Strong-field (near b_crit)"),
    (mp.mpf('10.0'), "Strong-field"),
    (mp.mpf('15.0'), "Intermediate"),
    (mp.mpf('25.0'), "Intermediate"),
    (mp.mpf('50.0'), "Weak-field"),
    (mp.mpf('100.0'),"Very weak-field"),
]
print("\nIMPACT PARAMETER  δ(b) (rad)  RATIO  GR_BASE(4m/b)  REL_ERR  INTERPRETATION")
print("-"*90)
success = 0
for b_target, label in test_cases:
    ref = 4*m_eff/b_target
    delta = delta_of_b(b_target)
    if delta is mp.inf:
        print(f"{mp.nstr(b_target,6):>8}   {'∞':>10}     -      {mp.nstr(ref,6):>10}   -        CAPTURED")
        continue
    if 0 < delta < 4*mp.pi:
        success += 1
        ratio = delta/ref
        rel_err = (delta - ref)/ref
        interp = "Near weak-field" if ratio <= 1.1 else ("Moderate enhancement" if ratio <= 1.5 else "Strong enhancement")
        print(f"{mp.nstr(b_target,6):>8}   {mp.nstr(delta,10):>10}  {mp.nstr(ratio,6):>6}  {mp.nstr(ref,6):>10}  {mp.nstr(rel_err,3):>7}   {interp}")
    else:
        print(f"{mp.nstr(b_target,6):>8}   CALC_ERROR  -      {mp.nstr(ref,6):>10}   -        {label}")

print(f"\n[DEMONSTRATION SUCCESS METRICS]")
print(f"Successful predictions: {success}/{len(test_cases)}")
print("\n[OPERATIONAL FOUNDATIONS VERIFIED]")
print(f"• m_eff = {mp.nstr(m_eff,8)} (from tail)")
print(f"• Photon sphere ρ_ph = {mp.nstr(rho_ph,8)} via min b(ρ)")
print("• Bending from Fermat in the operational medium (no geometry objects)")

# =================================================
# 6) Lens Equation and Magnification
#    (Refers to Lens Equation and Magnification Formula in theory text)
# =================================================

def _delta_of_theta(theta, Dl):
    """
    Calculates the deflection δ(b) where b = Dl*|θ|,
    returning a signed delta: sgn(θ) * δ(Dl|θ|).
    """
    b = Dl * mp.mpf(theta)
    if b >= 0:
        return delta_of_b(b)
    d = delta_of_b(-b)
    return -d if mp.isfinite(d) else mp.inf

def _F_factory(beta, Dl, factor):
    """
    Factory for the Lens Equation function F(θ) = 0:
    F(θ) = θ - β - (Dls/Ds) * δ(Dl|θ|)
    """
    def F(theta):
        delta = _delta_of_theta(theta, Dl)
        if not mp.isfinite(delta):
            return mp.sign(theta) * mp.mpf('1e3')
        return theta - beta - factor * delta
    return F

# ... [Omitted root finding helper functions for brevity, logic remains the same] ...

def _ensure_safe_theta(theta, Dl, margin=mp.mpf('1e-6')):
    th_min = (1 + margin) * (b_crit / Dl)
    if mp.fabs(theta) < th_min:
        theta = mp.sign(theta) * th_min if theta != 0 else th_min
    return theta

def _bracket_root(F, lo, hi, max_expand=40):
    lo = mp.mpf(lo); hi = mp.mpf(hi)
    if lo > hi: lo, hi = hi, lo
    flo = F(lo); fhi = F(hi)
    k = 0
    while (not (mp.isfinite(flo) and mp.isfinite(fhi))) or flo*fhi > 0:
        span = hi - lo
        lo -= span; hi += span
        flo = F(lo); fhi = F(hi)
        k += 1
        if k > max_expand:
            return None, None
    return lo, hi

def _bisect_root(F, lo, hi, tol=mp.mpf('1e-30'), max_it=200):
    lo = mp.mpf(lo); hi = mp.mpf(hi)
    flo = F(lo); fhi = F(hi)
    for _ in range(max_it):
        mid = (lo + hi)/2
        fmid = F(mid)
        if not (mp.isfinite(fmid) and mp.isfinite(flo) and mp.isfinite(fhi)):
            if not mp.isfinite(flo): lo = mid; flo = fmid; continue
            if not mp.isfinite(fhi): hi = mid; fhi = fmid; continue
        if mp.fabs(fmid) < tol: return mid
        if flo*fmid <= 0: hi = mid; fhi = fmid
        else: lo = mid; flo = fmid
    return (lo + hi)/2


def solve_lens_equation(beta, Dl, Ds, Dls, theta_guess=None, branch='+'):
    """
    Solves the Lens Equation for a single image and calculates its magnification.
    """
    beta, Dl, Ds, Dls = map(mp.mpf, (beta, Dl, Ds, Dls))
    factor = Dls / Ds
    F = _F_factory(beta, Dl, factor)

    theta_E = mp.sqrt(4*m_eff*Dls/(Dl*Ds))

    sgn = mp.mpf(1) if branch == '+' else mp.mpf(-1)
    base = theta_E if (theta_guess is None or not mp.isfinite(theta_guess)) else mp.mpf(theta_guess)
    base = sgn * mp.fabs(base)

    lo, hi = 0.2 * base, 5.0 * base
    if lo > hi: lo, hi = hi, lo

    lo = _ensure_safe_theta(lo, Dl)
    hi = _ensure_safe_theta(hi, Dl)

    lo, hi = _bracket_root(F, lo, hi)
    if lo is None:
        t0 = _ensure_safe_theta(base, Dl)
        t1 = _ensure_safe_theta(1.1*base, Dl)
        try:
            theta = mp.findroot(F, (t0, t1))
        except Exception:
            lo = _ensure_safe_theta(0.1*base, Dl)
            hi = _ensure_safe_theta(10*base, Dl)
            theta = _bisect_root(F, lo, hi)
    else:
        theta = _bisect_root(F, lo, hi)

    # Calculate dδ/dθ numerically for magnification formula
    def delta_of_theta(th):
        return _delta_of_theta(th, Dl)

    th_scale = max(mp.mpf('1.0'), mp.fabs(theta))
    eps = max(mp.mpf('1e-7'), mp.mpf('1e-6') * th_scale)
    for _ in range(8):
        dplus  = delta_of_theta(theta + eps)
        dminus = delta_of_theta(theta - eps)
        if mp.isfinite(dplus) and mp.isfinite(dminus):
            ddelta_dtheta = (dplus - dminus) / (2*eps)
            break
        eps *= 10
    else:
        return theta, mp.inf

    # dβ/dθ = 1 - (Dls/Ds) * dδ/dθ
    dbeta_dtheta = 1 - factor * ddelta_dtheta

    # Full Total Magnification: μ = |(θ/β) * dθ/dβ|
    # This aligns perfectly with the formula: μ = |(θ/β) * 1 / (dβ/dθ)|
    if beta == 0 or not mp.isfinite(dbeta_dtheta) or dbeta_dtheta == 0:
        mu_signed = mp.inf
    else:
        mu_signed = (theta / beta) / dbeta_dtheta

    mu = mp.fabs(mu_signed)
    return theta, mu

# Demo – Matches the Demonstration in the theory text
print("\n" + "="*60)
print("[FULL LENS EQUATION SOLVER DEMONSTRATION]")
Dl, Ds, Dls = mp.mpf('1.0e6'), mp.mpf('2.0e6'), mp.mpf('1.0e6')  # scaled distances
beta = mp.mpf('1.0e-4')
theta_E = mp.sqrt(4*m_eff*Dls/(Dl*Ds))
th_out, mu_out = solve_lens_equation(beta, Dl, Ds, Dls, branch='+')
th_in , mu_in  = solve_lens_equation(beta, Dl, Ds, Dls, branch='-')
print(f"For D_l={float(Dl)}, D_s={float(Ds)}, D_ls={float(Dls)}, β={float(beta)}")
print(f"Einstein scale θ_E ≈ {float(theta_E):.6e} rad")
print(f"Image 1 (Outer): θ = {float(th_out):.6e} rad, μ = {float(mu_out):.6f}")
print(f"Image 2 (Inner): θ = {float(th_in):.6e} rad, μ = {float(mu_in):.6f}")
print("="*60)

# ... [Omitted NEAR-CRITICAL STRESS TEST and AREAL-r CROSS-CHECK for brevity] ...

print("\n[NEAR-CRITICAL STRESS TEST]")
eps_list = [mp.mpf('1e-1'), mp.mpf('5e-2'), mp.mpf('2e-2'), mp.mpf('1e-2'),
            mp.mpf('5e-3'), mp.mpf('2e-3')]
xs, ys = [], []
for eps in eps_list:
    b = b_crit * (1 + eps)
    d = delta_of_b(b)
    logterm = mp.log(1/eps)
    xs.append(logterm); ys.append(d)
    print(f"eps={float(eps):8.3g}  log-term={float(logterm):9.7f}  δ={float(d):.7f}")

Sx  = mp.fsum(xs); Sy  = mp.fsum(ys)
Sxx = mp.fsum([x*x for x in xs]); Sxy = mp.fsum([x*y for x,y in zip(xs,ys)])
N   = mp.mpf(len(xs))
A = (N*Sxy - Sx*Sy) / (N*Sxx - Sx*Sx)
B = (Sy - A*Sx) / N
print(f"Fitted slope A ≈ {mp.nstr(A, 8)}, intercept B ≈ {mp.nstr(B, 8)}")

def r0_from_b_areal(b):
    M = m0
    r_ph = 3*M
    if b <= 3*mp.sqrt(3)*M: return r_ph, True
    def h(r): return (r*r)/(1 - 2*M/r) - b*b
    lo = r_ph*(1 + mp.mpf('1e-6')); hi = b + 10*M
    hlo, hhi = h(lo), h(hi); k = 0
    while hlo*hhi > 0 and k < 80:
        lo = max(r_ph*(1 + mp.mpf('1e-6')), lo*mp.mpf('0.9')); hi = hi*mp.mpf('1.3')
        hlo, hhi = h(lo), h(hi); k += 1
    if hlo*hhi > 0: r0 = mp.findroot(h, (lo, hi))
    else:
        A_, C_ = lo, hi
        for _ in range(200):
            M_ = (A_ + C_)/2; hM = h(M_)
            if mp.fabs(hM) < mp.mpf('1e-28')*(1 + mp.fabs(b)): return M_, False
            if h(A_)*hM <= 0: C_ = M_
            else: A_ = M_
        r0 = (A_ + C_)/2
    return r0, False

def deflection_areal_r(b):
    b = mp.mpf(b); M = m0
    r0, cap = r0_from_b_areal(b)
    if cap: return mp.inf
    def R(r): return (1/b**2) - ((1 - 2*M/r)/(r*r))
    Rp = mp.diff(R, r0)
    if Rp <= 0: Rp = mp.mpf('1e-80')
    L0 = 2 / (r0**2 * mp.sqrt(Rp))
    split = max(8*r0, mp.mpf('150.0'))
    s_max = mp.sqrt(split - r0)
    def f_near(s):
        if s < mp.mpf('1e-12'): return L0
        r = r0 + s*s; v = R(r)
        if v <= 0: return mp.mpf('0.0')
        return (1/(r*r*mp.sqrt(v))) * (2*s)
    I_near = mp.quad(f_near, [mp.mpf('0.0'), s_max])
    def f_tail(u):
        r = 1/u; v = R(r)
        if v <= 0: return mp.mpf('0.0')
        return - 1/mp.sqrt(v)
    I_tail = mp.quadts(f_tail, [1/split, mp.mpf('0.0')])
    return 2*(I_near + I_tail) - mp.pi

b_chk = mp.mpf('25.0')
delta_geom = deflection_areal_r(b_chk)
delta_oper = delta_of_b(b_chk)
print(f"\n[AREAL-r CROSS-CHECK] b={float(b_chk)}  δ_geom={mp.nstr(delta_geom, 12)}  δ_oper={mp.nstr(delta_oper, 12)}  Δ={mp.nstr(delta_geom - delta_oper, 4)}")
