# =========================================================
# Causal Light-Bending Magnification Data Generator (FAST)
# =========================================================
# Same physics & outputs as your original, but much faster:
# - Continuation + caching for delta(b)
# - Adaptive precision near b_crit
# - Bracketed root solve with continuation in beta
# - Stable finite-difference step for dδ/db
# =========================================================

import mpmath as mp
import numpy as np
import math
from functools import lru_cache

# ---------- Global precision (base); raised adaptively near b_crit
mp.mp.dps = 80

# ---------- Constants
m0  = mp.mpf('1.0')
DL  = mp.mpf('1e6')
DS  = mp.mpf('2e6')
DLS = mp.mpf('1e6')

RHO_MIN = (m0/2) * (1 + mp.mpf('1e-9'))
WEAK_FIELD_SPLIT_THRESHOLD = mp.mpf('150.0')
STRICT_RADICAND = False

# ---------- Operational profiles
def n_operational(rho):
    rho = mp.mpf(rho)
    a = m0/(2*rho)
    # n = psi^3 / f_t = (1+a)^3 / (1-a)
    return ((1 + a)**3) / (1 - a)

def dn_drho(rho):
    rho = mp.mpf(rho)
    k = m0/2
    a = k/rho
    n = n_operational(rho)
    factor = (3/(1 + a)) + (1/(1 - a))   # d ln n / da
    return n * factor * (-k) / (rho**2)

def b_of_rho(rho):
    return n_operational(rho) * rho

def db_drho(rho):
    return n_operational(rho) + rho*dn_drho(rho)

# ---------- Tail mass (for info) and photon sphere
def estimate_m_eff(sample_radii=(mp.mpf('1e4'), mp.mpf('2e4'), mp.mpf('3e4'))):
    vals = [ (n_operational(R)-1)*R/2 for R in sample_radii ]
    vals.sort(); k = len(vals)//2
    return vals[k] if len(vals)%2 else (vals[k-1]+vals[k])/2

m_eff = estimate_m_eff()

def photon_sphere():
    a = RHO_MIN * (1 + mp.mpf('1e-3'))
    # Find root of db/drho = 0
    rho_ph = mp.findroot(db_drho, (a*1.5, a*2.5))
    bcrit  = b_of_rho(rho_ph)
    return rho_ph, bcrit

rho_ph, b_crit = photon_sphere()

# ---------- Adaptive precision suggestion near b_crit
def suggested_dps(b, bcrit, dps_min=60, dps_max=120):
    b = mp.mpf(b); bcrit = mp.mpf(bcrit)
    if b <= bcrit:
        return dps_max
    tau = max(mp.mpf('1e-10'), (b/bcrit) - 1)
    boost = 18 * max(0, -mp.log10(tau))  # modest boost as b→bcrit
    return int(min(dps_max, max(dps_min, 60 + boost)))

# ---------- Turning point & deflection with continuation
class DeflectionCalculator:
    def __init__(self):
        self.last_b   = None
        self.last_rho = None

    def _rho0_from_b(self, b_target):
        B = mp.mpf(b_target)
        if B <= b_crit:
            return rho_ph, True

        # Try to reuse last_rho as a Newton seed if monotone step
        def g(r):  return b_of_rho(r) - B
        def gp(r): return db_drho(r)

        if (self.last_b is not None) and (self.last_rho is not None):
            r0 = self.last_rho
            try:
                rN = r0 - g(r0)/gp(r0)
                if rN > RHO_MIN and mp.isfinite(rN):
                    return rN, False
            except Exception:
                pass

        # Otherwise do a short bracket+bisection (robust and quick)
        s_asym = max(RHO_MIN*(1 + mp.mpf('1e-6')), B - 2*m_eff)
        lo, hi = s_asym, B + 10*m_eff
        glo, ghi = g(lo), g(hi)
        k = 0
        while glo*ghi > 0 and k < 60:
            lo = max(RHO_MIN*(1 + mp.mpf('1e-6')), lo*mp.mpf('0.9'))
            hi = hi*mp.mpf('1.3')
            glo, ghi = g(lo), g(hi)
            k += 1
        if glo*ghi > 0:
            # fallback: Newton with two seeds
            r0 = mp.findroot(g, (s_asym, hi))
            return r0, False
        A, C = lo, hi
        for _ in range(80):
            M = (A + C)/2
            gM = g(M)
            if mp.fabs(gM) < mp.mpf('1e-25')*(1 + mp.fabs(B)):
                return M, False
            if glo*gM <= 0:
                C = M; ghi = gM
            else:
                A = M; glo = gM
        return (A + C)/2, False

    def _deflection_from_rho0(self, rho0):
        rho0 = mp.mpf(rho0)
        b = b_of_rho(rho0)

        def rad(r):
            return n_operational(r)**2 - (b**2)/(r**2)

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
            if v <= 0:
                if STRICT_RADICAND: raise ValueError("Radicand < 0 (near)")
                return mp.mpf('0.0')
            return (b/(r*r*mp.sqrt(v))) * (2*s)

        I_near = mp.quad(f_near, [mp.mpf('0.0'), s_max])

        def f_tail(u):
            r = 1/u
            v = rad(r)
            if v <= 0:
                if STRICT_RADICAND: raise ValueError("Radicand < 0 (tail)")
                return mp.mpf('0.0')
            return - b / mp.sqrt(v)

        I_tail = mp.quadts(f_tail, [1/split, mp.mpf('0.0')])

        return mp.re(2*(I_near + I_tail) - mp.pi)

    @lru_cache(maxsize=4096)
    def delta(self, b_target_float):
        # cache key must be hashable -> use float
        b_target = mp.mpf(b_target_float)
        prev = mp.mp.dps
        mp.mp.dps = suggested_dps(b_target, b_crit)
        try:
            rho0, captured = self._rho0_from_b(b_target)
            if captured:
                return mp.inf
            dlt = self._deflection_from_rho0(rho0)
            # update continuation anchors
            self.last_b   = b_target
            self.last_rho = rho0
            return dlt
        finally:
            mp.mp.dps = prev

DEFLECT = DeflectionCalculator()

# Small wrapper to match your previous name
def delta_of_b(b):
    return DEFLECT.delta(float(b))

# ---------- Lens equation helpers
def _delta_of_theta(theta):
    b = DL * mp.fabs(theta)
    d = delta_of_b(b)
    if d is mp.inf:
        return mp.inf
    # sign convention: deflection changes sign with image parity
    return mp.sign(theta) * d

def _F(beta):
    factor = DLS / DS
    def F(theta):
        d = _delta_of_theta(theta)
        if not mp.isfinite(d):
            # push away from capture
            return mp.sign(theta) * mp.mpf('1e3')
        return theta - beta - factor*d
    return F

def _ensure_safe_theta(theta, margin=mp.mpf('1e-9')):
    th_min = (1 + margin) * (b_crit / DL)
    if mp.fabs(theta) < th_min:
        theta = mp.sign(theta) * th_min if theta != 0 else th_min
    return theta

def _bracket_root(F, lo, hi, max_expand=20):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    if lo > hi: lo, hi = hi, lo
    flo, fhi = F(lo), F(hi)
    k = 0
    while (not (mp.isfinite(flo) and mp.isfinite(fhi))) or flo*fhi > 0:
        span = hi - lo
        lo -= span; hi += span
        lo = _ensure_safe_theta(lo); hi = _ensure_safe_theta(hi)
        flo, fhi = F(lo), F(hi)
        k += 1
        if k > max_expand:
            return None, None
    return lo, hi

def _bisect_root(F, lo, hi, tol=mp.mpf('1e-30'), max_it=160):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    flo, fhi = F(lo), F(hi)
    for _ in range(max_it):
        mid  = (lo + hi)/2
        fmid = F(mid)
        if not (mp.isfinite(fmid) and mp.isfinite(flo) and mp.isfinite(fhi)):
            if not mp.isfinite(flo): lo, flo = mid, fmid; continue
            if not mp.isfinite(fhi): hi, fhi = mid, fmid; continue
        if mp.fabs(fmid) < tol:
            return mid
        if flo*fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return (lo + hi)/2

def solve_lens_equation(beta, branch, theta_seed=None):
    # branch: '+' for outer, '-' for inner
    beta = mp.mpf(beta)
    F = _F(beta)
    theta_E = mp.sqrt(4*m_eff*DLS/(DL*DS))
    sgn = mp.mpf(1) if branch == '+' else mp.mpf(-1)
    base = sgn * theta_E if (theta_seed is None) else mp.mpf(theta_seed)

    # build a bracket around the seed (continuation-friendly)
    lo, hi = _ensure_safe_theta(0.4*base), _ensure_safe_theta(2.5*base)
    out = _bracket_root(F, lo, hi)
    if out == (None, None):
        # fallback wider bracket
        lo, hi = _ensure_safe_theta(0.1*base), _ensure_safe_theta(8*base)
        out = _bracket_root(F, lo, hi)
        if out == (None, None):
            # last resort: secant
            try:
                return mp.findroot(F, (_ensure_safe_theta(base), _ensure_safe_theta(1.1*base)))
            except Exception:
                return None
    lo, hi = out
    return _bisect_root(F, lo, hi)

# ---------- Magnification with cached δ(b) and stable step
def magnification(theta):
    theta = mp.mpf(theta)
    b = DL * mp.fabs(theta)

    # central deflection (reuse cached)
    d0 = delta_of_b(b)
    if d0 is mp.inf:
        return mp.nan

    # beta from lens equation
    beta = theta - (DLS/DS) * mp.sign(theta) * d0

    # derivative dδ/db using a *relative* step (reuses cache for δ(b±h))
    h = max(mp.mpf('1e-6')*b, mp.mpf('1e-9'))
    dp = delta_of_b(b + h)
    dm = delta_of_b(max(b - h, b*mp.mpf('0.999999')))  # keep positive
    d_delta_db = (dp - dm) / (2*h)

    d_delta_dtheta = d_delta_db * DL * mp.sign(theta)
    d_beta_d_theta = 1 - (DLS/DS) * d_delta_dtheta

    if beta == 0 or d_beta_d_theta == 0 or (not mp.isfinite(d_beta_d_theta)):
        return mp.inf

    mu = mp.fabs((theta / beta) / d_beta_d_theta)
    return mu

# =================================================
# Sweep over beta with continuation in theta
# =================================================
if __name__ == "__main__":
    # Log-spaced betas
    beta_values = np.logspace(-5, -2, 40, base=10.0)

    print("# DATA FOR MAGNIFICATION CURVE (Figure: Total Magnification vs Source Angle)")
    print("# Constants: Dl=1e6, Ds=2e6, Dls=1e6, m_eff≈{}".format(mp.nstr(m_eff,7)))
    print("# Columns: beta (Source Angle), mu_outer (Outer Image Mag), mu_inner (Inner Image Mag), mu_total (Total Mag)")
    print("beta,mu_outer,mu_inner,mu_total")

    # seeds for continuation in theta
    theta_seed_plus  = None
    theta_seed_minus = None

    for beta in beta_values:
        beta = mp.mpf(beta)

        # solve each branch; use previous theta as seed (continuation)
        theta_plus  = solve_lens_equation(beta, branch='+', theta_seed=theta_seed_plus)
        theta_minus = solve_lens_equation(beta, branch='-', theta_seed=theta_seed_minus)

        mu_plus = mu_minus = mp.nan
        if theta_plus is not None:
            mu_plus = magnification(theta_plus)
            theta_seed_plus = theta_plus
        if theta_minus is not None:
            mu_minus = magnification(theta_minus)
            theta_seed_minus = theta_minus

        mu_total = mp.nan
        if mp.isfinite(mu_plus) and mp.isfinite(mu_minus):
            mu_total = mu_plus + mp.fabs(mu_minus)

        print(f"{float(beta):.10e},{float(mu_plus):.8f},{float(mu_minus):.8f},{float(mu_total):.8f}")