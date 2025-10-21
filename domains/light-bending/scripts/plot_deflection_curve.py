# deflection_plot_clean.py
# Plot δ(b) vs b for the operational (causal-budget) model vs the weak-field baseline.
# Robust to near-critical b and root-finding hiccups. No tricks or smoothing.

import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

mp.mp.dps = 80  # robust near b_crit

# ==========================
# 1) Operational profiles (match your pipeline)
# ==========================
alpha = mp.mpf('0.5')
beta  = mp.mpf('1.0')
m0    = mp.mpf('1.0')

RHO_MIN   = beta*m0*(1 + mp.mpf('1e-9'))
SPLIT_FAR = mp.mpf('150.0')

def f_t_operational(rho): 
    rho = mp.mpf(rho)
    return 1 - beta*m0/rho

def psi_operational(rho): 
    rho = mp.mpf(rho)
    return 1 + alpha*m0/rho

def n_operational(rho):
    rho = mp.mpf(rho)
    ft = f_t_operational(rho)
    if ft <= 0:
        # invalid region (would be captured anyway)
        raise ValueError("f_t<=0 for this ρ; outside domain")
    return (psi_operational(rho)**2) / ft

def estimate_m_eff():
    samples = [mp.mpf('100'), mp.mpf('200'), mp.mpf('300')]
    vals = sorted([(n_operational(R)-1)*R/2 for R in samples])
    return vals[1]  # median of 3

m_eff = estimate_m_eff()

def dn_drho(rho):
    rho = mp.mpf(rho)
    t = m0/rho
    num = beta + 2*alpha + 2*(alpha**2)*t - (alpha**2)*beta*(t**2)
    den = (1 - beta*t)**2
    dndt = num/den
    return dndt * (-m0/rho**2)

def b_of_rho(rho): 
    rho = mp.mpf(rho)
    return n_operational(rho)*rho

def db_drho(rho):  
    rho = mp.mpf(rho)
    return n_operational(rho) + rho*dn_drho(rho)

# Photon sphere by scanning for a sign change of db/drho, then polishing
def photon_sphere():
    a = RHO_MIN*(1 + mp.mpf('1e-3'))
    b = m0*mp.mpf('50')
    N = 400
    xs = [a + (b-a)*i/(N-1) for i in range(N)]
    ys = [db_drho(x) for x in xs]
    lo = hi = None
    for i in range(N-1):
        if ys[i] == 0:
            lo = hi = xs[i]; break
        if ys[i]*ys[i+1] < 0:
            lo, hi = xs[i], xs[i+1]; break
    if lo is None:
        # fallback: minimize b(ρ) on grid, then polish
        k = int(np.argmin([float(b_of_rho(x)) for x in xs]))
        seed = xs[k]
        rho_ph = mp.findroot(db_drho, seed)
    else:
        rho_ph = mp.findroot(db_drho, (lo, hi))
    return rho_ph, b_of_rho(rho_ph)

rho_ph, b_crit = photon_sphere()

# Outer turning point ρ0 for a given b (robust bracketing + bisection)
def rho0_from_b(b_target):
    B = mp.mpf(b_target)
    if B < b_crit:
        return rho_ph, True

    def g(r): 
        return b_of_rho(r) - B

    # start with an outer-branch estimate
    s_asym = max(RHO_MIN*(1+mp.mpf('1e-6')), B - 2*m_eff)
    lo = s_asym
    hi = B + 10*m_eff
    glo = g(lo)
    ghi = g(hi)

    # expand bracket safely
    for _ in range(100):
        if glo*ghi <= 0:
            break
        lo = max(RHO_MIN*(1+mp.mpf('1e-6')), lo*mp.mpf('0.9'))
        hi = hi*mp.mpf('1.3')
        glo, ghi = g(lo), g(hi)
    else:
        # fallback if no sign change found
        root = mp.findroot(g, (s_asym, hi))
        return root, False

    # now we have a valid bracket (lo, hi)
    A, C = lo, hi
    for _ in range(200):
        M = (A+C)/2
        gM = g(M)
        if mp.fabs(gM) < mp.mpf('1e-28')*(1+mp.fabs(B)):
            return M, False
        if g(A)*gM <= 0:
            C = M
        else:
            A = M
    return (A+C)/2, False

# Deflection integral with near-turn substitution and far-tail u=1/ρ
def deflection_from_rho0(rho0):
    rho0 = mp.mpf(rho0)
    b = b_of_rho(rho0)

    def rad(r):
        nr = n_operational(r)
        return nr*nr - (b*b)/(r*r)

    n0  = n_operational(rho0)
    dn0 = dn_drho(rho0)
    radp0 = 2*n0*dn0 + 2*(b*b)/(rho0**3)
    if radp0 <= mp.mpf('1e-80'):
        radp0 = mp.mpf('1e-80')
    L0 = 2*b/(rho0**2 * mp.sqrt(radp0))

    split = max(8*rho0, SPLIT_FAR)
    s_max = mp.sqrt(split - rho0)

    def f_near(s):
        if s < mp.mpf('1e-12'):
            return L0
        r = rho0 + s*s
        v = rad(r)
        if v < 0 and mp.fabs(v) < mp.mpf('1e-60'):
            v = mp.mpf('0')
        elif v < 0:
            return mp.mpf('0')
        return (b/(r*r*mp.sqrt(v))) * (2*s)

    I_near = mp.quad(f_near, [mp.mpf('0'), s_max])

    def f_tail(u):
        r = 1/u
        v = rad(r)
        if v < 0 and mp.fabs(v) < mp.mpf('1e-60'):
            v = mp.mpf('0')
        elif v < 0:
            return mp.mpf('0')
        if v == 0:
            return mp.mpf('0')
        return - b/mp.sqrt(v)

    I_tail = mp.quadts(f_tail, [1/split, mp.mpf('0')])

    delta = 2*(I_near + I_tail) - mp.pi
    return mp.re(delta)

def delta_of_b(b):
    rho0, captured = rho0_from_b(b)
    if captured:
        return mp.nan  # use NaN for plotting (masked)
    return deflection_from_rho0(rho0)

# ==========================
# 2) Plotting
# ==========================
def plot_deflection_curve(b_max=mp.mpf('25'), N=60, start_factor=mp.mpf('1.03'), log_axes=False):
    b_start = b_crit * start_factor  # step away from b_crit to ensure first point is finite
    b_vals = np.logspace(np.log10(float(b_start)), np.log10(float(b_max)), int(N))
    b_vals_mp = [mp.mpf(x) for x in b_vals]

    # quick sanity prints so you can see we’re computing real values
    for btest in (mp.mpf('10'), mp.mpf('50')):
        print(f"δ({float(btest):.0f}) ≈ {float(delta_of_b(btest)):.6f} rad")

    deltas = []
    for b in b_vals_mp:
        try:
            deltas.append(float(delta_of_b(b)))
        except Exception:
            deltas.append(np.nan)
    deltas = np.array(deltas)

    weak = (4*float(m_eff)) / b_vals

    finite = np.isfinite(deltas)
    print(f"Finite points: {finite.sum()}/{len(finite)}")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot only finite segments for the operational curve
    if finite.any():
        ax.plot(b_vals[finite], deltas[finite], label=r'Causal-Budget Model $\delta(b)$', linewidth=3)
    ax.plot(b_vals, weak, '--', label=r'Weak-Field Approx. $4m/b$', linewidth=2)
    ax.axvline(float(b_crit), color='r', linestyle=':', label=rf'Critical impact $b_{{\rm crit}}={float(b_crit):.2f}$')

    ax.set_title('Light Deflection: Causal-Budget Model vs. Weak-Field Approximation', fontsize=16)
    ax.set_xlabel(r'Impact Parameter $b$ ($\rho$ units)', fontsize=14)
    ax.set_ylabel(r'Deflection Angle $\delta(b)$ (radians)', fontsize=14)

    if log_axes:
        ax.set_xscale('log'); ax.set_yscale('log')

    if finite.any():
        ax.set_ylim(0, np.nanmax(deltas)*1.1)
    ax.set_xlim(float(b_crit) - 0.5, float(b_max))

    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle=':', alpha=0.7)

    out = 'deflection_vs_impact_parameter.png'
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved plot to {out}")

if __name__ == "__main__":
    plot_deflection_curve(log_axes=False)