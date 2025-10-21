# =========================================================
# Strong-field light bending via operational optical index n(ρ)
# =========================================================
#   n(ρ) = ((1 + r_s/(4ρ))**3) / (1 - r_s/(4ρ))
#
# Computes exact deflection δ(b) = 2 ∫ [b / (ρ² √(n² - b²/ρ²))] dρ − π
# and compares to the weak-field baseline δ ≈ 2 r_s / b.
# No geometry, no metric: purely operational medium n(ρ).
# =========================================================

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
import pandas as pd

mp.mp.dps = 100
r_s = mp.mpf(1)  # Schwarzschild radius units

# --- Operational definitions ---

def r_of_rho(rho):
    return rho * (1 + r_s/(4*rho))**2

def n_of_rho(rho):
    return (1 + r_s/(4*rho))**3 / (1 - r_s/(4*rho))

def b_from_rho0(rho0):
    return n_of_rho(rho0) * rho0

# --- Invert b → ρ0 (outer branch) ---

def rho0_from_b(b_target):
    """Return the OUTER turning point ρ0 solving n(ρ0)ρ0 = b_target."""
    B = mp.mpf(b_target)
    f = lambda rho: n_of_rho(rho)*rho - B
    rho_min = mp.mpf('0.26')  # just above the index pole

    low  = mp.mpf(max(rho_min, float(B) - 3.0))
    high = mp.mpf(float(B) + 3.0)

    it = 0
    while it < 80:
        f_low, f_high = f(low), f(high)
        if f_low < 0 and f_high > 0:
            break
        if f_low >= 0:
            low = mp.mpf(max(rho_min, 0.9 * low))
        if f_high <= 0:
            high = mp.mpf(1.3 * high)
        it += 1

    return mp.findroot(f, (low, high))

# --- Exact deflection integral (regularized) ---

def deflection_from_rho0(rho0):
    """Compute δ(b) = 2∫ b/(ρ²√(n² - b²/ρ²)) dρ − π, stabilized near ρ₀."""
    rho0 = mp.mpf(rho0)
    b    = b_from_rho0(rho0)

    def R(r):
        return n_of_rho(r)**2 - (b**2)/(r**2)

    # Adaptive offset from turning point (avoid singularity)
    dn0  = mp.diff(lambda x: n_of_rho(x), rho0)
    nr0  = n_of_rho(rho0)
    radp0 = 2*nr0*dn0 + 2*b**2/(rho0**3)
    eps = max(mp.mpf('1e-12')*rho0, mp.mpf('1e-10')/(abs(radp0)+mp.mpf('1e-40')))

    # Near-turn integral: ρ = ρ₀ + s²,  dρ = 2s ds
    def integrand_near(s):
        if s == 0:
            s = mp.mpf('1e-30')
        r = rho0 + s*s
        v = R(r)
        if v <= 0:
            v = mp.mpf('1e-40')
        return (b / (r*r*mp.sqrt(v))) * (2*s)

    split = max(mp.mpf('10'), mp.mpf('8')*rho0)
    s_max = mp.sqrt(split - rho0)
    I_near = mp.quad(integrand_near, [mp.mpf('0'), s_max])

    # Far tail: u = 1/ρ, dρ = -ρ² du ⇒ integrand = - b / √(n² - b²/ρ²)
    def integrand_tail(u):
        if u == 0:
            return mp.mpf('0')
        r = 1/u
        v = R(r)
        if v <= 0:
            v = mp.mpf('1e-40')
        return - b / mp.sqrt(v)

    I_tail = mp.quad(integrand_tail, [mp.mpf(1)/split, mp.mpf('0')])
    delta = 2*(I_near + I_tail) - mp.pi
    return mp.re(delta) if isinstance(delta, mp.mpc) else delta, b

# --- Photon sphere & critical impact parameter ---

rho_ph = mp.mpf('0.5') * (1 + mp.sqrt(3)/2)  # ≈ 0.9330127
b_crit = (3*mp.sqrt(3)/2) * r_s

print(f"Photon sphere: r_ph = 1.5 r_s,  rho_ph ≈ {rho_ph}")
print(f"Critical impact parameter: b_c = 3√3/2 r_s ≈ {b_crit}\n")

# --- Sampling plan ---
rho0_list_strong = [rho_ph*(1+eps) for eps in [0.40, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001]]
b_targets        = [5*r_s, 10*r_s, 20*r_s, 50*r_s, 100*r_s, 200*r_s]
rho0_list_weak   = [rho0_from_b(bt) for bt in b_targets]

# --- Run computations ---
rows = []

def add_row(rho0, tag):
    r0 = r_of_rho(rho0)
    delta, bval = deflection_from_rho0(rho0)
    wf = 2*r_s/bval if bval != 0 else mp.mpf('nan')
    rows.append([
        str(tag),
        float(rho0),
        float(r0),
        float(bval),
        float(delta) if mp.isfinite(delta) else np.nan,
        float(delta*180/np.pi) if mp.isfinite(delta) else np.nan,
        float(wf)
    ])

print("Computing strong-field points...")
for rho0 in rho0_list_strong:
    add_row(rho0, f"strong rho0/rho_ph={float(rho0/rho_ph):.3f}")

print("Computing weak-field points...")
for rho0 in rho0_list_weak:
    add_row(rho0, "weak by b")

df = pd.DataFrame(rows, columns=[
    "case",
    "rho0 (isotropic)",
    "r0 (areal)",
    "b",
    "delta (rad)",
    "delta (deg)",
    "weak-field baseline 2/b (rad)"
])

print("\nResults (r_s = 1 units):\n")
print(df.to_string(index=False))

# --- Weak-field sanity checks ---
print("\nSanity checks (weak field comparisons):")
for btest in [50, 100, 200]:
    rho0 = rho0_from_b(btest)
    delta, _ = deflection_from_rho0(rho0)
    print(f"  b={float(btest):.0f}: exact δ={float(delta):.5f}, 2/b={2/float(btest):.5f}, diff={float(delta - 2/btest):.2e}")

# --- Plot results ---
b_smooth = np.logspace(np.log10(float(b_crit)+1e-3), np.log10(200), 200)
weak_field_GR_smooth = 2 / b_smooth  # baseline 2/b since r_s=1

mask_valid = np.isfinite(df["delta (rad)"].values)
x  = df.loc[mask_valid, "b"].values
y  = df.loc[mask_valid, "delta (rad)"].values
y0 = df.loc[mask_valid, "weak-field baseline 2/b (rad)"].values

plt.figure(figsize=(8,6))
plt.plot(b_smooth, weak_field_GR_smooth, '--', label='Weak-field baseline (2/b)', linewidth=2)
plt.plot(x, y, 'o-', label='Exact (operational index)', linewidth=2, markersize=6)
plt.plot(x, y0, 's--', label='2/b evaluated at computed b', markersize=4)
plt.axvline(float(b_crit), color='k', linestyle=':', linewidth=2,
            label=f'Photon sphere  b_c = {float(b_crit):.3f}')
plt.xlabel('Impact parameter  b / r_s')
plt.ylabel('Deflection angle  δ (rad)')
plt.title('Deflection vs Impact Parameter: Strong → Weak Field\n(Operational index vs weak-field baseline)')
plt.xscale('log'); plt.yscale('log'); plt.grid(True, alpha=0.3); plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# --- Summary comparisons ---
print("\n" + "="*60)
print("KEY COMPARISONS:")
print("="*60)
for _, row in df.iterrows():
    if np.isfinite(row['delta (rad)']):
        exact = row['delta (rad)']
        base  = row['weak-field baseline 2/b (rad)']
        ratio = exact / base if base > 0 else float('inf')
        print(f"b = {row['b']:.3f}: Exact = {exact:.5f} rad, Baseline = {base:.5f} rad, Ratio = {ratio:.3f}")

print("\nExpected behavior:")
print("- b >> b_c: Ratio → 1 (weak-field agreement)")
print("- b → b_c: Ratio → ∞ (strong-field divergence)")
print("="*60)