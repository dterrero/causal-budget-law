import numpy as np
import matplotlib.pyplot as plt

# -------------------- Parameters --------------------
N = 64                 # grid points per side
L = 2*np.pi            # domain size
nu = 5e-4              # viscosity
dt = 1e-3              # time step
steps = 1000           # total steps
save_every = 2         # diagnostics cadence
P_target = 2e-3        # target power injection (per unit volume)
k_forc_max = 2.5       # force only low-k shell (0<|k|<=k_forc_max)
np.random.seed(2)

# -------------------- Grids & wave numbers --------------------
x = np.linspace(0, L, N, endpoint=False)
k1 = np.fft.fftfreq(N, d=L/N) * 2*np.pi
kx, ky, kz = np.meshgrid(k1, k1, k1, indexing='ij')
k2 = kx**2 + ky**2 + kz**2
k = np.sqrt(k2)
k2[0,0,0] = 1.0  # avoid divide-by-zero in projectors

# 2/3-rule dealiasing mask
kmax = np.max(np.abs(k1))
dealias = (np.abs(kx) <= (2/3)*kmax) & (np.abs(ky) <= (2/3)*kmax) & (np.abs(kz) <= (2/3)*kmax)

# Low-k forcing mask (exclude k=0)
lowk = (k > 0) & (k <= k_forc_max)

# -------------------- Helpers --------------------
def project_div_free(vhat):
    """Helmholtz projection in Fourier space: (I - kk^T/k^2) vhat."""
    kdotv = kx*vhat[...,0] + ky*vhat[...,1] + kz*vhat[...,2]
    proj = np.empty_like(vhat)
    proj[...,0] = vhat[...,0] - kx * kdotv / k2
    proj[...,1] = vhat[...,1] - ky * kdotv / k2
    proj[...,2] = vhat[...,2] - kz * kdotv / k2
    return proj

def curl_hat(vhat):
    """Fourier curl of v (returns hat-space curl)."""
    return 1j*np.stack([
        ky*vhat[...,2] - kz*vhat[...,1],
        kz*vhat[...,0] - kx*vhat[...,2],
        kx*vhat[...,1] - ky*vhat[...,0]
    ], axis=-1)

def nonlinear_term(u_hat):
    """Compute N_hat = FFT[(u · ∇)u] with 2/3 dealiasing."""
    u = np.real(np.fft.ifftn(u_hat, axes=(0,1,2)))
    # gradients in physical space
    dudx = np.gradient(u, L/N, axis=0)
    dudy = np.gradient(u, L/N, axis=1)
    dudz = np.gradient(u, L/N, axis=2)
    # (u · ∇)u
    adv = u[...,0][...,None]*dudx + u[...,1][...,None]*dudy + u[...,2][...,None]*dudz
    N_hat = np.fft.fftn(adv, axes=(0,1,2))
    # dealiased and projected (nonlinear term is not div-free)
    N_hat *= dealias[...,None]
    N_hat = project_div_free(N_hat)
    return N_hat

# -------------------- Initial condition (divergence-free random low-k) --------------------
def random_divfree_lowk():
    amp = np.random.randn(N,N,N,3) + 1j*np.random.randn(N,N,N,3)
    amp *= np.exp(-(k/6.0)**2)[...,None]      # concentrate at low k
    amp *= lowk[...,None]
    amp = project_div_free(amp)
    amp *= dealias[...,None]
    return amp

u_hat = random_divfree_lowk()

# -------------------- Constant-power forcing (feedback) --------------------
def forcing_hat(u_hat, P_target):
    """Low-k, constant-power forcing: f = alpha * b, with b = P_lowk(u)."""
    b = np.zeros_like(u_hat)
    b[lowk] = u_hat[lowk]
    b = project_div_free(b)
    b *= dealias[...,None]
    # In physical space, <u·b> = <b·b> when b is the low-k projection of u
    b_phys = np.real(np.fft.ifftn(b, axes=(0,1,2)))
    bb = np.mean(np.sum(b_phys**2, axis=-1))
    alpha = P_target / max(bb, 1e-14)
    return alpha * b

# -------------------- Diagnostics --------------------
def energy(u_hat):
    u = np.real(np.fft.ifftn(u_hat, axes=(0,1,2)))
    return 0.5*np.mean(np.sum(u**2, axis=-1))

def power(u_hat, f_hat):
    u = np.real(np.fft.ifftn(u_hat, axes=(0,1,2)))
    f = np.real(np.fft.ifftn(f_hat, axes=(0,1,2)))
    return np.mean(np.sum(u*f, axis=-1))

def enstrophy(u_hat):
    c_hat = curl_hat(u_hat)
    return nu * np.mean(np.sum(np.abs(c_hat)**2, axis=-1))

# -------------------- Time stepping (RK2) --------------------
E_list, P_list, eps_list, tau_star, t = [], [], [], [], []
bb_filt = 0.0
beta = 0.1            # low-pass on bb to avoid huge alpha at start
alpha_cap = 10.0      # safety cap on forcing gain

for n in range(steps):
    # ----- stage-0 forcing with filtered power denominator -----
    # build base b as before
    b = np.zeros_like(u_hat)
    b[lowk] = u_hat[lowk]
    b = project_div_free(b); b *= dealias[...,None]
    b_phys = np.real(np.fft.ifftn(b, axes=(0,1,2)))
    bb = np.mean(np.sum(b_phys**2, axis=-1))
    bb_filt = (1-beta)*bb_filt + beta*bb
    alpha = P_target / max(bb_filt, 1e-14)
    alpha = np.clip(alpha, 0.0, alpha_cap)
    f_hat_n = alpha * b

    # ----- stage-0 diagnostics (use u^n with f^n) -----
    u_n = np.real(np.fft.ifftn(u_hat, axes=(0,1,2)))
    f_n = np.real(np.fft.ifftn(f_hat_n, axes=(0,1,2)))
    E_n  = 0.5*np.mean(np.sum(u_n**2, axis=-1))
    P_n  = np.mean(np.sum(u_n*f_n, axis=-1))
    eps_n = nu * np.mean(np.sum(np.abs(curl_hat(u_hat))**2, axis=-1))

    # ---- optional subsample: log ONCE, not twice ----
    if n % save_every == 0:
        E_list.append(E_n); P_list.append(P_n); eps_list.append(eps_n)
        tau_star.append(E_n/abs(P_n) if abs(P_n) > 1e-14 else np.nan)
        t.append(n*dt)

    # ----- stage 1 -----
    N1 = nonlinear_term(u_hat)
    rhs1 = f_hat_n - N1 - nu*k2[...,None]*u_hat
    uh1  = project_div_free(u_hat + dt*rhs1); uh1 *= dealias[...,None]

    # ----- stage 2 (midpoint forcing, reuse same filter) -----
    b1 = np.zeros_like(uh1); b1[lowk] = uh1[lowk]
    b1 = project_div_free(b1); b1 *= dealias[...,None]
    b1_phys = np.real(np.fft.ifftn(b1, axes=(0,1,2)))
    bb1 = np.mean(np.sum(b1_phys**2, axis=-1))
    bb_filt = (1-beta)*bb_filt + beta*bb1
    alpha1 = np.clip(P_target / max(bb_filt, 1e-14), 0.0, alpha_cap)
    f_hat_1 = alpha1 * b1

    N2 = nonlinear_term(uh1)
    rhs2 = f_hat_1 - N2 - nu*k2[...,None]*uh1
    u_hat = project_div_free(u_hat + 0.5*dt*(rhs1 + rhs2)); u_hat *= dealias[...,None]

# --------- Cumulatives (use the effective logging step) ---------
t       = np.array(t)
E_arr   = np.array(E_list)
P_arr   = np.array(P_list)
eps_arr = np.array(eps_list)
tau_arr = np.array(tau_star)

dt_eff = save_every*dt
cum_eps      = np.cumsum(eps_arr)      * dt_eff
cum_PoverE   = np.cumsum(np.nan_to_num(P_arr/E_arr)) * dt_eff


# -------------------- Plots --------------------
plt.figure(figsize=(6.4,4.8))
plt.plot(t, eps_arr/np.max(eps_arr), label='Enstrophy (normalized)')
plt.plot(t, np.nan_to_num(1/tau_arr)/np.nanmax(np.nan_to_num(1/tau_arr)),
         '--', label='1/τ* = P/E (normalized)')
plt.xlabel('t'); plt.legend(); plt.grid(); plt.tight_layout()
plt.show()

plt.figure(figsize=(6.4,4.8))
plt.scatter(np.nan_to_num(1/tau_arr), eps_arr, s=6)
plt.xlabel('1/τ* = P/E'); plt.ylabel('ε = ν⟨|∇×u|²⟩')
plt.title('Instantaneous correlation'); plt.grid(); plt.tight_layout()
plt.show()

plt.figure(figsize=(6.4,4.8))
plt.plot(t, cum_eps/np.max(cum_eps), label='∫ ε dt (normalized)')
plt.plot(t, cum_PoverE/np.max(cum_PoverE), '--', label='∫ (P/E) dt (normalized)')
plt.xlabel('t'); plt.ylabel('Cumulative normalized integrals')
plt.title('Causal-Budget Integral Relation'); plt.legend(); plt.grid(); plt.tight_layout()
plt.show()
