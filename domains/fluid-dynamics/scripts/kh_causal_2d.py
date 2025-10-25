# kh_causal_2d.py
# 2-D Kelvin–Helmholtz shear layer with causal-capacity projection
# Periodic box [0,2π]^2, pseudo-spectral vorticity–streamfunction

import os, csv, math, time
import numpy as np, os
import numpy.fft as fft
import matplotlib.pyplot as plt

# -------------------------------
# Utilities
# -------------------------------
def dealias_mask(N):
    kcut = int(np.floor(N/3))
    m = np.zeros((N,N), dtype=bool)
    m[:kcut,:kcut] = True
    m[:kcut,-kcut:] = True
    m[-kcut:,:kcut] = True
    m[-kcut:,-kcut:] = True
    return m

def shell_average(kx, ky, Ahat):
    # isotropic 1D spectrum E(k) from |Û|^2/2 shell average
    k = np.sqrt(kx**2 + ky**2)
    kmax = int(np.max(k))
    bins = np.arange(0.5, kmax+1.5, 1.0)
    E = np.zeros(len(bins)-1)
    counts = np.zeros_like(E)
    spec_density = 0.5*np.abs(Ahat)**2
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            kk = k[i,j]
            b = int(np.floor(kk))
            if b>=0 and b<len(E):
                E[b] += spec_density[i,j]
                counts[b] += 1
    E[counts>0] /= counts[counts>0]
    kcenters = np.arange(1, len(E)+1)
    return kcenters, E

def energy_spectrum_2d(u, v):
    # FFT (unit box – absolute scaling doesn’t affect slope)
    U = np.fft.rfftn(u); V = np.fft.rfftn(v)
    kx = np.fft.fftfreq(u.shape[0])[:,None]
    ky = np.fft.rfftfreq(u.shape[1])[None,:]
    kk = np.sqrt((kx**2 + ky**2))
    E2D = 0.5*(np.abs(U)**2 + np.abs(V)**2)

    # shell average
    kmax = int(np.ceil(np.max(kk)*min(u.shape)))
    bins = np.arange(0, kmax+1)
    which = np.digitize(kk.flatten()*min(u.shape), bins)  # simple binning
    Ek = np.bincount(which, weights=E2D.flatten())
    Nk = np.bincount(which)
    valid = (Nk > 0)
    k_shell = bins[valid]
    Ek_1d = Ek[valid]/Nk[valid]
    return k_shell, Ek_1d

# -------------------------------
# Spectral grid
# -------------------------------
class SpectralGrid2D:
    def __init__(self, N=256, Lx=2*np.pi, Ly=2*np.pi):
        self.N = N
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx/N, Ly/N
        x = np.arange(N)*self.dx
        y = np.arange(N)*self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        kx = fft.fftfreq(N, d=self.dx/(2*np.pi))
        ky = fft.fftfreq(N, d=self.dy/(2*np.pi))
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.k2 = self.KX**2 + self.KY**2
        self.k2[0,0] = 1.0 # avoids division by zero; set psi_hat[0,0]=0 later
        self.dealias = dealias_mask(N)

# -------------------------------
# KH initial condition
# -------------------------------
def init_kh(grid, delta=0.05, U0=1.0, noise=1e-3, seed=1):
    """
    Hyperbolic-tangent shear layer aligned with x. Periodic in y via double-layer construction.
    u_x(y) = U0 * tanh((y - Ly/4)/delta) - U0 * tanh((y - 3Ly/4)/delta)
    u_y = small sinusoidal perturbation + random noise
    """
    np.random.seed(seed)
    X, Y, Ly = grid.X, grid.Y, grid.Ly
    yy1 = (Y - 0.25*Ly)/delta
    yy2 = (Y - 0.75*Ly)/delta
    ux = U0*np.tanh(yy1) - U0*np.tanh(yy2)
    uy = 0.05*U0*np.sin(2*X) + noise*(np.random.randn(*X.shape))
    # streamfunction to make divergence-free initial field
    # curl(psi) = u => solve ∇²ψ = ω with ω = ∂v/∂x - ∂u/∂y from (ux,uy)
    uy_x = np.gradient(uy, grid.dx, axis=0)
    ux_y = np.gradient(ux, grid.dy, axis=1)
    omega0 = uy_x - ux_y
    return omega0

# -------------------------------
# Core solver
# -------------------------------
class KH_Causal2D:
    def __init__(self, N=256, Re=20000.0, c=1.0, dt=None, T=10.0,
                 causal_on=True, theta=1.0, outdir="kh_out"):
        self.grid = SpectralGrid2D(N=N)
        self.N = N
        self.nu = 1.0/Re
        self.c = c
        self.theta = theta       # v_int^2 = theta * nu * |S|
        self.causal_on = causal_on
        self.T = T
        # CFL-based dt if not provided
        self.dt = dt if dt is not None else 0.3*min(self.grid.dx, self.grid.dy)
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "diagnostics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t","E","Z","eff","max_usage","viol_pct"])

    # Poisson invert: ∇²ψ = -ω (note sign convention for streamfunction)
    def poisson_invert(self, omega_hat):
        psi_hat = -omega_hat / self.grid.k2
        psi_hat[0,0] = 0.0
        return psi_hat

    def velocity_from_psi(self, psi_hat):
        # u = +∂ψ/∂y,  v = -∂ψ/∂x  (so that ∇²ψ = −ω)
        u_hat =  1j * self.grid.KY * psi_hat
        v_hat = -1j * self.grid.KX * psi_hat
        u = np.real(fft.ifft2(u_hat))
        v = np.real(fft.ifft2(v_hat))
        return u, v, u_hat, v_hat

    def strain_magnitude(self, u, v):
        ux = np.gradient(u, self.grid.dx, axis=0)
        uy = np.gradient(u, self.grid.dy, axis=1)
        vx = np.gradient(v, self.grid.dx, axis=0)
        vy = np.gradient(v, self.grid.dy, axis=1)
        S11 = ux
        S22 = vy
        S12 = 0.5*(uy + vx)
        # |S| = sqrt(2 S:S) = sqrt(2(S11^2 + S22^2 + 2 S12^2))
        return np.sqrt(2.0*(S11**2 + S22**2 + 2.0*S12**2))

    def causal_project(self, u, v):
        if not self.causal_on:
            return u, v, np.zeros_like(u), 0.0, 0.0
        # compute v_int from strain magnitude
        Smag = self.strain_magnitude(u, v)
        v_int_sq = self.theta * self.nu * Smag
        speed_sq = u**2 + v**2
        total = speed_sq + v_int_sq
        # scale factor s(x)
        cap = np.maximum(0.0, self.c**2 - v_int_sq)
        s2 = np.where(total > self.c**2, cap/np.maximum(speed_sq, 1e-30), 1.0)
        s = np.sqrt(np.clip(s2, 0.0, 1.0))
        up = s*u
        vp = s*v
        # diagnostics
        viol = (total > self.c**2)
        viol_pct = 100.0*np.count_nonzero(viol)/viol.size
        max_usage = float(np.max(total/self.c**2))
        eff = float(np.mean(speed_sq / np.maximum(total,1e-30)))  # causal efficiency
        return up, vp, v_int_sq, max_usage, viol_pct

    def leray_project(self, u, v):
        # Project to divergence-free in Fourier space
        uh = fft.fft2(u); vh = fft.fft2(v)
        kx = self.grid.KX; ky = self.grid.KY; k2 = self.grid.k2
        # û_perp = û - k (k·û)/|k|^2
        dot = kx*uh + ky*vh
        uhp = uh - kx*dot/k2
        vhp = vh - ky*dot/k2
        uhp[0,0] = 0; vhp[0,0] = 0
        u_perp = np.real(fft.ifft2(uhp))
        v_perp = np.real(fft.ifft2(vhp))
        return u_perp, v_perp, uhp, vhp

    def curl_from_velocity(self, u, v):
        vx = np.gradient(v, self.grid.dx, axis=0)
        uy = np.gradient(u, self.grid.dy, axis=1)
        return vx - uy

    def nonlinear_term(self, omega, u, v):
        dwdx = np.gradient(omega, self.grid.dx, axis=0)
        dwdy = np.gradient(omega, self.grid.dy, axis=1)
        return -(u*dwdx + v*dwdy)

    def step(self, omega_hat):
        # Compute psi, u, v
        psi_hat = self.poisson_invert(omega_hat)
        u, v, uh, vh = self.velocity_from_psi(psi_hat)

        # Nonlinear term (physical space), dealiased
        NL = self.nonlinear_term(np.real(fft.ifft2(omega_hat)), u, v)
        NLh = fft.fft2(NL)
        NLh[~self.grid.dealias] = 0.0

        # Semi-implicit diffusion: ω^{n+1/2} (Heun/RK2)
        dt = self.dt; nu = self.nu; k2 = self.grid.k2
        w_tilde = (omega_hat + dt*NLh) / (1.0 + dt*nu*k2)

        # Causal projection on velocity from w_tilde:
        psi_hat_t = self.poisson_invert(w_tilde)
        ut, vt, _, _ = self.velocity_from_psi(psi_hat_t)
        up, vp, v_int_sq, max_usage, viol_pct = self.causal_project(ut, vt)
        # Leray projection to restore incompressibility
        up, vp, uhp, vhp = self.leray_project(up, vp)
        # Enforce curl-consistency for next step
        w_proj = self.curl_from_velocity(up, vp)
        w_hat_proj = fft.fft2(w_proj)
        w_hat_proj[~self.grid.dealias] = 0.0

        # Second stage (Heun): recompute NL at projected state
        psi_hat2 = self.poisson_invert(w_hat_proj)
        u2, v2, _, _ = self.velocity_from_psi(psi_hat2)
        NL2 = self.nonlinear_term(np.real(fft.ifft2(w_hat_proj)), u2, v2)
        NL2h = fft.fft2(NL2); NL2h[~self.grid.dealias] = 0.0

        # Final update (Heun average) with semi-implicit diffusion
        w_next = (omega_hat + 0.5*dt*(NLh + NL2h)) / (1.0 + dt*nu*k2)
        # Replace by projected curl to enforce constraint strongly
        omega_hat = w_hat_proj

        # Diagnostics
        E = 0.5*np.mean(up**2 + vp**2)
        Z = 0.5*np.mean(np.real(fft.ifft2(omega_hat))**2)
        eff = float(np.mean((up**2+vp**2) / np.maximum(up**2+vp**2+v_int_sq,1e-30)))
        return omega_hat, E, Z, eff, max_usage, viol_pct

    def run(self, omega0, save_every=50, plot_every=200):
        omega_hat = fft.fft2(omega0); omega_hat[~self.grid.dealias] = 0.0
        t=0.0; nsteps = int(self.T/self.dt); t0 = time.time()

        for n in range(nsteps+1):
            if n%save_every==0 or n==nsteps:
                # Snapshot & spectrum
                psi_hat = self.poisson_invert(omega_hat)
                u, v, uh, vh = self.velocity_from_psi(psi_hat)
                w = np.real(fft.ifft2(omega_hat))
                # spectra of velocity (uses both components)
                k, E1D = energy_spectrum_2d(u, v)
                np.savez(os.path.join(self.outdir, f"spec_{n:06d}.npz"), k=k, Ek=E1D, t=t)

                # plot
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                im = ax[0].imshow(w.T, origin='lower', extent=(0, 2*np.pi, 0, 2*np.pi),
                                cmap='RdBu', vmin=-5, vmax=5)
                ax[0].set_title(f"Vorticity at t={t:.2f}")
                plt.colorbar(im, ax=ax[0])

                kplt = k.astype(float)  # avoid “integers to negative powers” issues
                ax[1].loglog(kplt[1:], E1D[1:] + 1e-20)
                ax[1].loglog(kplt[1:], kplt[1:]**(-3.0), ls='--', label=r"$k^{-3}$")
                ax[1].set_title("Velocity spectrum"); ax[1].legend()

                plt.tight_layout()
                fig.savefig(os.path.join(self.outdir, f"kh_{n:06d}.png"), dpi=150)
                plt.close(fig)

            omega_hat, E, Z, eff, max_usage, viol_pct = self.step(omega_hat)
            # log
            with open(os.path.join(self.outdir,"diagnostics.csv"), "a", newline="") as f:
                csv.writer(f).writerow([t, E, Z, eff, max_usage, viol_pct])
            t += self.dt

        print(f"Done in {time.time()-t0:.1f}s. Output -> {self.outdir}")
        return
# -------------------------------
# Run script
# -------------------------------
if __name__ == "__main__":
    N   = 256          # 384 for richer scales (slower)
    Re  = 20000.0      # high-Re shear layer
    c   = 0.8          # capacity (U0 nondim=1) — try 0.6..1.0
    T   = 12.0         # total time
    dt  = None         # CFL-based if None
    out = "kh_causal_out"

    sim = KH_Causal2D(N=N, Re=Re, c=c, dt=dt, T=T, causal_on=True, theta=1.0, outdir=out)
    omega0 = init_kh(sim.grid, delta=0.05, U0=1.0, noise=1e-3, seed=2)
    sim.run(omega0, save_every=50, plot_every=200)