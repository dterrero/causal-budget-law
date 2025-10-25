#!/usr/bin/env python3
# kh_ccns_les_benchmark_v2.py
# Quantitative benchmark: CCNS vs NSE on 2-D KH with improved stability and late-time stats.
#
# Key upgrades:
# - Smaller default CFL for stability (dt = cfl * min(dx,dy), default cfl=0.15)
# - Separate CFL for control (--cfl_control) to keep NSE stable if needed
# - Extended capacity sweep to c in [0.6,0.7,0.8,0.9,1.0,1.2,1.5,2.0]
# - Late-time averaging window (--avg_frac, default last 1/3 of the run)
#
# Outputs per case: diagnostics.csv, slope.csv, spec_*.npz, kh_*.png (snapshots)
# Summary: bench_out/benchmark_summary.csv
#
# Usage example:
#   python kh_ccns_les_benchmark_v2.py --control --T 12 --N 256 --Re 20000 \
#       --sweep_c 0.6 0.7 0.8 0.9 1.0 1.2 1.5 2.0 --avg_frac 0.33
#
import os, csv, math, time, argparse
import numpy as np
import numpy.fft as fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def dealias_mask(N):
    kcut = int(np.floor(N/3))
    m = np.zeros((N,N), dtype=bool)
    m[:kcut,:kcut] = True
    m[:kcut,-kcut:] = True
    m[-kcut:,:kcut] = True
    m[-kcut:,-kcut:] = True
    return m

def energy_spectrum_2d(u, v):
    U = np.fft.rfftn(u); V = np.fft.rfftn(v)
    kx = np.fft.fftfreq(u.shape[0])[:,None]
    ky = np.fft.rfftfreq(u.shape[1])[None,:]
    kk = np.sqrt((kx**2 + ky**2))
    E2D = 0.5*(np.abs(U)**2 + np.abs(V)**2)
    kmax = int(np.ceil(np.max(kk)*min(u.shape)))
    bins = np.arange(0, kmax+1)
    which = np.digitize(kk.flatten()*min(u.shape), bins)
    Ek = np.bincount(which, weights=E2D.flatten())
    Nk = np.bincount(which)
    valid = (Nk > 0)
    k_shell = bins[valid]
    Ek_1d = Ek[valid]/Nk[valid]
    return k_shell.astype(float), Ek_1d

def fit_slope_loglog(k, Ek, k1=6, k2=40, min_pts=5):
    sel = (k >= k1) & (k <= k2) & (Ek > 0)
    if np.count_nonzero(sel) < min_pts:
        return np.nan
    x = np.log(k[sel]); y = np.log(Ek[sel])
    n, _ = np.polyfit(x, y, 1)
    return n

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
        self.k2[0,0] = 1.0
        self.dealias = dealias_mask(N)

def init_kh(grid, delta=0.05, U0=1.0, noise=1e-3, seed=2):
    np.random.seed(seed)
    X, Y, Ly = grid.X, grid.Y, grid.Ly
    yy1 = (Y - 0.25*Ly)/delta
    yy2 = (Y - 0.75*Ly)/delta
    ux = U0*np.tanh(yy1) - U0*np.tanh(yy2)
    uy = 0.05*U0*np.sin(2*X) + noise*(np.random.randn(*X.shape))
    uy_x = np.gradient(uy, grid.dx, axis=0)
    ux_y = np.gradient(ux, grid.dy, axis=1)
    omega0 = uy_x - ux_y
    return omega0

class KH_CCNS_2D:
    def __init__(self, N=256, Re=2e4, c=0.8, theta=1.0, cfl=0.15, dt=None, T=12.0,
                 causal_on=True, outdir="bench_out"):
        self.grid = SpectralGrid2D(N=N)
        self.N = N
        self.nu = 1.0/float(Re)
        self.c = float(c)
        self.theta = float(theta)
        self.causal_on = bool(causal_on)
        self.T = float(T)
        base_dt = cfl*min(self.grid.dx, self.grid.dy)
        self.dt = dt if dt is not None else base_dt
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "diagnostics.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t","E","Z","eff","max_usage","viol_pct","dE_proj"])

    def poisson_invert(self, omega_hat):
        psi_hat = -omega_hat / self.grid.k2
        psi_hat[0,0] = 0.0
        return psi_hat

    def velocity_from_psi(self, psi_hat):
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
        S11 = ux; S22 = vy; S12 = 0.5*(uy + vx)
        return np.sqrt(2.0*(S11**2 + S22**2 + 2.0*S12**2))

    def causal_project(self, u, v):
        if not self.causal_on:
            z = np.zeros_like(u); s = np.ones_like(u)
            return u, v, z, 1.0, 0.0, s, 0.0
        Smag = self.strain_magnitude(u, v)
        v_int_sq = self.theta * self.nu * Smag
        speed_sq = u**2 + v**2
        total = speed_sq + v_int_sq
        cap = np.maximum(0.0, self.c**2 - v_int_sq)
        s2 = np.where(total > self.c**2, cap/np.maximum(speed_sq, 1e-30), 1.0)
        s  = np.sqrt(np.clip(s2, 0.0, 1.0))
        up = s*u; vp = s*v
        viol = (total > self.c**2)
        viol_pct = 100.0*np.count_nonzero(viol)/viol.size
        max_usage = float(np.max(total/self.c**2))
        eff = float(np.mean(speed_sq / np.maximum(total,1e-30)))
        dE_proj = 0.5*np.mean((1.0 - s2) * speed_sq)
        return up, vp, v_int_sq, max_usage, viol_pct, s, dE_proj

    def leray_project(self, u, v):
        uh = fft.fft2(u); vh = fft.fft2(v)
        kx = self.grid.KX; ky = self.grid.KY; k2 = self.grid.k2
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
        psi_hat = self.poisson_invert(omega_hat)
        u, v, uh, vh = self.velocity_from_psi(psi_hat)

        NL = self.nonlinear_term(np.real(fft.ifft2(omega_hat)), u, v)
        NLh = fft.fft2(NL); NLh[~self.grid.dealias] = 0.0

        dt = self.dt; nu = self.nu; k2 = self.grid.k2
        w_tilde = (omega_hat + dt*NLh) / (1.0 + dt*nu*k2)

        psi_hat_t = self.poisson_invert(w_tilde)
        ut, vt, _, _ = self.velocity_from_psi(psi_hat_t)
        up, vp, v_int_sq, max_usage, viol_pct, s, dE_proj = self.causal_project(ut, vt)
        up, vp, uhp, vhp = self.leray_project(up, vp)
        w_proj = self.curl_from_velocity(up, vp)
        w_hat_proj = fft.fft2(w_proj); w_hat_proj[~self.grid.dealias] = 0.0

        psi_hat2 = self.poisson_invert(w_hat_proj)
        u2, v2, _, _ = self.velocity_from_psi(psi_hat2)
        NL2 = self.nonlinear_term(np.real(fft.ifft2(w_hat_proj)), u2, v2)
        NL2h = fft.fft2(NL2); NL2h[~self.grid.dealias] = 0.0

        _ = (omega_hat + 0.5*dt*(NLh + NL2h)) / (1.0 + dt*nu*k2)  # Heun value (unused under strong enforcement)
        omega_hat = w_hat_proj

        E = 0.5*np.mean(up**2 + vp**2)
        Z = 0.5*np.mean(np.real(fft.ifft2(omega_hat))**2)
        eff = float(np.mean((up**2+vp**2) / np.maximum(up**2+vp**2+v_int_sq,1e-30)))
        return omega_hat, E, Z, eff, max_usage, viol_pct, dE_proj

    def run(self, omega0, k1=6, k2=40, save_every=50, plot_every=200):
        omega_hat = fft.fft2(omega0); omega_hat[~self.grid.dealias] = 0.0
        t=0.0; nsteps = int(self.T/self.dt)
        slope_file = os.path.join(self.outdir, "slope.csv")
        if os.path.exists(slope_file): os.remove(slope_file)

        for n in range(nsteps+1):
            if n%save_every==0 or n==nsteps:
                psi_hat = self.poisson_invert(omega_hat)
                u, v, uh, vh = self.velocity_from_psi(psi_hat)
                w = np.real(fft.ifft2(omega_hat))
                k, E1D = energy_spectrum_2d(u, v)
                np.savez(os.path.join(self.outdir, f"spec_{n:06d}.npz"), k=k, Ek=E1D, t=t)
                nfit = fit_slope_loglog(k, E1D, k1=k1, k2=k2)
                with open(slope_file, "a", newline="") as f:
                    csv.writer(f).writerow([t, nfit])

                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                im = ax[0].imshow(w.T, origin='lower', extent=(0, 2*np.pi, 0, 2*np.pi),
                                  cmap='RdBu', vmin=-5, vmax=5)
                ax[0].set_title(f"Vorticity at t={t:.2f}"); plt.colorbar(im, ax=ax[0])
                kplt = k.astype(float)
                ax[1].loglog(kplt[1:], E1D[1:] + 1e-20, label="E(k)")
                ax[1].loglog(kplt[1:], kplt[1:]**(-3.0), ls='--', label=r"$k^{-3}$")
                ax[1].set_title(f"Spectrum; slope~{nfit:.2f} on [{k1},{k2}]"); ax[1].legend()
                plt.tight_layout(); fig.savefig(os.path.join(self.outdir, f"kh_{n:06d}.png"), dpi=130); plt.close(fig)

            omega_hat, E, Z, eff, max_usage, viol_pct, dE_proj = self.step(omega_hat)
            with open(os.path.join(self.outdir,"diagnostics.csv"), "a", newline="") as f:
                csv.writer(f).writerow([t, E, Z, eff, max_usage, viol_pct, dE_proj])
            t += self.dt

def run_case(base_outdir, label, causal_on, c, theta, N, Re, T, cfl, delta=0.05, seed=2, k1=6, k2=40):
    outdir = os.path.join(base_outdir, label)
    os.makedirs(outdir, exist_ok=True)
    sim = KH_CCNS_2D(N=N, Re=Re, c=c, theta=theta, T=T, causal_on=causal_on, outdir=outdir, cfl=cfl)
    omega0 = init_kh(sim.grid, delta=delta, U0=1.0, noise=1e-3, seed=seed)
    sim.run(omega0, k1=k1, k2=k2)
    return outdir

def summarize(outdir, avg_frac=0.33):
    import pandas as pd, numpy as np, os
    diag_path = os.path.join(outdir, "diagnostics.csv")
    slope_path = os.path.join(outdir, "slope.csv")
    out = {"label": os.path.basename(outdir)}
    if os.path.exists(diag_path):
        d = pd.read_csv(diag_path)
        if "t" in d.columns and len(d):
            tmax = float(np.nanmax(d["t"]))
            tcut = tmax*(1.0 - float(avg_frac))
            d = d[d["t"] >= tcut] if np.isfinite(tcut) else d
        def safe_mean(col): return float(np.nanmean(d[col])) if col in d.columns and len(d[col]) else np.nan
        out.update(dict(
            E_mean=safe_mean("E"), Z_mean=safe_mean("Z"),
            eff_mean=safe_mean("eff"),
            viol_mean=safe_mean("viol_pct"),
            proj_dE_mean=safe_mean("dE_proj"),
        ))
    if os.path.exists(slope_path):
        s = pd.read_csv(slope_path, header=None, names=["t","n"])
        if len(s):
            tmax = float(np.nanmax(s["t"])); tcut = tmax*(1.0 - float(avg_frac))
            s = s[s["t"] >= tcut] if np.isfinite(tcut) else s
            out.update(dict(n_mean=float(np.nanmean(s["n"])), n_std=float(np.nanstd(s["n"]))))
    return out

def main():
    ap = argparse.ArgumentParser(description="CCNS vs NSE benchmark on 2-D KH (stabilized, late-time stats)")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--Re", type=float, default=20000.0)
    ap.add_argument("--T", type=float, default=12.0)
    ap.add_argument("--k1", type=int, default=6)
    ap.add_argument("--k2", type=int, default=40)
    ap.add_argument("--base_out", type=str, default="bench_out")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--control", action="store_true", help="run control (NSE, causal_off)")
    ap.add_argument("--sweep_c", nargs="*", type=float, default=[0.6,0.7,0.8,0.9,1.0,1.2,1.5,2.0])
    ap.add_argument("--theta_list", nargs="*", type=float, default=[])
    ap.add_argument("--avg_frac", type=float, default=0.33, help="use final (avg_frac) fraction of time for stats")
    ap.add_argument("--cfl", type=float, default=0.15, help="CFL factor for CCNS runs")
    ap.add_argument("--cfl_control", type=float, default=0.10, help="CFL factor for control run (can be smaller)")
    args = ap.parse_args()

    os.makedirs(args.base_out, exist_ok=True)
    rows = []
    case_dirs = []

    # Control
    if args.control:
        label = "control_NSE"
        outdir = run_case(args.base_out, label, causal_on=False, c=1.0, theta=1.0,
                          N=args.N, Re=args.Re, T=args.T, cfl=args.cfl_control,
                          delta=args.delta, seed=args.seed, k1=args.k1, k2=args.k2)
        s = summarize(outdir, avg_frac=args.avg_frac)
        s.update(dict(label=label, causal_on=False, c=np.nan, theta=1.0))
        rows.append(s); case_dirs.append((label, outdir))

    # CCNS capacity sweep
    for c in args.sweep_c:
        label = f"ccns_c{c:.1f}"
        outdir = run_case(args.base_out, label, causal_on=True, c=c, theta=1.0,
                          N=args.N, Re=args.Re, T=args.T, cfl=args.cfl,
                          delta=args.delta, seed=args.seed, k1=args.k1, k2=args.k2)
        s = summarize(outdir, avg_frac=args.avg_frac)
        s.update(dict(label=label, causal_on=True, c=c, theta=1.0))
        rows.append(s); case_dirs.append((label, outdir))

    # Optional theta sweep at c=0.8
    for th in args.theta_list:
        label = f"ccns_c0.8_th{th:.1f}"
        outdir = run_case(args.base_out, label, causal_on=True, c=0.8, theta=th,
                          N=args.N, Re=args.Re, T=args.T, cfl=args.cfl,
                          delta=args.delta, seed=args.seed, k1=args.k1, k2=args.k2)
        s = summarize(outdir, avg_frac=args.avg_frac)
        s.update(dict(label=label, causal_on=True, c=0.8, theta=th))
        rows.append(s); case_dirs.append((label, outdir))

    # Write summary
    summary_path = os.path.join(args.base_out, "benchmark_summary.csv")
    if rows:
        # Ensure column order
        fieldnames = ["label","causal_on","c","theta","n_mean","n_std","E_mean","Z_mean","eff_mean","viol_mean","proj_dE_mean"]
        # fill missing keys
        for r in rows:
            for k in fieldnames:
                r.setdefault(k, np.nan)
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)

    print("Wrote:", summary_path)

if __name__ == "__main__":
    main()
