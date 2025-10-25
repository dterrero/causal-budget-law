#!/usr/bin/env python3
# ccns_c_reference_clean.py
# 2-D compressible Navier–Stokes with:
#   - HLLC convective flux
#   - physical viscous + heat flux
#   - RK2 time stepping
#   - causal-capacity projection enforcing |u|^2 + v_int^2 <= c^2
#
# The causal projector implements the local throughput constraint
# |u|^2 + v_int^2 <= c^2 with
#   v_int^2 = theta_mu * nu * |S|
#   nu = mu / rho
#   |S| = sqrt(2 u_x^2 + 2 v_y^2 + (u_y + v_x)^2)
# and c = kappa * a, with a = sqrt(gamma p / rho).
# This is the discrete version of the capacity law described in the paper. :contentReference[oaicite:1]{index=1}

import argparse, csv, math, numpy as np, os
from typing import Tuple

GAMMA = 1.4     # ratio of specific heats
Rgas  = 1.0     # gas constant (nondimensional)

# ---------------------------------------------------------------------------
# Primitive/conservative conversions
# ---------------------------------------------------------------------------

def cons_to_prim(U: np.ndarray):
    """
    U[...,0] = rho
    U[...,1] = rho*u
    U[...,2] = rho*v
    U[...,3] = rho*E  (E = e + 0.5(u^2+v^2))

    Returns (rho,u,v,p,a,T,e) as arrays with same leading shape.
    """
    rho = U[...,0]
    u   = U[...,1] / np.maximum(rho,1e-14)
    v   = U[...,2] / np.maximum(rho,1e-14)
    E   = U[...,3] / np.maximum(rho,1e-14)

    ke  = 0.5*(u*u + v*v)
    e   = E - ke
    p   = (GAMMA - 1.0)*rho*e
    a   = np.sqrt(np.maximum(GAMMA*p/np.maximum(rho,1e-14), 1e-14))
    T   = p/(Rgas*np.maximum(rho,1e-14))

    return rho, u, v, p, a, T, e


def prim_to_cons(rho: np.ndarray,
                 u:   np.ndarray,
                 v:   np.ndarray,
                 p:   np.ndarray) -> np.ndarray:
    """
    Build conservative vector from primitive fields (rho,u,v,p).
    """
    E = p/((GAMMA - 1.0)*np.maximum(rho,1e-14)) + 0.5*(u*u + v*v)
    return np.stack([rho, rho*u, rho*v, rho*E], axis=-1)


# ---------------------------------------------------------------------------
# Riemann solver: HLLC
# ---------------------------------------------------------------------------

def hllc_flux(WL: Tuple[np.ndarray,...],
              WR: Tuple[np.ndarray,...],
              normal: Tuple[float,float]=(1.0,0.0)) -> np.ndarray:
    """
    Vectorized HLLC flux for compressible Euler in 2D.

    WL, WR are tuples (rho,u,v,p,a,T,e) for left/right states on a face.
    normal = (nx,ny) is unit normal pointing from L -> R.

    Returns flux[...,4] for each face.
    """
    nx, ny = normal

    rhoL,uL,vL,pL,aL,TL,eL = WL
    rhoR,uR,vR,pR,aR,TR,eR = WR

    # Normal velocities
    unL = uL*nx + vL*ny
    unR = uR*nx + vR*ny

    # Total specific enthalpy H = e + 0.5(u^2+v^2) + p/rho
    HL  = eL + 0.5*(uL*uL+vL*vL) + pL/np.maximum(rhoL,1e-14)
    HR  = eR + 0.5*(uR*uR+vR*vR) + pR/np.maximum(rhoR,1e-14)

    aMax = np.maximum(aL, aR)

    SL = np.minimum(unL, unR) - aMax
    SR = np.maximum(unL, unR) + aMax

    num = pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR)
    den = rhoL*(SL - unL) - rhoR*(SR - unR) + 1e-14
    SM  = np.clip(num/den, SL + 1e-10, SR - 1e-10)

    # pressure in star region (Toro's standard construction)
    pStar = np.maximum(
        pL + rhoL*(SL - unL)*(SM - unL),
        1e-10
    )

    # Conserved left/right
    UL = np.stack([
        rhoL,
        rhoL*uL,
        rhoL*vL,
        rhoL*(eL + 0.5*(uL*uL+vL*vL))
    ], axis=0)

    UR = np.stack([
        rhoR,
        rhoR*uR,
        rhoR*vR,
        rhoR*(eR + 0.5*(uR*uR+vR*vR))
    ], axis=0)

    # Physical fluxes left/right
    FL = np.stack([
        rhoL*unL,
        rhoL*unL*uL + pL*nx,
        rhoL*unL*vL + pL*ny,
        (rhoL*HL)*unL
    ], axis=0)

    FR = np.stack([
        rhoR*unR,
        rhoR*unR*uR + pR*nx,
        rhoR*unR*vR + pR*ny,
        (rhoR*HR)*unR
    ], axis=0)

    # Star states
    rhoLstar = rhoL*(SL - unL)/(SL - SM + 1e-14)
    rhoRstar = rhoR*(SR - unR)/(SR - SM + 1e-14)

    uLstar = uL + (SM - unL)*nx
    vLstar = vL + (SM - unL)*ny
    uRstar = uR + (SM - unR)*nx
    vRstar = vR + (SM - unR)*ny

    ELstar = eL + (SM - unL)*(pStar - pL)/(rhoL*(SL - unL) + 1e-14)
    ERstar = eR + (SM - unR)*(pStar - pR)/(rhoR*(SR - SM) + 1e-14)

    ULstar = np.stack([
        rhoLstar,
        rhoLstar*uLstar,
        rhoLstar*vLstar,
        rhoLstar*(ELstar + 0.5*(uLstar*uLstar+vLstar*vLstar))
    ], axis=0)

    URstar = np.stack([
        rhoRstar,
        rhoRstar*uRstar,
        rhoRstar*vRstar,
        rhoRstar*(ERstar + 0.5*(uRstar*uRstar+vRstar*vRstar))
    ], axis=0)

    # Assemble flux using wave-speed logic
    flux = np.zeros_like(UL)

    maskL = SL >= 0.0
    maskM = (SM >= 0.0) & (SL < 0.0)
    maskR = (SR >  0.0) & (SM < 0.0)
    maskO = ~(maskL | maskM | maskR)

    flux[:, maskL] = FL[:, maskL]
    flux[:, maskM] = (FL + SL[np.newaxis,...]*(ULstar - UL))[:, maskM]
    flux[:, maskR] = (FR + SR[np.newaxis,...]*(URstar - UR))[:, maskR]
    flux[:, maskO] = FR[:, maskO]

    # sanitize nans/infs
    flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    return flux  # shape (4, ...)


# ---------------------------------------------------------------------------
# Finite differences / helpers
# ---------------------------------------------------------------------------

def grad_centered(f: np.ndarray, dx: float, dy: float):
    """
    Periodic centered differences for scalar field f(x,y).
    Returns (df/dx, df/dy) arrays of same shape.
    """
    dfdx = (np.roll(f,-1,axis=0) - np.roll(f,1,axis=0)) / (2.0*dx)
    dfdy = (np.roll(f,-1,axis=1) - np.roll(f,1,axis=1)) / (2.0*dy)
    return dfdx, dfdy


def strain_mag(u: np.ndarray, v: np.ndarray, dx: float, dy: float):
    """
    Magnitude of the symmetric strain-rate tensor S.
    |S| = sqrt( 2 u_x^2 + 2 v_y^2 + (u_y + v_x)^2 ).
    """
    ux, uy = grad_centered(u, dx, dy)
    vx, vy = grad_centered(v, dx, dy)

    term = 2.0*(ux*ux + vy*vy) + (uy + vx)*(uy + vx)
    return np.sqrt(np.maximum(term, 0.0))


# ---------------------------------------------------------------------------
# Causal projection operator
# ---------------------------------------------------------------------------

def causal_project(U: np.ndarray,
                   kappa: float,
                   theta_mu: float,
                   mu: float,
                   dx: float,
                   dy: float,
                   eps_u: float=1e-14):
    """
    Enforce the local throughput cap:
        |u|^2 + v_int^2 <= c^2
    with
        v_int^2 = theta_mu * nu * |S|,  nu = mu / rho,
        c = kappa * a,
    where a is local sound speed.

    We rescale velocity if over capacity and convert the removed kinetic
    energy into internal energy (heating). The projection is L^2-nonexpansive
    in kinetic energy and matches the KKT active-set interpretation. :contentReference[oaicite:2]{index=2}
    """
    rho,u,v,p,a,T,e = cons_to_prim(U)

    nu    = mu / np.maximum(rho,1e-14)       # kinematic viscosity
    Smag  = strain_mag(u, v, dx, dy)         # |S|
    v_int_sq = theta_mu * nu * Smag          # internal throughput share

    c   = kappa * a                          # capacity
    speed2 = u*u + v*v                       # |u|^2
    total  = speed2 + v_int_sq               # |u|^2 + v_int^2

    cap    = np.maximum(0.0, c*c - v_int_sq) # allowed external share
    s2     = np.where(total > c*c,
                      cap/np.maximum(speed2, eps_u),
                      1.0)
    s2     = np.clip(s2, 0.0, 1.0)
    s      = np.sqrt(s2)

    u_star = s*u
    v_star = s*v

    # kinetic energy removed by projection (per mass)
    ke_old = 0.5*(u*u + v*v)
    ke_new = 0.5*(u_star*u_star + v_star*v_star)
    dEk    = np.maximum(ke_old - ke_new, 0.0)

    # reheat internal energy
    e_star = np.maximum(e + dEk, 1e-10)
    p_star = (GAMMA-1.0)*rho*e_star

    U_star = prim_to_cons(rho, u_star, v_star, p_star)

    # diagnostics
    phi_active = np.count_nonzero(total > c*c)/total.size
    Qproj_mean = float(np.mean(rho*dEk))  # mean (rho * removed_ke)

    return U_star, phi_active, Qproj_mean


# ---------------------------------------------------------------------------
# Viscous + heat fluxes and full flux divergence
# ---------------------------------------------------------------------------

def viscous_fluxes(U: np.ndarray,
                   mu: float,
                   kappa_th: float,
                   dx: float,
                   dy: float):
    """
    Compute viscous + heat flux at each cell.
    We'll later average to faces.

    tau = mu * (grad u + grad u^T - 2/3 div(u) I)
    q   = -kappa_th * grad T

    Return Fx_visc, Fy_visc each shape (...,4):
      Fx_visc[...,0] = 0
      Fx_visc[...,1] = tau_xx
      Fx_visc[...,2] = tau_xy
      Fx_visc[...,3] = u*tau_xx + v*tau_xy + q_x
    similarly for Fy_visc.
    """
    rho,u,v,p,a,T,e = cons_to_prim(U)

    dudx, dudy = grad_centered(u, dx, dy)
    dvdx, dvdy = grad_centered(v, dx, dy)
    dTdx, dTdy = grad_centered(T, dx, dy)

    divu = dudx + dvdy

    tau_xx = mu*(2.0*dudx - (2.0/3.0)*divu)
    tau_yy = mu*(2.0*dvdy - (2.0/3.0)*divu)
    tau_xy = mu*(dudy + dvdx)

    qx = -kappa_th * dTdx
    qy = -kappa_th * dTdy

    Fx_visc = np.stack([
        np.zeros_like(u),
        tau_xx,
        tau_xy,
        u*tau_xx + v*tau_xy + qx
    ], axis=-1)

    Fy_visc = np.stack([
        np.zeros_like(u),
        tau_xy,
        tau_yy,
        u*tau_xy + v*tau_yy + qy
    ], axis=-1)

    return Fx_visc, Fy_visc


def flux_divergence(U: np.ndarray,
                    dx: float,
                    dy: float,
                    mu: float,
                    kappa_th: float) -> np.ndarray:
    """
    Compute ∇·F for compressible NS:
      F = F_inv (Euler/HLLC) - F_visc
    using periodic BC in x,y.

    Steps:
      1. Build left/right states at x-faces and y-faces by simple
         first-order reconstruction.
      2. Get inviscid flux via HLLC at those faces.
      3. Get viscous flux per cell, average to faces, subtract.
      4. Take finite-volume divergence with periodic wrap.
    """
    Nx, Ny, _ = U.shape

    # --- Inviscid fluxes (HLLC) ---
    # Faces normal to x: between (i,j) [L] and (i+1,j) [R]
    U_Lx = U
    U_Rx = np.roll(U, -1, axis=0)

    WL = cons_to_prim(U_Lx)
    WR = cons_to_prim(U_Rx)
    Fx = hllc_flux(WL, WR, normal=(1.0,0.0))  # shape (4,Nx,Ny)

    # Faces normal to y: between (i,j) [L] and (i,j+1) [R]
    U_Ly = U
    U_Ry = np.roll(U, -1, axis=1)

    WLy = cons_to_prim(U_Ly)
    WRy = cons_to_prim(U_Ry)
    Fy = hllc_flux(WLy, WRy, normal=(0.0,1.0))  # shape (4,Nx,Ny)

    # --- Viscous + heat flux ---
    Fxv_cell, Fyv_cell = viscous_fluxes(U, mu, kappa_th, dx, dy)
    # Average to faces:
    Fxv_face = 0.5*(Fxv_cell + np.roll(Fxv_cell, -1, axis=0))  # (Nx,Ny,4)
    Fyv_face = 0.5*(Fyv_cell + np.roll(Fyv_cell, -1, axis=1))  # (Nx,Ny,4)

    # subtract viscous from inviscid
    # Note: Fx is (4,Nx,Ny); Fxv_face is (Nx,Ny,4)
    Fx_total = Fx - np.moveaxis(Fxv_face, -1, 0)  # (4,Nx,Ny)
    Fy_total = Fy - np.moveaxis(Fyv_face, -1, 0)  # (4,Nx,Ny)

    # --- Finite-volume divergence ---
    # d/dx [F_x(i+1/2) - F_x(i-1/2)] / dx
    Fx_fwd = Fx_total
    Fx_bwd = np.roll(Fx_total, 1, axis=1)  # careful: axis? Wait:
    # NOTE: Fx_total shape (4,Nx,Ny) with faces at i+1/2.
    # Shifting in x-direction is axis=1 after moveaxis? No:
    # We currently have axes: (comp, Nx, Ny).
    # x-direction is axis=1. So:
    Fx_bwd = np.roll(Fx_total, 1, axis=1)
    div_x = (Fx_fwd - Fx_bwd) / dx  # (4,Nx,Ny)

    # d/dy [F_y(j+1/2) - F_y(j-1/2)] / dy
    Fy_fwd = Fy_total
    Fy_bwd = np.roll(Fy_total, 1, axis=2)  # y-direction is axis=2
    div_y = (Fy_fwd - Fy_bwd) / dy        # (4,Nx,Ny)

    divF = div_x + div_y                  # (4,Nx,Ny)
    # Move comp axis back to last:
    divF = np.moveaxis(divF, 0, -1)       # (Nx,Ny,4)

    return divF


# ---------------------------------------------------------------------------
# Diagnostics / initialization
# ---------------------------------------------------------------------------

def max_wave_speed(U: np.ndarray) -> float:
    """
    CFL speed = max(|u| + a) over domain, no artificial clamp.
    """
    rho,u,v,p,a,T,e = cons_to_prim(U)
    wloc = np.sqrt(u*u + v*v) + a
    return float(np.max(wloc))


def diagnostics(U: np.ndarray,
                dx: float,
                dy: float) -> Tuple[float,float]:
    """
    Return (E, Z):
      E = mean(0.5|u|^2)
      Z = mean(0.5 w^2)
    where w = v_x - u_y is scalar vorticity in 2D.
    """
    rho,u,v,p,a,T,e = cons_to_prim(U)
    E = 0.5*np.mean(u*u + v*v)

    ux, uy = grad_centered(u, dx, dy)
    vx, vy = grad_centered(v, dx, dy)
    w  = vx - uy
    Z  = 0.5*np.mean(w*w)
    return float(E), float(Z)


def init_mixing_layer(Nx: int,
                      Ny: int,
                      Lx: float,
                      Ly: float,
                      M0: float=2.0,
                      delta: float=0.05,
                      noise: float=1e-3,
                      seed: int=1) -> np.ndarray:
    """
    Kelvin–Helmholtz style shear layer, periodic in x and y.
    Two oppositely directed tanh jets plus a sinusoidal perturbation.
    """
    np.random.seed(seed)
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X,Y = np.meshgrid(x,y, indexing='ij')

    rho0 = np.ones((Nx,Ny))
    p0   = np.ones((Nx,Ny))/GAMMA
    a0   = np.sqrt(GAMMA*p0/rho0)

    Uhi, Ulo = M0*a0, -M0*a0

    yy1 = (Y - 0.25*Ly)/delta
    yy2 = (Y - 0.75*Ly)/delta

    u = Uhi*np.tanh(yy1) + Ulo*np.tanh(yy2)
    v = 0.02*Uhi*np.sin(2.0*X) + noise*np.random.randn(Nx,Ny)

    return prim_to_cons(rho0, u, v, p0)


def enforce_positivity(U: np.ndarray,
                       rho_floor: float=1e-10,
                       e_floor: float=1e-10) -> np.ndarray:
    """
    Clamp rho>0 and e>0 cellwise, recompute p and conservative variables.
    """
    rho,u,v,p,a,T,e = cons_to_prim(U)
    rho = np.maximum(rho, rho_floor)
    e   = np.maximum(e,   e_floor)
    p   = (GAMMA-1.0)*rho*e
    return prim_to_cons(rho, u, v, p)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(Nx: int=200,
        Ny: int=128,
        Lx: float=2*np.pi,
        Ly: float=2*np.pi,
        Tfinal: float=0.1,
        CFL: float=0.3,
        Re: float=2e4,
        Pr: float=0.72,
        kappa: float=1.5,
        theta_mu: float=1.0,
        outdir: str="ccns_c_out"):
    """
    Advance the 2-D compressible flow with causal-capacity projection
    to time Tfinal.

    Writes diagnostics.csv with:
      t, E, Z, phi_active, Qproj_mean
    """
    # grid / material coeffs
    dx, dy = Lx/Nx, Ly/Ny
    mu  = 1.0/Re   # dynamic viscosity (nondimensional)
    k_th = mu * GAMMA * Rgas / (Pr*(GAMMA-1.0))  # conductivity

    # initial condition
    U  = init_mixing_layer(Nx, Ny, Lx, Ly)
    t, step = 0.0, 0

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir,"diagnostics.csv"), "w", newline="") as fdiag:
        w = csv.writer(fdiag)
        w.writerow(["t","E","Z","phi","Qproj_mean"])

        while t < Tfinal:
            # timestep from CFL
            dt = CFL * min(dx,dy) / max_wave_speed(U)

            # ---- RK stage 1 ----
            divF  = flux_divergence(U, dx, dy, mu, k_th)
            U1    = enforce_positivity(U - dt*divF)
            U1, phi1, Q1 = causal_project(U1, kappa, theta_mu, mu, dx, dy)

            # ---- RK stage 2 ----
            divF1 = flux_divergence(U1, dx, dy, mu, k_th)
            U2tmp = U + 0.5*( -dt*divF - dt*divF1 )
            U2    = enforce_positivity(U2tmp)
            U2, phi2, Q2 = causal_project(U2, kappa, theta_mu, mu, dx, dy)

            # diagnostics (every 10 steps or at end)
            if (step % 10 == 0) or (t+dt >= Tfinal):
                E, Z = diagnostics(U, dx, dy)
                phi  = 0.5*(phi1+phi2)
                Qm   = 0.5*(Q1+Q2)
                w.writerow([t, E, Z, phi, Qm])
                fdiag.flush()

            # advance
            U = U2
            t = t + dt
            step += 1

    print(f"[done] wrote diagnostics to {outdir}/diagnostics.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="2-D compressible causal-capacity Navier–Stokes solver"
    )
    ap.add_argument("--Nx", type=int, default=200)
    ap.add_argument("--Ny", type=int, default=128)
    ap.add_argument("--Lx", type=float, default=2*math.pi)
    ap.add_argument("--Ly", type=float, default=2*math.pi)
    ap.add_argument("--Tfinal", type=float, default=0.1)
    ap.add_argument("--CFL", type=float, default=0.3)
    ap.add_argument("--Re", type=float, default=2e4)
    ap.add_argument("--Pr", type=float, default=0.72)
    ap.add_argument("--kappa", type=float, default=1.5,
                    help="capacity prefactor c = kappa * a")
    ap.add_argument("--theta_mu", type=float, default=1.0,
                    help="internal-throughput prefactor v_int^2 = theta_mu * nu * |S|")
    ap.add_argument("--outdir", type=str, default="ccns_c_out")

    args = ap.parse_args()
    run(Nx=args.Nx,
        Ny=args.Ny,
        Lx=args.Lx,
        Ly=args.Ly,
        Tfinal=args.Tfinal,
        CFL=args.CFL,
        Re=args.Re,
        Pr=args.Pr,
        kappa=args.kappa,
        theta_mu=args.theta_mu,
        outdir=args.outdir)

if __name__ == "__main__":
    main()
