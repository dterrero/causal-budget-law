#!/usr/bin/env python3
import argparse
import json
import math
import numpy as np
import scipy.constants as const

# =========================
# Constants & base helpers
# =========================
h      = const.h
mu0    = const.mu_0
muB    = const.physical_constants['Bohr magneton'][0]
muN    = const.physical_constants['nuclear magneton'][0]
alpha  = const.alpha
gJ     = 2.00231930436256
a0     = const.physical_constants['Bohr radius'][0]
fR     = const.Rydberg * const.c  # Rydberg frequency
me     = const.m_e
hbar   = const.hbar
c      = const.c

# ---------- n* correction constants (STRUCTURAL, Li-only fade) ----------
D1, D2, ZC, P_FADE = 0.0, 3.797894, 2.455635, 8.0  # Li-only effective n* fade

# ---------- nuclear moments (μ/μ_N) for isotope scaling tests ----------
nuclear_mu = {
    'Rb-87': 2.751,     # μ/μ_N
    'Rb-85': 1.353348,  # μ/μ_N
    'K-39' : 0.391466,
    'K-41' : 0.214,     # (~0.214–0.215)
}

# ---------- dataset (electronic anchors) ----------
alkali = {
    'Li-7':   {'Z': 3,  'n': 2, 'I': 3/2, 'gI': 3.256427/(3/2), 'f_exp': 0.803504e9},
    'Na-23':  {'Z': 11, 'n': 3, 'I': 3/2, 'gI': 2.217522/(3/2), 'f_exp': 1.771626e9},
    'K-39':   {'Z': 19, 'n': 4, 'I': 3/2, 'gI': 0.391466/(3/2), 'f_exp': 0.461722e9},
    'Rb-87':  {'Z': 37, 'n': 5, 'I': 3/2, 'gI': 1.834/(3/2),    'f_exp': 6.83468261090429e9},
    'Cs-133': {'Z': 55, 'n': 6, 'I': 7/2, 'gI': 2.582025/(7/2), 'f_exp': 9.192631770e9},
}
HEAVY_SET = ['K-39', 'Rb-87', 'Cs-133']
ALL_SET = list(alkali.keys())

# ---------- relativistic contact factor ----------
def R_rel(Z, gamma_rel=2.0):
    """Relativistic enhancement for s_{1/2} contact density."""
    x = min(Z * alpha, 0.999)
    return (1.0 - x*x)**(-gamma_rel)

def kappa0_from_hydrogen():
    """Hydrogen anchor: universal normalization of the causal contact law."""
    fH = (13.605693122994 * const.e) / const.h
    psi0_sq_H = 1.0/(np.pi * a0**3)
    return psi0_sq_H * (hbar * c) / (me * (2*np.pi*fH)**2)

def psi0_from_causal_NR(Z, n_star):
    """
    Nonrelativistic causal contact density using bare-Z confinement,
    adjusted by the effective quantum number n_star. (NR: no R_rel here.)
    """
    k0 = kappa0_from_hydrogen()
    f_conf = fR * (Z**2 / n_star**3)
    base = k0 * me * (2*np.pi*f_conf)**2 / (hbar * c)
    return base  # NR part only

def f_clock_from_psi(psi0_sq, I, gI):
    """Hyperfine clock from |ψ(0)|^2."""
    A = (2.0/3.0) * mu0 * gI * muN * gJ * muB * psi0_sq
    A_over_h = A / h
    return A_over_h * (I + 0.5), A_over_h

# =========================
# Analytic β(Z) & n*(Z)
# =========================
def w_fade(Z, Zc, p):
    """Fading weight function for n* correction (local to light elements)."""
    return 1.0 / (1.0 + (Z / Zc)**p)

def n_star_Z(Z, n):
    """Effective principal quantum number (n -> n* for light elements; mainly Li)."""
    w = w_fade(Z, ZC, P_FADE)
    return max(n - D1*w - D2*(w**2), 0.6)

# Analytic intercept from structural numbers:
B0_ANALYTIC = 7.0/26.0            # 0.269230769...
# Relativistic shape (held fixed from heavy-locked shape)
B1_SHAPE = -0.813057
B2_SHAPE =  4.700252

# Quadratic & cubic knobs (wired via CLI)
B2_SCALE_DEFAULT = 0.90
B2_SCALE = B2_SCALE_DEFAULT
B3_SHAPE_DEFAULT = 0.0
B3_SHAPE = B3_SHAPE_DEFAULT

def beta_base(Z):
    x = (alpha*Z)**2
    return (B0_ANALYTIC
            + B1_SHAPE*x
            + (B2_SCALE*B2_SHAPE)*(x**2)
            + B3_SHAPE*(x**3))

# Local analytic anomaly (Na bump)
C_BETA_ANALYTIC = 1.0/28.0     # 0.035714...
BUMP_Z0         = 11.0         # center at Na
BUMP_SIGMA      = 3.0          # narrow; K/Rb/Cs essentially untouched

def beta_with_bump(Z, use_bump=True):
    b = beta_base(Z)
    if not use_bump:
        return b
    w = np.exp(-0.5*((Z - BUMP_Z0)/BUMP_SIGMA)**2)
    return b + C_BETA_ANALYTIC * w

# =========================
# Core prediction helpers
# =========================
def predict_with_explicit_beta(Z, n, I, gI, beta, gamma_rel=2.0, lambda_rel=1.0):
    """
    Compute clock frequency using an explicit β. lambda_rel rescales γ_rel (analytic mode).
    Returns: f_pred, Zeff_screened, n_star, psi_final, A_over_h
    """
    n_star = n_star_Z(Z, n)
    # 1) NR Zeff scale from Causal Law (using n*)
    psi_NR = psi0_from_causal_NR(Z, n_star)
    Zeff_bare = (psi_NR * np.pi * a0**3 * n_star**3)**(1.0/3.0)
    # 2) Apply screening Zeff = Zeff_bare * Z^{-β}
    Zeff_screened = Zeff_bare * (Z ** (-beta))
    # 3) Final density & relativity (γ scaled in analytic mode)
    gamma_eff = gamma_rel * lambda_rel
    psi_final = (Zeff_screened**3)/(np.pi*a0**3*n_star**3) * R_rel(Z, gamma_eff)
    f_pred, A_over_h = f_clock_from_psi(psi_final, I, gI)
    return f_pred, Zeff_screened, n_star, psi_final, A_over_h

def predict_core(Z, n, I, gI, *, mode='calibrated', lambda_rel=0.7, gamma_rel=2.0, use_bump=True):
    """
    Wrapper choosing calibrated vs analytic mode.
    - calibrated: λ not used (lambda_rel=1.0 internally), β=beta_with_bump(Z)
    - analytic  : λ used to rescale γ, β=beta_with_bump(Z) (same β form, different relativity strength)
    """
    beta = beta_with_bump(Z, use_bump=use_bump)
    lam  = 1.0 if mode == 'calibrated' else float(lambda_rel)
    return predict_with_explicit_beta(Z, n, I, gI, beta=beta, gamma_rel=gamma_rel, lambda_rel=lam) + (beta,)

def isotope_scale(f_ref, mu_ref, I_ref, mu_iso, I_iso):
    """
    Pure nuclear scaling at fixed electronic factor:
    f_iso = f_ref * [ (μ_iso/I_iso)*(I_iso+1/2) ] / [ (μ_ref/I_ref)*(I_ref+1/2) ]
    """
    num = (mu_iso / I_iso) * (I_iso + 0.5)
    den = (mu_ref / I_ref) * (I_ref + 0.5)
    return f_ref * (num / den)

# =========================
# Tiny residuals "plot"
# =========================
def ascii_bar(err_pct, width=28):
    """Draw a small symmetric ASCII bar centered at 0."""
    span = max(1.0, max(abs(err_pct), 10.0))  # normalize
    n = min(width, int(round(abs(err_pct) / span * width)))
    if err_pct < 0:
        return "[" + "-"*n + "|" + " "*(width-n) + "]"
    else:
        return "[" + " "*(width-n) + "|" + "+"*n + "]"

def set_stats(keys, baseline):
    errs = [abs(baseline[k]['err_pct']) for k in keys if k in baseline]
    if not errs:
        return float('nan'), float('nan')
    rms = math.sqrt(sum(e*e for e in errs) / len(errs))
    mae = sum(errs) / len(errs)
    return rms, mae

# =========================
# λ sweep helper
# =========================
def lambda_sweep(mode, gamma_rel, use_bump, lam_grid, keys, print_rows=False):
    """
    Return (best_lambda, best_rms). Optionally print sweep rows.
    Guaranteed to return a tuple even on edge cases.
    """
    lam_grid = np.asarray(lam_grid).ravel()
    if lam_grid.size == 0:
        if print_rows:
            print("[λ-sweep] Empty lambda grid; skipping.")
        return (None, float("inf"))
    if not keys:
        if print_rows:
            print("[λ-sweep] Empty key set; skipping.")
        return (None, float("inf"))

    best_lam, best_rms = (None, float("inf"))
    rows = []

    for lam in lam_grid:
        errs = []
        for key in keys:
            Z, n, I, gI, f_exp = (alkali[key][k] for k in ['Z','n','I','gI','f_exp'])
            f_pred, *_ = predict_core(
                Z, n, I, gI,
                mode=mode,
                lambda_rel=float(lam),
                gamma_rel=gamma_rel,
                use_bump=use_bump
            )
            errs.append(((f_pred - f_exp) / f_exp * 100.0) ** 2)

        rms = math.sqrt(sum(errs) / len(errs))
        rows.append((float(lam), float(rms)))
        if rms <= best_rms:
            best_lam, best_rms = float(lam), float(rms)

    if print_rows:
        print("--- λ sweep (analytic mode): RMS % error on set ---")
        for lam, rms in rows:
            tag = " <— best so far" if abs(rms - best_rms) < 1e-12 or rms <= best_rms else ""
            print(f"λ={lam:0.3f}: RMS={rms:0.3f}%{tag}")

    return (best_lam, best_rms)


def _rms_for_params(lam, b3, keys, gamma_rel, use_bump):
    """Compute RMS %% error for a given (lambda_rel, b3) on a set of keys."""
    global B3_SHAPE
    old_b3 = B3_SHAPE
    B3_SHAPE = float(b3)
    errs2 = []
    for key in keys:
        Z, n, I, gI, f_exp = (alkali[key][k] for k in ['Z','n','I','gI','f_exp'])
        f_pred, *_ = predict_core(Z, n, I, gI,
                                  mode="analytic",
                                  lambda_rel=float(lam),
                                  gamma_rel=gamma_rel,
                                  use_bump=use_bump)
        err_pct = (f_pred - f_exp)/f_exp * 100.0
        errs2.append(err_pct*err_pct)
    B3_SHAPE = old_b3
    return math.sqrt(sum(errs2)/len(errs2))

def grid_search_lambda_b3(lam_grid, b3_grid, gamma_rel, use_bump,
                          keys_heavy=HEAVY_SET, keys_all=ALL_SET,
                          print_table=False, return_rms_grids=False):
    
    best_heavy = (None, None, float("inf"))
    best_all   = (None, None, float("inf"))

    # optional print header
    if print_table:
        hdr = "b3 \\ λ  " + "  ".join([f"{lam:.3f}" for lam in lam_grid])
        print("\n--- 2D grid (heavy-set RMS %): ---")
        print(hdr)

    for b3 in b3_grid:
        row_vals = []
        for lam in lam_grid:
            rms_h = _rms_for_params(lam, b3, keys_heavy, gamma_rel, use_bump)
            if rms_h < best_heavy[2]:
                best_heavy = (lam, b3, rms_h)
            rms_a = _rms_for_params(lam, b3, keys_all,   gamma_rel, use_bump)
            if rms_a < best_all[2]:
                best_all = (lam, b3, rms_a)
            row_vals.append(rms_h)
        if print_table:
            print(f"{b3:7.3f}  " + "  ".join([f"{v:6.3f}" for v in row_vals]))

    print(f"\n[grid] Best (HEAVY): λ={best_heavy[0]:.3f}, b3={best_heavy[1]:.3f}  -> RMS={best_heavy[2]:.3f}%")
    print(f"[grid] Best (ALL)  : λ={best_all[0]:.3f}, b3={best_all[1]:.3f}  -> RMS={best_all[2]:.3f}%\n")


    if return_rms_grids:
        RMS_heavy = np.zeros((len(b3_grid), len(lam_grid)))
        RMS_all   = np.zeros((len(b3_grid), len(lam_grid)))
        for i_b3, b3 in enumerate(b3_grid):
            for i_lam, lam in enumerate(lam_grid):
                RMS_heavy[i_b3, i_lam] = _rms_for_params(lam, b3, keys_heavy, gamma_rel, use_bump)
                RMS_all[i_b3, i_lam]   = _rms_for_params(lam, b3, keys_all,   gamma_rel, use_bump)
        return best_heavy, best_all, RMS_heavy, RMS_all

    return best_heavy, best_all

def lambda_rel_from_dirac_match(Z_list, alpha=const.alpha):
    """
    Derive a = λ_rel * γ_rel by projecting the exact Dirac s_{1/2} factor
    R_Dirac(Z) = 1 / [γ(2γ-1)] onto (1 - x)^{-a}, x=(Zα)^2.
    Returns λ_rel = a / 2 (given your default γ_rel = 2).
    """
    import math
    num = 0.0
    den = 0.0
    for Z in Z_list:
        x = (alpha * Z) ** 2
        # Guard against x -> 1
        if x >= 1.0:
            x = 0.999999
        gamma = math.sqrt(1.0 - x)
        R_dirac = 1.0 / (gamma * (2.0 * gamma - 1.0))
        l1 = math.log(R_dirac)
        l2 = math.log(1.0 - x)
        num += l1 * l2
        den += l2 * l2
    a_star = -num / den
    return 0.5 * a_star  # because your model uses (1-x)^{-λ_rel*γ_rel} with γ_rel=2


# -------------------------
# Derived-λ from Dirac + FNS + BW (no data)
# -------------------------
A_default = {   # light mapping for common isotopes
    19: 39,   # K-39
    37: 87,   # Rb-87
    55: 133,  # Cs-133
    3 : 7,    # Li-7
    11: 23,   # Na-23
}

def nuclear_radius_m(A, r0_fm=1.2):
    """R_N in meters; r0 in fm."""
    return r0_fm * 1e-15 * (A ** (1.0/3.0))

def delta_fns(Z, A, C_FNS=1.0, r0_fm=1.2):
    """
    Finite-nuclear-size correction (first-order proxy).
    Negative sign = contact density suppressed by extended charge.
    """
    x = (alpha * Z)**2
    RN = nuclear_radius_m(A, r0_fm=r0_fm)
    return - C_FNS * x * (RN / a0)**2

def epsilon_bw(Z, A, C_BW=0.30, r0_fm=1.2):
    """
    Bohr–Weisskopf (magnetization distribution) proxy.
    Negative = suppression from spread magnetization in s-state.
    """
    RN = nuclear_radius_m(A, r0_fm=r0_fm)
    return - C_BW * (alpha * Z) * (RN / a0)

def lambda_from_dirac_matching_for_Z(Z, gamma_rel=2.0, C_FNS=1.0, C_BW=0.30, r0_fm=1.2, A_map=None):
    """
    Closed-form per-element λ from matching:
      (1 - (Zα)^2)^(-λγ) ≈ (1 - (Zα)^2)^(-γ) * (1 + δ_FNS + ε_BW)
    => λ = 1 + ln(1+δFNS+εBW) / [γ ln(1 - (Zα)^2)]
    """
    if A_map is None:
        A_map = A_default
    A = A_map.get(Z, int(round(2.5*Z)))  # fallback A≈2.5Z
    x = (alpha * Z)**2
    T = 1.0 + delta_fns(Z, A, C_FNS, r0_fm) + epsilon_bw(Z, A, C_BW, r0_fm)
    # guardrails
    T = max(T, 1e-6)
    denom = gamma_rel * math.log(max(1.0 - x, 1e-12))
    return 1.0 + math.log(T) / denom

def lambda_rel_from_dirac_match(Z_list, gamma_rel=2.0, C_FNS=1.0, C_BW=0.30, r0_fm=1.2, A_map=None):
    """Average the per-element λ across the provided Z_list."""
    vals = [lambda_from_dirac_matching_for_Z(
                Z, gamma_rel=gamma_rel, C_FNS=C_FNS, C_BW=C_BW, r0_fm=r0_fm, A_map=A_map)
            for Z in Z_list]
    return float(sum(vals)/len(vals))

# =========================
# CLI
# =========================
def main():
    # Wire CLI → globals
    global B2_SCALE, B3_SHAPE
    
    p = argparse.ArgumentParser(description="Predictive Causal–Dirac Model (dual-mode) with reporting")

    # Mode & physics knobs
    p.add_argument("--mode", choices=["calibrated", "analytic"], default="calibrated",
                   help="Choose model mode. 'calibrated' reproduces your near-perfect baseline.")
    p.add_argument("--lambda-rel", type=float, default=0.7,
               help="Relativity scaling λ (only used in MODE=analytic; scales γ_rel → λ·γ_rel).")
    p.add_argument("--gamma-rel", type=float, default=2.0,
                   help="Base relativistic exponent γ for R_rel.")
    p.add_argument("--b2-scale", type=float, default=B2_SCALE_DEFAULT,
                   help="Scale factor on quadratic β term (multiplies B2_SHAPE).")
    p.add_argument("--b3-shape", type=float, default=B3_SHAPE_DEFAULT,
                   help="Coefficient on cubic term in β(Z): B3 * x^3, x=(αZ)^2.")

    # β(Z) options
    p.add_argument("--no-bump", action="store_true",
                   help="Disable the Na-centered Gaussian bump in β(Z).")

    # Output controls
    p.add_argument("--detail", type=str, default="",
                   help="Print intermediate quantities for one element (e.g., Cs-133).")
    p.add_argument("--csv", type=str, default="",
                   help="Write baseline table to CSV file (path).")
    p.add_argument("--json", type=str, default="",
                   help="Write a JSON dump of results (path).")
    p.add_argument("--print-coeffs", action="store_true",
                   help="Print active β-shape coefficients and sample beta_base(Z) values.")

    # λ optimization controls
    p.add_argument("--no-sweep", action="store_true",
                   help="In analytic mode, skip all λ sweeps.")
    p.add_argument("--lam-range", nargs=3, type=float, default=[0.5, 1.5, 41],
                   help="Sweep range for λ: start stop numpoints (only used in analytic mode).")
    p.add_argument("--sweep-print", action="store_true",
                   help="Print the λ sweep table (if a sweep is performed).")
    p.add_argument("--rms-set", choices=["heavy", "all"], default="heavy",
                   help="Which set to use for λ optimization in analytic mode (heavy or all five).")
    p.add_argument("--auto-lambda", action="store_true",
                   help="In analytic mode, pick λ by minimizing RMS on the set chosen by --rms-set.")
    p.add_argument("--subset", type=str, default="",
                   help="Comma-separated element keys to include (overrides --rms-set).")
    p.add_argument("--grid-search", action="store_true",
                help="Sweep over (lambda_rel, b3_shape) and report best pairs.")
    p.add_argument("--b3-range", nargs=3, type=float, default=[0.11, 0.13, 9],
                help="Sweep range for b3: start stop numpoints.")
    p.add_argument("--apply-grid-best", choices=["", "heavy", "all"], default="",
                help="If set, apply the best (λ,b3) from that set before baseline.")
    p.add_argument("--grid-print", action="store_true",
                help="Print a compact RMS table for the heavy set.")
    p.add_argument("--derive-lambda", action="store_true",
               help="Derive lambda_rel from Dirac+FNS+BW matching (no data fit). Overrides --lambda-rel if set.")
    p.add_argument("--fns-c", type=float, default=1.0,
                help="Coefficient C_FNS for finite nuclear size correction (dimensionless).")
    p.add_argument("--bw-c", type=float, default=0.30,
                help="Coefficient C_BW for Bohr–Weisskopf correction (dimensionless).")
    p.add_argument("--r0-fm", type=float, default=1.2,
                help="Nuclear radius constant r0 in fm for R_N = r0 * A^(1/3).")
    p.add_argument("--json-heavy", help="Output JSON for heavy-set RMS grid")
    p.add_argument("--json-all", help="Output JSON for all-set RMS grid")


    args = p.parse_args()
    B2_SCALE = float(args.b2_scale)
    B3_SHAPE = float(args.b3_shape)

    # Build optimization/summary set
    if args.subset.strip():
        KEYS = [k.strip() for k in args.subset.split(",") if k.strip()]
    else:
        KEYS = HEAVY_SET if args.rms_set == "heavy" else ALL_SET

    # Guard unknown keys
    unknown = [k for k in KEYS if k not in alkali]
    if unknown:
        raise ValueError(f"Unknown element keys in --subset: {unknown}")

    MODE = args.mode
    LAMBDA_REL = float(args.lambda_rel)
    GAMMA_REL = float(args.gamma_rel)
    USE_BUMP = not args.no_bump

    # --- Choose LAMBDA_REL (derived → auto → manual) ---
    if MODE == "analytic":
        if args.derive_lambda:
            LAMBDA_REL = lambda_rel_from_dirac_match(
                Z_list=[19, 37, 55],      # K, Rb, Cs
                C_FNS=args.fns_c,
                C_BW=args.bw_c,
                r0_fm=args.r0_fm
            )
            print(f"[derived] λ_rel = {LAMBDA_REL:.3f} from Dirac+FNS+BW "
                f"(Z={ [19,37,55] }, r0={args.r0_fm:.2f} fm, "
                f"C_FNS={args.fns_c:.3f}, C_BW={args.bw_c:.3f})\n")

   # ---- Grid / apply-best logic (single place) ----
    best_heavy = None
    best_all   = None

    if args.grid_search:
        lam_start, lam_stop, lam_num = args.lam_range
        b3_start,  b3_stop,  b3_num  = args.b3_range
        lam_grid = np.linspace(lam_start, lam_stop, int(lam_num))
        b3_grid  = np.linspace(b3_start,  b3_stop,  int(b3_num))

        # Run the grid; support either 2-return or 4-return implementations
        RMS_heavy = RMS_all = None
        try:
            # Newer variant that can return RMS grids
            best_heavy, best_all, RMS_heavy, RMS_all = grid_search_lambda_b3(
                lam_grid, b3_grid,
                gamma_rel=GAMMA_REL, use_bump=USE_BUMP,
                keys_heavy=HEAVY_SET, keys_all=ALL_SET,
                print_table=args.grid_print,
                return_rms_grids=True
            )
        except TypeError:
            # Older variant that returns only the best pairs
            best_heavy, best_all = grid_search_lambda_b3(
                lam_grid, b3_grid,
                gamma_rel=GAMMA_REL, use_bump=USE_BUMP,
                keys_heavy=HEAVY_SET, keys_all=ALL_SET,
                print_table=args.grid_print
            )

        # Optionally apply the best (λ, b3) from THIS run
        if args.apply_grid_best in ("heavy", "all"):
            pick = best_heavy if args.apply_grid_best == "heavy" else best_all
            if not pick or pick[0] is None:
                raise RuntimeError("Grid search did not produce a best (λ, b3) pair.")
            LAMBDA_REL = float(pick[0])
            B3_SHAPE   = float(pick[1])   # global knob used by beta_base()
            print(f"[grid→baseline] Using {args.apply_grid_best.upper()} best: "
                f"λ={LAMBDA_REL:.3f}, b3={B3_SHAPE:.3f}\n")

        # Optional JSON exports for plotting (only if RMS grids were returned)
        if RMS_heavy is not None and getattr(args, "json_heavy", None):
            out_heavy = {
                "lambda_grid": lam_grid.tolist(),
                "b3_grid": b3_grid.tolist(),
                "rms_grid": RMS_heavy.tolist(),
                "best": {"lambda": float(best_heavy[0]),
                        "b3": float(best_heavy[1]),
                        "rms": float(best_heavy[2])}
            }
            with open(args.json_heavy, "w") as f:
                json.dump(out_heavy, f, indent=2)
            print(f"[JSON] Heavy-set RMS grid saved to {args.json_heavy}")

        if RMS_all is not None and getattr(args, "json_all", None):
            out_all = {
                "lambda_grid": lam_grid.tolist(),
                "b3_grid": b3_grid.tolist(),
                "rms_grid": RMS_all.tolist(),
                "best": {"lambda": float(best_all[0]),
                        "b3": float(best_all[1]),
                        "rms": float(best_all[2])}
            }
            with open(args.json_all, "w") as f:
                json.dump(out_all, f, indent=2)
            print(f"[JSON] All-set RMS grid saved to {args.json_all}")

    else:
        # Not running a grid this invocation — optionally apply a previously saved best
        if args.apply_grid_best in ("heavy", "all"):
            fname = args.json_heavy if args.apply_grid_best == "heavy" else args.json_all
            if not fname:
                raise RuntimeError("Provide --json-heavy/--json-all with --apply-grid-best when not using --grid-search.")
            with open(fname) as f:
                data = json.load(f)
            best = data.get("best", {})
            if "lambda" not in best or "b3" not in best:
                raise RuntimeError(f"File {fname} is missing a 'best' record with 'lambda' and 'b3'.")
            LAMBDA_REL = float(best["lambda"])
            B3_SHAPE   = float(best["b3"])   # global knob
            print(f"[apply-grid-best] Using {args.apply_grid_best.upper()} best: "
                f"λ={LAMBDA_REL:.3f}, b3={B3_SHAPE:.3f}\n")

    # ---- If analytic & auto-λ requested, sweep first so baseline uses chosen λ ----
    if MODE == "analytic" and not args.no_sweep and args.auto_lambda:
        start, stop, num = args.lam_range
        lam_grid = np.linspace(start, stop, int(num))
        best_lam, best_rms = lambda_sweep("analytic", GAMMA_REL, USE_BUMP, lam_grid, KEYS,
                                          print_rows=args.sweep_print)
        if best_lam is None:
            raise RuntimeError("Auto-λ sweep failed (empty grid or key set). "
                               "Try --lam-range or check --rms-set/--subset.")
        LAMBDA_REL = float(best_lam)
        print(f"[auto-λ] Picked λ={LAMBDA_REL:.3f} (RMS={best_rms:.3f}% on {KEYS})\n")

    # ---- Banner (printed AFTER auto-λ so it shows the picked value) ----
    print("\n=== UNIVERSAL CHECKS FOR THE ANALYTIC CAUSAL–DIRAC MODEL (Dual-Mode) ===\n")
    print(f"MODE = '{MODE}'   (LAMBDA_REL={LAMBDA_REL} only used in MODE='analytic')")
    print(f"B2_SCALE = {B2_SCALE:.3f}")
    print(f"B3_SHAPE = {B3_SHAPE:.6f}")
    print(f"Hydrogen anchor: kappa0 = {kappa0_from_hydrogen():.6e}")
    print(f"β0 intercept (analytic) = 7/26 = {7/26:.9f}")
    print(f"Na anomaly (analytic)   = 1/28 = {1/28:.9f}\n")

    if args.print_coeffs:
        print(f"--- β(Z) shape coefficients ---")
        print(f"b0={B0_ANALYTIC:.6f}, b1={B1_SHAPE:.6f}, b2={B2_SHAPE:.6f}, "
              f"b2_scale={B2_SCALE:.3f}, b3={B3_SHAPE:.6f}")
        for Z in (19, 37, 55):
            bb = beta_base(Z)
            print(f"beta_base(Z={Z}) = {bb:.6f}")
        print()

    # ---- Baseline table (with chosen mode) ----
    print(f"--- Baseline (gamma_rel={GAMMA_REL}), MODE: {MODE} ---")
    header = "Element   Z  n   n*     β_eff     β_req     f_pred(GHz)  f_exp(GHz)   err%"
    print(header)
    print("-"*len(header))

    baseline = {}
    for name, d in alkali.items():
        Z, n, I, gI, f_exp = d['Z'], d['n'], d['I'], d['gI'], d['f_exp']

        f_pred, Zeff_scr, n_star, psi_final, A_over_h, beta_eff = predict_core(
            Z, n, I, gI, mode=MODE, lambda_rel=LAMBDA_REL, gamma_rel=GAMMA_REL, use_bump=USE_BUMP
        )

        # f0 with β=0 (no bump), to compute required beta from f = f0 * Z^{-3β}
        f0, *_ = predict_with_explicit_beta(
            Z, n, I, gI, beta=0.0, gamma_rel=GAMMA_REL,
            lambda_rel=(1.0 if MODE=="calibrated" else LAMBDA_REL)
        )
        beta_req = (np.log(f0 / f_exp) / (3.0 * np.log(Z))) if (f0 > 0 and Z > 1) else np.nan

        err = (f_pred - f_exp)/f_exp * 100.0
        baseline[name] = {
            'Z': Z, 'n': n, 'I': I, 'gI': gI, 'f_exp': f_exp,
            'f_pred': f_pred, 'beta_eff': beta_eff, 'beta_req': beta_req,
            'n_star': n_star, 'Zeff_screened': Zeff_scr,
            'psi_final': psi_final, 'A_over_h': A_over_h,
            'err_pct': err
        }

        print(f"{name:8s} {Z:2d} {n:1d} {n_star:5.3f}  {beta_eff:8.6f}  {beta_req:8.6f}  "
              f"{(f_pred/1e9):12.3f}  {(f_exp/1e9):10.3f}  {err:7.2f}")

    # Summary
    rrms, mae = set_stats(KEYS, baseline)
    print(f"\n--- Summary on {KEYS} ---")
    print(f"RMS error: {rrms:.3f}%   MAE: {mae:.3f}%\n")

    print("Note: β_req(Z) computed from f0 (β=0) and f_exp via  f = f0 · Z^{-3β}.\n")

    # ---- Isotope scaling (same Z,n → electronic factor cancels) ----
    print("--- Isotope scaling (same Z,n → same electronic factor) ---")
    # Rb-87 → Rb-85
    f87 = baseline['Rb-87']['f_pred']
    ratio_rb = isotope_scale(
        f_ref=1.0,
        mu_ref=nuclear_mu['Rb-87'], I_ref=alkali['Rb-87']['I'],
        mu_iso=nuclear_mu['Rb-85'], I_iso=5/2
    )
    pred_f85 = f87 * ratio_rb
    print(f"Rb-87 → Rb-85: nuclear ratio = {ratio_rb:.6f};  pred f85 = {pred_f85/1e9:.6f} GHz")
    # K-39 → K-41
    f39 = baseline['K-39']['f_pred']
    ratio_k = isotope_scale(
        f_ref=1.0,
        mu_ref=nuclear_mu['K-39'], I_ref=alkali['K-39']['I'],
        mu_iso=nuclear_mu['K-41'], I_iso=3/2
    )
    pred_f41 = f39 * ratio_k
    print(f"K-39  → K-41 : nuclear ratio = {ratio_k:.6f};  pred f41 = {pred_f41/1e9:.6f} GHz")
    print("\n(As expected, the electronic factor cancels; ratios match the pure nuclear factor.)\n")

    # ---- Relativistic sensitivity ----
    print(f"--- Relativistic sensitivity: Δf/f vs γ_rel (baseline γ={GAMMA_REL}) ---")
    for key in ['K-39', 'Rb-87', 'Cs-133']:
        Z, n, I, gI = alkali[key]['Z'], alkali[key]['n'], alkali[key]['I'], alkali[key]['gI']
        f_base, *_ = predict_core(Z, n, I, gI, mode=MODE, lambda_rel=LAMBDA_REL,
                                  gamma_rel=GAMMA_REL, use_bump=USE_BUMP)
        rows = []
        for g in [1.8, 1.9, 2.0, 2.1, 2.2]:
            f_g, *_ = predict_core(Z, n, I, gI, mode=MODE, lambda_rel=LAMBDA_REL,
                                   gamma_rel=g, use_bump=USE_BUMP)
            rows.append((g, (f_g/f_base - 1.0)*100.0))
        row_str = "  ".join([f"γ={g:.1f}: {df:+6.3f}%" for g, df in rows])
        print(f"{key:8s}  {row_str}")

    # ---- Required β(Z) comparison (elements only) ----
    print("\n--- Required β(Z) comparison (all five) ---")
    print("Element   Z   β_eff(analytic)   β_req(from f0→f_exp)   Δβ")
    print("------------------------------------------------------------")
    for name, d in alkali.items():
        Z = d['Z']
        beta_eff = baseline[name]['beta_eff']
        f0, *_ = predict_with_explicit_beta(
            Z, d['n'], d['I'], d['gI'], beta=0.0, gamma_rel=GAMMA_REL,
            lambda_rel=(1.0 if MODE=="calibrated" else LAMBDA_REL)
        )
        beta_req = (np.log(f0 / d['f_exp']) / (3.0 * np.log(Z))) if (f0 > 0 and Z > 1) else np.nan
        delta = beta_eff - beta_req
        print(f"{name:8s} {Z:2d}     {beta_eff:10.6f}          {beta_req:10.6f}   {delta: .3e}")

    # ---- Residuals “plot” (ASCII bars) on heavy set ----
    print("\n--- Residuals (heavy set) ---")
    for key in HEAVY_SET:
        e = baseline[key]['err_pct']
        print(f"{key:8s} {e:+7.2f}%  {ascii_bar(e)}")

    # ---- Optional per-element DETAIL ----
    if args.detail:
        key = args.detail
        if key in alkali:
            Z, n, I, gI = alkali[key]['Z'], alkali[key]['n'], alkali[key]['I'], alkali[key]['gI']
            f_pred, Zeff_scr, n_star, psi_final, A_over_h, beta_eff = predict_core(
                Z, n, I, gI, mode=MODE, lambda_rel=LAMBDA_REL, gamma_rel=GAMMA_REL, use_bump=USE_BUMP
            )
            psi_NR = psi0_from_causal_NR(Z, n_star)
            Zeff_bare = (psi_NR * np.pi * a0**3 * n_star**3)**(1.0/3.0)

            print(f"\n--- DETAIL: {key} ---")
            print(f"Z={Z}, n={n}, n*={n_star:.6f}, β_eff={beta_eff:.6f}")
            print(f"Zeff_bare (NR from causal)  = {Zeff_bare:.6f}")
            print(f"Zeff_screened               = {Zeff_scr:.6f}")
            print(f"|psi(0)|^2 (final, rel) [m^-3] = {psi_final:.6e}")
            print(f"A/h [Hz] = {A_over_h:.6e}")
            print(f"f_pred [GHz] = {f_pred/1e9:.6f}\n")
        else:
            print(f"\n[detail] Unknown element key: {key}\n")

    # ---- Optional CSV output ----
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Element","Z","n","n_star","beta_eff","beta_req","f_pred_GHz","f_exp_GHz","err_percent"])
            for name, row in baseline.items():
                Z, n, f_exp = row["Z"], row["n"], row["f_exp"]
                f_pred      = row["f_pred"]
                n_star      = row["n_star"]
                beta_eff    = row["beta_eff"]
                beta_req    = row["beta_req"]
                err         = row["err_pct"]
                w.writerow([
                    name, Z, n,
                    f"{n_star:.6f}", f"{beta_eff:.6f}", f"{beta_req:.6f}",
                    f"{f_pred/1e9:.6f}", f"{f_exp/1e9:.6f}", f"{err:.3f}"
                ])
        print(f"\n[CSV] Baseline written to {args.csv}\n")

    # ---- Optional JSON output ----
    if args.json:
        out = {
            "mode": MODE,
            "lambda_rel": LAMBDA_REL,
            "gamma_rel": GAMMA_REL,
            "use_bump": USE_BUMP,
            "kappa0": kappa0_from_hydrogen(),
            "beta_shape": {
                "b0": B0_ANALYTIC, "b1": B1_SHAPE, "b2": B2_SHAPE,
                "b2_scale": B2_SCALE, "b3": B3_SHAPE,
                "bump_amp": C_BETA_ANALYTIC, "bump_center": BUMP_Z0, "bump_sigma": BUMP_SIGMA
            },
            "nstar_fade": {"D1": D1, "D2": D2, "Zc": ZC, "p": P_FADE},
            "elements": {}
        }
        for name, row in baseline.items():
            out["elements"][name] = {
                "Z": row["Z"], "n": row["n"], "I": row["I"], "gI": row["gI"],
                "f_exp_Hz": row["f_exp"], "f_pred_Hz": row["f_pred"],
                "beta_eff": row["beta_eff"], "beta_req": row["beta_req"],
                "n_star": row["n_star"], "Zeff_screened": row["Zeff_screened"],
                "psi_final_m3": row["psi_final"], "A_over_h_Hz": row["A_over_h"],
                "err_percent": row["err_pct"]
            }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[JSON] Results written to {args.json}\n")

    # ---- Optional sweep table after baseline ----
    if MODE == "analytic" and not args.no_sweep and not args.auto_lambda and args.sweep_print:
        start, stop, num = args.lam_range
        lam_grid = np.linspace(start, stop, int(num))
        lambda_sweep("analytic", GAMMA_REL, USE_BUMP, lam_grid, KEYS, print_rows=True)

if __name__ == "__main__":
    main()
