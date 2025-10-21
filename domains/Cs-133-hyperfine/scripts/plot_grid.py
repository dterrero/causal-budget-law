#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_rms_from_json(path):
    """Load RMS data from JSON output (baseline or sweep grid)."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_rms_grids(data_heavy, data_all):
    """
    Build λ–b3 RMS arrays for heavy and all sets.
    JSON must include nested 'rms_grid' with same λ,b3 axes.
    """
    lam_grid = np.array(data_heavy["lambda_grid"])
    b3_grid  = np.array(data_heavy["b3_grid"])
    RMS_heavy = np.array(data_heavy["rms_grid"])
    RMS_all   = np.array(data_all["rms_grid"])

    # Ensure correct orientation (b3 rows, λ columns)
    if RMS_heavy.shape != (len(b3_grid), len(lam_grid)):
        RMS_heavy = RMS_heavy.T
    if RMS_all.shape != (len(b3_grid), len(lam_grid)):
        RMS_all = RMS_all.T

    return lam_grid, b3_grid, RMS_heavy, RMS_all


def plot_rms_grids(lam_grid, b3_grid, RMS_heavy, RMS_all,
                   best_lam, best_b3, best_heavy, best_all,
                   save_pdf=True):
    """Plot dual contour maps for heavy/all RMS grids."""
    L, B3 = np.meshgrid(lam_grid, b3_grid)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, data, title, best_val in zip(
        axes,
        [RMS_heavy, RMS_all],
        [f"Heavy Set RMS (min={best_heavy:.3f}%)",
         f"All-Five Set RMS (min={best_all:.3f}%)"],
        [best_heavy, best_all],
    ):
        cont = ax.contourf(L, B3, data, levels=20, cmap="viridis")
        lines = ax.contour(L, B3, data, colors="white", linewidths=0.6)
        ax.clabel(lines, inline=True, fontsize=8, fmt="%.4f")

        # Highlight best point
        ax.contour(L, B3, data, levels=[best_val], colors="red",
                   linewidths=1.2, linestyles="--")
        ax.plot(best_lam, best_b3, "r*", markersize=10)
        ax.text(best_lam + 0.0004, best_b3 - 0.0015,
                f"{best_val:.3f}%", color="red", fontsize=8, weight="bold")

        # Axis labels and annotations
        ax.text(lam_grid.min() + 0.0004, b3_grid.min() + 0.001,
                f"Best λ={best_lam:.3f}, b3={best_b3:.3f}",
                fontsize=8, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        ax.set_xlabel(r"$\lambda_{\mathrm{rel}}$")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(r"$b_3$ (shape parameter)")

    # Colorbar referencing both panels
    cbar = fig.colorbar(axes[1].collections[0], ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("RMS error (%)")

    fig.suptitle("Causal–Dirac Analytic RMS Error Grids", fontsize=13, weight="bold")
    plt.subplots_adjust(top=0.88, wspace=0.15)

    if save_pdf:
        plt.savefig("rms_contours_dual.pdf", bbox_inches="tight")

    plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot RMS contour grids from Causal–Dirac JSON outputs")
    p.add_argument("--heavy", required=True, help="Path to JSON file for heavy-set RMS data")
    p.add_argument("--all", required=True, help="Path to JSON file for all-set RMS data")
    p.add_argument("--no-save", action="store_true", help="Do not save the PDF file")
    args = p.parse_args()

    heavy = load_rms_from_json(args.heavy)
    allset = load_rms_from_json(args.all)

    lam_grid, b3_grid, RMS_heavy, RMS_all = build_rms_grids(heavy, allset)

    best_lam   = heavy["best"]["lambda"]
    best_b3    = heavy["best"]["b3"]
    best_heavy = heavy["best"]["rms"]
    best_all   = allset["best"]["rms"]

    plot_rms_grids(lam_grid, b3_grid, RMS_heavy, RMS_all,
                   best_lam, best_b3, best_heavy, best_all,
                   save_pdf=not args.no_save)


if __name__ == "__main__":
    main()