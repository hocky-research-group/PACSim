#!/usr/bin/env python
"""
Diagnostic plots for a Frenkel-Ladd / Einstein-crystal thermodynamic-integration run.

Given the output directory of ``run_ti.py`` (containing ``free_energy.json``, the per-window
``window_XX/trajectory.gsd`` files, and ``delta_a1_samples.npz``), this produces a single
four-panel figure ``ti_diagnostics.png`` that lets you check the TI is well behaved:

  (A) dA2 integrand   : <U_spring(bare)>/NkT vs the Einstein coupling s (the quantity integrated in
                        Eq. dA2). Should be smooth and monotonically decreasing with s.
  (B) window overlap  : per-frame distributions of the bare spring energy for every window, overlaid
                        and colored by s. Overlap is useful for the optional MBAR cross-check; the
                        quadrature TI estimator itself does not require inter-window overlap.
  (C) convergence     : running (cumulative) mean of the bare spring energy per window, normalized to
                        its final value. Each curve should flatten to 1 well before the end.
  (D) dA1 reweighting : histogram of beta*(U_sol - U_lattice) over the ideal-Einstein Gaussian
                        samples, with the effective sample fraction. A narrow distribution => the
                        ideal -> interacting reweighting is well conditioned (small log correction).

Usage (standalone, re-plot without rerunning the TI):
    python ti_diagnostics.py path/to/ti_output_dir
"""

import json
import os
import sys
import numpy as np
import gsd.hoomd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

_KB = 0.0083144626  # kJ/mol/K (Boltzmann constant in the PACSim molar unit system)


def _window_bare_series(output_dir, window_index, coupling, n, kt, discard_first_frame=True):
    """Per-frame bare Einstein energy U_bare/NkT for one window (bare = coupled restraint / s)."""
    traj = os.path.join(output_dir, "windows", f"window_{window_index:02d}", "trajectory.gsd")
    with gsd.hoomd.open(traj, "r") as frames:
        coupled = np.array([float(np.asarray(fr.log["harmonic_restraint_energy"])[0]) for fr in frames])
    if discard_first_frame and coupled.size > 1:
        coupled = coupled[1:]
    return (coupled / coupling) / (n * kt)


def make_diagnostics(output_dir, filename="diagnostics.png"):
    """Write the four-panel TI diagnostics figure; return its path."""
    result = json.load(open(os.path.join(output_dir, "free_energy.json")))
    n = int(result["n_particles"])
    kt = _KB * float(result["temperature_K"])
    rows = sorted(result["windows_data"], key=lambda r: r["lambda_ein"])
    couplings = np.array([r["lambda_ein"] for r in rows])
    bare_mean = np.array([r["bare_spring_energy_kJmol"] for r in rows]) / (n * kt)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    cmap = matplotlib.colormaps["viridis"]
    norm = LogNorm(vmin=couplings.min(), vmax=couplings.max())

    # (A) dA2 integrand -----------------------------------------------------------------------
    axA = axes[0, 0]
    axA.plot(couplings, bare_mean, "-", color="0.6", zorder=1)
    axA.scatter(couplings, bare_mean, c=couplings, cmap=cmap, norm=norm, s=45, zorder=2,
                edgecolor="k", linewidth=0.4)
    axA.set_xscale("log")
    axA.set_xlabel(r"Einstein coupling $s$")
    axA.set_ylabel(r"$\langle U_{\rm spring}^{\rm bare}\rangle_s / Nk_BT$")
    axA.set_title(r"(A) $\Delta A_2$ integrand (Gauss-Legendre nodes)")
    axA.grid(alpha=0.3)

    # (B) window overlap ----------------------------------------------------------------------
    axB = axes[0, 1]
    series = []
    for r in rows:
        s = _window_bare_series(output_dir, int(r["window"]), r["lambda_ein"], n, kt)
        series.append(s)
    # Bare spring energy spans decades (steep integrand at small s), so use log-spaced bins/axis
    # to make the overlap between neighbouring windows visible across the whole range.
    lo = max(1e-3, min(float(s[s > 0].min()) for s in series if np.any(s > 0)))
    hi = max(float(s.max()) for s in series)
    bins = np.logspace(np.log10(lo), np.log10(hi), 70)
    for r, s in zip(rows, series):
        axB.hist(s[s > 0], bins=bins, density=True, histtype="stepfilled", alpha=0.4,
                 color=cmap(norm(r["lambda_ein"])))
    axB.set_xscale("log")
    axB.set_xlabel(r"$U_{\rm spring}^{\rm bare} / Nk_BT$  (per frame, log scale)")
    axB.set_ylabel("probability density")
    axB.set_title("(B) window overlap (needed for MBAR, not TI)")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axB, label=r"coupling $s$")

    # (C) per-window convergence --------------------------------------------------------------
    axC = axes[1, 0]
    for r, s in zip(rows, series):
        run_mean = np.cumsum(s) / np.arange(1, s.size + 1)
        axC.plot(np.linspace(0, 1, s.size), run_mean / run_mean[-1],
                 color=cmap(norm(r["lambda_ein"])), lw=1.0)
    axC.axhline(1.0, color="k", lw=0.8, ls="--")
    axC.set_ylim(0.6, 1.4)
    axC.set_xlabel("fraction of production")
    axC.set_ylabel(r"running mean / final mean")
    axC.set_title("(C) per-window convergence")
    axC.grid(alpha=0.3)

    # (D) dA1 reweighting ---------------------------------------------------------------------
    axD = axes[1, 1]
    npz_path = os.path.join(output_dir, "delta_a1_samples.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        samples = d["u_sol_samples"]; u_lat = float(d["u_lattice"]); temperature = float(d["temperature_K"])
        beta = 1.0 / (_KB * temperature)
        expo = beta * (samples - u_lat)           # exponent in <exp[-beta(U_sol-U_lattice)]>
        w = np.exp(-(expo - expo.min())); ess = (w.sum() ** 2) / (w ** 2).sum() / w.size
        axD.hist(expo, bins=50, density=True, color="tab:purple", alpha=0.7)
        axD.axvline(expo.mean(), color="k", ls="--", lw=1.0, label=f"mean = {expo.mean():.2f}")
        axD.set_xlabel(r"$\beta\,(U_{\rm sol}-U_{\rm lattice})$  (ideal-Einstein samples)")
        axD.set_ylabel("probability density")
        axD.set_title(f"(D) $\\Delta A_1$ reweighting  (ESS = {100*ess:.0f}%)")
        axD.legend(fontsize=8)
    else:
        axD.text(0.5, 0.5, "delta_a1_samples.npz not found", ha="center", va="center")
        axD.set_axis_off()

    logcorr = result.get("dA1_log_correction_NkT", float("nan"))
    fig.suptitle(f"TI diagnostics: A_sol/NkT = {result['A_sol_NkT']:.3f}  "
                 f"(A0 {result['A0_NkT']:.2f}, dA1 {result['dA1_NkT']:.2f}, dA2 {result['dA2_NkT']:.2f}; "
                 f"N={n}, dA1 log corr {logcorr:+.3f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(output_dir, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python ti_diagnostics.py <run_ti_output_dir>")
    print("wrote", make_diagnostics(sys.argv[1]))
