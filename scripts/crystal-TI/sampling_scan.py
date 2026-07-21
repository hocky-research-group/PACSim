#!/usr/bin/env python
"""
How long should each TI window run? -- a sub-sampling convergence test.

Given a finished ``run_ti.py`` output directory, this uses the first fraction f of each window's
frames (f = 1/8, 1/4, 1/2, 3/4, 1) and recomputes both estimators of dA2:

  * dA2_TI   -- the Gauss-Legendre quadrature of the per-window <U_spring^bare>_s (does NOT need
                inter-window overlap; its error is purely the statistical error of the window means).
  * dA2_MBAR -- the multistate estimate (needs overlap; biased when windows don't overlap).

It reports, versus f:
  * dA2_TI with its propagated statistical error (from each window's block/autocorrelation-based
    standard error), and
  * the |dA2_MBAR - dA2_TI| gap.

Interpretation -- which knob to turn:
  * If dA2_TI's statistical error and the MBAR-TI gap both shrink like ~1/sqrt(f) (i.e. ~1/sqrt(t)),
    the residual is STATISTICAL: run each window LONGER.
  * If the MBAR-TI gap PLATEAUS while dA2_TI is already converged, the residual is the MBAR
    OVERLAP BIAS: add MORE windows (run_ti.py --refine-mbar-tol), not longer runs. For an N-particle
    solid the per-window free-energy gaps are O(N), so MBAR overlap is intrinsically poor and needs
    many windows -- this is why TI is the estimator of choice for solids.

How long to run (rationale): the TI dA2 error is
    sigma(dA2)/NkT = sqrt( sum_k c_k^2 * se_k^2 ) ,   se_k = std(U_bare^k)/sqrt(N_k/g_k) ,
where c_k = (Gauss-Legendre weight_k * ds/dw_k)/(N kT) and g_k is the statistical inefficiency
(~2*tau+1) of window k. It is dominated by the weak-spring (small-s) windows. Pick the production
length so the dominant window has enough independent samples that sigma(dA2) is below your target;
the printed per-window (g_k, N_eff) show which windows limit it.

Usage:
    python sampling_scan.py <run_ti_output_dir>
"""

import argparse
import json
import os
import sys
import numpy as np
from openmm import unit

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from free_energy import FrenkelLaddSchedule  # noqa: E402
from colloids.units import energy_unit, length_unit, temperature_unit  # noqa: E402
import mbar_analysis  # noqa: E402
from pymbar import timeseries  # noqa: E402

_KB = 0.0083144626  # kJ/mol/K
_spring_constant_unit = energy_unit / (length_unit ** 2)


def _statistical_inefficiency(series):
    try:
        return float(timeseries.statistical_inefficiency(series))
    except Exception:
        return 1.0


def scan(output_dir, fractions=(0.125, 0.25, 0.5, 0.75, 1.0)):
    result = json.load(open(os.path.join(output_dir, "free_energy.json")))
    n = int(result["n_particles"])
    kt = _KB * float(result["temperature_K"])
    temperature = float(result["temperature_K"]) * temperature_unit
    spring_constant = float(result["lambda_E_kJ_per_mol_nm2"]) * _spring_constant_unit
    rows = sorted(result["windows_data"], key=lambda r: r["lambda_ein"])
    s_vals = [float(r["lambda_ein"]) for r in rows]
    bare_full = [mbar_analysis.read_bare_energies(
        os.path.join(output_dir, "windows", f"window_{int(r['window']):02d}", "trajectory.gsd"),
        r["lambda_ein"]) for r in rows]

    # Gauss-Legendre TI schedule -> per-window quadrature coefficients c_k (for error propagation).
    schedule = FrenkelLaddSchedule(n_points=len(rows), temperature=temperature,
                                   spring_constant=spring_constant)
    coeff = schedule._gl_weights * schedule._ds_dw / (n * kt)   # dA2 = -sum(coeff_k * mean_k)

    # Per-window statistical inefficiency and effective sample size at full length.
    g = np.array([_statistical_inefficiency(b) for b in bare_full])
    n_eff = np.array([len(b) / gk for b, gk in zip(bare_full, g)])

    print(f"Sampling-convergence scan: {output_dir}")
    print(f"  N = {n};  windows = {len(rows)};  frames/window = {len(bare_full[0])}")
    print("  per-window statistical inefficiency g (~2*tau+1) and N_eff at full length:")
    dom = int(np.argmax(coeff ** 2 * (np.array([b.std() for b in bare_full]) ** 2) / n_eff))
    for k, (s, gk, ne) in enumerate(zip(s_vals, g, n_eff)):
        mark = "  <-- dominates dA2 error" if k == dom else ""
        print(f"    s={s:.4e}  g={gk:5.1f}  N_eff={ne:6.1f}{mark}")

    print(f"\n  {'fraction':>8} {'frames':>7} {'dA2_TI':>10} {'sigma(TI)':>10} {'dA2_MBAR':>10} {'|MBAR-TI|':>10}")
    for f in fractions:
        bare_f = [b[:max(2, int(round(f * len(b))))] for b in bare_full]
        means = np.array([b.mean() for b in bare_f])                       # <U_bare> per window (kJ/mol)
        dA2_ti = schedule.integrate_delta_a2_per_particle([m * energy_unit for m in means], n)
        # statistical error of dA2_TI from block/autocorrelation standard errors of each window mean
        se = np.array([b.std() / np.sqrt(max(1.0, len(b) / _statistical_inefficiency(b)))
                       for b in bare_f])
        sigma_ti = float(np.sqrt(np.sum((coeff * se) ** 2)))
        dA2_mbar = mbar_analysis.delta_a2_mbar(bare_f, s_vals, n, float(result["temperature_K"]))[0]
        print(f"  {f:>8.3f} {len(bare_f[0]):>7d} {dA2_ti:>10.4f} {sigma_ti:>10.4f} "
              f"{dA2_mbar:>10.4f} {abs(dA2_mbar - dA2_ti):>10.4f}")

    print("\n  Read-off: if sigma(TI) and |MBAR-TI| both fall ~1/sqrt(fraction) -> longer runs help "
          "(statistical).\n  If |MBAR-TI| plateaus while sigma(TI) is small -> add windows "
          "(--refine-mbar-tol), not length (overlap bias).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", help="finished run_ti.py output directory (free_energy.json + windows/)")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5, 0.75, 1.0])
    args = p.parse_args()
    scan(args.output_dir, args.fractions)


if __name__ == "__main__":
    main()
