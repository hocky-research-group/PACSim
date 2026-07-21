#!/usr/bin/env python
"""
MBAR cross-check of the Frenkel-Ladd dA2 (the spring-coupling part of A_sol), with conditioning
guardrails and a TI/MBAR hybrid.

``run_ti.py`` combines the Einstein spring-coupling windows by thermodynamic integration (TI):
dA2/NkT = -(1/N) integral_0^1 <U_spring^bare>_s ds, a smooth quadrature of per-window averages that
does NOT require the windows to overlap. This module recombines the SAME window data with the
Multistate Bennett Acceptance Ratio (MBAR) estimator [Shirts & Chodera, J. Chem. Phys. 129, 124105
(2008), doi:10.1063/1.2978177], which DOES use the inter-window overlap.

Key simplification: the PACS interaction energy U_PACS is identical in every window (only the spring
coupling s changes), so it is a per-sample constant across states and cancels in MBAR. The reduced
potential of sample n at state k is therefore simply u_k(x_n) = beta * s_k * U_spring^bare(x_n), read
directly from the logged restraint energy (bare = logged coupled energy / that window's s). The
endpoints s=0 (real solid) and s=1 (interacting Einstein crystal) are unsampled states and
dA2/NkT = [f(s=0) - f(s=1)] / N.

WHY GUARDRAILS ARE NEEDED. For an N-particle solid the total spring free energy spans O(N) kT, so
adjacent-window overlap is intrinsically ~ e^{-O(N)} -- often exactly zero for the weak-spring
(small-s) windows. With zero overlap the MBAR equations are ill-conditioned: the solution depends on
the solver's initial guess (we have seen an 0.8 NkT spread between init-from-TI-profile and
init-from-zeros for the same data). A single number from MBAR is then meaningless. This module
therefore:

  1. seeds MBAR with the TI free-energy profile f(s)=beta*int_0^s <U_bare> ds' (close to the solution;
     BAR init is NOT used -- it raises a BoundsError on zero-overlap data that pymbar 4.0.3 mishandles);
  2. inspects bidirectional adjacent-state overlap for the sampled INTERIOR bridges
     (s_1<->...<->s_K), and Kish effective-sample fractions for reweighting to the two unsampled
     ENDPOINTS (s=0 and s=1). PyMBAR's overlap-matrix column for an unsampled state is identically
     zero because N_k=0, so that matrix entry is not an endpoint-overlap diagnostic;
  3. reports, in order of preference:
       * FULL MBAR   if every sampled-state interior overlap and endpoint ESS fraction clears the floor;
       * HYBRID      if the interior is well bridged but endpoint ESS is too small -- then TI integrates the
                     tiny [0,s_1] and [s_K,1] segments and MBAR supplies the interior f(s_K)-f(s_1),
                     so "the error is only from the endpoints" and the hybrid is trustworthy;
       * neither     if an INTERIOR bridge is below the floor -- MBAR is ill-conditioned; the tool
                     flags it and directs you to add windows (run_ti.py --refine-mbar-tol), and only
                     the TI value should be trusted.

Usage:
    python mbar_analysis.py <run_ti_output_dir>          # e.g. .../ti
"""

import argparse
import json
import os
import numpy as np
import gsd.hoomd
from pymbar import MBAR

_KB = 0.0083144626  # kJ/mol/K
_DEFAULT_OVERLAP_FLOOR = 0.01
_SOLVER_PROTOCOL = ({"method": "adaptive", "options": {"maximum_iterations": 2000, "tolerance": 1e-10}},)


def read_bare_energies(trajectory, coupling, discard_first_frame=True):
    """Per-frame bare spring energy Lambda_E*sum d^2 (kJ/mol) from a window trajectory (= logged/s)."""
    with gsd.hoomd.open(trajectory, "r") as frames:
        coupled = np.array([float(np.asarray(fr.log["harmonic_restraint_energy"])[0]) for fr in frames])
    if discard_first_frame and coupled.size > 1:
        coupled = coupled[1:]
    return coupled / coupling


def _ti_profile(s_query, s_sorted, means, beta):
    """f(s) = beta * integral_0^s <U_bare> ds' for each s in s_query, from the sampled window means.

    <U_bare>(s) is linearly interpolated on the sampled nodes and flat-extrapolated past the ends."""
    s_nodes = np.concatenate(([0.0], s_sorted, [1.0]))
    g_nodes = np.concatenate(([means[0]], means, [means[-1]]))
    out = np.empty(len(s_query))
    for i, sk in enumerate(s_query):
        grid = np.linspace(0.0, float(sk), 96)
        out[i] = beta * np.trapezoid(np.interp(grid, s_nodes, g_nodes), grid)
    return out


def _ti_segment(a, b, s_sorted, means, beta):
    """beta * integral_a^b <U_bare> ds  (one endpoint segment of the TI, for the hybrid estimator)."""
    s_nodes = np.concatenate(([0.0], s_sorted, [1.0]))
    g_nodes = np.concatenate(([means[0]], means, [means[-1]]))
    grid = np.linspace(float(a), float(b), 128)
    return beta * np.trapezoid(np.interp(grid, s_nodes, g_nodes), grid)


def analyze(bare_list, s_values, n_particles, temperature_K, overlap_floor=_DEFAULT_OVERLAP_FLOOR,
            init_scale=1.0):
    """
    Single MBAR solve + conditioning analysis of the spring-coupling windows.

    :returns: dict with dA2_full, dA2_hybrid (both /NkT), an adjacent-quality vector over the states
        (endpoint ESS fraction, sampled-state overlaps, endpoint ESS fraction), the split into
        interior-overlap / endpoint-efficiency minima, a recommendation
        ('full' | 'hybrid' | 'refine'), the recommended dA2, and the sorted couplings.
    """
    n = int(n_particles)
    beta = 1.0 / (_KB * float(temperature_K))
    order = np.argsort(s_values)
    s_sorted = np.array(s_values, dtype=float)[order]
    bare_sorted = [np.asarray(bare_list[i]) for i in order]
    means = np.array([b.mean() for b in bare_sorted])

    # States: s=0 (unsampled) + sampled windows + s=1 (unsampled).
    s_states = np.array([0.0] + list(s_sorted) + [1.0])
    N_k = np.array([0] + [b.size for b in bare_sorted] + [0])
    all_bare = np.concatenate(bare_sorted)
    u_kn = beta * np.outer(s_states, all_bare)
    f_init = init_scale * _ti_profile(s_states, s_sorted, means, beta)

    mbar = MBAR(u_kn, N_k, initial_f_k=f_init, solver_protocol=_SOLVER_PROTOCOL)
    delta_f = mbar.compute_free_energy_differences(compute_uncertainty=False)["Delta_f"]
    i0, i1 = 0, len(s_states) - 1                 # s=0 and s=1
    k1, kK = 1, len(s_states) - 2                 # first and last SAMPLED state
    dA2_full = float(delta_f[i1, i0]) / n         # [f(0)-f(1)]/N

    # Hybrid: TI over [0,s_1] and [s_K,1]; MBAR interior f(s_K)-f(s_1) = Delta_f[k1,kK].
    seg_lo = _ti_segment(0.0, s_sorted[0], s_sorted, means, beta)     # f(s_1)-f(0)
    seg_hi = _ti_segment(s_sorted[-1], 1.0, s_sorted, means, beta)    # f(1)-f(s_K)
    f_interior = float(delta_f[k1, kK])                              # f(s_K)-f(s_1)
    dA2_hybrid = -(seg_lo + f_interior + seg_hi) / n                 # (f(0)-f(1))/N

    diagnostics_ok = False
    diagnostic_error = None
    try:
        overlap = mbar.compute_overlap()["matrix"]
        # Use the smaller direction for each pair of SAMPLED states. With equal N_k the matrix is
        # symmetric, but the explicit minimum remains valid if window lengths differ.
        interior_ovl = np.array([
            min(float(overlap[k, k + 1]), float(overlap[k + 1, k]))
            for k in range(1, overlap.shape[0] - 2)
        ])
        n_eff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)
        endpoint_eff = n_eff[[0, -1]] / float(all_bare.size)
        # Compatibility/reporting vector: one quality metric per adjacent bridge. Endpoint entries
        # are ESS fractions, while interior entries are bidirectional overlap probabilities.
        adjacent = np.concatenate((endpoint_eff[:1], interior_ovl, endpoint_eff[1:]))
        raw_adjacent = np.array([float(overlap[k, k + 1])
                                 for k in range(overlap.shape[0] - 1)])
        diagnostics_ok = True
    except Exception as exc:
        overlap, adjacent, raw_adjacent = None, np.array([]), np.array([])
        endpoint_eff, interior_ovl = np.array([]), np.array([])
        diagnostic_error = f"{type(exc).__name__}: {exc}"

    # Endpoint quality is importance-reweighting ESS, not an overlap-matrix entry: an unsampled
    # state's overlap column is zero by construction. Interior quality remains adjacent overlap.
    endpoint_ovl = endpoint_eff
    min_interior = float(interior_ovl.min()) if interior_ovl.size else 1.0
    min_endpoint = float(endpoint_ovl.min()) if endpoint_ovl.size else 0.0

    if not diagnostics_ok:
        recommendation, dA2 = "refine", float("nan")
    elif min_interior >= overlap_floor and min_endpoint >= overlap_floor:
        recommendation, dA2 = "full", dA2_full
    elif min_interior >= overlap_floor:
        recommendation, dA2 = "hybrid", dA2_hybrid
    else:
        recommendation, dA2 = "refine", float("nan")

    return {"dA2_full": dA2_full, "dA2_hybrid": dA2_hybrid, "dA2": dA2,
            "recommendation": recommendation, "adjacent": adjacent, "interior_overlap": interior_ovl,
            "endpoint_overlap": endpoint_ovl, "min_interior_overlap": min_interior,
            # Keep min_endpoint_overlap as a compatibility alias for older result consumers.
            "min_endpoint_overlap": min_endpoint, "min_endpoint_efficiency": min_endpoint,
            "overlap_matrix": overlap, "s_sorted": s_sorted,
            "endpoint_efficiency": endpoint_eff, "raw_adjacent_overlap": raw_adjacent,
            "overlap_floor": overlap_floor, "diagnostics_ok": diagnostics_ok,
            "diagnostic_error": diagnostic_error}


def delta_a2_mbar(bare_list, s_values, n_particles, temperature_K, n_bootstraps=0):
    """Back-compat wrapper returning full dA2, uncertainty, overlap matrix, states, and quality."""
    r = analyze(bare_list, s_values, n_particles, temperature_K)
    return r["dA2_full"], float("nan"), r["overlap_matrix"], r["s_sorted"], r["adjacent"]


def analyze_output_dir(output_dir, discard_first_frame=True, overlap_floor=_DEFAULT_OVERLAP_FLOOR):
    """Run analyze() on a run_ti output directory; returns (analysis_dict, result_json)."""
    result = json.load(open(os.path.join(output_dir, "free_energy.json")))
    rows = sorted(result["windows_data"], key=lambda r: r["lambda_ein"])
    s_windows = [float(r["lambda_ein"]) for r in rows]
    bare = [read_bare_energies(
        os.path.join(output_dir, "windows", f"window_{int(r['window']):02d}", "trajectory.gsd"),
        r["lambda_ein"], discard_first_frame) for r in rows]
    r = analyze(bare, s_windows, result["n_particles"], result["temperature_K"], overlap_floor)
    return r, result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="run_ti.py output directory (contains free_energy.json + windows/)")
    parser.add_argument("--keep-first-frame", action="store_true")
    parser.add_argument("--overlap-floor", type=float, default=_DEFAULT_OVERLAP_FLOOR,
                        help="minimum sampled-state overlap or endpoint ESS fraction for a bridge "
                             "to count as usable (default 0.01)")
    parser.add_argument("--check-init", action="store_true",
                        help="also solve from a perturbed initial guess and report the init-dependence "
                             "(a direct conditioning test; slow, doubles the solve time)")
    args = parser.parse_args()

    r, result = analyze_output_dir(args.output_dir, not args.keep_first_frame, args.overlap_floor)
    dA2_ti = float(result["dA2_NkT"])
    a0, dA1 = float(result["A0_NkT"]), float(result["dA1_NkT"])
    a_sol_ti = float(result["A_sol_NkT"])

    print(f"MBAR cross-check: {args.output_dir}")
    print(f"  N = {result['n_particles']};  windows = {len(result['windows_data'])};  "
          f"overlap floor = {r['overlap_floor']}")
    print(f"  dA2/NkT :  TI = {dA2_ti:+.4f}   full-MBAR = {r['dA2_full']:+.4f}   "
          f"hybrid = {r['dA2_hybrid']:+.4f}")
    if r["adjacent"].size:
        print(f"  adjacent quality (endpoint ESS fraction; interior overlap; endpoint ESS fraction): "
              + " ".join(f"{o:.3f}" for o in r["adjacent"]))
        print(f"    min interior bridge = {r['min_interior_overlap']:.3f}   "
              f"min endpoint ESS fraction = {r['min_endpoint_overlap']:.3f}")

    if args.check_init:
        r2 = analyze(  # perturbed init (0.5x TI profile) to expose ill-conditioning
            [read_bare_energies(os.path.join(args.output_dir, "windows",
             f"window_{int(row['window']):02d}", "trajectory.gsd"), row["lambda_ein"],
             not args.keep_first_frame) for row in sorted(result["windows_data"], key=lambda x: x["lambda_ein"])],
            [float(row["lambda_ein"]) for row in sorted(result["windows_data"], key=lambda x: x["lambda_ein"])],
            result["n_particles"], result["temperature_K"], args.overlap_floor, init_scale=0.5)
        print(f"  init-dependence (TI-profile vs 0.5x): full {abs(r['dA2_full']-r2['dA2_full']):.4f}  "
              f"hybrid {abs(r['dA2_hybrid']-r2['dA2_hybrid']):.4f} NkT  "
              f"(large => ill-conditioned)")

    rec = r["recommendation"]
    if rec == "full":
        val = r["dA2_full"]; a_sol = a0 + dA1 + val
        print(f"  => well conditioned. Report FULL MBAR: dA2 = {val:+.4f}, A_sol = {a_sol:+.4f} "
              f"(TI {a_sol_ti:+.4f}; |diff| {abs(a_sol-a_sol_ti):.4f} NkT).")
    elif rec == "hybrid":
        val = r["dA2_hybrid"]; a_sol = a0 + dA1 + val
        print(f"  => interior well bridged, endpoints are not: the error is only from the endpoints, so "
              f"report the HYBRID (TI endpoints + MBAR interior): dA2 = {val:+.4f}, A_sol = {a_sol:+.4f} "
              f"(TI {a_sol_ti:+.4f}; |diff| {abs(a_sol-a_sol_ti):.4f} NkT).")
    elif not r["diagnostics_ok"]:
        print(f"  => MBAR QUALITY DIAGNOSTICS FAILED: {r['diagnostic_error']}. "
              f"No MBAR estimate is reportable; trust TI (dA2 = {dA2_ti:+.4f}).")
    else:
        print(f"  => ILL-CONDITIONED: an interior bridge is below the floor "
              f"({r['min_interior_overlap']:.3f} < {r['overlap_floor']}). MBAR cannot bridge the "
              f"weak-spring windows; its value is initialization-dependent and NOT trustworthy. "
              f"Add windows there (run_ti.py --refine-mbar-tol) and trust TI (dA2 = {dA2_ti:+.4f}) meanwhile.")


if __name__ == "__main__":
    main()
