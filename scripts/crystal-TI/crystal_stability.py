#!/usr/bin/env python
"""
Simple crystal-stability analysis for a PACSim (periodic) trajectory.

Two structure-agnostic signals are used together (both must pass the configured thresholds; they are
reported separately because different crystals give clearer signals in different metrics):

  * Radial distribution function g(r): compared between the start and the end of the trajectory.
    A crystal keeps sharp peaks at fixed positions, so the start and end g(r) stay highly correlated;
    on melting the peaks broaden/merge and the correlation drops. This is the primary signal and
    works even for crystals (e.g. Th3P4) whose bond-orientational order gives a weak Q6.
  * Lindemann parameter sqrt(<|r - r0|^2>) / d_nn: the rms displacement of the particles from their
    initial lattice sites in units of the nearest-neighbor distance. Lindemann's rule of thumb is
    that a solid melts around 0.1-0.15.

Box breathing (NPT): to keep both signals sensitive to *structural* change rather than to a change
in box size, every frame is first mapped into the reference (initial) box via its fractional
coordinates (an affine rescaling). Under a constant-volume run this is a no-op; under an NPT run it
removes the trivial shift of the g(r) peaks and the affine part of the displacement that come purely
from the box changing size, so a crystal that merely breathes with the barostat is still recognized
as stable.

The global Steinhardt Q6 is also printed for reference, but it is NOT used in the verdict because it
is a poor indicator for some crystal structures.

Usage:
    python crystal_stability.py trajectory.gsd [--rdf-rmax-factor 4] [--rdf-correlation 0.9]
                                               [--lindemann 0.15] [--neighbors 8]
"""

import argparse
import numpy as np
import gsd.hoomd
import freud


def _global_q6(box, positions, n_neighbors):
    steinhardt = freud.order.Steinhardt(l=6)
    steinhardt.compute((box, positions), neighbors={"num_neighbors": n_neighbors, "exclude_ii": True})
    return float(steinhardt.order)


def _nearest_neighbor_distance(box, reference):
    aabb = freud.locality.AABBQuery(box, reference)
    neighbors = aabb.query(reference, {"num_neighbors": 1, "exclude_ii": True})
    return float(np.sqrt(np.median([distance ** 2 for (_, _, distance) in neighbors])))


def _positions_in_reference_box(frame, reference_box):
    """Map a frame's positions into the reference box via fractional coordinates (affine rescaling)."""
    box = freud.box.Box.from_box(frame.configuration.box)
    positions = np.array(frame.particles.position, dtype=float)
    return reference_box.make_absolute(box.make_fractional(positions))


def _averaged_rdf(frames, reference_box, r_max, bins):
    """Average g(r) over the given frames (rescaled into the reference box) using one accumulator."""
    rdf = freud.density.RDF(bins=bins, r_max=r_max)
    for frame in frames:
        rdf.compute((reference_box, _positions_in_reference_box(frame, reference_box)), reset=False)
    return rdf.bin_centers.copy(), rdf.rdf.copy()


def analyze(trajectory_filename, rdf_rmax_factor=4.0, rdf_correlation_threshold=0.9,
            lindemann_threshold=0.15, n_neighbors=8, rdf_bins=120):
    with gsd.hoomd.open(trajectory_filename, "r") as traj:
        frames = list(traj)
    if len(frames) < 2:
        raise ValueError("Need at least two frames to assess stability.")
    reference = np.array(frames[0].particles.position, dtype=float)
    box0 = freud.box.Box.from_box(frames[0].configuration.box)
    d_nn = _nearest_neighbor_distance(box0, reference)

    # --- per-frame Q6 and Lindemann (displacement measured in the reference box) ---
    rows = []
    for frame in frames:
        box = freud.box.Box.from_box(frame.configuration.box)
        positions = np.array(frame.particles.position, dtype=float)
        q6 = _global_q6(box, positions, n_neighbors)  # bond angles are scale invariant
        positions_in_reference = _positions_in_reference_box(frame, box0)
        disp = box0.wrap(positions_in_reference - reference)
        lindemann = float(np.sqrt(np.mean(np.sum(disp ** 2, axis=1))) / d_nn)
        rows.append((int(frame.configuration.step), q6, lindemann))
    lindemann_max = max(r[2] for r in rows)

    # --- start vs end RDF (skip frame 0, the perfect lattice; average over thirds of the run) ---
    r_max = min(rdf_rmax_factor * d_nn, 0.49 * min(box0.L))
    third = max(1, (len(frames) - 1) // 3)
    start_frames = frames[1:1 + third]
    end_frames = frames[-third:]
    r, g_start = _averaged_rdf(start_frames, box0, r_max, rdf_bins)
    _, g_end = _averaged_rdf(end_frames, box0, r_max, rdf_bins)
    rdf_correlation = float(np.corrcoef(g_start, g_end)[0, 1])

    stable = (rdf_correlation >= rdf_correlation_threshold) and (lindemann_max <= lindemann_threshold)

    print(f"trajectory: {trajectory_filename}")
    print(f"N = {frames[0].particles.N};  d_nn = {d_nn:.3f};  frames = {len(frames)};  r_max = {r_max:.3f}")
    print(f"{'step':>10} {'Q6(info)':>9} {'Lindemann':>11}")
    for step, q6, lindemann in rows:
        print(f"{step:>10} {q6:>9.3f} {lindemann:>11.4f}")
    print(f"\nStart-vs-end g(r) correlation = {rdf_correlation:.4f} "
          f"(threshold {rdf_correlation_threshold}); peak g(r): start {g_start.max():.2f}, end {g_end.max():.2f}")
    print(f"Lindemann parameter: maximum {lindemann_max:.4f} (threshold {lindemann_threshold})")
    print(f"VERDICT: crystal is {'STABLE (stays ordered)' if stable else 'NOT stable (melted/disordered)'}")
    return {"stable": stable, "rdf_correlation": rdf_correlation, "lindemann_max": lindemann_max,
            "r": r, "g_start": g_start, "g_end": g_end, "rows": rows, "d_nn": d_nn}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory", help="GSD trajectory file")
    parser.add_argument("--rdf-rmax-factor", type=float, default=4.0,
                        help="RDF r_max as a multiple of the nearest-neighbor distance")
    parser.add_argument("--rdf-correlation", type=float, default=0.9,
                        help="start-vs-end g(r) correlation must stay above this to be 'stable'")
    parser.add_argument("--lindemann", type=float, default=0.15,
                        help="Lindemann parameter must stay below this to be 'stable'")
    parser.add_argument("--neighbors", type=int, default=8, help="neighbors used for the (informational) Q6")
    args = parser.parse_args()
    analyze(args.trajectory, args.rdf_rmax_factor, args.rdf_correlation, args.lindemann, args.neighbors)


if __name__ == "__main__":
    main()
