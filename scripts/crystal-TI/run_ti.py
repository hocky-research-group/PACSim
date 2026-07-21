#!/usr/bin/env python
"""
End-to-end Frenkel-Ladd / Einstein-crystal free-energy driver for a PACS colloidal crystal.

Given a PACS run YAML (the physics parameters plus an initial *equilibrated* crystal configuration),
this script computes the Helmholtz free energy A_sol of the crystal by thermodynamic integration
from an ideal Einstein-crystal reference:

    A_sol = A0 + dA1 + dA2                                    (Vega et al. 2008; see free_energy.py)

  * A0  - analytic ideal-Einstein free energy: fixed-COM Einstein crystal plus the COM-release
          term (Vega eq. 48), so the assembled A_sol is the UNCONSTRAINED solid free energy.
  * dA2 - Gauss-Legendre thermodynamic integration over the Einstein spring strength. Each window is
          run as a separate `pacsim-run run.yaml -t ti.yaml` invocation (one coupling per window);
          the mean restraint energy is read back from each window's GSD trajectory log.
  * dA1 - ideal -> interacting Einstein crystal, computed by directly Gaussian-sampling the ideal
          Einstein crystal (independent Gaussians about the lattice, centre-of-mass projected out)
          and scoring the PACS interaction energy on those configurations with a PACS-only context.
          No PACS-scaling / interaction switch is needed.

The spring constant Lambda_E is chosen from a small target Einstein displacement -- by default a
fraction of the crystal's MEASURED unrestrained rms displacement (auto-tuned per structure), or of
the interaction decay length min(Debye, brush) under --no-autotune-spring -- which keeps the dA1
reweighting well-conditioned. The method is validated against the Lennard-Jones Frenkel-Ladd
benchmark (A_sol = 3.104 vs the literature value 3.11 N kB T).

Platform note: the Einstein restraint now uses periodicdistance and is stable on the CUDA and OpenCL
platforms (an earlier naive restraint expression was discontinuous across periodic images and diverged
on GPUs). The TI windows still default to CPU for reproducibility, but you
can run them on a GPU with --window-platform CUDA. The PACS-only dA1 energy evaluations have no
restraint and default to a GPU (OpenCL locally; pass --energy-platform CUDA on a CUDA machine).

Usage:
    python run_ti.py run.yaml --output-dir ti_run [--windows 12] [--equil-steps 20000]
                              [--prod-steps 60000] [--sample-interval 200]
"""

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import gsd.hoomd
from openmm import unit

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from free_energy import (ideal_einstein_free_energy_per_particle, delta_a1_per_particle,  # noqa: E402
                         FrenkelLaddSchedule, FreeEnergyResult)
from colloids.run_parameters import RunParameters  # noqa: E402
from colloids.ti_parameters import TIParameters  # noqa: E402
from colloids.colloids_run import set_up_simulation, check_frame  # noqa: E402
from colloids.helper_functions import get_cell_from_box, read_gsd_file  # noqa: E402
from colloids.units import energy_unit, length_unit, temperature_unit, time_unit  # noqa: E402

_KB = (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).value_in_unit(energy_unit / temperature_unit)
_spring_constant_unit = energy_unit / (length_unit ** 2)
# Keep the time step at or below this fraction of the fastest Einstein-spring oscillation period so
# that the stiff restraint is integrated stably and accurately.
_MAX_DT_PER_SPRING_PERIOD = 0.15


def _recommended_mbar_value(result):
    """Return the guardrail-approved MBAR estimate, or NaN while refinement is still required."""
    if result["recommendation"] == "full":
        return float(result["dA2_full"])
    if result["recommendation"] == "hybrid":
        return float(result["dA2_hybrid"])
    return float("nan")


def _require_equal_mobile_masses(masses):
    """Validate the equal-mass assumption of the current fixed-CM Einstein reference."""
    values = np.asarray(masses, dtype=float)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("A TI calculation requires at least one mobile particle with finite positive mass.")
    if not np.allclose(values, values[0], rtol=1.0e-12, atol=0.0):
        raise ValueError(
            "The current fixed-center-of-mass Einstein reference assumes equal mobile-particle "
            "masses. Unequal masses require the mass-weighted constraint/reference correction of "
            "Khanna et al., J. Chem. Phys. 154, 164509 (2021).")
    return values


def _thermostat_temperature(parameters: RunParameters) -> unit.Quantity:
    """The thermostat (integrator) temperature, which sets kT for the free-energy analysis."""
    temperature = parameters.integrator_parameters.get("temperature")
    if temperature is None or not isinstance(temperature, unit.Quantity):
        raise ValueError("The integrator must define a temperature (a thermostatted integrator such "
                         "as LangevinMiddleIntegrator) for a thermodynamic-integration run.")
    return temperature


def _minimum_image_nearest_neighbor(positions: np.ndarray, cell: np.ndarray) -> float:
    """Smallest minimum-image pair distance (nm) for positions in a row-vector cell."""
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    minimum = np.inf
    for shift in shifts:
        offset = shift @ cell
        deltas = positions[:, None, :] - (positions[None, :, :] + offset)
        distances = np.sqrt((deltas ** 2).sum(-1))
        if np.all(shift == 0):
            np.fill_diagonal(distances, np.inf)
        minimum = min(minimum, float(distances.min()))
    return minimum


def _minimum_image_displacement(positions, reference, cell, reference_cell=None):
    """Minimum-image displacement (nm) of `positions` from `reference`, valid for TILTED cells.

    Uses fractional coordinates through the full (row-vector) cell matrix, so a particle whose stored
    position jumped by a whole box vector (including tilt components) on crossing a periodic boundary
    is wrapped correctly -- a component-wise wrap is wrong for non-orthorhombic cells (e.g. hexagonal
    Li3Fe). If `reference_cell` differs (box breathing under NPT), each set is reduced in its own cell
    so the affine box change cancels, and the displacement is expressed in the current `cell`."""
    if reference_cell is None:
        reference_cell = cell
    frac = positions @ np.linalg.inv(cell)
    frac0 = reference @ np.linalg.inv(reference_cell)
    dfrac = frac - frac0
    dfrac -= np.round(dfrac)
    return dfrac @ cell


def _stable_time_step(coupling, spring_constant_value, min_mass, base_step_size):
    """Cap the time step (ps) so the Einstein spring at this coupling is integrated stably.

    For U = coupling * Lambda_E * sum d^2, each Cartesian degree of freedom oscillates with angular
    frequency omega = sqrt(2 * coupling * Lambda_E / m). In the PACSim molar unit system,
    kJ/mol/nm^2/amu equals 1/ps^2, so omega is in 1/ps directly.
    """
    omega = math.sqrt(2.0 * coupling * spring_constant_value / min_mass)
    period = 2.0 * math.pi / omega
    return min(base_step_size, _MAX_DT_PER_SPRING_PERIOD * period)


def _prepare_window(base_parameters, initial_configuration_abs, window_dir, coupling, spring_constant,
                    equil_steps, prod_steps, sample_interval, step_size, platform, seed):
    """Write a TI window's run.yaml + ti.yaml; return (run_yaml, ti_yaml, trajectory_filename)."""
    os.makedirs(window_dir, exist_ok=True)
    integrator_parameters = dict(base_parameters.integrator_parameters)
    integrator_parameters["stepSize"] = step_size * time_unit
    window_parameters = dataclasses.replace(
        base_parameters,
        initial_configuration=initial_configuration_abs,
        platform_name=platform,
        integrator_parameters=integrator_parameters,
        equilibration_steps=equil_steps,
        run_steps=prod_steps,
        state_data_interval=sample_interval,
        trajectory_interval=sample_interval,
        checkpoint_interval=prod_steps,
        velocity_seed=seed,
        output_prefix=None,
        trajectory_filename=os.path.join(window_dir, "trajectory.gsd"),
        state_data_filename=os.path.join(window_dir, "state.csv"),
        checkpoint_filename=os.path.join(window_dir, "checkpoint.chk"),
        final_configuration_gsd_filename=None,
    )
    run_yaml = os.path.join(window_dir, "run.yaml")
    ti_yaml = os.path.join(window_dir, "ti.yaml")
    window_parameters.to_yaml(run_yaml)
    TIParameters(spring_constant=spring_constant, coupling=float(coupling),
                 fix_center_of_mass=True).to_yaml(ti_yaml)
    return run_yaml, ti_yaml, window_parameters.trajectory_filename


def _run_window_subprocess(run_yaml, ti_yaml, log_path):
    """Run one `pacsim-run` window as a subprocess, capturing its output to log_path."""
    with open(log_path, "w") as log_file:
        subprocess.run(["pacsim-run", run_yaml, "-t", ti_yaml],
                       stdout=log_file, stderr=subprocess.STDOUT, check=True)


def _collect_window(trajectory_filename, coupling, discard_first_frame):
    """Read a finished window's trajectory; return (bare_energy, coupled_mean, coupled_std) in kJ/mol."""
    with gsd.hoomd.open(trajectory_filename, "r") as trajectory:
        restraint_energies = [float(np.asarray(frame.log["harmonic_restraint_energy"])[0])
                              for frame in trajectory]
    if discard_first_frame and len(restraint_energies) > 1:
        restraint_energies = restraint_energies[1:]
    coupled_mean = float(np.mean(restraint_energies))
    # Logged energy is the coupled energy coupling * Lambda_E * sum d^2; the bare Einstein energy is
    # that divided by the coupling.
    return coupled_mean / float(coupling), coupled_mean, float(np.std(restraint_energies))


def _compute_delta_a1(base_parameters, run_yaml_dir, temperature, spring_constant_value,
                      n_samples, energy_platform, seed):
    """dA1/(N kT) by Gaussian sampling of the ideal Einstein crystal, scored with a PACS-only context."""
    eval_parameters = dataclasses.replace(base_parameters, platform_name=energy_platform)
    frame = read_gsd_file(eval_parameters.initial_configuration, eval_parameters.frame_index)
    check_frame(eval_parameters, frame)
    simulation = set_up_simulation(eval_parameters, frame)  # PACS only (no restraint)

    reference = np.array(frame.particles.position, dtype=float)
    mobile = np.array(frame.particles.mass, dtype=float) > 0.0
    _require_equal_mobile_masses(np.asarray(frame.particles.mass, dtype=float)[mobile])
    n_mobile = int(mobile.sum())
    kt = _KB * temperature.value_in_unit(temperature_unit)

    def pacs_energy(positions):
        simulation.context.setPositions(positions * length_unit)
        return simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(energy_unit)

    u_lattice = pacs_energy(reference)
    # For U = Lambda_E * sum d^2, each Cartesian degree of freedom has variance kT / (2 Lambda_E).
    sigma = math.sqrt(kt / (2.0 * spring_constant_value))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples)
    for s in range(n_samples):
        displacement = rng.normal(0.0, sigma, (n_mobile, 3))
        displacement -= displacement.mean(axis=0)  # project out centre-of-mass translation (fixed CM)
        positions = reference.copy()
        positions[mobile] += displacement
        samples[s] = pacs_energy(positions)
    delta_a1 = delta_a1_per_particle(samples * energy_unit, u_lattice * energy_unit, n_mobile, temperature)
    return delta_a1, u_lattice, n_mobile, samples


def _reduce_box_tilts(box):
    """Return a lattice-equivalent HOOMD box [Lx,Ly,Lz,xy,xz,yz] in OpenMM strict reduced form.

    Tilted cells built at the reduced-form boundary (e.g. hexagonal Li3Fe with xy = -0.5773: the
    b_x component is exactly -a_x/2 up to float32 rounding of the per-condition lattice scale) can
    land infinitesimally outside OpenMM's requirement |b_x| <= a_x/2, making setPeriodicBoxVectors
    raise 'must be in reduced form' unpredictably. Substituting the equivalent lattice vectors
    (b -= round(b_x/a_x) a, etc.) fixes this WITHOUT moving any particle position."""
    lx, ly, lz, xy, xz, yz = (float(v) for v in box[:6])
    bx = xy * ly                                     # row vectors: a=(lx,0,0), b=(bx,ly,0), c=(cx,cy,lz)
    cx, cy = xz * lz, yz * lz
    n = round(bx / lx); bx -= n * lx                 # b -> b - n a
    n = round(cy / ly); cy -= n * ly; cx -= n * bx   # c -> c - n b   (uses the reduced b)
    n = round(cx / lx); cx -= n * lx                 # c -> c - n a
    return np.array([lx, ly, lz, bx / ly, cx / lz, cy / lz], dtype=float)


def _write_config(template_src, dst, positions, box, frame_index):
    """Write a GSD with the given positions (nm) and box (tilts lattice-reduced for OpenMM), copying
    all other per-particle fields (radii/diameter, charge, mass, types) from template_src[frame_index]."""
    with gsd.hoomd.open(template_src, "r") as f:
        s = f[frame_index]
    frame = gsd.hoomd.Frame()
    frame.configuration.box = _reduce_box_tilts(np.asarray(box, dtype=float))
    frame.particles.N = int(s.particles.N)
    frame.particles.types = list(s.particles.types)
    frame.particles.typeid = np.array(s.particles.typeid)
    frame.particles.position = np.asarray(positions, dtype=float)
    frame.particles.mass = np.array(s.particles.mass)
    frame.particles.diameter = np.array(s.particles.diameter)
    frame.particles.charge = np.array(s.particles.charge)
    with gsd.hoomd.open(dst, "w") as out:
        out.append(frame)


def _pacs_energy(base_parameters, config_path, frame_index, platform="CPU", minimize=False):
    """PACS interaction energy (kJ/mol) of a configuration, optionally after local energy minimization
    (no restraint). Returns (energy_kJ_per_mol, positions_nm)."""
    parameters = dataclasses.replace(base_parameters, initial_configuration=config_path, platform_name=platform)
    frame = read_gsd_file(config_path, frame_index)
    check_frame(parameters, frame)
    simulation = set_up_simulation(parameters, frame)   # PACS only (no restraint)
    simulation.context.setPositions(np.array(frame.particles.position, dtype=float) * length_unit)
    if minimize:
        simulation.minimizeEnergy()
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit(energy_unit)
    positions = np.asarray(state.getPositions(asNumpy=True).value_in_unit(length_unit), dtype=float)
    return energy, positions


def _measure_rms_displacement(base_parameters, config_path, frame_index, temperature, platform, steps,
                              sample_interval=100, equil_fraction=0.3):
    """Mean-square displacement (nm^2, 3D per mobile particle) of the UNRESTRAINED crystal about its
    reference sites, from a short NVT run. Used to auto-tune the Einstein spring so the ideal-Einstein
    fluctuations match the real crystal's (the Frenkel-Ladd / Vega prescription) -- important because
    the NPT-relaxed, denser structures are stiffer than the Debye/brush length heuristic assumes."""
    parameters = dataclasses.replace(base_parameters, initial_configuration=config_path, platform_name=platform)
    frame = read_gsd_file(config_path, frame_index)
    check_frame(parameters, frame)
    simulation = set_up_simulation(parameters, frame)   # PACS only (no restraint)
    reference = np.array(frame.particles.position, dtype=float)
    mobile = np.array(frame.particles.mass, dtype=float) > 0.0
    cell = get_cell_from_box(frame.configuration.box)      # full matrix: valid for tilted cells too
    simulation.context.setPositions(reference * length_unit)
    simulation.context.setVelocitiesToTemperature(temperature)
    equil = int(steps * equil_fraction)
    if equil:
        simulation.step(equil)
    n_samples = max(1, (steps - equil) // sample_interval)
    msd = []
    for _ in range(n_samples):
        simulation.step(sample_interval)
        pos = np.asarray(simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
                         .value_in_unit(length_unit), dtype=float)
        disp = _minimum_image_displacement(pos[mobile], reference[mobile], cell)
        disp -= disp.mean(axis=0)                            # remove COM drift (fixed-CM reference)
        msd.append(float((disp ** 2).sum(axis=1).mean()))
    return float(np.mean(msd))


def _npt_relax_configuration(base_parameters, initial_configuration_abs, output_dir, args):
    """Relax the crystal under an ANISOTROPIC Monte-Carlo barostat at args.npt_pressure (each box axis
    scales independently, so no crystal symmetry is pre-assumed; cell TILT angles still stay fixed),
    then return a Frenkel-Ladd reference built from the RELAXED, ENERGY-MINIMIZED structure at the
    (per-axis) mean relaxed box.

    Rationale: the CIF/LatticeBuilder lattice scaled to a new box is not in general the mechanical
    (free-energy) minimum -- under a steep PACS surface-gap potential the particles relax off those
    sites, sometimes far (e.g. Th3P4). So the unrestrained NPT run finds the true relaxed structure;
    we take its final configuration at the mean box, energy-minimize it there, and use those minimized
    positions as the FL reference sites (this determines the reference self-consistently rather than
    trusting the CIF lattice constant). Returns (config_path, scale, box0_nm, box1_nm, u0_kJ, u1_kJ)
    where u0 is the as-built ideal-lattice energy and u1 the minimized reference energy (u1 <= u0 when
    healthy)."""
    npt_dir = os.path.join(output_dir, "npt")
    os.makedirs(npt_dir, exist_ok=True)
    # Guard against builder output whose tilted box sits marginally outside OpenMM's strict reduced
    # form (float32 rounding at the |b_x| = a_x/2 boundary, e.g. hexagonal Li3Fe): substitute the
    # lattice-equivalent reduced cell (positions untouched) before any simulation reads it.
    frame_in = read_gsd_file(initial_configuration_abs, base_parameters.frame_index)
    box_in = np.asarray(frame_in.configuration.box, dtype=float)
    if not np.allclose(_reduce_box_tilts(box_in), box_in, rtol=0, atol=1e-12):
        reduced_config = os.path.abspath(os.path.join(npt_dir, "initial_reduced.gsd"))
        _write_config(initial_configuration_abs, reduced_config,
                      np.array(frame_in.particles.position, dtype=float), box_in,
                      base_parameters.frame_index)
        initial_configuration_abs = reduced_config
    trajectory = os.path.join(npt_dir, "npt.gsd")
    npt_parameters = dataclasses.replace(
        base_parameters, initial_configuration=initial_configuration_abs, platform_name=args.npt_platform,
        npt_pressure=[args.npt_pressure * unit.bar] * 3, npt_frequency=args.npt_frequency,
        equilibration_steps=args.npt_equil_steps, run_steps=args.npt_steps,
        trajectory_interval=max(1, args.npt_steps // 100), state_data_interval=max(1, args.npt_steps // 100),
        checkpoint_interval=args.npt_steps, minimize_energy_initially=False, velocity_seed=args.seed,
        output_prefix=None, trajectory_filename=trajectory,
        state_data_filename=os.path.join(npt_dir, "npt.state.csv"),
        checkpoint_filename=os.path.join(npt_dir, "npt.chk"), final_configuration_gsd_filename=None,
    )
    run_yaml = os.path.join(npt_dir, "npt_run.yaml")
    npt_parameters.to_yaml(run_yaml)
    with open(os.path.join(npt_dir, "npt.log"), "w") as log_file:
        subprocess.run(["pacsim-run", run_yaml], stdout=log_file, stderr=subprocess.STDOUT, check=True)

    with gsd.hoomd.open(trajectory, "r") as frames:
        boxes = np.array([list(fr.configuration.box) for fr in frames], dtype=float)
        last = frames[-1]
        last_positions = np.array(last.particles.position, dtype=float)
        last_box = np.array(last.configuration.box, dtype=float)
    if boxes.size == 0:
        raise RuntimeError("NPT relaxation produced no trajectory frames to average the box over.")
    tail = boxes[len(boxes) // 2:] if len(boxes) > 1 else boxes    # production tail (post-equilibration)
    box0 = np.array(read_gsd_file(initial_configuration_abs, base_parameters.frame_index).configuration.box,
                    dtype=float)
    mean_box = tail[:, :3].mean(axis=0)
    scales = mean_box / box0[:3]                                  # per-axis scales (anisotropic barostat)
    scale = float(np.prod(scales) ** (1.0 / 3.0))                 # volume-equivalent isotropic scale
    # Adjust the equilibrated NPT snapshot to exactly the per-axis mean box, write it, then minimize.
    ref_box = np.array(last_box); ref_box[:3] = mean_box
    snapshot = os.path.join(npt_dir, "npt_snapshot.gsd")
    _write_config(trajectory, snapshot, last_positions * (mean_box / last_box[:3]), ref_box, -1)
    relaxed = os.path.abspath(os.path.join(npt_dir, "npt_relaxed.gsd"))
    u1, min_positions = _pacs_energy(base_parameters, snapshot, 0, platform=args.npt_platform, minimize=True)
    _write_config(snapshot, relaxed, min_positions, ref_box, 0)
    u0, _ = _pacs_energy(base_parameters, initial_configuration_abs, base_parameters.frame_index)

    # Structural check (the verdict must not be energy-only): displacement of the final NPT frame from
    # the as-built lattice sites, compared in fractional coordinates so pure box breathing cancels,
    # reported in units of the nearest-neighbor distance (a Lindemann-like number; ~<0.1 = ordered
    # crystal, >~0.15 = melted/rearranged).
    frame0 = read_gsd_file(initial_configuration_abs, base_parameters.frame_index)
    pos0 = np.array(frame0.particles.position, dtype=float)
    mobile0 = np.array(frame0.particles.mass, dtype=float) > 0.0
    frac0 = pos0 / box0[:3]
    frac1 = last_positions / last_box[:3]
    dfrac = frac1 - frac0
    dfrac -= np.round(dfrac)                                       # minimum image (orthorhombic)
    disp_nm = np.sqrt(((dfrac * mean_box) ** 2).sum(axis=1))[mobile0]
    cell0 = get_cell_from_box(box0)
    d_nn0 = _minimum_image_nearest_neighbor(pos0[mobile0], cell0)
    lindemann = float(disp_nm.mean() / d_nn0)
    n_particles = int(mobile0.sum())
    return {"config": relaxed, "scale": scale, "scales": [float(x) for x in scales],
            "box0_nm": float(box0[0]), "box1_nm": float(mean_box[0]),
            "box0_nm_xyz": [float(x) for x in box0[:3]], "box1_nm_xyz": [float(x) for x in mean_box],
            "u_asbuilt": float(u0), "u_relaxed": float(u1), "lindemann": lindemann,
            "n_particles": n_particles}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_yaml", help="PACS run YAML (physics + initial equilibrated crystal)")
    parser.add_argument("--output-dir", default="ti_run", help="directory for per-window outputs and results")
    parser.add_argument("--windows", type=int, default=12,
                        help="minimum number of Gauss-Legendre TI windows (auto-scaled up with the "
                             "spring stiffness unless --no-auto-windows)")
    parser.add_argument("--no-auto-windows", action="store_true",
                        help="use exactly --windows nodes; do NOT auto-scale with the w-range "
                             "(risks a quadrature bias for stiff auto-tuned springs)")
    parser.add_argument("--equil-steps", type=int, default=20000, help="equilibration steps per window")
    parser.add_argument("--prod-steps", type=int, default=60000, help="production steps per window")
    parser.add_argument("--sample-interval", type=int, default=200, help="reporting interval (steps)")
    parser.add_argument("--einstein-displacement-fraction", type=float, default=0.05,
                        help="target rms Einstein displacement, as a fraction of the crystal's measured "
                             "real rms displacement when auto-tuning (default), or of the interaction "
                             "decay length (min Debye/brush) under --no-autotune-spring; sets Lambda_E. "
                             "Lower it if the dA1 log correction is not small.")
    parser.add_argument("--spring-constant", type=float, default=None,
                        help="Lambda_E in kJ/mol/nm^2; overrides --einstein-displacement-fraction")
    parser.add_argument("--delta-a1-samples", type=int, default=4000,
                        help="number of ideal-Einstein Gaussian samples for dA1")
    parser.add_argument("--debroglie", type=float, default=1.0,
                        help="thermal de Broglie wavelength (nm) convention for A0 (only shifts the absolute value)")
    parser.add_argument("--window-platform", default="CPU",
                        help="OpenMM platform for the TI windows (CPU default for reproducibility; "
                             "the periodic restraint is also validated on OpenCL and CUDA)")
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="number of TI windows to run concurrently as separate pacsim-run "
                             "subprocesses. Small colloid systems barely use the GPU, so many windows "
                             "share one A100 efficiently (see gpu_bench); set to --windows to run all "
                             "windows of a crystal at once on a CUDA node.")
    parser.add_argument("--energy-platform", default="OpenCL",
                        help="OpenMM platform for the PACS-only dA1 energy evaluations")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--keep-first-frame", action="store_true",
                        help="include the first (post-equilibration) frame of each window in the average")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip writing the TI diagnostics figure (diagnostics.png)")
    parser.add_argument("--mbar", action="store_true",
                        help="also combine the windows with MBAR and report it alongside the TI result")
    parser.add_argument("--refine-mbar-tol", type=float, default=None,
                        help="adaptively insert extra windows at the lowest-overlap gaps until MBAR and "
                             "TI agree on A_sol/N to within this tolerance (kcal/mol per particle, e.g. 0.1); "
                             "implies --mbar")
    parser.add_argument("--max-refine", type=int, default=10,
                        help="maximum number of extra windows inserted during MBAR refinement")
    parser.add_argument("--mbar-overlap-floor", type=float, default=0.01,
                        help="minimum adjacent MBAR overlap for an interior bridge to count as usable; "
                             "refinement inserts windows until every interior bridge clears this (default 0.01)")
    parser.add_argument("--no-npt", action="store_true",
                        help="skip the default NPT pre-relaxation and run TI on the as-built (CIF-scaled) "
                             "lattice. By default the box is relaxed under an isotropic barostat at "
                             "--npt-pressure, the equilibrated structure is energy-minimized at the mean "
                             "box, and the TI (still NVT) references that relaxed+minimized structure")
    parser.add_argument("--no-autotune-spring", action="store_true",
                        help="use the Debye/brush length heuristic for the Einstein spring instead of "
                             "auto-tuning it to the crystal's measured rms displacement")
    parser.add_argument("--autotune-steps", type=int, default=6000,
                        help="unrestrained NVT steps used to measure the rms displacement for spring auto-tuning")
    parser.add_argument("--npt-pressure", type=float, default=0.0,
                        help="barostat pressure in bar for the NPT pre-relaxation (default 0 = zero-pressure density)")
    parser.add_argument("--npt-equil-steps", type=int, default=20000,
                        help="barostat equilibration steps for the NPT pre-relaxation")
    parser.add_argument("--npt-steps", type=int, default=20000,
                        help="barostat production steps averaged for the relaxed box")
    parser.add_argument("--npt-frequency", type=int, default=25,
                        help="barostat volume-move attempt interval in steps")
    parser.add_argument("--npt-platform", default=None,
                        help="OpenMM platform for the NPT relaxation run (defaults to --window-platform)")
    args = parser.parse_args()
    if args.npt_platform is None:
        args.npt_platform = args.window_platform

    run_yaml_dir = os.path.dirname(os.path.abspath(args.run_yaml))
    os.makedirs(args.output_dir, exist_ok=True)
    base_parameters = RunParameters.from_yaml(args.run_yaml)
    initial_configuration_abs = os.path.abspath(os.path.join(run_yaml_dir, base_parameters.initial_configuration))

    npt_info = None
    if not args.no_npt:
        relax = _npt_relax_configuration(base_parameters, initial_configuration_abs, args.output_dir, args)
        initial_configuration_abs = relax["config"]
        # Point ALL downstream steps at the relaxed lattice: A0/reference read initial_configuration_abs,
        # the windows are passed it explicitly, and dA1 reads base_parameters.initial_configuration -- so
        # base_parameters must be updated too or dA1 (U_lattice) would stay at the original density.
        base_parameters = dataclasses.replace(base_parameters, initial_configuration=initial_configuration_abs)
        # Validation gate (structural + physically scaled energy, NOT a bare percent-of-|U| test):
        #   * energy: a P=0 relax toward equilibrium should not raise the reference energy by a
        #     thermally significant amount PER PARTICLE (a percent-of-|U| criterion false-positives for
        #     weakly bound crystals, e.g. CsCl at +10/-60 where +379 kJ/mol was only 0.35 kT/particle);
        #   * structure: the final NPT frame must remain near the lattice sites (Lindemann-like
        #     displacement/d_nn below the ~0.15 melting rule of thumb).
        npt_kt = _KB * _thermostat_temperature(base_parameters).value_in_unit(temperature_unit)
        rise_kt = (relax["u_relaxed"] - relax["u_asbuilt"]) / (relax["n_particles"] * npt_kt)
        energy_bad = rise_kt > 0.5
        structure_bad = relax["lindemann"] > 0.15
        npt_scale = relax["scale"]
        npt_info = {"pressure_bar": args.npt_pressure, "box_before_nm": relax["box0_nm"],
                    "box_after_nm": relax["box1_nm"], "isotropic_scale": npt_scale,
                    "volume_ratio": npt_scale ** 3,
                    "U_asbuilt_kJ_per_mol": relax["u_asbuilt"],
                    "U_relaxed_minimized_kJ_per_mol": relax["u_relaxed"],
                    "relaxation_energy_kT_per_particle": rise_kt,
                    "lindemann_displacement_over_dnn": relax["lindemann"],
                    "relaxation_warning": bool(energy_bad or structure_bad)}
        print(f"  NPT pre-relax (P={args.npt_pressure} bar): box {relax['box0_nm']:.3f} -> "
              f"{relax['box1_nm']:.3f} nm (scale {npt_scale:.5f}, V x{npt_scale ** 3:.4f}); "
              f"TI reference = relaxed+minimized structure "
              f"(dU {rise_kt:+.3f} kT/particle, Lindemann {relax['lindemann']:.3f})")
        if structure_bad:
            print(f"  *** WARNING: the crystal LEFT its lattice sites during the NPT relaxation "
                  f"(displacement/d_nn = {relax['lindemann']:.3f} > 0.15). It melted or rearranged at "
                  f"these conditions; do not report this tethered-basin TI as an equilibrium crystal "
                  f"free energy. ***")
        elif energy_bad:
            print(f"  *** WARNING: the relaxed+minimized reference is {rise_kt:.2f} kT/particle ABOVE "
                  f"the as-built lattice despite remaining ordered. The NPT/minimization likely did not "
                  f"converge; inspect before trusting this free energy. ***")

    temperature = _thermostat_temperature(base_parameters)
    kt = _KB * temperature.value_in_unit(temperature_unit)
    frame = read_gsd_file(initial_configuration_abs, base_parameters.frame_index)
    reference = np.array(frame.particles.position, dtype=float)
    mobile = np.array(frame.particles.mass, dtype=float) > 0.0
    mobile_masses = _require_equal_mobile_masses(np.asarray(frame.particles.mass, dtype=float)[mobile])
    n_mobile = int(mobile.sum())
    cell = get_cell_from_box(frame.configuration.box)
    volume = abs(float(np.linalg.det(cell)))
    d_nn = _minimum_image_nearest_neighbor(reference[mobile], cell)

    autotuned_msd = None
    if args.spring_constant is not None:
        spring_constant_value = args.spring_constant
        characteristic_length = None
    elif not args.no_autotune_spring:
        # Auto-tune to the crystal's ACTUAL fluctuation scale: measure the real (unrestrained) rms
        # displacement and make the ideal-Einstein rms a small fraction of it. The Einstein must stay
        # NARROWER than the real crystal so its samples remain in the near-harmonic core of the (very
        # steep) PACS potential -- matching the full real rms puts them in the anharmonic tails and
        # destroys the dA1 reweighting. This adapts to each structure's stiffness (unlike the fixed
        # Debye/brush heuristic), which matters for the NPT-relaxed, denser lattices.
        real_msd = _measure_rms_displacement(base_parameters, initial_configuration_abs,
                                             base_parameters.frame_index, temperature,
                                             args.energy_platform, args.autotune_steps)
        target_displacement = args.einstein_displacement_fraction * math.sqrt(real_msd)
        spring_constant_value = 3.0 * kt / (2.0 * target_displacement ** 2)
        autotuned_msd = real_msd
        characteristic_length = None
        print(f"  spring auto-tuned: real rms displacement {math.sqrt(real_msd):.4f} nm; Einstein rms "
              f"{target_displacement:.4f} nm ({args.einstein_displacement_fraction:g} x real)")
    else:
        # Fallback heuristic: the dA1 reweighting is well conditioned only if the ideal Einstein crystal
        # stays inside the region where U_PACS is nearly harmonic, i.e. the rms displacement must be
        # small compared to the interaction decay length (the smaller of the Debye and brush lengths).
        characteristic_length = min(base_parameters.debye_length.value_in_unit(length_unit),
                                    base_parameters.brush_length.value_in_unit(length_unit))
        target_displacement = args.einstein_displacement_fraction * characteristic_length
        spring_constant_value = 3.0 * kt / (2.0 * target_displacement ** 2)
    spring_constant = spring_constant_value * _spring_constant_unit

    # Auto-scale the quadrature with the spring stiffness. The dA2 integrand lives on
    # w = ln(s*kappa + c), a range of ln(1 + kappa/c) that GROWS with Lambda_E; a fixed node count
    # that resolves a soft spring under-resolves a stiff one. A spring-independence test (CsCl N=128,
    # Lambda 100->2651) showed a systematic +0.5 NkT dA2 bias at 16 nodes that 32 nodes removed; 22 nodes
    # (10/unit) was still ~0.1 NkT low, while 32 (~15/unit) converged -- so 15 nodes per unit of
    # w-range (min: the user setting) keeps the quadrature error at or below the ~0.05 NkT noise.
    kappa = spring_constant_value / kt          # beta * Lambda_E * (1 nm)^2, dimensionless
    w_range = math.log1p(kappa / math.exp(3.5))
    n_windows = int(args.windows)
    if not args.no_auto_windows:
        n_windows = max(n_windows, int(math.ceil(15.0 * w_range)))

    print(f"PACS Frenkel-Ladd TI: {args.run_yaml}")
    print(f"  N (mobile) = {n_mobile};  V = {volume:.4e} nm^3;  T = {temperature.value_in_unit(temperature_unit):.2f} K;"
          f"  d_nn = {d_nn:.3f} nm")
    print(f"  Lambda_E = {spring_constant_value:.4e} kJ/mol/nm^2;  windows = {n_windows}"
          f"{f' (auto-scaled from {args.windows} for w-range {w_range:.2f})' if n_windows != args.windows else ''};  "
          f"window platform = {args.window_platform}")

    # --- A0 (analytic) ---
    a0 = ideal_einstein_free_energy_per_particle(
        n_mobile, volume * length_unit ** 3, temperature, spring_constant,
        debroglie_wavelength=args.debroglie * length_unit)

    # --- dA2 (Gauss-Legendre TI over the springs; one pacsim-run per window) ---
    schedule = FrenkelLaddSchedule(n_points=n_windows, temperature=temperature,
                                   spring_constant=spring_constant)
    couplings = schedule.couplings()
    base_step_size = base_parameters.integrator_parameters["stepSize"].value_in_unit(time_unit)
    min_mass = float(mobile_masses.min())
    # Prepare every window (write run.yaml + ti.yaml), then run them as pacsim-run subprocesses up to
    # args.max_parallel at a time, then collect the mean spring energies in coupling order. Small
    # colloid systems barely use the GPU, so running many windows concurrently on one A100 is a large
    # speed-up with negligible per-window slowdown (see gpu_bench).
    window_specs = []
    for w, coupling in enumerate(couplings):
        window_dir = os.path.join(args.output_dir, "windows", f"window_{w:02d}")
        # Cap the time step for stiff springs, and scale the step counts so the total simulated time
        # (and the number of recorded frames) is preserved regardless of the reduced step size.
        step_size = _stable_time_step(float(coupling), spring_constant_value, min_mass, base_step_size)
        scale = base_step_size / step_size
        equil_steps = int(math.ceil(args.equil_steps * scale))
        prod_steps = int(math.ceil(args.prod_steps * scale))
        sample_interval = max(1, int(round(args.sample_interval * scale)))
        run_yaml, ti_yaml, trajectory_filename = _prepare_window(
            base_parameters, initial_configuration_abs, window_dir, coupling, spring_constant,
            equil_steps, prod_steps, sample_interval, step_size, args.window_platform, args.seed + w + 1)
        window_specs.append({"window": w, "coupling": float(coupling), "run_yaml": run_yaml,
                             "ti_yaml": ti_yaml, "trajectory": trajectory_filename,
                             "log": os.path.join(window_dir, "pacsim.log"),
                             "step_size": step_size, "prod_steps": prod_steps})

    max_parallel = max(1, args.max_parallel)
    print(f"  dA2: running {len(window_specs)} windows on {args.window_platform}, "
          f"up to {max_parallel} concurrently ...")
    if max_parallel == 1:
        for spec in window_specs:
            _run_window_subprocess(spec["run_yaml"], spec["ti_yaml"], spec["log"])
            print(f"    window {spec['window']:2d} done", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_spec = {executor.submit(_run_window_subprocess, spec["run_yaml"], spec["ti_yaml"],
                                              spec["log"]): spec for spec in window_specs}
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                future.result()  # re-raise any subprocess failure
                print(f"    window {spec['window']:2d} done", flush=True)

    bare_energies, window_rows = [], []
    for spec in window_specs:
        w, coupling = spec["window"], spec["coupling"]
        bare, coupled_mean, coupled_std = _collect_window(
            spec["trajectory"], coupling, not args.keep_first_frame)
        bare_energies.append(bare * energy_unit)
        window_rows.append({"window": w, "lambda_ein": coupling,
                            "bare_spring_energy_kJmol": bare, "coupled_mean_kJmol": coupled_mean,
                            "coupled_std_kJmol": coupled_std, "step_size_ps": spec["step_size"],
                            "prod_steps": spec["prod_steps"]})
        print(f"  window {w:2d}: lambda_ein = {coupling:.4e}  dt = {spec['step_size']:.4f} ps  "
              f"<U_spring(bare)>/NkT = {bare / (n_mobile * kt):8.3f}")
    delta_a2 = schedule.integrate_delta_a2_per_particle(bare_energies, n_mobile)

    # --- dA1 (ideal Einstein reweighting via direct Gaussian sampling) ---
    delta_a1, u_lattice, _, delta_a1_samples = _compute_delta_a1(
        base_parameters, run_yaml_dir, temperature, spring_constant_value,
        args.delta_a1_samples, args.energy_platform, args.seed + 999)
    log_correction = delta_a1 - u_lattice / (n_mobile * kt)
    print(f"  U_lattice/NkT = {u_lattice / (n_mobile * kt):.4f};  dA1 log correction/NkT = {log_correction:+.4f} "
          f"(should be small, ~0.02)")
    if abs(log_correction) > 0.1:
        print(f"  WARNING: the dA1 log correction ({log_correction:+.3f} NkT) is not small, so A_sol may be "
              f"poorly converged. Use a stiffer Lambda_E (a smaller --einstein-displacement-fraction).")

    result = FreeEnergyResult(a0=a0, delta_a1=delta_a1, delta_a2=delta_a2, n_particles=n_mobile)
    print("\n" + str(result))

    summary = {
        "run_yaml": os.path.abspath(args.run_yaml), "n_particles": n_mobile, "volume_nm3": volume,
        "temperature_K": temperature.value_in_unit(temperature_unit), "d_nn_nm": d_nn,
        "lambda_E_kJ_per_mol_nm2": spring_constant_value, "windows": int(n_windows),
        "windows_requested": int(args.windows), "w_range": w_range,
        "equil_steps": args.equil_steps, "prod_steps": args.prod_steps,
        "A0_NkT": a0, "dA1_NkT": delta_a1, "dA2_NkT": delta_a2,
        "U_lattice_NkT": u_lattice / (n_mobile * kt), "dA1_log_correction_NkT": log_correction,
        "A_sol_NkT": result.a_sol, "A_sol_FL_NkT": result.a_sol_frenkel_ladd, "windows_data": window_rows,
        "npt_equilibration": npt_info, "autotuned_msd_nm2": autotuned_msd,
        "spring_rule": {"rule": ("explicit" if args.spring_constant is not None
                                 else "fraction_of_measured_rms" if not args.no_autotune_spring
                                 else "fraction_of_decay_length"),
                        "fraction": args.einstein_displacement_fraction,
                        "target_displacement_nm": math.sqrt(1.5 * kt / spring_constant_value),
                        "measured_rms_nm": (math.sqrt(autotuned_msd) if autotuned_msd else None)},
    }
    output_json = os.path.join(args.output_dir, "free_energy.json")
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {output_json}")

    # Save the dA1 samples so the diagnostics can be (re)plotted without rerunning.
    np.savez(os.path.join(args.output_dir, "delta_a1_samples.npz"),
             u_sol_samples=delta_a1_samples, u_lattice=u_lattice,
             temperature_K=temperature.value_in_unit(temperature_unit), n_mobile=n_mobile)

    if not args.no_plots:
        try:
            import ti_diagnostics
            png = ti_diagnostics.make_diagnostics(args.output_dir)
            print(f"Wrote {png}")
        except Exception as exc:  # plotting must never break a completed free-energy calculation
            print(f"  (diagnostics plot skipped: {exc})")

    # --- optional MBAR pipeline + adaptive window refinement ---
    if args.mbar or args.refine_mbar_tol is not None:
        _run_mbar_pipeline(args, window_specs, base_parameters, initial_configuration_abs,
                           spring_constant, spring_constant_value, min_mass, base_step_size,
                           a0, delta_a1, delta_a2, result.a_sol, n_mobile, temperature)


def _run_mbar_pipeline(args, window_specs, base_parameters, initial_configuration_abs, spring_constant,
                       spring_constant_value, min_mass, base_step_size, a0, delta_a1, delta_a2,
                       a_sol_ti, n_mobile, temperature):
    """MBAR re-combination of the windows, adaptively inserting extra windows at the lowest-overlap
    gaps until MBAR agrees with the TI A_sol to within --refine-mbar-tol (kcal/mol per particle)."""
    import mbar_analysis
    kb_kcal = 0.001987204259  # kcal/mol/K
    temperature_k = temperature.value_in_unit(temperature_unit)
    kt_kcal = kb_kcal * temperature_k
    keep_first = args.keep_first_frame

    s_vals = [spec["coupling"] for spec in window_specs]
    bare = [mbar_analysis.read_bare_energies(spec["trajectory"], spec["coupling"], not keep_first)
            for spec in window_specs]
    next_index = len(window_specs)

    def _run_extra_window(index, s_new):
        window_dir = os.path.join(args.output_dir, "windows", f"window_{index:02d}")
        step_size = _stable_time_step(s_new, spring_constant_value, min_mass, base_step_size)
        scale = base_step_size / step_size
        equil_steps = int(math.ceil(args.equil_steps * scale))
        prod_steps = int(math.ceil(args.prod_steps * scale))
        sample_interval = max(1, int(round(args.sample_interval * scale)))
        run_yaml, ti_yaml, trajectory = _prepare_window(
            base_parameters, initial_configuration_abs, window_dir, s_new, spring_constant,
            equil_steps, prod_steps, sample_interval, step_size, args.window_platform,
            args.seed + 1000 + index)
        _run_window_subprocess(run_yaml, ti_yaml, os.path.join(window_dir, "pacsim.log"))
        return trajectory

    tol_kcal = args.refine_mbar_tol
    max_refine = args.max_refine if tol_kcal is not None else 0
    trace = []
    print("\n  === MBAR pipeline ===  (dA2 by MBAR vs the TI quadrature; refinement tol "
          f"{'off' if tol_kcal is None else f'{tol_kcal} kcal/mol/particle'})")
    overlap_floor = args.mbar_overlap_floor
    r = None
    for iteration in range(max_refine + 1):
        r = mbar_analysis.analyze(bare, s_vals, n_mobile, temperature_k, overlap_floor)
        s_sorted = r["s_sorted"]
        # Do not promote an ill-conditioned full/hybrid value after the guardrail says "refine".
        # The diagnostic values remain available, but the reported estimate and gap stay NaN/null.
        best = _recommended_mbar_value(r)
        gap_nkt = abs((a0 + delta_a1 + best) - a_sol_ti) if math.isfinite(best) else float("nan")
        gap_kcal = gap_nkt * kt_kcal if math.isfinite(gap_nkt) else float("nan")
        trustworthy = r["recommendation"] in ("full", "hybrid")
        trace.append({"windows": len(s_vals), "dA2_full_NkT": r["dA2_full"],
                      "dA2_hybrid_NkT": r["dA2_hybrid"], "recommendation": r["recommendation"],
                      "min_interior_overlap": r["min_interior_overlap"],
                      "min_endpoint_efficiency": r["min_endpoint_efficiency"],
                      "diagnostics_ok": r["diagnostics_ok"],
                      "gap_NkT": gap_nkt, "gap_kcal_per_mol_particle": gap_kcal})
        gap_label = f"{gap_kcal:.3f} kcal/mol" if math.isfinite(gap_kcal) else "n/a (refine)"
        print(f"  windows={len(s_vals):3d}:  TI={delta_a2:+.3f}  full-MBAR={r['dA2_full']:+.3f}  "
              f"hybrid={r['dA2_hybrid']:+.3f}  min-interior-ovl={r['min_interior_overlap']:.3f}  "
              f"[{r['recommendation']}]  |best-TI|={gap_label}")
        if tol_kcal is None or (trustworthy and gap_kcal < tol_kcal):
            break
        if not r["diagnostics_ok"]:
            print(f"  MBAR quality diagnostics failed; stopping refinement without reporting an "
                  f"MBAR value ({r['diagnostic_error']}).")
            break
        if iteration == max_refine:
            print(f"  reached --max-refine={max_refine}; stopping "
                  f"(min interior overlap {r['min_interior_overlap']:.3f}, "
                  f"gap {gap_label}).")
            break
        # Insert one window at the geometric midpoint of the lowest-overlap INTERIOR pair of sampled
        # windows (r["interior_overlap"][j] bridges s_sorted[j] and s_sorted[j+1]); the s=0/s=1
        # endpoint bridges are excluded -- TI handles those segments in the hybrid estimator.
        interior = r["interior_overlap"]
        candidates = [(float(interior[j]), float(s_sorted[j]), float(s_sorted[j + 1]))
                      for j in range(len(interior))]
        worst_overlap, lo, hi = min(candidates, key=lambda c: c[0])
        s_new = math.sqrt(lo * hi)
        print(f"    -> insert window at s={s_new:.4e} (worst interior overlap {worst_overlap:.3f} in "
              f"[{lo:.4e}, {hi:.4e}])")
        trajectory = _run_extra_window(next_index, s_new)
        bare.append(mbar_analysis.read_bare_energies(trajectory, s_new, not keep_first))
        s_vals.append(s_new)
        next_index += 1

    best = _recommended_mbar_value(r)
    gap_nkt = abs((a0 + delta_a1 + best) - a_sol_ti) if math.isfinite(best) else float("nan")
    refined = {"a_sol_TI_NkT": a_sol_ti,
               "a_sol_MBAR_NkT": a0 + delta_a1 + best if math.isfinite(best) else float("nan"),
               "dA2_TI_NkT": delta_a2, "dA2_full_MBAR_NkT": r["dA2_full"],
               "dA2_hybrid_NkT": r["dA2_hybrid"], "dA2_reported_NkT": best,
               "recommendation": r["recommendation"], "min_interior_overlap": r["min_interior_overlap"],
               # Compatibility alias plus the accurately named endpoint-quality field.
               "min_endpoint_overlap": r["min_endpoint_overlap"],
               "min_endpoint_efficiency": r["min_endpoint_efficiency"], "overlap_floor": overlap_floor,
               "n_windows_final": len(s_vals), "n_windows_initial": len(window_specs),
               "gap_NkT": gap_nkt,
               "gap_kcal_per_mol_particle": gap_nkt * kt_kcal if math.isfinite(gap_nkt) else float("nan"),
               "tol_kcal_per_mol_particle": tol_kcal,
               "converged": bool(tol_kcal is not None and r["recommendation"] in ("full", "hybrid")
                                 and math.isfinite(gap_nkt) and gap_nkt * kt_kcal < tol_kcal),
               "s_values_final": sorted(float(s) for s in s_vals), "trace": trace}
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            value = float(obj)
            return value if math.isfinite(value) else None
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    out = os.path.join(args.output_dir, "mbar_refined.json")
    with open(out, "w") as f:
        json.dump(_json_safe(refined), f, indent=2)
    print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
