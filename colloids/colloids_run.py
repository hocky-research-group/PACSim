import argparse
import inspect
import os
import sys
from typing import Optional, Sequence
import warnings
import gsd.hoomd
import numpy as np
import openmm
from openmm import app
from colloids import (ColloidPotentialsAlgebraic, ColloidPotentialsParameters, ShiftedLennardJonesWalls,
                      ImplicitSubstrateWall, DepletionPotential, Gravity, HarmonicRestraint, PlumedPotential,
                      __version__)
from colloids.gsd_reporter import GSDReporter
from colloids.helper_functions import get_cell_from_box, read_gsd_file, write_gsd_file
import colloids.integrators as integrators
from colloids.run_parameters import RunParameters
from colloids.ti_parameters import TIParameters
from colloids.status_reporter import StatusReporter
import colloids.update_reporters as update_reporters
from colloids.units import electric_potential_unit, length_unit


def simple_formatwarning(msg: str, category: Warning, filename: str, lineno: int, line: Optional[str] = None) -> str:
    """
    Simpler format for warnings that excludes the line with the code that caused the warning.

    :param msg:
        The warning message.
    :type msg: str
    :param category:
        The warning category.
    :type category: Warning
    :param filename:
        The filename where the warning occurred.
    :type filename: str
    :param lineno:
        The line number where the warning occurred.
    :type lineno: int
    :param line:
        The line of code that caused the warning (not used).
    :type line: Optional[str]

    :return:
        The formatted warning message.
    :rtype: str
    """
    return f"{filename}:{lineno}: {category.__name__}: {msg}\n"


warnings.formatwarning = simple_formatwarning


class ExampleAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # TODO ADD OPTION FOR PLATFORM PROPERTIES?
        default_parameters = RunParameters()
        default_parameters.to_yaml("example.yaml")
        parser.exit()


def initialize_barostat(parameters: RunParameters, integrator: openmm.Integrator) -> Optional[openmm.Force]:
    """
    Instantiate the optional Monte Carlo barostat for an NPT run.

    A single ``npt_pressure`` quantity gives an isotropic MonteCarloBarostat; a list of three
    pressures gives an anisotropic MonteCarloAnisotropicBarostat. The barostat temperature is the
    thermostat temperature of the integrator (not the ``potential_temperature``, which sets the
    strength of the colloidal potentials and may differ from the thermostat temperature).

    :param parameters:
        The run parameters.
    :type parameters: RunParameters
    :param integrator:
        The integrator of the simulation. Its temperature is used for the barostat, so the
        integrator must be a thermostatted integrator (i.e., it must define a temperature).
    :type integrator: openmm.Integrator

    :return:
        The barostat force, or None if no barostat is requested.
    :rtype: Optional[openmm.Force]

    :raises ValueError:
        If a barostat is requested but the integrator does not define a temperature (e.g., a
        VerletIntegrator), because constant-pressure sampling requires a thermostat.
    """
    if parameters.npt_pressure is None:
        return None
    try:
        temperature = integrator.getTemperature()
    except AttributeError:
        raise ValueError("An NPT barostat requires a thermostatted integrator that defines a "
                         "temperature (e.g., LangevinMiddleIntegrator or NoseHooverIntegrator), but "
                         f"the chosen integrator '{parameters.integrator}' does not.")
    if isinstance(parameters.npt_pressure, list):
        pressure_x, pressure_y, pressure_z = parameters.npt_pressure
        scale = parameters.npt_scale if parameters.npt_scale is not None else [True, True, True]
        return integrators.MonteCarloAnisotropicBarostat(
            temperature, pressure_x, pressure_y, pressure_z,
            scale[0], scale[1], scale[2], parameters.npt_frequency)
    return integrators.MonteCarloBarostat(temperature, parameters.npt_pressure, parameters.npt_frequency)


def check_frame(parameters: RunParameters, frame: gsd.hoomd.Frame) -> None:
    """Check the frame and the run parameters."""
    for diameter in frame.particles.diameter:
        if not diameter > 0.0:
            raise ValueError("Every diameter must be greater than zero.")
    for mass in frame.particles.mass:
        if not mass >= 0.0:
            raise ValueError("Every mass must be greater than or equal to zero.")
    for constraint_value in frame.constraints.value:
        if not constraint_value > 0.0:
            raise ValueError("Every constraint distance must be greater than zero.")

    if any(parameters.wall_directions):
        # Check for orthogonal box vectors if walls should be included.
        if not all(a == 0.0 for a in frame.configuration.box[3:]):
            raise ValueError("If any wall is included, the box vectors must be parallel to the coordinate axes.")
        # If not all walls are present, the box of OpenMM needs to be enlarged because OpenMM will use periodic
        # boundaries, and we do not want to let particles interact through the walls. The enlargement of the box does
        # currently not consider the depletant radius, which is why it should be small enough.
        if not all(parameters.wall_directions):
            if parameters.use_depletion:
                if (parameters.depletant_radius
                        > (parameters.cutoff_factor * parameters.debye_length - 2.0 * parameters.brush_length) / 2.0):
                    raise ValueError("The depletant radius is too large for the cutoff factor and brush length when "
                                     "partial walls are included (r_d <= (cutoff_factor * lambda_D - 2 * L) / 2)")

    if parameters.use_depletion:
        assert (parameters.depletant_radius is not None and parameters.depletant_radius.value_in_unit(length_unit) > 0.0)
        for diameter in frame.particles.diameter:
            if parameters.depletant_radius.value_in_unit(length_unit) / (diameter / 2.0) > 0.1547:
                warnings.warn("Size ratio of depletant to colloid particles is too large. "
                              "Analytical computation of depletion potential may be invalid."
                              "See Dijkstra et. al., Journal of Physics: Condensed Matter, 1999, Volume 11, "
                              "pp 10079 - 10106.")

    # Explicit substrate is detected by immobile particles with mass 0.0.
    use_explicit_substrate = any(mass == 0.0 for mass in frame.particles.mass)
    if use_explicit_substrate and parameters.use_implicit_substrate:
        raise ValueError("Cannot use both explicit and implicit substrate.")
    if use_explicit_substrate or parameters.use_implicit_substrate:
        if not parameters.wall_directions[-1]:
            raise ValueError("A substrate can only be used if z walls are active.")


def set_up_harmonic_restraint(ti_parameters: TIParameters, frame: gsd.hoomd.Frame,
                              reference_frame: gsd.hoomd.Frame) -> HarmonicRestraint:
    """
    Build the Einstein-crystal harmonic restraint for a thermodynamic-integration run.

    Every mobile particle (mass greater than zero) is restrained to its position in the reference
    frame, unless ``ti_parameters.restrain_types`` limits the restraint to particular types.

    :param ti_parameters:
        The thermodynamic-integration parameters.
    :type ti_parameters: TIParameters
    :param frame:
        The frame that is used as the initial configuration of the run (used for the particle types
        and masses).
    :type frame: gsd.hoomd.Frame
    :param reference_frame:
        The frame that supplies the reference (lattice) positions r0.
    :type reference_frame: gsd.hoomd.Frame

    :return:
        The harmonic restraint with all restrained particles added.
    :rtype: HarmonicRestraint

    :raises ValueError:
        If the reference frame does not have the same number of particles as the run frame.
        If a type in restrain_types is not present in the frame.
    """
    if reference_frame.particles.N != frame.particles.N:
        raise ValueError("The reference configuration must have the same number of particles as the "
                         "initial configuration.")
    if ti_parameters.restrain_types is not None:
        for restrain_type in ti_parameters.restrain_types:
            if restrain_type not in frame.particles.types:
                raise ValueError(f"Type {restrain_type} of restrain_types is not in the frame.")

    restraint = HarmonicRestraint(spring_constant=ti_parameters.spring_constant,
                                  coupling=ti_parameters.coupling)
    reference_positions = reference_frame.particles.position
    restrained_any = False
    for i in range(frame.particles.N):
        # Never restrain immobile substrate particles (mass zero).
        if not frame.particles.mass[i] > 0.0:
            continue
        if ti_parameters.restrain_types is not None:
            if frame.particles.types[frame.particles.typeid[i]] not in ti_parameters.restrain_types:
                continue
        restraint.add_particle(i, reference_positions[i] * length_unit)
        restrained_any = True
    if not restrained_any:
        raise ValueError("No particles were restrained. Check the masses and restrain_types.")
    return restraint


def set_up_simulation(parameters: RunParameters, frame: gsd.hoomd.Frame,
                      ti_parameters: Optional[TIParameters] = None,
                      reference_frame: Optional[gsd.hoomd.Frame] = None) -> app.Simulation:
    radii = frame.particles.diameter / 2.0 * length_unit
    surface_potentials = frame.particles.charge * electric_potential_unit

    # ----------------------------------- Set up system and parameters. ------------------------------------------------
    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res", chain)

    atoms = []
    for type_id in frame.particles.typeid:
        atoms.append(topology.addAtom(frame.particles.types[type_id], None, residue))

    system = openmm.System()

    cell = get_cell_from_box(frame.configuration.box)
    include_walls = any(parameters.wall_directions)
    all_walls = all(parameters.wall_directions)
    if include_walls:
        box_vector_one = cell[0]
        box_vector_two = cell[1]
        box_vector_three = cell[2]
        assert (box_vector_one[1] == 0.0 and box_vector_one[2] == 0.0 and
                box_vector_two[0] == 0.0 and box_vector_two[2] == 0.0 and
                box_vector_three[0] == 0.0 and box_vector_three[1] == 0.0)
        wall_distances = (box_vector_one[0] * length_unit if parameters.wall_directions[0] else None,
                          box_vector_two[1] * length_unit if parameters.wall_directions[1] else None,
                          box_vector_three[2] * length_unit if parameters.wall_directions[2] else None)
        final_cell = cell.copy()
        if not all_walls:
            assert (not parameters.use_depletion
                    or parameters.depletant_radius
                    > (parameters.cutoff_factor * parameters.debye_length - 2.0 * parameters.brush_length) / 2.0)
        else:
            if parameters.use_pbc:
                warnings.warn("All walls are included, so particles will not be able to leave the box. Consider disabling periodic boundary conditions to improve performance.")
        for index, wall_direction in enumerate(parameters.wall_directions):
            if wall_direction:
                # The shifted Lennard Jones walls diverge at distance r = radius - 1 from the location of the wall,
                # where radius is the radius of the particle. The minimum distance between periodic images through
                # a wall is thus 2 * radius_min - 2, where radius_min is the smallest radius in the system.
                # The maximum cutoff of the electrostatic interactions is
                # 2 * radius_max + cutoff_factor * debye_length. In order to prevent particles from interacting
                # through the walls, we thus increase the length of the periodic box vectors (not the wall) by
                # 2 * (radius_max - radius_min) + 2 + cutoff_factor * debye_length.
                final_cell[index][index] += \
                    (2.0 * (max(radii) - min(radii)) + 2.0 * length_unit
                        + parameters.cutoff_factor * parameters.debye_length).value_in_unit(length_unit)
    else:
        wall_distances = None
        final_cell = cell

    if parameters.use_pbc:
        topology.setPeriodicBoxVectors(final_cell)
        system.setDefaultPeriodicBoxVectors(openmm.Vec3(*final_cell[0]), openmm.Vec3(*final_cell[1]),
                                            openmm.Vec3(*final_cell[2]))

    # Explicit substrate is detected by immobile particles with mass 0.0.
    use_substrate = any(mass == 0.0 for mass in frame.particles.mass) or parameters.use_implicit_substrate
    assert not (any(mass == 0.0 for mass in frame.particles.mass) and parameters.use_implicit_substrate)

    # TODO: Prevent printing the traceback when the platform is not existing.
    platform = openmm.Platform.getPlatformByName(parameters.platform_name)

    integrator = getattr(integrators, parameters.integrator)(**parameters.integrator_parameters)

    potentials_parameters = ColloidPotentialsParameters(
        brush_density=parameters.brush_density, brush_length=parameters.brush_length,
        debye_length=parameters.debye_length, temperature=parameters.potential_temperature,
        dielectric_constant=parameters.dielectric_constant)

    # ---------------------------------------- Create all forces. ------------------------------------------------------
    colloid_potentials = ColloidPotentialsAlgebraic(
        colloid_potentials_parameters=potentials_parameters, use_log=parameters.use_log,
        cutoff_factor=parameters.cutoff_factor, periodic_boundary_conditions=parameters.use_pbc,
        steric_radius_average=parameters.steric_radius_average,
        electrostatic_radius_average=parameters.electrostatic_radius_average)

    if include_walls:
        slj_walls = ShiftedLennardJonesWalls(wall_distances, parameters.epsilon, parameters.alpha,
                                             parameters.wall_directions, use_substrate, use_pbc=parameters.use_pbc)
    else:
        slj_walls = None

    if parameters.use_depletion:
        depletion_potential = DepletionPotential(parameters.depletion_phi, parameters.depletant_radius,
                                                 brush_length=parameters.brush_length,
                                                 temperature=parameters.potential_temperature,
                                                 periodic_boundary_conditions=parameters.use_pbc)
    else:
        depletion_potential = None

    if parameters.use_gravity:
        gravitational_potential = Gravity(parameters.gravitational_acceleration, parameters.water_density,
                                          parameters.particle_density)
    else:
        gravitational_potential = None

    if parameters.use_implicit_substrate:
        substrate_wall = ImplicitSubstrateWall(colloid_potentials_parameters=potentials_parameters,
                                               wall_distance_z=wall_distances[2],
                                               substrate_charge=parameters.substrate_wall_charge,
                                               use_log=parameters.use_log)
    else:
        substrate_wall = None

    if parameters.use_plumed:
        plumed = PlumedPotential(parameters.plumed_script)
    else:
        plumed = None

    if ti_parameters is not None:
        assert reference_frame is not None
        harmonic_restraint = set_up_harmonic_restraint(ti_parameters, frame, reference_frame)
    else:
        harmonic_restraint = None

    # --------------------------- Add all particles and constraints to the system. -------------------------------------
    for mass in frame.particles.mass:
        system.addParticle(mass)

    for i in range(frame.constraints.N):
        if not len(frame.constraints.group[i]) == 2:
            raise ValueError("Every constraint must have exactly two particles.")
        system.addConstraint(frame.constraints.group[i][0], frame.constraints.group[i][1], frame.constraints.value[i])

    # ------------------------------------- Add all particles to the forces. -------------------------------------------
    # Be careful to add the particles in the same order as to the system.
    for i in range(frame.particles.N):
        is_substrate = frame.particles.mass[i] == 0.0
        colloid_potentials.add_particle(radius=radii[i], surface_potential=surface_potentials[i],
                                        substrate_flag=is_substrate)
        if include_walls and not is_substrate:
            slj_walls.add_particle(index=i, radius=radii[i])
        if parameters.use_depletion:
            depletion_potential.add_particle(radius=radii[i], substrate_flag=is_substrate)
        if parameters.use_gravity and not is_substrate:
            gravitational_potential.add_particle(index=i, radius=radii[i])
        if parameters.use_implicit_substrate:
            assert not is_substrate
            substrate_wall.add_particle(index=i, radius=radii[i], surface_potential=surface_potentials[i])
        if parameters.use_plumed:
            plumed.add_particle()

    for i in range(frame.constraints.N):
        colloid_potentials.add_exclusion(frame.constraints.group[i][0], frame.constraints.group[i][1])
        if parameters.use_depletion:
            depletion_potential.add_exclusion(frame.constraints.group[i][0], frame.constraints.group[i][1])

    # -------------------------------------- Add all forces to the system. ---------------------------------------------
    for force in colloid_potentials.yield_potentials():
        force.setForceGroup(system.getNumForces())
        system.addForce(force)

    if include_walls:
        for force in slj_walls.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)

    if parameters.use_depletion:
        for force in depletion_potential.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)

    if parameters.use_gravity:
        for force in gravitational_potential.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)

    if parameters.use_implicit_substrate:
        for force in substrate_wall.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)

    if parameters.use_plumed:
        for force in plumed.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)

    if harmonic_restraint is not None:
        for force in harmonic_restraint.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)
        # The Einstein-crystal method requires the center of mass to be fixed to remove the
        # quasi-divergence of the thermodynamic-integration integrand at small coupling.
        if ti_parameters.fix_center_of_mass:
            cm_motion_remover = openmm.CMMotionRemover(1)
            cm_motion_remover.setForceGroup(system.getNumForces())
            system.addForce(cm_motion_remover)

    barostat = initialize_barostat(parameters, integrator)
    if barostat is not None:
        # The Monte Carlo barostat scales the periodic box, so it requires periodic boundary
        # conditions (i.e., not all walls active).
        if all_walls:
            raise ValueError("An NPT barostat requires periodic boundary conditions, but all walls "
                             "are active (fully closed box). Disable at least one wall or the barostat.")
        barostat.setForceGroup(system.getNumForces())
        system.addForce(barostat)

    # -------------------------------------- Set up the simulation. ----------------------------------------------------
    if parameters.platform_name == "CUDA":
        simulation = app.Simulation(topology, system, integrator, platform,
                                    platformProperties={"Precision": "mixed"})
    else:
        simulation = app.Simulation(topology, system, integrator, platform)

    return simulation


def _ensure_parent_directory(path: Optional[str]) -> None:
    """Create the parent directory of an output file if it does not exist (e.g. for output_prefix)."""
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def set_up_reporters(parameters: RunParameters, simulation: app.Simulation, append_file: bool,
                     total_number_steps: int, initial_frame: gsd.hoomd.Frame) -> None:
    # Create the output directory (e.g. the output_prefix folder) before any reporter opens a file.
    for output_path in (parameters.trajectory_filename, parameters.state_data_filename,
                        parameters.checkpoint_filename):
        _ensure_parent_directory(output_path)
    # With walls, the OpenMM box is artificially enlarged, so the true (fixed) cell is recorded.
    # Without walls the box may change during the run (e.g. under an NPT barostat), so pass cell=None
    # to record the live simulation box each frame.
    gsd_cell = (get_cell_from_box(initial_frame.configuration.box) * length_unit
                if any(parameters.wall_directions) else None)
    simulation.reporters.append(GSDReporter(parameters.trajectory_filename, parameters.trajectory_interval,
                                            initial_frame.particles.diameter / 2.0 * length_unit,
                                            initial_frame.particles.charge * electric_potential_unit, simulation,
                                            append_file=append_file, cell=gsd_cell))
    simulation.reporters.append(StatusReporter(max(1, total_number_steps // 100), total_number_steps,
                                               desc="Production"))
    simulation.reporters.append(app.StateDataReporter(parameters.state_data_filename,
                                                      parameters.state_data_interval, time=True,
                                                      kineticEnergy=True, potentialEnergy=True, temperature=True,
                                                      speed=True, append=append_file))

    if parameters.update_reporter is not None:
        update_reporter = getattr(update_reporters, parameters.update_reporter)
        try:
            simulation.reporters.append(update_reporter(simulation=simulation, append_file=append_file,
                                                        **parameters.update_reporter_parameters))
        except TypeError:
            raise TypeError(
                f"UpdateReporter does not accept the given arguments {parameters.update_reporter_parameters}. "
                f"The expected signature is {inspect.signature(update_reporter)} (the simulation and append_file "
                f"arguments should not be specified).")
    # The CheckpointReporter should always be last to ensure that all other reporters have been executed before it.
    simulation.reporters.append(app.CheckpointReporter(parameters.checkpoint_filename,
                                                       parameters.checkpoint_interval))


def colloids_run(argv: Sequence[str]) -> app.Simulation:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=f"""
PACSim Version {__version__}: A Flexible Simulation Framework for Polymer-Attenuated Coulombic Self-Assembly.

Perform a molecular-dynamics simulation using OpenMM.
""")
    parser.add_argument("yaml_file", help="YAML file with PACSim parameters", type=str)
    parser.add_argument("-c", "--checkpoint_file", help="OpenMM checkpoint file", type=str,
                        default=None)
    parser.add_argument("-t", "--ti_file", help="YAML file with thermodynamic-integration "
                        "(Frenkel-Ladd / Einstein-crystal) parameters", type=str, default=None)
    parser.add_argument("--example", help="write an example YAML file and exit", action=ExampleAction)
    args = parser.parse_args(args=argv)

    if not args.yaml_file.endswith(".yaml"):
        raise ValueError("The YAML file must have the .yaml extension.")

    parameters = RunParameters.from_yaml(args.yaml_file)

    frame = read_gsd_file(parameters.initial_configuration, parameters.frame_index)

    check_frame(parameters, frame)

    if args.ti_file is not None:
        if not args.ti_file.endswith(".yaml"):
            raise ValueError("The TI file must have the .yaml extension.")
        ti_parameters = TIParameters.from_yaml(args.ti_file)
        if ti_parameters.reference_configuration is not None:
            reference_frame = read_gsd_file(ti_parameters.reference_configuration,
                                            ti_parameters.reference_frame_index)
        else:
            reference_frame = frame
    else:
        ti_parameters = None
        reference_frame = None

    simulation = set_up_simulation(parameters, frame, ti_parameters, reference_frame)

    if args.checkpoint_file is not None:
        if not args.checkpoint_file.endswith(".chk"):
            raise ValueError("The checkpoint file must have the .chk extension.")

        simulation.loadCheckpoint(args.checkpoint_file)

        set_up_reporters(parameters, simulation, True, parameters.run_steps, frame)
    else:
        simulation.context.setPositions(frame.particles.position)

        if parameters.velocity_seed is not None:
            if not np.all(frame.particles.velocity == 0.0):
                warnings.warn("The initial velocities in the GSD file are ignored because a velocity seed is provided.")
            if parameters.velocity_seed < 0:
                simulation.context.setVelocitiesToTemperature(parameters.potential_temperature)
            else:
                simulation.context.setVelocitiesToTemperature(parameters.potential_temperature,
                                                              parameters.velocity_seed)
        else:
            if np.all(frame.particles.velocity == 0.0):
                warnings.warn(
                    "All initial velocities in the GSD file are zero. Set a velocity seed to assign random "
                    "values based on the temperature (use a negative seed to generate a random seed automatically).")
            simulation.context.setVelocities(frame.particles.velocity)

        if parameters.minimize_energy_initially:
            # Add reporter during minimization?
            # See https://openmm.github.io/openmm-cookbook/dev/notebooks/cookbook/report_minimization.html
            simulation.minimizeEnergy()

        if parameters.equilibration_steps > 0:
            simulation.reporters.append(StatusReporter(
                max(1, parameters.equilibration_steps // 100), parameters.equilibration_steps, desc="Equilibration"))
            simulation.step(parameters.equilibration_steps)
            simulation.reporters = []

        # Reset the current step to zero after the equilibration.
        simulation.currentStep = 0

        set_up_reporters(parameters, simulation, False, parameters.run_steps, frame)

    simulation.step(parameters.run_steps)

    if parameters.final_configuration_gsd_filename is not None:
        _ensure_parent_directory(parameters.final_configuration_gsd_filename)
        write_gsd_file(parameters.final_configuration_gsd_filename, simulation,
                       frame.particles.diameter / 2.0 * length_unit,
                       frame.particles.charge * electric_potential_unit,
                       get_cell_from_box(frame.configuration.box) * length_unit)

    return simulation


def main():
    colloids_run(sys.argv[1:])


if __name__ == '__main__':
    main()
