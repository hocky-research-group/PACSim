import argparse
import inspect
import itertools
import sys
from typing import Sequence
import MDAnalysis.analysis.distances
import numpy as np
import numpy.random as npr
import numpy.typing as npt
import openmm
import gsd.hoomd
from openmm import app
from openmm import unit
from colloids import (ColloidPotentialsAlgebraic, ColloidPotentialsParameters, ColloidPotentialsTabulated,
                      ShiftedLennardJonesWalls, DepletionPotential, Gravity)
from colloids.gsd_reporter import GSDReporter
from colloids.helper_functions import (read_gsd_file, write_gsd_file, write_xyz_file)
import colloids.integrators as integrators
from colloids.run_parameters import RunParameters
from colloids.status_reporter import StatusReporter
import colloids.update_reporters as update_reporters


class ExampleAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # TODO ADD OPTION FOR PLATFORM PROPERTIES?
        # TODO PUT EQUILIBRATION STEPS?
        default_parameters = RunParameters()
        default_parameters.to_yaml("example.yaml")
        parser.exit()


def set_up_simulation(parameters: RunParameters, frame: gsd.hoomd.Frame):
    # ----------------------------------- Set up system and parameters. ------------------------------------------------
    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res", chain)

    atoms = []
    for t in frame.particles.typeid:
        atoms.append(topology.addAtom(frame.particles.types[t], None, residue))

    system = openmm.System()

    include_walls = any(parameters.wall_directions)
    all_walls = all(parameters.wall_directions)

    box_vector_one = np.array([frame.configuration.box[0], 0.0, 0.0]) 
    box_vector_two = np.array([0.0, frame.configuration.box[1], 0.0]) 
    box_vector_three = np.array([0.0, 0.0, frame.configuration.box[2]]) 
    final_cell = np.array([box_vector_one, box_vector_two, box_vector_three]) * (unit.nano * unit.meter)

    if include_walls:
        if not (box_vector_one[1] == 0.0 and box_vector_one[2] == 0.0 and
                box_vector_two[0] == 0.0 and box_vector_two[2] == 0.0 and
                box_vector_three[0] == 0.0 and box_vector_three[1] == 0.0):
            raise ValueError("If any wall is included, the box vectors must be parallel to the coordinate axes.")
        wall_distances = (box_vector_one[0] * (unit.nano * unit.meter) if parameters.wall_directions[0] else None,
                          box_vector_two[1] * (unit.nano * unit.meter) if parameters.wall_directions[1] else None,
                          box_vector_three[2] * (unit.nano * unit.meter) if parameters.wall_directions[2] else None)

        if not all_walls:
            if parameters.use_depletion:
                if (parameters.depletant_radius
                        > (parameters.cutoff_factor * parameters.debye_length - 2.0 * parameters.brush_length) / 2.0):
                    raise ValueError("the depletant radius is too large for the cutoff factor and brush length when "
                                     "partial walls are included (r_d <= (cutoff_factor * lambda_D - 2 * L) / 2)")
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
                        (2.0 * (max(parameters.radii.values()) - min(parameters.radii.values()))
                         + 2.0 * (unit.nano * unit.meter)
                         + parameters.cutoff_factor * parameters.debye_length).value_in_unit(unit.nano * unit.meter)
    else:
        wall_distances = None

    if not all_walls:
        topology.setPeriodicBoxVectors(final_cell)
        system.setDefaultPeriodicBoxVectors(openmm.Vec3(*final_cell[0]), openmm.Vec3(*final_cell[1]),
                                            openmm.Vec3(*final_cell[2]))

    # TODO: Prevent printing the traceback when the platform is not existing.
    platform = openmm.Platform.getPlatformByName(parameters.platform_name)

    integrator = getattr(integrators, parameters.integrator)(**parameters.integrator_parameters)

    potentials_parameters = ColloidPotentialsParameters(
        brush_density=parameters.brush_density, brush_length=parameters.brush_length,
        debye_length=parameters.debye_length, temperature=parameters.potential_temperature,
        dielectric_constant=parameters.dielectric_constant
    )

    substrate_positions = []
    substrate_in_initial_configuration = False
    if parameters.use_substrate:
        assert all_walls
        if parameters.substrate_type in frame.particles.types:
            print("[INFO] Substrate type is present in the initial configuration.")
            substrate_in_initial_configuration = True
        else:
            substrate_in_initial_configuration = False
            substrate_positions = substrate_positions_hexagonal(parameters.radii[parameters.substrate_type], cell)
            for _ in substrate_positions:
                # Setting the mass to zero tells the integrator that the particle is immobile.
                # See http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.System.html.
                topology.addAtom(parameters.substrate_type, None, residue)

    # ---------------------------------------- Create all forces. ------------------------------------------------------
    if parameters.use_tabulated:
        # TODO: Maybe generalize tabulated potentials to more than two types.
        # Use a dictionary instead of a set to preserve the order of the types.
        # See https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
        # Works since Python 3.7.
        set_of_types = list(dict.fromkeys(types))
        if not len(set_of_types) == 2:
            raise ValueError("Tabulated potentials only supports two types.")
        first_type = set_of_types.pop()
        second_type = set_of_types.pop()
        colloid_potentials = ColloidPotentialsTabulated(
            radius_one=parameters.radii[first_type], radius_two=parameters.radii[second_type],
            surface_potential_one=parameters.surface_potentials[first_type],
            surface_potential_two=parameters.surface_potentials[second_type],
            colloid_potentials_parameters=potentials_parameters, use_log=parameters.use_log,
            cutoff_factor=parameters.cutoff_factor, periodic_boundary_conditions=not all_walls)
    else:
        colloid_potentials = ColloidPotentialsAlgebraic(
            colloid_potentials_parameters=potentials_parameters, use_log=parameters.use_log,
            cutoff_factor=parameters.cutoff_factor, periodic_boundary_conditions=not all_walls)

    if include_walls:
        slj_walls = ShiftedLennardJonesWalls(wall_distances, parameters.epsilon, parameters.alpha,
                                             parameters.wall_directions, parameters.use_substrate)
    else:
        slj_walls = None

    if parameters.use_depletion:
        depletion_potential = DepletionPotential(parameters.depletion_phi, parameters.depletant_radius,
                                                 brush_length=parameters.brush_length,
                                                 temperature=parameters.potential_temperature,
                                                 periodic_boundary_conditions=not all_walls)
    else:
        depletion_potential = None

    if parameters.use_gravity:
        assert all_walls
        gravitational_potential = Gravity(parameters.gravitational_acceleration, parameters.water_density,
                                          parameters.particle_density)
    else:
        gravitational_potential = None

    # ------------------------------------- Add all particles to the system. -------------------------------------------
    for mass in frame.particles.mass:
        system.addParticle(mass)

    if parameters.use_clusters:
        for constraint_indices in range(frame.constraints.N):
            constraint = frame.constraints.group[constraint_indices]
            distance = frame.constraints.value[constraint_indices]

            i, j = constraint
            system.addConstraint(i, j, distance)

    if parameters.use_substrate and substrate_in_initial_configuration:
        assert substrate_positions == []

    if parameters.use_substrate and not substrate_in_initial_configuration:
        for _ in substrate_positions:
            system.addParticle(parameters.masses[parameters.substrate_type])

    # ------------------------------------- Add all particles to the forces. -------------------------------------------
    # Be careful to add the particles in the same order as to the system.
    for i, t in enumerate(frame.particles.typeid):
        colloid_potentials.add_particle(radius=frame.particles.diameter[i] / 2.0 * unit.nanometer,
                                        surface_potential=frame.particles.charge[i] * (unit.milli * unit.volt),
                                        substrate_flag=(frame.particles.types[t] == parameters.substrate_type))
        if include_walls:
            if t != parameters.substrate_type:
                slj_walls.add_particle(index=i, radius=frame.particles.diameter[i] / 2.0 * unit.nanometer)
        if parameters.use_depletion:
            depletion_potential.add_particle(radius=frame.particles.diameter[i] / 2.0 * unit.nanometer,
                                             substrate_flag=(frame.particles.types[t] == parameters.substrate_type))
        if parameters.use_gravity:
            if t != parameters.substrate_type:
                gravitational_potential.add_particle(index=i, radius=frame.particles.diameter[i] / 2.0 * unit.nanometer,)

    if parameters.use_clusters:
        # add exclusions
        for constraint in frame.constraints.group:
            i, j = constraint
            colloid_potentials.add_exclusion(i, j)

        if parameters.use_depletion:
            for constraint in frame.constraints.group:
                i, j = constraint
                depletion_potential.add_exclusion(i, j)

    if parameters.use_substrate and not substrate_in_initial_configuration:
        # No need to add the substrate particles to the wall and gravitational potential as they are immobile.
        for _ in substrate_positions:
            colloid_potentials.add_particle(radius=parameters.radii[parameters.substrate_type],
                                            surface_potential=parameters.surface_potentials[parameters.substrate_type],
                                            substrate_flag=True)
        if parameters.use_depletion:
            for _ in substrate_positions:
                depletion_potential.add_particle(radius=parameters.radii[parameters.substrate_type],
                                                 substrate_flag=True)

    # -------------------------------------- Add all forces to the system. ---------------------------------------------
    for force in colloid_potentials.yield_potentials():
        system.addForce(force)

    if include_walls:
        for force in slj_walls.yield_potentials():
            system.addForce(force)

    if parameters.use_depletion:
        for force in depletion_potential.yield_potentials():
            system.addForce(force)

    if parameters.use_gravity:
        assert all_walls
        for force in gravitational_potential.yield_potentials():
            system.addForce(force)
        assert not system.usesPeriodicBoundaryConditions()

    # -------------------------------------- Set up the simulation. ----------------------------------------------------
    if parameters.platform_name == "CUDA" or parameters.platform_name == "OpenCL":
        # Set different force groups for the nonbonded potentials to allow for different cutoffs on the OpenCL and CUDA
        # platforms.
        cutoffs = []
        for force in system.getForces():
            if isinstance(force, (openmm.NonbondedForce, openmm.CustomNonbondedForce)):
                assert (force.getNonbondedMethod() == openmm.NonbondedForce.CutoffPeriodic
                        or force.getNonbondedMethod() == openmm.NonbondedForce.CutoffNonPeriodic)
                cutoff_distance = force.getCutoffDistance()
                cutoff_distance_index = -1
                for other_cutoff_index in range(len(cutoffs)):
                    if abs((cutoff_distance - cutoffs[other_cutoff_index]).value_in_unit(
                            unit.nano * unit.meter)) < 1.0e-6:
                        cutoff_distance_index = other_cutoff_index
                if cutoff_distance_index == -1:
                    cutoffs.append(cutoff_distance)
                    cutoff_distance_index = len(cutoffs) - 1
                else:
                    force.setCutoffDistance(cutoffs[cutoff_distance_index])
                force.setForceGroup(cutoff_distance_index)

    if parameters.platform_name == "CUDA":
        simulation = app.Simulation(topology, system, integrator, platform,
                                    platformProperties={"Precision": "mixed"})
    else:
        simulation = app.Simulation(topology, system, integrator, platform)

    extra_positions = np.array([p for p in itertools.chain(substrate_positions) if p is not None])
    return simulation, extra_positions


def set_up_reporters(parameters: RunParameters, simulation: app.Simulation, append_file: bool,
                     total_number_steps: int, frame: gsd.hoomd.Frame) -> None:
    simulation.reporters.append(GSDReporter(parameters.trajectory_filename, parameters.trajectory_interval,
                                            frame, simulation, append_file=append_file))
    simulation.reporters.append(StatusReporter(max(1, total_number_steps // 100), total_number_steps))
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
                f"The expected signature is {inspect.signature(update_reporter)} (the simulation argument need not be "
                f"specified).")
    # The CheckpointReporter should always be last to ensure that all other reporters have been executed before it.
    simulation.reporters.append(app.CheckpointReporter(parameters.checkpoint_filename,
                                                       parameters.checkpoint_interval))


def colloids_run(argv: Sequence[str]) -> app.Simulation:
    parser = argparse.ArgumentParser(description="Run OpenMM for a colloids system.")
    parser.add_argument("yaml_file", help="YAML file with simulation parameters", type=str)
    parser.add_argument("--example", help="write an example YAML file and exit", action=ExampleAction)
    args = parser.parse_args(args=argv)

    if not args.yaml_file.endswith(".yaml"):
        raise ValueError("The YAML file must have the .yaml extension.")

    parameters = RunParameters.from_yaml(args.yaml_file)

    frame = read_gsd_file(parameters.initial_configuration)

    simulation, extra_positions = set_up_simulation(parameters, frame)

    simulation.context.setPositions(np.concatenate((frame.particles.position, extra_positions)) if len(extra_positions) > 0
                                    else frame.particles.position)
    if parameters.velocity_seed is not None:
        simulation.context.setVelocitiesToTemperature(parameters.potential_temperature,
                                                      parameters.velocity_seed)
    else:
        simulation.context.setVelocitiesToTemperature(parameters.potential_temperature)

    if parameters.minimize_energy_initially:
        # TODO: Do we want this?
        # Add reporter during minimization?
        # See https://openmm.github.io/openmm-cookbook/dev/notebooks/cookbook/report_minimization.html
        simulation.minimizeEnergy()

    set_up_reporters(parameters, simulation, False, parameters.run_steps, frame)

    simulation.step(parameters.run_steps)

    # TODO: Automatically plot energies etc.
    # TODO: CHECK ALL SURFACE SEPARATIONS

    if parameters.final_configuration_gsd_filename is not None:
        write_gsd_file(parameters.final_configuration_gsd_filename, simulation, frame)

    return simulation


def main():
    colloids_run(sys.argv[1:])


if __name__ == '__main__':
    main()
