import argparse
import inspect
import itertools
import sys
from typing import Sequence
import numpy as np
import numpy.random as npr
import numpy.typing as npt
import openmm
from openmm import app
from openmm import unit
from colloids import (ColloidPotentialsAlgebraic, ColloidPotentialsParameters, ColloidPotentialsTabulated,
                      ShiftedLennardJonesWalls, SubstrateWall, DepletionPotential, Gravity)
from colloids.gsd_reporter import GSDReporter
from colloids.helper_functions import (generate_fibonacci_sphere_grid_points, read_xyz_file, write_gsd_file,
                                       write_xyz_file)
import colloids.integrators as integrators
from colloids.run_parameters import RunParameters
from colloids.substrate import substrate_positions_hexagonal
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


def set_up_simulation(parameters: RunParameters, types: Sequence[str], cell: npt.NDArray[float],
                      positions: npt.NDArray[float]) -> (app.Simulation, npt.NDArray[float]):
    # ----------------------------------- Set up system and parameters. ------------------------------------------------
    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res", chain)

    atoms = []
    for t in types:
        atoms.append(topology.addAtom(t, None, residue))

    system = openmm.System()

    include_walls = any(parameters.wall_directions)
    all_walls = all(parameters.wall_directions)
    add_implicit_substrate = parameters.use_substrate and parameters.substrate_type == "wall"

    if include_walls:
        box_vector_one = cell[0]
        box_vector_two = cell[1]
        box_vector_three = cell[2]
        if not (box_vector_one[1] == 0.0 and box_vector_one[2] == 0.0 and
                box_vector_two[0] == 0.0 and box_vector_two[2] == 0.0 and
                box_vector_three[0] == 0.0 and box_vector_three[1] == 0.0):
            raise ValueError("If any wall is included, the box vectors must be parallel to the coordinate axes.")
        wall_distances = (box_vector_one[0] * (unit.nano * unit.meter) if parameters.wall_directions[0] else None,
                          box_vector_two[1] * (unit.nano * unit.meter) if parameters.wall_directions[1] else None,
                          box_vector_three[2] * (unit.nano * unit.meter) if parameters.wall_directions[2] else None)
        final_cell = cell.copy()
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
        final_cell = cell

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

    # ------------------------------------ Create additional particles. ------------------------------------------------
    snowman_positions = []
    if parameters.use_snowman:
        if parameters.snowman_seed is not None and parameters.snowman_seed > 0:
            npr.seed(parameters.snowman_seed)
        nanometer = unit.nano * unit.meter
        for i, t in enumerate(types):
            if t in parameters.snowman_bond_types:
                snowman_type = parameters.snowman_bond_types[t]
                snowman_atom = topology.addAtom(snowman_type, None, residue)
                topology.addBond(atoms[i], snowman_atom)
                offset = list(generate_fibonacci_sphere_grid_points(
                    1, parameters.snowman_distances[t].value_in_unit(nanometer),
                    parameters.snowman_seed is None or parameters.snowman_seed > 0))[0]
                assert abs(np.linalg.norm(offset) - parameters.snowman_distances[t].value_in_unit(nanometer)) < 1.0e-12
                snowman_positions.append(positions[i] + offset)
            else:
                snowman_positions.append(None)

    if parameters.use_substrate and not add_implicit_substrate:
        substrate_positions = substrate_positions_hexagonal(parameters.radii[parameters.substrate_type], cell)
        for _ in substrate_positions:
            # Setting the mass to zero tells the integrator that the particle is immobile.
            # See http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.System.html.
            topology.addAtom(parameters.substrate_type, None, residue)
    else:
        substrate_positions = []

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

    if add_implicit_substrate:
        substrate_wall = SubstrateWall(colloid_potentials_parameters=potentials_parameters, 
                                       wall_distance=wall_distances[2],
                                       wall_charge=parameters.surface_potentials[parameters.substrate_type], 
                                       use_log=parameters.use_log)
    
    
    if parameters.use_depletion:
        depletion_potential = DepletionPotential(parameters.depletion_phi, parameters.depletant_radius,
                                                 brush_length=parameters.brush_length,
                                                 temperature=parameters.potential_temperature,
                                                 periodic_boundary_conditions=not all_walls)
    else:
        depletion_potential = None

    if parameters.use_gravity:
        gravitational_potential = Gravity(parameters.gravitational_acceleration, parameters.water_density,
                                          parameters.particle_density)
    else:
        gravitational_potential = None

    # ------------------------------------- Add all particles to the system. -------------------------------------------
    for t in types:
        system.addParticle(parameters.masses[t])

    snowman_indices = []
    if parameters.use_snowman:
        assert len(types) == len(snowman_positions)
        for i, (t, snowman_position) in enumerate(zip(types, snowman_positions)):
            if snowman_position is not None:
                snowman_type = parameters.snowman_bond_types[t]
                snowman_index = system.addParticle(parameters.masses[snowman_type])
                snowman_indices.append(snowman_index)
                system.addConstraint(i, snowman_index, parameters.snowman_distances[t])
            else:
                snowman_indices.append(None)

    if parameters.use_substrate and not add_implicit_substrate:
        for _ in substrate_positions:
            system.addParticle(parameters.masses[parameters.substrate_type])

    # ------------------------------------- Add all particles to the forces. -------------------------------------------
    # Be careful to add the particles in the same order as to the system.
    for i, t in enumerate(types):
        colloid_potentials.add_particle(radius=parameters.radii[t],
                                        surface_potential=parameters.surface_potentials[t],
                                        substrate_flag=(t == parameters.substrate_type),
                                        type_flag=(t == 'P'))
        if include_walls:
            slj_walls.add_particle(index=i, radius=parameters.radii[t])
        if add_implicit_substrate:
            substrate_wall.add_particle(index=i, radius=parameters.radii[t],
                                        surface_potential=parameters.surface_potentials[t])
        if parameters.use_depletion:
            depletion_potential.add_particle(radius=parameters.radii[t], substrate_flag=False)
        if parameters.use_gravity:
            gravitational_potential.add_particle(index=i, radius=parameters.radii[t])

    if parameters.use_snowman:
        assert len(types) == len(snowman_positions) == len(snowman_indices)
        for i, (t, snowman_position, snowman_index) in enumerate(zip(types, snowman_positions, snowman_indices)):
            if snowman_position is not None:
                assert snowman_index is not None
                snowman_type = parameters.snowman_bond_types[t]
                colloid_potentials.add_particle(radius=parameters.radii[snowman_type],
                                                surface_potential=parameters.surface_potentials[snowman_type],
                                                substrate_flag=False)
                colloid_potentials.add_exclusion(i, snowman_index)
            else:
                assert snowman_index is None

        if include_walls:
            for i, (t, snowman_position, snowman_index) in enumerate(zip(types, snowman_positions, snowman_indices)):
                if snowman_position is not None:
                    assert snowman_index is not None
                    slj_walls.add_particle(index=snowman_index,
                                           radius=parameters.radii[parameters.snowman_bond_types[t]])
                else:
                    assert snowman_index is None

        if parameters.use_depletion:
            for i, (t, snowman_position, snowman_index) in enumerate(zip(types, snowman_positions, snowman_indices)):
                if snowman_position is not None:
                    assert snowman_index is not None
                    snowman_type = parameters.snowman_bond_types[t]
                    depletion_potential.add_particle(radius=parameters.radii[snowman_type], substrate_flag=False)
                    depletion_potential.add_exclusion(i, snowman_index)
                else:
                    assert snowman_index is None

        if parameters.use_gravity:
            for i, (t, snowman_position, snowman_index) in enumerate(zip(types, snowman_positions, snowman_indices)):
                if snowman_position is not None:
                    assert snowman_index is not None
                    gravitational_potential.add_particle(index=snowman_index,
                                                         radius=parameters.radii[parameters.snowman_bond_types[t]])
                else:
                    assert snowman_index is None

    if parameters.use_substrate and not add_implicit_substrate:
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

    if add_implicit_substrate:
        for force in substrate_wall.yield_potentials():
            system.addForce(force)

    if parameters.use_depletion:
        for force in depletion_potential.yield_potentials():
            system.addForce(force)

    if parameters.use_gravity:
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

    extra_positions = np.array([p for p in itertools.chain(snowman_positions, substrate_positions) if p is not None])
    return simulation, extra_positions


def set_up_reporters(parameters: RunParameters, simulation: app.Simulation, append_file: bool,
                     total_number_steps: int, cell: npt.NDArray[float]) -> None:
    simulation.reporters.append(GSDReporter(parameters.trajectory_filename, parameters.trajectory_interval,
                                            parameters.radii, parameters.surface_potentials, simulation,
                                            append_file=append_file,
                                            cell=cell * (unit.nano * unit.meter)))
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
    parameters.check_types_of_initial_configuration()

    types, positions, cell = read_xyz_file(parameters.initial_configuration)

    simulation, extra_positions = set_up_simulation(parameters, types, cell, positions)

    simulation.context.setPositions(np.concatenate((positions, extra_positions)) if len(extra_positions) > 0
                                    else positions)
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

    set_up_reporters(parameters, simulation, False, parameters.run_steps, cell)

    simulation.step(parameters.run_steps)

    # TODO: Automatically plot energies etc.
    # TODO: CHECK ALL SURFACE SEPARATIONS

    if parameters.final_configuration_gsd_filename is not None:
        write_gsd_file(parameters.final_configuration_gsd_filename, simulation, parameters.radii,
                       parameters.surface_potentials, cell * (unit.nano * unit.meter))

    if parameters.final_configuration_xyz_filename is not None:
        write_xyz_file(parameters.final_configuration_xyz_filename, simulation, cell * (unit.nano * unit.meter))

    return simulation


def main():
    colloids_run(sys.argv[1:])


if __name__ == '__main__':
    main()