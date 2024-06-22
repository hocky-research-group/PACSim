import argparse
from typing import Iterable
import numpy as np
import numpy.typing as npt
import numpy.random as npr
import openmm
from openmm import app
from openmm import unit
from colloids import (ColloidPotentialsAlgebraic, ColloidPotentialsParameters, ColloidPotentialsTabulated,
                      ShiftedLennardJonesWalls)
from colloids.gsd_reporter import GSDReporter
from colloids.helper_functions import (generate_fibonacci_sphere_grid_points, read_xyz_file, write_gsd_file,
                                       write_xyz_file)
from colloids.run_parameters import RunParameters
from colloids.status_reporter import StatusReporter


class ExampleAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # TODO ADD OPTION FOR PLATFORM PROPERTIES?
        # TODO PUT EQUILIBRATION STEPS?
        default_parameters = RunParameters()
        default_parameters.to_yaml("example.yaml")
        parser.exit()


def set_up_simulation(parameters: RunParameters, types: Iterable[str],
                      cell: npt.NDArray[float]) -> (app.Simulation, npt.NDArray[float]):
    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res1", chain)

    atoms = []
    for t in types:
        atoms.append(topology.addAtom(t, None, residue))

    system = openmm.System()

    include_walls = any(parameters.wall_directions)
    all_walls = all(parameters.wall_directions)
    if include_walls:
        box_vector_one = cell[0]
        box_vector_two = cell[1]
        box_vector_three = cell[2]
        if not (box_vector_one[1] == 0.0 and box_vector_one[2] == 0.0 and
                box_vector_two[0] == 0.0 and box_vector_two[2] == 0.0 and
                box_vector_three[0] == 0.0 and box_vector_three[1] == 0.0):
            raise ValueError("If any wall is included, the box vectors must be parallel to the coordinate axes.")
        wall_distances = (box_vector_one[0] * (unit.nano * unit.meter) if parameters.wall_directions[0] else None,
                          box_vector_two[1] * (unit.nano * unit.meter)if parameters.wall_directions[1] else None,
                          box_vector_three[2] * (unit.nano * unit.meter) if parameters.wall_directions[2] else None)
        final_cell = cell.copy()
        if not all_walls:
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

    # Prevent printing the traceback when the platform is not existing.
    platform = openmm.Platform.getPlatformByName(parameters.platform_name)

    # TODO: ALLOW FOR DIFFERENT INTEGRATORS?
    integrator = openmm.LangevinIntegrator(parameters.temperature,
                                           parameters.collision_rate,
                                           parameters.timestep)
    if parameters.integrator_seed is not None:
        integrator.setRandomNumberSeed(parameters.integrator_seed)

    potentials_parameters = ColloidPotentialsParameters(
        brush_density=parameters.brush_density, brush_length=parameters.brush_length,
        debye_length=parameters.debye_length, temperature=parameters.temperature,
        dielectric_constant=parameters.dielectric_constant
    )

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

    for t in types:
        system.addParticle(parameters.masses[t])
        colloid_potentials.add_particle(radius=parameters.radii[t],
                                        surface_potential=parameters.surface_potentials[t])

    if include_walls:
        slj_walls = ShiftedLennardJonesWalls(wall_distances, parameters.epsilon, parameters.alpha,
                                             parameters.wall_directions)
        for i, t in enumerate(types):
            slj_walls.add_particle(index=i, radius=parameters.radii[t])
    else:
        slj_walls = None

    snowman_positions = []
    if parameters.snowman_masses is not None:
        assert parameters.snowman_radii is not None
        assert parameters.snowman_distances is not None
        if parameters.snowman_seed is not None:
            npr.seed(parameters.snowman_seed)
        nanometer = unit.nano * unit.meter
        for i, t in enumerate(types):
            snowman_type = t + t
            if parameters.snowman_masses[snowman_type] is not None:
                assert parameters.snowman_radii[snowman_type] is not None
                assert parameters.snowman_distances[snowman_type] is not None
                snowman_atom = topology.addAtom(snowman_type, None, residue)
                topology.addBond(atoms[i], snowman_atom)
                snowman_index = system.addParticle(parameters.snowman_masses[snowman_type])
                colloid_potentials.add_particle(radius=parameters.snowman_radii[snowman_type],
                                                surface_potential=parameters.snowman_surface_potentials[snowman_type])
                system.addConstraint(i, snowman_index, parameters.snowman_distances[snowman_type])
                colloid_potentials.add_exclusion(i, snowman_index)
                if include_walls:
                    slj_walls.add_particle(index=snowman_index, radius=parameters.snowman_radii[snowman_type])
                pos = list(generate_fibonacci_sphere_grid_points(
                    1, parameters.snowman_distances[snowman_type].value_in_unit(nanometer),
                    True))[0]
                snowman_positions.append(pos)
    snowman_positions = np.array(snowman_positions)

    for force in colloid_potentials.yield_potentials():
        system.addForce(force)
    if include_walls:
        for force in slj_walls.yield_potentials():
            system.addForce(force)

    if parameters.platform_name == "CUDA" or parameters.platform_name == "OpenCL":
        simulation = app.Simulation(topology, system, integrator, platform,
                                    platformProperties={"Precision": "mixed"})
    else:
        simulation = app.Simulation(topology, system, integrator, platform)

    return simulation, snowman_positions


def set_up_reporters(parameters: RunParameters, simulation: app.Simulation, append_file: bool,
                     total_number_steps: int, cell: npt.NDArray[float]) -> None:
    snowman_radii = parameters.snowman_radii if parameters.snowman_radii is not None else {}
    snowman_surface_potentials = (parameters.snowman_surface_potentials
                                  if parameters.snowman_surface_potentials is not None else {})
    assert all(r not in snowman_radii for r in parameters.radii)
    assert all(r not in snowman_surface_potentials for r in parameters.surface_potentials)
    simulation.reporters.append(GSDReporter(parameters.trajectory_filename, parameters.trajectory_interval,
                                            parameters.radii |
                                            {k: v for k, v in snowman_radii.items() if v is not None},
                                            parameters.surface_potentials |
                                            {k: v for k, v in snowman_surface_potentials.items() if v is not None},
                                            simulation, append_file=append_file, cell=cell * (unit.nano * unit.meter)))
    simulation.reporters.append(StatusReporter(max(1, total_number_steps // 100), total_number_steps))
    simulation.reporters.append(app.StateDataReporter(parameters.state_data_filename,
                                                      parameters.state_data_interval, time=True,
                                                      kineticEnergy=True, potentialEnergy=True, temperature=True,
                                                      speed=True, append=append_file))
    simulation.reporters.append(app.CheckpointReporter(parameters.checkpoint_filename,
                                                       parameters.checkpoint_interval))


def main():
    parser = argparse.ArgumentParser(description="Run OpenMM for a colloids system.")
    parser.add_argument("yaml_file", help="YAML file with simulation parameters", type=str)
    parser.add_argument("--example", help="write an example YAML file and exit", action=ExampleAction)
    args = parser.parse_args()

    if not args.yaml_file.endswith(".yaml"):
        raise ValueError("The YAML file must have the .yaml extension.")

    parameters = RunParameters.from_yaml(args.yaml_file)
    parameters.check_types_of_initial_configuration()

    types, positions, cell = read_xyz_file(parameters.initial_configuration)

    simulation, extra_positions = set_up_simulation(parameters, types, cell)

    simulation.context.setPositions(np.concatenate((positions, extra_positions)) if len(extra_positions) > 0
                                    else positions)
    if parameters.velocity_seed is not None:
        simulation.context.setVelocitiesToTemperature(parameters.temperature,
                                                      parameters.velocity_seed)
    else:
        simulation.context.setVelocitiesToTemperature(parameters.temperature)

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
                       parameters.surface_potentials)

    if parameters.final_configuration_xyz_filename is not None:
        write_xyz_file(parameters.final_configuration_xyz_filename, simulation)


if __name__ == '__main__':
    main()
