import argparse
import numpy.typing as npt
import openmm
from openmm import app
from colloids import ColloidPotentialsAlgebraic, ColloidPotentialsParameters, ColloidPotentialsTabulated
from colloids.gsd_reporter import GSDReporter
from colloids.helper_functions import read_xyz_file, write_gsd_file, write_xyz_file
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


def set_up_simulation(parameters: RunParameters, types: npt.NDArray[str],
                      cell: npt.NDArray[float]) -> app.Simulation:
    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res1", chain)

    for t in types:
        topology.addAtom(t, None, residue)

    topology.setPeriodicBoxVectors(cell)

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(*cell[0]), openmm.Vec3(*cell[1]), openmm.Vec3(*cell[2]))
    # Prevent printing the traceback when the platform is not existing.
    platform = openmm.Platform.getPlatformByName(parameters.platform_name)

    # TODO: ALLOW FOR DIFFERENT INTEGRATORS?
    #integrator = openmm.LangevinMiddleIntegrator(parameters.temperature,
                                                 #parameters.collision_rate,
                                                 #parameters.timestep)
   
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
            cutoff_factor=parameters.cutoff_factor)
    else:
        colloid_potentials = ColloidPotentialsAlgebraic(
            colloid_potentials_parameters=potentials_parameters, use_log=parameters.use_log,
            cutoff_factor=parameters.cutoff_factor)

    for t in types:
        system.addParticle(parameters.masses[t])
        colloid_potentials.add_particle(radius=parameters.radii[t],
                                        surface_potential=parameters.surface_potentials[t])
    for force in colloid_potentials.yield_potentials():
        system.addForce(force)

    if parameters.platform_name == "CUDA" or parameters.platform_name == "OpenCL":
        simulation = app.Simulation(topology, system, integrator, platform,
                                    platformProperties={"Precision": "mixed"})
    else:
        simulation = app.Simulation(topology, system, integrator, platform)

    return simulation


def set_up_reporters(parameters: RunParameters, simulation: app.Simulation, append_file: bool,
                     total_number_steps: int) -> None:
    simulation.reporters.append(GSDReporter(parameters.trajectory_filename, parameters.trajectory_interval,
                                            parameters.radii, parameters.surface_potentials, simulation,
                                            append_file=append_file))
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

    simulation = set_up_simulation(parameters, types, cell)

    simulation.context.setPositions(positions)
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

    set_up_reporters(parameters, simulation, False, parameters.run_steps)

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
