import argparse
import sys
from typing import Sequence
from openmm import app
from openmm import unit
from colloids.colloids_run import set_up_simulation, set_up_reporters
from colloids.helper_functions import read_xyz_file, write_gsd_file, write_xyz_file
from colloids.run_parameters import RunParameters


def colloids_resume(argv: Sequence[str]) -> app.Simulation:
    parser = argparse.ArgumentParser(description="Resume OpenMM for a colloids system.")
    parser.add_argument("yaml_file", help="YAML file with simulation parameters", type=str)
    parser.add_argument("checkpoint_file", help="checkpoint file of OpenMM", type=str)
    parser.add_argument("number_steps", help="number of steps to run", type=int)
    args = parser.parse_args(args=argv)

    if not args.yaml_file.endswith(".yaml"):
        raise ValueError("The YAML file must have the .yaml extension.")
    if not args.checkpoint_file.endswith(".chk"):
        raise ValueError("The checkpoint file must have the .chk extension.")
    if not args.number_steps > 0:
        raise ValueError("The number of steps must be positive.")

    parameters = RunParameters.from_yaml(args.yaml_file)
    types, positions, cell = read_xyz_file(parameters.initial_configuration)

    simulation, _ = set_up_simulation(parameters, types, cell, positions)

    simulation.loadCheckpoint(args.checkpoint_file)

    set_up_reporters(parameters, simulation, True, args.number_steps, cell)

    simulation.step(args.number_steps)

    if parameters.final_configuration_gsd_filename is not None:
        write_gsd_file(parameters.final_configuration_gsd_filename, simulation, parameters.radii,
                       parameters.surface_potentials, cell * (unit.nano * unit.meter))

    if parameters.final_configuration_xyz_filename is not None:
        write_xyz_file(parameters.final_configuration_xyz_filename, simulation, cell * (unit.nano * unit.meter))

    return simulation


def main():
    colloids_resume(sys.argv[1:])


if __name__ == '__main__':
    main()
