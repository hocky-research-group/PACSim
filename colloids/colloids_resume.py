import argparse
import sys
from typing import Sequence
from openmm import app
from colloids.colloids_run import set_up_simulation, set_up_reporters, check_frame, get_cell_from_box
from colloids.helper_functions import read_gsd_file, write_gsd_file
from colloids.run_parameters import RunParameters
from colloids.units import electric_potential_unit, length_unit, time_unit


def colloids_resume(argv: Sequence[str]) -> app.Simulation:
    parser = argparse.ArgumentParser(description="Resume OpenMM for a colloids system.")
    parser.add_argument("yaml_file", help="YAML file with simulation parameters", type=str)
    parser.add_argument("checkpoint_file", help="checkpoint file of OpenMM", type=str)
    parser.add_argument("number_steps", help="number of steps to run", type=int)
    args = parser.parse_args(args=argv)

    if not args.yaml_file.endswith(".yaml"):
        raise ValueError("The YAML file must have the .yaml extension.")
    if (not args.checkpoint_file.endswith(".chk")) and (not args.checkpoint_file.endswith(".gsd")):
        raise ValueError("The checkpoint file must have the .chk or .gsd extension.")
    if not args.number_steps > 0:
        raise ValueError("The number of steps must be positive.")

    parameters = RunParameters.from_yaml(args.yaml_file)

    if args.checkpoint_file.endswith(".gsd"):
        # If the checkpoint file is a GSD file, read the gsd file
        frame = read_gsd_file(args.checkpoint_file, -1)
        check_frame(parameters, frame)

        simulation = set_up_simulation(parameters, frame)
        simulation.context.setPositions(frame.particles.position * length_unit)
        simulation.context.setVelocities(frame.particles.velocity * length_unit / time_unit)

    elif args.checkpoint_file.endswith(".chk"):
        # If the checkpoint file is a checkpoint file, load the state from it
        frame = read_gsd_file(parameters.initial_configuration, parameters.frame_index)
        check_frame(parameters, frame)

        simulation = set_up_simulation(parameters, frame)
        simulation.loadCheckpoint(args.checkpoint_file)

    set_up_reporters(parameters, simulation, True, args.number_steps, frame)

    simulation.step(args.number_steps)

    if parameters.final_configuration_gsd_filename is not None:
        write_gsd_file(parameters.final_configuration_gsd_filename, simulation,
                       frame.particles.diameter / 2.0 * length_unit,
                       frame.particles.charge * electric_potential_unit,
                       get_cell_from_box(frame.configuration.box) * length_unit)

    return simulation


def main():
    colloids_resume(sys.argv[1:])


if __name__ == '__main__':
    main()
