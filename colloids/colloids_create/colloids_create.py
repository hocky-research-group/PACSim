import argparse
from openmm import unit
from colloids.run_parameters import RunParameters
from colloids.colloids_create.configuration_parameters import ConfigurationParameters
from colloids.colloids_create.cubic_lattice_with_satellites_generator import (CubicLattice,
                                                                              CubicLatticeWithSatellitesGenerator)


class ExampleAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        default_parameters = ConfigurationParameters()
        default_parameters.to_yaml("example_configuration.yaml")
        parser.exit()


def main():
    parser = argparse.ArgumentParser(description="Create an initial configuration for an OpenMM simulation of a "
                                                 "colloids system.")
    parser.add_argument("simulation_parameters", help="YAML file with simulation parameters", type=str)
    parser.add_argument("configuration_parameters", help="YAML file with configuration parameters",
                        type=str)
    parser.add_argument("--example", help="write an example configuration YAML file and exit",
                        action=ExampleAction)
    args = parser.parse_args()

    if not args.simulation_parameters.endswith(".yaml"):
        raise ValueError("The YAML file for the simulation parameters must have the .yaml extension.")
    if not args.configuration_parameters.endswith(".yaml"):
        raise ValueError("The YAML file for the configuration parameters must have the .yaml extension.")

    run_parameters = RunParameters.from_yaml(args.simulation_parameters)
    configuration_parameters = ConfigurationParameters.from_yaml(args.configuration_parameters)

    relevant_radii = {k: v for k, v in run_parameters.radii.items()
                      if k != run_parameters.substrate_type
                      and (k not in run_parameters.snowman_bond_types.values()
                           if run_parameters.snowman_bond_types is not None else True)}

    if not len(relevant_radii) == 2:
        raise ValueError("This script can only generate an initial configuration for two types of particles.")

    # Sort entries in dictionary by value in descending order.
    radii = sorted(relevant_radii.items(), key=lambda r: r[1], reverse=True)

    lattice_spacing = 2.0 * radii[0][1] * configuration_parameters.lattice_spacing_factor
    orbit_distance = ((radii[0][1] + radii[1][1] + 2.0 * run_parameters.brush_length)
                      * configuration_parameters.orbit_factor)
    padding_distance = radii[0][1] * configuration_parameters.padding_factor
    generator = CubicLatticeWithSatellitesGenerator(
        run_parameters.initial_configuration, CubicLattice.from_string(configuration_parameters.lattice_type),
        lattice_spacing, configuration_parameters.lattice_repeats, orbit_distance, padding_distance,
        configuration_parameters.satellites_per_center, radii[0][0], radii[1][0])
    generator.write_positions()

    run_parameters.check_types_of_initial_configuration()


if __name__ == '__main__':
    main()
