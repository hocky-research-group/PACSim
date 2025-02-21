import argparse
import gsd.hoomd
import numpy as np
from openmm import unit
from colloids.run_parameters import RunParameters
from colloids.colloids_create.configuration_parameters import ConfigurationParameters
from colloids.colloids_create.cluster_generator import (CubicLattice,
                                                                              ClusterGenerator)
from colloids.colloids_create.substrate_modifier import SubstrateModifier


class ExampleAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        default_parameters = ConfigurationParameters()
        default_parameters.to_yaml("example_configuration.yaml")
        parser.exit()


def _check_frame_changes(frame: gsd.hoomd.Frame, accountable_name: str) -> None:
    """
    Check if the accountable class did not populate the frame.particles.type_shapes, frame.particles.diameter,
    frame.particles.charge, and frame.particles.mass attributes of the given frame.

    :param frame:
        The frame.
    :type frame: gsd.hoomd.Frame
    :param accountable_name:
        The name of the accountable class.
    :type accountable_name: str

    :raises ValueError:
        If the accountable class populated the frame.particles.type_shapes attribute.
        If the accountable class populated the frame.particles.diameter attribute.
        If the accountable class populated the frame.particles.charge attribute.
        If the accountable class populated the frame.particles.mass attribute.
    """
    if frame.particles.type_shapes is not None:
        raise ValueError(f"Class {accountable_name} must not populate the frame.particles.type_shapes attribute.")
    if frame.particles.diameter is not None:
        raise ValueError(f"Class {accountable_name} must not populate the frame.particles.diameter attribute.")
    if frame.particles.charge is not None:
        raise ValueError(f"Class {accountable_name} must not populate the frame.particles.charge attribute.")
    if frame.particles.mass is not None:
        raise ValueError(f"Class {accountable_name} must not populate the frame.particles.mass attribute.")


def check_frame_types(frame: gsd.hoomd.Frame, masses: dict[str, unit.Quantity], radii: dict[str, unit.Quantity],
                      surface_potentials: dict[str, unit.Quantity]) -> None:
    """
    Check if the frame contains all types of particles that are in the masses, radii, and surface potentials
    dictionaries and vice versa.

    :param frame:
        The frame to check.
    :type frame: gsd.hoomd.Frame
    :param masses:
        The masses dictionary with the particle types as keys and the masses as values.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii dictionary with the particle types as keys and the radii as values.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials dictionary with the particle types as keys and the surface potentials as values.
    :type surface_potentials: dict[str, unit.Quantity]

    :raises ValueError:
        If a type in the frame is not in the masses dictionary.
        If a type in the frame is not in the radii dictionary.
        If a type in the frame is not in the surface potentials dictionary.
    """
    for t in frame.particles.types:
        if t not in masses:
            raise ValueError(f"Type {t} of the frame is not in the masses dictionary.")
        if t not in radii:
            raise ValueError(f"Type {t} of the frame is not in the radii dictionary.")
        if t not in surface_potentials:
            raise ValueError(f"Type {t} of the frame is not in the surface potentials dictionary.")


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

    if not run_parameters.initial_configuration.endswith(".gsd"):
        raise ValueError("The initial configuration must have the .gsd extension.")

    generator = ClusterGenerator(configuration_parameters)

    frame, constraints = generator.generate_configuration()
    _check_frame_changes(frame, generator.__class__.__name__)

    if configuration_parameters.use_substrate:
        substrate_modifier = SubstrateModifier(configuration_parameters)
        substrate_modifier.modify_configuration(frame)
        _check_frame_changes(frame, substrate_modifier.__class__.__name__)

    # Check if the frame has the necessary attributes.
    check_frame_types(frame, configuration_parameters.masses, configuration_parameters.radii,
                      configuration_parameters.surface_potentials)
    
    frame.particles.mass = np.array([configuration_parameters.masses[frame.particles.types[i]].value_in_unit(unit.amu)
                                     for i in frame.particles.typeid], dtype=np.float32)
    millivolt = unit.milli * unit.volt
    frame.particles.charge = np.array(
        [configuration_parameters.surface_potentials[frame.particles.types[i]].value_in_unit(millivolt)
         for i in frame.particles.typeid], dtype=np.float32)
    nanometer = unit.nano * unit.meter
    frame.particles.diameter = np.array(
        [2.0 * configuration_parameters.radii[frame.particles.types[i]].value_in_unit(nanometer)
         for i in frame.particles.typeid], dtype=np.float32)
    
    # Interpret the constraints as a list of pairs of particle indices and distances
    constraints_values = []
    constraints_groups = []

    for i, constraint_data in enumerate(constraints):
        constraint_pairs = constraint_data[0]
        constraint_distances = constraint_data[1]

        for ind in range(len(constraint_pairs)):
            pair = constraint_pairs[ind]
            constraint_distance = constraint_distances[ind]

            if pair > i:
                constraints_values.append(constraint_distance)
                constraints_groups.append((i, pair))

    # Write the constraints to the gsd 
    frame.constraints.N = len(constraints_values)
    frame.constraints.value = np.array(constraints_values, dtype=np.float32)
    frame.constraints.group = np.array(constraints_groups, dtype=np.uint32)

    with gsd.hoomd.open(name=run_parameters.initial_configuration, mode="w") as f:
        f.append(frame)


if __name__ == '__main__':
    main()
