import argparse
import warnings
from ase.io.lammpsdata import read_lammps_data
import gsd.hoomd
import numpy as np
from openmm import unit
from colloids.colloids_create.configuration_parameters import ConfigurationParameters
from colloids.colloids_create.cluster_generator import ClusterGenerator
from colloids.colloids_create.substrate_modifier import SubstrateModifier
from colloids.units import electric_potential_unit, length_unit, mass_unit


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
        If a type in the masses dictionary is not in the frame.
        If a type in the radii dictionary is not in the frame.
        If a type in the surface potentials dictionary is not in the frame.
    """
    for t in frame.particles.types:
        if t not in masses:
            raise ValueError(f"Type {t} of the frame is not in the masses dictionary.")
        if t not in radii:
            raise ValueError(f"Type {t} of the frame is not in the radii dictionary.")
        if t not in surface_potentials:
            raise ValueError(f"Type {t} of the frame is not in the surface potentials dictionary.")
    for t in masses:
        if t not in frame.particles.types:
            warnings.warn(f"Type {t} of the masses dictionary is not in the frame.")
    for t in radii:
        if t not in frame.particles.types:
            warnings.warn(f"Type {t} of the radii dictionary is not in the frame.")
    for t in surface_potentials:
        if t not in frame.particles.types:
            warnings.warn(f"Type {t} of the surface potentials dictionary is not in the frame.")


def main():
    parser = argparse.ArgumentParser(description="Create an initial configuration for an OpenMM simulation of a "
                                                 "colloids system.")
    parser.add_argument("configuration_parameters", help="YAML file with configuration parameters",
                        type=str)
    parser.add_argument("save_file", help="file for the generated gsd to be saved under", type=str)
    parser.add_argument("--example", help="write an example configuration YAML file and exit",
                        action=ExampleAction)
    args = parser.parse_args()

    if not args.configuration_parameters.endswith(".yaml"):
        raise ValueError("The YAML file for the configuration parameters must have the .yaml extension.")
    if not args.save_file.endswith(".gsd"):
        raise ValueError("The initial configuration must have the .gsd extension.")

    configuration_parameters = ConfigurationParameters.from_yaml(args.configuration_parameters)

    # We assume that the lammps-data file uses "nano" units where distances are measured in nanometers.
    # However, ase would transform the distances in the lammps-data file to Angstroms by multiplying them by 10 if
    # we specify units="nano". For units="metal", the ase distances are equal to the distances in the lammps-data
    # file. We then just pretend that the distances are in nanometers.
    clusters = [read_lammps_data(spec, units="metal") for spec in configuration_parameters.cluster_specifications]
    generator = ClusterGenerator(clusters, configuration_parameters.cluster_relative_weights,
                                 configuration_parameters.lattice_repeats,
                                 configuration_parameters.cluster_padding_factor,
                                 configuration_parameters.padding_factor,
                                 configuration_parameters.random_rotation)

    frame = generator.generate_configuration()
    _check_frame_changes(frame, generator.__class__.__name__)

    if configuration_parameters.use_explicit_substrate:
        substrate_modifier = SubstrateModifier(configuration_parameters.radii[configuration_parameters.substrate_type],
                                               configuration_parameters.substrate_type)
        substrate_modifier.modify_configuration(frame)
        _check_frame_changes(frame, substrate_modifier.__class__.__name__)

    # Check if the frame has the necessary attributes.
    check_frame_types(frame, configuration_parameters.masses, configuration_parameters.radii,
                      configuration_parameters.surface_potentials)

    frame.particles.mass = np.array([configuration_parameters.masses[frame.particles.types[i]].value_in_unit(mass_unit)
                                     for i in frame.particles.typeid], dtype=np.float32)
    frame.particles.charge = np.array(
        [configuration_parameters.surface_potentials[frame.particles.types[i]].value_in_unit(electric_potential_unit)
         for i in frame.particles.typeid], dtype=np.float32)
    frame.particles.diameter = np.array(
        [2.0 * configuration_parameters.radii[frame.particles.types[i]].value_in_unit(length_unit)
         for i in frame.particles.typeid], dtype=np.float32)

    with gsd.hoomd.open(name=args.save_file, mode="w") as f:
        f.append(frame)


if __name__ == '__main__':
    main()
