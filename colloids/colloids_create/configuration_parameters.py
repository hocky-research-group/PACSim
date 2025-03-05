from dataclasses import dataclass, field
from typing import Optional, Union
import warnings
from ase.io.lammpsdata import read_lammps_data
import numpy as np
from openmm import unit
from colloids.abstracts import Parameters
from colloids.units import electric_potential_unit, length_unit, mass_unit


@dataclass(order=True, frozen=True)
class ConfigurationParameters(Parameters):
    """
    Data class for the parameters of the colloids configuration to be created for an OpenMM simulation.

    This dataclass can be written to and read from a yaml file. The yaml file contains the parameters as key-value
    pairs. Any OpenMM quantities are converted to Quantity objects that can be represented in a readable way in the
    yaml file.

    The base configuration is constructed from a single cluster of colloids. The cluster is defined in a lammps-data
    file together with lattice vectors. The cluster is then repeated in all three directions of the lattice
    vectors to create the base configuration. Every replica of the cluster can optionally be randomly rotated.

    This dataclass assumes that the style of units in the lammps-data file is "nano" (see
    https://docs.lammps.org/units.html).

    Any bonds in the cluster definition in the lammps-data file are added as constraints, with the constraint distance
    equal to the current bond length in the cluster definition. The bond lengths are not modified during the simulation.

    In the lammps-data file, only the lattice vectors, the positions of the colloids in the Atoms section, and the bonds
    in the Bonds section are used. All other sections and information are ignored. In particular, the masses, radii, and
    surface potentials of the different types of colloidal particles appearing in the lammps-data file should be
    specified in the masses, radii, and surface_potentials dictionaries of this data class (and, for instance, not in
    the Masses section of the lammps-data file).

    After the base configuration has been created, it can be modified by adding a substrate at the bottom of the
    simulation box.

    :param cluster_specification:
        The filename of the cluster definition in lammps-data format.
        Defaults to cluster.lmp.
    :type cluster_specification: str
    :param lattice_repeats:
        The number of repeats of the lattice in the three directions of the lattice vectors of the cluster.
        If only a single integer is given, the same number of repeats is used in all directions.
        Every repeat must be positive.
        Defaults to 8.
    :type lattice_repeats: Union[int, list[int]]
    :param random_rotation:
        A boolean that indicates whether every replica of the cluster should be randomly rotated.
        Defaults to False.
    :type random_rotation: bool
    :param padding_distance:
        The additional distance that is added on the outside as a padding between the clusters and the walls.
        Must be non-negative.
        Defaults to 0 nm.
    :type padding_distance: unit.Quantity
    :param masses:
        The masses of the different types of colloidal particles that appear in the cluster definition.
        The keys of the dictionary are the types of the colloidal particles and the values are the masses.
        The unit of the masses must be compatible with atomic mass units and the values must be greater than zero,
        except for immobile particles (as the substrate), which should have a mass of zero.
        Defaults to {"P": 1.0 * amu, "N": (95.0 / 105.0) ** 3 * amu}.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii of the different types of colloidal particles that appear in the initial configuration file.
        The keys of the dictionary are the types of the colloidal particles and the values are the radii.
        The unit of the radii must be compatible with nanometers and the values must be greater than zero.
        Defaults to {"P": 105.0 * nanometer, "N": 95.0 * nanometer}.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials of the different types of colloidal particles that appear in the initial configuration
        file.
        The keys of the dictionary are the types of the colloidal particles and the values are the surface potentials.
        The unit of the surface potentials must be compatible with millivolts.
        Defaults to {"P": 44.0 * millivolt, "N": -54.0 * millivolt}.
    :type surface_potentials: dict[str, unit.Quantity]
    :param use_substrate:
        A boolean indicating whether to place a substrate at the bottom of the simulation box.
        In a simulation with colloids_run, a substrate can only be used when all walls are active. The bottom wall is
        then replaced by the substrate. Also, a substrate can only be used with the algebraic colloid potentials.
        Defaults to False.
    :type use_substrate: bool
    :param substrate_type:
        The type of the substrate that is used at the bottom of the simulation box.
        If a substrate is used, the substrate type must not be None, and it must appear in the radii, masses, and
        surface_potentials dictionaries.
        Defaults to None.
    :type substrate_type: Optional[Union[str, int]]

    """
    cluster_specification: str = "cluster.lmp"
    lattice_repeats: Union[int, list[int]] = 8
    padding_distance: unit.Quantity = field(default_factory=lambda: 0.0 * length_unit)
    random_rotation: bool = False
    masses: dict[str, unit.Quantity] = field(default_factory=lambda: {"P": 1.0 * mass_unit,
                                                                      "N": (95.0 / 105.0) ** 3 * mass_unit})
    radii: dict[str, unit.Quantity] = field(default_factory=lambda: {"P": 105.0 * length_unit,
                                                                     "N": 95.0 * length_unit})
    surface_potentials: dict[str, unit.Quantity] = field(default_factory=lambda: {"P": 44.0 * electric_potential_unit,
                                                                                  "N": -54.0 * electric_potential_unit})
    use_substrate: bool = False
    substrate_type: Optional[str] = None

    def __post_init__(self):
        """Post-initialization method for the ConfigurationParameters class."""
        atoms = read_lammps_data(units="nano")
        types = set(str(atom.number) for atom in atoms)
        for t in types:
            if t not in self.masses:
                raise ValueError(f"Type {t} of the atoms in the lammps-data file is not in masses dictionary.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the atoms in the lammps-data file is not in radii dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the atoms in the lammps-data file is not in surface potentials "
                                 f"dictionary.")
        # If no cell is set in the lammps-data file, the lattice vectors are set to the identity matrix.
        if np.equal(atoms.get_cell(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).all():
            warnings.warn("The lattice vectors of the cluster are probably not set in the lammps-data file."
                          "The identity matrix is used as lattice vectors.")

        if isinstance(self.lattice_repeats, int):
            if self.lattice_repeats <= 0:
                raise ValueError("The number of lattice repeats must be positive.")
        else:
            if not (isinstance(self.lattice_repeats, list)
                    and all(isinstance(repeat, int) for repeat in self.lattice_repeats)
                    and len(self.lattice_repeats) == 3):
                raise TypeError("The lattice repeats must be an integer or a list of three integers.")
            if not all(repeat > 0 for repeat in self.lattice_repeats):
                raise ValueError("All lattice repeats must be positive.")

        if not self.padding_distance.unit.is_compatible(length_unit):
            raise TypeError("Padding distance must have a unit compatible with nanometers.")
        if self.padding_distance < 0.0 * length_unit:
            raise ValueError("Padding distance must be equal to or greater than zero.")

        for t in self.masses:
            if not self.masses[t].unit.is_compatible(mass_unit):
                raise TypeError(f"Mass of type {t} must have a unit compatible with atomic mass units.")
            if self.masses[t] < 0.0 * mass_unit:
                raise ValueError(f"Mass of type {t} must be greater than zero.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the masses dictionary is not in radii dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the masses dictionary is not in surface potentials dictionary.")
            if t not in types and t != self.substrate_type:
                raise ValueError(f"Non-substrate type {t} of the masses dictionary is not in the lammps-data file.")
        for t in self.radii:
            if not self.radii[t].unit.is_compatible(length_unit):
                raise TypeError(f"Radius of type {t} must have a unit compatible with nanometers.")
            if self.radii[t] <= 0.0 * length_unit:
                raise ValueError(f"Radius of type {t} must be greater than zero.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the radii dictionary is not in masses dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the radii dictionary is not in surface potentials dictionary.")
            if t not in types and t != self.substrate_type:
                raise ValueError(f"Non-substrate type {t} of the radii dictionary is not in the lammps-data file.")
        for t in self.surface_potentials:
            if not self.surface_potentials[t].unit.is_compatible(electric_potential_unit):
                raise TypeError(f"Surface potential of type {t} must have a unit compatible with millivolts.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in masses dictionary.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in radii dictionary.")
            if t not in types and t != self.substrate_type:
                raise ValueError(f"Non-substrate type {t} of the surface potentials dictionary is not in the "
                                 f"lammps-data file.")

        if self.use_substrate:
            if self.substrate_type is None:
                raise ValueError("The substrate type must be specified if a substrate is used.")
            if self.substrate_type not in self.radii:
                raise ValueError("The substrate type must be in the radii dictionary.")
            if self.substrate_type not in self.masses:
                raise ValueError("The substrate type must be in the masses dictionary.")
            if self.substrate_type not in self.surface_potentials:
                raise ValueError("The substrate type must be in the surface potentials dictionary.")
            if self.masses[self.substrate_type] != 0.0 * mass_unit:
                warnings.warn("The mass of the substrate type is not zero. Substrate will move during the simulation.")
        else:
            if self.substrate_type is not None:
                raise ValueError("The substrate type must not be specified if a substrate is not used.")
