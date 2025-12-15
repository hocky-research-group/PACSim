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

    The base configuration is constructed from several clusters of colloids. Each cluster is defined in a lammps-data
    file together with cell vectors. Every cluster of colloids is assumed to have the same cell vectors. To generate
    the initial configuration, the clusters are first centered. Then, the shared cell vectors of the clusters are
    repeated in all three directions. Every replica of the cell is then filled with a randomly selected cluster from the
    list of clusters. The clusters are selected based on their relative weights. Every cluster can optionally be
    randomly rotated.

    All colloid positions in the centered clusters must lie in the unit cell defined by the lattice vectors.

    To space out the clusters, one can increase a cluster padding factor that scales the lattice vectors. This will also
    scale the box size. Additionally, one can increase a padding factor that scales just the overall box size and thus
    increases the distance between the outwards facing colloids and the walls. To make the simulation box smaller, use
    a padding factor less than 1.

    This dataclass assumes that the style of units in the lammps-data file is "nano" (see
    https://docs.lammps.org/units.html), that is, positions are in nanometers.

    Any bonds in the cluster definition in the lammps-data file are added as constraints, with the constraint distance
    equal to the current bond length in the cluster definition. The bond lengths are not modified during the simulation.

    In the lammps-data file, only the lattice vectors, the positions of the colloids in the Atoms section, and the bonds
    in the Bonds section are used. All other sections and information are ignored. In particular, the masses, radii, and
    surface potentials of the different types of colloidal particles appearing in the lammps-data file should be
    specified in the masses, radii, and surface_potentials dictionaries in the yaml file of this data class (and, for
    instance, not in the Masses section of the lammps-data file).

    See https://docs.lammps.org/Howto_triclinic.html for more information about the lattice vectors in the lammps-data
    files.

    In the Atoms section of the lammps-data files, the different columns from left to right are as follows:
        Atom Index (should go from 1 to number of atoms).
        Molecule-ID (ignored)
        Atom type (these are the types appearing as keys in the mass/diameter/surface potential dictionaries in the yaml file)
        Charge (ignored)
        x position
        y position
        z position

    In the Bonds section of the lammps-data files, the different columns from left to right are as follows:
        Bond index (should go from 1 to number of bonds)
        Bond ID (ignored)
        Index of first atom involved in the bond.
        Index of second atom involved in the bond.

    After the base configuration has been created, it can be modified by adding a substrate at the bottom of the
    simulation box.

    Furthermore, it is possible to include a seed of colloids from a gsd file. If a seed file is specified, the seed is
    placed in the simulation box without transformation, overwriting any colloids that overlap with the seed. Overlaps
    are determined based on a specified overlap distance. Particles with a surface-to-surface distance smaller than the
    overlap distance are considered overlapping.

    :param cluster_specifications:
        The filenames of the cluster definitions in lammps-data format.
        Defaults to [cluster.lmp].
    :type cluster_specifications: list[str]
    :param cluster_relative_weights:
        The relative weights of the clusters. The weights are used to randomly select a cluster from the list of
        clusters when generating the initial configuration.
        The weights should be positive.
        Defaults to [1.0].
    :type cluster_relative_weights: Sequence[float]
    :param lattice_repeats:
        The number of repeats of the lattice in the three directions of the lattice vectors of the cluster.
        If only a single integer is given, the same number of repeats is used in all directions.
        Every repeat must be positive.
        Defaults to 8.
    :type lattice_repeats: Union[int, list[int]]
    :param cluster_padding_factor:
        The factor by which the lattice vectors of every replicated cluster are scaled to space out the clusters.
        The cluster padding factor must be greater than zero.
        Defaults to 1.0.
    :type cluster_padding_factor: float
    :param padding_factor:
        The factor by which the overall lattice vectors are scaled to increase the distance between the outwards facing
        colloids and the walls.
        The padding factor must be greater than zero.
        Defaults to 1.0.
    :type padding_factor: float
    :param random_rotation:
        A boolean that indicates whether every replica of the cluster should be randomly rotated.
        Defaults to False.
    :type random_rotation: bool
    :param masses:
        The masses of the different types of colloidal particles that appear in the cluster definition.
        The keys of the dictionary are the types of the colloidal particles and the values are the masses.
        The unit of the masses must be compatible with atomic mass units and the values must be greater than zero,
        except for immobile particles (as the substrate), which should have a mass of zero.
        Defaults to {"1": 1.0 * amu, "2": (95.0 / 105.0) ** 3 * amu}.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii of the different types of colloidal particles that appear in the initial configuration file.
        The keys of the dictionary are the types of the colloidal particles and the values are the radii.
        The unit of the radii must be compatible with nanometers and the values must be greater than zero.
        Defaults to {"1": 105.0 * nanometer, "2": 95.0 * nanometer}.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials of the different types of colloidal particles that appear in the initial configuration
        file.
        The keys of the dictionary are the types of the colloidal particles and the values are the surface potentials.
        The unit of the surface potentials must be compatible with millivolts.
        Defaults to {"1": 44.0 * millivolt, "2": -54.0 * millivolt}.
    :type surface_potentials: dict[str, unit.Quantity]
    :param seed_filename:
        The gsd file with a seed of colloids.
        Defaults to None.
    :type seed_filename: Optional[str]
    :param seed_frame_index:
        The frame index in the seed file to use. Negative indices are supported (e.g., -1 for the last frame).
        Only used if seed_file is specified.
        Defaults to -1.
    :type seed_frame_index: int
    :param seed_overlap_distance:
        The overlap distance for seeding. Particles in the base frame that overlap with the seed are removed.
        Must have units compatible with nanometers and be non-negative.
        Defaults to 0.0 * length_unit.
    :type seed_overlap_distance: unit.Quantity
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

    :raises TypeError:
        If the lattice repeats are not an integer or a list of three integers.
        If the masses, radii, or surface potentials do not have the correct units.
        If the masses, radii, or surface potentials dictionaries do not have strings as keys.
        If the substrate type is not a string.
    :raises ValueError:
        If the cluster specification file does not end in ".lmp."
        If the number of lattice repeats is not positive.
        If the (cluster) padding factor is not greater than zero.
        If the masses are not greater than or equal to zero.
        If the radii are not greater than zero.
        If the substrate type is not specified when a substrate is used or vice versa.
        If a non-substrate type of the masses, radii, or surface potentials dictionaries is not in the lammps-data file.
        If a type of the lammps-data file is not in the masses, radii, or surface potentials dictionaries.
        If the substrate type is not in the radii, masses, or surface potentials dictionaries.
        If the mass of the substrate type is not zero.
    """
    cluster_specifications: list[str] = field(default_factory=lambda: ["cluster.lmp"])
    cluster_relative_weights: list[float] = field(default_factory=lambda: [1.0])
    lattice_repeats: Union[int, list[int]] = 8
    cluster_padding_factor: float = 1.0
    padding_factor: float = 1.0
    random_rotation: bool = False
    masses: dict[str, unit.Quantity] = field(default_factory=lambda: {"1": 1.0 * mass_unit,
                                                                      "2": (95.0 / 105.0) ** 3 * mass_unit})
    radii: dict[str, unit.Quantity] = field(default_factory=lambda: {"1": 105.0 * length_unit,
                                                                     "2": 95.0 * length_unit})
    surface_potentials: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"1": 44.0 * electric_potential_unit, "2": -54.0 * electric_potential_unit})
    seed_filename: Optional[str] = None
    seed_frame_index: Optional[int] = None
    seed_overlap_distance: Optional[unit.Quantity] = None
    use_substrate: bool = False
    substrate_type: Optional[str] = None

    def __post_init__(self):
        """Post-initialization method for the ConfigurationParameters class."""
        if not all(cluster_specification.endswith(".lmp") for cluster_specification in self.cluster_specifications):
            raise ValueError("The cluster specification file must be of the lammps-data file format.")
        if not len(self.cluster_specifications) > 0:
            raise ValueError("At least one cluster must be provided.")
        if len(self.cluster_specifications) != len(self.cluster_relative_weights):
            raise ValueError("The number of clusters must match the number of cluster probabilities.")
        if not all(prob >= 0.0 for prob in self.cluster_relative_weights):
            raise ValueError("All cluster probabilities must be non-negative.")
        if any(prob == 0.0 for prob in self.cluster_relative_weights):
            warnings.warn("Some cluster probabilities are zero. These clusters will not be used in the initial "
                          "configuration.")
        for t in self.masses:
            if not isinstance(t, str):
                raise TypeError("The types of the masses dictionary must be strings.")
            if not self.masses[t].unit.is_compatible(mass_unit):
                raise TypeError(f"Mass of type {t} must have a unit compatible with atomic mass units.")
            if self.masses[t] < 0.0 * mass_unit:
                raise ValueError(f"Mass of type {t} must be greater than zero.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the masses dictionary is not in radii dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the masses dictionary is not in surface potentials dictionary.")
        for t in self.radii:
            if not isinstance(t, str):
                raise TypeError("The types of the radii dictionary must be strings.")
            if not self.radii[t].unit.is_compatible(length_unit):
                raise TypeError(f"Radius of type {t} must have a unit compatible with nanometers.")
            if self.radii[t] <= 0.0 * length_unit:
                raise ValueError(f"Radius of type {t} must be greater than zero.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the radii dictionary is not in masses dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the radii dictionary is not in surface potentials dictionary.")
        for t in self.surface_potentials:
            if not isinstance(t, str):
                raise TypeError("The types of the surface potentials dictionary must be strings.")
            if not self.surface_potentials[t].unit.is_compatible(electric_potential_unit):
                raise TypeError(f"Surface potential of type {t} must have a unit compatible with millivolts.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in masses dictionary.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in radii dictionary.")

        # We assume that the lammps-data file uses "nano" units where distances are measured in nanometers.
        # However, ase would transform the distances in the lammps-data file to Angstroms by multiplying them by 10 if
        # we specify units="nano". For units="metal", the ase distances are equal to the distances in the lammps-data
        # file. We then just pretend that the distances are in nanometers.
        found_types = set()
        cell = None
        for cluster_specification in self.cluster_specifications:
            atoms = read_lammps_data(cluster_specification, units="metal")
            # If no cell is set in the lammps-data file, the lattice vectors are set to the identity matrix.
            if np.equal(atoms.get_cell(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).all():
                warnings.warn("The lattice vectors of the cluster are probably not set in the lammps-data file."
                              "The identity matrix is used as lattice vectors.")
            if cell is None:
                cell = atoms.get_cell()
            else:
                if not np.allclose(atoms.get_cell(), cell):
                    raise ValueError("All clusters must have the same cell vectors.")
            types = set(str(atom.number) for atom in atoms)
            found_types.update(types)
            for t in types:
                if t not in self.masses:
                    raise ValueError(f"Type {t} of the atoms in the lammps-data file is not in masses dictionary.")
                if t not in self.radii:
                    raise ValueError(f"Type {t} of the atoms in the lammps-data file is not in radii dictionary.")
                if t not in self.surface_potentials:
                    raise ValueError(f"Type {t} of the atoms in the lammps-data file is not in surface potentials "
                                     f"dictionary.")
        for t in self.masses:
            if t not in found_types and t != self.substrate_type:
                warnings.warn(f"Non-substrate type {t} of the masses dictionary is not in the lammps-data file.")
        for t in self.radii:
            if t not in found_types and t != self.substrate_type:
                warnings.warn(f"Non-substrate type {t} of the radii dictionary is not in the lammps-data file.")
        for t in self.surface_potentials:
            if t not in found_types and t != self.substrate_type:
                warnings.warn(f"Non-substrate type {t} of the surface potentials dictionary is not in the lammps-data "
                              f"file.")

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

        if self.cluster_padding_factor <= 0.0:
            raise ValueError("Cluster padding factor must be greater than zero.")
        if self.padding_factor <= 0.0:
            raise ValueError("Padding factor must be greater than zero.")

        if self.seed_filename is not None:
            if not self.seed_filename.endswith(".gsd"):
                raise ValueError("The seed file must have the .gsd extension.")
            if self.seed_frame_index is None:
                raise ValueError("The seed frame index must be specified if a seed file is set.")
            if self.seed_overlap_distance is None:
                raise ValueError("The seed overlap distance must be specified if a seed file is set.")
            if not self.seed_overlap_distance.unit.is_compatible(length_unit):
                raise TypeError("The seed overlap distance must have a unit compatible with nanometers.")
            if self.seed_overlap_distance < 0.0 * length_unit:
                raise ValueError("The seed overlap distance must be non-negative.")
        else:
            if self.seed_frame_index is not None:
                raise ValueError("The seed frame index must not be specified if no seed file is set.")
            if self.seed_overlap_distance is not None:
                raise ValueError("The seed overlap distance must not be specified if no seed file is set.")

        if self.use_substrate:
            if self.substrate_type is None:
                raise ValueError("The substrate type must be specified if a substrate is used.")
            if not isinstance(self.substrate_type, str):
                raise TypeError("The substrate type must be a string.")
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
