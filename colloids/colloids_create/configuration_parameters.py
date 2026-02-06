from dataclasses import dataclass, field
from typing import Any, Optional, Union
import inspect
import warnings
from ase.io.lammpsdata import read_lammps_data
import numpy as np
from openmm import unit
from colloids.abstracts import Parameters
from colloids.units import electric_potential_unit, length_unit, mass_unit
import colloids.colloids_create.initial_modifiers as initial_modifiers
import colloids.colloids_create.final_modifiers as final_modifiers


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

    Any bonds in the cluster definition in the lammps-data file are added as constraints, with the constraint distance
    equal to the current bond length in the cluster definition. The bond lengths are not modified during the simulation.

    This dataclass assumes that the style of units in the lammps-data file is "nano" (see
    https://docs.lammps.org/units.html), that is, positions are in nanometers.

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

    If working with a lattice structure that you would like to expand into a supercell and/or resize to match the colloid
    radii, instead of supplying a lammps-data file, you can specify a .cif file and use the lattice builder functionality
    to obtain a lammps-data file that is then used to generate the base configuration. In this case, you must specify parameters
    to direct the scaling and resizing of the radii as well as the polymer brush length (which should match the brush
    length in the pair potential parameters, if using). 

    After the base configuration has been created, it can be modified by applying a series of configuration modifiers.
    These modifiers can modify the positions of the colloids, add or remove colloids, or modify other properties of the
    colloids. The modifiers are applied in two stages: initial modifiers (such as adding a substrate at the bottom of
    the simulation box) are applied before setting the particle properties (diameter, charge, mass), and final modifiers
    (such as including a seed of colloids from a gsd file while removing overlapping particles from the base
    configuration) are applied after setting the particle properties.

    :param cif: 
        The .cif file that specifies the desired lattice structure of the output configuration files.
        Defaults to None.
    :type cif: Optional[str]
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
    :param lattice_vector_scaling_factor:
        If using the lattice builder functionality, a scaling matrix for transforming the lattice vectors.
        If only a single integer is given, the same scale factor is used in all directions.
        Every scale factor in the matrix should be positive.
        Defaults to None. 
    :type lattice_vector_scaling_factor: Optional[Union[int, list[int]]]
    :param brush_length:
        If using the lattice builder functionality, the thickness of the brush in the Alexander-de Gennes polymer brush model 
        [i.e., L in eq. (1)].
        The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to None.
    :type brush_length: Optional[unit.Quantity]
    :param lattice_scale_factor:
        If using the lattice builder functionality, scale-up increment factor.
        The lattice scale factor must be greater than zero.
        Defaults to None.
    :type lattice_scale_factor: Optional[float]
    :param lattice_scale_start:
        If using the lattice builder functionality, starting scale factor.
        The lattice scale start factor must be greater than zero.
        Defaults to None.
    :type lattice_scale_start: Optional[float]
    :param radii_padding_factor: 
        If using the lattice builder functionality, extra gap added to radii.
        The unit of the radii padding factor should be compatible with nanometers and the value must be greater than zero.
        Defaults to None.
    :type radii_padding_factor: Optional[unit.Quantity]
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
    :param initial_modifiers:
        List of modifier class names to apply before setting particle properties (diameter, charge, mass).
        These modifiers run early in the configuration generation process.
        Possible choices can be found in the colloids_create.initial_modifiers module.
        If initial modifiers are specified, their parameters must be specified as well in the
        initial_modifiers_parameters list.
        Defaults to None.
    :type initial_modifiers: Optional[list[str]]
    :param initial_modifiers_parameters:
        List of dictionaries containing parameters for each initial modifier.
        Each dictionary is passed to the corresponding modifier's __init__ method.
        The list must have the same length as initial_modifiers.
        Defaults to None.
    :type initial_modifiers_parameters: Optional[list[dict[str, Any]]]
    :param final_modifiers:
        List of modifier class names to apply after setting particle properties (diameter, charge, mass).
        These modifiers run at the end of the configuration generation process.
        Possible choices can be found in the colloids_create.final_modifiers module.
        If final modifiers are specified, their parameters must be specified as well.
        Defaults to None.
    :type final_modifiers: Optional[list[str]]
    :param final_modifiers_parameters:
        List of dictionaries containing parameters for each final modifier.
        Each dictionary is passed to the corresponding modifier's __init__ method.
        The list must have the same length as final_modifiers.
        Defaults to None.
    :type final_modifiers_parameters: Optional[list[dict[str, Any]]]

    :raises TypeError:
        If the lattice repeats are not an integer or a list of three integers.
        If the masses, radii, or surface potentials do not have the correct units.
        If the masses, radii, or surface potentials dictionaries do not have strings as keys.
    :raises ValueError:
        If the cluster specification file does not end in ".lmp."
        If the number of lattice repeats is not positive.
        If the (cluster) padding factor is not greater than zero.
        If the masses are not greater than or equal to zero.
        If the radii are not greater than zero.
        If a type of the lammps-data file is not in the masses, radii, or surface potentials dictionaries.
        If an initial or final modifier is not found in the available modifiers.
        If initial_modifiers is specified but initial_modifiers_parameters is not, or vice versa.
        If final_modifiers is specified but final_modifiers_parameters is not, or vice versa.
        If the number of (initial or final) modifiers does not match the number of parameter dictionaries.
    """
    cluster_specifications: list[str] = field(default_factory=lambda: ["cluster.lmp"])
    cif: Optional[str] = None
    lattice_repeats: Union[int, list[int]] = 8
    cluster_relative_weights: list[float] = field(default_factory=lambda: [1.0])
    lattice_scale_factor: Optional[float] = None
    lattice_scale_start: Optional[float] = None
    lattice_vector_scaling_matrix: Optional[Union[int, list[int]]] = None
    cluster_padding_factor: float = 1.0
    padding_factor: float = 1.0
    radii_padding_factor: Optional[unit.Quantity] = None 
    random_rotation: bool = False
    masses: dict[str, unit.Quantity] = field(default_factory=lambda: {"1": 1.0 * mass_unit,
                                                                      "2": (95.0 / 105.0) ** 3 * mass_unit})
    radii: dict[str, unit.Quantity] = field(default_factory=lambda: {"1": 105.0 * length_unit,
                                                                     "2": 95.0 * length_unit})
    surface_potentials: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"1": 44.0 * electric_potential_unit, "2": -54.0 * electric_potential_unit})
    brush_length: Optional[unit.Quantity] = None
    initial_modifiers: Optional[list[str]] = None
    initial_modifiers_parameters: Optional[list[dict[str, Any]]] = None
    final_modifiers: Optional[list[str]] = None
    final_modifiers_parameters: Optional[list[dict[str, Any]]] = None

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
        if self.cluster_padding_factor <= 0.0:
            raise ValueError("Cluster padding factor must be greater than zero.")
        if self.padding_factor <= 0.0:
            raise ValueError("Padding factor must be greater than zero.")
        if self.cif:
            if not self.cif.endswith(".cif"):
                raise ValueError("The cif file must have the correct .cif extension.")
            if not self.lattice_scale_start:
                raise ValueError("Lattice scale start must be specified if using lattice builder method.")
            if not self.lattice_scale_factor:
                raise ValueError("Lattice scale factor must be specified if using lattice builder method.")
            if not self.lattice_vector_scaling_matrix:
                raise ValueError("Lattice vector scaling matrix must be specified if using lattice builder method.")
            if not self.radii_padding_factor:
                raise ValueError(("Radii padding factor must be specified if using lattice builder method."))
            if not self.radii_padding_factor.unit.is_compatible(length_unit):
                raise TypeError(f"Radii padding factor must have a unit compatible with nanometers.")
            if self.radii_padding_factor <= 0.0 * length_unit:
                raise ValueError(f"Radii padding factor must be greater than zero.")
            if not self.brush_length:
                    raise ValueError("Brush length must be specified if using lattice builder method.")
            if not self.brush_length.unit.is_compatible(length_unit):
                raise TypeError(f"Brush length must have a unit compatible with nanometers.")
            if self.brush_length <= 0.0 * length_unit:
                raise ValueError(f"Brush length must be greater than zero.")


        # We assume that the lammps-data file uses "nano" units where distances are measured in nanometers.
        # However, ase would transform the distances in the lammps-data file to Angstroms by multiplying them by 10 if
        # we specify units="nano". For units="metal", the ase distances are equal to the distances in the lammps-data
        # file. We then just pretend that the distances are in nanometers.
        found_types = set()
        cell = None
        for cluster_specification in self.cluster_specifications:
            try:
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
            except FileNotFoundError: #if cluster.lmp file is being generated via lattice builder
                pass

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

        if self.initial_modifiers is not None:
            if self.initial_modifiers_parameters is None:
                raise ValueError("The initial_modifiers_parameters must be specified if initial_modifiers is specified.")
            if len(self.initial_modifiers) != len(self.initial_modifiers_parameters):
                raise ValueError("The number of initial modifiers must match the number of initial modifier "
                                 "parameter dictionaries.")
            possible_modifiers = [name for name, obj in inspect.getmembers(initial_modifiers, inspect.isclass)
                                  if issubclass(obj, initial_modifiers.InitialModifier)
                                  and obj is not initial_modifiers.InitialModifier]
            for initial_modifier in self.initial_modifiers:
                if initial_modifier not in possible_modifiers:
                    raise ValueError(f"Initial modifier {initial_modifier} not found. Possible choices are: "
                                     f"{', '.join(possible_modifiers)}.")
        else:
            if self.initial_modifiers_parameters is not None:
                raise ValueError("The initial_modifiers_parameters must not be specified if initial_modifiers is not "
                                 "specified.")

        if self.final_modifiers is not None:
            if self.final_modifiers_parameters is None:
                raise ValueError("The final_modifiers_parameters must be specified if final_modifiers is specified.")
            if len(self.final_modifiers) != len(self.final_modifiers_parameters):
                raise ValueError("The number of final modifiers must match the number of final modifier "
                                 "parameter dictionaries.")
            possible_modifiers = [name for name, obj in inspect.getmembers(final_modifiers, inspect.isclass)
                                  if issubclass(obj, final_modifiers.FinalModifier)
                                  and obj is not final_modifiers.FinalModifier]
            for final_modifier in self.final_modifiers:
                if final_modifier not in possible_modifiers:
                    raise ValueError(f"Final modifier {final_modifier} not found. Possible choices are: "
                                     f"{', '.join(possible_modifiers)}.")
        else:
            if self.final_modifiers_parameters is not None:
                raise ValueError("The final_modifiers_parameters must not be specified if final_modifiers is not "
                                 "specified.")
