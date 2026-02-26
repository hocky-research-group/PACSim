from typing import Sequence, Union
from gsd.hoomd import Frame
import numpy as np
from openmm import unit
from pymatgen.core import Element
from pymatgen.io.cif import CifParser
from scipy.spatial import distance_matrix
from .abstracts import ConfigurationGenerator


class LatticeBuilder(ConfigurationGenerator):
    """
    Generator for an initial configuration in a gsd.hoomd.Frame instance for a colloid simulation based on a
    crystal lattice structure defined in a CIF file.

    The lattice structure is loaded from a CIF file and expanded into a supercell. The supercell is then uniformly
    scaled so that no particles overlap, accounting for colloid radii, brush length, and an extra radii padding gap.
    The optimal scale factor is computed directly as the maximum ratio of the sum of effective radii to the distance
    for all particle pairs, using a small (3, 3, 3) test supercell for efficiency. This scale factor is then applied to
    the full supercell defined by the lattice repeats.

    The scaled supercell is centered at the origin and embedded in a cubic orthorhombic simulation box. The box side
    length is chosen so that the outermost particle (including its effective radius) plus a lattice padding gap fits
    within the box in every direction.

    :param masses:
        The masses dictionary with the particle types as keys and the masses as values.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii dictionary with the particle types as keys and the radii as values.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials dictionary with the particle types as keys and the surface potentials as values.
    :type surface_potentials: dict[str, unit.Quantity]
    :param lattice_specification:
        The .cif file that specifies the desired lattice structure.
    :type lattice_specification: str
    :param lattice_repeats:
        Number of repetitions of the unit cell in each direction to create the supercell. This can be specified as a
        single integer (if the same number of repetitions is desired in all directions) or as a sequence of three
        integers (if different numbers of repetitions are desired in different directions).
    :type lattice_repeats: Union[int, Sequence[int]]
    :param brush_length:
        The thickness of the brush in the Alexander-de Gennes polymer brush model [i.e., L in eq. (1)].
        The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
    :type brush_length: unit.Quantity
    :param radii_padding:
        Extra gap added to the effective radii when checking for overlaps.
        The unit of the radii padding should be compatible with nanometers and the value must be greater than or equal
        to zero.
    :type radii_padding: unit.Quantity
    :param lattice_padding:
        Extra gap added to the box dimensions.
        The unit of the lattice padding should be compatible with nanometers and the value must be greater than or
        equal to zero.
    :type lattice_padding: unit.Quantity

    :raises ValueError:
        If the lattice specification file is not a .cif file.
        If the CIF file does not contain exactly one structure.
        If the lattice repeats is not a positive integer or a sequence of three positive integers.
        If the brush length is not compatible with nanometers or is not greater than zero.
        If the radii padding is not compatible with nanometers or is negative.
        If the lattice padding is not compatible with nanometers or is negative.
    """

    def __init__(self, masses: dict[str, unit.Quantity], radii: dict[str, unit.Quantity],
                 surface_potentials: dict[str, unit.Quantity], lattice_specification: str,
                 lattice_repeats: Union[int, Sequence[int]], brush_length: unit.Quantity,
                 radii_padding: unit.Quantity, lattice_padding: unit.Quantity) -> None:
        """Constructor of the LatticeBuilder class."""
        super().__init__(masses=masses, radii=radii, surface_potentials=surface_potentials)
        if not lattice_specification.endswith('.cif'):
            raise ValueError("The lattice specification must be a .cif file.")
        parser = CifParser(lattice_specification, site_tolerance=0.0, frac_tolerance=0.0)
        structures = parser.parse_structures(check_occu=False, primitive=False)
        if len(structures) != 1:
            raise ValueError("The CIF file must contain exactly one structure.")
        self._structure = structures[0]
        self._radii = radii
        self._brush_length = brush_length  # TODO: SHOULD BE TAKEN FROM RUN.YAML
        self._lattice_repeats = lattice_repeats
        self._radii_padding = radii_padding
        self._lattice_padding = lattice_padding
        # Label atoms as their element symbols based on atomic number, e.g. 'Fe', 'O', etc.
        self._type_map = {atomic_number: str(Element.from_Z(atomic_number))
                          for atomic_number in np.unique(self._structure.atomic_numbers)}
        if isinstance(self._lattice_repeats, int):
            if not self._lattice_repeats > 0:
                raise ValueError("The lattice repeats must be greater than zero.")
        else:
            if not isinstance(self._lattice_repeats, Sequence) or len(self._lattice_repeats) != 3:
                raise ValueError("The lattice repeats must be either a single integer or a sequence of three integers.")
            if not all(r > 0 for r in self._lattice_repeats):
                raise ValueError("All values in the lattice repeats must be greater than zero.")
        if not self._brush_length.unit.is_compatible(unit.nanometer):
            raise TypeError("The brush length must have a unit that is compatible with nanometers.")
        if not self._brush_length.value_in_unit(unit.nanometer) > 0.0:
            raise ValueError("The brush length must have a value greater than zero.")
        if not self._radii_padding.unit.is_compatible(unit.nanometer):
            raise TypeError("The radii padding must have a unit that is compatible with nanometers.")
        if not self._radii_padding.value_in_unit(unit.nanometer) >= 0.0:
            raise ValueError("The radii padding must have a value greater than or equal to zero.")
        if not self._lattice_padding.unit.is_compatible(unit.nanometer):
            raise TypeError("The lattice padding must have a unit that is compatible with nanometers.")
        if not self._lattice_padding.value_in_unit(unit.nanometer) >= 0.0:
            raise ValueError("The lattice padding must have a value greater than or equal to zero.")

    def types(self) -> set[str]:
        """
        Return the set of particle types that will be generated by this configuration generator.

        :return:
            The set of particle types that will be generated by this configuration generator.
        :rtype: set[str]
        """
        return set(self._type_map.values())

    def generate_configuration(self) -> Frame:
        """
        Generate the initial positions of the colloids in a gsd.hoomd.Frame instance.

        :return:
            The initial configuration of the colloids.
        :rtype: gsd.hoomd.Frame
        """
        # Find optimal scale factor using a small test supercell to speed up the search.
        small_supercell = self._structure.make_supercell((3, 3, 3), in_place=False, to_unit_cell=True)
        dists = distance_matrix(small_supercell.cart_coords, small_supercell.cart_coords)
        effective_radii = [self._radii[self._type_map[atomic_number]].value_in_unit(unit.nanometer)
                           + self._brush_length.value_in_unit(unit.nanometer)
                           + self._radii_padding.value_in_unit(unit.nanometer)
                           for atomic_number in small_supercell.atomic_numbers]
        required_scale_factor = 0.0
        for i in range(len(effective_radii)):
            for j in range(i + 1, len(effective_radii)):
                required_scale_factor = max(required_scale_factor,
                                            (effective_radii[i] + effective_radii[j]) / dists[i, j])
        # TODO: CAN I ASSERT THAT THIS IS CORRECT?

        # Apply scale factor to the full supercell.
        structure_full = self._structure.make_supercell(self._lattice_repeats, in_place=False)
        positions = structure_full.cart_coords * required_scale_factor

        # Center at origin.
        positions -= positions.mean(axis=0)
        types = [self._type_map[atomic_number] for atomic_number in structure_full.atomic_numbers]
        effective_radii = [self._radii[t].value_in_unit(unit.nanometer)
                           + self._brush_length.value_in_unit(unit.nanometer)
                           + self._radii_padding.value_in_unit(unit.nanometer)
                           for t in types]

        # Embed in cubic box with padding.
        box_length = 2.0 * (np.max(np.abs(positions)) + np.max(effective_radii)
                            + self._lattice_padding.value_in_unit(unit.nanometer))

        # --- Build the Frame ---
        frame = Frame()
        frame.particles.N = len(positions)
        frame.particles.types = sorted(self.types())
        frame.particles.typeid = np.array(
            [frame.particles.types.index(t) for t in types], dtype=np.uint32)
        frame.particles.position = np.array(positions, dtype=np.float32)
        frame.configuration.box = np.array([box_length, box_length, box_length, 0.0, 0.0, 0.0], dtype=np.float32)

        return frame
