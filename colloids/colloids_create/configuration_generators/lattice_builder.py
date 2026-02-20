from typing import Sequence, Union
from gsd.hoomd import Frame
import numpy as np
from openmm import unit
from pymatgen.io.cif import CifParser
from scipy.spatial import distance_matrix
from .abstracts import ConfigurationGenerator


class LatticeBuilder(ConfigurationGenerator):
    """
    Generator for an initial configuration in a gsd.hoomd.Frame instance for a colloid simulation based on a
    crystal lattice structure defined in a CIF file.

    The lattice structure is loaded from a CIF file, expanded into a supercell, and then uniformly scaled until no
    particles overlap (accounting for colloid radii, brush length, and an extra padding gap). The resulting lattice
    structure is then centered at the origin.

    A smaller test supercell is used to find the optimal scale factor before applying it to the full supercell defined
    by the lattice vector scaling matrix.

    :param lattice_specification:
        The .cif file that specifies the desired lattice structure.
    :type lattice_specification: str
    :param lattice_vector_scaling_matrix:
        A scaling matrix for transforming the lattice vectors into a supercell.
        If only a single integer is given, the same scale factor is used in all directions.
        Every scale factor in the matrix should be positive.
    :type lattice_vector_scaling_matrix: Union[int, Sequence[int]]
    :param radii:
        The radii of the different types of colloidal particles.
        The keys of the dictionary are the types of the colloidal particles and the values are the radii.
        The unit of the radii must be compatible with nanometers and the values must be greater than zero.
    :type radii: dict[str, unit.Quantity]
    :param brush_length:
        The thickness of the brush in the Alexander-de Gennes polymer brush model [i.e., L in eq. (1)].
        The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
    :type brush_length: unit.Quantity
    :param lattice_scale_factor:
        Scale-up increment factor used when searching for a non-overlapping configuration.
        The lattice scale factor must be greater than zero.
    :type lattice_scale_factor: float
    :param lattice_scale_start:
        Starting scale factor for the overlap search.
        The lattice scale start factor must be greater than zero.
    :type lattice_scale_start: float
    :param radii_padding_factor:
        Extra gap added to the effective radii when checking for overlaps.
        The unit of the radii padding factor should be compatible with nanometers and the value must be greater than zero.
    :type radii_padding_factor: unit.Quantity
    """

    def __init__(self, lattice_specification: str, lattice_vector_scaling_matrix: Union[int, Sequence[int]],
                 radii: dict[str, unit.Quantity], brush_length: unit.Quantity, lattice_scale_factor: float,
                 lattice_scale_start: float, radii_padding_factor: unit.Quantity) -> None:
        """Constructor of the LatticeBuilder class."""
        super().__init__()
        if not lattice_specification.endswith('.cif'):
            raise ValueError("The lattice specification must be a .cif file.")
        parser = CifParser(lattice_specification, site_tolerance=0.0, frac_tolerance=0.0)
        structures = parser.parse_structures(check_occu=False)
        if len(structures) != 1:
            raise ValueError("The CIF file must contain exactly one structure.")
        self._structure = structures[0]
        self._radii = radii  # TODO: This should just be passed as in final modifiers.
        self._brush_length = brush_length
        self._lattice_vector_scaling_matrix = lattice_vector_scaling_matrix
        self._lattice_scale_factor = lattice_scale_factor
        self._lattice_scale_start = lattice_scale_start
        self._radii_padding_factor = radii_padding_factor

    @staticmethod
    def _set_colloid_labels(atomic_numbers):
        """Label atoms as '1' ... 'N' based on atomic number."""
        element_list = np.unique(atomic_numbers).tolist()
        type_map = {atomic_number: element_list.index(atomic_number) + 1 for atomic_number in atomic_numbers}
        type_list = [str(type_map[atomic_number]) for atomic_number in atomic_numbers]
        return type_list

    @staticmethod
    def _check_overlap(distances, radii):
        """Check if any particles overlap."""
        n = len(radii)
        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] <= (radii[i] + radii[j]):
                    return True
        return False

    @staticmethod
    def _give_smallest_connection(distances, radii):
        """
        Return smallest effective distance (gap between particle surfaces).

        Effective distance = d - (r_i + r_j).
        """
        n = len(radii)  # TODO: I'm sure this can be sped up.
        min_dist = np.inf
        min_pair = None

        for i in range(n):
            for j in range(i + 1, n):
                eff = distances[i, j] - (radii[i] + radii[j])
                if eff < min_dist:
                    min_dist = eff
                    min_pair = (i, j)

        return min_dist, min_pair

    def generate_configuration(self, test_matrix=(3, 3, 3)) -> Frame:
        """
        Generate the initial positions of the colloids in a gsd.hoomd.Frame instance.

        Loads the CIF structure, finds the optimal lattice scale factor using a small test supercell,
        then applies it to the full supercell defined by the lattice vector scaling matrix.

        :param test_matrix:
            The supercell size used for the overlap test (should be small for speed).
            Defaults to (3, 3, 3).
        :type test_matrix: tuple[int, int, int]

        :return:
            The initial configuration of the colloids.
        :rtype: gsd.hoomd.Frame
        """
        # --- Find optimal scale factor on a small test supercell ---
        sc_test = self._structure.make_supercell(test_matrix, in_place=False)
        positions_test = sc_test.cart_coords

        types_test = self._set_colloid_labels(sc_test.atomic_numbers)

        # Effective radii: colloid radius + brush length + padding
        radii_test = [self._radii[str(t)].value_in_unit(unit.nanometer) for t in types_test]
        radii_test = np.array(radii_test) + self._brush_length.value_in_unit(unit.nanometer) + \
            self._radii_padding_factor.value_in_unit(unit.nanometer)

        dists = distance_matrix(positions_test, positions_test)
        i = 0
        scale = self._lattice_scale_start

        while self._check_overlap(dists, radii_test):  # TODO: Why not just move it?
            scale = self._lattice_scale_start + i * self._lattice_scale_factor
            positions_exp = positions_test * scale
            dists = self._calculate_distances(positions_exp)
            i += 1

        # --- Apply scale factor to the full supercell ---
        structure_full = self._structure.make_supercell(self._lattice_vector_scaling_matrix, in_place=False)
        positions = structure_full.cart_coords * scale

        # Center at origin
        positions -= positions.mean(axis=0)

        types = self._set_colloid_labels(structure_full.atomic_numbers)

        # Compute effective radii for box sizing
        radii_full = [self._radii[str(t)].value_in_unit(unit.nanometer) for t in types]
        radii_full = np.array(radii_full) + self._brush_length.value_in_unit(unit.nanometer) + \
            self._radii_padding_factor.value_in_unit(unit.nanometer)
        box = np.max(positions, axis=0) + np.max(radii_full)

        # --- Build the Frame ---
        frame = Frame()
        frame.particles.N = len(positions)
        frame.particles.types = sorted(set(types))
        frame.particles.typeid = np.array(
            [frame.particles.types.index(t) for t in types], dtype=np.uint32
        )
        frame.particles.position = np.array(positions, dtype=np.float32)
        frame.configuration.box = np.array(
            [box[0], box[1], box[2], 0.0, 0.0, 0.0], dtype=np.float32
        )

        return frame
