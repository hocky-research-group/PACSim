from math import acos, pi
from random import uniform
import re
from typing import Sequence, Union
from ase import Atoms
from gsd.hoomd import Frame
import numpy as np
from colloids.colloids_create import ConfigurationGenerator


class ClusterGenerator(ConfigurationGenerator):
    """
    Generator for an initial configuration in a gsd.hoomd.Frame instance for a colloid simulation based on a cluster
    of colloids.

    To generate the initial configuration, the given single cluster of colloids is centered and repeated in all three
    directions of its lattice vectors. Every replica of the cluster can optionally be randomly rotated.

    To space out the clusters, one can increase a cluster padding factor that scales the lattice vectors. Additionally,
    one can increase a padding factor that scales the overall box size and thus increases the distance between the
    outwards facing colloids and the walls.

    This dataclass assumes that the distances in the cluster are in units of nanometers.

    Any bonds in the cluster are added as constraints, with the constraint distance equal to the current bond length in
    the cluster definition. The bond lengths are not modified during the simulation.

    :param cluster:
        The cluster of colloids that is repeated in all three directions of its lattice vectors.
        All colloid positions in the centered cluster should lie in the unit cell defined by the lattice vectors.
    :type cluster: Atoms
    :param lattice_repeats:
        The number of repeats of the lattice in the three directions of the lattice vectors of the cluster.
        If only a single integer is given, the same number of repeats is used in all directions.
        Every repeat should be positive.
    :type lattice_repeats: Union[int, list[int]]
    :param cluster_padding_factor:
        The factor by which the lattice vectors of every replicated cluster are scaled to space out the clusters.
        The cluster padding factor should be greater than zero.
    :type cluster_padding_factor: float
    :param padding_factor:
        The factor by which the overall lattice vectors are scaled to increase the distance between the outwards facing
        colloids and the walls.
        The padding factor should be greater than zero.
    :type padding_factor: float

    :raises ValueError:
        If the cluster padding factor is not greater than zero.
        If the padding factor is not greater than zero.
    """

    def __init__(self, cluster: Atoms, lattice_repeats: Union[int, Sequence[int]], cluster_padding_factor: float,
                 padding_factor: float, random_rotation: bool) -> None:
        """Constructor of the ClusterGenerator class."""
        super().__init__()
        # The format of these arguments is already checked in configuration_parameters.py.
        self._cluster = cluster
        self._lattice_repeats = lattice_repeats
        self._cluster_padding_factor = cluster_padding_factor
        self._padding_factor = padding_factor
        self._random_rotation = random_rotation
        if self._cluster_padding_factor <= 0.0:
            raise ValueError("The cluster padding factor must be greater than zero.")
        if self._padding_factor <= 0.0:
            raise ValueError("The padding factor must be greater than zero.")

    @staticmethod
    def _extract_bonded_indices(bond_string: str) -> list[int]:
        """
        Extract the bonded indices from a bond string in the format of the ASE Atoms.arrays["bonds"] attribute.

        The bond string in ASE that was constructed from a lammps-data file looks like '1(1),2(1),3(1)',
        '2(1)', or '_'. If the string is '_', no bonds are present and an empty list is returned. Otherwise, the bonded
        indices are given by the integers in front of the parentheses (with the ignored bond type within the
        parentheses). These are extracted and returned as a list of integers.

        :param bond_string:
            The bond string in the format of the ASE Atoms.arrays["bonds"] attribute.
        :type bond_string: str

        :return:
            The bonded indices.
        :rtype: list[int]
        """
        if bond_string == "_":
            return []
        return [int(num) for num in re.findall(r"(\d+)\(", bond_string)]

    def generate_configuration(self) -> Frame:
        """
        Generate the initial positions of the colloids in a gsd.hoomd.Frame instance together with constraints.

        The generated frame should contain the following attributes:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid
        - frame.configuration.box
        - frame.constraints.N (optionally if constraints are present)
        - frame.constraints.value (optionally if constraints are present)
        - frame.constraints.group (optionally if constraints are present)

        The generated frame should not populate the following attributes:
        - frame.particles.type_shapes
        - frame.particles.diameter
        - frame.particles.charge
        - frame.particles.mass attributes

        :return:
            The initial configuration of the colloids.
        :rtype: gsd.hoomd.Frame

        :raises ValueError:
            If some positions in the centered and padded cluster are outside the unit cell.
        """
        centered_padded_cluster = self._cluster.copy()
        centered_padded_cluster.set_cell(np.array([self._cluster_padding_factor * self._cluster.cell[c]
                                                   for c in range(3)]))
        # Do not use the vacuum argument of the center method, as it does not simply add vacuum but can also decrease
        # the size of the cell to get the desired vaccum.
        centered_padded_cluster.center()
        unwrapped_positions = centered_padded_cluster.get_positions(wrap=False)
        wrapped_positions = centered_padded_cluster.get_positions(wrap=True)
        if not np.allclose(unwrapped_positions, wrapped_positions):
            raise ValueError("Some positions in the centered and padded cluster are outside of the unit cell.")
        if not self._random_rotation:
            repeated_cluster = centered_padded_cluster.repeat(self._lattice_repeats)
        else:
            # Adapted from ase's repeat function.
            if isinstance(self._lattice_repeats, int):
                r = (self._lattice_repeats, self._lattice_repeats, self._lattice_repeats)
            else:
                r = self._lattice_repeats
            repeated_cluster = centered_padded_cluster.copy()
            for name, a in repeated_cluster.arrays.items():
                repeated_cluster.arrays[name] = np.tile(a, (np.prod(r),) + (1,) * (len(a.shape) - 1))
            i0 = 0
            for r0 in range(r[0]):
                for r1 in range(r[1]):
                    for r2 in range(r[2]):
                        copied_cluster = centered_padded_cluster.copy()
                        # Generate uniformly randomized rotations.
                        # See Properties section here: https://en.wikipedia.org/wiki/Euler_angles
                        random_phi_deg = uniform(0.0, 2.0 * pi) * 180.0 / pi  # alpha on Wikipedia.
                        random_theta_deg = acos(uniform(-1.0, 1.0)) * 180.0 / pi  # beta on Wikipedia.
                        random_psi_deg = uniform(0.0, 2.0 * pi) * 180.0 / pi  # gamma on Wikipedia.
                        copied_cluster.euler_rotate(phi=random_phi_deg, theta=random_theta_deg, psi=random_psi_deg,
                                                    center="COP")
                        unwrapped_positions = copied_cluster.get_positions(wrap=False)
                        wrapped_positions = copied_cluster.get_positions(wrap=True)
                        if not np.allclose(wrapped_positions, unwrapped_positions):
                            raise ValueError("Parts of the rotated cluster lie outside the original box, increase the "
                                             "rotation padding distance to allow for rotations.")
                        repeated_cluster.arrays["positions"][i0:i0 + len(centered_padded_cluster)] = unwrapped_positions
                        repeated_cluster.arrays["positions"][i0:i0 + len(centered_padded_cluster)] += np.dot(
                            (r0, r1, r2), centered_padded_cluster.get_cell())
                        i0 += len(centered_padded_cluster)
            repeated_cluster.set_cell(np.array([r[c] * centered_padded_cluster.cell[c] for c in range(3)]))

        old_cell = repeated_cluster.get_cell()
        repeated_cluster.set_cell(np.array([self._padding_factor * old_cell[c] for c in range(3)]))
        repeated_cluster.center(about=(0.0, 0.0, 0.0))

        frame = Frame()
        frame.particles.N = len(repeated_cluster)
        frame.particles.types = tuple(set(str(atom.number) for atom in repeated_cluster))
        frame.particles.typeid = np.array([frame.particles.types.index(str(atom.number))
                                           for atom in repeated_cluster], dtype=np.uint32)
        frame.particles.position = repeated_cluster.positions.astype(np.float32)
        # See http://docs.openmm.org/7.6.0/userguide/theory/05_other_features.html
        # See https://hoomd-blue.readthedocs.io/en/v2.9.3/box.html
        frame.configuration.box = np.array(
            [repeated_cluster.cell[0][0], repeated_cluster.cell[1][1], repeated_cluster.cell[2][2],
             repeated_cluster.cell[1][0] / repeated_cluster.cell[1][1],
             repeated_cluster.cell[2][0] / repeated_cluster.cell[2][2],
             repeated_cluster.cell[2][1] / repeated_cluster.cell[2][2]], dtype=np.float32)

        if "bonds" in centered_padded_cluster.arrays:
            # Note that the replication of the bonds does not work correctly so we have to use only the original ones.
            original_bonds = centered_padded_cluster.arrays["bonds"]
            # By sorting and converting to a set we remove duplicates.
            bond_pairs = set(tuple(sorted((first_index, second_index)))
                             for first_index, bonds in enumerate(original_bonds)
                             for second_index in self._extract_bonded_indices(bonds))
            assert len(repeated_cluster) % len(centered_padded_cluster) == 0
            number_repetitions = len(repeated_cluster) // len(centered_padded_cluster)

            all_constraints = np.array(
                [[first_index + i * len(centered_padded_cluster),
                  second_index + i * len(centered_padded_cluster)]
                 for i in range(number_repetitions) for first_index, second_index in bond_pairs], dtype=np.uint32)
            all_positions = repeated_cluster.get_positions()
            all_positions = all_positions[:len(centered_padded_cluster)]
            all_distances = np.linalg.norm(
                all_positions[:, np.newaxis] - all_positions[np.newaxis, :], axis=-1)
            all_values = np.array(
                [all_distances[(first_index, second_index)]
                 for i in range(number_repetitions)
                for first_index, second_index in bond_pairs], dtype=np.float32)

            frame.constraints.N = len(bond_pairs) * number_repetitions
            frame.constraints.group = all_constraints
            frame.constraints.value = all_values

            # Useful for visualization although not necessary for the simulation.
            frame.bonds.N = len(bond_pairs) * number_repetitions
            frame.bonds.types = ["b"]
            frame.bonds.typeid = np.zeros(frame.bonds.N, dtype=np.uint32)
            frame.bonds.group = all_constraints

        return frame
