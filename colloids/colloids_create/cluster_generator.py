from enum import auto, Enum
from typing import Union
from ase import Atom, build, Atoms

from gsd.hoomd import Frame
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from math import pi, ceil

import warnings
from openmm import unit
from colloids.colloids_create import ConfigurationGenerator


class CubicLattice(Enum):
    # TODO: Add docstrings.
    SC = auto()
    FCC = auto()
    BCC = auto()

    def to_ase_string(self):
        return self.name.lower()

    @staticmethod
    def from_string(string: str):
        return CubicLattice[string.upper()]


class ClusterGenerator(ConfigurationGenerator):
    _nanometer = unit.nano * unit.meter

    def __init__(self, configuration_parameters: ConfigurationGenerator) -> None:
        super().__init__()

        """
        :param lattice_constant:
            The lattice constant of the lattice.
        :type lattice_constant: unit.Quantity
        :param total_clusters:
            The number of clusters to generate.
        :type total_clusters: int
        :param cluster_order:
            The order of the clusters to be placed in a simple cubic latice.
        :type cluster_order: Union[str, list[str]]
        :param box_size:
            The size of the box in each dimension or for a square box the length of a side.
        :type box_size: unit.Quantity
        :param cluster_specifications:
            The specifications of the clusters. A dictionary with the cluster name as the key and the value: a dictionary with the
            identity of the atoms in the cluster as the key and the positions of the atoms in the cluster.
        :type cluster_specifications: dict[str, dict[str, Union[list[str], list[unit.Quantity]]]
        :param colloid_radii:
            A dictionary of the radii of the colloids.
        :type colloid_radii: dict[str, unit.Quantity]
        :param masses:
            A dictionary of the masses of the colloids.
        :type masses: dict[str, unit.Quantity]
        :param random_rotation:
            Whether to rotate the cluster randomly.
        :type random_rotation: bool
        """
        box_size: unit.Quantity = configuration_parameters.box_size
        lattice_constant: unit.Quantity = configuration_parameters.lattice_constant
        total_clusters: int = configuration_parameters.total_clusters
        cluster_order: Union[str, list[str]] = configuration_parameters.cluster_order
        padding_distance: unit.Quantity = configuration_parameters.padding_distance
        cluster_specifications: dict[str, dict[str, Union[list[str], list[unit.Quantity]]]] = configuration_parameters.cluster_specifications
        colloid_radii: dict[str, unit.Quantity] = configuration_parameters.radii
        masses: dict[str, unit.Quantity] = configuration_parameters.masses
        random_rotation: bool = configuration_parameters.random_rotation

        self.masses = masses
        self.box_size = box_size
        self._lattice_constant = lattice_constant
        self._total_clusters = total_clusters
        self._padding_distance = padding_distance
        self._cluster_order = cluster_order
        self._cluster_specifications = cluster_specifications
        self._colloid_radii = colloid_radii
        self._random_rotation = random_rotation

    def generate_configuration(self) -> tuple[Frame, list[tuple[int]]]:
        # Create the lattice.
        self.build_positions()
        # Tags are a linear combination of the intracluster ids, the cluster ids, and the cluster numbers. They can 
        # be decomposed into the intracluster ids by taking the floor after dividing by number of cluster types times the
        # total number of clusters, cluster numbers by taking the floor after dividing by the number of cluster types mod
        # the total number of clusters, and cluster ids by taking the modulo base the total number of clusters and modulo
        # base the number of cluster types.

        # cluster_numbers = [(tag // n_cluster_types) % n_clusters for tag in tags]
        # cluster_ids = [(tag % n_cluster_types) % n_clusters for tag in tags]
        # intracluster_ids = [(tag // n_cluster_types) // n_clusters for tag in tags]

        """
        n_clusters = np.max(self.cluster_numbers) + 1
        n_cluster_types = len(set(self._cluster_order))
        tags = self.cluster_ids  + np.array(self.intracluster_ids) * n_cluster_types * n_clusters + self.cluster_numbers * n_cluster_types
        """
        
        self.get_constraint_dict()
        self.get_constraint_map()
        self.get_constraint_dists()

        # Create the frame.
        frame = Frame()
        frame.particles.N = len(self.positions)
        frame.particles.position = self.positions
        
        # See http://docs.openmm.org/7.6.0/userguide/theory/05_other_features.html
        # See https://hoomd-blue.readthedocs.io/en/v2.9.3/box.html
        frame.configuration.box = np.array([self.box_size[0].value_in_unit(self._nanometer), self.box_size[1].value_in_unit(self._nanometer), 
                                            self.box_size[2].value_in_unit(self._nanometer), 0, 0, 0], dtype=np.float32)
        
        frame.particles.types = tuple(self.colloid_id_dict.keys())
        frame.particles.typeid = [self.colloid_id_dict[colloid_type] for colloid_type in self.colloid_types]

        return frame, list(zip(self.constraint_map, self.constraint_dists, self.intracluster_ids, self.cluster_ids, self.cluster_numbers))

    def write_positions(self) -> None:
        # Save positions as xyz
        with open("positions.xyz", "w") as f:
            f.write(f"{len(self.atoms)}\n")
            f.write("Lattice\n")
            for i, atom in enumerate(self.atoms):
                f.write(f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n")

    def build_positions(self) -> None:
        """
        Build the positions of the clusters on a sphere using the Fibonacci lattice.

        :param total_clusters:
            The number of clusters to generate
        :type number_points: int
        :param lattice_constant:
            The lattice constant in each dimension.
        :type lattice_constant: tuple[float]
        :param cluster_order:
            The order of the clusters to be placed in a simple cubic latice.
        :type cluster_order: list[str]
        :param cluster_specifications:
            The specifications of the clusters. A dictionary with the cluster name as the key and a dictionary with the
            identity of the atoms in the cluster as the key and the positions of the atoms in the cluster as the value.
        :type cluster_specifications: dict[str, dict[str, Union[str, list[list[float]]]]
        :param random_rotation:
            Whether to rotate the cluster randomly.
        """
        # the number of clusters total
        self._total_clusters = self._total_clusters - (self._total_clusters % len(self._cluster_order))
        # the number of repeats in each dimension
        number_repeats_per_dimension = [ceil(self._total_clusters**(1/3))] * 3
        # the number of times the order of the clusters is repeated
        n_repeats = self._total_clusters // len(self._cluster_order)

        n_colloids = {}
        for cluster in set(self._cluster_order):
            n_colloids[cluster] = len(self._cluster_specifications[cluster]["identity"])
        n_colloids_per_repeat = sum([n_colloids[cluster] for cluster in self._cluster_order])

        n_colloids_total = n_colloids_per_repeat * n_repeats

        # the intracluster ids are the identity of the colloids in the cluster
        intracluster_ids = []
        for cluster in self._cluster_order:
            intracluster_ids += list(range(len(self._cluster_specifications[cluster]["identity"])))
        intracluster_ids = intracluster_ids * n_repeats
            

        # the colloids types are the identity of the colloids in the cluster
        colloid_types = []
        for cluster in self._cluster_order:
            colloid_types += self._cluster_specifications[cluster]["identity"]

        colloid_types = colloid_types * n_repeats

        # create the cluster ids (dict method is used to remove duplicates and keep the order)
        cluster_id_dict = {cluster: i for i, cluster in enumerate(list(dict.fromkeys(self._cluster_order)))}
        colloid_id_dict = {colloid: i for i, colloid in enumerate(list(dict.fromkeys(colloid_types)))}

        # the cluster ids are a one hot encoding of the colloid types
        cluster_ids = []
        for cluster in self._cluster_order:
            cluster_ids += [cluster_id_dict[cluster]] * n_colloids[cluster]
        cluster_ids = np.tile(np.array(cluster_ids), n_repeats)

        # the cluster numbers are a unique number for each cluster based on genertation order
        cluster_numbers = []
        for i, cluster in enumerate(self._cluster_order * n_repeats):
            cluster_numbers += [i] * n_colloids[cluster]
        cluster_numbers = np.array(cluster_numbers)

        repeat_index = np.repeat(np.arange(n_repeats), n_colloids_per_repeat)

        # create the positions
        positions = np.zeros((n_colloids_total, 3))
        cluster_index = 0
        for x_i in range(number_repeats_per_dimension[0]):
            for y_i in range(number_repeats_per_dimension[1]):
                for z_i in range(number_repeats_per_dimension[2]):
                    # the index of the cluster in the cluster order
                    relative_cluster_index = cluster_index % len(self._cluster_order)
                    # the number of times the cluster (of this specific index in the order) has been repeated
                    n_repeat = cluster_index // len(self._cluster_order)
                    # use above to find where to insert the cluster in the positions tensor
                    positions_mask = (cluster_numbers == cluster_index) * (repeat_index == n_repeat)

                    # the cluster name
                    cluster = self._cluster_order[relative_cluster_index]

                    # create the positions of the cluster
                    relative_positions = self._cluster_specifications[cluster]["coordinates"]
                    if self._random_rotation:
                        rotation = Rotation.from_euler("xyz", np.random.uniform(0, 2*pi, 3))
                        relative_positions = rotation.apply(relative_positions.value_in_unit(unit.nanometer))
                    offset = np.array([x_i, y_i, z_i]) * self._lattice_constant.value_in_unit(unit.nanometer)
                    positions[positions_mask] = relative_positions + offset

                    cluster_index += 1
                    if cluster_index == self._total_clusters:
                        break
                if cluster_index == self._total_clusters:
                    break
            if cluster_index == self._total_clusters:
                break
            
        self.positions = positions
        self.cluster_numbers = cluster_numbers.squeeze()
        self.cluster_ids = cluster_ids
        self.intracluster_ids = intracluster_ids
        self.cluster_id_dict = cluster_id_dict
        self.colloid_types = colloid_types
        self.colloid_id_dict = colloid_id_dict

    def get_constraint_dict(self) -> dict[str, dict[int, npt.NDArray[np.floating]]]:
        """
        Get the positions of the colloids in the cluster.

        :param cluster_specifications:
            The specifications of the clusters. A dictionary with the cluster name as the key and a dictionary with the
            identity of the atoms in the cluster as the key and the positions of the atoms in the cluster as the value.
        :type cluster_specifications: dict[str, dict[str, Union[str, list[list[float]]]]
        """
        constraints = {}

        for cluster in self._cluster_specifications:
            coordinates = np.array(self._cluster_specifications[cluster]["coordinates"].value_in_unit(unit.nanometer))
            distance_matrix = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1)

            constraints[cluster] = {}
            for i in range(len(coordinates)):
                constraints[cluster][i] = distance_matrix[i]

        self.constraint_dist_dict = constraints

    def get_constraint_map(self) -> list[npt.NDArray[np.int32]]:
        """
        Get the constraint map for the clusters.

        :param cluster_numbers:
            The number of the cluster.
        :type cluster_numbers: npt.NDArray[np.int32]
        """
        n_colloids = len(self.cluster_numbers)
        constraint_map = []

        for i in range(n_colloids):
            cluster_number = self.cluster_numbers[i]
            current_index_onehot = np.zeros(n_colloids)
            current_index_onehot[i] = 1
            
            in_cluster = np.where(np.logical_and(self.cluster_numbers == cluster_number, current_index_onehot == 0))

            constraint_map.append(in_cluster[0])

        self.constraint_map = constraint_map

    def get_constraint_dists(self) -> list[npt.NDArray[np.floating]]:
        constraint_dists = []
        reverse_cluster_id_dict = {v: k for k, v in self.cluster_id_dict.items()}

        for i in range(len(self.constraint_map)):
            cluster_id = self.cluster_ids[i]
            intracluster_id = self.intracluster_ids[i]
            
            # the distances between colloids in the cluster type specified by the cluster_id and containing the colloid specified by intracluster_id
            dists = self.constraint_dist_dict[reverse_cluster_id_dict[cluster_id]][intracluster_id]
            dists = np.delete(dists, intracluster_id)

            constraint_dists.append(dists)

        self.constraint_dists = constraint_dists

if __name__ == "__main__":
    lattice_constant = 6.0 * unit.nanometer
    total_clusters = 2400
    cluster_order = ["A", "B","B"]
    padding_distance = 0.0 * unit.nanometer
    cluster_specifications = {
        "A": {
            "identity": ["A"],
            "coordinates": [[0.0, 0.0, 0.0]]
        },
        "B": {
            "identity": ["B", "B", "B", "B", "C"],
            "coordinates": [[1.0, 1.0, 1.0],
                            [-1.0, -1.0, 1.0],
                            [1.0, -1.0, -1.0],
                            [-1.0, 1.0, -1.0],
                            [0.0, 0.0, 0.0]]
        }
    }
    colloid_radii = {
        "A": 1.0 * unit.nanometer,
        "B": 1.0 * unit.nanometer
    }
    masses = {
        "A": 1.0 * unit.amu,
        "B": 1.0 * unit.amu
    }
    generator = ClusterGenerator(lattice_constant, total_clusters, cluster_order,
                                padding_distance, cluster_specifications, colloid_radii, masses, random_rotation=True)
    frame, constraints = generator.generate_configuration()
    print(frame)
    print(constraints)