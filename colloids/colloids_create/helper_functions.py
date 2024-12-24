from math import acos, cos, pi, sin, sqrt
from typing import Iterator
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from math import ceil
from typing import Union
from openmm import unit


def build_positions(n_clusters: Union[int, tuple[int]], lattice_constant: tuple[float], cluster_order: list[str], 
                    cluster_specifications: dict[str, dict[str, Union[str, list[list[float]]]]], 
                    random_rotation: bool = False) -> npt.NDArray[np.floating]:
    """
    Build the positions of the clusters on a sphere using the Fibonacci lattice.

    :param n_clusters:
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
    n_clusters = n_clusters - (n_clusters % len(cluster_order))
    # the number of repeats in each dimension
    number_repeats_per_dimension = [ceil(n_clusters**(1/3))] * 3
    # the number of times the order of the clusters is repeated
    n_repeats = n_clusters // len(cluster_order)

    n_colloids = {}
    for cluster in set(cluster_order):
        n_colloids[cluster] = len(cluster_specifications[cluster]["identity"])
    n_colloids_per_repeat = sum([n_colloids[cluster] for cluster in cluster_order])

    n_colloids_total = n_colloids_per_repeat * n_repeats

    # the intracluster ids are the identity of the colloids in the cluster
    intracluster_ids = []
    for cluster in cluster_order:
        intracluster_ids += list(range(len(cluster_specifications[cluster]["identity"])))
    intracluster_ids = intracluster_ids * n_repeats
        

    # the colloids types are the identity of the colloids in the cluster
    colloid_types = []
    for cluster in cluster_order:
        colloid_types += cluster_specifications[cluster]["identity"]

    colloid_types = colloid_types * n_repeats

    # create the cluster ids (dict method is used to remove duplicates and keep the order)
    cluster_ids = []
    cluster_id_dict = {cluster: i for i, cluster in enumerate(list(dict.fromkeys(cluster_order)))}

    # the cluster ids are a one hot encoding of the colloid types
    for cluster in cluster_order:
        cluster_ids += [cluster_id_dict[cluster]] * n_colloids[cluster]
    cluster_ids = np.tile(np.array(cluster_ids), n_repeats)

    # the cluster numbers are a unique number for each cluster based on genertation order
    cluster_numbers = []
    for i, cluster in enumerate(cluster_order * n_repeats):
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
                relative_cluster_index = cluster_index % len(cluster_order)
                # the number of times the cluster (of this specific index in the order) has been repeated
                n_repeat = cluster_index // len(cluster_order)
                # use above to find where to insert the cluster in the positions tensor
                positions_mask = (cluster_numbers == cluster_index) * (repeat_index == n_repeat)

                # the cluster name
                cluster = cluster_order[relative_cluster_index]

                # create the positions of the cluster
                relative_positions = cluster_specifications[cluster]["coordinates"]
                if random_rotation:
                    rotation = Rotation.from_euler("xyz", np.random.uniform(0, 2*pi, 3))
                    relative_positions = rotation.apply(relative_positions)
                offset = np.array([x_i, y_i, z_i]) * lattice_constant.value_in_unit(unit.nanometer)
                positions[positions_mask] = relative_positions + offset

                cluster_index += 1
                if cluster_index == n_clusters:
                    break
            if cluster_index == n_clusters:
                break
        if cluster_index == n_clusters:
            break
        
    return positions, intracluster_ids, colloid_types, cluster_ids, cluster_numbers

def get_constraint_dict(cluster_specifications: dict[str, dict[str, Union[str, list[list[float]]]]]) -> dict[str, dict[int, npt.NDArray[np.floating]]]:
    """
    Get the positions of the colloids in the cluster.

    :param cluster_specifications:
        The specifications of the clusters. A dictionary with the cluster name as the key and a dictionary with the
        identity of the atoms in the cluster as the key and the positions of the atoms in the cluster as the value.
    :type cluster_specifications: dict[str, dict[str, Union[str, list[list[float]]]]
    """
    constraints = {}

    for cluster in cluster_specifications:
        coordinates = cluster_specifications[cluster]["coordinates"]
        distance_matrix = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1)

        constraints[cluster] = {}
        for i in range(len(coordinates)):
            constraints[cluster][i] = distance_matrix[i]

    return constraints

def get_constrain_map(cluster_numbers: npt.NDArray[np.int32]) -> list[npt.NDArray[np.int32]]:
    """
    Get the constraint map for the clusters.

    :param cluster_numbers:
        The number of the cluster.
    :type cluster_numbers: npt.NDArray[np.int32]
    """
    n_colloids = len(cluster_numbers)
    constraint_map = []

    for i in range(n_colloids):
        cluster_number = cluster_numbers[i]
        in_cluster = np.where(cluster_numbers == cluster_number)

        constraint_map.append(in_cluster)

    return constraint_map