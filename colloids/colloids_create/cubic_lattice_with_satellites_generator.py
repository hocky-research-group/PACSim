from enum import auto, Enum
from typing import Union
from ase import Atom, build, Atoms
from gsd.hoomd import Frame
import numpy as np
import warnings
from openmm import unit
from colloids.colloids_create import ConfigurationGenerator
from colloids.colloids_create.helper_functions import build_positions, get_constraint_dict, get_constraint_map, get_constraint_dists


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


class CubicLatticeWithSatellitesGenerator(ConfigurationGenerator):
    _nanometer = unit.nano * unit.meter

    def __init__(self, lattice_constant: unit.Quantity, 
                 total_clusters: int, 
                 cluster_order: Union[str, list[str]], 
                 padding_distance: unit.Quantity, 
                 cluster_specifications: dict[str, dict[str, Union[str, list[list[float]]]]], 
                 colloid_radii: dict[str, unit.Quantity], 
                 masses: dict[str, unit.Quantity], 
                 random_rotation: bool = False) -> None:
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
        :param padding_distance:
        The minimum distance between the clusters at initialization.
        :type padding_distance: unit.Quantity
        :param cluster_specifications:
        The specifications of the clusters. A dictionary with the cluster name as the key and the value: a dictionary with the
        identity of the atoms in the cluster as the key and the positions of the atoms in the cluster.
        :type cluster_specifications: dict[str, dict[str, Union[str, list[list[float]]]]
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


        if not lattice_constant.unit.is_compatible(self._nanometer):
            raise TypeError("The lattice constant must have a unit that is compatible with nanometers.")
        if not lattice_constant.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("The lattice constant must have a value greater than zero.")
        if isinstance(total_clusters, int):
            if not total_clusters > len(cluster_order):
                raise ValueError("The number of lattice repeats must be greater than the length of the cluster order.")
        if not padding_distance.unit.is_compatible(self._nanometer):
            raise TypeError("The padding distance must have a unit that is compatible with nanometers.")
        if not padding_distance.value_in_unit(self._nanometer) >= 0.0:
            raise ValueError("The padding distance must have a value greater than or equal to zero.")

        clusters_requested = set(cluster_order)
        clusters_specified = set(cluster_specifications.keys())

        if clusters_requested > clusters_specified:
            raise ValueError("The clusters requested by ordering must be contained in clusters specified by cluster file.")
        if clusters_requested < clusters_specified:
            left_out_clusters = clusters_requested - clusters_specified
            warnings.warn(f"The clusters {left_out_clusters} specified in cluster file are not used in the ordering.")
        if not all(cluster_specifications[cluster].keys() == {"identity", "coordinates"} for cluster in clusters_requested):
            raise ValueError("The cluster specifications must contain the keys 'identity', 'coordinates'.")
        if not all(isinstance(cluster_specifications[cluster]["identity"], list) for cluster in clusters_requested):
            raise TypeError("The cluster identities must be a list of strings.")
        if not all(isinstance(cluster_specifications[cluster]["identity"][0], str) for cluster in clusters_requested):
            raise TypeError("The cluster identities must be a list of strings.")
        if not all(colloid_radius.unit.is_compatible(self._nanometer) for colloid_radius in colloid_radii.values()):
            raise TypeError("The colloid radii must have units that are compatible with nanometers.")
        
        orbit_distance = padding_distance.value_in_unit(self._nanometer)
        for cluster in clusters_specified:
            coordinates = cluster_specifications[cluster]["coordinates"]
            identities = cluster_specifications[cluster]["identity"]

            for i, coordinate in enumerate(coordinates):
                if not isinstance(coordinate, list):
                    raise TypeError("The coordinates must be a list of lists.")
                if not all(isinstance(coord_xyz, float) for coord_xyz in coordinate):
                    raise TypeError("The coordinates must be a list of lists of floats (that will take on units of the nm).")
                if not all(len(coordinate) == 3 for coordinate in coordinates):
                    raise ValueError("The coordinates must be a list of lists of length three.")
                
            if not isinstance(identities, list):
                raise TypeError("The identities must be a list.")
            if not all(isinstance(identity, str) for identity in identities):
                raise TypeError("The identities must be a list of strings.")
            if not len(identities) == len(coordinates):
                raise ValueError("The identities and coordinates must have the same length.")
            if not all(identity in colloid_radii for identity in identities):
                raise ValueError("The identities must be contained in the colloid radii dictionary.")
        
            
            coordinates = np.array(coordinates)
            colloid_center_distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)

            cluster_colloid_radii = np.array([colloid_radii[identity].value_in_unit(self._nanometer) for identity in identities])
            pairwise_colloid_radii = cluster_colloid_radii[:, None] + cluster_colloid_radii[None, :]

            cluster_orbit_distance = np.max(colloid_center_distances) + np.max(pairwise_colloid_radii) + padding_distance.value_in_unit(self._nanometer)
            orbit_distance = max(orbit_distance, cluster_orbit_distance)
        orbit_distance *= unit.nanometer

        if not orbit_distance < lattice_constant:
            raise ValueError("The orbit distance must be smaller than the lattice constant.")

        self._lattice = lattice
        self.masses = masses
        self._lattice_constant = lattice_constant
        self._total_clusters = total_clusters
        self._padding_distance = padding_distance
        self._cluster_order = cluster_order
        self._cluster_specifications = cluster_specifications
        self._colloid_radii = colloid_radii
        self._random_rotation = random_rotation

    def generate_configuration(self) -> tuple[Frame, list[tuple[int]]]:
        # Create the lattice.
        positions, intracluster_ids, colloid_types, cluster_ids, cluster_numbers, cluster_id_dict = build_positions(self._total_clusters, self._lattice_constant, 
                                                                                  self._cluster_order, self._cluster_specifications, random_rotation=self._random_rotation)
        # Tags are a linear combination of the intracluster ids, the cluster ids, and the cluster numbers. They can 
        # be decomposed into the intracluster ids by taking the floor after dividing by number of cluster types times the
        # total number of clusters, cluster numbers by taking the floor after dividing by the number of cluster types mod
        # the total number of clusters, and cluster ids by taking the modulo base the total number of clusters and modulo
        # base the number of cluster types.

        # cluster_numbers = [(tag // n_cluster_types) % n_clusters for tag in tags]
        # cluster_ids = [(tag % n_cluster_types) % n_clusters for tag in tags]
        # intracluster_ids = [(tag // n_cluster_types) // n_clusters for tag in tags]

        n_clusters = np.max(cluster_numbers) + 1
        n_cluster_types = len(set(self._cluster_order))
        tags = cluster_ids  + np.array(intracluster_ids) * n_cluster_types * n_clusters + cluster_numbers * n_cluster_types

        constraint_dist_dict = get_constraint_dict(self._cluster_specifications)
        constraint_map = get_constraint_map(cluster_numbers)
        constraint_dists = get_constraint_dists(constraint_map, constraint_dist_dict, cluster_ids)

        masses = [self.masses[identity] for identity in colloid_types]
        atoms = Atoms(symbols=["X"] * len(tags), positions=positions.tolist(), tags=tags.tolist(), masses=masses)

        # Center the center atoms around the origin.
        atoms.center(about=(0.0, 0.0, 0.0))

        # Create the frame.
        frame = Frame()
        frame.particles.N = len(atoms)
        frame.particles.position = atoms.positions.astype(np.float32)
        
        # See http://docs.openmm.org/7.6.0/userguide/theory/05_other_features.html
        # See https://hoomd-blue.readthedocs.io/en/v2.9.3/box.html
        frame.configuration.box = np.array([atoms.cell[0][0], atoms.cell[1][1], atoms.cell[2][2],
                                            atoms.cell[1][0] / atoms.cell[1][1], atoms.cell[2][0] / atoms.cell[2][2],
                                            atoms.cell[2][1] / atoms.cell[2][2]], dtype=np.float32)

        frame.particles.types = np.array(colloid_types)
        frame.particles.mass = np.array([mass.value_in_unit(unit.amu) for mass in masses], dtype=np.float32)
        frame.particles.diameter = np.array([2.0 * self._colloid_radii[identity].value_in_unit(unit.nanometer) for identity in colloid_types], dtype=np.float32)

        frame.particles.position = atoms.positions.astype(np.float32)
        frame.particles.N = len(atoms)

        self.atoms = atoms

        return frame, list(zip(intracluster_ids, cluster_ids, cluster_numbers, constraint_map, constraint_dists))

    def write_positions(self) -> None:
        # Save positions as xyz
        with open("positions.xyz", "w") as f:
            f.write(f"{len(self.atoms)}\n")
            f.write("Lattice\n")
            for i, atom in enumerate(self.atoms):
                f.write(f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n")

if __name__ == "__main__":
    lattice = CubicLattice.SC
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
            "identity": ["B", "B", "A", "A", "A", "A"],
            "coordinates": [[1.0, 2.5, 0.0],
                            [-1.0, 2.5, 0.0],
                            [-1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.5, -0.25, 0.0],
                            [-0.5, -0.25, 0.0]]
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
    generator = CubicLatticeWithSatellitesGenerator(lattice_constant, total_clusters, cluster_order,
                                                    padding_distance, cluster_specifications, colloid_radii, masses, random_rotation=True)
    frame, constraints = generator.generate_configuration()
    print(frame)
    print(constraints)