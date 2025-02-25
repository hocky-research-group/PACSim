from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
from openmm import unit
from colloids.abstracts import Parameters

_nanometer = unit.nano * unit.meter # There is a bug in the openmm unit system that causes a memory leak when using a unit.x constant.
_millivolt = unit.milli * unit.volt
_amu = unit.amu
@dataclass(order=True, frozen=True)
class ConfigurationParameters(Parameters):
    """
    Data class for the parameters of the colloids configuration to be created for an OpenMM simulation.

    This dataclass can be written to and read from a yaml file. The yaml file contains the parameters as key-value
    pairs. Any OpenMM quantities are converted to Quantity objects that can be represented in a readable way in the
    yaml file.

    The base configuration consists of two types of colloids. It is created by placing the colloidal particles with the
    bigger radius on a cubic lattice structure. The colloidal particles with a smaller radius are then placed as
    satellites around these centers.

    After the base configuration has been created, it can be modified by adding a substrate at the bottom of the
    simulation box, and by adding snowman heads to given colloid types.

    :param total_clusters:
        The total number of clusters to generate in the initial configuration.
        Must be a positive integer.
    :type total_clusters: int
    :param lattice_constant:
        The lattice constant of the cubic (or orthorhombic) lattice structure in each dimension (rhombohedral tbi).
        The unit must be compatible with nanometers and the value must be greater than zero.
    :type lattice_constant: union[unit.Quantity, tuple[unit.Quantity]]
    :param box_size:
        The size of the simulation box in each dimension.
    :type box_size: unit.Quantity
    :param padding_distance:
        The minimum distance between colloids in the different clusters. Defaults to 0.0 * _nanometer.
    :type padding_distance: unit.Quantity
    :param padding_factor:
        The fraction of the lattice constant that is used as padding between the colloids and the edge of the box.
        Defaults to 0.5.
    :type padding_factor: float
    :param cluster_order:
        The order in which the clusters should be placed in the lattice.
        Must be a list of strings with the names of the clusters.
    :type cluster_order: list[str]
    :param masses:
        The masses of the different types of colloidal particles that appear in the initial configuration file.
        The keys of the dictionary are the types of the colloidal particles and the values are the masses.
        The unit of the masses must be compatible with atomic mass units and the values must be greater than zero,
        except for immobile particles (as the substrate), which should have a mass of zero.
        Defaults to {"P": 1.0 * _amu, "N": (95.0 / 105.0) ** 3 * _amu}.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii of the different types of colloidal particles that appear in the initial configuration file.
        The keys of the dictionary are the types of the colloidal particles and the values are the radii.
        The unit of the radii must be compatible with nanometers and the values must be greater than zero.
        Defaults to {"P": 105.0 * (_nanometer), "N": 95.0 * (_nanometer)}.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials of the different types of colloidal particles that appear in the initial configuration
        file.
        The keys of the dictionary are the types of the colloidal particles and the values are the surface potentials.
        The unit of the surface potentials must be compatible with millivolts.
        Defaults to {"P": 44.0 * (_millivolt), "N": -54.0 * (_millivolt)}.
    :type surface_potentials: dict[str, unit.Quantity]
    :param cluster_specifications:
        The specifications of the clusters. A dictionary with the cluster name as the key and a dictionary as the value.
        The inner dictionary has the identity of the atoms filed under "identity" and the positions of the atoms in the
        cluster filed under "coordinates". Must not be None.
    :type cluster_specifications: dict[str, dict[str, Union[str, list[list[unit.Quantity]]]]]
    :param use_substrate:
        A boolean that indicates whether a substrate should be added to the configuration.
        Defaults to False.
    :type use_substrate: bool
    :param substrate_type:
        The type of the substrate particles. Defaults to "S".
    :type substrate_type: str
    :param random_rotation:
        A boolean that indicates whether the clusters should be randomly rotated.
        Defaults to False.
    :type random_rotation: bool
    """
    total_clusters: int
    lattice_constant: Union[unit.Quantity, tuple[unit.Quantity]]
    box_size: unit.Quantity
    cluster_order: list[str] = None
    random_rotation: bool = False
    padding_factor: float = 0.5
    cluster_specifications: dict[str, dict[str, Union[list[str], list[unit.Quantity]]]] = None
    padding_distance: unit.Quantity = field(default_factory=lambda: 0.0 * _nanometer)
    masses: dict[str, unit.Quantity] = field(default_factory=lambda: {"P": 1.0 * _amu, "N": (95.0 / 105.0) ** 3 * _amu})
    radii: dict[str, unit.Quantity] = field(default_factory=lambda: {"P": 105.0 * (_nanometer), "N": 95.0 * (_nanometer)})
    surface_potentials: dict[str, unit.Quantity] = field(default_factory=lambda: {"P": 44.0 * (_millivolt), "N": -54.0 * (_millivolt)})
    use_substrate: bool = False
    substrate_type: str = None


    def __post_init__(self):
        """Post-initialization method for the ConfigurationParameters class."""

        # Check the lattice specifications and number of clusters.
        if not isinstance(self.total_clusters, int):
            raise TypeError("The total number of clusters must be an integer.")
        if self.total_clusters <= 0:
            raise ValueError("The total number of clusters must be a positive integer.")
        if not isinstance(self.lattice_constant, unit.Quantity) and not isinstance(self.lattice_constant, tuple):
            raise TypeError("The lattice constant must be a Quantity or a tuple of Quantities.")
        if isinstance(self.lattice_constant, unit.Quantity):
            if not self.lattice_constant.unit.is_compatible(_nanometer):
                raise TypeError("The lattice constant must have a unit compatible with nanometers.")
            if any(lattice_constant <= 0.0 * (_nanometer) for lattice_constant in self.lattice_constant):
                raise ValueError("The lattice constant must be greater than zero.")
        if isinstance(self.lattice_constant, tuple):
            for l in self.lattice_constant:
                if not l.unit.is_compatible(_nanometer):
                    raise TypeError("The lattice constant must have a unit compatible with nanometers.")
                if l <= 0.0 * (_nanometer):
                    raise ValueError("The lattice constant must be greater than zero.")
        if not isinstance(self.random_rotation, bool):
            raise TypeError("The random rotation must be a boolean.")
        
        # Check that there is sufficient volume for the unit cells.
        if not len(self.lattice_constant) == 3:
            raise ValueError("The lattice constant must be three quantities compatible with nanometer.")
        if not all(isinstance(i, unit.Quantity) for i in self.lattice_constant):
            raise TypeError("The lattice constant must be a list of quantities.")
        if len(self.box_size) == 6:
            raise ValueError("Tilted box not supported currently.")
        if not len(self.box_size) == 3:
            raise ValueError("The box size must be a list of three quantities.")
        effective_repeats = np.array([self.box_size[i].value_in_unit(_nanometer) / self.lattice_constant[i].value_in_unit(_nanometer) for i in range(3)])
        effective_clusters = np.prod(np.floor(effective_repeats - 2.0 * self.padding_factor))
        if self.cluster_order is not None:
            n_unit_cells = len(self.cluster_order)
        else:
            key = list(self.cluster_specifications.keys())[0]
            cluster_names = list(dict.fromkeys(self.cluster_specifications[key]["cluster"]))
            n_clusters_per_unit_cell = len(cluster_names)
            n_unit_cells = self.total_clusters // n_clusters_per_unit_cell

        if n_unit_cells > effective_clusters:
            raise ValueError("The volume of the unit cell times the number of clusters must be less than the volume of the box.")
        
        # Check the cluster order.
        clusters_in_specifications = set(self.cluster_specifications.keys())
        if isinstance(self.cluster_order, list):
            clusters_in_order = set(self.cluster_order)
            if not clusters_in_order == clusters_in_specifications:
                raise ValueError("The cluster order and the cluster specifications must have the same keys.")
            unit_cell = False
        elif self.cluster_order is None:
            if len(self.cluster_specifications) > 1:
                raise ValueError("The cluster order must be specified if there is more than one cluster, unit cell is not permitted.")
            if not all("cluster" in self.cluster_specifications[cluster] for cluster in self.cluster_specifications):
                raise ValueError("The cluster specifications must have a cluster key when using it as a unit cell.")
            if self.random_rotation:
                raise ValueError("Random rotation is not permitted when using the unit cell.")
            unit_cell = True
        else:
            raise TypeError("The cluster order must be a list of strings or None.")
        
        if not all("identity" in self.cluster_specifications[cluster] for cluster in self.cluster_specifications):
            raise ValueError("The cluster specifications must have an identity key.")
        if not all("coordinates" in self.cluster_specifications[cluster] for cluster in self.cluster_specifications):
            raise ValueError("The cluster specifications must have a coordinates key.")

        # Check the cluster specifications.
        if not isinstance(self.cluster_specifications, dict):
            raise TypeError("The cluster specifications must be a dictionary.")
        for cluster in self.cluster_specifications:
            if "identity" not in self.cluster_specifications[cluster]:
                raise ValueError(f"Cluster {cluster} must have an identity key in the cluster specifications.")
            if not isinstance(self.cluster_specifications[cluster]["identity"], list):
                raise TypeError(f"Identity of cluster {cluster} must be a list of strings.")
            if not all(isinstance(i, str) for i in self.cluster_specifications[cluster]["identity"]):
                raise TypeError(f"Identity of cluster {cluster} must be a list of strings.")
            if "coordinates" not in self.cluster_specifications[cluster]:
                raise ValueError(f"Cluster {cluster} must have a coordinates key in the cluster specifications.")
            if not self.cluster_specifications[cluster]["coordinates"].unit.is_compatible(_nanometer):
                raise TypeError(f"Coordiantes of cluster {cluster} must hvae units compatible with nm.")
            if not all(len(i) == 3 for i in self.cluster_specifications[cluster]["coordinates"]):
                raise ValueError(f"Coordinates of cluster {cluster} must be a iterable of triplets of nm.")

        # Check the masses, radii, and surface potentials of colloids.
        for t in self.masses:
            if not self.masses[t].unit.is_compatible(_amu):
                raise TypeError(f"Mass of type {t} must have a unit compatible with atomic mass units.")
            if self.masses[t] < 0.0 * _amu:
                raise ValueError(f"Mass of type {t} must be greater than zero.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the masses dictionary is not in radii dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the masses dictionary is not in surface potentials dictionary.")
        for t in self.radii:
            if not self.radii[t].unit.is_compatible(_nanometer):
                raise TypeError(f"Radius of type {t} must have a unit compatible with nanometers.")
            if self.radii[t] <= 0.0 * (_nanometer):
                raise ValueError(f"Radius of type {t} must be greater than zero.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the radii dictionary is not in masses dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the radii dictionary is not in surface potentials dictionary.")
        for t in self.surface_potentials:
            if not self.surface_potentials[t].unit.is_compatible(_millivolt):
                raise TypeError(f"Surface potential of type {t} must have a unit compatible with millivolts.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in masses dictionary.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in radii dictionary.")
        colloids_in_specifications = set([k for k in self.cluster_specifications.values() for k in k["identity"]])
        for t in colloids_in_specifications:
            if t not in self.masses:
                raise ValueError(f"Colloid type {t} in cluster_specifications is not in masses dictionary.")
            if t not in self.radii:
                raise ValueError(f"Colloid type {t} in cluster_specifications is not in radii dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Colloid type {t} in cluster_specifications is not in surface potentials dictionary.")

        # Check the substrate type.
        if not isinstance(self.use_substrate, bool):
            raise TypeError("The use substrate must be a boolean.")
        if self.use_substrate:
            if not isinstance(self.substrate_type, str):
                raise TypeError("The substrate type must be a string.")
            if self.substrate_type not in self.masses:
                raise ValueError("The substrate type must be in the masses dictionary.")
            elif self.masses[self.substrate_type] != 0.0 * _amu:
                raise ValueError("The mass of the substrate type must be zero.")
            if self.substrate_type not in self.radii:
                raise ValueError("The substrate type must be in the radii dictionary.")
            if self.substrate_type not in self.surface_potentials:
                raise ValueError("The substrate type must be in the surface potentials dictionary.")
            
            # Add the substrate to the colloids in the specifications. 
            # Substrate should be a separate type from any fixed atoms of the same type in the cluster specifications.
            self.masses["__substrate__"] = self.masses[self.substrate_type]
            self.radii["__substrate__"] = self.radii[self.substrate_type]
            self.surface_potentials["__substrate__"] = self.surface_potentials[self.substrate_type]
        if "__substrate__" in colloids_in_specifications:
            raise ValueError("The substrate type must not be in the cluster specifications. Restricted type name.")


        # Check the longest distance between colloids plus radii of those colloids is less than the (shortest) lattice constant.
        if not unit_cell:
            for cluster in self.cluster_specifications:
                coordinates = np.array(self.cluster_specifications[cluster]["coordinates"].value_in_unit(_nanometer))
                distance_matrix = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1)

                radius_vector = np.array([self.radii[identity].value_in_unit(_nanometer) for identity in self.cluster_specifications[cluster]["identity"]])
                radius_matrix = radius_vector[:, np.newaxis] + radius_vector[np.newaxis, :]

                min_lattice_vec = min(self.lattice_constant)
                max_distance = np.max(distance_matrix + radius_matrix)
                if max_distance + self.padding_distance.value_in_unit(_nanometer) > min_lattice_vec.value_in_unit(_nanometer):
                    raise ValueError(f"The distance between the colloids plus the radii of the colloids must be less than the lattice constant for cluster {cluster}.")

        else:
            for cluster in self.cluster_specifications:

                coordinates = np.array(self.cluster_specifications[cluster]["coordinates"].value_in_unit(_nanometer))
                distance_matrix = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1)

                radius_vector = np.array([self.radii[identity].value_in_unit(_nanometer) for identity in self.cluster_specifications[cluster]["identity"]])
                radius_matrix = radius_vector[:, np.newaxis] + radius_vector[np.newaxis, :]

                clusters = self.cluster_specifications[cluster]["cluster"]
                is_in_same_cluster = np.array([[clusters[i] == clusters[j] for j in range(len(clusters))] for i in range(len(clusters))])
                min_distance = np.min((distance_matrix - radius_matrix)[~is_in_same_cluster])
                if min_distance < self.padding_distance.value_in_unit(_nanometer):
                    raise ValueError(f"The distance between the colloids minus the radii of the colloids must be greater than the padding disctance for unit cell.")
