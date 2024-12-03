from typing import Optional
import gsd.hoomd
import numpy as np
from numpy.linalg import norm
import numpy.random as npr
from openmm import unit
from colloids.colloids_create import ConfigurationModifier
from colloids.colloids_create.helper_functions import generate_fibonacci_sphere_grid_points


class SnowmanModifier(ConfigurationModifier):
    """
    Modifier of an existing configuration in a gsd.hoomd.Frame instance for a colloid simulation that adds snowman heads
    to the base particles in the configuration.

    :param snowman_bond_types:
        Dictionary mapping from the type of the base particle to the type of the head particle in the snowman colloid.
        Snowman heads are attached to every particle with the given base particle types in this dictionary.
        Every base-particle type must appear in the given frame in the modify_configuration method.
        Every head-particle type must not appear in the given frame in the modify_configuration method.
    :type snowman_bond_types: dict[str, str]
    :param snowman_distances:
        Dictionary mapping from the type of the base particle to the desired distance to the snowman head.
        Every type appearing in the snowman bond types dictionary must have a corresponding distance in this dictionary.
        The unit of every distance must be compatible with nanometers and the value must be greater than zero.
    :type snowman_distances: dict[str, unit.Quantity]
    :param snowman_seed:
        The seed for the random number generator that is used to sample the positions of the snowman heads.
        If zero or smaller than zero, the positions of the snowman heads are not randomized.
        If None, a random seed is used.
        Defaults to None.
    :type snowman_seed: Optional[int]

    :raises ValueError:
        If a type in the snowman bond types dictionary is not in the snowman distances dictionary.
        If a type in the snowman distances dictionary is not in the snowman bond types dictionary.
        If the distance of a type is not greater than zero.
    :raises TypeError:
        If the distance of a type is not a Quantity with a proper unit.
    """
    _nanometer = unit.nano * unit.meter

    def __init__(self, snowman_bond_types: dict[str, str], snowman_distances: dict[str, unit.Quantity],
                 snowman_seed: Optional[int] = None) -> None:
        """Constructor of the SnowmanModifier class."""
        super().__init__()
        self._snowman_seed = snowman_seed
        self._snowman_bond_types = snowman_bond_types
        self._snowman_distances = snowman_distances
        for t in self._snowman_bond_types.keys():
            if t not in self._snowman_distances:
                raise ValueError(f"Type {t} of the snowman bond types dictionary is not in snowman distances "
                                 f"dictionary.")
        for t in self._snowman_distances.keys():
            if t not in self._snowman_bond_types:
                raise ValueError(f"Type {t} of the snowman distances dictionary is not in snowman bond types "
                                 f"dictionary.")
            if not self._snowman_distances[t].unit.is_compatible(self._nanometer):
                raise TypeError(f"Distance of type {t} must have a unit compatible with nanometers.")
            if self._snowman_distances[t] <= 0.0 * self._nanometer:
                raise ValueError(f"Distance of type {t} must be greater than zero.")

    def modify_configuration(self, frame: gsd.hoomd.Frame, constraints: dict[str, float]) -> None:
        """
        Modify the given configuration and constraints in-place by adding snowman heads.

        This method modifies the following attributes of the given frame:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid
        - frame.bonds.N
        - frame.bonds.types
        - frame.bonds.typeid
        - frame.bonds.group

        Adds the snowmen constraints to the dictionary mapping from the bond type to a constraint distance in
        nanometers.

        :param frame:
            The frame to modify.
        :type frame: gsd.hoomd.Frame
        :param constraints:
            The constraints to modify.
        :type constraints: dict[str, float]

        :raises ValueError:
            If a snowman body type is not in the given frame.
            If a snowman head type is already in the given
        """
        if self._snowman_seed is not None and self._snowman_seed > 0:
            npr.seed(self._snowman_seed)
        assert frame.particles.N > 0
        assert frame.particles.position is not None
        assert frame.particles.types is not None
        assert frame.particles.typeid is not None
        if frame.bonds.types is None:
            frame.bonds.types = []
        if frame.bonds.typeid is None:
            frame.bonds.typeid = np.array([], dtype=np.uint32)
        if frame.bonds.group is None:
            frame.bonds.group = np.empty((0, 2), dtype=np.uint32)
        new_constraints = {}
        for body_type, head_type in self._snowman_bond_types.values():
            if body_type not in frame.particles.types:
                raise ValueError(f"Type {body_type} of the snowman bond types dictionary is not in the given frame.")
            if head_type in frame.particles.types:
                raise ValueError(f"Type {head_type} of the snowman bond types dictionary is already in the given "
                                 f"frame.")
            if f"{body_type}-{head_type}" in constraints:
                raise ValueError(f"Constraint for bond type {body_type}-{head_type} already exists.")
            frame.bonds.types.append(f"{body_type}-{head_type}")
            new_constraints[f"{body_type}-{head_type}"] = self._snowman_distances[body_type].value_in_unit(
                self._nanometer)
            bond_index = len(frame.bonds.types) - 1
            frame.particles.types = frame.particles.types + (head_type,)
            head_index = len(frame.particles.types) - 1
            body_type_index = frame.particles.types.index(body_type)

            new_positions = []
            new_type_ids = []
            new_bond_type_ids = []
            new_bond_groups = []

            for i, (body_position, type_index) in enumerate(zip(frame.particles.position, frame.particles.typeid)):
                if type_index != body_type_index:
                    continue
                offset = list(generate_fibonacci_sphere_grid_points(
                    1, self._snowman_distances[body_type].value_in_unit(self._nanometer),
                    self._snowman_seed is None or self._snowman_seed > 0))[0].astype(np.float32)
                assert norm(offset) - self._snowman_distances[body_type].value_in_unit(self._nanometer) < 1.0e-12
                new_positions.append(body_position + offset)
                new_type_ids.append(head_index)
                new_bond_type_ids.append(bond_index)
                new_bond_groups.append([i, frame.particles.N + len(new_positions) - 1])

            new_positions = np.array(new_positions, dtype=np.float32)
            new_type_ids = np.array(new_type_ids, dtype=np.uint32)
            new_bond_type_ids = np.array(new_bond_type_ids, dtype=np.uint32)
            new_bond_groups = np.array(new_bond_groups, dtype=np.uint32)

            frame.particles.N += len(new_positions)
            frame.particles.position = np.concatenate((frame.particles.position, new_positions), axis=0)
            frame.particles.typeid = np.concatenate((frame.particles.typeid, new_type_ids), axis=0)
            frame.bonds.N += len(new_bond_groups)
            frame.bonds.typeid = np.concatenate((frame.bonds.typeid, new_bond_type_ids), axis=0)
            frame.bonds.group = np.concatenate((frame.bonds.group, new_bond_groups), axis=0)

            constraints.update(new_constraints)
