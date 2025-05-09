from math import sqrt
from random import uniform
from gsd.hoomd import Frame
import numpy as np
from openmm import unit
from colloids.colloids_create import ConfigurationModifier
from colloids.units import electric_potential_unit, length_unit


class RandomSnowmanHeadsModifier(ConfigurationModifier):
    """
    TODO
    """

    def __init__(self, snowman_body_type: str, snowman_head_type: str, cosine_theta: float, lower_radius: unit.Quantity,
                 upper_radius: unit.Quantity, surface_potential: unit.Quantity) -> None:
        """Constructor of the SubstrateModifier class."""
        super().__init__()
        if not -1.0 <= cosine_theta <= 1.0:
            raise ValueError("The cosine of the angle theta must be between -1.0 and 1.0.")
        if not lower_radius.unit.is_compatible(length_unit):
            raise TypeError("The lower radius must have a unit that is compatible with nanometers.")
        if not lower_radius.value_in_unit(length_unit) > 0.0:
            raise ValueError("The lower radius must have a value greater than zero.")
        if not upper_radius.unit.is_compatible(length_unit):
            raise TypeError("The upper radius must have a unit that is compatible with nanometers.")
        if not upper_radius.value_in_unit(length_unit) > 0.0:
            raise ValueError("The upper radius must have a value greater than zero.")
        if not surface_potential.unit.is_compatible(electric_potential_unit):
            raise TypeError("The surface potential must have a unit that is compatible with volts.")
        self._snowman_body_type = snowman_body_type
        self._snowman_head_type = snowman_head_type
        self._cosine_theta = cosine_theta
        self._lower_radius = lower_radius.value_in_unit(length_unit)
        self._upper_radius = upper_radius.value_in_unit(length_unit)
        self._surface_potential = surface_potential.value_in_unit(electric_potential_unit)

    def modify_configuration(self, frame: Frame) -> None:
        """
        Modify the given configuration and constraints in-place by TODO.

        This method modifies the following attributes of the given frame:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid

        :param frame:
            The frame to modify.
        :type frame: gsd.hoomd.Frame
        """
        assert frame.particles.N > 0
        assert frame.particles.position is not None
        assert frame.particles.types is not None
        assert frame.particles.typeid is not None
        assert frame.particles.mass is not None
        assert frame.particles.charge is not None
        assert frame.particles.diameter is not None

        if self._snowman_body_type not in frame.particles.types:
            raise ValueError("The snowman body type must be in the frame.")
        if self._snowman_head_type in frame.particles.types:
            raise ValueError(f"The snowman head type {self._snowman_head_type} is already in the given frame.")

        head_positions = []
        head_diameters = []
        head_masses = []
        head_charges = []
        distances = []
        constraints = []
        assert (len(frame.particles.typeid) == len(frame.particles.position) == len(frame.particles.diameter)
                == len(frame.particles.mass) == len(frame.particles.charge) == frame.particles.N)
        for index, (typeid, body_position, body_diameter, body_mass) in enumerate(zip(frame.particles.typeid,
                                                                                      frame.particles.position,
                                                                                      frame.particles.diameter,
                                                                                      frame.particles.mass)):
            if frame.particles.types[typeid] == self._snowman_body_type:
                body_radius = body_diameter / 2.0
                head_radius = uniform(self._lower_radius, self._upper_radius)
                distance = sqrt(body_radius * body_radius + head_radius * head_radius
                                - 2.0 * body_radius * head_radius * self._cosine_theta)
                head_positions.append([body_position[0] + distance, body_position[1], body_position[2]])
                head_diameters.append(2.0 * head_radius)
                head_masses.append(body_mass / ((body_radius / head_radius) ** 3))
                head_charges.append(self._surface_potential)
                distances.append(distance)
                constraints.append([index, len(frame.particles.position) + len(head_positions) - 1])

        frame.particles.types = frame.particles.types + (self._snowman_head_type,)
        assert frame.particles.types.index(self._snowman_head_type) not in frame.particles.typeid
        snowman_head_index = len(frame.particles.types) - 1
        frame.particles.N += len(head_positions)
        frame.particles.position = np.concatenate((frame.particles.position, np.array(head_positions)), axis=0)
        frame.particles.typeid = np.concatenate((frame.particles.typeid,
                                                 np.full(len(head_positions), snowman_head_index)), axis=0)
        frame.particles.mass = np.concatenate(
            (frame.particles.mass, np.array(head_masses, dtype=np.float32)), axis=0)
        frame.particles.charge = np.concatenate(
            (frame.particles.charge, np.array(head_charges, dtype=np.float32)), axis=0)
        frame.particles.diameter = np.concatenate(
            (frame.particles.diameter, np.array(head_diameters, dtype=np.float32)), axis=0)

        if frame.constraints.N == 0:
            frame.constraints.N = len(constraints)
            frame.constraints.group = np.array(constraints, dtype=np.uint32)
            frame.constraints.value = np.array(distances, dtype=np.float32)
            assert frame.bonds.N == 0
            frame.bonds.N = len(constraints)
            frame.bonds.types = ["s"]
            frame.bonds.typeid = np.zeros(frame.bonds.N, dtype=np.uint32)
            frame.bonds.group = np.array(constraints, dtype=np.uint32)
        else:
            frame.constraints.N += len(constraints)
            frame.constraints.group = np.concatenate((frame.constraints.group,
                                                      np.array(constraints, dtype=np.uint32)),
                                                     axis=0)
            frame.constraints.value = np.concatenate((frame.constraints.value,
                                                      np.array(distances, dtype=np.float32)),
                                                     axis=0)
            assert frame.bonds.N == frame.constraints.N
            frame.bonds.N += len(constraints)
            assert "s" not in frame.bonds.types
            frame.bonds.types = frame.bonds.types + ["s"]
            frame.bonds.typeid = np.concatenate((frame.bonds.typeid,
                                                 np.full(len(constraints), len(frame.bonds.types) - 1)), axis=0)

            frame.bonds.group = np.concatenate((frame.bonds.group, np.array(constraints, dtype=np.uint32)),
                                               axis=0)
