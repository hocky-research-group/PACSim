from abc import ABC, abstractmethod
from gsd.hoomd import Frame


class ConfigurationGenerator(ABC):
    """
    Abstract base class for a generator of an initial configuration in a gsd.hoomd.Frame instance for a colloid
    simulation.
    """

    def __init__(self) -> None:
        """Constructor of the ConfigurationGenerator class."""
        pass

    @abstractmethod
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
        """
        raise NotImplementedError


class ConfigurationModifier(ABC):
    """
    Abstract base class for a modifier of an existing configuration in a gsd.hoomd.Frame instance for a colloid
    simulation.
    """

    def __init__(self) -> None:
        """Constructor of the ConfigurationModifier class."""
        pass

    @abstractmethod
    def modify_configuration(self, frame: Frame) -> None:
        """
        Modify the given configuration and constraints in-place (for instance, by adding a substrate).

        The overloading method should only modify the following attributes of the given frame:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid
        - frame.configuration.box
        - frame.constraints.N (optionally if constraints are present)
        - frame.constraints.value (optionally if constraints are present)
        - frame.constraints.group (optionally if constraints are present)

        The overloading method should not modify the following attributes of the given frame:
        - frame.particles.type_shapes
        - frame.particles.diameter
        - frame.particles.charge
        - frame.particles.mass attributes

        :param frame:
            The frame to modify.
        :type frame: gsd.hoomd.Frame
        """
        raise NotImplementedError
