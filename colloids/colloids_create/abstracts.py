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
    def generate_configuration(self) -> tuple[Frame, list[tuple[int, int]]]:
        """
        Generate the initial positions of the colloids in a gsd.hoomd.Frame instance together with constraints.

        The generated frame should contain the following attributes:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid
        - frame.configuration.box
        - frame.bonds.N (optionally if bonds are present)
        - frame.bonds.types (optionally if bonds are present)
        - frame.bonds.typeid (optionally if bonds are present)
        - frame.bonds.group (optionally if bonds are present)

        The generated frame should not populate the following attributes:
        - frame.particles.type_shapes
        - frame.particles.diameter
        - frame.particles.charge
        - frame.particles.mass attributes

        This method can generate constraints in addition to the positions of the colloids. The constraints should be
        returned as a dictionary mapping from the bond type to a constraint distance in nanometers.

        :return:
            The initial configuration of the colloids, the constraints.
        :rtype: gsd.hoomd.Frame, dict[str, float]
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
    def modify_configuration(self, frame: Frame, constraints: dict[str, float]) -> None:
        """
        Modify the given configuration and constraints in-place (for instance, by adding a substrate).

        The overloading method should only modify the following attributes of the given frame:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid
        - frame.configuration.box
        - frame.bonds.N (optionally if bonds are present)
        - frame.bonds.types (optionally if bonds are present)
        - frame.bonds.typeid (optionally if bonds are present)
        - frame.bonds.group (optionally if bonds are present)

        The overloading method should not modify the following attributes of the given frame:
        - frame.particles.type_shapes
        - frame.particles.diameter
        - frame.particles.charge
        - frame.particles.mass attributes

        This method should only add (and not overwrite) constraints to the dictionary mapping from the bond type to a
        constraint distance in nanometers.

        :param frame:
            The frame to modify.
        :type frame: gsd.hoomd.Frame
        :param constraints:
            The constraints to modify.
        :type constraints: dict[str, float]
        """
        raise NotImplementedError
