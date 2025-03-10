from gsd.hoomd import Frame
import numpy as np
import numpy.typing as npt
from openmm import unit
from colloids.colloids_create import ConfigurationModifier
from colloids.units import length_unit


class SubstrateModifier(ConfigurationModifier):
    """
    Modifier of an existing configuration in a gsd.hoomd.Frame instance for a colloid simulation that adds a substrate
    in a hexagonal pattern at the bottom of the simulation box.

    This class can only modify configurations that have orthogonal box vectors.

    :param substrate_radius:
        The radius of the substrate particles.
        The unit must be compatible with nanometers and the value must be greater than zero.
    :type substrate_radius: unit.Quantity

    :raises TypeError:
        If the substrate_radius is not a Quantity with a proper unit.
    :raises ValueError:
        If the substrate_radius is not greater than zero.
    """

    def __init__(self, substrate_radius: unit.Quantity, substrate_type: str) -> None:
        """Constructor of the SubstrateModifier class."""
        super().__init__()
        if not substrate_radius.unit.is_compatible(length_unit):
            raise TypeError("The substrate radius must have a unit that is compatible with nanometers.")
        if not substrate_radius > 0.0 * length_unit:
            raise ValueError("The substrate radius must have a value greater than zero.")
        self._substrate_radius = substrate_radius
        self._substrate_type = substrate_type

    @staticmethod
    def _generate_substrate_positions_hexagonal(substrate_radius: float,
                                                box: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Generate the positions of the substrate particles in a hexagonal lattice at the bottom of the simulation box.

        :param substrate_radius:
            The radius of the substrate particles (in nanometers without an explicit unit).
        :type substrate_radius: unit.Quantity
        :param box:
            The box lengths (in nanometers without an explicit unit) and tilt factors (dimensionless) of the simulation
            box.
        :type box: npt.NDArray[float]

        :return:
            The positions of the substrate particles.
        :rtype: npt.NDArray[unit.Quantity]

        :raises ValueError:
            If the box is not a one-dimensional array with six elements.
            If the box vectors are not orthogonal.
        """
        if not box.shape == (6,):
            raise ValueError("The box must be a one-dimensional array with six elements.")
        if not (box[3] == 0.0 and box[4] == 0.0 and box[5] == 0.0):
            raise ValueError("The box vectors must be orthogonal (all tilt factors zero) in order to allow for a "
                             "substrate.")

        # The substrate is a hexagonal lattice of particles within the walls in the x and y directions.
        diameter_substrate = 2.0 * substrate_radius
        # Orthogonal box was already tested above.
        box_length_x = box[0]
        box_length_y = box[1]
        box_length_z = box[2]
        x_spacing_hexagonal_lattice = diameter_substrate
        y_spacing_hexagonal_lattice = (3.0 ** 0.5) * diameter_substrate / 2.0

        number_substrate_x = box_length_x // x_spacing_hexagonal_lattice
        shift_x = (box_length_x - number_substrate_x * x_spacing_hexagonal_lattice) / 2.0
        assert 0.0 <= shift_x < x_spacing_hexagonal_lattice / 2.0

        number_substrate_y = box_length_y // y_spacing_hexagonal_lattice
        shift_y = (box_length_y - number_substrate_y * y_spacing_hexagonal_lattice) / 2.0
        assert 0.0 <= shift_y < y_spacing_hexagonal_lattice / 2.0

        # The x coordinates in the first row.
        x_one_positions = np.linspace(-box_length_x / 2.0 + x_spacing_hexagonal_lattice / 2.0 + shift_x,
                                      box_length_x / 2.0 - x_spacing_hexagonal_lattice / 2.0 - shift_x,
                                      num=int(number_substrate_x))
        assert abs(x_one_positions[1] - x_one_positions[0] - x_spacing_hexagonal_lattice) < 1.0e-10
        # The x coordinates in the second row that are shifted by the radius of the substrate particles.
        # The number of particles in the second row has to be one less than in the first row because shift_x is smaller
        # than the radius of the substrate particles.
        x_two_positions = np.linspace(-box_length_x / 2.0 + x_spacing_hexagonal_lattice + shift_x,
                                      box_length_x / 2.0 - x_spacing_hexagonal_lattice - shift_x,
                                      num=int(number_substrate_x - 1))
        assert abs(x_two_positions[1] - x_two_positions[0] - x_spacing_hexagonal_lattice) < 1.0e-10

        # The y coordinates in the different rows.
        y_positions = np.linspace(-box_length_y / 2.0 + y_spacing_hexagonal_lattice / 2.0 + shift_y,
                                  box_length_y / 2.0 - y_spacing_hexagonal_lattice / 2.0 - shift_y,
                                  num=int(number_substrate_y))
        assert abs(y_positions[1] - y_positions[0] - y_spacing_hexagonal_lattice) < 1.0e-10

        substrate_positions = []
        for y_index, y_position in enumerate(y_positions):
            x_positions = x_one_positions if y_index % 2 == 0 else x_two_positions
            for x_position in x_positions:
                substrate_positions.append(np.array([x_position, y_position, -box_length_z / 2.0]))

        return np.array(substrate_positions)

    def modify_configuration(self, frame: Frame) -> None:
        """
        Modify the given configuration and constraints in-place by adding a substrate.

        This method modifies the following attributes of the given frame:
        - frame.particles.N
        - frame.particles.position
        - frame.particles.types
        - frame.particles.typeid

        :param frame:
            The frame to modify.
        :type frame: gsd.hoomd.Frame

        :raises ValueError:
            If the substrate type is already in the given frame.
        """
        assert frame.particles.N > 0
        assert frame.particles.position is not None
        assert frame.particles.types is not None
        assert frame.particles.typeid is not None
        if self._substrate_type in frame.particles.types:
            raise ValueError(f"The substrate type {self._substrate_type} is already in the given frame.")
        substrate_positions = self._generate_substrate_positions_hexagonal(
            self._substrate_radius.value_in_unit(length_unit), frame.configuration.box).astype(np.float32)

        frame.particles.types = frame.particles.types + (self._substrate_type,)
        assert frame.particles.types.index(self._substrate_type) not in frame.particles.typeid
        substrate_index = len(frame.particles.types) - 1
        frame.particles.N += len(substrate_positions)
        frame.particles.position = np.concatenate((frame.particles.position, substrate_positions), axis=0)
        frame.particles.typeid = np.concatenate((frame.particles.typeid,
                                                 np.full(len(substrate_positions), substrate_index)))
