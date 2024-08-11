import numpy as np
from openmm import unit
import numpy.typing as npt


def substrate_positions_hexagonal(substrate_radius: unit.Quantity,
                                  cell: npt.NDArray[float]) -> npt.NDArray[unit.Quantity]:
    """
    Generate the positions of the substrate particles in a hexagonal lattice at the bottom of the simulation box.

    :param substrate_radius:
        The radius of the substrate particles.
        The unit must be compatible with nanometers and the value must be greater than zero.
    :type substrate_radius: unit.Quantity
    :param cell:
        The box vectors (in nanometers without an explicit unit) of the simulation box.
        The box vectors must be orthogonal.
    :type cell: npt.NDArray[float]

    :return:
        The positions of the substrate particles.
    :rtype: npt.NDArray[unit.Quantity]
    """
    box_vector_one = cell[0]
    box_vector_two = cell[1]
    box_vector_three = cell[2]
    if not (box_vector_one[1] == 0.0 and box_vector_one[2] == 0.0 and
            box_vector_two[0] == 0.0 and box_vector_two[2] == 0.0 and
            box_vector_three[0] == 0.0 and box_vector_three[1] == 0.0):
        raise ValueError("The box vectors must be parallel to the coordinate axes in order to allow for a substrate.")

    # The substrate is a hexagonal lattice of particles within the walls in the x and y directions.
    diameter_substrate = 2.0 * substrate_radius
    # Orthogonal box was already tested above.
    box_length_x = cell[0][0]
    box_length_y = cell[1][1]
    box_length_z = cell[2][2]
    x_spacing_hexagonal_lattice = diameter_substrate.value_in_unit(unit.nano * unit.meter)
    y_spacing_hexagonal_lattice = (3.0 ** 0.5) * diameter_substrate.value_in_unit(unit.nano * unit.meter) / 2.0

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

    return np.array(substrate_positions) * (unit.nano * unit.meter)
