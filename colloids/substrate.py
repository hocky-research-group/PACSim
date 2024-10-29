from enum import auto, Enum
import math
from typing import Iterator
import numpy as np
import numpy.typing as npt
from openmm import CustomExternalForce, unit
from colloids import ColloidPotentialsParameters
from colloids.abstracts import OpenMMPotentialAbstract


class SubstrateType(Enum):
    """
    The possible types of the substrate at the bottom of the simulation box.
    """

    EXPLICIT = auto()
    """
    Substrate particles are explicitly modeled at the bottom of the simulation box.
    """
    IMPLICIT = auto()
    """
    Substrate particles are implicitly modeled at the bottom of the simulation box.
    """

    @staticmethod
    def from_string(string: str) -> "SubstrateType":
        """
        Convert a string to a SubstrateType.

        :param string:
            The string to convert to a SubstrateType.
        :type string: str

        :return:
            The SubstrateType corresponding to the string.
        :rtype: SubstrateType
        """
        return SubstrateType[string.upper()]


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


class ImplicitSubstrate(OpenMMPotentialAbstract):
    """
    This class sets up an implicit layer of substrate particles at the bottom of the simulation box using the
    CustomExternalForce class of OpenMM. This is an alternative to explicitly modeling the substrate using a layer of
    fixed particles, and is designed to reduce computational costs. Either implicit or explicit substrate may be used,
    but not both.

    A substrate can only be used when all SLJ walls are active. The bottom wall in the z direction will be replaced by
    the substrate wall.

    The implicit substrate is modeled as a single substrate particle with a flat surface (that is, with an infinite
    radius) at the bottom of the simulation box. The substrate particle is charged and interacts with the colloidal
    particles via the same steric and electrostatic pair potentials as, e.g., defined in the ColloidPotentialsAlgebraic
    class. An infinite substrate radius implies a value of 2 * radius for the prefactors in the steric and electrostatic
    potentials, where radius is the radius of the colloidal particles.

    :param colloid_potentials_parameters:
        The parameters of the steric and electrostatic pair potentials between colloidal particles.
    :type colloid_potentials_parameters: ColloidPotentialsParameters
    :param wall_distance_z:
        A distance specifying the dimensions of the simulation box in the z direction.
        This is used to determine the location of the substrate wall at -wall_distance/2.
        The unit must be compatible with nanometer and the value must be greater than zero.
    :type wall_distance_z: unit.Quantity
    :param substrate_charge:
        The charge of the implicit substrate particles.
    :type substrate_charge: unit.Quantity
    :param use_log:
        If True, the electrostatic force uses the more accurate equation involving a logarithm [i.e., eq. (12.5.2) in
        Hunter, Foundations of Colloid Science (Oxford University Press, 2001), 2nd edition] instead of the simpler
        equation that only involves an exponential [i.e., eq. (12.5.5) in Hunter, Foundations of Colloid Science
        (Oxford University Press, 2001), 2nd edition].
        Defaults to True.
    :type use_log: bool

    :raises TypeError:
        If the substrate charge for an active substrate wall is not a Quantity with a proper unit.
        If the wall distance for an active substrate wall is not a Quantity with a proper unit.
    :raises ValueError:
        If the wall distance for an active substrate wall is not greater than zero.
    """

    _nanometer = unit.nano * unit.meter

    def __init__(self, colloid_potentials_parameters: ColloidPotentialsParameters, wall_distance_z: unit.Quantity,
                 substrate_charge: unit.Quantity, use_log: bool = False) -> None:
        """Constructor of the ImplicitSubstrate class."""
        super().__init__()

        if not wall_distance_z.unit.is_compatible(self._nanometer):
            raise TypeError("wall distance must have a unit that is compatible with nanometers")
        if not wall_distance_z.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("wall distance must have a value greater than zero")
        if not substrate_charge.unit.is_compatible(unit.milli * unit.volt):
            raise TypeError("substrate charge must have a unit that is compatible with volts")

        self._parameters = colloid_potentials_parameters
        self._substrate_charge = substrate_charge
        self._wall_distance = wall_distance_z
        self._use_log = use_log
        self._substrate_wall_potential = self._set_up_substrate_wall_potential()

    def _set_up_substrate_wall_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the substrate wall."""

        """Set up the basic functional form of the steric potential from the Alexander-de Gennes polymer brush model."""
        # 2 / (1 / radius1 + 1 / radius2) = 2 * radius1 if radius2 = infinity.
        steric_potential = (
            "step(two_l - h) * "
            "steric_prefactor * 2 * radius * brush_length * brush_length * ("
            "28.0 * ((two_l / h)^0.25 - 1.0) "
            "+ 20.0 / 11.0 * (1.0 - (h / two_l)^2.75)"
            "+ 12.0 * (h / two_l - 1.0)) "
        )

        """Set up the basic functional form of the electrostatic potential from DLVO theory."""
        if self._use_log:
            # 2 / (1 / radius1 + 1 / radius2) = 2 * radius1 if radius2 = infinity.
            electrostatic_potential = (
                "electrostatic_prefactor * 2 * radius * psi * substrate_psi * log(1.0 + exp(-h / debye_length));")

        else:
            # 2 / (1 / radius1 + 1 / radius2) = 2 * radius1 if radius2 = infinity.
            electrostatic_potential = (
                "electrostatic_prefactor * 2 * radius * psi * substrate_psi * exp(-h / debye_length);")

        substrate_string = steric_potential + electrostatic_potential
        # +z so that close to zero when z~-L/2.
        # substrate_wall_distance is L / 2 - radius.
        substrate_string += ("h = substrate_wall_distance + z;"
                             "two_l = 2.0 * brush_length;")

        substrate_wall_potential = CustomExternalForce(substrate_string)

        # Steric prefactor is k_B * T * 16 * pi * sigma^(3/2) / 35 (see Hocky paper)
        substrate_wall_potential.addGlobalParameter(
            "steric_prefactor",
            (unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature
             * 16.0 * math.pi * (self._parameters.brush_density ** (3 / 2)) / 35.0
             * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole / (self._nanometer ** 3))
        )
        # Brush length L (see Hocky paper)
        substrate_wall_potential.addGlobalParameter("brush_length",
                                                    self._parameters.brush_length.value_in_unit(self._nanometer))

        # Electrostatic prefactor is 2 * pi * epsilon
        substrate_wall_potential.addGlobalParameter(
            "electrostatic_prefactor",
            (2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
             * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
                unit.kilojoule_per_mole / (self._nanometer * (unit.milli * unit.volt) ** 2)))
        substrate_wall_potential.addGlobalParameter("debye_length",
                                                    self._parameters.debye_length.value_in_unit(self._nanometer))
        substrate_wall_potential.addGlobalParameter("substrate_psi",
                                                    self._substrate_charge.value_in_unit((unit.milli * unit.volt)))

        substrate_wall_potential.addPerParticleParameter("radius")
        # Psi should be given in millivolts.
        substrate_wall_potential.addPerParticleParameter("psi")
        substrate_wall_potential.addPerParticleParameter("substrate_wall_distance")

        return substrate_wall_potential

    def add_particle(self, index: int, radius: unit.Quantity, surface_potential: unit.Quantity) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param index:
            The index of the particle in the OpenMM system.
        :type index: int
        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param surface_potential:
            The surface potential of the colloid.
            The unit of the surface_potential must be compatible with millivolts.
        :type surface_potential: unit.Quantity

        :raises TypeError:
            If the radius or surface potential is not a Quantity with a proper unit.
        :raises ValueError:
            If the radius is not greater than zero.
        :raises RuntimeError:
            If this method is called after the yield_potentials method (via the abstract base class).
        """
        super().add_particle()
        if not radius.unit.is_compatible(self._nanometer):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        if not surface_potential.unit.is_compatible(unit.milli * unit.volt):
            raise TypeError("argument surface_potential must have a unit that is compatible with volts")

        substrate_wall_distance = (self._wall_distance / 2.0 - radius + 1.0 * self._nanometer).value_in_unit(
            self._nanometer)  # TODO: Why shift 1 nm?

        self._substrate_wall_potential.addParticle(index, [radius.value_in_unit(self._nanometer),
                                                           surface_potential.value_in_unit(unit.milli * unit.volt),
                                                           substrate_wall_distance])

    def yield_potentials(self) -> Iterator[CustomExternalForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the substrate wall.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields the substrate wall handled by this class.
        :rtype: Iterator[CustomExternalForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        yield self._substrate_wall_potential
