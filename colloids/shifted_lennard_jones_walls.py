from typing import Iterator
from openmm import CustomExternalForce, unit
from colloids.abstracts import OpenMMPotentialAbstract 
import warnings


class ShiftedLennardJonesWalls(OpenMMPotentialAbstract):
    """
    This class sets up the shifted Lennard-Jones potentials for closed-wall simulations using the CustomExternalForce
    class of openmm.

    The shifted Lennard-Jones potential as a wall follows the implementation of hoomd (see
    https://hoomd-blue.readthedocs.io/en/v2.9.4/module-md-wall.html#hoomd.md.wall.slj).

    This class allows to independently switch on walls in the x, y, and z directions. The walls are placed at
    +-box_length/2 for every specified direction.

    The shifted Lennard-Jones potential acts on colloid particles within a certain cutoff distance of every wall. This
    cutoff distance depends on the particle radius and is given by r_cut - delta, where r_cut = radius * 2^(1/6) and
    delta = radius - 1. Outside of this range, the external force acting on a particle is 0.

    The Lennard-Jones potential is shifted so that it starts smoothly at zero at the cutoff distance.

    The shifted Lennard-Jones potential as a function of the distance r to the wall is given by:
    slj(r) = 4 * epsilon * ((radius / (r - delta))^12 - alpha * (radius / (r - delta))^6)
             - 4 * epsilon * ((radius / r_cut)^12 - alpha * (radius / r_cut)^6)
    

    :param box_length:
        The dimensions of the simulation box. This is used to determine the location of the SLJ walls at +-box_length/2.
        The unit of the box_length must be compatible with nanometer and the value must be greater than zero.
    :type box_length: unit.Quantity
    :param epsilon:
        The unshifted Lennard-Jones potential well-depth.
        The unit of the epsilon must be compatible with kilojoules_per_mole and the value must be greater than zero.
    :type epsilon: unit.Quantity
    :param alpha:
        Factor determining the strength of the attractive part of the Lennard-Jones potential.
        This factor has to satisfy 0 <= alpha <= 1.
        Note that the force of this potential is only continuous if alpha = 1.
    :type alpha: float
    :param wall_directions:
        A tuple of three booleans indicating whether the walls in the x, y, and z directions are active.
        Defaults to (True, True, True).
    :type wall_directions: tuple[int]

    :raises TypeError:
        If box_length or epsilon is not a Quantity with a proper unit.
    :raises ValueError:
        If box_length or epsilon is not greater than zero.
        If alpha is not in the interval [0, 1].
        If no wall direction is active.
    """

    _nanometer = unit.nano * unit.meter

    def __init__(self, box_length: unit.Quantity, epsilon: unit.Quantity, alpha: float,
                 wall_directions: tuple[int] = (True, True, True)) -> None:
        """Constructor of the ShiftedLennardJonesWalls class."""
        super().__init__()

        if not box_length.unit.is_compatible(self._nanometer):
            raise TypeError("argument box_length must have a unit that is compatible with nanometers")
        if not box_length.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("argument box_length must have a value greater than zero")
        if not epsilon.unit.is_compatible(unit.kilojoule_per_mole):
            raise TypeError("argument epsilon must have a unit that is compatible with kilojoules_per_mole")
        if not epsilon.value_in_unit(unit.kilojoule_per_mole) > 0.0:
            raise ValueError("argument epsilon must have a value greater than zero")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("argument alpha must satisfy 0 <= alpha <= 1")
        if alpha != 1.0:
            warnings.warn("The force of the shifted Lennard-Jones potential as a wall is only continuous if alpha = 1.")
        if not any(wall_directions):
            raise ValueError("At least one wall direction must be active.")

        self._box_length = box_length
        self._epsilon = epsilon
        self._alpha = alpha
        self._wall_directions = wall_directions
        self._slj_potential = self._set_up_slj_potential()

    def _set_up_slj_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the shifted Lennard Jones potential."""

        # TODO: Use self._wall_directions to switch on walls in the x, y, and z directions.
        slj_potential = CustomExternalForce(
                    "step(abs(x) - (box_length/2 - r_cut - delta)) * ("
                    "4 * epsilon * "
                    "((sigma/(box_length/2 - abs(x) - delta))^12 "
                    "- alpha * (sigma / (box_length/2 - abs(x) - delta))^6)"
                    "-4 * epsilon * "
                    "((sigma/ r_cut)^12 "
                    "- alpha * (sigma / r_cut)^6))"
                    "+step(abs(y) - (box_length/2 - r_cut - delta)) * ("
                    "4 * epsilon * "
                    "((sigma/(box_length/2 - abs(y) - delta))^12 "
                    "- alpha * (sigma / (box_length/2 - abs(y) - delta))^6)"
                    "-4 * epsilon * "
                    "((sigma/ r_cut)^12 "
                    "- alpha * (sigma / r_cut)^6))"
                    "+step(abs(z) - (box_length/2 - r_cut - delta)) * ("
                    "4 * epsilon * "
                    "((sigma/(box_length/2 - abs(z) - delta))^12 "
                    "- alpha * (sigma / (box_length/2 - abs(z) - delta))^6)"
                    "-4 * epsilon * "
                    "((sigma/ r_cut)^12 "
                    "- alpha * (sigma / r_cut)^6));"
                    "sigma = radius;"
                    "delta = radius -1;"
                    "r_cut = radius  * 2^(1/6)"
                )   

        slj_potential.addGlobalParameter("box_length", self._box_length.value_in_unit(self._nanometer))
        slj_potential.addGlobalParameter("epsilon", self._epsilon.value_in_unit(unit.kilojoule_per_mole))
        slj_potential.addGlobalParameter("alpha", self._alpha)
        slj_potential.addPerParticleParameter("radius")
      
        return slj_potential

    def add_particle(self, radius: unit.Quantity) -> None:
        """
        Add a colloid with a given radius to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        
        :raises TypeError:
            If the radius is not a Quantity with a proper unit (via the abstract base class).
        :raises ValueError:
            If the radius is not greater than zero (via the abstract base class).

        """
        super().add_particle()
        if not radius.unit.is_compatible(self._nanometer):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        self._slj_potential.addParticle([radius.value_in_unit(self._nanometer)])

    def yield_potentials(self) -> Iterator[CustomExternalForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the shifted Lennard-Jones walls.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields shifted Lennard-Jones walls handled by this class.
        :rtype: Iterator[CustomExternalForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        yield self._slj_potential
