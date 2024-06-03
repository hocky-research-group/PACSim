from typing import Iterator, Optional, Sequence
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
    +-wall_distance / 2 for every specified direction with its specified wall_distance.

    The shifted Lennard-Jones potential acts on colloid particles within a certain cutoff distance of every wall. This
    cutoff distance depends on the particle radius and is given by r_cut - delta, where r_cut = radius * 2^(1/6) and
    delta = radius - 1. Outside of this range, the external force acting on a particle is 0.

    The Lennard-Jones potential is shifted so that it starts smoothly at zero at the cutoff distance.

    The shifted Lennard-Jones potential as a function of the distance r to the wall is given by:
    slj(r) = 4 * epsilon * ((radius / (r - delta))^12 - alpha * (radius / (r - delta))^6)
             - 4 * epsilon * ((radius / r_cut)^12 - alpha * (radius / r_cut)^6)
    

    :param wall_distances:
        A list of three distances specifying the dimensions of the simulation box in the x, y, and z directions.
        This is used to determine the location of the SLJ walls at +-wall_distance/2 for every active wall direction.
        For any inactive wall direction (see wall_directions parameter), the corresponding wall distance must be None.
        For any active wall direction, the corresponding wall distance must be specified.
        The unit of any wall distance must be compatible with nanometer and the value must be greater than zero.
    :type wall_distances: Sequence[Optional[unit.Quantity]]
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
        A list of three booleans indicating whether the walls in the x, y, and z directions are active.
        Defaults to [False, False, False].
    :type wall_directions: list[bool]

    :raises TypeError:
        If epsilon or any wall distance for an active wall direction is not a Quantity with a proper unit.

    :raises ValueError:
        If epsilon or any wall distance for an active wall direction is not greater than zero.
        If alpha is not in the interval [0, 1].
        If no wall direction is active.
        If not exactly three wall directions are specified.
        If not exactly three wall distances are specified.
        If a wall distance is specified for an inactive wall direction.
        If a wall distance is not specified for an active wall direction.
    """

    _nanometer = unit.nano * unit.meter

    def __init__(self, wall_distances: Sequence[Optional[unit.Quantity]], epsilon: unit.Quantity, alpha: float,
                 wall_directions: Sequence[bool] = (True, True, True)) -> None:
        """Constructor of the ShiftedLennardJonesWalls class."""
        super().__init__()

        if not any(wall_directions):
            raise ValueError("at least one wall direction must be active")
        if len(wall_directions) != 3:
            raise ValueError("wall directions must be specified for three dimensions")
        if len(wall_distances) != 3:
            raise ValueError("wall distances must be specified for three dimensions")
        for wdir, wdist in zip(wall_directions, wall_distances):
            if wdir:
                if wdist is None:
                    raise ValueError("wall distance must be specified for any active wall direction")
                if not wdist.unit.is_compatible(self._nanometer):
                    raise TypeError("any wall distance must have a unit that is compatible with nanometers")
                if not wdist.value_in_unit(self._nanometer) > 0.0:
                    raise ValueError("any wall distance must have a value greater than zero")
            else:
                if wdist is not None:
                    raise ValueError("wall distance must not be specified for inactive wall direction")
        if not epsilon.unit.is_compatible(unit.kilojoule_per_mole):
            raise TypeError("argument epsilon must have a unit that is compatible with kilojoules per mole")
        if not epsilon.value_in_unit(unit.kilojoule_per_mole) > 0.0:
            raise ValueError("argument epsilon must have a value greater than zero")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("argument alpha must satisfy 0 <= alpha <= 1")
        if alpha != 1.0:
            warnings.warn("The force of the shifted Lennard-Jones potential as a wall is only continuous if alpha = 1.")

        self._wall_distances = wall_distances
        self._epsilon = epsilon
        self._alpha = alpha
        self._wall_directions = wall_directions
        self._slj_potential = self._set_up_slj_potential()

    def _set_up_slj_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the shifted Lennard Jones potential."""

        slj_x = ("step(abs(x) - (wall_distance_x / 2.0 - r_cut - delta)) * ("
                 "4.0 * epsilon * "
                 "((sigma / (wall_distance_x / 2.0 - abs(x) - delta))^12 "
                 " - alpha * (sigma / (wall_distance_x / 2.0 - abs(x) - delta))^6)"
                 "- 4.0 * epsilon * "
                 "((sigma / r_cut)^12 "
                 " - alpha * (sigma / r_cut)^6))")
        slj_y = ("step(abs(y) - (wall_distance_y / 2.0 - r_cut - delta)) * ("
                 "4.0 * epsilon * "
                 "((sigma / (wall_distance_y / 2.0 - abs(y) - delta))^12 "
                 " - alpha * (sigma / (wall_distance_y / 2.0 - abs(y) - delta))^6)"
                 "- 4.0 * epsilon * "
                 "((sigma / r_cut)^12 "
                 " - alpha * (sigma / r_cut)^6))")
        slj_z = ("step(abs(z) - (wall_distance_z / 2.0 - r_cut - delta)) * ("
                 "4.0 * epsilon * "
                 "((sigma / (wall_distance_z / 2.0 - abs(z) - delta))^12 "
                 " - alpha * (sigma / (wall_distance_z / 2.0 - abs(z) - delta))^6)"
                 "- 4.0 * epsilon * "
                 "((sigma / r_cut)^12 "
                 " - alpha * (sigma / r_cut)^6))")

        slj_string = "+".join(slj for slj, wdir in zip([slj_x, slj_y, slj_z], self._wall_directions) if wdir)
        assert slj_string
        # 2^(1/6) = 1.122462048309373.
        slj_string += "; sigma = radius; delta = radius - 1.0; r_cut = radius * 1.122462048309373"

        slj_potential = CustomExternalForce(slj_string)
        slj_potential.addGlobalParameter("epsilon", self._epsilon.value_in_unit(unit.kilojoule_per_mole))
        slj_potential.addGlobalParameter("alpha", self._alpha)
        slj_potential.addPerParticleParameter("radius")
        if self._wall_directions[0]:
            slj_potential.addGlobalParameter("wall_distance_x", self._wall_distances[0].value_in_unit(self._nanometer))
        if self._wall_directions[1]:
            slj_potential.addGlobalParameter("wall_distance_y", self._wall_distances[1].value_in_unit(self._nanometer))
        if self._wall_directions[2]:
            slj_potential.addGlobalParameter("wall_distance_z", self._wall_distances[2].value_in_unit(self._nanometer))

        return slj_potential

    def add_particle(self, index: int, radius: unit.Quantity) -> None:
        """
        Add a colloid with a given radius to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param index:
            The index of the particle in the OpenMM system.
        :type index: int
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
        for wall_distance in self._wall_distances:
            if wall_distance is not None:
                if not wall_distance / 2.0 > radius * 2**(1/6) + radius - 1.0 * self._nanometer:
                    raise ValueError("The colloid radius leads to a cutoff radius * 2^(1/6) + radius - 1 in the "
                                     "shifted Lennard-Jones wall that exceeds half of the wall distance.")
        self._slj_potential.addParticle(index, [radius.value_in_unit(self._nanometer)])

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
        super().yield_potentials()
        yield self._slj_potential
