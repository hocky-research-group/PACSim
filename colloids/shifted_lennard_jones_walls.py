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
        A list of three booleans indicating whether the walls in the x, y, and z directions are active.
        Defaults to [True, True, True].
    :type wall_directions: list[bool]

    :raises TypeError:
        If box_length or epsilon is not a Quantity with a proper unit.
    :raises ValueError:
        If box_length or epsilon is not greater than zero.
        If alpha is not in the interval [0, 1].
        If no wall direction is active.
        If fewer or more than three wall directions are specified.
    """

    _nanometer = unit.nano * unit.meter
    _kilojoule_per_mole = unit.kilojoule / unit.mole

    def __init__(self, box_length: unit.Quantity, epsilon: unit.Quantity, alpha: float,
                 wall_directions: list[bool] = [True, True, True]) -> None:
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
        if len(wall_directions)!=3:
            raise ValueError("Wall directions must be specified for three dimensions.")

        self._box_length = box_length
        self._epsilon = epsilon
        self._alpha = alpha
        self._wall_directions = wall_directions
        self._slj_potential = self._set_up_slj_potential()
    
    
    def _set_up_slj_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the shifted Lennard Jones potential."""

        slj_x = ("step(abs(x) - (box_length/2 - r_cut - delta)) * ("
                    "\n 4 * epsilon * "
                    "\n ((sigma/(box_length/2 - abs(x) - delta))^12 "
                    "\n - alpha * (sigma / (box_length/2 - abs(x) - delta))^6)"
                    "\n -4 * epsilon * "
                    "\n ((sigma/ r_cut)^12 "
                    "\n - alpha * (sigma / r_cut)^6))")
        slj_y = ( "+step(abs(y) - (box_length/2 - r_cut - delta)) * ("
                            "\n 4 * epsilon * "
                            "\n ((sigma/(box_length/2 - abs(y) - delta))^12 "
                            "\n - alpha * (sigma / (box_length/2 - abs(y) - delta))^6)"
                            "\n -4 * epsilon * "
                            "\n ((sigma/ r_cut)^12 "
                            "\n - alpha * (sigma / r_cut)^6))")
        slj_z = ( "+step(abs(z) - (box_length/2 - r_cut - delta)) * ("
                            "\n 4 * epsilon * "
                            "\n ((sigma/(box_length/2 - abs(z) - delta))^12 "
                            "\n - alpha * (sigma / (box_length/2 - abs(z) - delta))^6)"
                            "\n -4 * epsilon * "
                            "\n ((sigma/ r_cut)^12 "
                            "\n - alpha * (sigma / r_cut)^6));")

        walls = [slj_x, slj_y, slj_z]

        #Use wall_directions to selectively turn on walls in x, y, and z directions
        
        walls_dict = dict({walls[i]: self._wall_directions[i] for i in range(3)})

        slj_force = []

        for key, val in walls_dict.items():
            if val==True:
                if len(slj_force)==0: 
                    slj_force.append(key)
                else: 
                    slj_force.append(str("+" + key))
            
                
        var_defs = ["delta = radius_negative -1;",
                            "r_cut = radius_negative * 2^(1/6)"]

        for i in var_defs:
            slj_force.append(i)
            
        slj_force=tuple(slj_force)
                
        slj_str = "\n".join(slj_force)

        slj_potential = CustomExternalForce(slj_str)

        slj_potential.addGlobalParameter("box_length", self._box_length.value_in_unit(self._nanometer))
        slj_potential.addGlobalParameter("epsilon", self._epsilon.value_in_unit(self._kilojoule_per_mole))
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
