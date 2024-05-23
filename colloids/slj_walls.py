import math
#from typing import Iterator
from openmm import CustomExternalForce, unit
from colloids.abstracts import OpenMMPotentialAbstract 
from colloids.slj_walls_parameters import ShiftedLennardJonesWallsParameters

class ShiftedLennardJonesWalls(OpenMMPotentialAbstract):
 """
    This class sets up the Shifted Lennard Jones potentials for closed-wall simulations using the CustomExternalForce 
    class of openmm.

    The Shifted Lennard Jones potential acts on colloid particles within a certain distance of the wall. This distance 
    depends on the particle radius and the length of the box box_length and is defined as box_length/2 - r_cut - delta, where
    r_cut is radius  * 2^(1/6) and  delta is radius -1. Outside of this range, the external force acting on a particle is 0.
    
    The potential depends on the factors:
      -epsilon, the Lennard Jones potential well-depth
      -sigma, the Lennard Jones potential bond length (set equal to the particle radius)
      -alpha, a cutoff for the shifted Lennard Jones potential that affects the continuinty and differentiability of the 
        potential functional form

    Before the potential is generated in order to add it to the openmm system (using the system.addForce method), the 
    add_particle method has to be called for each colloid particle in the system to define its radius and the x, y, and
    z coordinates of its position.

    :param radius:
        The radius of the the colloid.
        The unit of the radius must be compatible with nanometers and the value must be greater than zero.
    :type radius: unit.Quantity

    :param x:
        The x-position of the the colloid.
        The unit must be compatible with nanometers.
    :type x: unit.Quantity

    :param y:
        The y-position of the the colloid.
        The unit must be compatible with nanometers.
    :type y: unit.Quantity

    :param z:
        The z-position of the the colloid.
        The unit must be compatible with nanometers.
    :type z: unit.Quantity

    :param slj_parameters:
        The parameters of the Shifted Lennard Jones potentials.
        Defaults to the default parameters of the ShiftedLennardJonesWallsParameters class.
    :type slj_parameters: ShiftedLennardJonesWallsParameters
    

    :raises TypeError:
        If the radius is not a Quantity with a proper unit.
    :raises ValueError:
        If the radius is not greater than zero.

    """

  
    def __init__(self, slj_wall_parameters: ShiftedLennardJonesWallsParameters = ShiftedLennardJonesWallsParameters(),
                 #use_log: bool = True) 
                  -> None:
        """Constructor of the ShiftedLennardJonesWalls class."""
        super().__init__(slj_wall_parameters)

        #self._use_log = use_log
        self._slj_potential = self._set_up_slj_potential()
        self._max_radius = -math.inf * self._nanometer
        #self._cutoff_factor = cutoff_factor


    def _set_up_slj_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the shifted Lennard Jones potential."""

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

        slj_potential.addGlobalParameter("box_length", box_length)
        slj_potential.addGlobalParameter("epsilon", epsilon) #*2.477709860209665*unit.kilojoule_per_mole)
        slj_potential.addGlobalParameter("alpha", alpha)
      
        slj_potential.addPerParticleParameter("radius")
      
      return slj_potential
      

    def add_particle(self, radius: unit.Quantity, x: unit.Quantity, y: unit.Quantity, z: unit.Quantity ) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity

        :param x:
            The x coordinate of the colloid particle position.
            The unit must be compatible with nanometers.
        :type x: unit.Quantity

        :param y:
            The y coordinate of the colloid particle position.
            The unit must be compatible with nanometers.
        :type y: unit.Quantity

        :param z:
            The z coordinate of the colloid particle position.
            The unit must be compatible with nanometers.
        :type z: unit.Quantity
        
        :raises TypeError:
            If the radius is not a Quantity with a proper unit (via the abstract base class).
        :raises ValueError:
            If the radius is not greater than zero (via the abstract base class).

        """
        super().add_particle(radius, x, y, z)

        if radius.in_units_of(self._nanometer) > self._max_radius:
            self._max_radius = radius.in_units_of(self._nanometer)

        self._slj_potential.addParticle([radius.value_in_unit(self._nanometer), x, y, z])


if __name__ == '__main__':
    ShiftedLennardJonesWalls()
