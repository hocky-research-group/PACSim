import math
#from typing import Iterator
from openmm import CustomExternalForce, unit
from colloids.abstracts import OpenMMPotentialAbstract 
from colloids.helper_functions import read_xyz_file
from colloids.run_parameters import RunParameters


class ShiftedLennardJonesWallsParameters(object):
    """
    This class stores the parameters to compute the shifted Lennard Jones potential forces for closed-wall simulations .

    :raises TypeError:
        If the is not a Quantity with a proper unit.
    :raises ValueError:
        If the is not greater than zero.
    
    """

    def __init__(self):
        """Constructor of the ShiftedLennardJonesWallsParameters class."""

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file", help="YAML file with simulation parameters", type=str)
  
    args = parser.parse_args()

    if not args.yaml_file.endswith(".yaml"):
        raise ValueError("The YAML file must have the .yaml extension.")

    parameters = RunParameters.from_yaml(args.yaml_file)
    parameters.check_types_of_initial_configuration()

    box_length = read_xyz_file(parameters.initial_configuration)[2][0]
    #epsilon = 
    #alpha = 

    if __name__ == '__main__':
        parameters = ShiftedLennardJonesWallsParameters()
        print(parameters)


class ShiftedLennardJonesWalls(OpenMMPotentialAbstract):
####TODO: add documentation
  
    def __init__(self, slj_wall_parameters: ShiftedLennardJonesWallsParameters = ShiftedLennardJonesWallsParameters(),
                 use_log: bool = True) -> None:
        """Constructor of the ShiftedLennardJonesWalls class."""
        super().__init__(slj_wall_parameters)

        self._use_log = use_log
        self._slj_potential = self._set_up_slj_potential()
        self._max_radius = -math.inf * self._nanometer
        #self._cutoff_factor = cutoff_factor


    def _set_up_slj_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the shifted Lennard Jones potential."""

        slj_potential_x = CustomExternalForce(
                    "step(abs(x) - (box_length/2 - r_cut - delta)) * ("
                    "4 * epsilon * "
                    "((sigma/(box_length/2 - abs(x) - delta))^12 "
                    "- alpha * (sigma / (box_length/2 - abs(x) - delta))^6)"
                    "-4 * epsilon * "
                    "((sigma/ r_cut)^12 "
                    "- alpha * (sigma / r_cut)^6));"
                    "delta = radius -1;"
                    "r_cut = radius * 2^(1/6)"
                )
        
        slj_potential_y = CustomExternalForce(
                    "step(abs(y) - (box_length/2 - r_cut - delta)) * ("
                    "4 * epsilon * "
                    "((sigma/(box_length/2 - abs(y) - delta))^12 "
                    "- alpha * (sigma / (box_length/2 - abs(y) - delta))^6)"
                    "-4 * epsilon * "
                    "((sigma/ r_cut)^12 "
                    "- alpha * (sigma / r_cut)^6));"
                    "delta = radius -1;"
                    "r_cut = radius * 2^(1/6)"
                )
        
        slj_potential_z = CustomExternalForce(
                    "step(abs(z) - (box_length/2 - r_cut - delta)) * ("
                    "4 * epsilon * "
                    "((sigma/(box_length/2 - abs(z) - delta))^12 "
                    "- alpha * (sigma / (box_length/2 - abs(z) - delta))^6)"
                    "-4 * epsilon * "
                    "((sigma/ r_cut)^12 "
                    "- alpha * (sigma / r_cut)^6));"
                    "sigma = radius;"
                    "delta = radius -1;"
                    "r_cut = radius * 2^(1/6)"
                )
    
        slj_potential_x.addGlobalParameter("box_length", box_length)
        slj_potential_x.addGlobalParameter("epsilon", epsilon*2.477709860209665*unit.kilojoule_per_mole)
        slj_potential_x.addGlobalParameter("alpha", alpha)
      
        slj_potential_x.addPerParticleParameter("radius")

        slj_potential_y.addGlobalParameter("box_length", box_length)
        slj_potential_y.addGlobalParameter("epsilon", epsilon*2.477709860209665*unit.kilojoule_per_mole)
        slj_potential_y.addGlobalParameter("alpha", alpha)
      
        slj_potential_y.addPerParticleParameter("radius")

        slj_potential_z.addGlobalParameter("box_length", box_length)
        slj_potential_z.addGlobalParameter("epsilon", epsilon*2.477709860209665*unit.kilojoule_per_mole)
        slj_potential_z.addGlobalParameter("alpha", alpha)
      
        slj_potential_z.addPerParticleParameter("radius")
      
      return slj_potential_x, slj_potential_y, slj_potential_z
      

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
            The unit must be compatible with nanometers and the value must be greater than zero.
        :type x: unit.Quantity

        :param y:
            The y coordinate of the colloid particle position.
            The unit must be compatible with nanometers and the value must be greater than zero.
        :type y: unit.Quantity

        :param z:
            The z coordinate of the colloid particle position.
            The unit must be compatible with nanometers and the value must be greater than zero.
        :type z: unit.Quantity
        
        :raises TypeError:
            If the radius is not a Quantity with a proper unit (via the abstract base class).
        :raises ValueError:
            If the radius is not greater than zero (via the abstract base class).
        :raises RuntimeError:
            If the method yield_potentials was called before this method (via the abstract base class).
        """
        super().add_particle(radius, x, y, z)

        if radius.in_units_of(self._nanometer) > self._max_radius:
            self._max_radius = radius.in_units_of(self._nanometer)

        self._slj_potential_x.addParticle([radius.value_in_unit(self._nanometer), x, y, z])
        self._slj_potential_y.addParticle([radius.value_in_unit(self._nanometer), x, y, z])
        self._slj_potential_z.addParticle([radius.value_in_unit(self._nanometer), x, y, z])


if __name__ == '__main__':
    ShiftedLennardJonesWalls()
