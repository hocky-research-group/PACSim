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
        #self._max_radius = -math.inf * self._nanometer
        #self._cutoff_factor = cutoff_factor


    def _set_up_slj(self) -> CustomExternalForce:
       """Set up the basic functional form of the shifted Lennard Jones potential."""
      slj_potential = CustomExternalForce(
         "step(-box_length/2 + r_cut + delta, box_length/2 - r_cut - delta) * "
        "4*epsilon*(
          (radius/(x-delta))^12+(radius/(x-delta))^6) +
          (radius/(y-delta))^12+(radius/(y-delta))^6)
          (radius/(z-delta))^12+(radius/(z-delta))^6)
            )
        " 
       "r_cut = radius * 2 **(1/6)"
       "delta = radius - 1"
      )


      slj_potential.addGlobalParameter("epsilon", 1.0)

      slj_potential.addGlobalParameter("box_length",
                                            self._parameters.box_length.value_in_unit(self._nanometer))

      slj_potential.addPerParticleParameter("x")
      slj_potential.addPerParticleParameter("y")
      slj_potential.addPerParticleParameter("z")
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

        self._slj_potential.addParticle([radius.value_in_unit(self._nanometer), x, y, z])
        

    def yield_potentials(self) -> Iterator[CustomNonbondedForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the steric and electrostatic pair
        potentials between colloids in a solution in an openmm system.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields the steric and electrostatic potentials handled by this class.
        :rtype: Iterator[CustomNonbondedForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        assert not math.isinf(self._max_radius.value_in_unit(self._nanometer))

        self._steric_potential.setNonbondedMethod(self._steric_potential.CutoffPeriodic)
        self._steric_potential.setCutoffDistance(
            (2.0 * self._max_radius + 2.0 * self._parameters.brush_length).value_in_unit(self._nanometer))
        self._steric_potential.setUseLongRangeCorrection(False)
        self._steric_potential.setUseSwitchingFunction(False)
        # Set different force groups for steric and electrostatic potentials to allow for different cutoffs on the
        # OpenCL and CUDA platforms.
        self._steric_potential.setForceGroup(0)

        self._electrostatic_potential.setNonbondedMethod(self._electrostatic_potential.CutoffPeriodic)
        self._electrostatic_potential.setCutoffDistance(
            (2.0 * self._max_radius
             + self._cutoff_factor * self._parameters.debye_length).value_in_unit(self._nanometer))
        self._electrostatic_potential.setUseLongRangeCorrection(False)
        self._electrostatic_potential.setUseSwitchingFunction(True)
        self._electrostatic_potential.setSwitchingDistance(
            (2.0 * self._max_radius
             + (self._cutoff_factor - 1.0) * self._parameters.debye_length).value_in_unit(self._nanometer))
        self._electrostatic_potential.setForceGroup(1)

        yield self._steric_potential
        yield self._electrostatic_potential


if __name__ == '__main__':
    ShiftedLennardJonesWalls()
