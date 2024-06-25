from typing import Iterator, Optional, Sequence
from openmm import CustomExternalForce, unit
from colloids.abstracts import OpenMMPotentialAbstract
import warnings
import numpy as np


class Gravity(OpenMMPotentialAbstract):
    """
    This class sets up a gravitational potential  using the CustomExternalForce class of openmm.

    :param gravitational_constant: 
        The acceleration due to gravity.
        The unit must be compatible with meters per second squared.
    :type gravitational_constant: unit.Quantity
    :param water_density:
        The density of water. This is used to compute relative particle densities when calculating gravitational force.
        The units must be compatible with grams per centimeter cubed.
    :type water_density: unit.Quantity
    
    :raises TypeError:
        If gravitational_constant or water_density is not a Quantity with a proper unit.

    :raises ValueError:
        If water_density is not greater than zero.
    """

    _nanometer = unit.nano * unit.meter

    def __init__(self, gravitational_constant: unit.Quantity, water_density: unit.Quantity):

        """Constructor of the Gravity class."""
        super().__init__()

        if not gravitational_constant.unit.is_compatible(unit.meter/unit.second**2):
            raise TypeError("argument gravitational constant must have a unit that is compatible with meters per second squared")
        #if not particle_density.unit.is_compatible(unit.gram/unit.centimeter**3):
         #   raise TypeError("argument particle_density must have a unit compatible with grams per centimeter cubed.")
        #if not particle_density.value_in_unit(unit.gram/unit.centimeter**3) > 0.0:
        #    raise ValueError("argument particle_density must have a value greater than zero")
        if not water_density.unit.is_compatible(unit.gram/unit.centimeter**3):
            raise TypeError("argument water_density must have a unit compatible with grams per centimeter cubed.")
        if not water_density.value_in_unit(unit.gram/unit.centimeter**3) > 0.0:
            raise ValueError("argument water_density must have a value greater than zero")

        self._gravitational_constant = gravitational_constant
        #self._particle_density = particle_density
        self._water_density = water_density
      
        self._gravitational_potential = self._set_up_gravitational_potential()

    def _set_up_gravitational_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the gravitational potential."""

        pi = np.pi

        u_grav = CustomExternalForce(
                "(gravitational_constant * particle_mass * z);"
                "density_difference = particle_density - water_density;"
                "particle_mass = (particle_density - water_density) * 4/3 * pi * radius^3;"
            )

        gravitational_potential = CustomExternalForce(u_grav)
        gravitational_potential.addGlobalParameter("gravitational_constant", self._gravitational_constant.value_in_unit(unit.meter/unit.second**2))
        gravitational_potential.addGlobalParameter("pi", pi)
        #gravitational_potential.addGlobalParameter("particle_density", self._particle_density.value_in_unit(unit.gram/unit.centimer**3))
        gravitational_potential.addGlobalParameter("water_density", self._water_density.value_in_unit(unit.gram/unit.centimeter**3))
        
        gravitational_potential.addPerParticleParameter("radius")
        gravitational_potential.addPerParticleParameter("particle_density")
    
        return gravitational_potential

    def add_particle(self, index: int, radius: unit.Quantity, particle_density: unit.Quantity) -> None:
        """
        Add a colloid with a given radius and particle density to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param index:
            The index of the particle in the OpenMM system.
        :type index: int
        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param particle_density:
        :type particle_density: unit.Quantity
        
        :raises TypeError:
            If the radius or particle density is not a Quantity with a proper unit (via the abstract base class).
        :raises ValueError:
            If the radius or particle density is not greater than zero (via the abstract base class).

        """
        super().add_particle()
        if not radius.unit.is_compatible(self._nanometer):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        if not particle_density.unit.is_compatible(unit.gram/unit.centimeter**3):
            raise TypeError("argument particle_density must have a unit compatible with grams per centimeter cubed.")
        if not particle_density.value_in_unit(unit.gram/unit.centimeter**3) > 0.0:
            raise ValueError("argument particle_density must have a value greater than zero")

        self._gravitational_potential.addParticle(index, [radius.value_in_unit(self._nanometer), particle_density.value_in_unit(unit.gram/unit.centimeter**3)])

    def yield_potentials(self) -> Iterator[CustomExternalForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the gravitational potential.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields the gravitational potential handled by this class.
        :rtype: Iterator[CustomExternalForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        yield self._gravitational_potential
