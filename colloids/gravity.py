import math
from typing import Iterator
from openmm import CustomExternalForce, unit
from colloids.abstracts import OpenMMPotentialAbstract


class Gravity(OpenMMPotentialAbstract):
    """
    This class sets up a gravitational potential using the CustomExternalForce class of OpenMM.

    The gravitational potential is given by the formula U = m * g * z, where m is the effective mass of the particle,
    g is the gravitational acceleration, and z is the height of the particle above the reference origin [0, 0, 0].
    The effective mass of the spherical colloids is calculated from the effective particle density and the radius of the
    particle. The effective particle density is assumed to be the difference between the particle density and the water
    density.

    :param gravitational_acceleration:
        The acceleration due to gravity.
        The unit must be compatible with meters per second squared and the value must be greater than zero.
    :type gravitational_acceleration: unit.Quantity
    :param water_density:
        The density of water. This is used to compute effective particle density when calculating the gravitational
        potential.
        The units must be compatible with grams per centimeter cubed and the value must be greater than zero.
    :type water_density: unit.Quantity
    :param particle_density:
        The density of the colloidal particles. This is used to compute the effective particle density when calculating
        the gravitational potential.
        The units must be compatible with grams per centimeter cubed and the value must be greater than zero.
    :type particle_density: unit.Quantity
    
    :raises TypeError:
        If the gravitational_acceleration, water_density, or particle_density is not a Quantity with a proper unit.
    :raises ValueError:
        If gravitational_acceleration, water_density, or particle_density is not greater than zero.
    """

    _centimeter = unit.centi * unit.meter
    _nanometer = unit.nano * unit.meter

    def __init__(self, gravitational_acceleration: unit.Quantity, water_density: unit.Quantity,
                 particle_density: unit.Quantity) -> None:
        """Constructor of the Gravity class."""
        super().__init__()

        if not gravitational_acceleration.unit.is_compatible(unit.meter / unit.second ** 2):
            raise TypeError(
                "argument gravitational constant must have a unit that is compatible with meters per second squared")
        if not gravitational_acceleration.value_in_unit(unit.meter / unit.second ** 2) > 0.0:
            raise ValueError("argument gravitational constant must have a value greater than zero")
        if not water_density.unit.is_compatible(unit.gram / self._centimeter ** 3):
            raise TypeError("argument water_density must have a unit compatible with grams per centimeter cubed.")
        if not water_density.value_in_unit(unit.gram / self._centimeter ** 3) > 0.0:
            raise ValueError("argument water_density must have a value greater than zero")
        if not particle_density.unit.is_compatible(unit.gram / self._centimeter ** 3):
            raise TypeError("argument particle_density must have a unit compatible with grams per centimeter cubed.")
        if not particle_density.value_in_unit(unit.gram / self._centimeter ** 3) > 0.0:
            raise ValueError("argument particle_density must have a value greater than zero")

        self._gravitational_acceleration = gravitational_acceleration
        self._water_density = water_density
        self._particle_density = particle_density
        self._gravitational_potential = self._set_up_gravitational_potential()

    def _set_up_gravitational_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the gravitational potential."""
        gravitational_potential = CustomExternalForce("gravitational_acceleration * particle_mass * z")

        gravitational_potential.addGlobalParameter(
            "gravitational_acceleration",
            self._gravitational_acceleration.value_in_unit(self._nanometer / unit.picosecond ** 2))
        gravitational_potential.addPerParticleParameter("particle_mass")

        return gravitational_potential

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
            If the radius is not a Quantity with a proper unit.
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
        self._gravitational_potential.addParticle(
            index,
            [((self._particle_density - self._water_density)
              * 4.0 / 3.0 * math.pi * (radius ** 3) * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.amu)])

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
