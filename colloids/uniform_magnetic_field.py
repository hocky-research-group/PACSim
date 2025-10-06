import math
from typing import Iterator
from networkx import radius
from openmm import CustomExternalForce, unit
from colloids.abstracts import OpenMMPotentialAbstract
from colloids.units import energy_unit, length_unit


class UniformMagneticField(OpenMMPotentialAbstract):
    """
    This class sets up a uniform magnetic field potential using the CustomExternalForce class of OpenMM.

    The magnetic field potential is given by the formula U = -q * B * z, where q is the effective charge of the particle,
    B is the magnetic field strength, and z is the height of the particle above the reference origin [0, 0, 0].
    The effective mass of the spherical colloids is calculated from the effective particle density and the radius of the
    particle. The effective particle density is assumed to be the difference between the particle density and the water
    density.

    :param magnetic_field_strength:
        The strength of the magnetic field.
        The unit must be compatible with tesla and the value must be greater than zero.
    :type magnetic_field_strength: unit.Quantity
    :param periodic_boundary_conditions:
        Whether periodic boundary conditions are applied in the simulation.
        This is important to correctly handle the potential when a particle crosses the periodic boundary.
    :type periodic_boundary_conditions: bool

    :raises TypeError:
        If the magnetic_field_strength is not a Quantity with a proper unit.
    :raises ValueError:
        If magnetic_field_strength is not greater than zero.
    """

    _force_unit = energy_unit / length_unit

    def __init__(self, magnetic_field_strength: unit.Quantity, periodic_boundary_conditions: bool) -> None:
        """Constructor of the UniformMagneticField class."""
        super().__init__()

        if not magnetic_field_strength.unit.is_compatible(self._force_unit):
            raise TypeError(
                "argument magnetic_field_strength must have a unit that is compatible with volts per meter")
        if not magnetic_field_strength.value_in_unit(self._force_unit) > 0.0:
            raise ValueError("argument magnetic_field_strength must have a value greater than zero")
        
        self._magnetic_field_strength = magnetic_field_strength
        self._periodic_boundary_conditions = periodic_boundary_conditions
        self._magnetic_field_potential = self._set_up_magnetic_field_potential()

    def _set_up_magnetic_field_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the magnetic field potential."""
        magnetic_field_potential = CustomExternalForce("-particle_magnetic_moment * magnetic_field_strength * x")
        magnetic_field_potential.addGlobalParameter(
            "magnetic_field_strength",
            self._magnetic_field_strength.value_in_unit(self._force_unit))
        magnetic_field_potential.addPerParticleParameter("particle_magnetic_moment")
        print(f"Setting magnetic field strength to {self._magnetic_field_strength.value_in_unit(self._force_unit)}")

        return magnetic_field_potential

    def add_particle(self, index: int, magnetic_moment: unit.Quantity) -> None:
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
        if not isinstance(magnetic_moment, float):
            raise TypeError("argument magnetic_moment must be of type float")

        self._magnetic_field_potential.addParticle(
            index,
            [magnetic_moment]
        )

    def yield_potentials(self) -> Iterator[CustomExternalForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the magnetic field potential.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields the magnetic field potential handled by this class.
        :rtype: Iterator[CustomExternalForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        yield self._magnetic_field_potential
