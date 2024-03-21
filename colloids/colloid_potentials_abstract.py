from abc import ABC, abstractmethod
from typing import Iterator
from openmm import CustomNonbondedForce, unit
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters


class ColloidPotentialsAbstract(ABC):
    """
    Abstract class for the steric and electrostatic pair potentials between colloids in a solution with periodic
    boundary conditions using the CustomNonbondedForces class of openmm.

    The potentials are given in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). They should be implemented in the inheriting classes in one or
    several CustomNonbondedForce instances.

    The inheriting classes must implement the add_particle and yield_potentials methods.

    The steric potential from the Alexander-de Gennes polymer brush model between two colloids depends on their radii
    r_1 and r_2. Similarly, the electrostatic potential from DLVO theory between two colloids depends on their radii r_1
    and r_2 and their surface potentials psi_1 and psi_2. Before the finalized potentials are generated via the
    yield_potentials method in order to add them to the openmm system (using the system.addForce method), the
    add_particle method has to be called for each colloid in the system to define its radius and surface potential.

    :param colloid_potentials_parameters:
        The parameters of the steric and electrostatic pair potentials between colloidal particles.
    :type colloid_potentials_parameters: ColloidPotentialsParameters
    """

    def __init__(self, colloid_potentials_parameters: ColloidPotentialsParameters):
        """Constructor of the ColloidPotentialsAbstract class."""
        self._parameters = colloid_potentials_parameters
        self._add_particle_called = False
        self._yield_potentials_called = False

    @abstractmethod
    def add_particle(self, radius: unit.Quantity, surface_potential: unit.Quantity) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        Note that the overriding method in the inheriting class should call this method first because it checks the
        input arguments and that the method yield_potentials was not called before.

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param surface_potential:
            The surface potential of the colloid.
            The unit of the surface_potential must be compatible with millivolts.
        :type surface_potential: unit.Quantity

        :raises TypeError:
            If the radius or surface_potential is not a Quantity with a proper unit.
        :raises ValueError:
            If the radius is not greater than zero.
        :raises RuntimeError:
            If the method yield_potentials was called before this method.
        """
        if not radius.unit.is_compatible(unit.nanometer):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(unit.nanometer) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        if not surface_potential.unit.is_compatible(unit.milli * unit.volt):
            raise TypeError("argument surface_potential must have a unit that is compatible with volts")
        if self._yield_potentials_called:
            raise RuntimeError("method add_particle must be called for every particle in the system before the method "
                               "yield_potentials is used")
        self._add_particle_called = True

    # noinspection PyTypeChecker
    @abstractmethod
    def yield_potentials(self) -> Iterator[CustomNonbondedForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the steric and electrostatic pair
        potentials between colloids in a solution in an openmm system.

        This method has to be called after the method add_particle was called for every particle in the system. Note
        that the overriding method in the inheriting class should call this method first because it checks that the
        method add_particle was called before.

        The generated potentials can be added to the openmm system using the system.addForce method.

        :return:
            A generator that yields all potentials handled by this class.
        :rtype: Iterator[CustomNonbondedForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method.
        """
        if not self._add_particle_called:
            raise RuntimeError("method add_particle must be called for every particle in the system before the method "
                               "yield_potentials is used")
        self._yield_potentials_called = True
