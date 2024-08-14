import math
from typing import Iterator
import numpy as np
import numpy.typing as npt
from openmm import Continuous1DFunction, CustomNonbondedForce, unit
from colloids.abstracts import ColloidPotentialsAbstract
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters


class ColloidPotentialsTabulated(ColloidPotentialsAbstract):
    """
    This class sets up the steric and electrostatic pair potentials between colloids in a solution using the
    CustomNonbondedForces class of openmm with tabulated functions.

    The potentials are given in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). Any references to equations or symbols in the code refer to this
    paper.

    The steric potential from the Alexander-de Gennes polymer brush model between two colloids depends on their radii
    r_1 and r_2. Similarly, the electrostatic potential from DLVO theory between two colloids depends on their radii r_1
    and r_2 and their surface potentials psi_1 and psi_2. Before the finalized potentials are generated via the
    yield_potentials method in order to add them to the openmm system (using the system.addForce method), the
    add_particle method has to be called for each colloid in the system to define its radius and surface potential.

    This class assumes that there are only two types of colloids in the system. It defines three CustomNonbondedForce
    instances containing both the steric and electrostatic potentials for the pair potentials between (i) two colloids
    of the first type, (ii) two colloids of the second type, and (iii) between a colloid of the first type and a colloid
    of the second type. It requires the radii and surface potentials of the two types of colloids on initialization.

    The potential of every CustomNonbondedForce instance has a cutoff at a surface-to-surface separation of
    cutoff_factor * debye_length between the involved types of colloids. Here, debye_length is the Debye screening
    length that is stored in the ColloidPotentialsParameters instance, and cutoff_factor is set on initialization. A
    switching function reduces the interaction at surface-to-surface separations larger than
    (cutoff_factor - 1) * debye_length to make the potential and forces go smoothly to 0 at the cutoff distance.

    The cutoffs can be set to be periodic or non-periodic.

    Note that the steric potential from the Alexander-de Gennes polymer brush model uses the mixing rule
    r = r_1 + r_2 / 2.0 for the prefactor [see eq. (1)], whereas the electrostatic potential from DLVO theory uses
    r = 2.0 / (1.0 / r_1 + 1.0 / r_2) for the prefactor.

    :param radius_one:
        The radius of the first type of colloid.
        The unit of the radius_one must be compatible with nanometers and the value must be greater than zero.
    :type radius_one: unit.Quantity
    :param radius_two:
        The radius of the second type of colloid.
        The unit of the radius_two must be compatible with nanometers and the value must be greater than zero.
    :type radius_two: unit.Quantity
    :param surface_potential_one:
        The surface potential of the first type of colloid.
        The unit of the surface_potential_one must be compatible with millivolts.
    :type surface_potential_one: unit.Quantity
    :param surface_potential_two:
        The surface potential of the second type of colloid.
        The unit of the surface_potential_two must be compatible with millivolts.
    :param colloid_potentials_parameters:
        The parameters of the steric and electrostatic pair potentials between colloidal particles.
        Defaults to the default parameters of the ColloidPotentialsParameters class.
    :type colloid_potentials_parameters: ColloidPotentialsParameters
    :param use_log:
        If True, the electrostatic force uses the more accurate equation involving a logarithm [i.e., eq. (12.5.2) in
        Hunter, Foundations of Colloid Science (Oxford University Press, 2001), 2nd edition] instead of the simpler
        equation that only involves an exponential [i.e., eq. (12.5.5) in Hunter, Foundations of Colloid Science
        (Oxford University Press, 2001), 2nd edition].
        Defaults to True.
    :type use_log: bool
    :param cutoff_factor:
        The factor by which the Debye length is multiplied to get the cutoff distance of the forces.
        Defaults to 21.0.
    :type cutoff_factor: float
    :param periodic_boundary_conditions:
        Whether this force should use periodic cutoffs for the steric and electrostatic potentials.
    :type periodic_boundary_conditions: bool

    :raises TypeError:
        If the radius_one, radius_two, surface_potential_one, or surface_potential_two is not a Quantity with a proper
        unit.
    :raises ValueError:
        If the radius_one or radius_two is not greater than zero.
    :raises ValueError:
        If the cutoff factor is not greater than zero.
    """

    def __init__(self, radius_one: unit.Quantity, radius_two: unit.Quantity,
                 surface_potential_one: unit.Quantity, surface_potential_two: unit.Quantity,
                 colloid_potentials_parameters: ColloidPotentialsParameters = ColloidPotentialsParameters(),
                 use_log: bool = True, cutoff_factor: float = 21.0, periodic_boundary_conditions: bool = True) -> None:
        """Constructor of the ColloidPotentialsTabulated class."""
        super().__init__(colloid_potentials_parameters, periodic_boundary_conditions)
        if not cutoff_factor > 0.0:
            raise ValueError("The cutoff factor must be greater than zero.")
        if not radius_one.unit.is_compatible(self._nanometer):
            raise TypeError("argument radius_one must have a unit that is compatible with nanometer")
        if not radius_one.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("argument radius_one must have a value greater than zero")
        if not radius_two.unit.is_compatible(self._nanometer):
            raise TypeError("argument radius_two must have a unit that is compatible with nanometer")
        if not radius_two.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("argument radius_two must have a value greater than zero")
        if not surface_potential_one.unit.is_compatible(self._millivolt):
            raise TypeError("argument surface_potential_one must have a unit that is compatible with millivolts")
        if not surface_potential_two.unit.is_compatible(self._millivolt):
            raise TypeError("argument surface_potential_two must have a unit that is compatible with millivolts")
        if surface_potential_one == surface_potential_two:
            raise ValueError("the surface potentials of the two types of colloids must be different")

        self._radius_one = radius_one.in_units_of(self._nanometer)
        self._radius_two = radius_two.in_units_of(self._nanometer)
        self._surface_potential_one = surface_potential_one.in_units_of(self._millivolt)
        self._surface_potential_two = surface_potential_two.in_units_of(self._millivolt)
        self._use_log = use_log
        self._cutoff_factor = cutoff_factor
        self._maximum_surface_separation = self._cutoff_factor * self._parameters.debye_length
        self._switch_off_distance = (self._cutoff_factor - 1.0) * self._parameters.debye_length
        self._number_samples = 5000
        self._potential_11, self._potential_22, self._potential_12 = self._set_up_potentials()
        self._current_particle_index = 0
        self._indices_one = []
        self._indices_two = []

    def _steric_potential(self, prefactor: float, h_values: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Return the steric potential from the Alexander-de Gennes polymer brush model for the given surface-to-surface
        separations.
        """
        double_brush_length = 2.0 * self._parameters.brush_length.value_in_unit(self._nanometer)
        h_over_double_brush_length = h_values / double_brush_length
        double_brush_length_over_h = double_brush_length / h_values
        return prefactor * np.where(h_values <= double_brush_length,
                                    28.0 * (np.power(double_brush_length_over_h, 0.25) - 1.0)
                                    + 20.0 / 11.0 * (1.0 - np.power(h_over_double_brush_length, 2.75))
                                    + 12.0 * (h_over_double_brush_length - 1.0),
                                    0.0)

    def _electrostatic_potential(self, prefactor: float, h_values: npt.NDArray[float]) -> npt.NDArray[float]:
        """Return the electrostatic potential from DLVO theory for the given surface-to-surface separations."""
        debye_length = self._parameters.debye_length.value_in_unit(self._nanometer)
        if self._use_log:
            return prefactor * np.log(1.0 + np.exp(-h_values / debye_length))
        else:
            return prefactor * np.exp(-h_values / debye_length)

    def _set_up_potentials(self) -> (CustomNonbondedForce, CustomNonbondedForce, CustomNonbondedForce):
        """Set up the CustomNonbondedForce instances based on tabulated functions."""
        r_values_11 = np.linspace(
            1.00005 * 2.0 * self._radius_one.value_in_unit(self._nanometer),
            (2.0 * self._radius_one + self._maximum_surface_separation).value_in_unit(self._nanometer),
            num=self._number_samples)
        r_values_22 = np.linspace(
            1.00005 * 2.0 * self._radius_two.value_in_unit(self._nanometer),
            (2.0 * self._radius_two + self._maximum_surface_separation).value_in_unit(self._nanometer),
            num=self._number_samples)
        r_values_12 = np.linspace(
            (1.00005 * (self._radius_one + self._radius_two)).value_in_unit(self._nanometer),
            ((self._radius_one + self._radius_two) + self._maximum_surface_separation).value_in_unit(self._nanometer),
            num=self._number_samples)

        h_values_11 = r_values_11 - 2.0 * self._radius_one.value_in_unit(self._nanometer)
        h_values_22 = r_values_22 - 2.0 * self._radius_two.value_in_unit(self._nanometer)
        h_values_12 = r_values_12 - (self._radius_one + self._radius_two).value_in_unit(self._nanometer)

        steric_prefactor_11 = (
                unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature * 16.0 * math.pi * self._radius_one *
                (self._parameters.brush_length ** 2) * (self._parameters.brush_density ** (3 / 2)) / 35.0
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        steric_prefactor_22 = (
                unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature * 16.0 * math.pi * self._radius_two *
                (self._parameters.brush_length ** 2) * (self._parameters.brush_density ** (3 / 2)) / 35.0
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        steric_prefactor_12 = (
                unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature * 16.0 * math.pi *
                (self._radius_one + self._radius_two) / 2.0 * (self._parameters.brush_length ** 2) *
                (self._parameters.brush_density ** (3 / 2)) / 35.0
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)

        steric_potential_11 = self._steric_potential(steric_prefactor_11, h_values_11)
        steric_potential_22 = self._steric_potential(steric_prefactor_22, h_values_22)
        steric_potential_12 = self._steric_potential(steric_prefactor_12, h_values_12)

        electrostatic_prefactor_11 = (
                2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
                * self._radius_one * self._surface_potential_one * self._surface_potential_one
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        electrostatic_prefactor_22 = (
                2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
                * self._radius_two * self._surface_potential_two * self._surface_potential_two
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        electrostatic_prefactor_12 = (
                2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
                * 2.0 / (1.0 / self._radius_one + 1.0 / self._radius_two)
                * self._surface_potential_one * self._surface_potential_two
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)

        electrostatic_potential_11 = self._electrostatic_potential(electrostatic_prefactor_11, h_values_11)
        electrostatic_potential_22 = self._electrostatic_potential(electrostatic_prefactor_22, h_values_22)
        electrostatic_potential_12 = self._electrostatic_potential(electrostatic_prefactor_12, h_values_12)

        tabulated_function_11 = Continuous1DFunction(
            steric_potential_11 + electrostatic_potential_11, r_values_11[0], r_values_11[-1], False)
        tabulated_function_22 = Continuous1DFunction(
            steric_potential_22 + electrostatic_potential_22, r_values_22[0], r_values_22[-1], False)
        tabulated_function_12 = Continuous1DFunction(
            steric_potential_12 + electrostatic_potential_12, r_values_12[0], r_values_12[-1], False)

        potential_11 = CustomNonbondedForce("tabulated_function_11(r)")
        potential_11.addTabulatedFunction("tabulated_function_11", tabulated_function_11)
        if self._periodic_boundary_conditions:
            potential_11.setNonbondedMethod(potential_11.CutoffPeriodic)
        else:
            potential_11.setNonbondedMethod(potential_11.CutoffNonPeriodic)
        potential_11.setCutoffDistance(r_values_11[-1])
        potential_11.setUseSwitchingFunction(True)
        potential_11.setSwitchingDistance((self._switch_off_distance
                                          + 2.0 * self._radius_one).value_in_unit(self._nanometer))
        potential_11.setUseLongRangeCorrection(False)

        potential_22 = CustomNonbondedForce("tabulated_function_22(r)")
        potential_22.addTabulatedFunction("tabulated_function_22", tabulated_function_22)
        if self._periodic_boundary_conditions:
            potential_22.setNonbondedMethod(potential_22.CutoffPeriodic)
        else:
            potential_22.setNonbondedMethod(potential_22.CutoffNonPeriodic)
        potential_22.setCutoffDistance(r_values_22[-1])
        potential_22.setUseSwitchingFunction(True)
        potential_22.setSwitchingDistance((self._switch_off_distance
                                          + 2.0 * self._radius_two).value_in_unit(self._nanometer))
        potential_22.setUseLongRangeCorrection(False)

        potential_12 = CustomNonbondedForce("tabulated_function_12(r)")
        potential_12.addTabulatedFunction("tabulated_function_12", tabulated_function_12)
        if self._periodic_boundary_conditions:
            potential_12.setNonbondedMethod(potential_12.CutoffPeriodic)
        else:
            potential_12.setNonbondedMethod(potential_12.CutoffNonPeriodic)
        potential_12.setCutoffDistance(r_values_12[-1])
        potential_12.setUseSwitchingFunction(True)
        potential_12.setSwitchingDistance((self._switch_off_distance
                                          + self._radius_one + self._radius_two).value_in_unit(self._nanometer))
        potential_12.setUseLongRangeCorrection(False)

        return potential_11, potential_22, potential_12

    def add_particle(self, radius: unit.Quantity, surface_potential: unit.Quantity,
                     substrate_flag: bool = False) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        If the substrate flag is True, the colloid is considered to be a substrate particle. Substrate particles do
        not interact with each other. Since this class does not support substrate particles anyway (since this would
        require more than two types of colloids), the substrate flag should always be false.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param surface_potential:
            The surface potential of the colloid.
            The unit of the surface_potential must be compatible with millivolts.
        :type surface_potential: unit.Quantity
        :param substrate_flag:
            Whether the colloid is a substrate particle.
        :type substrate_flag: bool

        :raises TypeError:
            If the radius or surface_potential is not a Quantity with a proper unit (via the abstract base class).
        :raises ValueError:
            If the radius is not greater than zero (via the abstract base class).
        :raises RuntimeError:
            If the method yield_potentials was called before this method (via the abstract base class).
        :raises ValueError:
            If the given radius is not compatible with the radius of the first or the second colloid that was specified
            during the initialization of this class.
        :raises ValueError:
            If the given surface potential is not compatible with the surface potential of the first or the second
            colloid that was specified during the initialization of this class.
        :raises ValueError:
            If the substrate flag is True.
        """
        super().add_particle(radius, surface_potential, substrate_flag)

        if surface_potential == self._surface_potential_one:
            if not radius == self._radius_one:
                raise ValueError(
                    "the given radius must be compatible with the radius of the first colloid that was specified "
                    "during the initialization of this class (because the given surface potential is the same "
                    "as the surface potential of the first colloid)")
            self._indices_one.append(self._current_particle_index)
        elif surface_potential == self._surface_potential_two:
            if not radius == self._radius_two:
                raise ValueError(
                    "the given radius must be compatible with the radius of the second colloid that was specified "
                    "during the initialization of this class (because the given surface potential is the same "
                    "as the surface potential of the second colloid)")
            self._indices_two.append(self._current_particle_index)
        else:
            raise ValueError(
                "the given surface potential must be the same as the surface potential of the first or the second "
                "colloid that was specified during the initialization of this class")
        if substrate_flag:
            raise ValueError("this class does not support substrate particles")
        self._potential_11.addParticle([])
        self._potential_22.addParticle([])
        self._potential_12.addParticle([])
        self._current_particle_index += 1

    def add_exclusion(self, particle_one: int, particle_two: int) -> None:
        """
        Exclude a particle pair from the non-bonded interactions handled by this class.

        :param particle_one:
            The index of the first particle.
        :type particle_one: int
        :param particle_two:
            The index of the second particle.
        :type particle_two: int
        """
        self._potential_11.addExclusion(particle_one, particle_two)
        self._potential_22.addExclusion(particle_one, particle_two)
        self._potential_12.addExclusion(particle_one, particle_two)

    def yield_potentials(self) -> Iterator[CustomNonbondedForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the steric and electrostatic pair
        potentials between colloids in a solution in an openmm system.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields the tabulated potentials handled by this class.
        :rtype: Iterator[CustomNonbondedForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        self._potential_11.addInteractionGroup(self._indices_one, self._indices_one)
        self._potential_22.addInteractionGroup(self._indices_two, self._indices_two)
        self._potential_12.addInteractionGroup(self._indices_one, self._indices_two)
        yield self._potential_11
        yield self._potential_22
        yield self._potential_12


if __name__ == '__main__':
    ColloidPotentialsTabulated(radius_one=105 * (unit.nano * unit.meter), radius_two=95.0 * (unit.nano * unit.meter),
                               surface_potential_one=44.0 * (unit.milli * unit.volt),
                               surface_potential_two=-54.0 * (unit.milli * unit.volt))
