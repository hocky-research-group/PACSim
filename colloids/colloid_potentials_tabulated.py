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

    The potentials are based on the models described in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). Any references to equations or symbols in the code refer to this
    paper.

    The steric potential from the Alexander-de Gennes polymer brush model and the electrostatic potential from DLVO
    theory depend on the radii and surface potentials of the colloids. Before the finalized potentials are generated via
    the yield_potentials method to add them to the openmm system (using the system.addForce method), the add_particle
    method has to be called for each colloid in the system to define its radius and surface potential.

    This class can handle multiple types of colloids in the system. It defines a CustomNonbondedForce instance for each
    pair of colloid types, containing both the steric and electrostatic potentials. It requires the radii and surface
    potentials of the colloids on initialization.

    The potential of every CustomNonbondedForce instance has a cutoff at a surface-to-surface separation of
    cutoff_factor * debye_length between the involved types of colloids. Here, debye_length is the Debye screening
    length that is stored in the ColloidPotentialsParameters instance, and cutoff_factor is set on initialization. A
    switching function reduces the interaction at surface-to-surface separations larger than
    (cutoff_factor - 1) * debye_length to make the potential and forces go smoothly to 0 at the cutoff distance.

    The cutoffs can be set to be periodic or non-periodic.

    Note that the steric potential from the Alexander-de Gennes polymer brush model uses the mixing rule
    r = (r_1 + r_2) / 2.0 for the prefactor [see eq. (1)], whereas the electrostatic potential from DLVO theory uses
    r = 2.0 / (1.0 / r_1 + 1.0 / r_2) for the prefactor.

    :param radii:
        A dictionary mapping colloid types to their radii.
        The unit of each radius must be compatible with nanometers and the value must be greater than zero.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        A dictionary mapping colloid types to their surface potentials.
        The unit of each surface potential must be compatible with millivolts.
    :type surface_potentials: dict[str, unit.Quantity]
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
        If any radius or surface potential is not a Quantity with a proper unit.
    :raises ValueError:
        If any radius is not greater than zero.
        If the cutoff factor is not greater than zero.
    """

    def __init__(self, radii: dict[str, unit.Quantity], surface_potentials: dict[str, unit.Quantity],
                 colloid_potentials_parameters: ColloidPotentialsParameters = ColloidPotentialsParameters(),
                 use_log: bool = True, cutoff_factor: float = 21.0, periodic_boundary_conditions: bool = True) -> None:
        """Constructor of the ColloidPotentialsTabulated class."""
        super().__init__(colloid_potentials_parameters, periodic_boundary_conditions)
        if not cutoff_factor > 0.0:
            raise ValueError("The cutoff factor must be greater than zero.")
        if not all([radius.unit.is_compatible(self._nanometer) for radius in radii.values()]):
            raise TypeError("All radii must have a unit that is compatible with nanometer")
        if not all([radii > 0.0 * unit.nanometer for radii in radii.values()]):
            raise ValueError("All radii must have a value greater than zero")
        if not all([surface_potential.unit.is_compatible(self._millivolt) for surface_potential in surface_potentials.values()]):
            raise TypeError("All surface potentials must have a unit that is compatible with millivolts")

        self._radii = {key: value.in_units_of(self._nanometer) for key, value in radii.items()}
        self._surface_potentials = {key: value.in_units_of(self._millivolt) for key, value in surface_potentials.items()}
        self._use_log = use_log
        self._cutoff_factor = cutoff_factor
        self._maximum_surface_separation = self._cutoff_factor * self._parameters.debye_length
        self._switch_off_distance = (self._cutoff_factor - 1.0) * self._parameters.debye_length
        self._number_samples = 5000
        self._potentials = self._set_up_potentials()
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
        n_potentials = int((len(self._radii) ** 2 + len(self._radii)) / 2)
        r_values = np.zeros((n_potentials, self._number_samples))
        radii_sums = np.zeros((n_potentials))
        inverted_radii_sums = np.zeros((n_potentials))
        surface_potential_prod = np.zeros((n_potentials))
        keys = []
        rind = 0
        for i, radius_one in enumerate(self._radii.values()):
            for j, radius_two in enumerate(self._radii.values()):
                if i <= j:
                    keys.append((i, j))
                    r_values[rind, :] = np.linspace(
                        (1.00005 * (radius_one + radius_two)).value_in_unit(self._nanometer),
                        ((radius_one + radius_two) + self._maximum_surface_separation).value_in_unit(self._nanometer),
                        num=self._number_samples)
                    surface_potential_prod[rind] = (self._surface_potentials[list(self._radii.keys())[i]] *
                                                    self._surface_potentials[list(self._radii.keys())[j]]).value_in_unit(self._millivolt ** 2)
                    radii_sums[rind] = (radius_one + radius_two).value_in_unit(self._nanometer)
                    inverted_radii_sums[rind] = (2.0 / ((1.0 / radius_one) + (1.0 / radius_two))).value_in_unit(self._nanometer)
                    rind += 1

        h_values = r_values - radii_sums[:, np.newaxis]

        steric_prefactors = (
                unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature * 16.0 * math.pi * (radii_sums / 2) *
                (self._parameters.brush_length ** 2) * (self._parameters.brush_density ** (3 / 2)) / 35.0
                * unit.AVOGADRO_CONSTANT_NA * self._nanometer).value_in_unit(unit.kilojoule_per_mole)

        steric_potentials = [self._steric_potential(prefactor, h_values[ind, :]) for ind, prefactor in
                             enumerate(steric_prefactors)]


        electrostatic_prefactors = (
            2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
            * inverted_radii_sums * (self._nanometer) * surface_potential_prod * (self._millivolt ** 2)
            * unit.AVOGADRO_CONSTANT_NA ).value_in_unit(unit.kilojoule_per_mole)

        electrostatic_potentials = [self._electrostatic_potential(prefactor, h_values[ind, :]) for ind, prefactor in
                                    enumerate(electrostatic_prefactors)]

        tabulated_functions = []
        potentials = []

        for ind, (steric_potential, electrostatic_potential) in enumerate(zip(steric_potentials, electrostatic_potentials)):
            tabulated_function = Continuous1DFunction(
            steric_potential + electrostatic_potential, r_values[ind, 0], r_values[ind, -1], False)
            tabulated_functions.append(tabulated_function)

            potential = CustomNonbondedForce(f"tabulated_function_{ind}(r)")
            potential.addTabulatedFunction(f"tabulated_function_{ind}", tabulated_function)
            if self._periodic_boundary_conditions:
                potential.setNonbondedMethod(potential.CutoffPeriodic)
            else:
                potential.setNonbondedMethod(potential.CutoffNonPeriodic)
                potential.setCutoffDistance(r_values[ind, -1])
                potential.setUseSwitchingFunction(True)
                potential.setSwitchingDistance(self._switch_off_distance.value_in_unit(self._nanometer))
                potential.setUseLongRangeCorrection(False)
                potentials.append(potential)

        return tuple(potentials)

    def add_particle(self, radius: unit.Quantity, surface_potential: unit.Quantity,
                     substrate_flag: bool = False) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        If the substrate flag is True, the colloid is considered to be a substrate particle. 

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

        for potential in self._potentials:
            potential.addParticle([])
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
        for potential in self._potentials:
            potential.addExclusion(particle_one, particle_two)

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
        for potential in self._potentials:
            potential.addInteractionGroup(self._indices_one, self._indices_two)
            yield potential


if __name__ == '__main__':
    ColloidPotentialsTabulated(radii={"colloid_one": 1.0 * unit.nanometer, "colloid_two": 2.0 * unit.nanometer},
                                 surface_potentials={"colloid_one": 44.0 * unit.volt,
                                                    "colloid_two": -54.0 * unit.volt},
                                 colloid_potentials_parameters=ColloidPotentialsParameters(),
                                 use_log=True, cutoff_factor=21.0, periodic_boundary_conditions=True)