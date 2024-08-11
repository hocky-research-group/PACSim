import math
from typing import Iterator, Sequence
import numpy as np
from openmm import Context, CustomNonbondedForce, LangevinIntegrator, Platform, System, unit, Vec3
from scipy.optimize import minimize, root_scalar
from colloids.abstracts import ColloidPotentialsAbstract
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters


class ColloidPotentialsAlgebraic(ColloidPotentialsAbstract):
    """
    This class sets up the steric and electrostatic pair potentials between colloids in a solution using the
    CustomNonbondedForces class of openmm with an algebraic expression.

    The potentials are given in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). Any references to equations or symbols in the code refer to this
    paper.

    The steric potential from the Alexander-de Gennes polymer brush model between two colloids depends on their radii
    r_1 and r_2. Similarly, the electrostatic potential from DLVO theory between two colloids depends on their radii r_1
    and r_2 and their surface potentials psi_1 and psi_2. Before the finalized potentials are generated via the
    yield_potentials method in order to add them to the openmm system (using the system.addForce method), the
    add_particle method has to be called for each colloid in the system to define its radius and surface potential.

    The cutoff of the electrostatic potential is set to 2.0 * r_max + cutoff_factor * debye_length, where r_max is the
    largest radius of the colloids in the system, debye_length is the Debye screening length that is stored in the
    ColloidPotentialsParameters instance, and cutoff_factor is set on initialization. The largest r_max is automatically
    determined when the add_particle method is called. A switching function reduces the interaction at distances larger
    than 2.0 * r_max + (cutoff_factor - 1) * debye_length to make the potential and forces go smoothly to 0 at the
    cutoff distance.

    The cutoff of the steric potential is set to 2.0 * r_max + 2.0 * brush_length, where brush_length is the thickness
    of the polymer brush.

    The cutoffs can be set to be periodic or non-periodic.

    Note that the steric potential from the Alexander-de Gennes polymer brush model uses the mixing rule
    r = r_1 + r_2 / 2.0 for the prefactor [see eq. (1)], whereas the electrostatic potential from DLVO theory uses
    r = 2.0 / (1.0 / r_1 + 1.0 / r_2) for the prefactor.

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
        The factor by which the Debye length is multiplied to get the cutoff distance of the electrostatic force.
        Defaults to 21.0.
    :type cutoff_factor: float
    :param periodic_boundary_conditions:
        Whether this force should use periodic cutoffs for the steric and electrostatic potentials.
    :type periodic_boundary_conditions: bool

    :raises ValueError:
        If the cutoff factor is not greater than zero.
    """

    def __init__(self, colloid_potentials_parameters: ColloidPotentialsParameters = ColloidPotentialsParameters(),
                 use_log: bool = True, cutoff_factor: float = 21.0, periodic_boundary_conditions: bool = True) -> None:
        """Constructor of the ColloidPotentialsAlgebraic class."""
        super().__init__(colloid_potentials_parameters, periodic_boundary_conditions)
        if not cutoff_factor > 0.0:
            raise ValueError("The cutoff factor must be greater than zero.")

        self._use_log = use_log
        self._steric_potential = self._set_up_steric_potential()
        self._electrostatic_potential = self._set_up_electrostatic_potential()
        self._max_radius = -math.inf * self._nanometer
        self._cutoff_factor = cutoff_factor

    def _set_up_steric_potential(self) -> CustomNonbondedForce:
        """Set up the basic functional form of the steric potential from the Alexander-de Gennes polymer brush model."""
        steric_potential = CustomNonbondedForce(
            "step(two_l - h) * "
            "steric_prefactor * rs / 2.0 * brush_length * brush_length * ("
            "28.0 * ((two_l / h)^0.25 - 1.0) "
            "+ 20.0 / 11.0 * (1.0 - (h / two_l)^2.75)"
            "+ 12.0 * (h / two_l - 1.0)); "
            "h = r - rs;"
            "rs = radius1 + radius2;"
            "two_l = 2.0 * brush_length"
        )
        # Prefactor is k_B * T * 16 * pi * sigma^(3/2) / 35 (see Hocky paper)
        steric_potential.addGlobalParameter(
            "steric_prefactor",
            (unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature
             * 16.0 * math.pi * (self._parameters.brush_density ** (3 / 2)) / 35.0
             * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole / (self._nanometer ** 3))
        )
        # Brush length L (see Hocky paper)
        steric_potential.addGlobalParameter("brush_length",
                                            self._parameters.brush_length.value_in_unit(self._nanometer))
        steric_potential.addPerParticleParameter("radius")
        return steric_potential

    def _set_up_electrostatic_potential(self) -> CustomNonbondedForce:
        """Set up the basic functional form of the electrostatic potential from DLVO theory."""
        if self._use_log:
            electrostatic_potential = CustomNonbondedForce(
                "electrostatic_prefactor * radius * psi1 * psi2 * log(1.0 + exp(-h / debye_length)); "
                "radius = 2.0 / (1.0 / radius1 + 1.0 / radius2);"
                "h = r - rs;"
                "rs = radius1 + radius2"
            )
        else:
            electrostatic_potential = CustomNonbondedForce(
                "electrostatic_prefactor * radius * psi1 * psi2 * exp(-h / debye_length); "
                "radius = 2.0 / (1.0 / radius1 + 1.0 / radius2);"
                "h = r - rs;"
                "rs = radius1 + radius2"
            )
        # Prefactor is 2 * pi * epsilon
        electrostatic_potential.addGlobalParameter(
            "electrostatic_prefactor",
            (2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
             * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
                unit.kilojoule_per_mole / (self._nanometer * self._millivolt ** 2)))
        electrostatic_potential.addGlobalParameter("debye_length",
                                                   self._parameters.debye_length.value_in_unit(self._nanometer))
        electrostatic_potential.addPerParticleParameter("radius")
        # Psi should be given in millivolts.
        electrostatic_potential.addPerParticleParameter("psi")
        return electrostatic_potential

    def add_particle(self, radius: unit.Quantity, surface_potential: unit.Quantity) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param surface_potential:
            The surface potential of the colloid.
            The unit of the surface_potential must be compatible with millivolts.
        :type surface_potential: unit.Quantity

        :raises TypeError:
            If the radius or surface_potential is not a Quantity with a proper unit (via the abstract base class).
        :raises ValueError:
            If the radius is not greater than zero (via the abstract base class).
        :raises RuntimeError:
            If the method yield_potentials was called before this method (via the abstract base class).
        """
        super().add_particle(radius, surface_potential)

        if radius.in_units_of(self._nanometer) > self._max_radius:
            self._max_radius = radius.in_units_of(self._nanometer)

        self._steric_potential.addParticle([radius.value_in_unit(self._nanometer)])
        self._electrostatic_potential.addParticle([radius.value_in_unit(self._nanometer),
                                                   surface_potential.value_in_unit(self._millivolt)])

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

        if self._periodic_boundary_conditions:
            self._steric_potential.setNonbondedMethod(self._steric_potential.CutoffPeriodic)
        else:
            self._steric_potential.setNonbondedMethod(self._steric_potential.CutoffNonPeriodic)
        self._steric_potential.setCutoffDistance(
            (2.0 * self._max_radius + 2.0 * self._parameters.brush_length).value_in_unit(self._nanometer))
        self._steric_potential.setUseLongRangeCorrection(False)
        self._steric_potential.setUseSwitchingFunction(False)

        if self._periodic_boundary_conditions:
            self._electrostatic_potential.setNonbondedMethod(self._electrostatic_potential.CutoffPeriodic)
        else:
            self._electrostatic_potential.setNonbondedMethod(self._electrostatic_potential.CutoffNonPeriodic)
        self._electrostatic_potential.setCutoffDistance(
            (2.0 * self._max_radius
             + self._cutoff_factor * self._parameters.debye_length).value_in_unit(self._nanometer))
        self._electrostatic_potential.setUseLongRangeCorrection(False)
        self._electrostatic_potential.setUseSwitchingFunction(True)
        self._electrostatic_potential.setSwitchingDistance(
            (2.0 * self._max_radius
             + (self._cutoff_factor - 1.0) * self._parameters.debye_length).value_in_unit(self._nanometer))

        yield self._steric_potential
        yield self._electrostatic_potential

    def tune_surface_potential(self, other_radius: unit.Quantity, other_surface_potential: unit.Quantity,
                               tuned_radius: unit.Quantity, tuned_potential_depth: unit.Quantity) -> unit.Quantity:
        """
        Tune the surface potential of a colloid with a given radius so that the potential depth of the combined steric
        and electrostatic potentials with another colloid is equal to the given potential depth.

        :param other_radius:
            The radius of the other colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type other_radius: unit.Quantity
        :param other_surface_potential:
            The surface potential of the other colloid.
            The unit of the surface_potential must be compatible with millivolts.
        :type other_surface_potential: unit.Quantity
        :param tuned_radius:
            The radius of the colloid whose surface potential will be tuned.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type tuned_radius: unit.Quantity
        :param tuned_potential_depth:
            The desired potential depth of the combined steric and electrostatic potential with the other colloid.
            The unit of the potential_depth must be compatible with kilojoules per mole and the value must be smaller
            than zero.
        :type tuned_potential_depth: unit.Quantity

        :return:
            The tuned surface potential of the colloid in millivolts.
        :rtype: unit.Quantity

        :raises TypeError:
            If other_radius, other_radius, or tuned_potential_depth is not a Quantity with a proper unit (via the
            abstract base class).
        :raises ValueError:
            If other_radius or tuned_radius is not greater than zero (via the abstract base class).
            If the tuned_potential_depth is not smaller than zero (via the abstract base class).
        """
        super().tune_surface_potential(other_radius, other_surface_potential, tuned_radius, tuned_potential_depth)

        system = System()
        platform = Platform.getPlatformByName("Reference")
        dummy_integrator = LangevinIntegrator(0.0, 0.0, 0.0)
        electrostatic_potential = self._set_up_electrostatic_potential()
        steric_potential = self._set_up_steric_potential()
        electrostatic_potential.setNonbondedMethod(electrostatic_potential.NoCutoff)
        steric_potential.setNonbondedMethod(steric_potential.NoCutoff)
        system.addParticle(1.0 * unit.amu)
        system.addParticle(1.0 * unit.amu)
        # Force the surface potential of the other colloid to be positive.
        electrostatic_potential.addParticle([other_radius.value_in_unit(self._nanometer),
                                             abs(other_surface_potential.value_in_unit(self._millivolt))])
        # We use the global electrostatic prefactor to tune the (negative) surface potential.
        electrostatic_potential.addParticle([tuned_radius.value_in_unit(self._nanometer), 1.0])
        steric_potential.addParticle([other_radius.value_in_unit(self._nanometer)])
        steric_potential.addParticle([tuned_radius.value_in_unit(self._nanometer)])
        system.addForce(electrostatic_potential)
        system.addForce(steric_potential)
        context = Context(system, dummy_integrator, platform)
        original_electrostatic_prefactor = context.getParameter("electrostatic_prefactor")
        radius_sum = (other_radius + tuned_radius).value_in_unit(self._nanometer)

        def potential_energy(surface_separation: Sequence[float], surface_potential: float) -> float:
            # Surface separation must be a numpy array for the minimize function.
            assert len(surface_separation) == 1
            assert surface_potential <= 0.0
            # We use the global electrostatic prefactor to tune the negative surface potential.
            context.setParameter("electrostatic_prefactor", original_electrostatic_prefactor * surface_potential)
            context.setPositions([Vec3(0.0, 0.0, 0.0),
                                  Vec3(radius_sum + surface_separation[0], 0.0, 0.0)])
            return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        def deviation_potential_energy(surface_potential: float) -> float:
            minimum_energy_result = minimize(potential_energy, np.array([10.0]), args=(surface_potential,), tol=1.0e-3)
            if not minimum_energy_result.success:
                raise RuntimeError(minimum_energy_result.message + " Minimization failed.")
            assert len(minimum_energy_result.x) == 1
            return (potential_energy(minimum_energy_result.x, surface_potential)
                    - tuned_potential_depth.value_in_unit(unit.kilojoule_per_mole))

        result = root_scalar(
            deviation_potential_energy,
            bracket=[-10.0 * abs(other_surface_potential.value_in_unit(self._millivolt)), 0.0],
            method="brentq")
        if not result.converged:
            raise RuntimeError(result.flag)

        # Choose the opposite sign of the surface potential.
        return -math.copysign(result.root, other_surface_potential.value_in_unit(self._millivolt)) * self._millivolt


if __name__ == '__main__':
    ColloidPotentialsAlgebraic()
