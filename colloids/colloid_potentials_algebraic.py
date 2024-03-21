import math
from openmm import CustomNonbondedForce, unit
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters


class ColloidPotentialsAlgebraic(object):
    """
    This class sets up the steric and electrostatic pair potentials between colloids in a solution with periodic
    boundary conditions using the CustomNonbondedForces class of openmm with an algebraic expression.

    The potentials are given in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). Any references to equations or symbols in the code refer to this
    paper.

    The steric potential from the Alexander-de Gennes polymer brush model between two colloids depends their radii r_1
    and r_2. Similarly, the electrostatic potential from DLVO theory between two colloids depends on their radii r_1 and
    r_2 and their surface potentials psi_1 and psi_2. Before the potentials can be accessed via the steric_potential and
    the electrostatic_potential properties, the add_particle method has to be called for each colloid in the system to
    define its radius and surface potential.

    The cutoff of the electrostatic potential is set to 2.0 * r_max + 21.0 * debye_length, where r_max is the largest
    radius of the colloids in the system and debye_length is the Debye screening length that is stored in the
    ColloidPotentialsParameters instance. The largest r_max is automatically determined when the add_particle method is
    called. A switching function reduces the interaction at distances larger than 2.0 * r_max + 20.0 * debye_length to
    make the potential and forces go smoothly to 0 at the cutoff distance.

    The cutoff of the steric potential is set to 2.0 * r_max + 2.0 * brush_length, where brush_length is the thickness
    of the polymer brush.

    Note that the steric potential from the Alexander-de Gennes polymer brush model uses the mixing rule
    r = r_1 + r_2 / 2.0 for the prefactor [see eq. (1)], whereas the electrostatic potential from DLVO theory uses
    r = 2.0 / (1.0 / r_1 + 1.0 / r_2) for the prefactor.

    TODO Implement the forces in a different class based on lookup tables and benchmark.

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
    """

    def __init__(self, colloid_potentials_parameters: ColloidPotentialsParameters = ColloidPotentialsParameters(),
                 use_log: bool = True) -> None:
        """Constructor of the ColloidPotentialsAlgebraic class."""
        self._parameters = colloid_potentials_parameters
        self._use_log = use_log
        self._steric_potential = self._set_up_steric_potential()
        self._electrostatic_potential = self._set_up_electrostatic_potential()
        self._max_radius = -math.inf * unit.nanometer

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
             * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole / (unit.nanometer ** 3))
        )
        # Brush length L (see Hocky paper)
        steric_potential.addGlobalParameter("brush_length",
                                            self._parameters.brush_length.value_in_unit(unit.nanometer))
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
                unit.kilojoule_per_mole / (unit.nanometer * (unit.milli * unit.volt) ** 2)))
        electrostatic_potential.addGlobalParameter("debye_length",
                                                   self._parameters.debye_length.value_in_unit(unit.nanometer))
        electrostatic_potential.addPerParticleParameter("radius")
        # Psi should be given in millivolts.
        electrostatic_potential.addPerParticleParameter("psi")
        return electrostatic_potential

    def add_particle(self, radius: unit.Quantity, surface_potential: unit.Quantity) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        This function has to be called for every particle in the system.

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
        """
        if not radius.unit.is_compatible(unit.nanometer):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(unit.nanometer) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        if not surface_potential.unit.is_compatible(unit.milli * unit.volt):
            raise TypeError("argument surface_potential must have a unit that is compatible with volts")

        if radius.in_units_of(unit.nanometer) > self._max_radius:
            self._max_radius = radius.in_units_of(unit.nanometer)

        self._steric_potential.addParticle([radius.value_in_unit(unit.nanometer)])
        self._electrostatic_potential.addParticle([radius.value_in_unit(unit.nanometer),
                                                   surface_potential.value_in_unit(unit.milli * unit.volt)])

    @property
    def steric_potential(self) -> CustomNonbondedForce:
        """
        Return the steric potential/force between the colloids in the system.

        :return: The steric potential/force.
        :rtype: CustomNonbondedForce

        :raises RuntimeError:
            If no particles have been added to the system via the add_particle method.
        """
        if math.isinf(self._max_radius.value_in_unit(unit.nanometer)):
            raise RuntimeError("particles have to be added to the system via the add_particle method before the "
                               "steric_force can be accessed")
        self._steric_potential.setNonbondedMethod(self._steric_potential.CutoffPeriodic)
        self._steric_potential.setCutoffDistance(
            (2.0 * self._max_radius + 2.0 * self._parameters.brush_length).value_in_unit(unit.nanometer))
        self._steric_potential.setUseLongRangeCorrection(False)
        self._steric_potential.setUseSwitchingFunction(False)
        return self._steric_potential

    @property
    def electrostatic_potential(self) -> CustomNonbondedForce:
        """
        Return the electrostatic potential/force between the colloids in the system.

        :return: The electrostatic potential/force.
        :rtype: CustomNonbondedForce

        :raises RuntimeError:
            If no particles have been added to the system via the add_particle method.
        """
        if math.isinf(self._max_radius.value_in_unit(unit.nanometer)):
            raise RuntimeError("particles have to be added to the system via the add_particle method before the "
                               "electrostatic_force can be accessed")
        self._electrostatic_potential.setNonbondedMethod(self._electrostatic_potential.CutoffPeriodic)
        self._electrostatic_potential.setCutoffDistance(
            (2.0 * self._max_radius + 21.0 * self._parameters.debye_length).value_in_unit(unit.nanometer))
        self._electrostatic_potential.setUseLongRangeCorrection(False)
        self._electrostatic_potential.setUseSwitchingFunction(True)
        self._electrostatic_potential.setSwitchingDistance(
            (2.0 * self._max_radius + 20.0 * self._parameters.debye_length).value_in_unit(unit.nanometer))
        return self._electrostatic_potential


if __name__ == '__main__':
    ColloidPotentialsAlgebraic()
