import math
from openmm import CustomNonbondedForce, unit


class ColloidForces(object):
    """
    This class sets up the steric and electrostatic pair forces between colloids in a solution with periodic boundary
    conditions using the CustomNonbondedForces class of openmm.

    The forces are based on Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). Any references to equations or symbols in the code refer to this
    paper.

    The steric force from the Alexander-de Gennes polymer brush model between two colloids depends their radii r_1 and
    r_2. Similarly, the electrostatic force from DLVO theory between two colloids depends on their radii r_1 and r_2
    and their surface potentials psi_1 and psi_2. Before the forces can be accessed via the steric_force and the
    electrostatic_force properties, the add_particle method has to be called for each colloid in the system to define
    its radius and surface potential.

    The cutoff of the electrostatic force is set to 2.0 * r_max + 20.0 * debye_length, where r_max is the largest radius
    of the colloids in the system and debye_length is the Debye screening length. The largest r_max is automatically
    determined when the add_particle method is called. The cutoff of the steric force is set to
    2.0 * r_max + 2.0 * brush_length, where brush_length is the thickness of the polymer brush.

    Note that the steric force from the Alexander-de Gennes polymer brush model uses the mixing rule
    r = r_1 + r_2 / 2.0 for the prefactor [see eq. (1)], whereas the electrostatic force from DLVO theory uses
    r = 2.0 / (1.0 / r_1 + 1.0 / r_2) for the prefactor.

    TODO Implement the forces in a different class based on lookup tables.

    :param brush_density:
        The polymer surface density in the Alexander-de Gennes polymer brush model [i.e., sigma in eq. (1)].
        The unit of the brush_density must be compatible with 1/nanometer^2 and the value must be greater than zero.
        Defaults to 0.09/nanometer^2.
    :type brush_density: unit.Quantity
    :param brush_length:
        The thickness of the brush in the Alexander-de Gennes polymer brush model [i.e., L in eq. (1)].
        The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 10.0 nanometers.
    :type brush_length: unit.Quantity
    :param debye_length:
        The Debye screening length within DLVO theory [i.e., lambda_D].
        The unit of the debye_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 5.0 nanometers.
    :type debye_length: unit.Quantity
    :param temperature:
        The temperature of the system [i.e., T].
        The unit of the temperature must be compatible with kelvin and the value must be greater than zero.
        Defaults to 298.0 kelvin.
    :type temperature: unit.Quantity
    :param dielectric_constant:
        The dielectric constant of the solvent [i.e., epsilon] and the value must be greater than zero.
        Defaults to 80.0 (i.e., water).
    :type dielectric_constant: float

    :raises TypeError:
        If the brush_density, brush_length, debye_length, or temperature is not a Quantity with a proper unit.
    :raises ValueError:
        If the brush_density, brush_length, debye_length, temperature, or dielectric_constant is not greater than zero.
    """

    _VACUUM_PERMITTIVITY = 8.8541878128e-12 * unit.joule / (unit.volt ** 2 * unit.meter)

    def __init__(self, brush_density: unit.Quantity = 0.09 / (unit.nanometer ** 2),
                 brush_length: unit.Quantity = 10.0 * unit.nanometer,
                 debye_length: unit.Quantity = 5.0 * unit.nanometer,
                 temperature: unit.Quantity = 298.0 * unit.kelvin,
                 dielectric_constant: float = 80.0) -> None:
        """Constructor of the ColloidForces class."""
        if not brush_density.unit.is_compatible(unit.nanometer ** -2):
            raise TypeError("argument brush_density must have a unit that is compatible with 1/nanometer^2")
        if not brush_density.value_in_unit(unit.nanometer ** -2) > 0.0:
            raise ValueError("argument brush_density must have a value greater than zero")
        if not brush_length.unit.is_compatible(unit.nanometer):
            raise TypeError("argument brush_length must have a unit that is compatible with nanometers")
        if not brush_length.value_in_unit(unit.nanometer) > 0.0:
            raise ValueError("argument brush_length must have a value greater than zero")
        if not debye_length.unit.is_compatible(unit.nanometer):
            raise TypeError("argument debye_length must have a unit that is compatible with nanometers")
        if not debye_length.value_in_unit(unit.nanometer) > 0.0:
            raise ValueError("argument debye_length must have a value greater than zero")
        if not temperature.unit.is_compatible(unit.kelvin):
            raise TypeError("argument temperature must have a unit that is compatible with kelvin")
        if not temperature.value_in_unit(unit.kelvin) > 0.0:
            raise ValueError("argument temperature must have a value greater than zero")
        if not dielectric_constant > 0.0:
            raise ValueError("argument dielectric_constant must have a value greater than zero")

        self._brush_density = brush_density.in_units_of(unit.nanometer ** -2)
        self._brush_length = brush_length.in_units_of(unit.nanometer)
        self._debye_length = debye_length.in_units_of(unit.nanometer)
        self._temperature = temperature.in_units_of(unit.kelvin)
        self._dielectric_constant = dielectric_constant

        self._steric_force = self._set_up_steric_force()
        self._electrostatic_force = self._set_up_electrostatic_force()

        self._max_radius = -math.inf * unit.nanometer

    def _set_up_steric_force(self) -> CustomNonbondedForce:
        """Set up the basic functional form of the steric force from the Alexander-de Gennes polymer brush model."""
        steric_force = CustomNonbondedForce(
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
        steric_force.addGlobalParameter(
            "steric_prefactor",
            (unit.BOLTZMANN_CONSTANT_kB * self._temperature
             * 16.0 * math.pi * (self._brush_density ** (3 / 2)) / 35.0
             * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole / (unit.nanometer ** 3))
        )
        # Brush length L (see Hocky paper)
        steric_force.addGlobalParameter("brush_length", self._brush_length.value_in_unit(unit.nanometer))
        steric_force.addPerParticleParameter("radius")
        return steric_force

    def _set_up_electrostatic_force(self) -> CustomNonbondedForce:
        """Set up the basic functional form of the electrostatic force from DLVO theory."""
        electrostatic_force = CustomNonbondedForce(
            "electrostatic_prefactor * radius * psi1 * psi2 * exp(-h / debye_length); "
            "radius = 2.0 / (1.0 / radius1 + 1.0 / radius2);"
            "h = r - rs;"
            "rs = radius1 + radius2"
        )
        # Prefactor is 2 * pi * epsilon
        electrostatic_force.addGlobalParameter(
            "electrostatic_prefactor",
            (2.0 * math.pi * self._VACUUM_PERMITTIVITY * self._dielectric_constant
             * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
                unit.kilojoule_per_mole / (unit.nanometer * (unit.milli * unit.volt) ** 2)))
        electrostatic_force.addGlobalParameter("debye_length", self._debye_length.value_in_unit(unit.nanometer))
        electrostatic_force.addPerParticleParameter("radius")
        # Psi should be given in millivolts.
        electrostatic_force.addPerParticleParameter("psi")
        return electrostatic_force

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

        self._steric_force.addParticle([radius.value_in_unit(unit.nanometer)])
        self._electrostatic_force.addParticle([radius.value_in_unit(unit.nanometer),
                                               surface_potential.value_in_unit(unit.milli * unit.volt)])

    @property
    def steric_force(self) -> CustomNonbondedForce:
        """
        Return the steric force between the colloids in the system.

        :return: The steric force.
        :rtype: CustomNonbondedForce

        :raises RuntimeError:
            If no particles have been added to the system via the add_particle method.
        """
        if math.isinf(self._max_radius.value_in_unit(unit.nanometer)):
            raise RuntimeError("particles have to be added to the system via the add_particle method before the "
                               "steric_force can be accessed")
        self._steric_force.setNonbondedMethod(self._steric_force.CutoffPeriodic)
        self._steric_force.setCutoffDistance(
            (2.0 * self._max_radius + 2.0 * self._brush_length).value_in_unit(unit.nanometer))
        self._steric_force.setUseLongRangeCorrection(False)
        self._steric_force.setUseSwitchingFunction(False)
        return self._steric_force

    @property
    def electrostatic_force(self) -> CustomNonbondedForce:
        """
        Return the electrostatic force between the colloids in the system.

        :return: The electrostatic force.
        :rtype: CustomNonbondedForce

        :raises RuntimeError:
            If no particles have been added to the system via the add_particle method.
        """
        if math.isinf(self._max_radius.value_in_unit(unit.nanometer)):
            raise RuntimeError("particles have to be added to the system via the add_particle method before the "
                               "electrostatic_force can be accessed")
        self._electrostatic_force.setNonbondedMethod(self._electrostatic_force.CutoffPeriodic)
        self._electrostatic_force.setCutoffDistance(
            (2.0 * self._max_radius + 20.0 * self._debye_length).value_in_unit(unit.nanometer))
        self._electrostatic_force.setUseLongRangeCorrection(False)
        self._electrostatic_force.setUseSwitchingFunction(False)
        return self._electrostatic_force


if __name__ == '__main__':
    ColloidForces()
