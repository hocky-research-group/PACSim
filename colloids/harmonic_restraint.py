from typing import Iterator, Optional, Sequence
import numpy as np
import numpy.typing as npt
from openmm import CustomExternalForce, unit
from colloids.abstracts import OpenMMPotentialAbstract
from colloids.units import energy_unit, length_unit


class HarmonicRestraint(OpenMMPotentialAbstract):
    r"""
    This class sets up a harmonic (Einstein-crystal) restraint that ties every particle to a fixed
    reference position using the CustomExternalForce class of OpenMM.

    The restraint implements the translational Einstein-crystal field used in the Frenkel-Ladd
    method to compute the free energy of solids (see D. Frenkel and A. J. C. Ladd,
    J. Chem. Phys. 81, 3188 (1984); C. Vega et al., J. Phys.: Condens. Matter 20, 153101 (2008)).

    The potential energy of a single particle at position r that is restrained to the reference
    position r0 is

        u(r) = lambda_ein * spring_constant * |r - r0|^2
             = lambda_ein * spring_constant * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2).

    Following the convention of Vega et al. (2008), the spring constant (denoted Lambda_E in the
    review) multiplies the *squared* displacement without an additional factor of one half, and it
    therefore has units of energy per squared length. With this convention the free energy of the
    ideal Einstein crystal is

        A0 / (N kB T) = -(1/N) ln[ (1 / Lambda_dB^(3N)) (pi / (beta Lambda_E))^(3(N-1)/2) N^(3/2) V ]

    for an atomic solid with fixed center of mass (eq. (48) of the review), where Lambda_dB is the
    thermal de Broglie wavelength (conventionally set to the unit of length) and V is the volume.

    The global parameter ``lambda_ein`` is the dimensionless coupling parameter of the
    thermodynamic integration. It scales the strength of the whole restraint so that a single
    OpenMM system can be reused for every window of the integration by simply changing this global
    parameter with ``simulation.context.setParameter("lambda_ein", value)`` (or with an update
    reporter). Note that the energy that is logged for this force by the GSDReporter is the
    *coupled* energy lambda_ein * Lambda_E * |r - r0|^2; the free-energy analysis divides by
    lambda_ein to recover the bare Einstein energy Lambda_E * |r - r0|^2 that enters the
    thermodynamic-integration integrand.

    Because this force is placed in its own force group and given a distinct name in the OpenMM
    system, its energy is automatically recorded in the log of the GSD trajectory by the
    GSDReporter. This provides the harmonic-restraint energy that is needed to compute the free
    energy afterwards.

    :param spring_constant:
        The spring constant Lambda_E of the harmonic restraint.
        The unit of the spring constant must be compatible with kilojoules per mole per squared
        nanometer and the value must be greater than zero.
    :type spring_constant: unit.Quantity
    :param coupling:
        The initial value of the dimensionless coupling parameter ``lambda_ein`` that scales the
        strength of the restraint. It must satisfy 0 <= coupling.
        Defaults to 1.0.
    :type coupling: float

    :raises TypeError:
        If the spring constant is not a Quantity with a unit compatible with
        kilojoules per mole per squared nanometer.
    :raises ValueError:
        If the spring constant is not greater than zero.
        If the coupling is negative.
    """

    _spring_constant_unit = energy_unit / (length_unit ** 2)
    _name = "harmonic_restraint_energy"
    _coupling_parameter_name = "lambda_ein"
    _spring_constant_parameter_name = "spring_constant"

    def __init__(self, spring_constant: unit.Quantity, coupling: float = 1.0) -> None:
        """Constructor of the HarmonicRestraint class."""
        super().__init__()

        if not spring_constant.unit.is_compatible(self._spring_constant_unit):
            raise TypeError("argument spring_constant must have a unit that is compatible with "
                            "kilojoules per mole per squared nanometer")
        if not spring_constant.value_in_unit(self._spring_constant_unit) > 0.0:
            raise ValueError("argument spring_constant must have a value greater than zero")
        if not coupling >= 0.0:
            raise ValueError("argument coupling must be greater than or equal to zero")

        self._spring_constant = spring_constant
        self._coupling = coupling
        self._restraint_potential = self._set_up_restraint_potential()

    def _set_up_restraint_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the harmonic restraint."""
        restraint_potential = CustomExternalForce(
            f"{self._coupling_parameter_name} * {self._spring_constant_parameter_name} * "
            "((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
        restraint_potential.addGlobalParameter(self._coupling_parameter_name, self._coupling)
        restraint_potential.addGlobalParameter(
            self._spring_constant_parameter_name,
            self._spring_constant.value_in_unit(self._spring_constant_unit))
        restraint_potential.addPerParticleParameter("x0")
        restraint_potential.addPerParticleParameter("y0")
        restraint_potential.addPerParticleParameter("z0")
        return restraint_potential

    def add_particle(self, index: int, reference_position: Sequence[unit.Quantity]) -> None:
        """
        Restrain the particle with the given index to the given reference position.

        This method has to be called for every particle that should be restrained before the method
        yield_potentials is used. Particles that should not be restrained (for instance an immobile
        substrate) should simply not be added.

        :param index:
            The index of the particle in the OpenMM system.
        :type index: int
        :param reference_position:
            The reference position (x0, y0, z0) to which the particle is restrained.
            The unit of the reference position must be compatible with nanometers.
        :type reference_position: Sequence[unit.Quantity]

        :raises TypeError:
            If the reference position is not a Quantity with a unit compatible with nanometers.
        :raises ValueError:
            If the reference position does not have exactly three components.
        :raises RuntimeError:
            If this method is called after the yield_potentials method (via the abstract base
            class).
        """
        super().add_particle()
        reference_position = unit.Quantity(reference_position)
        if not reference_position.unit.is_compatible(length_unit):
            raise TypeError("argument reference_position must have a unit that is compatible with "
                            "nanometers")
        reference_position_value = np.atleast_1d(reference_position.value_in_unit(length_unit))
        if not reference_position_value.shape == (3,):
            raise ValueError("argument reference_position must have exactly three components")
        self._restraint_potential.addParticle(index, [float(reference_position_value[0]),
                                                      float(reference_position_value[1]),
                                                      float(reference_position_value[2])])

    def yield_potentials(self) -> Iterator[CustomExternalForce]:
        """
        Generate the harmonic restraint that is necessary to properly include the Einstein-crystal
        field in the OpenMM system.

        This method has to be called after the method add_particle was called for every particle
        that should be restrained.

        :return:
            A generator that yields the harmonic restraint handled by this class.
        :rtype: Iterator[CustomExternalForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base
            class).
        """
        super().yield_potentials()
        self._restraint_potential.setName(self._name)
        yield self._restraint_potential
