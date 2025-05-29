import math
from typing import Iterator
from openmm import unit
from openmm import CustomNonbondedForce
from colloids.abstracts import OpenMMNonbondedPotentialAbstract
from colloids.units import energy_unit, length_unit, temperature_unit


class WCAPotential(OpenMMNonbondedPotentialAbstract):
    """
    This class sets up the Lennard Jones potential between colloids with DNA patches.
    This was used in Colloidal alloys with preassembled clusters and spheres https://www.nature.com/articles/nmat4869.

    The cutoff distance for the Lennard Jones potential is set to max(sigma_colloid) + sigma_depletant where sigma_colloid
    is the diameter of the largest particle in the system plus two lengths of the polymer brush, and sigma_depletant is
    the diameter of the depletant. The cutoff can be set to be periodic or non-periodic.

    :param epsilon:
        The "radius" of the depletants, if treated as hard spheres.
        The unit of the epsilon must be compatible with kJ and the value must be greater than zero.
    :type epsilon: unit.Quantity
    :param radii:
    
    :param n:
        The order of the Lennard-Jones potential, which is typically 6 for the attractive part.
        The unit of the order must be compatible with dimensionless and the value must be greater than zero.
    :type n: float
    :param interactions:
        The interactions between the particles, which is a list of tuples containing the names of the particles that interact.
    :type interactions: tuple[str, str]
    :param periodic_boundary_conditions:
        Whether this force should use periodic cutoffs for the lennard jones potential.
    :type periodic_boundary_conditions: bool
    """
    
    def __init__(self, epsilon: unit.Quantity, radii: tuple[unit.Quantity], interactions: tuple[str, str],
                 n: float, periodic_boundary_conditions: bool = True):
        """Constructor of the LennardJonesPotential class."""
        super().__init__()


        if not epsilon.unit.is_compatible(energy_unit):
            raise TypeError("The epsilon parameter must have a unit compatible with kJ.")
        if epsilon.value_in_unit(energy_unit) <= 0.0:
            raise ValueError("The epsilon parameter must be greater than zero.")
        if not all(radius.unit.is_compatible(length_unit) for radius in radii):
            raise TypeError("All radii must have a unit compatible with nanometers.")
        if not all(radius.value_in_unit(length_unit) > 0.0 for radius in radii):
            raise ValueError("All radii must have a value greater than zero.")
        if not isinstance(n, (int, float)) or n <= 0:
            raise TypeError("The order parameter must be a positive number.")
        if not isinstance(interactions, tuple) or len(interactions) != 2:
            raise TypeError("The interactions parameter must be a tuple of two strings representing the particle types.")
        if not all(isinstance(type_name, str) for type_name in interactions):
            raise TypeError("All interaction type names must be strings.")
        
        self._epsilon = epsilon
        self._sigma = radii[0] + radii[1]
        self._n = n
        self._interactions = interactions
        self._periodic_boundary_conditions = periodic_boundary_conditions
        self._max_radius = -math.inf * length_unit
        self._lennard_jones_potential = self._set_up_lennard_jones_potential()

    def _set_up_lennard_jones_potential(self) -> CustomNonbondedForce:
        """Set up the basic functional form of the Jennard Jones potential"""
        lennard_jones_potential = CustomNonbondedForce(
            "select(flag1 * flag2 * (1 - type1_match) * (1 - type2_match),"
            "0, 4 * epsilon * (an^2 - an + 1/4));"
            "an = a^(-n);"
            "a = r / sigma;"
        )
        lennard_jones_potential.addGlobalParameter("epsilon", self._epsilon.value_in_unit(energy_unit))
        lennard_jones_potential.addGlobalParameter("sigma", self._sigma.value_in_unit(length_unit))
        lennard_jones_potential.addGlobalParameter("n", self._n)
        lennard_jones_potential.addPerParticleParameter("type1_match")
        lennard_jones_potential.addPerParticleParameter("type2_match")
        lennard_jones_potential.addPerParticleParameter("flag")
        return lennard_jones_potential
    
    def add_particle(self, type: str, substrate_flag: bool = False) -> None:
        """
        Add a colloid with a given radius to the system.

        If the substrate flag is True, the colloid is considered to be a substrate particle. Substrate particles do
        not interact with each other. In this class, this is achieved by setting the flag per-particle parameter to 1
        for substrate particles and to 0 for non-substrate particles. This flag is used in the algebraic expression of
        the steric and electrostatic potentials. Interaction groups would also work but are considerably slower
        (see https://github.com/openmm/openmm/issues/2698).

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param substrate_flag:
            Whether the colloid is a substrate particle.
        :type substrate_flag: bool

        :raises TypeError:
            If the radius is not a Quantity with a proper unit.
        :raises ValueError:
            If the radius is not greater than zero.
        :raises RuntimeError:
            If the method yield_potentials was called before this method (via the abstract base class).
        """
        super().add_particle()
        if not isinstance(type, str):
            raise TypeError("argument type must be a string representing the particle type")
        
        type_flag1 = 0
        type_flag2 = 0
        if type == self._interactions[0]:
            type_flag1 = 1
        if type == self._interactions[1]:
            type_flag2 = 1
        
        self._lennard_jones_potential.addParticle(type_flag1, type_flag2, int(substrate_flag))

    def yield_potentials(self) -> Iterator[CustomNonbondedForce]:
        """
        Generate the lennard jones pair potential between colloids in a solution in an openmm system.

        This method has to be called after the method add_particle is called for every particle in the system.

        :return:
            A generator that yields the lennard_jones potential handled by this class.
        :rtype: Iterator[CustomNonbondedForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        assert not math.isinf(self._max_radius.value_in_unit(length_unit))

        r_cut = (2 ** (1 / self._n) * self._sigma).value_in_unit(length_unit)

        if self._periodic_boundary_conditions:
            self._lennard_jones_potential.setNonbondedMethod(self._lennard_jones_potential.CutoffPeriodic)
        else:
            self._lennard_jones_potential.setNonbondedMethod(self._lennard_jones_potential.CutoffNonPeriodic)
        self._lennard_jones_potential.setCutoffDistance(r_cut)
        self._lennard_jones_potential.setUseLongRangeCorrection(False)
        self._lennard_jones_potential.setUseSwitchingFunction(False)

        yield self._lennard_jones_potential

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
        self._lennard_jones_potential.addExclusion(particle_one, particle_two)
