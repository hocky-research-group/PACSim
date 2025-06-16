import math
from typing import Iterator
from openmm import unit
from openmm import CustomNonbondedForce
from colloids.abstracts import OpenMMNonbondedPotentialAbstract
from colloids.units import energy_unit, length_unit, temperature_unit


class LennardJonesPotential(OpenMMNonbondedPotentialAbstract):
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
        The radii of the colloids in the system, which is a dictionary with the particle type as key and the radius as value.
        The unit of the radius must be compatible with nanometers and the value must be greater than zero.
    :type radii: dict[str, unit.Quantity]
    :param n:
        The order of the Lennard-Jones potential, which is typically 6 for the attractive part.
        The unit of the order must be compatible with dimensionless and the value must be greater than zero.
    :type n: float
    :param interactions:
        The interactions between the particles, which is a list of tuples containing the names of the particles that interact 
        and the type of interaction. The first two elements of the tuple are the names of the particles that interact, and the 
        third element is a string representing the type of interaction (attractive or repulsive).
    :type interactions: tuple[str, str]
    :param periodic_boundary_conditions:
        Whether this force should use periodic cutoffs for the lennard jones potential.
    :type periodic_boundary_conditions: bool
    """
    
    def __init__(self, radii: dict[str, unit.Quantity], interactions: tuple[str, str, str, unit.Quantity],
                 n: float, periodic_boundary_conditions: bool = True):
        """Constructor of the LennardJonesPotential class."""
        super().__init__()


        if not isinstance(n, (int, float)) or n <= 0:
            raise TypeError("The order parameter must be a positive number.")
        if not all(isinstance(interaction, tuple) or len(interaction) != 4 for interaction in interactions):
            raise TypeError("The interactions parameter must be a tuple of three strings representing the particle types.")
        if not all(isinstance(type_name, str) for interaction in interactions for type_name in interaction[:2]):
            raise TypeError("All interaction type names must be strings.")
        
        self._n = n
        self.radii = radii
        self._interactions = interactions
        self._periodic_boundary_conditions = periodic_boundary_conditions
        self._max_radius = -math.inf * length_unit
        self._lennard_jones_potentials: list[CustomNonbondedForce] = [self._set_up_lennard_jones_potential(interaction) 
                                                                      for  interaction in interactions]

    def _set_up_lennard_jones_potential(self, interaction: str) -> CustomNonbondedForce:
        """Set up the basic functional form of the Jennard Jones potential"""
        assert interaction[2] in ["attractive", "repulsive"], "Interaction type must be either 'attractive' or 'repulsive'."

        interaction_numbers = f'{interaction[0]}{interaction[1]}'

        if interaction[2] == "attractive":
            lennard_jones_potential = CustomNonbondedForce(
                "select(flag1 * flag2 * (1 - (1 - (1 - type1_match1) * (1 - type2_match2))"
                " * (1 - (1 - type1_match2) * (1 - type2_match1))),"
                f"0, 4 * epsilon{interaction_numbers} * (an^2 - an));"
                "an = a^(-n);"
                f"a = r / sigma{interaction_numbers};"
            )

        if interaction[2] == "repulsive":
            lennard_jones_potential = CustomNonbondedForce(
                "select(flag1 * flag2 * (1 - (1 - (1 - type1_match1) * (1 - type2_match2))"
                " * (1 - (1 - type1_match2) * (1 - type2_match1))),"
                f"0, 4 * epsilon{interaction_numbers} * (an^2 - an + 1/4));"
                "an = a^(-n);"
                f"a = r / sigma{interaction_numbers};"
            )

        sigma = self.radii[interaction[0]] + self.radii[interaction[1]]

        lennard_jones_potential.addGlobalParameter(f"epsilon{interaction_numbers}", interaction[3])
        lennard_jones_potential.addGlobalParameter(f"sigma{interaction_numbers}", sigma.value_in_unit(length_unit))
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
        
        for i, lennard_jones_potential in enumerate(self._lennard_jones_potentials):
            type_flag1 = 0
            type_flag2 = 0
            if type == self._interactions[i][0]:
                type_flag1 = 1
            if type == self._interactions[i][1]:
                type_flag2 = 1
            lennard_jones_potential.addParticle([type_flag1, type_flag2, int(substrate_flag)])

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

        for i, lennard_jones_potential in enumerate(self._lennard_jones_potentials):
            sigma = self.radii[self._interactions[i][0]] + self.radii[self._interactions[i][1]]

            if self._interactions[i][2] == "attractive":
                r_min = (1 + math.log(2.0) / self._n) * sigma
                r_cut = (3 * r_min - 2 * sigma).value_in_unit(length_unit)

            elif self._interactions[i][2] == "repulsive":
                r_cut = (2 ** (1 / self._n) * sigma).value_in_unit(length_unit)

            if self._periodic_boundary_conditions:
                lennard_jones_potential.setNonbondedMethod(lennard_jones_potential.CutoffPeriodic)
            else:
                lennard_jones_potential.setNonbondedMethod(lennard_jones_potential.CutoffNonPeriodic)
            lennard_jones_potential.setCutoffDistance(r_cut)
            lennard_jones_potential.setUseLongRangeCorrection(False)
            lennard_jones_potential.setUseSwitchingFunction(False)

            yield lennard_jones_potential

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
        for lennard_jones_potential in self._lennard_jones_potentials:
            lennard_jones_potential.addExclusion(particle_one, particle_two)
