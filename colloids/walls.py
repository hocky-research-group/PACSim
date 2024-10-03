from typing import Iterator, Optional, Sequence
from openmm import CustomExternalForce, unit
from colloids.abstracts import OpenMMPotentialAbstract
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters
import math
import warnings

class SubstrateWall(OpenMMPotentialAbstract):
    """
    This class sets up a charged wall for simulations with implicit substrate using the CustomExternalForce class 
    of OpenMM. This is an alternative to explicitly modeling the substrate using a layer of fixed particles, and is
    designed to reduce computational costs of modeling substrate. Either implicit or explicit substrate may be used, 
    but not both. 

    A substrate wall can only be used when all SLJ walls are active. The bottom wall in the z direction will be replaced
    by the substrate wall.
    
    The substrate wall is placed at the bottom of the simulation box in the z direction, at +-wall_distance / 2. 

    :param wall_distance:
        A distance specifying the dimensions of the simulation box in the z direction.
        This is used to determine the location of the substrate wall at +-wall_distance/2 at the bottom of the simulation box.
        The unit of any wall distance must be compatible with nanometer and the value must be greater than zero.
    :type wall_distance: Optional[unit.Quantity]

    :raises TypeError:
        If the wall distance for an active substrate wall is not a Quantity with a proper unit.
        If the wall charge for an active substrate wall is not a Quantity with a proper unit (via abstract base class)

    :raises ValueError:
        If the wall distance for an active substrate wall is not greater than zero.
        If the wall distance or wall charge is specified when the substrate wall is inactive.
        If the wall distance or wall charge is not specified for an active substrate wall.
        If a substrate wall is active while an explicit substrate is also being used. 
    """

    _nanometer = unit.nano * unit.meter

    def __init__(self, colloid_potentials_parameters: ColloidPotentialsParameters, 
                 substrate_type: Optional[str], wall_distance: Optional[unit.Quantity], 
                 wall_charge: Optional[unit.Quantity], use_log: bool = False) -> None:
        """Constructor of the SubstrateWall class."""
        super().__init__()
    
        if substrate_type == "wall":
            if wall_distance is None:
                raise ValueError("wall distance must be specified when substrate wall is on")
            if not wall_distance.unit.is_compatible(self._nanometer):
                raise TypeError("wall distance must have a unit that is compatible with nanometers")
            if not wall_distance.value_in_unit(self._nanometer) > 0.0:
                raise ValueError("wall distance must have a value greater than zero")
        else:
            if wall_distance is not None:
                raise ValueError("wall distance must not be specified for when substrate wall is inactive")
       

        self._parameters = colloid_potentials_parameters
        self._wall_distance = wall_distance
        self._wall_charge = wall_charge
        self._use_log = use_log

        self._substrate_wall_potential = self._set_up_substrate_wall_potential()

    def _set_up_substrate_wall_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the substrate wall."""
        
        """Set up the basic functional form of the steric potential from the Alexander-de Gennes polymer brush model."""
        steric_potential = (
            "step(two_l - h) * "
            "steric_prefactor * rs / 2.0 * brush_length * brush_length * ("
            "28.0 * ((two_l / h)^0.25 - 1.0) "
            "+ 20.0 / 11.0 * (1.0 - (h / two_l)^2.75)"
            "+ 12.0 * (h / two_l - 1.0))); "
            "h = r - rs;"
            "rs = radius + z;"
            "two_l = 2.0 * brush_length"
        )

        """Set up the basic functional form of the electrostatic potential from DLVO theory."""
        if self._use_log:
            electrostatic_potential = (
                "electrostatic_prefactor * radius_avg * psi * wall_charge * log(1.0 + exp(-h / debye_length))); "
                "radius_avg = 2.0 / (1.0 / radius + 1.0 / radius_substrate);"
                "h = r - rs;"
                "rs = radius + z"
            )
        else:
            electrostatic_potential = (
                "electrostatic_prefactor * radius_avg * psi * wall_charge * exp(-h / debye_length)); "
                "radius_avg = 2.0 / (1.0 / radius + 1.0 / z);"
                "h = r - rs;"
                "rs = radius + z"
            )

        wall_string = "+".join(pot for pot in zip([steric_potential, electrostatic_potential]))
        assert wall_string

        substrate_wall_potential = CustomExternalForce(wall_string)

         # Steric prefactor is k_B * T * 16 * pi * sigma^(3/2) / 35 (see Hocky paper)
        substrate_wall_potential.addGlobalParameter(
            "steric_prefactor",
            (unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature
            * 16.0 * math.pi * (self._parameters.brush_density ** (3 / 2)) / 35.0
            * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole / (self._nanometer ** 3))
        )
        # Brush length L (see Hocky paper)
        substrate_wall_potential.addGlobalParameter("brush_length",
                                            self._parameters.brush_length.value_in_unit(self._nanometer))
       
       # Electrostatic prefactor is 2 * pi * epsilon 
        substrate_wall_potential.addGlobalParameter(
            "electrostatic_prefactor",
            (2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
             * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
                unit.kilojoule_per_mole / (self._nanometer * self._millivolt ** 2)))
        substrate_wall_potential.addGlobalParameter("debye_length",
                                                   self._parameters.debye_length.value_in_unit(self._nanometer))
        
        #dlvo_walls_potential.addGlobalParameter("radius_substrate")
        substrate_wall_potential.addGlobalParameter("wall_charge", self._wall_charge)
        
        substrate_wall_potential.addPerParticleParameter("radius")
        # Psi should be given in millivolts.
        substrate_wall_potential.addPerParticleParameter("psi")


    def add_particle(self, index: int, radius: unit.Quantity, surface_potential: unit.Quantity) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param index:
            The index of the particle in the OpenMM system.
        :type index: int
        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param surface_potential:
            The surface potential of the colloid.
            The unit of the surface_potential must be compatible with millivolts.
        :type surface_potential: unit.Quantity

        
        :raises TypeError:
            If the radius or surface potential is not a Quantity with a proper unit.
        :raises ValueError:
            If the radius is not greater than zero.
        :raises RuntimeError:
            If this method is called after the yield_potentials method (via the abstract base class).
        """
        super().add_particle(radius, surface_potential)
        if not radius.unit.is_compatible(self._nanometer):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        if not surface_potential.unit.is_compatible(unit.milli * unit.volt):
            raise TypeError("argument surface_potential must have a unit that is compatible with volts") 
        if self._wall_distance is not None:
            if not self._wall_distance / 2.0 > radius * 2 ** (1 / 6) + radius - 1.0 * self._nanometer:
                raise ValueError("The colloid radius leads to a cutoff radius * 2^(1/6) + radius - 1 in the "
                                    "substrate wall that exceeds half of the wall distance.")

        self._substrate_wall_potential.addParticle(index, radius, surface_potential)

    def yield_potentials(self) -> Iterator[CustomExternalForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the substrate wall.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields the substrate wall handled by this class.
        :rtype: Iterator[CustomExternalForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        yield self._substrate_wall_potential



class ShiftedLennardJonesWalls(OpenMMPotentialAbstract):
    """
    This class sets up the shifted Lennard-Jones potentials for closed-wall simulations using the CustomExternalForce
    class of openmm.

    The shifted Lennard-Jones potential as a wall follows the implementation of hoomd (see
    https://hoomd-blue.readthedocs.io/en/v2.9.4/module-md-wall.html#hoomd.md.wall.slj).

    This class allows to independently switch on walls in the x, y, and z directions. The walls are placed at
    +-wall_distance / 2 for every specified direction with its specified wall_distance.

    The shifted Lennard-Jones potential acts on colloid particles within a certain cutoff distance of every wall. This
    cutoff distance depends on the particle radius and is given by r_cut - delta, where r_cut = radius * 2^(1/6) and
    delta = radius - 1. Outside of this range, the external force acting on a particle is 0.

    The Lennard-Jones potential is shifted so that it starts smoothly at zero at the cutoff distance.

    The shifted Lennard-Jones potential as a function of the distance r to the wall is given by:
    slj(r) = 4 * epsilon * ((radius / (r - delta))^12 - alpha * (radius / (r - delta))^6)
             - 4 * epsilon * ((radius / r_cut)^12 - alpha * (radius / r_cut)^6)
    

    :param wall_distances:
        A list of three distances specifying the dimensions of the simulation box in the x, y, and z directions.
        This is used to determine the location of the SLJ walls at +-wall_distance/2 for every active wall direction.
        For any inactive wall direction (see wall_directions parameter), the corresponding wall distance must be None.
        For any active wall direction, the corresponding wall distance must be specified.
        The unit of any wall distance must be compatible with nanometer and the value must be greater than zero.
    :type wall_distances: Sequence[Optional[unit.Quantity]]
    :param epsilon:
        The unshifted Lennard-Jones potential well-depth.
        The unit of the epsilon must be compatible with kilojoules_per_mole and the value must be greater than zero.
    :type epsilon: unit.Quantity
    :param alpha:
        Factor determining the strength of the attractive part of the Lennard-Jones potential.
        This factor has to satisfy 0 <= alpha <= 1.
        Note that the force of this potential is only continuous if alpha = 1.
    :type alpha: float
    :param wall_directions:
        A list of three booleans indicating whether the walls in the x, y, and z directions are active.
        Defaults to [False, False, False].
    :type wall_directions: list[bool]
    :param use_substrate:
        A boolean indicating whether the bottom wall is replaced by a substrate.
        This is only possible if all wall directions are active.
        Defaults to False.
    :type use_substrate: bool

    :raises TypeError:
        If epsilon or any wall distance for an active wall direction is not a Quantity with a proper unit.

    :raises ValueError:
        If epsilon or any wall distance for an active wall direction is not greater than zero.
        If alpha is not in the interval [0, 1].
        If no wall direction is active.
        If not exactly three wall directions are specified.
        If not exactly three wall distances are specified.
        If a wall distance is specified for an inactive wall direction.
        If a wall distance is not specified for an active wall direction.
        If not all wall directions are active if a substrate is used.
    """

    _nanometer = unit.nano * unit.meter

    def __init__(self, wall_distances: Sequence[Optional[unit.Quantity]], epsilon: unit.Quantity, alpha: float,
                 wall_directions: Sequence[bool] = (True, True, True), use_substrate: bool = False) -> None:
        """Constructor of the ShiftedLennardJonesWalls class."""
        super().__init__()

        if not any(wall_directions):
            raise ValueError("at least one wall direction must be active")
        if len(wall_directions) != 3:
            raise ValueError("wall directions must be specified for three dimensions")
        if len(wall_distances) != 3:
            raise ValueError("wall distances must be specified for three dimensions")
        for wdir, wdist in zip(wall_directions, wall_distances):
            if wdir:
                if wdist is None:
                    raise ValueError("wall distance must be specified for any active wall direction")
                if not wdist.unit.is_compatible(self._nanometer):
                    raise TypeError("any wall distance must have a unit that is compatible with nanometers")
                if not wdist.value_in_unit(self._nanometer) > 0.0:
                    raise ValueError("any wall distance must have a value greater than zero")
            else:
                if wdist is not None:
                    raise ValueError("wall distance must not be specified for inactive wall direction")
        if not epsilon.unit.is_compatible(unit.kilojoule_per_mole):
            raise TypeError("argument epsilon must have a unit that is compatible with kilojoules per mole")
        if not epsilon.value_in_unit(unit.kilojoule_per_mole) > 0.0:
            raise ValueError("argument epsilon must have a value greater than zero")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("argument alpha must satisfy 0 <= alpha <= 1")
        if alpha != 1.0:
            warnings.warn("The force of the shifted Lennard-Jones potential as a wall is only continuous if alpha = 1.")
        if use_substrate:
            if not all(wall_directions):
                raise ValueError("all wall directions must be active if a substrate is used")

        self._wall_distances = wall_distances
        self._epsilon = epsilon
        self._alpha = alpha
        self._wall_directions = wall_directions
        self._use_substrate = use_substrate
        self._slj_potential = self._set_up_slj_potential()

    def _set_up_slj_potential(self) -> CustomExternalForce:
        """Set up the basic functional form of the shifted Lennard Jones potential."""

        slj_x = ("step(periodicdistance(x, 0, 0, 0, 0, 0) - cutoff_x) * ("
                 "four_epsilon * "
                 "((radius / (wall_distance_x_over_two_minus_delta - periodicdistance(x, 0, 0, 0, 0, 0)))^12 "
                 "- alpha * (radius / (wall_distance_x_over_two_minus_delta - periodicdistance(x, 0, 0, 0, 0, 0)))^6)"
                 "+ shift)")
        slj_y = ("step(periodicdistance(0, y, 0, 0, 0, 0) - cutoff_y) * ("
                 "four_epsilon * "
                 "((radius / (wall_distance_y_over_two_minus_delta - periodicdistance(0, y, 0, 0, 0, 0)))^12 "
                 "- alpha * (radius / (wall_distance_y_over_two_minus_delta - periodicdistance(0, y, 0, 0, 0, 0)))^6)"
                 "+ shift)")
        slj_z = ("step(periodicdistance(0, 0, z, 0, 0, 0) - cutoff_z) * ("
                 "four_epsilon * "
                 "((radius / (wall_distance_z_over_two_minus_delta - periodicdistance(0, 0, z, 0, 0, 0)))^12 "
                 "- alpha * (radius / (wall_distance_z_over_two_minus_delta - periodicdistance(0, 0, z, 0, 0, 0)))^6)"
                 "+ shift)")
        # Using periodicdistance switches on periodic boundary conditions in the OpenMM system.
        # If there are walls in all directions, we don't want periodic boundary conditions though.
        if all(self._wall_directions):
            slj_x = slj_x.replace("periodicdistance(x, 0, 0, 0, 0, 0)", "abs(x)")
            slj_y = slj_y.replace("periodicdistance(0, y, 0, 0, 0, 0)", "abs(y)")
            slj_z = slj_z.replace("periodicdistance(0, 0, z, 0, 0, 0)", "abs(z)")
            if self._use_substrate:
                # Only the bottom wall is replaced by a substrate.
                # The top wall is still a shifted Lennard-Jones wall.
                slj_z = slj_z.replace("abs(z)", "z")

        slj_string = "+".join(slj for slj, wdir in zip([slj_x, slj_y, slj_z], self._wall_directions) if wdir)
        assert slj_string

        slj_potential = CustomExternalForce(slj_string)
        slj_potential.addGlobalParameter("four_epsilon",
                                         4.0 * self._epsilon.value_in_unit(unit.kilojoule_per_mole))
        slj_potential.addGlobalParameter("alpha", self._alpha)
        slj_potential.addPerParticleParameter("radius")
        slj_potential.addPerParticleParameter("shift")

        if self._wall_directions[0]:
            slj_potential.addPerParticleParameter("wall_distance_x_over_two_minus_delta")
            slj_potential.addPerParticleParameter("cutoff_x")
        if self._wall_directions[1]:
            slj_potential.addPerParticleParameter("wall_distance_y_over_two_minus_delta")
            slj_potential.addPerParticleParameter("cutoff_y")
        if self._wall_directions[2]:
            slj_potential.addPerParticleParameter("wall_distance_z_over_two_minus_delta")
            slj_potential.addPerParticleParameter("cutoff_z")

        return slj_potential

    def add_particle(self, index: int, radius: unit.Quantity) -> None:
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
        if not radius.unit.is_compatible(self._nanometer):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        for wall_distance in self._wall_distances:
            if wall_distance is not None:
                if not wall_distance / 2.0 > radius * 2 ** (1 / 6) + radius - 1.0 * self._nanometer:
                    raise ValueError("The colloid radius leads to a cutoff radius * 2^(1/6) + radius - 1 in the "
                                     "shifted Lennard-Jones wall that exceeds half of the wall distance.")
        rcut = (2.0 ** (1.0 / 6.0)) * radius
        per_particle_parameters = [
            radius.value_in_unit(self._nanometer),  # radius
            (-4.0 * self._epsilon * ((radius / rcut) ** 12 - self._alpha * (radius / rcut) ** 6)).value_in_unit(
                unit.kilojoule_per_mole)  # shift
        ]
        if self._wall_directions[0]:
            per_particle_parameters.append(
                (self._wall_distances[0] / 2.0 - radius + 1.0 * self._nanometer).value_in_unit(
                    self._nanometer)  # wall_distance_x_over_two_minus_delta
            )
            per_particle_parameters.append(
                (self._wall_distances[0] / 2.0 - rcut - radius + 1.0 * self._nanometer).value_in_unit(
                    self._nanometer)  # cutoff_x
            )
        if self._wall_directions[1]:
            per_particle_parameters.append(
                (self._wall_distances[1] / 2.0 - radius + 1.0 * self._nanometer).value_in_unit(
                    self._nanometer)  # wall_distance_y_over_two_minus_delta
            )
            per_particle_parameters.append(
                (self._wall_distances[1] / 2.0 - rcut - radius + 1.0 * self._nanometer).value_in_unit(
                    self._nanometer)  # cutoff_y
            )
        if self._wall_directions[2]:
            per_particle_parameters.append(
                (self._wall_distances[2] / 2.0 - radius + 1.0 * self._nanometer).value_in_unit(
                    self._nanometer)  # wall_distance_z_over_two_minus_delta
            )
            per_particle_parameters.append(
                (self._wall_distances[2] / 2.0 - rcut - radius + 1.0 * self._nanometer).value_in_unit(
                    self._nanometer)  # cutoff_z
            )

        self._slj_potential.addParticle(index, per_particle_parameters)

    def yield_potentials(self) -> Iterator[CustomExternalForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the shifted Lennard-Jones walls.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields shifted Lennard-Jones walls handled by this class.
        :rtype: Iterator[CustomExternalForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        yield self._slj_potential
