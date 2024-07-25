import math
from typing import Iterator
from openmm import unit
from openmm import CustomNonbondedForce
from colloids.abstracts import ColloidPotentialsAbstract
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters

class DepletionPotential(ColloidPotentialsAbstract):

    """
    This class sets up the depletion potential between colloids in a solution with a nonadsorbing polymer background. 
    Since the attractive force arises from the fact that the polymer molecules are depleted at the surface of the colloids, 
    the force is called the depletion force. The depletion force is well-modeled by the Asakura-Oosawa potential. 
    To completely describe the pair potentials in a system of colloids within solution of nonadsorbing polymers, this attractive
    depletion force can be paired with the repulsive force as described by the Alexander-de Gennes polymer brush model 
    between two colloids. (See ColloidPotentialsParameters() for more information.)

    The cutoff distance for the depletion potential is set to max(sigma_colloid) + sigma_depletant where sigma_colloid is the 
    diameter of the largest particle in the system plus two lengths of the polymer brush, and sigma_depletant is the diameter
    of the depletant plus two lengths of the polymer brush. A switching function is used to make the potential and forces go 
    smoothly to 0 at the cutoff distance. The cutoff can be set to be periodic or non-periodic.

    :param depletion_phi:
        The number density of polymers in the solution.
        The value must be between 0 and 1.
    :type depletion_phi: float
    :param depletant_radius:
        The "radius" of the polymers, if treated as hard spheres.
        The unit of the depletant_radius must be compatible with nanometers and the value must be greater than zero.
    :type depletant_radius: unit.Quantity
    :param brush_length:
        The thickness of the polymer brush as described by the Alexander-de Gennes polymer brush model.
        (See ColloidPotentialsParameters() for more information.)
        The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 10.0 nanometers.
    :type brush_length: unit.Quantity
    :param periodic_boundary_conditions:
        Whether this force should use periodic cutoffs for the depletion potential.
    :type periodic_boundary_conditions: bool

    """
    
    _nanometer = unit.nano * unit.meter
    
    def __init__(self, depletion_phi: float, depletant_radius: unit.Quantity, 
                colloid_potentials_parameters: ColloidPotentialsParameters = ColloidPotentialsParameters(), 
                periodic_boundary_conditions: bool = True):
        """Constructor of the DepletionPotential class."""
        
        super().__init__(colloid_potentials_parameters, periodic_boundary_conditions)

        self._depletion_phi = depletion_phi
        self._depletant_radius = depletant_radius
        self._depletion_potential = self._set_up_depletion_potential()
        self._max_radius = -math.inf * self._nanometer


    def _set_up_depletion_potential(self) -> CustomNonbondedForce:
        """Set up the basic functional form of the Asakura-Oosawa depletion potential for a colloidal solution 
        in a background of non-adsorbing polymers."""

        depletion_potential = CustomNonbondedForce(
            "step(sigma_colloid + sigma_depletant - r) * "
            "(AO_prefactor * (1 - term1 + term2));"
            "AO_prefactor =  -phi * (1+q)^3/q^3;"
            "term1 = 3*r/ (2 * sigma_colloid * (1+q));"
            "term2 = r^3 / (2 * sigma_colloid^3 *(1+q)^3);"
            "q = sigma_depletant/sigma_colloid;"
            "sigma_colloid = ((2 * radius1) + 2*brush_length);"
            "sigma_depletant = ((2 * depletant_radius) + 2*brush_length);"
        )

        depletion_potential.addGlobalParameter("phi", (self._depletion_phi))
        
        depletion_potential.addGlobalParameter("depletant_radius",
                                            self._depletant_radius.value_in_unit(self._nanometer))
        
        depletion_potential.addGlobalParameter("brush_length",
                                            self._parameters.brush_length.value_in_unit(self._nanometer))

        depletion_potential.addPerParticleParameter("radius")

        return depletion_potential
    
    def add_particle(self, radius: unit.Quantity, surface_potential: unit.Quantity) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity


        :raises TypeError:
            If the radius is not a Quantity with a proper unit (via the abstract base class).
        :raises ValueError:
            If the radius is not greater than zero (via the abstract base class).
        :param surface_potential:
            The surface potential of the colloid.
            The unit of the surface_potential must be compatible with millivolts.
        :type surface_potential: unit.Quantity
        :raises RuntimeError:
            If the method yield_potentials was called before this method (via the abstract base class).
        """
        super().add_particle(radius, surface_potential)

        if radius.in_units_of(self._nanometer) > self._max_radius:
            self._max_radius = radius.in_units_of(self._nanometer)
        
        self._depletion_potential.addParticle([radius.value_in_unit(self._nanometer)])

    def yield_potentials(self) -> Iterator[CustomNonbondedForce]:
        """
        Generate the depletion pair potential between colloids in a solution in an openmm system.

        This method has to be called after the method add_particle is called for every particle in the system.

        :return:
            A generator that yields the depletion potential handled by this class.
        :rtype: Iterator[CustomNonbondedForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        assert not math.isinf(self._max_radius.value_in_unit(self._nanometer))

        if self._periodic_boundary_conditions:
            self._depletion_potential.setNonbondedMethod(self._depletion_potential.CutoffPeriodic)
        else:
            self._depletion_potential.setNonbondedMethod(self._depletion_potential.CutoffNonPeriodic)
        self._depletion_potential.setCutoffDistance(
            ((2.0 * self._max_radius + 2.0 * self._parameters.brush_length)
            + (2.0 * self._depletant_radius + 2.0 * self._parameters.brush_length)).value_in_unit(self._nanometer))
        self._depletion_potential.setUseLongRangeCorrection(False)
        self._depletion_potential.setUseSwitchingFunction(False)

        yield self._depletion_potential

    
    
