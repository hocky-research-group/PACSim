import math
from typing import Iterator
import openmm
from openmm import unit
from openmm import CustomNonbondedForce
from colloids.abstracts import OpenMMPotentialAbstract

class DepletionPotentialsAlgebraic(OpenMMPotentialAbstract):
    
    _nanometer = unit.nano * unit.meter
    
    def __init__(self, phi: float, depletant_radius: unit.Quantity, brush_length: unit.Quantity):
        """Constructor of the DepletionPotentialsAlgebraic class."""
        
        super().__init__()

        self._phi = phi
        self._depletant_radius = depletant_radius
        self._brush_length = brush_length
        self._depletion_potential = self._set_up_depletion_potential()
         #self._steric_potential = self._set_up_steric_potential()
        self._max_radius = -math.inf * self._nanometer
        
    '''def _set_up_steric_potential(self) -> CustomNonbondedForce:
        """Set up the basic functional form of the steric potential from the Alexander-de Gennes polymer 
        brush model."""
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
        return steric_potential'''

    def _set_up_depletion_potential(self) -> CustomNonbondedForce:
        """Set up the basic functional form of the Asakura-Oosawa potential (generalized to asymmetric sphere) 
        for a colloidal solution in a background of non-adsorbing polymers."""

        depletion_potential = CustomNonbondedForce(
            "step(sigmaD1 + sigmaD2 + 2*depletant_radius - rcc) * "
            "depletion_prefactor * (q1+q2+2-n)**2*(n+2*(q1+q2+2)-3/n*(q1**2+q2**2-2*q1*q2));"
            "sigmaD1 = brush_length + radius1;"
            "sigmaD2 = brush_length + radius2;"
            "rcc = h + radius1 + radius2;"
            "q1 = sigmaD1/depletant_radius;"
            "q2 = sigmaD2/depletant_radius;"
            "n = rcc/depletant_radius;"
        )

        depletion_potential.addGlobalParameter("depletion_prefactor", (self._phi / 16.0))
        
        depletion_potential.addGlobalParameter("depletant_radius",
                                            self._depletant_radius.value_in_unit(self._nanometer))
        
        depletion_potential.addGlobalParameter("brush_length",
                                            self._brush_length.value_in_unit(self._nanometer))

        depletion_potential.addPerParticleParameter("radius")

        return depletion_potential
    
    def add_particle(self, radius: unit.Quantity) -> None:
        """
        Add a colloid with a given radius to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity


        :raises TypeError:
            If the radius is not a Quantity with a proper unit (via the abstract base class).
        :raises ValueError:
            If the radius is not greater than zero (via the abstract base class).
        :raises RuntimeError:
            If the method yield_potentials was called before this method (via the abstract base class).
        """
        super().add_particle(radius)

        if radius.in_units_of(self._nanometer) > self._max_radius:
            self._max_radius = radius.in_units_of(self._nanometer)

    def yield_potentials(self) -> Iterator[CustomNonbondedForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the steric (brush) and depletion pair
        potentials between colloids in a solution in an openmm system.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            A generator that yields the steric and AO potentials handled by this class.
        :rtype: Iterator[CustomNonbondedForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method (via the abstract base class).
        """
        super().yield_potentials()
        assert not math.isinf(self._max_radius.value_in_unit(self._nanometer))


        #yield self._steric_potential
        yield self._depletion_potential

if __name__ == '__main__':
    DepletionPotentialsAlgebraic()
    
    
