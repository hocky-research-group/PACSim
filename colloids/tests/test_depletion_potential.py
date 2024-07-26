from openmm import Context, LangevinIntegrator, Platform, System, unit, Vec3
import pytest
from colloids.depletion_potential import DepletionPotential
from colloids import ColloidPotentialsParameters
import numpy as np


class TestParameters(object):

    @pytest.fixture
    def radius_one(self):
        return 105.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def radius_two(self):
        return 85.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def depletant_radius(self):
        return 5.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def brush_density(self):
        return 0.09 / ((unit.nano * unit.meter) ** 2)

    @pytest.fixture
    def brush_length(self):
        return 10.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def phi(self):
        return 0.5

    @pytest.fixture
    def debye_length(self):
        return 5.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def dielectric_constant(self): 
        return 80.0

    @pytest.fixture
    def temperature(self):
        return 298.0 * unit.kelvin

    @pytest.fixture
    def colloid_potentials_parameters(self, brush_density, brush_length, debye_length, temperature,
                                      dielectric_constant):
        return ColloidPotentialsParameters(brush_density=brush_density, brush_length=brush_length,
                                           debye_length=debye_length, temperature=temperature,
                                           dielectric_constant=dielectric_constant)

    @pytest.fixture
    def test_positions(self):
        return np.linspace(0.0 , 25.0, num=1000) * unit.nanometer

    @pytest.fixture
    def maximum_surface_separation(self):
        return 100.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def side_length(self, radius_one, radius_two, maximum_surface_separation):
        # Make system very large so that we do not care about periodic boundaries.
        return 20.0 * (maximum_surface_separation + 2.0 * max(radius_one, radius_two))

    @pytest.fixture
    def openmm_system(self, side_length):
        system = System()
        # Make system very large so that we do not care about periodic boundaries.
        side_length_value = side_length.value_in_unit(unit.nano * unit.meter)
        system.setDefaultPeriodicBoxVectors(Vec3(side_length_value, 0.0, 0.0),
                                            Vec3(0.0, side_length_value, 0.0),
                                            Vec3(0.0, 0.0, side_length_value))
        return system

    @pytest.fixture
    def depletion_potential(self, phi, depletant_radius, colloid_potentials_parameters):
        return DepletionPotential(phi, depletant_radius, colloid_potentials_parameters)

    @pytest.fixture
    def openmm_platform(self):
        return Platform.getPlatformByName("Reference")

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)


class TestDepletionPotentialExceptions(TestParameters):
    def test_exception_radius(self, depletion_potential, radius_one):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            depletion_potential.add_particle(
                radius=radius_one / ((unit.nano * unit.meter) ** 2))
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            depletion_potential.add_particle(
                radius=-radius_one)


    def test_exception_no_particles_added(self, depletion_potential):
        with pytest.raises(RuntimeError):
            for _ in depletion_potential.yield_potentials():
                pass

    def test_exception_add_particle_after_yield_potentials(self, depletion_potential, radius_one):
        depletion_potential.add_particle(radius=radius_one)
        for _ in depletion_potential.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            depletion_potential.add_particle(radius=radius_one)


    def test_exception_phi(self, colloid_potentials_parameters):
        # Test exception on negative phi
        with pytest.raises(ValueError):
            DepletionPotential(depletion_phi = -0.5, depletant_radius=5* (unit.nano * unit.meter), 
                                colloid_potentials_parameters=colloid_potentials_parameters)
                                        #brush_length=10* (unit.nano * unit.meter))

        # Test exception on phi > 1
        with pytest.raises(ValueError):
            DepletionPotential(depletion_phi = 2.0, depletant_radius=5* (unit.nano * unit.meter),
                                colloid_potentials_parameters=colloid_potentials_parameters)
                                       # brush_length=10* (unit.nano * unit.meter))


    def test_exception_depletant_radius(self, colloid_potentials_parameters):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            DepletionPotential(depletion_phi = 0.5, depletant_radius=5 * unit.joule, 
                                colloid_potentials_parameters=colloid_potentials_parameters)
                                        #brush_length=10* (unit.nano * unit.meter))

        # Test exception on negative depletant radius
        with pytest.raises(ValueError):
            DepletionPotential(depletion_phi = 0.5, depletant_radius=-5* (unit.nano * unit.meter), 
                                colloid_potentials_parameters=colloid_potentials_parameters)
                                      #  brush_length=10* (unit.nano * unit.meter))

# noinspection DuplicatedCode
class TestDepletionPotentialForTwoParticles(TestParameters):
    @pytest.fixture(autouse=True)
    def add_two_particles(self, openmm_system, depletion_potential, radius_one, radius_two):
        openmm_system.addParticle(mass=1.0)
        depletion_potential.add_particle(radius=radius_one)
        openmm_system.addParticle(mass=1.0)
        depletion_potential.add_particle(radius=radius_two)
        for potential in depletion_potential.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

   
    @staticmethod
    def depletion_potential_expected(h, radius_one, radius_two, brush_length, phi, depletant_radius):

        rho_colloid1 = (2 * radius_one + 2*brush_length)/2 #*unit.nanometer
        rho_colloid2 = (2 * radius_two + 2*brush_length)/2 #*unit.nanometer
        
        # size ratio 
        q1 = rho_colloid1/depletant_radius
        q2 = rho_colloid2/depletant_radius
        
        # center to center separation
        rcc = h + rho_colloid1 + rho_colloid2  
        
        n = rcc/depletant_radius
        
        return np.where(rcc <= (rho_colloid1 + rho_colloid2 + 2*depletant_radius),
                        -phi/16*(q1+q2+2-n)**2*(n+2*(q1+q2+2)-3/n*(q1**2+q2**2-2*q1*q2)),
                        0.0) 
    
    @pytest.mark.parametrize("expected_function", [(depletion_potential_expected)])

    def test_depletion_potential(self, openmm_context, test_positions, radius_one, radius_two, brush_length, phi, depletant_radius,
        expected_function):

        openmm_potentials = np.zeros(len(test_positions)) #use surface separation as test positions
        #expected_potentials = np.zeros(len(test_positions))
        for index, pos in enumerate(test_positions):
            r_value = pos + radius_one + radius_two +2*brush_length #openmm function takes input of center-to-center distance
            openmm_context.setPositions([[r_value.value_in_unit(unit.nanometer), 0.0, 0.0], [0.0, 0.0, 0.0]])
            state = openmm_context.getState(getEnergy=True)
            openmm_potentials[index] = (
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        
        expected_potentials = expected_function(test_positions, radius_one, radius_two, brush_length, phi, depletant_radius)
        
        assert openmm_potentials == pytest.approx(expected_potentials, #.value_in_unit(unit.kilojoule_per_mole), 
            rel=1.0e-7, abs=1.0e-13)
       
                 
if __name__ == '__main__':
    pytest.main([__file__])
