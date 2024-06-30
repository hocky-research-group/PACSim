from openmm import Context, LangevinIntegrator, Platform, System, unit, Vec3
import pytest
from colloids.depletion_potentials_algebraic import DepletionPotentialsAlgebraic
import numpy as np


class TestParameters(object):

    @staticmethod
    def depletion_potential_expected(h, radius_colloid, brush_length, phi, radius_depletant):

        #diameters 
        sigma_colloid = (2 * radius_colloid + 2*brush_length) 
        sigma_depletant = (2 * radius_depletant + 2*brush_length) 

        # size ratio 
        q = sigma_depletant/sigma_colloid 
        
        #surface-to-surface separation
        r = h + 2*radius_colloid + 2*brush_length

        AO_prefactor = -phi * (1+q)**3/q**3

        term1 = 3*r / (2 * sigma_colloid * (1+q))

        term2 = r**3 / (2 * sigma_colloid**3 *(1+q)**3)
        
        return np.where(r <= (sigma_colloid + sigma_depletant),
                        0.0,
                        (AO_prefactor * (1 - term1 + term2)) 
                        )

    @pytest.fixture
    def radius_one(self):
        return 325.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def radius_two(self):
        return 325.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def radius_depletant(self):
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
    def temperature(self):
        return 298.0 * unit.kelvin

    @pytest.fixture
    def test_positions(self):
        return np.linspace(15.0 , 36.0, num=1000) * unit.nanometer

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
    def depletion_potentials_algebraic(self, phi, depletant_radius, brush_length):
        return DepletionPotentialsAlgebraic(self, phi, depletant_radius, brush_length)

    # Keep this to adjust for future implementation of tabulated pair potentials
    @pytest.fixture
    def depletion_potentials(self, depletion_potentials_algebraic):
        return depletion_potentials_algebraic 

    '''@pytest.fixture
    def colloid_potentials_tabulated(self, radius_one, radius_two, surface_potential_one, surface_potential_two,
                                     colloid_potentials_parameters):
        return ColloidPotentialsTabulated(radius_one=radius_one, radius_two=radius_two,
                                          surface_potential_one=surface_potential_one,
                                          surface_potential_two=surface_potential_two,
                                          colloid_potentials_parameters=colloid_potentials_parameters, use_log=False,
                                          cutoff_factor=21.0, periodic_boundary_conditions=True)

    @pytest.fixture(params=["algebraic", "tabulated"])
    def colloid_potentials(self, colloid_potentials_algebraic, colloid_potentials_tabulated,  request):
        if request.param == "algebraic":
            return colloid_potentials_algebraic
        else:
            assert request.param == "tabulated"
            return colloid_potentials_tabulated'''


    @pytest.fixture
    def openmm_platform(self):
        return Platform.getPlatformByName("Reference")

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)


class TestDepletionPotentialsAlgebraicExceptions(TestParameters):
    def test_exception_radius(self, depletion_potentials, radius_one):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            depletion_potentials.add_particle(
                radius=radius_one / ((unit.nano * unit.meter) ** 2))
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            depletion_potentials.add_particle(
                radius=-radius_one)


    def test_exception_no_particles_added(self, depletion_potentials):
        with pytest.raises(RuntimeError):
            for _ in depletion_potentials.yield_potentials():
                pass

    def test_exception_add_particle_after_yield_potentials(self, depletion_potentials, radius_one):
        depletion_potentials.add_particle(radius=radius_one)
        for _ in depletion_potentials.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            depletion_potentials.add_particle(radius=radius_one)

    def test_exception_wrong_particle_size_ratio(self, depletion_potentials):
        # This is fine for a depletant of radius 5nm.
        depletion_potentials.add_particle(index=0, radius=236.0 * (unit.nano * unit.meter))
        # This is not fine for a depletant radius of 5 nm.
        with pytest.raises(ValueError):
            depletion_potentials.add_particle(index=1, radius=23.0 * (unit.nano * unit.meter))

    def test_exception_phi(self, depletion_potentials):
        # Test exception on negative phi
        with pytest.raises(ValueError):
            depletion_potentials(phi = -0.5, depletant_radius=5* (unit.nano * unit.meter), 
                                        brush_length=10* (unit.nano * unit.meter))

        # Test exception on phi > 1
        with pytest.raises(ValueError):
            depletion_potentials(phi = 2.0, depletant_radius=5* (unit.nano * unit.meter), 
                                        brush_length=10* (unit.nano * unit.meter))


    def test_exception_depletant_radius(self, depletion_potentials):
        # Test exception on wrong unit.
        with pytest.raises(ValueError):
            depletion_potentials(phi = 0.5, depletant_radius=5/ ((unit.nano * unit.meter) ** 2), 
                                        brush_length=10* (unit.nano * unit.meter))

        # Test exception on negative depletant radius
        with pytest.raises(TypeError):
            depletion_potentials(phi = 0.5, depletant_radius=-5* (unit.nano * unit.meter), 
                                        brush_length=10* (unit.nano * unit.meter))


    '''def test_exception_colloid_potentials_tabulated_add_wrong_particles(
            self, colloid_potentials_tabulated, radius_one, radius_two, surface_potential_one, surface_potential_two):
        with pytest.raises(ValueError):
            colloid_potentials_tabulated.add_particle(radius=1000.0 * (unit.nano * unit.meter),
                                                      surface_potential=surface_potential_one)
        with pytest.raises(ValueError):
            colloid_potentials_tabulated.add_particle(radius=1000.0 * (unit.nano * unit.meter),
                                                      surface_potential=surface_potential_two)
        with pytest.raises(ValueError):
            colloid_potentials_tabulated.add_particle(radius=radius_one,
                                                      surface_potential=1000.0 * (unit.milli * unit.volt))
        with pytest.raises(ValueError):
            colloid_potentials_tabulated.add_particle(radius=radius_two,
                                                      surface_potential=1000.0 * (unit.milli * unit.volt))
        with pytest.raises(ValueError):
            colloid_potentials_tabulated.add_particle(radius=radius_one, surface_potential=surface_potential_two)
        with pytest.raises(ValueError):
            colloid_potentials_tabulated.add_particle(radius=radius_two, surface_potential=surface_potential_one)'''


# noinspection DuplicatedCode
class TestDepletionPotentialsForTwoParticles(TestParameters):
    @pytest.fixture(autouse=True)
    def add_two_particles(self, openmm_system, depletion_potentials, radius_one, radius_two):
        openmm_system.addParticle(mass=1.0)
        depletion_potentials.add_particle(radius=radius_one)
        openmm_system.addParticle(mass=1.0)
        depletion_potentials.add_particle(radius=radius_two)
        for potential in depletion_potentials.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)


    @pytest.mark.parametrize("surface_separation,expected_function",
                             [((lambda pos, *args: np.array((pos* (unit.nano * unit.meter), 0.0, 0.0)),
                             [0.0, 0.0, 0.0]),
                             TestParameters.depletion_potential_expected)
                      ])

    def test_depletion_potentials(self, openmm_context, test_positions, radius_one, brush_length, phi, radius_depletant, 
        expected_function):

        openmm_potentials = np.zeros(len(test_positions)) #use surface separation as test positions
        for index, pos in enumerate(test_positions):
            r_value = pos + 2*radius_one +2*brush_length #function takes input of center-to-center distance
            openmm_context.setPositions([[r_value.value_in_unit(unit.nanometer), 0.0, 0.0], [0.0, 0.0, 0.0]])
            state = openmm_context.getState(getEnergy=True)
            openmm_potentials[index] = (
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        
        expected_potentials = expected_function(test_positions, radius_one, brush_length, phi, radius_depletant)

        assert openmm_potentials == pytest.approx(expected_potentials.value_in_unit(unit.kilojoule_per_mole), 
            rel=1.0e-7, abs=1.0e-13)
       
                 
if __name__ == '__main__':
    pytest.main([__file__])
