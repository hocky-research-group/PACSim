from openmm import Context, LangevinIntegrator, Platform, System, unit, Vec3
import pytest
from colloids import ShiftedLennardJonesWalls
import numpy as np

def expected_slj_potential(x, box_length, radius, epsilon, alpha):

    return np.where(np.abs(x) < box_length / 2 - rcut - delta,
                    0.0,
                    4 * epsilon * (np.power(sigma / (box_length / 2 - np.abs(x) - delta), 12)
                                   - alpha * np.power(sigma / (box_length / 2 - np.abs(x) - delta), 6))
                    - 4 * epsilon * (np.power(sigma / rcut, 12)
                                     - alpha * np.power(sigma / rcut, 6)))

radius = 105.0
sigma = radius
rcut = radius*(2.0**(1.0/6.0))
delta = radius - 1
box_length = 1000

test_positions = np.linspace(-box_length / 2 + 200, box_length / 2 - 200, num=10000)
expected_potentials = expected_slj_potential(test_positions, box_length, rcut,
                         delta, 1, radius, 0)

class TestSLJParameters(object):
    @pytest.fixture
    def radius(self):
        return 105.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def box_length(self):
        return 1000.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def epsilon(self):
        return 1.0 * (unit.kilojoule / unit.mole)

    @pytest.fixture
    def alpha(self):
        return 1.0

    @pytest.fixture
    def wall_directions(self):
        return [True, True, True]

    '''@pytest.fixture
    def slj_wall_parameters(self, radius, box_length, epsilon, alpha, wall_directions):
        
        return radius, box_length, epsilon, alpha, wall_directions'''
    

    @pytest.fixture
    def openmm_system(self, box_length):
        system = System()
        system.setDefaultPeriodicBoxVectors(Vec3(box_length, 0.0, 0.0),
                                            Vec3(0.0, box_length, 0.0),
                                            Vec3(0.0, 0.0, box_length))
        return system


    @pytest.fixture
    def slj_potential(self): #radius, box_length, epsilon, alpha, wall_directions):
        return ShiftedLennardJonesWalls.yield_potentials()

    @pytest.fixture
    def openmm_platform(self):
        return Platform.getPlatformByName("Reference")

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)


class TestSLJPotentialsExceptions(TestSLJParameters):
    def test_exception_radius(self, slj_potential, radius):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            slj_potential.add_particle(
                radius=radius / ((unit.nano * unit.meter))
            )
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            slj_potential.add_particle(
                radius=-radius)
                                           
    def test_exception_no_particles_added(self, slj_potential):
        with pytest.raises(RuntimeError):
            for _ in slj_potential.yield_potentials():
                pass

    def test_exception_add_particle_after_yield_potentials(self, slj_potential, radius):
        slj_potential.add_particle(radius=radius)
        for _ in slj_potential.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            slj_potential.add_particle(radius)

   
    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @pytest.mark.parametrize("slj_potential","expected",
                             [   # 
                                 (test_positions * (unit.nano * unit.meter), expected_potentials * unit.kilojoule_per_mole),
                             ])

    def test_slj_potential(self, openmm_context, positions, expected):
        openmm_context.setPositions([positions])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))


if __name__ == '__main__':
    pytest.main([__file__])
