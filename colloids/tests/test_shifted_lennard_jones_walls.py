from openmm import Context, LangevinIntegrator, Platform, System, unit, Vec3
import pytest
from colloids import ShiftedLennardJonesWalls
import numpy as np

def expected_slj_walls_potential(pos, box_length, rcut, delta, epsilon, sigma, alpha):

    return np.where(np.abs(pos) < box_length / 2 - rcut - delta,
                    0.0,
                    4 * epsilon * (np.power(sigma / (box_length / 2 - np.abs(pos) - delta), 12)
                                   - alpha * np.power(sigma / (box_length / 2 - np.abs(pos) - delta), 6))
                    - 4 * epsilon * (np.power(sigma / rcut, 12)
                                     - alpha * np.power(sigma / rcut, 6)))

def generate_parametrize_values(wall_direction, num_test_values, box_length, rcut, delta, epsilon, sigma, alpha):
    
    test_positions = np.linspace(-box_length / 2 + 200, box_length / 2 - 200, num=num_test_values)

    pots_tuple=[]

    if wall_direction == True:

        pots= expected_slj_walls_potential(test_positions, box_length, rcut,
                         delta, epsilon, sigma, alpha)
    else:
        pots= np.zeros(num_test_values)

    for i in range(num_test_values):
        pots_tuple.append((test_positions[i] * (unit.nano * unit.meter), pots[i] * unit.kilojoule_per_mole))
    
    pots_tuple = tuple(pots_tuple)

    return pots_tuple
    

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
    def sigma(self):
        return 105.0
    
    @pytest.fixture
    def delta(self):
        return 105.0 - 1

    @pytest.fixture 
    def rcut(self):
        return 105.0*(2.0**(1.0/6.0))

    @pytest.fixture
    def num_test_values(self):
        return 1000

    @pytest.fixture
    def wall_directions(self):
        return [True, True, True]


    @pytest.fixture
    def openmm_system(self, box_length):
        system = System()
        system.setDefaultPeriodicBoxVectors(Vec3(box_length, 0.0, 0.0),
                                            Vec3(0.0, box_length, 0.0),
                                            Vec3(0.0, 0.0, box_length))
        return system


    @pytest.fixture
    def openmm_platform(self):
        return Platform.getPlatformByName("Reference")

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)

    @pytest.fixture
    def slj_potential(self, box_length, epsilon, alpha, wall_directions):
        return ShiftedLennardJonesWalls(box_length, epsilon, alpha,
                                             wall_directions)

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


class TestSLJWallPotentials(TestSLJParameters):
    
    @pytest.fixture(autouse=True)
    def add_particle(self, openmm_system, slj_potential, radius):
        openmm_system.addParticle(mass=1.0)
        slj_potential.add_particle(radius=radius)
        
        for potential in slj_potential.yield_potentials():
            openmm_system.addForce(potential)

    @pytest.fixture
    def test_positions(self, box_length, num_test_values):
        return np.linspace(-box_length / 2 + 200, box_length / 2 - 200, num=num_test_values)

    @pytest.fixture
    def expected_walls_potential(self, wall_direction, num_test_values, box_length, rcut, delta, epsilon, sigma, alpha):
        xyz_params = []
        for i in self.wall_directions:
            xyz_params.append(generate_parametrize_values(i, num_test_values, box_length, rcut, delta, epsilon, sigma, alpha))
        return xyz_params
    
    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @pytest.mark.parametrize("slj_potential","expected_x",
                             [   
                                 expected_walls_potential[0]
                             ])

    @pytest.mark.parametrize("slj_potential","expected_y",
                             [   
                                 expected_walls_potential[1]
                             ])

    @pytest.mark.parametrize("slj_potential","expected_z",
                             [   
                                 expected_walls_potential[2]
                             ])

    def test_slj_walls_potential(self, openmm_context, test_positions, expected_x, expected_y, expected_z):
        # test x direction
        openmm_context.setPositions([test_positions, 0.0, 0.0])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected_x.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))

        # test y direction
        openmm_context.setPositions([0.0, test_positions, 0.0])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected_y.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))

        # test z direction
        openmm_context.setPositions([0.0, 0.0, test_positions])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected_y.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))


if __name__ == '__main__':
    pytest.main([__file__])
