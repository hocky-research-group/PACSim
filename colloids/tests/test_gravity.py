from openmm import Context, LangevinIntegrator, Platform, System, unit, Vec3
import pytest
from colloids import gravity
import numpy as np


class TestGravityParameters(object):
    @staticmethod
    def expected_gravitational_potential(z, g, particle_mass, energy_conversion_factor):
        return (g * particle_mass * z / energy_conversion_factor) 

    @pytest.fixture
    def radius(self):
        return 105.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def energy_conversion_factor(self):
        return 10 ** 36

 ## should i be adding the units to g 
    @pytest.fixture
    def g(self):
        return 9.8 * unit.meters / unit.second**2

  ## should i add amu?
    @pytest.fixture
    def particle_mass(self):
        return 1.0 

    @pytest.fixture
    def box_length(self):
        return 1000.0 * unit.nanometer 

 """for the functions wall_distances, num_test_values, and test_positions below, is the test_positions just taking
 the dimensions of the box and creating an evenly spaced sequence with 1000-step intervals? If so, should I keep this
 or does it not pertain to the gravity function?"""
 
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
    def expected_gravitational_potential(self, z, g, particle_mass):
        return gravity(self, z, g, particle_mass)

class TestGravityExceptions(TestGravityParameters):
    def test_exception_g(self, g, expected_gravitational_potential):
        # Test exception on wrong unit of g.
        with pytest.raises(TypeError):
            expected_gravitational_potential.add_particle(index=0, g=g / ((unit.nanometer) ** 2))
        # Test exception on negative g.
    ## confused with what the noinspetion PyTypeChecker means
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            expected_gravitational_potential.add_particle(index=0, g = -g)

##Do i need to keep this test?
    def test_exception_radius_too_large(self, expected_gravitational_potential):
        # This is fine for the smallest wall distance of 1000 nm.
        expected_gravitational_potential.add_particle(index=0, radius=236.0 * (unit.nanometer))
        # This is not fine for the smallest wall distance of 1000 nm.
        with pytest.raises(ValueError):
            expected_gravitational_potential.add_particle(index=1, radius=237.0 * (unit.nano * unit.meter))
        # This is fine for the smallest relevant wall distance of 1500 nm.
       expected_gravitational_potential.add_particle(index=0, radius=353.0 * (unit.nano * unit.meter))
        # This is not fine for the smallest relevant wall distance of 1500 nm.
        with pytest.raises(ValueError):
           expected_gravitational_potential.add_particle(index=1, radius=354.0 * (unit.nano * unit.meter))

    def test_exception_no_particles_added(self, expected_gravitational_potential):
        with pytest.raises(RuntimeError):
            for _ in expected_gravitational_potential.yield_potentials():
                pass
        with pytest.raises(RuntimeError):
            for _ in expected_gravitational_potential.yield_potentials():
                pass
## I don't understand what this test is doing
    def test_exception_add_particle_after_yield_potentials(self, radius, expected_gravitational_potential):
        expected_gravitational_potential.add_particle(index=0, radius=radius)
        for _ in expected_gravitational_potential.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            expected_gravitational_potential.add_particle(index=1, radius=radius)
       expected_gravitational_potential.add_particle(index=0, radius=radius)
        for _ inexpected_gravitational_potential.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
           expected_gravitational_potential.add_particle(index=1, radius=radius)

    def test_exception_mass_negative(self, particle_mass, expected_gravitational_potential):
        # Test exception mass is negative
        with pytest.raises(ValueError):
            expected_gravitational_potential.add_particle(index=0, particle_mass = -particle_mass)

    def test_exception_z(self):
        # Test exception on wrong units of z
        with pytest.raises(TypeError):
            expected_gravitational_potential.add_particle(index=0, z=z / ((unit.nanometer) ** 2))


class  TestGravityParameters(TestGravityParameters):
    
    @pytest.fixture(autouse=True)
    def add_particle(self, openmm_system, expected_gravitational_potential, radius):
        openmm_system.addParticle(mass=1.0)
        expected_gravitational_potential.add_particle(index=0, radius=radius)
        for potential in expected_gravitational_potential.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_particle fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @pytest.mark.parametrize("direction,expected_function",
                             [
                                 (0,  TestGravityParameters.slj_walls_potential_active),
                                 (1,  TestGravityParameters.slj_walls_potential_active),
                                 (2,  TestGravityParameters.slj_walls_potential_active)
                             ])
    def test_slj_walls_potential(self, openmm_context, test_positions, wall_distances, epsilon, alpha_all, radius,
                                 all_wall_directions, direction, expected_function):
        openmm_potentials = np.empty(len(test_positions[direction]))
        for index, dir_position in enumerate(test_positions[direction]):
            position = [0.0, 0.0, 0.0]
            position[direction] = dir_position
            openmm_context.setPositions([position])
            openmm_state = openmm_context.getState(getEnergy=True)
            openmm_potentials[index] = openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        assert all_wall_directions[direction]
        wall_distance = wall_distances[direction].value_in_unit(unit.nano * unit.meter)
        rcut = radius.value_in_unit(unit.nano * unit.meter) * 2**(1.0/6.0)
        delta = radius.value_in_unit(unit.nano * unit.meter) - 1.0
        expected_potentials = expected_function(test_positions[direction], wall_distance, rcut, delta,
                                                epsilon.value_in_unit(unit.kilojoule_per_mole),
                                                radius.value_in_unit(unit.nano * unit.meter), alpha_all)
        assert np.any(expected_potentials > 0.0)
        assert np.all(expected_potentials >= 0.0)
        assert openmm_potentials == pytest.approx(expected_potentials, rel=1.0e-7, abs=1.0e-13)


class TestShiftedLennardJonesWallPotentialsSome( TestGravityParameters):
    @pytest.fixture(autouse=True)
    def add_particle(self, openmm_system,expected_gravitational_potential, radius):
        openmm_system.addParticle(mass=1.0)
       expected_gravitational_potential.add_particle(index=0, radius=radius)
        for potential inexpected_gravitational_potential.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_particle fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @pytest.mark.parametrize("direction,expected_function",
                             [
                                 (0, lambda pos, *args: np.zeros_like(pos)),
                                 (1,  TestGravityParameters.slj_walls_potential_active),
                                 (2, lambda pos, *args: np.zeros_like(pos))
                             ])
    def test_slj_walls_potential(self, openmm_context, test_positions, wall_distances, epsilon, alpha_some, radius,
                                 some_wall_directions, direction, expected_function):
        openmm_potentials = np.empty(len(test_positions[direction]))
        for index, dir_position in enumerate(test_positions[direction]):
            position = [0.0, 0.0, 0.0]
            position[direction] = dir_position
            openmm_context.setPositions([position])
            openmm_state = openmm_context.getState(getEnergy=True)
            openmm_potentials[index] = openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        wall_distance = wall_distances[direction].value_in_unit(unit.nano * unit.meter)
        rcut = radius.value_in_unit(unit.nano * unit.meter) * 2 ** (1.0 / 6.0)
        delta = radius.value_in_unit(unit.nano * unit.meter) - 1.0
        expected_potentials = expected_function(test_positions[direction], wall_distance, rcut, delta,
                                                epsilon.value_in_unit(unit.kilojoule_per_mole),
                                                radius.value_in_unit(unit.nano * unit.meter), alpha_some)
        if some_wall_directions[direction]:
            assert np.any(expected_potentials > 0.0)
            assert np.all(expected_potentials >= 0.0)
        else:
            assert np.all(expected_potentials == 0.0)
        assert openmm_potentials == pytest.approx(expected_potentials, rel=1.0e-7, abs=1.0e-13)


if __name__ == '__main__':
    pytest.main([__file__])
