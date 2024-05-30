from openmm import Context, LangevinIntegrator, Platform, System, unit, Vec3
import pytest
from colloids import ShiftedLennardJonesWalls
import numpy as np


class TestShiftedLennardJonesWallsParameters(object):
    @staticmethod
    def slj_walls_potential_active(pos, box_length, rcut, delta, epsilon, sigma, alpha):
        return np.where(np.abs(pos) < box_length / 2 - rcut - delta,
                        0.0,
                        4 * epsilon * (np.power(sigma / (box_length / 2 - np.abs(pos) - delta), 12)
                                       - alpha * np.power(sigma / (box_length / 2 - np.abs(pos) - delta), 6))
                        - 4 * epsilon * (np.power(sigma / rcut, 12)
                                         - alpha * np.power(sigma / rcut, 6)))

    @pytest.fixture
    def radius(self):
        return 105.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def box_lengths(self):
        return [1000.0 * (unit.nano * unit.meter), 1500.0 * (unit.nano * unit.meter), 1200.0 * (unit.nano * unit.meter)]

    @pytest.fixture
    def epsilon(self):
        return 1.0 * unit.kilojoule_per_mole

    @pytest.fixture
    def alpha_all(self):
        return 1.0

    @pytest.fixture
    def alpha_some(self):
        return 0.0

    @pytest.fixture
    def sigma(self):
        return 105.0

    @pytest.fixture
    def num_test_values(self):
        return 1000

    @pytest.fixture
    def test_positions(self, box_lengths, num_test_values):
        # noinspection PyUnresolvedReferences
        return [np.linspace(-box_length.value_in_unit(unit.nanometer) / 2.0 + 200.0,
                            box_length.value_in_unit(unit.nanometer) / 2.0 - 200.0, num=num_test_values)
                for box_length in box_lengths]

    @pytest.fixture
    def all_wall_directions(self):
        return [True, True, True]

    @pytest.fixture
    def some_wall_directions(self):
        return [False, True, False]

    @pytest.fixture
    def openmm_system(self, box_lengths):
        system = System()
        system.setDefaultPeriodicBoxVectors(Vec3(box_lengths[0], 0.0, 0.0),
                                            Vec3(0.0, box_lengths[1], 0.0),
                                            Vec3(0.0, 0.0, box_lengths[2]))
        return system

    @pytest.fixture
    def openmm_platform(self):
        return Platform.getPlatformByName("Reference")

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)

    @pytest.fixture
    def slj_potential_all(self, box_lengths, epsilon, alpha_all, all_wall_directions):
        return ShiftedLennardJonesWalls(box_lengths, epsilon, alpha_all, all_wall_directions)

    @pytest.fixture
    def slj_potential_some(self, box_lengths, epsilon, alpha_some, some_wall_directions):
        new_box_lengths = [None if not i else box_lengths[j] for j, i in enumerate(some_wall_directions)]
        return ShiftedLennardJonesWalls(new_box_lengths, epsilon, alpha_some, some_wall_directions)


class TestShiftedLennardJonesWallsExceptions(TestShiftedLennardJonesWallsParameters):
    def test_exception_radius(self, radius, slj_potential_all, slj_potential_some):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            slj_potential_all.add_particle(index=0, radius=radius / ((unit.nano * unit.meter) ** 2))
        with pytest.raises(TypeError):
            slj_potential_some.add_particle(index=0, radius=radius / ((unit.nano * unit.meter) ** 2))
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            slj_potential_all.add_particle(index=0, radius=-radius)
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            slj_potential_some.add_particle(index=0, radius=-radius)
                                           
    def test_exception_no_particles_added(self, slj_potential_all, slj_potential_some):
        with pytest.raises(RuntimeError):
            for _ in slj_potential_all.yield_potentials():
                pass
        with pytest.raises(RuntimeError):
            for _ in slj_potential_some.yield_potentials():
                pass

    def test_exception_add_particle_after_yield_potentials(self, radius, slj_potential_all, slj_potential_some):
        slj_potential_all.add_particle(index=0, radius=radius)
        for _ in slj_potential_all.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            slj_potential_all.add_particle(index=1, radius=radius)
        slj_potential_some.add_particle(index=0, radius=radius)
        for _ in slj_potential_some.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            slj_potential_some.add_particle(index=1, radius=radius)


class TestShiftedLennardJonesWallPotentialsAll(TestShiftedLennardJonesWallsParameters):
    
    @pytest.fixture(autouse=True)
    def add_particle(self, openmm_system, slj_potential_all, radius):
        openmm_system.addParticle(mass=1.0)
        slj_potential_all.add_particle(index=0, radius=radius)
        for potential in slj_potential_all.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_particle fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @pytest.mark.parametrize("direction,expected_function",
                             [
                                 (0, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active),
                                 (1, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active),
                                 (2, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active)
                             ])
    def test_slj_walls_potential(self, openmm_context, test_positions, box_lengths, epsilon, alpha_all, radius,
                                 all_wall_directions, direction, expected_function):
        openmm_potentials = np.empty(len(test_positions[direction]))
        for index, dir_position in enumerate(test_positions[direction]):
            position = [0.0, 0.0, 0.0]
            position[direction] = dir_position
            openmm_context.setPositions([position])
            openmm_state = openmm_context.getState(getEnergy=True)
            openmm_potentials[index] = openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        assert all_wall_directions[direction]
        box_length = box_lengths[direction].value_in_unit(unit.nano * unit.meter)
        rcut = radius.value_in_unit(unit.nano * unit.meter) * 2**(1.0/6.0)
        delta = radius.value_in_unit(unit.nano * unit.meter) - 1.0
        expected_potentials = expected_function(test_positions[direction], box_length, rcut, delta,
                                                epsilon.value_in_unit(unit.kilojoule_per_mole),
                                                radius.value_in_unit(unit.nano * unit.meter), alpha_all)
        assert np.any(expected_potentials > 0.0)
        assert np.all(expected_potentials >= 0.0)
        assert openmm_potentials == pytest.approx(expected_potentials, rel=1.0e-7, abs=1.0e-13)


class TestShiftedLennardJonesWallPotentialsSome(TestShiftedLennardJonesWallsParameters):
    @pytest.fixture(autouse=True)
    def add_particle(self, openmm_system, slj_potential_some, radius):
        openmm_system.addParticle(mass=1.0)
        slj_potential_some.add_particle(index=0, radius=radius)
        for potential in slj_potential_some.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_particle fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @pytest.mark.parametrize("direction,expected_function",
                             [
                                 (0, lambda pos, *args: np.zeros_like(pos)),
                                 (1, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active),
                                 (2, lambda pos, *args: np.zeros_like(pos))
                             ])
    def test_slj_walls_potential(self, openmm_context, test_positions, box_lengths, epsilon, alpha_some, radius,
                                 some_wall_directions, direction, expected_function):
        openmm_potentials = np.empty(len(test_positions[direction]))
        for index, dir_position in enumerate(test_positions[direction]):
            position = [0.0, 0.0, 0.0]
            position[direction] = dir_position
            openmm_context.setPositions([position])
            openmm_state = openmm_context.getState(getEnergy=True)
            openmm_potentials[index] = openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        box_length = box_lengths[direction].value_in_unit(unit.nano * unit.meter)
        rcut = radius.value_in_unit(unit.nano * unit.meter) * 2 ** (1.0 / 6.0)
        delta = radius.value_in_unit(unit.nano * unit.meter) - 1.0
        expected_potentials = expected_function(test_positions[direction], box_length, rcut, delta,
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
