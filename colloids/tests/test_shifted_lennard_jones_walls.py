from openmm import Context, LangevinIntegrator, NonbondedForce, OpenMMException, Platform, System, unit, Vec3
import pytest
from colloids import ShiftedLennardJonesWalls
import numpy as np


@pytest.mark.filterwarnings(
    "ignore:The force of the shifted Lennard-Jones potential as a wall is only continuous if alpha = 1.")
class TestShiftedLennardJonesWallsParameters(object):
    @staticmethod
    def slj_walls_potential_active(pos, wall_distance, rcut, delta, epsilon, sigma, alpha):
        return np.where(np.abs(pos) < wall_distance / 2 - rcut - delta,
                        0.0,
                        4 * epsilon * (np.power(sigma / (wall_distance / 2 - np.abs(pos) - delta), 12)
                                       - alpha * np.power(sigma / (wall_distance / 2 - np.abs(pos) - delta), 6))
                        - 4 * epsilon * (np.power(sigma / rcut, 12)
                                         - alpha * np.power(sigma / rcut, 6)))

    @pytest.fixture
    def radius(self):
        return 105.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def wall_distances(self):
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
    def test_positions(self, wall_distances, num_test_values):
        # noinspection PyUnresolvedReferences
        return [np.linspace(-wall_distance.value_in_unit(unit.nanometer) / 2.0 + 200.0,
                            wall_distance.value_in_unit(unit.nanometer) / 2.0 - 200.0, num=num_test_values)
                for wall_distance in wall_distances]

    @pytest.fixture
    def all_wall_directions(self):
        return [True, True, True]

    @pytest.fixture
    def some_wall_directions(self):
        return [False, True, False]

    @pytest.fixture
    def openmm_system(self, wall_distances):
        system = System()
        system.setDefaultPeriodicBoxVectors(Vec3(wall_distances[0], 0.0, 0.0),
                                            Vec3(0.0, wall_distances[1], 0.0),
                                            Vec3(0.0, 0.0, wall_distances[2]))
        return system

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)

    @pytest.fixture
    def slj_potential_all(self, wall_distances, epsilon, alpha_all, all_wall_directions):
        return ShiftedLennardJonesWalls(wall_distances, epsilon, alpha_all, all_wall_directions, False)

    @pytest.fixture
    def slj_potential_some(self, wall_distances, epsilon, alpha_some, some_wall_directions):
        new_wall_distances = [None if not i else wall_distances[j] for j, i in enumerate(some_wall_directions)]
        return ShiftedLennardJonesWalls(new_wall_distances, epsilon, alpha_some, some_wall_directions, False)

    @pytest.fixture
    def slj_potential_all_substrate(self, wall_distances, epsilon, alpha_all, all_wall_directions):
        return ShiftedLennardJonesWalls(wall_distances, epsilon, alpha_all, all_wall_directions, True)


class TestShiftedLennardJonesWallsExceptions(TestShiftedLennardJonesWallsParameters):
    def test_exception_radius(self, radius, slj_potential_all, slj_potential_some, slj_potential_all_substrate):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            slj_potential_all.add_particle(index=0, radius=radius / ((unit.nano * unit.meter) ** 2))
        with pytest.raises(TypeError):
            slj_potential_some.add_particle(index=0, radius=radius / ((unit.nano * unit.meter) ** 2))
        with pytest.raises(TypeError):
            slj_potential_all_substrate.add_particle(index=0, radius=radius / ((unit.nano * unit.meter) ** 2))
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            slj_potential_all.add_particle(index=0, radius=-radius)
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            slj_potential_some.add_particle(index=0, radius=-radius)
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            slj_potential_all_substrate.add_particle(index=0, radius=-radius)

    def test_exception_radius_too_large(self, slj_potential_all, slj_potential_some, slj_potential_all_substrate):
        # This is fine for the smallest wall distance of 1000 nm.
        slj_potential_all.add_particle(index=0, radius=236.0 * (unit.nano * unit.meter))
        # This is not fine for the smallest wall distance of 1000 nm.
        with pytest.raises(ValueError):
            slj_potential_all.add_particle(index=1, radius=237.0 * (unit.nano * unit.meter))
        # This is fine for the smallest relevant wall distance of 1500 nm.
        slj_potential_some.add_particle(index=0, radius=353.0 * (unit.nano * unit.meter))
        # This is not fine for the smallest relevant wall distance of 1500 nm.
        with pytest.raises(ValueError):
            slj_potential_some.add_particle(index=1, radius=354.0 * (unit.nano * unit.meter))
        # This is fine for the smallest wall distance of 1000 nm.
        slj_potential_all_substrate.add_particle(index=0, radius=236.0 * (unit.nano * unit.meter))
        # This is not fine for the smallest wall distance of 1000 nm.
        with pytest.raises(ValueError):
            slj_potential_all_substrate.add_particle(index=1, radius=237.0 * (unit.nano * unit.meter))

    def test_exception_no_particles_added(self, slj_potential_all, slj_potential_some, slj_potential_all_substrate):
        with pytest.raises(RuntimeError):
            for _ in slj_potential_all.yield_potentials():
                pass
        with pytest.raises(RuntimeError):
            for _ in slj_potential_some.yield_potentials():
                pass
        with pytest.raises(RuntimeError):
            for _ in slj_potential_all_substrate.yield_potentials():
                pass

    def test_exception_add_particle_after_yield_potentials(self, radius, slj_potential_all, slj_potential_some,
                                                           slj_potential_all_substrate):
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
        slj_potential_all_substrate.add_particle(index=0, radius=radius)
        for _ in slj_potential_all_substrate.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            slj_potential_all_substrate.add_particle(index=1, radius=radius)

    def test_exception_alpha(self):
        # Test exception on negative alpha
        with pytest.raises(ValueError):
            ShiftedLennardJonesWalls(wall_distances=[1000.0, 1500.0, 2000.0] * (unit.nano * unit.meter),
                                     epsilon=1.0 * unit.kilojoule_per_mole, alpha=-1.0,
                                     wall_directions=[True, True, True])

        # Test exception on alpha > 1
        with pytest.raises(ValueError):
            ShiftedLennardJonesWalls(wall_distances=[1000.0, 1500.0, 2000.0] * (unit.nano * unit.meter),
                                     epsilon=1.0 * unit.kilojoule_per_mole, alpha=2.0,
                                     wall_directions=[True, True, True])

    def test_exception_epsilon(self):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            ShiftedLennardJonesWalls(wall_distances=[1000.0, 1500.0, 2000.0] * (unit.nano * unit.meter),
                                     epsilon=1.0 * unit.joule, alpha=1.0,
                                     wall_directions=[True, True, True])

        # Test exception on negative epsilon
        with pytest.raises(ValueError):
            ShiftedLennardJonesWalls(wall_distances=[1000.0, 1500.0, 2000.0] * (unit.nano * unit.meter),
                                     epsilon=-1.0 * unit.kilojoule_per_mole, alpha=1.0,
                                     wall_directions=[True, True, True])

    def test_exception_wall_directions(self):
        # Test exception no active wall directions
        with pytest.raises(ValueError):
            ShiftedLennardJonesWalls(wall_distances=[1000.0, 1500.0, 2000.0] * (unit.nano * unit.meter),
                                     epsilon=1.0 * unit.kilojoule_per_mole, alpha=1.0,
                                     wall_directions=[False, False, False])

        # Test exception length of wall direction sequence !=3
        with pytest.raises(ValueError):
            ShiftedLennardJonesWalls(wall_distances=[1000.0, 1500.0, 2000.0] * (unit.nano * unit.meter),
                                     epsilon=1.0 * unit.kilojoule_per_mole, alpha=1.0,
                                     wall_directions=[True, False, True, True])

    def test_exception_wall_distances(self):
        # Test exception on wrong unit 
        with pytest.raises(TypeError):
            ShiftedLennardJonesWalls(wall_distances=[1000.0, 1500.0, 2000.0] * ((unit.nano * unit.meter) ** 2),
                                     epsilon=1.0 * unit.kilojoule_per_mole, alpha=1.0,
                                     wall_directions=[True, True, True])

        # Test exception length of wall distance sequence !=3
        with pytest.raises(ValueError):
            ShiftedLennardJonesWalls(wall_distances=[1000.0, 1500.0, 2000.0, 2500.0] * (unit.nano * unit.meter),
                                     epsilon=1.0 * unit.kilojoule_per_mole, alpha=1.0,
                                     wall_directions=[True, True, True])

        # Test exception wall distance negative.
        with pytest.raises(ValueError):
            ShiftedLennardJonesWalls(wall_distances=[-1000.0, 1500.0, 2000.0] * (unit.nano * unit.meter),
                                     epsilon=1.0 * unit.kilojoule_per_mole, alpha=1.0,
                                     wall_directions=[True, True, True])

        # Test exception wall distance not specified for active wall direction
        with pytest.raises(ValueError):
            ShiftedLennardJonesWalls(wall_distances=[1000.0 * (unit.nano * unit.meter),
                                                     None,
                                                     2000.0 * (unit.nano * unit.meter)],
                                     epsilon=1.0 * unit.kilojoule_per_mole, alpha=1.0,
                                     wall_directions=[True, True, True])

        # Test exception wall distance specified for inactive wall direction
        with pytest.raises(ValueError):
            ShiftedLennardJonesWalls(wall_distances=[1000.0, 1500.0, 2000.0] * (unit.nano * unit.meter),
                                     epsilon=1.0 * unit.kilojoule_per_mole, alpha=1.0,
                                     wall_directions=[True, False, True])

    def test_exception_substrate(self):
        # Test exception on missing wall distance and wall direction for substrate.
        with pytest.raises(ValueError):
            ShiftedLennardJonesWalls(
                wall_distances=[1000.0 * (unit.nano * unit.meter), None, 2000.0 * (unit.nano * unit.meter)],
                epsilon=1.0 * unit.kilojoule_per_mole, alpha=1.0, wall_directions=[True, False, True],
                use_substrate=True)


class TestShiftedLennardJonesWallPotentialsAll(TestShiftedLennardJonesWallsParameters):
    @pytest.fixture(autouse=True)
    def add_particle(self, openmm_system, slj_potential_all, radius):
        openmm_system.addParticle(mass=1.0)
        slj_potential_all.add_particle(index=0, radius=radius)
        for potential in slj_potential_all.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_particle fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture(params=[("Reference", 1.0e-7), ("CPU", 1.0e-7), ("CUDA", 1.0e-3), ("OpenCL", 1.0e-3)])
    def openmm_context_rel(self, openmm_system, openmm_dummy_integrator, request):
        try:
            platform = Platform.getPlatformByName(request.param[0])
        except OpenMMException:
            pytest.skip("Platform {} not available.".format(request.param[0]))
        return Context(openmm_system, openmm_dummy_integrator, platform), request.param[1]

    @pytest.mark.parametrize("direction,expected_function",
                             [
                                 (0, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active),
                                 (1, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active),
                                 (2, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active)
                             ])
    def test_slj_walls_potential(self, openmm_context_rel, test_positions, wall_distances, epsilon, alpha_all, radius,
                                 all_wall_directions, direction, expected_function):
        openmm_context, rel = openmm_context_rel
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
        assert openmm_potentials == pytest.approx(expected_potentials, rel=rel, abs=1.0e-13)


class TestShiftedLennardJonesWallPotentialsSome(TestShiftedLennardJonesWallsParameters):
    @pytest.fixture(autouse=True)
    def add_particle(self, openmm_system, slj_potential_some, radius):
        openmm_system.addParticle(mass=1.0)
        slj_potential_some.add_particle(index=0, radius=radius)
        for potential in slj_potential_some.yield_potentials():
            openmm_system.addForce(potential)
        # Add a nonbonded force with a periodic cutoff to the system so that it uses periodic boundary conditions.
        # This tests a potential issue on the CUDA/OpenCL platforms: https://github.com/openmm/openmm/issues/4611
        nonbonded_force = NonbondedForce()
        nonbonded_force.addParticle(0.0, 1.0, 0.0)
        nonbonded_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
        nonbonded_force.setCutoffDistance(5.0)
        openmm_system.addForce(nonbonded_force)

    # This function cannot be moved to TestParameters class because add_particle fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture(params=[("Reference", 1.0e-7), ("CPU", 1.0e-7), ("CUDA", 1.0e-3), ("OpenCL", 1.0e-3)])
    def openmm_context_rel(self, openmm_system, openmm_dummy_integrator, request):
        try:
            platform = Platform.getPlatformByName(request.param[0])
        except OpenMMException:
            pytest.skip("Platform {} not available.".format(request.param[0]))
        return Context(openmm_system, openmm_dummy_integrator, platform), request.param[1]

    @pytest.mark.parametrize("direction,expected_function",
                             [
                                 (0, lambda pos, *args: np.zeros_like(pos)),
                                 (1, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active),
                                 (2, lambda pos, *args: np.zeros_like(pos))
                             ])
    def test_slj_walls_potential(self, openmm_context_rel, test_positions, wall_distances, epsilon, alpha_some, radius,
                                 some_wall_directions, direction, expected_function):
        openmm_context, rel = openmm_context_rel
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
        assert openmm_potentials == pytest.approx(expected_potentials, rel=rel, abs=1.0e-13)


class TestShiftedLennardJonesWallPotentialsSubstrate(TestShiftedLennardJonesWallsParameters):
    @pytest.fixture(autouse=True)
    def add_particle(self, openmm_system, slj_potential_all_substrate, radius):
        openmm_system.addParticle(mass=1.0)
        slj_potential_all_substrate.add_particle(index=0, radius=radius)
        for potential in slj_potential_all_substrate.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_particle fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture(params=[("Reference", 1.0e-7), ("CPU", 1.0e-7), ("CUDA", 1.0e-3), ("OpenCL", 1.0e-3)])
    def openmm_context_rel(self, openmm_system, openmm_dummy_integrator, request):
        try:
            platform = Platform.getPlatformByName(request.param[0])
        except OpenMMException:
            pytest.skip("Platform {} not available.".format(request.param[0]))
        return Context(openmm_system, openmm_dummy_integrator, platform), request.param[1]

    @pytest.mark.parametrize("direction,expected_function",
                             [
                                 (0, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active),
                                 (1, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active),
                                 (2, TestShiftedLennardJonesWallsParameters.slj_walls_potential_active)
                             ])
    def test_slj_walls_potential(self, openmm_context_rel, test_positions, wall_distances, epsilon, alpha_all, radius,
                                 all_wall_directions, direction, expected_function):
        openmm_context, rel = openmm_context_rel
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
        # With a substrate, only the upper wall is active in the z direction.
        expected_potentials = np.where(
            direction != 2 or test_positions[direction] >= 0.0,
            expected_function(test_positions[direction], wall_distance, rcut, delta,
                              epsilon.value_in_unit(unit.kilojoule_per_mole),
                              radius.value_in_unit(unit.nano * unit.meter), alpha_all),
            0.0)
        assert np.any(expected_potentials > 0.0)
        assert np.all(expected_potentials >= 0.0)
        assert openmm_potentials == pytest.approx(expected_potentials, rel=rel, abs=1.0e-13)


if __name__ == '__main__':
    pytest.main([__file__])
