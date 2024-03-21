from openmm import Context, LangevinIntegrator, Platform, System, unit
import pytest
from colloids.colloid_potentials_algebraic import ColloidPotentialsAlgebraic
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters


class TestParameters(object):
    @pytest.fixture(scope="class")
    def radius_one(self):
        return 325.0 * unit.nanometer

    @pytest.fixture(scope="class")
    def radius_two(self):
        return 65.0 * unit.nanometer

    @pytest.fixture(scope="class")
    def surface_potential_one(self):
        return 50.0 * (unit.milli * unit.volt)

    @pytest.fixture(scope="class")
    def surface_potential_two(self):
        return -40.0 * (unit.milli * unit.volt)

    @pytest.fixture(scope="class")
    def brush_density(self):
        return 0.09 / (unit.nanometer ** 2)

    @pytest.fixture(scope="class")
    def brush_length(self):
        return 10.0 * unit.nanometer

    @pytest.fixture(scope="class")
    def debye_length(self):
        return 5.0 * unit.nanometer

    @pytest.fixture(scope="class")
    def dielectric_constant(self):
        return 80.0

    @pytest.fixture(scope="class")
    def temperature(self):
        return 298.0 * unit.kelvin

    @pytest.fixture(scope="class")
    def colloid_potentials_parameters(self, brush_density, brush_length, debye_length, temperature,
                                      dielectric_constant):
        return ColloidPotentialsParameters(brush_density=brush_density, brush_length=brush_length,
                                           debye_length=debye_length, temperature=temperature,
                                           dielectric_constant=dielectric_constant)

    @pytest.fixture(scope="class")
    def maximum_surface_separation(self, radius_one, radius_two, debye_length):
        return 100.0 * unit.nanometer

    @pytest.fixture(scope="class")
    def side_length(self, radius_one, radius_two, maximum_surface_separation):
        # Make system very large so that we do not care about periodic boundaries.
        return 20.0 * (maximum_surface_separation + 2.0 * max(radius_one, radius_two))

    @pytest.fixture(scope="class")
    def openmm_system(self, side_length):
        system = System()
        # Make system very large so that we do not care about periodic boundaries.
        side_length_value = side_length.value_in_unit(unit.nanometer)
        # noinspection PyTypeChecker
        system.setDefaultPeriodicBoxVectors([side_length_value, 0.0, 0.0],
                                            [0.0, side_length_value, 0.0],
                                            [0.0, 0.0, side_length_value])
        return system

    @pytest.fixture(scope="class")
    def colloid_potentials_algebraic(self, colloid_potentials_parameters):
        return ColloidPotentialsAlgebraic(colloid_potentials_parameters=colloid_potentials_parameters, use_log=False)

    @pytest.fixture(scope="class")
    def colloid_potentials_algebraic_log(self, colloid_potentials_parameters):
        return ColloidPotentialsAlgebraic(colloid_potentials_parameters=colloid_potentials_parameters, use_log=True)

    @pytest.fixture(scope="class")
    def openmm_platform(self):
        return Platform.getPlatformByName("Reference")

    @pytest.fixture(scope="class")
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)


class TestColloidPotentialsAlgebraicExceptions(object):
    def test_exceptions_no_particles_added(self):
        potentials = ColloidPotentialsAlgebraic()
        with pytest.raises(RuntimeError):
            _ = potentials.steric_potential
        with pytest.raises(RuntimeError):
            _ = potentials.electrostatic_potential


# noinspection DuplicatedCode
class TestColloidPotentialsAlgebraicForTwoParticles(TestParameters):
    @pytest.fixture(autouse=True, scope="class")
    def add_two_particles(self, openmm_system, colloid_potentials_algebraic,
                          radius_one, radius_two, surface_potential_one, surface_potential_two):
        openmm_system.addParticle(mass=1.0)
        colloid_potentials_algebraic.add_particle(radius=radius_one, surface_potential=surface_potential_one)
        openmm_system.addParticle(mass=1.0)
        colloid_potentials_algebraic.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        openmm_system.addForce(colloid_potentials_algebraic.steric_potential)
        openmm_system.addForce(colloid_potentials_algebraic.electrostatic_potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture(scope="class")
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    # noinspection DuplicatedCode
    def test_potential_parameters(self, openmm_context, radius_one, radius_two, brush_length, debye_length):
        assert len(openmm_context.getSystem().getForces()) == 2
        steric_force = openmm_context.getSystem().getForce(0)
        electrostatic_force = openmm_context.getSystem().getForce(1)

        assert steric_force.usesPeriodicBoundaryConditions()
        assert electrostatic_force.usesPeriodicBoundaryConditions()

        assert not steric_force.getUseLongRangeCorrection()
        assert not electrostatic_force.getUseLongRangeCorrection()

        assert not steric_force.getUseSwitchingFunction()
        assert electrostatic_force.getUseSwitchingFunction()
        assert electrostatic_force.getSwitchingDistance() == 2.0 * max(radius_one, radius_two) + 20.0 * debye_length

        assert steric_force.getNonbondedMethod() == steric_force.CutoffPeriodic
        assert electrostatic_force.getNonbondedMethod() == electrostatic_force.CutoffPeriodic

        assert steric_force.getCutoffDistance() == 2.0 * max(radius_one, radius_two) + 2.0 * brush_length
        assert electrostatic_force.getCutoffDistance() == 2.0 * max(radius_one, radius_two) + 21.0 * debye_length

    @pytest.mark.parametrize("surface_separation,expected",
                             [   # Test at h=0.
                                 (10.0 * unit.nanometer, 1505.829355134808 * unit.kilojoule_per_mole),
                                 # Test at h=2L.
                                 (20.0 * unit.nanometer, -10.63613061419315 * unit.kilojoule_per_mole),
                                 # Test slightly below h=2L where steric potential is not zero.
                                 ((20.0 - 0.1) * unit.nanometer, -10.84996692702675 * unit.kilojoule_per_mole),
                                 # Test slightly above h=2L where steric potential is strictly zero.
                                 ((20.0 + 0.1) * unit.nanometer, -10.42552111714948 * unit.kilojoule_per_mole),
                                 # Test at h=3L.
                                 (30.0 * unit.nanometer, -1.439443749213437 * unit.kilojoule_per_mole),
                                 # Test at h=20*debye_length, where electrostatic potential should not yet be cutoff.
                                 (100.0 * unit.nanometer, -1.196938817005087e-6 * unit.kilojoule_per_mole)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, surface_separation, expected):
        openmm_context.setPositions([[radius_one + radius_two + surface_separation, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-8, abs=1.0e-13))


# noinspection DuplicatedCode
class TestColloidPotentialsAlgebraicWithLogForTwoParticles(TestParameters):
    @pytest.fixture(autouse=True, scope="class")
    def add_two_particles(self, openmm_system, colloid_potentials_algebraic_log,
                          radius_one, radius_two, surface_potential_one, surface_potential_two):
        openmm_system.addParticle(mass=1.0)
        colloid_potentials_algebraic_log.add_particle(radius=radius_one, surface_potential=surface_potential_one)
        openmm_system.addParticle(mass=1.0)
        colloid_potentials_algebraic_log.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        openmm_system.addForce(colloid_potentials_algebraic_log.steric_potential)
        openmm_system.addForce(colloid_potentials_algebraic_log.electrostatic_potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture(scope="class")
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    # noinspection DuplicatedCode
    def test_potential_parameters(self, openmm_context, radius_one, radius_two, brush_length, debye_length):
        assert len(openmm_context.getSystem().getForces()) == 2
        steric_force = openmm_context.getSystem().getForce(0)
        electrostatic_force = openmm_context.getSystem().getForce(1)

        assert steric_force.usesPeriodicBoundaryConditions()
        assert electrostatic_force.usesPeriodicBoundaryConditions()

        assert not steric_force.getUseLongRangeCorrection()
        assert not electrostatic_force.getUseLongRangeCorrection()

        assert not steric_force.getUseSwitchingFunction()
        assert electrostatic_force.getUseSwitchingFunction()
        assert electrostatic_force.getSwitchingDistance() == 2.0 * max(radius_one, radius_two) + 20.0 * debye_length

        assert steric_force.getNonbondedMethod() == steric_force.CutoffPeriodic
        assert electrostatic_force.getNonbondedMethod() == electrostatic_force.CutoffPeriodic

        assert steric_force.getCutoffDistance() == 2.0 * max(radius_one, radius_two) + 2.0 * brush_length
        assert electrostatic_force.getCutoffDistance() == 2.0 * max(radius_one, radius_two) + 21.0 * debye_length

    @pytest.mark.parametrize("surface_separation,expected",
                             [   # Test at h=0.
                                 (10.0 * unit.nanometer, 1510.711567854979 * unit.kilojoule_per_mole),
                                 # Test at h=2L.
                                 (20.0 * unit.nanometer, -10.53990009001303 * unit.kilojoule_per_mole),
                                 # Test slightly below h=2L where steric potential is not zero.
                                 ((20.0 - 0.1) * unit.nanometer, -10.74983348860948 * unit.kilojoule_per_mole),
                                 # Test slightly above h=2L where steric potential is strictly zero.
                                 ((20.0 + 0.1) * unit.nanometer, -10.33304182104529 * unit.kilojoule_per_mole),
                                 # Test at h=3L.
                                 (30.0 * unit.nanometer, -1.437662679662967 * unit.kilojoule_per_mole),
                                 # Test at h=20*debye_length, where electrostatic potential should not yet be cutoff.
                                 (100.0 * unit.nanometer, -1.196938856264202e-6 * unit.kilojoule_per_mole)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, surface_separation, expected):
        openmm_context.setPositions([[radius_one + radius_two + surface_separation, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-8, abs=1.0e-13))


# noinspection DuplicatedCode
class TestColloidPotentialsAlgebraicForFourParticles(TestParameters):
    @pytest.fixture(autouse=True, scope="class")
    def add_four_particles(self, openmm_system, colloid_potentials_algebraic,
                           radius_one, radius_two, surface_potential_one, surface_potential_two):
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic.add_particle(radius=radius_one, surface_potential=surface_potential_one)
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        openmm_system.addForce(colloid_potentials_algebraic.steric_potential)
        openmm_system.addForce(colloid_potentials_algebraic.electrostatic_potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture(scope="class")
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    # noinspection DuplicatedCode
    def test_potential_parameters(self, openmm_context, radius_one, radius_two, brush_length, debye_length):
        assert len(openmm_context.getSystem().getForces()) == 2
        steric_force = openmm_context.getSystem().getForce(0)
        electrostatic_force = openmm_context.getSystem().getForce(1)

        assert steric_force.usesPeriodicBoundaryConditions()
        assert electrostatic_force.usesPeriodicBoundaryConditions()

        assert not steric_force.getUseLongRangeCorrection()
        assert not electrostatic_force.getUseLongRangeCorrection()

        assert not steric_force.getUseSwitchingFunction()
        assert electrostatic_force.getUseSwitchingFunction()
        assert electrostatic_force.getSwitchingDistance() == 2.0 * max(radius_one, radius_two) + 20.0 * debye_length

        assert steric_force.getNonbondedMethod() == steric_force.CutoffPeriodic
        assert electrostatic_force.getNonbondedMethod() == electrostatic_force.CutoffPeriodic

        assert steric_force.getCutoffDistance() == 2.0 * max(radius_one, radius_two) + 2.0 * brush_length
        assert electrostatic_force.getCutoffDistance() == 2.0 * max(radius_one, radius_two) + 21.0 * debye_length

    def test_potential(self, openmm_context, radius_one, radius_two):
        openmm_context.setPositions([[0.0, 0.0, 0.0],
                                     # Place at h=30 with reference to first particle.
                                     [2.0 * radius_one + 30.0 * unit.nanometer, 0.0, 0.0],
                                     # Place at h=20 with reference to first particle.
                                     [0.0, radius_one + radius_two + 20.0 * unit.nanometer, 0.0],
                                     # Place at h=10 with reference to first particle.
                                     [0.0, 0.0, radius_one + radius_two + 10.0 * unit.nanometer]])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(1500.591138580165, rel=1.0e-8, abs=1.0e-13))


# noinspection DuplicatedCode
class TestColloidPotentialsAlgebraicWithLogForFourParticles(TestParameters):
    @pytest.fixture(autouse=True, scope="class")
    def add_four_particles(self, openmm_system, colloid_potentials_algebraic_log,
                           radius_one, radius_two, surface_potential_one, surface_potential_two):
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic_log.add_particle(radius=radius_one, surface_potential=surface_potential_one)
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic_log.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        openmm_system.addForce(colloid_potentials_algebraic_log.steric_potential)
        openmm_system.addForce(colloid_potentials_algebraic_log.electrostatic_potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture(scope="class")
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    # noinspection DuplicatedCode
    def test_potential_parameters(self, openmm_context, radius_one, radius_two, brush_length, debye_length):
        assert len(openmm_context.getSystem().getForces()) == 2
        steric_force = openmm_context.getSystem().getForce(0)
        electrostatic_force = openmm_context.getSystem().getForce(1)

        assert steric_force.usesPeriodicBoundaryConditions()
        assert electrostatic_force.usesPeriodicBoundaryConditions()

        assert not steric_force.getUseLongRangeCorrection()
        assert not electrostatic_force.getUseLongRangeCorrection()

        assert not steric_force.getUseSwitchingFunction()
        assert electrostatic_force.getUseSwitchingFunction()
        assert electrostatic_force.getSwitchingDistance() == 2.0 * max(radius_one, radius_two) + 20.0 * debye_length

        assert steric_force.getNonbondedMethod() == steric_force.CutoffPeriodic
        assert electrostatic_force.getNonbondedMethod() == electrostatic_force.CutoffPeriodic

        assert steric_force.getCutoffDistance() == 2.0 * max(radius_one, radius_two) + 2.0 * brush_length
        assert electrostatic_force.getCutoffDistance() == 2.0 * max(radius_one, radius_two) + 21.0 * debye_length

    def test_potential(self, openmm_context, radius_one, radius_two):
        openmm_context.setPositions([[0.0, 0.0, 0.0],
                                     # Place at h=30 with reference to first particle.
                                     [2.0 * radius_one + 30.0 * unit.nanometer, 0.0, 0.0],
                                     # Place at h=20 with reference to first particle.
                                     [0.0, radius_one + radius_two + 20.0 * unit.nanometer, 0.0],
                                     # Place at h=10 with reference to first particle.
                                     [0.0, 0.0, radius_one + radius_two + 10.0 * unit.nanometer]])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(1505.562902813702, rel=1.0e-8, abs=1.0e-13))


if __name__ == '__main__':
    pytest.main([__file__])
