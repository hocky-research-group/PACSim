from openmm import Context, LangevinIntegrator, Platform, System, unit, Vec3
import pytest
from colloids import ColloidPotentialsParameters, ColloidPotentialsAlgebraic, ColloidPotentialsTabulated


class TestParameters(object):
    @pytest.fixture
    def radius_one(self):
        return 325.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def radius_two(self):
        return 65.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def surface_potential_one(self):
        return 50.0 * (unit.milli * unit.volt)

    @pytest.fixture
    def surface_potential_two(self):
        return -40.0 * (unit.milli * unit.volt)

    @pytest.fixture
    def brush_density(self):
        return 0.09 / ((unit.nano * unit.meter) ** 2)

    @pytest.fixture
    def brush_length(self):
        return 10.0 * (unit.nano * unit.meter)

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
    def maximum_surface_separation(self, radius_one, radius_two, debye_length):
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
    def colloid_potentials_algebraic(self, colloid_potentials_parameters):
        return ColloidPotentialsAlgebraic(colloid_potentials_parameters=colloid_potentials_parameters, use_log=False,
                                          cutoff_factor=21.0, periodic_boundary_conditions=True)

    @pytest.fixture
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
            return colloid_potentials_tabulated

    @pytest.fixture
    def colloid_potentials_algebraic_log(self, colloid_potentials_parameters):
        return ColloidPotentialsAlgebraic(colloid_potentials_parameters=colloid_potentials_parameters, use_log=True,
                                          cutoff_factor=21.0, periodic_boundary_conditions=True)

    @pytest.fixture
    def colloid_potentials_tabulated_log(self, radius_one, radius_two, surface_potential_one, surface_potential_two,
                                         colloid_potentials_parameters):
        return ColloidPotentialsTabulated(radius_one=radius_one, radius_two=radius_two,
                                          surface_potential_one=surface_potential_one,
                                          surface_potential_two=surface_potential_two,
                                          colloid_potentials_parameters=colloid_potentials_parameters,
                                          use_log=True, cutoff_factor=21.0, periodic_boundary_conditions=True)

    @pytest.fixture(params=["algebraic", "tabulated"])
    def colloid_potentials_log(self, colloid_potentials_algebraic_log, colloid_potentials_tabulated_log, request):
        if request.param == "algebraic":
            return colloid_potentials_algebraic_log
        else:
            assert request.param == "tabulated"
            return colloid_potentials_tabulated_log

    @pytest.fixture
    def openmm_platform(self):
        return Platform.getPlatformByName("Reference")

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)


class TestColloidPotentialsExceptions(TestParameters):
    def test_exception_radius(self, colloid_potentials, radius_one, surface_potential_one):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            colloid_potentials.add_particle(
                radius=radius_one / ((unit.nano * unit.meter) ** 2), surface_potential=surface_potential_one)
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            colloid_potentials.add_particle(
                radius=-radius_one, surface_potential=surface_potential_one)

    def test_exception_surface_potential(self, colloid_potentials, radius_one, surface_potential_one):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            colloid_potentials.add_particle(radius=radius_one,
                                            surface_potential=surface_potential_one / (unit.milli * unit.volt) ** 2)

    def test_exception_no_particles_added(self, colloid_potentials):
        with pytest.raises(RuntimeError):
            for _ in colloid_potentials.yield_potentials():
                pass

    def test_exception_add_particle_after_yield_potentials(self, colloid_potentials, radius_one, surface_potential_one):
        colloid_potentials.add_particle(radius=radius_one, surface_potential=surface_potential_one)
        for _ in colloid_potentials.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            colloid_potentials.add_particle(radius=radius_one, surface_potential=surface_potential_one)

    def test_exception_colloid_potentials_tabulated_same_surface_potential(
            self, radius_one, radius_two, surface_potential_one, colloid_potentials_parameters):
        with pytest.raises(ValueError):
            ColloidPotentialsTabulated(radius_one=radius_one, radius_two=radius_two,
                                       surface_potential_one=surface_potential_one,
                                       surface_potential_two=surface_potential_one,
                                       colloid_potentials_parameters=colloid_potentials_parameters, use_log=False,
                                       cutoff_factor=21.0, periodic_boundary_conditions=True)

    def test_exception_colloid_potentials_tabulated_add_wrong_particles(
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
            colloid_potentials_tabulated.add_particle(radius=radius_two, surface_potential=surface_potential_one)


# noinspection DuplicatedCode
class TestColloidPotentialsForTwoParticles(TestParameters):
    @pytest.fixture(autouse=True)
    def add_two_particles(self, openmm_system, colloid_potentials, radius_one, radius_two, surface_potential_one,
                          surface_potential_two):
        openmm_system.addParticle(mass=1.0)
        colloid_potentials.add_particle(radius=radius_one, surface_potential=surface_potential_one,
                                        substrate_flag=False)
        openmm_system.addParticle(mass=1.0)
        colloid_potentials.add_particle(radius=radius_two, surface_potential=surface_potential_two,
                                        substrate_flag=False)
        # Add another particle but exclude it from all interactions.
        openmm_system.addParticle(mass=2.0)
        colloid_potentials.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        colloid_potentials.add_exclusion(0, 2)
        colloid_potentials.add_exclusion(1, 2)
        for potential in colloid_potentials.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    # noinspection DuplicatedCode
    def test_potential_parameters(self, openmm_context, radius_one, radius_two, brush_length, debye_length,
                                  colloid_potentials):
        # This test is pretty specific to the implementation of the colloid potentials.
        if isinstance(colloid_potentials, ColloidPotentialsAlgebraic):
            assert len(openmm_context.getSystem().getForces()) == 2
            steric_force = openmm_context.getSystem().getForce(0)
            electrostatic_force = openmm_context.getSystem().getForce(1)

            assert steric_force.usesPeriodicBoundaryConditions()
            assert electrostatic_force.usesPeriodicBoundaryConditions()

            assert not steric_force.getUseLongRangeCorrection()
            assert not electrostatic_force.getUseLongRangeCorrection()

            assert not steric_force.getUseSwitchingFunction()
            assert electrostatic_force.getUseSwitchingFunction()
            assert (electrostatic_force.getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))

            assert steric_force.getNonbondedMethod() == steric_force.CutoffPeriodic
            assert electrostatic_force.getNonbondedMethod() == electrostatic_force.CutoffPeriodic

            assert (steric_force.getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 2.0 * brush_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (electrostatic_force.getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
        else:
            assert isinstance(colloid_potentials, ColloidPotentialsTabulated)
            assert len(openmm_context.getSystem().getForces()) == 3
            for force in openmm_context.getSystem().getForces():
                assert force.usesPeriodicBoundaryConditions()
                assert not force.getUseLongRangeCorrection()
                assert force.getUseSwitchingFunction()
                assert force.getNonbondedMethod() == force.CutoffPeriodic
                assert force.getNumInteractionGroups() == 1
            assert (openmm_context.getSystem().getForce(0).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_one + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(1).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_two + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(2).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx(
                        (radius_one + radius_two + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                        rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(0).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_one + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(1).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_two + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(2).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx(
                        (radius_one + radius_two + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                        rel=1e-12, abs=1e-12))
            assert openmm_context.getSystem().getForce(0).getInteractionGroupParameters(0) == [(0,), (0,)]
            assert openmm_context.getSystem().getForce(1).getInteractionGroupParameters(0) == [(1, 2), (1, 2)]
            assert openmm_context.getSystem().getForce(2).getInteractionGroupParameters(0) == [(0,), (1, 2)]

    @pytest.mark.parametrize("surface_separation,expected",
                             [   # Test at h=0.
                                 (10.0 * (unit.nano * unit.meter), 1505.829355134808 * unit.kilojoule_per_mole),
                                 # Test at h=2L.
                                 (20.0 * (unit.nano * unit.meter), -10.63613061419315 * unit.kilojoule_per_mole),
                                 # Test slightly below h=2L where steric potential is not zero.
                                 ((20.0 - 0.1) * (unit.nano * unit.meter),
                                  -10.84996692702675 * unit.kilojoule_per_mole),
                                 # Test slightly above h=2L where steric potential is strictly zero.
                                 ((20.0 + 0.1) * (unit.nano * unit.meter),
                                  -10.42552111714948 * unit.kilojoule_per_mole),
                                 # Test at h=3L.
                                 (30.0 * (unit.nano * unit.meter), -1.439443749213437 * unit.kilojoule_per_mole),
                                 # Test at h=20*debye_length, where electrostatic potential should not yet be cutoff.
                                 (100.0 * (unit.nano * unit.meter), -1.196938817005087e-6 * unit.kilojoule_per_mole)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, surface_separation, expected):
        openmm_context.setPositions([[radius_one + radius_two + surface_separation, 0.0, 0.0],
                                     [0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))


# noinspection DuplicatedCode
class TestColloidPotentialsWithLogForTwoParticles(TestParameters):
    @pytest.fixture(autouse=True)
    def add_two_particles(self, openmm_system, colloid_potentials_log,
                          radius_one, radius_two, surface_potential_one, surface_potential_two):
        openmm_system.addParticle(mass=1.0)
        colloid_potentials_log.add_particle(radius=radius_one, surface_potential=surface_potential_one)
        openmm_system.addParticle(mass=1.0)
        colloid_potentials_log.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        # Add another particle but exclude it from all interactions.
        openmm_system.addParticle(mass=2.0)
        colloid_potentials_log.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        colloid_potentials_log.add_exclusion(0, 2)
        colloid_potentials_log.add_exclusion(1, 2)
        for potential in colloid_potentials_log.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    # noinspection DuplicatedCode
    def test_potential_parameters(self, openmm_context, radius_one, radius_two, brush_length, debye_length,
                                  colloid_potentials_log):
        # This test is pretty specific to the implementation of the colloid potentials.
        if isinstance(colloid_potentials_log, ColloidPotentialsAlgebraic):
            assert len(openmm_context.getSystem().getForces()) == 2
            steric_force = openmm_context.getSystem().getForce(0)
            electrostatic_force = openmm_context.getSystem().getForce(1)

            assert steric_force.usesPeriodicBoundaryConditions()
            assert electrostatic_force.usesPeriodicBoundaryConditions()

            assert not steric_force.getUseLongRangeCorrection()
            assert not electrostatic_force.getUseLongRangeCorrection()

            assert not steric_force.getUseSwitchingFunction()
            assert electrostatic_force.getUseSwitchingFunction()
            assert (electrostatic_force.getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))

            assert steric_force.getNonbondedMethod() == steric_force.CutoffPeriodic
            assert electrostatic_force.getNonbondedMethod() == electrostatic_force.CutoffPeriodic

            assert (steric_force.getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 2.0 * brush_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (electrostatic_force.getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
        else:
            assert isinstance(colloid_potentials_log, ColloidPotentialsTabulated)
            assert len(openmm_context.getSystem().getForces()) == 3
            for force in openmm_context.getSystem().getForces():
                assert force.usesPeriodicBoundaryConditions()
                assert not force.getUseLongRangeCorrection()
                assert force.getUseSwitchingFunction()
                assert force.getNonbondedMethod() == force.CutoffPeriodic
                assert force.getNumInteractionGroups() == 1
            assert (openmm_context.getSystem().getForce(0).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_one + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(1).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_two + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(2).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx(
                        (radius_one + radius_two + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                        rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(0).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_one + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(1).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_two + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(2).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx(
                        (radius_one + radius_two + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                        rel=1e-12, abs=1e-12))
            assert openmm_context.getSystem().getForce(0).getInteractionGroupParameters(0) == [(0,), (0,)]
            assert openmm_context.getSystem().getForce(1).getInteractionGroupParameters(0) == [(1, 2), (1, 2)]
            assert openmm_context.getSystem().getForce(2).getInteractionGroupParameters(0) == [(0,), (1, 2)]

    @pytest.mark.parametrize("surface_separation,expected",
                             [   # Test at h=0.
                                 (10.0 * (unit.nano * unit.meter), 1510.711567854979 * unit.kilojoule_per_mole),
                                 # Test at h=2L.
                                 (20.0 * (unit.nano * unit.meter), -10.53990009001303 * unit.kilojoule_per_mole),
                                 # Test slightly below h=2L where steric potential is not zero.
                                 ((20.0 - 0.1) * (unit.nano * unit.meter),
                                  -10.74983348860948 * unit.kilojoule_per_mole),
                                 # Test slightly above h=2L where steric potential is strictly zero.
                                 ((20.0 + 0.1) * (unit.nano * unit.meter),
                                  -10.33304182104529 * unit.kilojoule_per_mole),
                                 # Test at h=3L.
                                 (30.0 * (unit.nano * unit.meter), -1.437662679662967 * unit.kilojoule_per_mole),
                                 # Test at h=20*debye_length, where electrostatic potential should not yet be cutoff.
                                 (100.0 * (unit.nano * unit.meter), -1.196938856264202e-6 * unit.kilojoule_per_mole)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, surface_separation, expected):
        openmm_context.setPositions([[radius_one + radius_two + surface_separation, 0.0, 0.0],
                                     [0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))


class TestColloidPotentialsAlgebraicForTwoSubstrateParticles(TestParameters):
    @pytest.fixture(autouse=True, params=["algebraic", "algebraic_log"])
    def add_two_substrate_particles(self, openmm_system, colloid_potentials_algebraic, colloid_potentials_algebraic_log,
                                    radius_one, radius_two, surface_potential_one, surface_potential_two, request):
        if request.param == "algebraic":
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic.add_particle(radius=radius_one, surface_potential=surface_potential_one,
                                                      substrate_flag=True)
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic.add_particle(radius=radius_two, surface_potential=surface_potential_two,
                                                      substrate_flag=True)
            for potential in colloid_potentials_algebraic.yield_potentials():
                openmm_system.addForce(potential)
        else:
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic_log.add_particle(radius=radius_one, surface_potential=surface_potential_one,
                                                      substrate_flag=True)
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic_log.add_particle(radius=radius_two, surface_potential=surface_potential_two,
                                                      substrate_flag=True)
            for potential in colloid_potentials_algebraic_log.yield_potentials():
                openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    # The potential should always be zero between the substrate particles.
    @pytest.mark.parametrize("surface_separation",
                             [   # Test at h=0.
                                 10.0 * (unit.nano * unit.meter),
                                 # Test at h=2L.
                                 20.0 * (unit.nano * unit.meter),
                                 # Test slightly below h=2L.
                                 (20.0 - 0.1) * (unit.nano * unit.meter),
                                 # Test slightly above h=2L.
                                 (20.0 + 0.1) * (unit.nano * unit.meter),
                                 # Test at h=3L.
                                 30.0 * (unit.nano * unit.meter),
                                 # Test at h=20*debye_length.
                                 100.0 * (unit.nano * unit.meter)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, surface_separation):
        openmm_context.setPositions([[radius_one + radius_two + surface_separation, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(0.0, rel=1.0e-7, abs=1.0e-13))


class TestColloidPotentialsTabulatedForTwoSubstrateParticles(TestParameters):
    def test_add_substrate_particle_raises_exception(self, colloid_potentials_tabulated,
                                                     colloid_potentials_tabulated_log, radius_one,
                                                     surface_potential_one):
        with pytest.raises(ValueError):
            colloid_potentials_tabulated.add_particle(radius=radius_one, surface_potential=surface_potential_one,
                                                      substrate_flag=True)
        with pytest.raises(ValueError):
            colloid_potentials_tabulated_log.add_particle(radius=radius_one, surface_potential=surface_potential_one,
                                                          substrate_flag=True)


class TestColloidPotentialsAlgebraicForOneParticleOneSubstrateParticle(TestParameters):
    @pytest.fixture(autouse=True)
    def add_two_particles(self, openmm_system, colloid_potentials_algebraic, radius_one, radius_two,
                          surface_potential_one, surface_potential_two):
        openmm_system.addParticle(mass=1.0)
        colloid_potentials_algebraic.add_particle(radius=radius_one, surface_potential=surface_potential_one,
                                                  substrate_flag=False)
        openmm_system.addParticle(mass=1.0)
        colloid_potentials_algebraic.add_particle(radius=radius_two, surface_potential=surface_potential_two,
                                                  substrate_flag=True)
        for potential in colloid_potentials_algebraic.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @pytest.mark.parametrize("surface_separation,expected",
                             [   # Test at h=0.
                                 (10.0 * (unit.nano * unit.meter), 1505.829355134808 * unit.kilojoule_per_mole),
                                 # Test at h=2L.
                                 (20.0 * (unit.nano * unit.meter), -10.63613061419315 * unit.kilojoule_per_mole),
                                 # Test slightly below h=2L where steric potential is not zero.
                                 ((20.0 - 0.1) * (unit.nano * unit.meter),
                                  -10.84996692702675 * unit.kilojoule_per_mole),
                                 # Test slightly above h=2L where steric potential is strictly zero.
                                 ((20.0 + 0.1) * (unit.nano * unit.meter),
                                  -10.42552111714948 * unit.kilojoule_per_mole),
                                 # Test at h=3L.
                                 (30.0 * (unit.nano * unit.meter), -1.439443749213437 * unit.kilojoule_per_mole),
                                 # Test at h=20*debye_length, where electrostatic potential should not yet be cutoff.
                                 (100.0 * (unit.nano * unit.meter), -1.196938817005087e-6 * unit.kilojoule_per_mole)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, surface_separation, expected):
        openmm_context.setPositions([[radius_one + radius_two + surface_separation, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))


# noinspection DuplicatedCode
class TestColloidPotentialsAlgebraicWithLogForOneParticleOneSubstrateParticle(TestParameters):
    @pytest.fixture(autouse=True)
    def add_two_particles(self, openmm_system, colloid_potentials_algebraic_log,
                          radius_one, radius_two, surface_potential_one, surface_potential_two):
        openmm_system.addParticle(mass=1.0)
        colloid_potentials_algebraic_log.add_particle(radius=radius_one, surface_potential=surface_potential_one,
                                                      substrate_flag=False)
        openmm_system.addParticle(mass=1.0)
        colloid_potentials_algebraic_log.add_particle(radius=radius_two, surface_potential=surface_potential_two,
                                                      substrate_flag=True)
        for potential in colloid_potentials_algebraic_log.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @pytest.mark.parametrize("surface_separation,expected",
                             [   # Test at h=0.
                                 (10.0 * (unit.nano * unit.meter), 1510.711567854979 * unit.kilojoule_per_mole),
                                 # Test at h=2L.
                                 (20.0 * (unit.nano * unit.meter), -10.53990009001303 * unit.kilojoule_per_mole),
                                 # Test slightly below h=2L where steric potential is not zero.
                                 ((20.0 - 0.1) * (unit.nano * unit.meter),
                                  -10.74983348860948 * unit.kilojoule_per_mole),
                                 # Test slightly above h=2L where steric potential is strictly zero.
                                 ((20.0 + 0.1) * (unit.nano * unit.meter),
                                  -10.33304182104529 * unit.kilojoule_per_mole),
                                 # Test at h=3L.
                                 (30.0 * (unit.nano * unit.meter), -1.437662679662967 * unit.kilojoule_per_mole),
                                 # Test at h=20*debye_length, where electrostatic potential should not yet be cutoff.
                                 (100.0 * (unit.nano * unit.meter), -1.196938856264202e-6 * unit.kilojoule_per_mole)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, surface_separation, expected):
        openmm_context.setPositions([[radius_one + radius_two + surface_separation, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))


# noinspection DuplicatedCode
class TestColloidPotentialsForFourParticles(TestParameters):
    @pytest.fixture(autouse=True)
    def add_four_particles(self, openmm_system, colloid_potentials,
                           radius_one, radius_two, surface_potential_one, surface_potential_two):
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials.add_particle(radius=radius_one, surface_potential=surface_potential_one)
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        # Add another particle but exclude it from all interactions
        openmm_system.addParticle(mass=2.0)
        colloid_potentials.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        colloid_potentials.add_exclusion(0, 4)
        colloid_potentials.add_exclusion(1, 4)
        colloid_potentials.add_exclusion(2, 4)
        colloid_potentials.add_exclusion(3, 4)
        for potential in colloid_potentials.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    # noinspection DuplicatedCode
    def test_potential_parameters(self, openmm_context, radius_one, radius_two, brush_length, debye_length,
                                  colloid_potentials):
        # This test is pretty specific to the implementation of the colloid potentials.
        if isinstance(colloid_potentials, ColloidPotentialsAlgebraic):
            assert len(openmm_context.getSystem().getForces()) == 2
            steric_force = openmm_context.getSystem().getForce(0)
            electrostatic_force = openmm_context.getSystem().getForce(1)

            assert steric_force.usesPeriodicBoundaryConditions()
            assert electrostatic_force.usesPeriodicBoundaryConditions()

            assert not steric_force.getUseLongRangeCorrection()
            assert not electrostatic_force.getUseLongRangeCorrection()

            assert not steric_force.getUseSwitchingFunction()
            assert electrostatic_force.getUseSwitchingFunction()
            assert (electrostatic_force.getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))

            assert steric_force.getNonbondedMethod() == steric_force.CutoffPeriodic
            assert electrostatic_force.getNonbondedMethod() == electrostatic_force.CutoffPeriodic

            assert (steric_force.getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 2.0 * brush_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (electrostatic_force.getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
        else:
            assert isinstance(colloid_potentials, ColloidPotentialsTabulated)
            assert len(openmm_context.getSystem().getForces()) == 3
            for force in openmm_context.getSystem().getForces():
                assert force.usesPeriodicBoundaryConditions()
                assert not force.getUseLongRangeCorrection()
                assert force.getUseSwitchingFunction()
                assert force.getNonbondedMethod() == force.CutoffPeriodic
                assert force.getNumInteractionGroups() == 1
            assert (openmm_context.getSystem().getForce(0).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_one + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(1).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_two + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(2).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx(
                        (radius_one + radius_two + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                        rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(0).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_one + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(1).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_two + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(2).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx(
                        (radius_one + radius_two + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                        rel=1e-12, abs=1e-12))
            assert openmm_context.getSystem().getForce(0).getInteractionGroupParameters(0) == [(0, 1), (0, 1)]
            assert openmm_context.getSystem().getForce(1).getInteractionGroupParameters(0) == [(2, 3, 4), (2, 3, 4)]
            assert openmm_context.getSystem().getForce(2).getInteractionGroupParameters(0) == [(0, 1), (2, 3, 4)]

    @pytest.mark.parametrize("positions,expected",
                             [
                                 ([[0.0, 0.0, 0.0],
                                     # Place at h=30 with reference to first particle.
                                     [680.0 * (unit.nano * unit.meter), 0.0, 0.0],
                                     # Place at h=20 with reference to first particle.
                                     [0.0, 410.0 * (unit.nano * unit.meter), 0.0],
                                     # Place at h=10 with reference to first particle.
                                     [0.0, 0.0, 400.0 * (unit.nano * unit.meter)],
                                     [100.0, 100.0, 100.0] * (unit.nano * unit.meter)],
                                  1500.591138580165 * unit.kilojoule_per_mole),
                                 ([[0.0, 0.0, 0.0],
                                   # Place at h=10 with reference to first particle.
                                   [660.0 * (unit.nano * unit.meter), 0.0, 0.0],
                                   # Place at h=30 with reference to first particle.
                                   [0.0, 420.0 * (unit.nano * unit.meter), 0.0],
                                   # Place at h=100 with reference to first particle.
                                   [0.0, 0.0, 490.0 * (unit.nano * unit.meter)],
                                   [100.0, 100.0, 100.0] * (unit.nano * unit.meter)],
                                  2933.97721160759 * unit.kilojoule_per_mole),
                                 ([# Place at h=25 with reference to last particle.
                                   [0.0, 0.0, 515.0],
                                   # Place at h=15 with reference to last particle.
                                   [0.0, 405.0, 0.0],
                                   # Place at h=10 with reference to last particle.
                                   [140, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [100.0, 100.0, 100.0] * (unit.nano * unit.meter)],
                                  14398.17405177216 * unit.kilojoule_per_mole)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, positions, expected):
        openmm_context.setPositions(positions)
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))


class TestColloidPotentialsAlgebraicForTwoParticlesTwoSubstrateParticles(TestParameters):
    @pytest.fixture(autouse=True)
    def add_four_particles(self, openmm_system, colloid_potentials_algebraic,
                           radius_one, radius_two, surface_potential_one, surface_potential_two):
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic.add_particle(radius=radius_one, surface_potential=surface_potential_one,
                                                      substrate_flag=False)
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic.add_particle(radius=radius_two, surface_potential=surface_potential_two,
                                                      substrate_flag=True)
        for potential in colloid_potentials_algebraic.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @pytest.mark.parametrize("positions,expected",
                             [
                                 ([[0.0, 0.0, 0.0],
                                     # Place at h=30 with reference to first particle.
                                     [680.0 * (unit.nano * unit.meter), 0.0, 0.0],
                                     # Place at h=20 with reference to first particle.
                                     [0.0, 410.0 * (unit.nano * unit.meter), 0.0],
                                     # Place at h=10 with reference to first particle.
                                     [0.0, 0.0, 400.0 * (unit.nano * unit.meter)]],
                                  1500.591138580165 * unit.kilojoule_per_mole),
                                 ([[0.0, 0.0, 0.0],
                                   # Place at h=10 with reference to first particle.
                                   [660.0 * (unit.nano * unit.meter), 0.0, 0.0],
                                   # Place at h=30 with reference to first particle.
                                   [0.0, 420.0 * (unit.nano * unit.meter), 0.0],
                                   # Place at h=100 with reference to first particle.
                                   [0.0, 0.0, 490.0 * (unit.nano * unit.meter)]],
                                  2933.97721160759 * unit.kilojoule_per_mole),
                                 ([# Place at h=25 with reference to last particle.
                                   [0.0, 0.0, 515.0],
                                   # Place at h=15 with reference to last particle.
                                   [0.0, 405.0, 0.0],
                                   # Place at h=10 with reference to last particle.
                                   [140, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]],
                                  13832.31028122305 * unit.kilojoule_per_mole)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, positions, expected):
        openmm_context.setPositions(positions)
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))


# noinspection DuplicatedCode
class TestColloidPotentialsWithLogForFourParticles(TestParameters):
    @pytest.fixture(autouse=True)
    def add_four_particles(self, openmm_system, colloid_potentials_log,
                           radius_one, radius_two, surface_potential_one, surface_potential_two):
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_log.add_particle(radius=radius_one, surface_potential=surface_potential_one)
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_log.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        # Add another particle but exclude it from all interactions
        openmm_system.addParticle(mass=2.0)
        colloid_potentials_log.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        colloid_potentials_log.add_exclusion(0, 4)
        colloid_potentials_log.add_exclusion(1, 4)
        colloid_potentials_log.add_exclusion(2, 4)
        colloid_potentials_log.add_exclusion(3, 4)
        for potential in colloid_potentials_log.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    # noinspection DuplicatedCode
    def test_potential_parameters(self, openmm_context, radius_one, radius_two, brush_length, debye_length,
                                  colloid_potentials_log):
        # This test is pretty specific to the implementation of the colloid potentials.
        if isinstance(colloid_potentials_log, ColloidPotentialsAlgebraic):
            assert len(openmm_context.getSystem().getForces()) == 2
            steric_force = openmm_context.getSystem().getForce(0)
            electrostatic_force = openmm_context.getSystem().getForce(1)

            assert steric_force.usesPeriodicBoundaryConditions()
            assert electrostatic_force.usesPeriodicBoundaryConditions()

            assert not steric_force.getUseLongRangeCorrection()
            assert not electrostatic_force.getUseLongRangeCorrection()

            assert not steric_force.getUseSwitchingFunction()
            assert electrostatic_force.getUseSwitchingFunction()
            assert (electrostatic_force.getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))

            assert steric_force.getNonbondedMethod() == steric_force.CutoffPeriodic
            assert electrostatic_force.getNonbondedMethod() == electrostatic_force.CutoffPeriodic

            assert (steric_force.getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 2.0 * brush_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (electrostatic_force.getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * max(radius_one, radius_two)
                                      + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
        else:
            assert isinstance(colloid_potentials_log, ColloidPotentialsTabulated)
            assert len(openmm_context.getSystem().getForces()) == 3
            for force in openmm_context.getSystem().getForces():
                assert force.usesPeriodicBoundaryConditions()
                assert not force.getUseLongRangeCorrection()
                assert force.getUseSwitchingFunction()
                assert force.getNonbondedMethod() == force.CutoffPeriodic
                assert force.getNumInteractionGroups() == 1
            assert (openmm_context.getSystem().getForce(0).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_one + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(1).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_two + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(2).getCutoffDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx(
                        (radius_one + radius_two + 21.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                        rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(0).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_one + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(1).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx((2.0 * radius_two + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                                     rel=1e-12, abs=1e-12))
            assert (openmm_context.getSystem().getForce(2).getSwitchingDistance().value_in_unit(unit.nano * unit.meter)
                    == pytest.approx(
                        (radius_one + radius_two + 20.0 * debye_length).value_in_unit(unit.nano * unit.meter),
                        rel=1e-12, abs=1e-12))
            assert openmm_context.getSystem().getForce(0).getInteractionGroupParameters(0) == [(0, 1), (0, 1)]
            assert openmm_context.getSystem().getForce(1).getInteractionGroupParameters(0) == [(2, 3, 4), (2, 3, 4)]
            assert openmm_context.getSystem().getForce(2).getInteractionGroupParameters(0) == [(0, 1), (2, 3, 4)]

    @pytest.mark.parametrize("positions,expected",
                             [
                                 ([[0.0, 0.0, 0.0],
                                     # Place at h=30 with reference to first particle.
                                     [680.0 * (unit.nano * unit.meter), 0.0, 0.0],
                                     # Place at h=20 with reference to first particle.
                                     [0.0, 410.0 * (unit.nano * unit.meter), 0.0],
                                     # Place at h=10 with reference to first particle.
                                     [0.0, 0.0, 400.0 * (unit.nano * unit.meter)],
                                     [100.0, 100.0, 100.0] * (unit.nano * unit.meter)],
                                  1505.562902813702 * unit.kilojoule_per_mole),
                                 ([[0.0, 0.0, 0.0],
                                   # Place at h=10 with reference to first particle.
                                   [660.0 * (unit.nano * unit.meter), 0.0, 0.0],
                                   # Place at h=30 with reference to first particle.
                                   [0.0, 420.0 * (unit.nano * unit.meter), 0.0],
                                   # Place at h=100 with reference to first particle.
                                   [0.0, 0.0, 490.0 * (unit.nano * unit.meter)],
                                   [100.0, 100.0, 100.0] * (unit.nano * unit.meter)],
                                  2915.670694976501 * unit.kilojoule_per_mole),
                                 ([# Place at h=25 with reference to last particle.
                                   [0.0, 0.0, 515.0],
                                   # Place at h=15 with reference to last particle.
                                   [0.0, 405.0, 0.0],
                                   # Place at h=10 with reference to last particle.
                                   [140, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [100.0, 100.0, 100.0] * (unit.nano * unit.meter)],
                                  14284.77185919515 * unit.kilojoule_per_mole)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, positions, expected):
        openmm_context.setPositions(positions)
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))


class TestColloidPotentialsAlgebraicWithLogForTwoParticlesTwoSubstrateParticles(TestParameters):
    @pytest.fixture(autouse=True)
    def add_four_particles(self, openmm_system, colloid_potentials_algebraic_log,
                           radius_one, radius_two, surface_potential_one, surface_potential_two):
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic_log.add_particle(radius=radius_one, surface_potential=surface_potential_one,
                                                          substrate_flag=True)
        for _ in range(2):
            openmm_system.addParticle(mass=1.0)
            colloid_potentials_algebraic_log.add_particle(radius=radius_two, surface_potential=surface_potential_two,
                                                          substrate_flag=False)
        for potential in colloid_potentials_algebraic_log.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @pytest.mark.parametrize("positions,expected",
                             [
                                 ([[0.0, 0.0, 0.0],
                                     # Place at h=30 with reference to first particle.
                                     [680.0 * (unit.nano * unit.meter), 0.0, 0.0],
                                     # Place at h=20 with reference to first particle.
                                     [0.0, 410.0 * (unit.nano * unit.meter), 0.0],
                                     # Place at h=10 with reference to first particle.
                                     [0.0, 0.0, 400.0 * (unit.nano * unit.meter)]],
                                  1500.171667764966 * unit.kilojoule_per_mole),
                                 ([[0.0, 0.0, 0.0],
                                   # Place at h=10 with reference to first particle.
                                   [660.0 * (unit.nano * unit.meter), 0.0, 0.0],
                                   # Place at h=30 with reference to first particle.
                                   [0.0, 420.0 * (unit.nano * unit.meter), 0.0],
                                   # Place at h=100 with reference to first particle.
                                   [0.0, 0.0, 490.0 * (unit.nano * unit.meter)]],
                                  -1.437663876601824 * unit.kilojoule_per_mole),
                                 ([# Place at h=25 with reference to last particle.
                                   [0.0, 0.0, 515.0],
                                   # Place at h=15 with reference to last particle.
                                   [0.0, 405.0, 0.0],
                                   # Place at h=10 with reference to last particle.
                                   [140, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]],
                                  688.4738152224481 * unit.kilojoule_per_mole)
                             ])
    def test_potential(self, openmm_context, radius_one, radius_two, positions, expected):
        openmm_context.setPositions(positions)
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=1.0e-7, abs=1.0e-13))


if __name__ == '__main__':
    pytest.main([__file__])
