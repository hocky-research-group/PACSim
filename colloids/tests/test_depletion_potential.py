from openmm import Context, LangevinIntegrator, Platform, System, unit, Vec3
import pytest
from colloids.depletion_potential import DepletionPotential
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
    def brush_length(self):
        return 10.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def temperature(self):
        return 300.0 * unit.kelvin

    @pytest.fixture
    def depletant_phi(self):
        return 0.5

    @pytest.fixture
    def maximum_separation(self, radius_one, radius_two, depletant_radius, brush_length):
        return 1.5 * (radius_one + radius_two + 2.0 * brush_length + 2.0 * depletant_radius)

    @pytest.fixture
    def test_separations(self, maximum_separation):
        return np.linspace(0.0, maximum_separation.value_in_unit(unit.nano * unit.meter),
                           num=1000)[1:] * (unit.nano * unit.meter)

    @pytest.fixture
    def side_length(self, radius_one, radius_two, maximum_separation):
        # Make system very large so that we do not care about periodic boundaries.
        return 20.0 * (maximum_separation + 2.0 * max(radius_one, radius_two))

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
    def depletion_potential(self, depletant_phi, depletant_radius, brush_length, temperature):
        return DepletionPotential(depletant_phi, depletant_radius, brush_length, temperature)

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
            depletion_potential.add_particle(radius=radius_one / ((unit.nano * unit.meter) ** 2))
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            depletion_potential.add_particle(radius=-radius_one)

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

    def test_exception_phi(self, depletant_radius, brush_length, temperature):
        # Test exception on negative phi
        with pytest.raises(ValueError):
            DepletionPotential(depletion_phi=-0.5, depletant_radius=depletant_radius, brush_length=brush_length,
                               temperature=temperature)

        # Test exception on phi > 1
        with pytest.raises(ValueError):
            DepletionPotential(depletion_phi=2.0, depletant_radius=depletant_radius, brush_length=brush_length,
                               temperature=temperature)

    def test_exception_depletant_radius(self, depletant_phi, brush_length, temperature):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            DepletionPotential(depletion_phi=depletant_phi, depletant_radius=5.0 * unit.joule,
                               brush_length=brush_length, temperature=temperature)

        # Test exception on negative depletant radius
        with pytest.raises(ValueError):
            DepletionPotential(depletion_phi=depletant_phi, depletant_radius=-5.0 * (unit.nano * unit.meter),
                               brush_length=brush_length, temperature=temperature)

    def test_exception_brush_length(self, depletant_phi, depletant_radius, temperature):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            DepletionPotential(depletion_phi=depletant_phi, depletant_radius=depletant_radius,
                               brush_length=5.0 * unit.joule, temperature=temperature)

        # Test exception on negative brush length
        with pytest.raises(ValueError):
            DepletionPotential(depletion_phi=depletant_phi, depletant_radius=depletant_radius,
                               brush_length=-5.0 * (unit.nano * unit.meter), temperature=temperature)

    def test_exception_temperature(self, depletant_phi, depletant_radius, brush_length):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            DepletionPotential(depletion_phi=depletant_phi, depletant_radius=depletant_radius,
                               brush_length=brush_length, temperature=5.0 * unit.joule)

        # Test exception on negative temperature
        with pytest.raises(ValueError):
            DepletionPotential(depletion_phi=depletant_phi, depletant_radius=depletant_radius,
                               brush_length=brush_length, temperature=-5.0 * unit.kelvin)


# noinspection DuplicatedCode
class TestDepletionPotentialForTwoParticles(TestParameters):
    @pytest.fixture(autouse=True)
    def add_two_particles(self, openmm_system, depletion_potential, radius_one, radius_two):
        openmm_system.addParticle(mass=1.0)
        depletion_potential.add_particle(radius=radius_one, substrate_flag=False)
        openmm_system.addParticle(mass=1.0)
        depletion_potential.add_particle(radius=radius_two, substrate_flag=True)
        for potential in depletion_potential.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @staticmethod
    def depletion_potential_expected(r, radius_one, radius_two, brush_length, depletant_phi, depletant_radius,
                                     temperature):
        rho_colloid1 = radius_one + brush_length
        rho_colloid2 = radius_two + brush_length

        # size ratio 
        q1 = rho_colloid1 / depletant_radius
        q2 = rho_colloid2 / depletant_radius

        n = r / depletant_radius
        kt = (unit.BOLTZMANN_CONSTANT_kB * temperature * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            unit.kilojoules_per_mole)

        return np.where(
            r <= (rho_colloid1 + rho_colloid2 + 2 * depletant_radius),
            - kt * depletant_phi / 16 * (q1 + q2 + 2 - n) ** 2 * (n + 2 * (q1 + q2 + 2)
                                                                  - 3 / n * (q1 ** 2 + q2 ** 2 - 2 * q1 * q2)),
            0.0)

    def test_depletion_potential(self, openmm_context, test_separations, radius_one, radius_two, brush_length,
                                 depletant_phi, depletant_radius, temperature):

        openmm_potentials = np.zeros(len(test_separations))  # use surface separation as test positions
        for index, sep in enumerate(test_separations):
            openmm_context.setPositions([[sep.value_in_unit(unit.nano * unit.meter), 0.0, 0.0], [0.0, 0.0, 0.0]])
            state = openmm_context.getState(getEnergy=True)
            openmm_potentials[index] = (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

        expected_potentials = self.depletion_potential_expected(test_separations, radius_one, radius_two, brush_length,
                                                                depletant_phi, depletant_radius, temperature)

        assert openmm_potentials == pytest.approx(expected_potentials, rel=1.0e-7, abs=1.0e-13)


# noinspection DuplicatedCode
class TestDepletionPotentialForTwoSubstrateParticles(TestParameters):
    @pytest.fixture(autouse=True)
    def add_two_particles(self, openmm_system, depletion_potential, radius_one, radius_two):
        openmm_system.addParticle(mass=1.0)
        depletion_potential.add_particle(radius=radius_one, substrate_flag=True)
        openmm_system.addParticle(mass=1.0)
        depletion_potential.add_particle(radius=radius_two, substrate_flag=True)
        # Add another particle but exclude it from all interactions.
        openmm_system.addParticle(mass=2.0)
        depletion_potential.add_particle(radius=radius_two, substrate_flag=False)
        depletion_potential.add_exclusion(0, 2)
        depletion_potential.add_exclusion(1, 2)
        for potential in depletion_potential.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestParameters class because add_two_particles fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    @staticmethod
    def depletion_potential_expected(r, radius_one, radius_two, brush_length, depletant_phi, depletant_radius,
                                     temperature):
        return np.array([0.0 for _ in range(len(r))])

    def test_depletion_potential(self, openmm_context, test_separations, radius_one, radius_two, brush_length,
                                 depletant_phi, depletant_radius, temperature):

        openmm_potentials = np.zeros(len(test_separations))  # use surface separation as test positions
        for index, sep in enumerate(test_separations):
            openmm_context.setPositions([[sep.value_in_unit(unit.nano * unit.meter), 0.0, 0.0], [0.0, 0.0, 0.0],
                                         [1.0, 1.0, 1.0]])
            state = openmm_context.getState(getEnergy=True)
            openmm_potentials[index] = (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

        expected_potentials = self.depletion_potential_expected(test_separations, radius_one, radius_two, brush_length,
                                                                depletant_phi, depletant_radius, temperature)

        assert openmm_potentials == pytest.approx(expected_potentials, rel=1.0e-7, abs=1.0e-13)


if __name__ == '__main__':
    pytest.main([__file__])
