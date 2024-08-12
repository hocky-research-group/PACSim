from openmm import Context, LangevinIntegrator, NonbondedForce, OpenMMException, Platform, System, unit, Vec3
import pytest
from colloids import Gravity
import numpy as np


class TestGravityParameters(object):
    @pytest.fixture
    def particle_density(self):
        return 1.05 * (unit.gram / (unit.centi * unit.meter) ** 3)

    @pytest.fixture
    def particle_radius(self):
        return 105.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def gravitational_acceleration(self):
        return 9.8 * unit.meter / unit.second ** 2

    @pytest.fixture
    def water_density(self):
        return 0.998 * (unit.gram / (unit.centi * unit.meter) ** 3)

    @pytest.fixture
    def box_length(self):
        return 1000.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def openmm_system(self, box_length):
        system = System()
        system.setDefaultPeriodicBoxVectors(Vec3(box_length, 0.0, 0.0),
                                            Vec3(0.0, box_length, 0.0),
                                            Vec3(0.0, 0.0, box_length))
        return system

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)

    @pytest.fixture
    def gravitational_potential(self, gravitational_acceleration, water_density, particle_density):
        return Gravity(gravitational_acceleration, water_density, particle_density)

    @pytest.fixture
    def z_num_test_values(self):
        return 1000

    @pytest.fixture
    def test_z_positions(self, box_length, z_num_test_values):
        return np.linspace(-box_length.value_in_unit(unit.nano * unit.meter) / 2.0,
                           box_length.value_in_unit(unit.nano * unit.meter) / 2.0,
                           num=z_num_test_values) * (unit.nano * unit.meter)


class TestGravityExceptions(TestGravityParameters):
    def test_exception_radius(self, particle_radius, gravitational_potential):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            gravitational_potential.add_particle(index=0, radius=particle_radius * (unit.nano * unit.meter))
        # Test exception on negative radius. 
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            gravitational_potential.add_particle(index=0, radius=-particle_radius)

    def test_exception_no_particles_added(self, gravitational_potential):
        with pytest.raises(RuntimeError):
            for _ in gravitational_potential.yield_potentials():
                pass

    def test_exception_add_particle_after_yield_potentials(self, particle_radius, gravitational_potential):
        gravitational_potential.add_particle(index=0, radius=particle_radius)
        for _ in gravitational_potential.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            gravitational_potential.add_particle(index=1, radius=particle_radius)

    def test_exception_gravitational_acceleration(self, particle_density, gravitational_acceleration, water_density):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            Gravity(gravitational_acceleration=gravitational_acceleration / unit.meter,
                    water_density=water_density, particle_density=particle_density)

        # Test exception on negative gravitational acceleration.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            Gravity(gravitational_acceleration=-gravitational_acceleration,
                    water_density=water_density, particle_density=particle_density)

    def test_exception_water_density(self, particle_density, gravitational_acceleration, water_density):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            Gravity(gravitational_acceleration=gravitational_acceleration,
                    water_density=water_density / unit.gram, particle_density=particle_density)

        # Test exception on negative water density.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            Gravity(gravitational_acceleration=gravitational_acceleration,
                    water_density=-water_density, particle_density=particle_density)

    def test_exception_particle_density(self, particle_density, gravitational_acceleration, water_density):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            Gravity(gravitational_acceleration=gravitational_acceleration, water_density=water_density,
                    particle_density=water_density / unit.gram)

        # Test exception on negative particle density.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            Gravity(gravitational_acceleration=gravitational_acceleration, water_density=water_density,
                    particle_density=-particle_density)


class TestGravity(TestGravityParameters):
    @staticmethod
    def gravitational_force_exp(particle_density, water_density, particle_radius, z, g):
        return ((particle_density - water_density) * 4.0 / 3.0 * np.pi * particle_radius ** 3) * z * g

    @pytest.fixture(autouse=True, params=["periodic", "non_periodic"])
    def add_particle(self, openmm_system, gravitational_potential, particle_radius, request):
        openmm_system.addParticle(mass=1.0)
        gravitational_potential.add_particle(index=0, radius=particle_radius)
        for potential in gravitational_potential.yield_potentials():
            openmm_system.addForce(potential)
        # Add a nonbonded force with a periodic cutoff to the system so that it uses periodic boundary conditions.
        # This tests a potential issue on the CUDA/OpenCL platforms: https://github.com/openmm/openmm/issues/4611
        if request.param == "periodic":
            nonbonded_force = NonbondedForce()
            nonbonded_force.addParticle(0.0, 1.0, 0.0)
            nonbonded_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
            nonbonded_force.setCutoffDistance(5.0)
            openmm_system.addForce(nonbonded_force)
            pytest.xfail("on the CUDA and OpenCL platforms, the gravitational potential will not be correctly computed "
                         "if the system uses periodic boundary conditions because the z positions are changed to lie "
                         "outside the range from -L/2 to L/2 (see https://github.com/openmm/openmm/issues/4611)")

    @pytest.fixture(params=[("Reference", 1.0e-7), ("CPU", 1.0e-7), ("CUDA", 1.0e-3), ("OpenCL", 1.0e-3)])
    def openmm_context_rel(self, openmm_system, openmm_dummy_integrator, request):
        try:
            platform = Platform.getPlatformByName(request.param[0])
        except OpenMMException:
            pytest.skip("Platform {} not available.".format(request.param[0]))
        return Context(openmm_system, openmm_dummy_integrator, platform), request.param[1]

    def test_gravitational_potentials(self, openmm_context_rel, particle_density, particle_radius,
                                      test_z_positions, gravitational_acceleration, water_density):
        openmm_context, rel = openmm_context_rel

        openmm_grav_potentials = np.zeros(len(test_z_positions))

        for index, dir_z_position in enumerate(test_z_positions):
            position = [0.0, 0.0, dir_z_position]
            openmm_context.setPositions([position])
            openmm_state = openmm_context.getState(getEnergy=True)
            openmm_grav_potentials[index] = openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        expected_numpy_grav_potentials = (self.gravitational_force_exp(
            particle_density, water_density, particle_radius, test_z_positions, gravitational_acceleration)
                                          * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        assert openmm_grav_potentials == pytest.approx(expected_numpy_grav_potentials, rel=rel, abs=1.0e-13)


if __name__ == '__main__':
    pytest.main([__file__])
