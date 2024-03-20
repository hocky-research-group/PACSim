from openmm import Context, LangevinIntegrator, Platform, System, unit
import pytest
from colloids.colloid_forces import ColloidForces


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
        return -50.0 * (unit.milli * unit.volt)

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
    def maximum_surface_separation(self, radius_one, radius_two, debye_length):
        return 2.0 * max(radius_one, radius_two) + 21.0 * debye_length

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
    def colloid_forces(self, brush_density, brush_length, debye_length, temperature, dielectric_constant):
        return ColloidForces(brush_density=brush_density, brush_length=brush_length, debye_length=debye_length,
                             temperature=temperature, dielectric_constant=dielectric_constant)

    @pytest.fixture(scope="class")
    def openmm_platform(self):
        return Platform.getPlatformByName("Reference")

    @pytest.fixture(scope="class")
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)


class TestExceptions(object):
    def test_exceptions_brush_density(self):
        with pytest.raises(TypeError):
            ColloidForces(brush_density=0.09 / unit.nanometer)
        with pytest.raises(ValueError):
            ColloidForces(brush_density=-0.09 / unit.nanometer ** 2)

    def test_exceptions_brush_length(self):
        with pytest.raises(TypeError):
            ColloidForces(brush_length=10.0 / unit.nanometer)
        with pytest.raises(ValueError):
            ColloidForces(brush_length=-10.0 * unit.nanometer)

    def test_exceptions_debye_length(self):
        with pytest.raises(TypeError):
            ColloidForces(debye_length=5.0 / unit.nanometer)
        with pytest.raises(ValueError):
            ColloidForces(debye_length=-5.0 * unit.nanometer)

    def test_exceptions_temperature(self):
        with pytest.raises(TypeError):
            ColloidForces(temperature=298.0 / unit.kelvin)
        with pytest.raises(ValueError):
            ColloidForces(temperature=-298.0 * unit.kelvin)

    def test_exceptions_dielectric_constant(self):
        with pytest.raises(ValueError):
            ColloidForces(dielectric_constant=-80.0)

    def test_exceptions_no_particles_added(self):
        forces = ColloidForces()
        with pytest.raises(RuntimeError):
            _ = forces.steric_force
        with pytest.raises(RuntimeError):
            _ = forces.electrostatic_force


class TestForcesForPair(TestParameters):
    @pytest.fixture(autouse=True, scope="class")
    def add_two_particles(self, openmm_system, colloid_forces,
                          radius_one, radius_two, surface_potential_one, surface_potential_two):
        openmm_system.addParticle(mass=1.0)
        colloid_forces.add_particle(radius=radius_one, surface_potential=surface_potential_one)
        openmm_system.addParticle(mass=1.0)
        colloid_forces.add_particle(radius=radius_two, surface_potential=surface_potential_two)
        openmm_system.addForce(colloid_forces.steric_force)
        openmm_system.addForce(colloid_forces.electrostatic_force)

    @pytest.fixture(scope="class")
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    def test_force(self, openmm_context):
        openmm_context.setPositions([[600.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        openmm_state = openmm_context.getState(getEnergy=True)
        print(openmm_state.getPotentialEnergy())