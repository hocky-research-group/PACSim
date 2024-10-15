from openmm import Context, LangevinIntegrator, OpenMMException, Platform, System, unit, Vec3
import pytest
from colloids import ColloidPotentialsParameters, SubstrateWall


class TestSubstrateWallParameters(object):

    @pytest.fixture
    def radius(self):
        return 105.0 * (unit.nano * unit.meter)
    
    @pytest.fixture
    def surface_potential(self):
        return -50.0 * (unit.milli * unit.volt)
    
    @pytest.fixture
    def wall_distance(self):
        return 1000.0 * (unit.nano * unit.meter)


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
    def wall_charge(self):
        return -47.0 * (unit.milli * unit.volt)


    @pytest.fixture
    def openmm_system(self):
        system = System()
        system.setDefaultPeriodicBoxVectors(Vec3(1000.0, 0.0, 0.0),
                                            Vec3(0.0, 1000.0, 0.0),
                                            Vec3(0.0, 0.0, 1000.0))
        return system

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)

    @pytest.fixture
    def substrate_wall_potential(self, colloid_potentials_parameters, wall_distance, wall_charge):
        return SubstrateWall(colloid_potentials_parameters, wall_distance, wall_charge, False)


class TestSubstrateWallExceptions(TestSubstrateWallParameters):
    def test_exception_radius(self, radius, surface_potential, substrate_wall_potential):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            substrate_wall_potential.add_particle(index=0, radius=radius / ((unit.nano * unit.meter) ** 2), surface_potential=surface_potential)
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            substrate_wall_potential.add_particle(index=0, radius=-radius, surface_potential=surface_potential)
    
    def test_exception_surface_potential(self, radius, surface_potential, substrate_wall_potential):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            substrate_wall_potential.add_particle(index=0, radius=radius, surface_potential=surface_potential/ (unit.milli * unit.volt) ** 2)
      
    def test_exception_no_particles_added(self, substrate_wall_potential):
        with pytest.raises(RuntimeError):
            for _ in substrate_wall_potential.yield_potentials():
                pass

    def test_exception_add_particle_after_yield_potentials(self, radius, surface_potential, substrate_wall_potential):
        substrate_wall_potential.add_particle(index=0, radius=radius, surface_potential=surface_potential)
        for _ in substrate_wall_potential.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            substrate_wall_potential.add_particle(index=0, radius=radius, surface_potential=surface_potential)


class TestSubstrateWallEnergies(TestSubstrateWallParameters):
    @pytest.fixture(autouse=True)
    def add_particle(self, openmm_system, substrate_wall_potential, radius, surface_potential):
        openmm_system.addParticle(mass=1.0)
        substrate_wall_potential.add_particle(index=0, radius=radius, surface_potential=surface_potential)

        for potential in substrate_wall_potential.yield_potentials():
            openmm_system.addForce(potential)

    # This function cannot be moved to TestSubstrateParameters class because add_particle fixture should be called before
    # the context is created (see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html).
    @pytest.fixture(params=[("Reference", 1.0e-7), ("CPU", 1.0e-7), ("CUDA", 1.0e-3), ("OpenCL", 1.0e-3)])
    def openmm_context_rel(self, openmm_system, openmm_dummy_integrator, request):
        try:
            platform = Platform.getPlatformByName(request.param[0])
        except OpenMMException:
            pytest.skip("Platform {} not available.".format(request.param[0]))
        return Context(openmm_system, openmm_dummy_integrator, platform), request.param[1]


    @pytest.mark.parametrize("surface_separation,expected",
                             [   # Test just above h=0.
                                 (0.1 * (unit.nano * unit.meter) , 1.974393318268673e-41 *unit.kilojoule_per_mole),
                                 # Test at h=2L.
                                 (20.0 * (unit.nano * unit.meter) ,3.689280145599771e-43 *unit.kilojoule_per_mole),
                                 # Test slightly below h=2L where steric potential is not zero.
                                 (19.9 * (unit.nano * unit.meter) , 3.76380854827503e-43 *unit.kilojoule_per_mole),
                                 # Test slightly above h=2L where steric potential is strictly zero.
                                 (20.1 * (unit.nano * unit.meter) , 3.616227504173863e-43 *unit.kilojoule_per_mole),
                                 # Test at h=3L.
                                 (30.0 * (unit.nano * unit.meter) , 4.992897734439567e-44 *unit.kilojoule_per_mole),
                                 # Test at h=20*debye_length, where electrostatic potential should not yet be cutoff.
                                 (100.0 * (unit.nano * unit.meter) , 4.1517378577335804e-50 *unit.kilojoule_per_mole),
                             ])


    def test_potential(self, openmm_context_rel, radius, surface_separation, expected):
        openmm_context, rel = openmm_context_rel
        openmm_context.setPositions([[0.0, 0.0, surface_separation+radius]])
        openmm_state = openmm_context.getState(getEnergy=True)
        assert (openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                == pytest.approx(expected.value_in_unit(unit.kilojoule_per_mole), rel=rel, abs=1.0e-13))

if __name__ == '__main__':
    pytest.main([__file__])
