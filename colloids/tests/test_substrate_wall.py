from openmm import Context, LangevinIntegrator, NonbondedForce, OpenMMException, Platform, System, unit, Vec3
import pytest
from colloids import ColloidPotentialsParameters, SubstrateWall
import numpy as np


class TestSubstrateWallParameters(object):

    @pytest.fixture
    def radius(self):
        return 105.0 * (unit.nano * unit.meter)
    
    @pytest.fixture
    def surface_potential(self):
        return 50.0 * (unit.milli * unit.volt)
    
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
    def num_test_values(self):
        return 1000

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
    def test_exception_radius(self, radius, surface_potential, substrate_wall_potential, wall_distance):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            substrate_wall_potential.add_particle(index=0, radius=radius / ((unit.nano * unit.meter) ** 2), surface_potential=surface_potential, wall_distance=wall_distance)
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            substrate_wall_potential.add_particle(index=0, radius=-radius, surface_potential=surface_potential, wall_distance=wall_distance)

    def test_exception_wall_distance(self, radius, surface_potential, substrate_wall_potential, wall_distance):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            substrate_wall_potential.add_particle(index=0, radius=radius, surface_potential=surface_potential, wall_distance=wall_distance/ ((unit.nano * unit.meter) ** 2))
        # Test exception on negative wall distance.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            substrate_wall_potential.add_particle(index=0, radius=-radius, surface_potential=surface_potential, wall_distance=-wall_distance)
    
    def test_exception_surface_potential(self, radius, surface_potential, substrate_wall_potential, wall_distance):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
            substrate_wall_potential.add_particle(index=0, radius=radius, surface_potential=surface_potential/ (unit.milli * unit.volt) ** 2, wall_distance=wall_distance)
    
    
    def test_exception_radius_too_large(self, substrate_wall_potential, surface_potential, wall_distance):
        # This is fine for a wall distance of 1000 nm.
        substrate_wall_potential.add_particle(index=0, radius=236.0 * (unit.nano * unit.meter), surface_potential=surface_potential, wall_distance=wall_distance)
        # This is not fine for a wall distance of 1000 nm.
        with pytest.raises(ValueError):
            substrate_wall_potential.add_particle(index=1, radius=237.0 * (unit.nano * unit.meter), surface_potential=surface_potential, wall_distance=wall_distance)
      
    def test_exception_no_particles_added(self, substrate_wall_potential):
        with pytest.raises(RuntimeError):
            for _ in substrate_wall_potential.yield_potentials():
                pass

    def test_exception_add_particle_after_yield_potentials(self, radius, surface_potential, substrate_wall_potential, wall_distance):
        substrate_wall_potential.add_particle(index=0, radius=radius, surface_potential=surface_potential, wall_distance=wall_distance)
        for _ in substrate_wall_potential.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            substrate_wall_potential.add_particle(index=1, radius=radius, surface_potential=surface_potential, wall_distance=wall_distance)


if __name__ == '__main__':
    pytest.main([__file__])
