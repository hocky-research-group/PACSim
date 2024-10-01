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
    def wall_distance(self):
        return 1000.0 * (unit.nano * unit.meter)

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
    def substrate_wall_potential(self, colloid_potentials_parameters, wall_distance, wall_charge):
        return SubstrateWall(colloid_potentials_parameters, "wall", wall_distance, wall_charge, False)


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
    
    
    def test_exception_radius_too_large(self, substrate_wall_potential, surface_potential):
        # This is fine for a wall distance of 1000 nm.
        substrate_wall_potential.add_particle(index=0, radius=236.0 * (unit.nano * unit.meter), surface_potential=surface_potential)
        # This is not fine for a wall distance of 1000 nm.
        with pytest.raises(ValueError):
            substrate_wall_potential.add_particle(index=1, radius=237.0 * (unit.nano * unit.meter), surface_potential=surface_potential)
      
    def test_exception_no_particles_added(self, substrate_wall_potential):
        with pytest.raises(RuntimeError):
            for _ in substrate_wall_potential.yield_potentials():
                pass

    def test_exception_add_particle_after_yield_potentials(self, radius, surface_potential, substrate_wall_potential):
        substrate_wall_potential.add_particle(index=0, radius=radius, surface_potential=surface_potential)
        for _ in substrate_wall_potential.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            substrate_wall_potential.add_particle(index=1, radius=radius, surface_potential=surface_potential)


    def test_exception_wall_distance(self, colloid_potentials_parameters, wall_charge):
        # Test exception on wrong unit 
        with pytest.raises(TypeError):
            SubstrateWall(colloid_potentials_parameters, "wall", wall_distance=1000.0 * ((unit.nano * unit.meter) ** 2), 
                          wall_charge=wall_charge, use_log= False)
                
        # Test exception wall distance negative.
        with pytest.raises(ValueError):
            SubstrateWall(colloid_potentials_parameters, "wall", wall_distance=-1000.0 * (unit.nano * unit.meter), 
                          wall_charge=wall_charge, use_log= False)


if __name__ == '__main__':
    pytest.main([__file__])
