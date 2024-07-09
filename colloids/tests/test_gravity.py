from openmm import Context, LangevinIntegrator, Platform, System, unit, Vec3
import pytest
from colloids import gravity
import numpy as np

'''In this file, you want to test if the implementation of gravitational force is working properly.
To do so, you will create a dummy system with a single particle, obtain its gravitational force from 
your openmm gravity function,and compare it to the force expected using a simple function in numpy that 
calculates "mgh"
You are going to compute the gravitational force as the particle moves along the z-axis, so you are calculating 
force as a function of z position.
There is no molecular dynamics happening, hence the dummy integrator that is necessary to initialize the system 
but won't actually do anything'''


class TestGravityParameters(object):
    '''Define the parameters you will need for the tests in this file. Use the decorator @pytest.fixture'''
   
   # Define density of a particle whose gravitational force will be measured. not mass
   ## if you look at gravity.py the function does not take mass as an argument
    
    @pytest.fixture
    def particle_density(self):
        return 1.05 * (unit.gram * unit.centimeter**3)

   # Define radius of a particle whose gravitational force will be measured
    @pytest.fixture
    def particle_radius(self):
        return 105.0 * (unit.nano * unit.meter)

   # Define the gravitational constant
    @pytest.fixture
    def g(self):
        return 9.8 * unit.meter / unit.second**2

    # Define density of water
    @pytest.fixture
    def water_density(self):
        return 0.998 * (unit.gram * unit.centimeter**3)

   # Define things you need for setting up the simulation: box length, openmm system, platform, dummy integrator
    @pytest.fixture
    def box_length(self):
        return 1000.0 * unit.nanometer 
    
    @pytest.fixture
    def openmm_system(self, box_length):
        system = System()
        system.setDefaultPeriodicBoxVectors(Vec3(box_length, 0.0, 0.0),
                                            Vec3(0.0, box_length, 0.0),
                                            Vec3(0.0, 0.0, box_length))
        return system
    
    @pytest.fixture
    def openmm_platform(self):
        return Platform.getPlatformByName("Reference")

    @pytest.fixture
    def openmm_dummy_integrator(self):
        return LangevinIntegrator(0.0, 0.0, 0.0)


    # Define a function that returns your Gravity function from gravity.py
    def gravitational_potential(self, gravitational_constant, water_density):
        return gravity(gravitational_constant, water_density)
    
    # Define an array of z positions to use to calculate the gravitational potential
    @pytest.fixture
    def z_num_test_values(self):
        return 1000

    @pytest.fixture
    def test_z_positions(self, box_length, z_num_test_values):
        # noinspection PyUnresolvedReferences
        return [np.linspace(-box_length.value_in_unit(unit.nanometer) / 2.0 + 200.0, 
                            box_length.value_in_unit(unit.nanometer) / 2.0 - 200.0, num=z_num_test_values)
                            for _ in box_length]

    # Using @staticmethod decorator, create a numpy function to calculate the expected gravitational force
    #given a particle's effective mass, z coordinate, and gravitational constant
    ##look up what a staticmethod decorator is to understand why it is being used here.

    """staticmethod decorater doesn't require self argument because it tells the interpretor that the method (function)
    is static and should be called on the class, not on an instance of the class. In other words, it can't modify
    the class state
    """
    @staticmethod
    def gravitational_force_exp(particle_density, water_density, particle_radius, z, g):
        return (np.absolute(water_density - particle_density))* 4/3 * np.pi * particle_radius**3

class TestGravityExceptions(TestGravityParameters):
    '''Test to make sure all of the variables being defined for use in gravity.py are properly initialized'''

    # test that radius has the right unit
    # test that radius >0
    def test_exception_radius(self, radius, gravitational_force_exp):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
           gravitational_force_exp.add_particle(index=0, radius=radius / ((unit.nano * unit.meter) ** 2))
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            gravitational_force_exp.add_particle(index=0, radius=-radius)
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            gravitational_force_exp.add_particle(index=0, radius=0)
    
    # test that particle density has the right unit
    # test that particle density >0
    def test_exception_particle_density(self, particle_density, gravitational_force_exp):
        # Test exception on wrong unit.
        with pytest.raises(TypeError):
           gravitational_force_exp.add_particle(index=0, particle_density=particle_density / (unit.gram / unit.centimeter** 2))
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            gravitational_force_exp.add_particle(index=0, particle_density=-particle_density)
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            gravitational_force_exp.add_particle(index=0, particle_density=0)

    # test to make sure a particle is added to the system
    def test_exception_no_particles_added(self, expected_gravitational_potential):
        with pytest.raises(RuntimeError):
            for _ in expected_gravitational_potential.yield_potentials():
                pass
        with pytest.raises(RuntimeError):
            for _ in expected_gravitational_potential.yield_potentials():
                pass
    
    #test to make sure particles are added before getting the potential
    def test_exception_add_particle_after_yield_potentials(self, particle_mass, expected_gravitational_potential):
        expected_gravitational_potential.add_particle(index=0, particle_mass=particle_mass)
        for _ in expected_gravitational_potential.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
            expected_gravitational_potential.add_particle(index=1, particle_mass=particle_mass)
        expected_gravitational_potential.add_particle(index=0, particle_mass=particle_mass)
        for _ in expected_gravitational_potential.yield_potentials():
            pass
        with pytest.raises(RuntimeError):
           expected_gravitational_potential.add_particle(index=1, particle_mass=particle_mass)
    
    # test that gravitational constant has the right units
    # note you don't need to test if g>0. technically someone can define negative acceleration for a system, it's valid
    ##if you look at run_parameters.py, there is no exception for negative g
    
    def test_exception_g(self, g, expected_gravitational_potential):
        # Test exception on wrong unit of g.
        with pytest.raises(TypeError):
            expected_gravitational_potential.add_particle(index=0, g=g / (unit.nanometer))


    # test that water density has the right unit
    # test that water density >0
    def test_exception_water_density(self, water_density, gravitational_force_exp):
            # Test exception on wrong unit.
            with pytest.raises(TypeError):
                gravitational_force_exp.add_particle(index=0, water_density=water_density / (unit.gram / unit.centimeter** 2))
            # Test exception on density > 0.
            with pytest.raises(ValueError):
                # noinspection PyTypeChecker
                gravitational_force_exp.add_particle(index=0, water_density=-water_density)
            with pytest.raises(ValueError):
                # noinspection PyTypeChecker
                gravitational_force_exp.add_particle(index=0, water_density=0)
  

class TestGravity(TestGravityParameters):
    '''Test to compare the force of gravity calculated in openmm with that in the numpy function'''

    #autouse=True automatically executes the fixture before the test fxn; no need to initialize openmm_system and gravitational_potential in the test fxn
    @pytest.fixture(autouse=True)
    def add_particle(self, openmm_system, gravitational_potential, radius):
        openmm_system.addParticle(mass=1.0)
        gravitational_potential.add_particle(index=0, radius=radius)
        for potential in gravitational_potential.yield_potentials():
            openmm_system.addForce(potential)

    @pytest.fixture
    def openmm_context(self, openmm_system, openmm_dummy_integrator, openmm_platform):
        return Context(openmm_system, openmm_dummy_integrator, openmm_platform)

    
    ## see https://realpython.com/pytest-python-testing/#what-makes-pytest-so-useful to read about the mark.parametrize decorator
    # and understand what to pass in 

    """""we can use @pytest.mark.parameterize to parameterize a single test definition, and the create variants of the 
    test with the parameters I specify"""
    @pytest.mark.parametrize("direction, expected_function",
                             [
                                 2, TestGravityParameters.gravitational_force_exp
                             ])
    
    def test_gravitational_potentials(self, openmm_context, test_z_positions, expected_function):
        # create an empty array to fill with gravitational potential values for each z position
        openmm_grav_potentials = np.empty(len(test_z_positions))

        # for each test position, set that as the particle's z position and get the openmm state/energy . save the energy
        for index, dir_z_position in enumerate(test_z_positions):
            position = [0.0, 0.0, 0.0]
            position[2] = [0.0, 0.0, dir_z_position]
            openmm_context.setPositions([position])
            openmm_state = openmm_context.getState(getEnergy=True)
            openmm_grav_potentials[index] = openmm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        # get the gravitational potentials for each z position from the numpy function
        expected_numpy_grav_potentials = expected_function(test_z_positions)
        
        #use an assert statement to compare the two arrays of gravitational potentials and make sure they're the same
        assert openmm_grav_potentials == pytest.approx(expected_numpy_grav_potentials, rel=1.0e-7, abs=1.0e-13)

        ## see the jupyter notebook on github for reference about how we compare the openmm and numpy function values


if __name__ == '__main__':
    pytest.main([__file__])
