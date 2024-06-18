from dataclasses import field
from openmm import unit

class Integrators():

    """
    Module to define the integrators for molecular dynamics simulations in OpenMM.

    For information about the integrators, see http://docs.openmm.org/latest/userguide/theory/04_integrators.html.

    :param temperature:
        The temperature of the heat bath coupled to the system (for integration at constant temperature).
        The unit of the temperature must be compatible with Kelvin and the value must be greater than zero.
        Defaults to 298.0  * unit.kelvin.
    :type temperature: unit.Quantity

    :param frictionCoeff:
        The friction coefficient which couples the system to the heat bath (for integration at constant temperature).
        The unit of the frictionCoeff must be compatible with inverse picoseconds and the value must be greater than zero.
        Defaults to 0.01 / unit.picosecond.
    :type frictionCoeff: unit.Quantity

    :param stepSize:
        The timestep with which to integrate the system.
        The unit of the stepSize must be compatible with picoseconds and the value must be greater than zero.
        Defaults to 0.05 * unit.picosecond.
    :type stepSize: unit.Quantity

    :param collisionFrequency:
        The frequency of the system's interaction with the heat bath (for the Nose Hoover integrator).
        The unit of the collisionFrequency must be compatible with inverse picoseconds and the value must be greater than zero.
        Defaults to 0.01 / unit.picosecond.
    :type collisionFrequency: unit.Quantity

    :param chainLength: 
        The number of beads in the Nose-Hoover chain (for the Nose Hoover integrator).
        Defaults to 3.
    :type chainLength: int

    :param numMTS: 
        The number of steps in the multiple-timestep chain propagation algorithm (for the Nose Hoover integrator).
        Defaults to 3.
    :type numMTS: int

    :param numYoshidaSuzuki: 
        The number of terms in the Yoshida-Suzuki multi-timestep decomposition used in the chain propagation algorithm.
        The value of numYoshidaSuzuki must be 1, 3, 5, or 7. 
        Defaults to 7. 
    :type numYoshidaSuzuki: int

    :param errorTol:
        The error tolerance for integrators that use variable timestep.
        The value of errorTol must be greater than 0.
        Defaults to 0.001.
    :type errorTol: float

    """

    def BrownianIntegrator(temperature: unit.Quantity = field(default_factory=lambda: 298.0 * unit.kelvin), 
                            frictionCoeff: unit.Quantity = field(default_factory=lambda: 0.01 / unit.picosecond), 
                            stepSize:unit.Quantity = field(default_factory=lambda: 0.05 * unit.picosecond)):
        """
        Returns the OpenMM Brownian integrator. This integrator uses Brownian dynamics to simulate a system 
        at constant temperature in contact with a heat bath.
        """
        return openmm.BrownianIntegrator(temperature, frictionCoeff, stepSize)

    def LangevinIntegrator(temperature: unit.Quantity = field(default_factory=lambda: 298.0 * unit.kelvin), 
                            frictionCoeff: unit.Quantity = field(default_factory=lambda: 0.01 / unit.picosecond), 
                            stepSize:unit.Quantity = field(default_factory=lambda: 0.05 * unit.picosecond)):
        """
        Returns the OpenMM Langevin integrator. This integrator uses Langevin dynamics to simulate a system 
        at constant temperature in contact with a heat bath.
        """
        return openmm.LangevinIntegrator(temperature, frictionCoeff, stepSize)

    def LangevinMiddleIntegrator(temperature: unit.Quantity = field(default_factory=lambda: 298.0 * unit.kelvin), 
                                frictionCoeff: unit.Quantity = field(default_factory=lambda: 0.01 / unit.picosecond), 
                                stepSize:unit.Quantity = field(default_factory=lambda: 0.05 * unit.picosecond)):
        """
        Returns the OpenMM Langevin middle integrator. This integrator uses Langevin dynamics, like the Langevin 
        integrator, but with LFMiddle discretization (half-step velocities) to more accurately sample the configuration 
        space at the expense of less accuracy for kinetic properties.
        """
        return openmm.LangevinMiddleIntegrator(temperature, frictionCoeff, stepSize)

    def NoseHooverIntegrator(temperature: unit.Quantity = field(default_factory=lambda: 298.0 * unit.kelvin), 
                            collisionFrequency: unit.Quantity = field(default_factory=lambda: 0.01 / unit.picosecond),
                            stepSize: unit.Quantity = field(default_factory=lambda: 0.05 * unit.picosecond),
                            chainLength: int = 3,
                            numMTS: int = 3,
                            numYoshidaSuzuki: int = 7):
        """
        Returns the OpenMM Nose Hoover integrator. This integrator simulates a system at constant temperature in contact 
        with a heat bath using a chain of one or more Nose Hoover thermostats.
        """
        return openmm.NoseHooverIntegrator(temperature, collisionFrequency, stepSize, chainLength, numMTS, numYoshidaSuzuki)

    def VariableLangevinIntegrator(temperature: unit.Quantity = field(default_factory=lambda: 298.0 * unit.kelvin), 
                                frictionCoeff: unit.Quantity = field(default_factory=lambda: 0.01 / unit.picosecond), 
                                errorTol: float = 0.001):
        """
        Returns the OpenMM variable Langevin integrator. This integrator uses the Langevin integration method, but 
        instead of a fixed timestep, the timestep is continuously adjusted such that the error in the integration remains
        below a specified error tolerance.
        """
        return openmm.VariableLangevinIntegrator(temperature, frictionCoeff, errorTol)
    
    def VariableVerletIntegrator(errorTol: float = 0.001):
        """
        Returns the OpenMM variable Verlet integrator. This integrator uses the Verlet integration method, but 
        instead of a fixed timestep, the timestep is continuously adjusted such that the error in the integration remains
        below a specified error tolerance.
        """
        return openmm.VariableVerletIntegrator(errorTol)

    def VerletIntegrator(stepSize:unit.Quantity = field(default_factory=lambda: 0.05 * unit.picosecond)):
        """
        Returns the OpenMM Verlet integrator. This integrator implements the leap-frog Verlet integration method.
        """
        return openmm.VerletIntegrator(stepSize)
        
