class Integrators():

    def BrownianIntegrator(temperature, frictionCoeff, stepSize):
        return openmm.BrownianIntegrator(temperature, frictionCoeff, stepSize)

    def LangevinIntegrator(temperature, frictionCoeff, stepSize):
        return openmm.LangevinIntegrator(temperature, frictionCoeff, stepSize)

    def LangevinMiddleIntegrator(temperature, frictionCoeff, stepSize):
        return openmm.LangevinMiddleIntegrator(temperature, frictionCoeff, stepSize)

    def NoseHooverIntegrator(temperature, collisionFrequency, stepSize, chainLength, numMTS, numYoshidaSuzuki):
        return openmm.NoseHooverIntegrator(temperature, collisionFrequency, stepSize, chainLength, numMTS, numYoshidaSuzuki)

    def VariableLangevinIntegrator(temperature, frictionCoeff, errorTol):
        return openmm.VariableLangevinIntegrator( temperature, frictionCoeff, errorTol)
    
    def VariableVerletIntegrator(errorTol):
        return openmm.VariableVerletIntegrator(errorTol)

    def VerletIntegrator(stepSize):
        return openmm.VerletIntegrator(stepSize)
        
