from typing import Optional
import numpy as np
import numpy.typing as npt
import openmm.app
from openmm import unit


class UpdateReporter(object):
    
    def __init__(self, filename: str, report_interval: int, simulation: openmm.app.Simulation, variant: str, 
                 start_value: unit.Quantity, end_value: unit.Quantity, continous=False):
        """Constructor of the UpdateReporter class."""
        
        if not report_interval > 0:
            raise ValueError("The report interval must be greater than zero.")
        assert simulation.topology.getNumChains() == 1
        assert simulation.topology.getNumResidues() == 1
        assert simulation.topology.getNumAtoms() == simulation.system.getNumParticles()
   
        self._filename = filename
        self._report_interval = report_interval
        self._variant = variant
        self._start_value = start_value
        self._end_value = end_value

        #self._file = gsd.hoomd.open(name=filename, mode="r+" if self._append_file else "w")
    
    def update_global_parameter(self, openmm_simulation: app.Simulation, parameter: str, modifier_function, continuous=False, frequency=Optional[int]):
        current_value = openmm_simulation.context.getParameter(parameter)
        if continuous==False:
        
            for i in range(frequency):
                runsteps = int(self._parameters.run_steps/frequency)
                openmm_simulation.step(runsteps)
                print(f"{parameter} current value:", current_value)
                current_value = modifier_function(current_value)
                openmm_simulation.context.setParameter(parameter, current_value)
        else: 
            #continuous: update every step
            for i in range(self._parameters.run_steps):
                openmm_simulation.step(1)
                print(f"{parameter} current value:", current_value)
                current_value = modifier_function(current_value)
                openmm_simulation.context.setParameter(parameter, current_value)

