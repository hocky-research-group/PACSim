import numpy as np
import openmm.app
from openmm import unit


class UpdateReporter(object):
    
    def __init__(self, filename: str, report_interval: int, simulation: openmm.app.Simulation, variant: str, 
                 start_value: unit.Quantity, end_value: unit.Quantity, total_number_steps: int, 
                 append_file: bool = False, continous=False):
        """Constructor of the UpdateReporter class."""
        
        if not filename.endswith(".csv"):
            raise ValueError("The file must have the .csv extension.")
        if not report_interval > 0:
            raise ValueError("The report interval must be greater than zero.")
        assert simulation.topology.getNumChains() == 1
        assert simulation.topology.getNumResidues() == 1
        assert simulation.topology.getNumAtoms() == simulation.system.getNumParticles()
   
        self._append_file = append_file
        self._out = open(filename, 'a' if self._append_file else 'w')
        self._ramp = continous
        if self._ramp:
            self._report_interval = 1
        else: 
            self._report_interval = report_interval
        self._variant = variant.split("-")[1]
        self._start_value = start_value
        self._end_value = end_value
        self._total_number_steps = total_number_steps
    
    def describeNextReport(self, simulation: openmm.app.Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        """Get information about the next report this reporter will generate.

        This method is called by OpenMM once this reporter is added to the list of reporters of a simulation.

        :param simulation:
            The simulation to generate a report for.
        :type simulation: openmm.app.Simulation

        :returns:
            (Number of steps until next report,
            Whether the next report requires positions (False),
            Whether the next report requires velocities (False),
            Whether the next report requires forces (False),
            Whether the next report requires energies (False),
            Whether positions should be wrapped to lie in a single periodic box (False))
        :rtype: tuple[int, bool, bool, bool, bool, bool]
        """
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return steps, False, False, False, False, False
    
    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Generate a report by storing information about the updated variant parameter in a .csv file.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        step = simulation.currentStep
        
        current_value = simulation.context.getParameter(self._variant)
        current_value += (self._end_value - self._start_value) * self._report_interval/self._total_number_steps
        
        simulation.context.setParameter(self._variant, current_value)
        
        #this is just to check that parameter update is happening properly
        ## remove when fully implemented
        current_value1 = simulation.context.getParameter(self._variant)

        print(step, current_value, current_value1, file=self._out)

    def __del__(self) -> None:
        """Destructor of the UpdateReporter class."""
        try:
            self._out.close()
        except AttributeError:
            # If another error occured, the '_out' attribute might not exist.
            pass

    
    
    