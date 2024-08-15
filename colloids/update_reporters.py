import math
from abc import abstractmethod, ABC
from typing import Any
import warnings
import openmm.app
from openmm import unit


class UpdateReporterAbstract(ABC):
    """
    Abstract class for reporters for an OpenMM simulation of colloids that change the value of a global parameter over 
    the course of the simulation.

    The inheriting class must implement the report method. The report method can be used to specify the way the 
    global parameter is updated.

    This class creates a .csv file that stores the current simulation step and current value of the parameter
    being updated.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param update_interval:
        The interval (in time steps) at which to update the value of the global parameter in the OpenMM simulation.
        The value must be greater than zero.
    :type update_interval: int
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param global_parameter_name:
        The name of the global parameter to be updated.
        This must be one of the global parameters passed into any of the OpenMM Force objects.
    :type global_parameter_name: str
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param append_file:
        If True, open an existing csv file to append to. If False, try to create a new file, and throw an error if the
        file already exists.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension.
        If the update_interval is not greater than zero.
        If the final_update_step is not greater than or equal to the update_interval.
        If the global_parameter_name is not in the simulation context.
    """
    
    def __init__(self, filename: str, update_interval: int, final_update_step, global_parameter_name: str, 
                 simulation: openmm.app.Simulation, append_file: bool = False):
                 
        """Constructor of the UpdateReporterAbstract class."""
        if not filename.endswith(".csv"):
            raise ValueError("The file must have the .csv extension.")
        if not update_interval > 0:
            raise ValueError("The update frequency must be greater than zero.")
        if not final_update_step >= update_interval:
            raise ValueError("The final update step must be greater than or equal to the update frequency.")
        self._update_interval = update_interval
        self._final_update_step = final_update_step
        self._global_parameter_name = global_parameter_name
        if self._global_parameter_name not in simulation.context.getParameters():
            raise ValueError(f"The global parameter {self._global_parameter_name} is not in the simulation context.")
        self._file = open(filename, "a" if append_file else "w")
        if not append_file:
            print(f"timestep,{self._global_parameter_name}", file=self._file)
    
    # noinspection PyPep8Naming
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
        if simulation.currentStep >= self._final_update_step:
            # 0 signals to not interrupt the simulation again.
            return 0, False, False, False, False, False
        steps = self._update_interval - simulation.currentStep % self._update_interval
        return steps, False, False, False, False, False

    # noinspection PyUnusedLocal
    @abstractmethod
    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
               
        """
        Update the value of a global parameter during the simulation and store the parameter value in the output
        .csv file.

        One should set and print in the inherting class.

        :param args:
            Parameters of the report method as positional arguments.
        :type args: Any
        :param kwargs:
            Parameters of the report method as keyword arguments.
        :type kwargs: Any

        """
        raise NotImplementedError

    def set_and_print(self, simulation, new_value) -> None:
        """
        Pass the updated value of the global parameter into the OpenMM simulation context to update it and print the 
        parameter value in the ouput .csv file.

        :param simulation:
        :param new_value:
        :return:
        """
        step = simulation.currentStep
        simulation.context.setParameter(self._global_parameter_name, new_value)
        print(f"{step},{new_value}", file=self._file)

    def __del__(self) -> None:
        """Destructor of the UpdateReporter class."""
        try:
            self._file.close()
        except AttributeError:
            # If another error occurred, the '_file' attribute might not exist.
            pass


class LinearMonotonicUpdateReporter(UpdateReporterAbstract):

    '''
    This class sets up a reporter to linearly, monotonically change the value of a force-related global parameter 
    over the course of an OpenMM simulation.

    The start value is determined by the current value of the global parameter in the OpenMM simulation. The end value
    is specified on initialization.

    :param update_interval:
        The interval (in time steps) at which to update the value of the global parameter in the OpenMM simulation.
        The value must be greater than zero.
    :type update_interval: int
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param global_parameter_name:
        The name of the global parameter to be updated.
        This must be one of the global parameters passed into any of the OpenMM Force objects.
    :type global_parameter_name: str
    :param start_value:
        The start value of the global parameter.
        OpenMM does not store the units of global parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: unit.Quantity
    :param end_value:
        The end value of the global parameter.
        OpenMM does not store the units of global parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type end_value: unit.Quantity

    :raises ValueError:
        If the filename does not end with the .csv extension.
        If the update_interval is not greater than zero.
        If the final_update_step is not greater than or equal to the update_interval.
        If the global_parameter_name is not in the simulation context.
    '''

    def __init__(self, filename: str, update_interval: int, final_update_step, global_parameter_name: str,
                 start_value: unit.Quantity, end_value: unit.Quantity, simulation: openmm.app.Simulation,
                 append_file: bool = False):
        """Constructor of the LinearMonotonicUpdateReporter class."""

        
        super().__init__(filename=filename, update_interval=update_interval, final_update_step=final_update_step,
                         global_parameter_name=global_parameter_name, simulation=simulation, append_file=append_file)
        if not start_value.unit.is_compatible(end_value.unit):
            raise ValueError(f"The start and end values have incompatible units.")
        self._start_value = start_value.value_in_unit_system(unit.md_unit_system)
        # Check if the start value of the global parameter matches the value in the OpenMM simulation.
        # If the file is being appended to, this check is not necessary since the simulation was resumed in which case
        # the start value is not necessarily the same as the value in the OpenMM simulation.
        if (not append_file
                and abs(self._start_value - simulation.context.getParameters()[self._global_parameter_name]) > 1.0e-12):
            warnings.warn("The start value of the global parameter does not match the value in the OpenMM simulation.")
        self._end_value = end_value.value_in_unit_system(unit.md_unit_system)
    
    
    def report(self, simulation: openmm.app.Simulation, state) -> None:
        """
        Linearly, monotonically update the value of a global parameter in an OpenMM simulation.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        """
        old_value = simulation.context.getParameter(self._global_parameter_name)
        new_value = old_value + (self._end_value - self._start_value) * self._update_interval / self._final_update_step
        self.set_and_print(simulation, new_value)

class LinearUnimodalUpdateReporter(UpdateReporterAbstract):

    '''
    This class sets up a reporter to linearly, unimodally change the value of a force-related global parameter 
    over the course of an OpenMM simulation.

    The start value is determined by the current value of the global parameter in the OpenMM simulation. The end value
    is specified on initialization. If the end value is greater than the start value, the global parameter value increases
    until the switch step, then decreases back to the start value. Otherwise, the global parameter value decreases until 
    the switch step, then increases back to the start value. 


    :param update_interval:
        The interval (in time steps) at which to update the value of the global parameter in the OpenMM simulation.
        The value must be greater than zero.
    :type update_interval: int
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param switch_step: 
        The step at which to switch from increasing to decreasing (or decreasing to increasing) the value of the 
        global parameter.
        The value must be greater than or equal to the update_interval, and less than or equal to the final_update_step.
    :type switch_step: int
    :param global_parameter_name:
        The name of the global parameter to be updated.
        This must be one of the global parameters passed into any of the OpenMM Force objects.
    :type global_parameter_name: str
    :param start_value:
        The start value of the global parameter.
        OpenMM does not store the units of global parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: unit.Quantity
    :param end_value:
        The max or min value of the global parameter. This value will be reached at the switch step.
        OpenMM does not store the units of global parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type end_value: unit.Quantity

    :raises ValueError:
        If the filename does not end with the .csv extension.
        If the update_interval is not greater than zero.
        If the final_update_step is not greater than or equal to the update_interval.
        If the switch_step is not greater than or equal to the update_interval and less than or equal to the 
        final_update_step.
        If the global_parameter_name is not in the simulation context.
    '''

    def __init__(self, filename: str, update_interval: int, final_update_step, switch_step, global_parameter_name: str,
                 start_value: unit.Quantity, end_value: unit.Quantity, simulation: openmm.app.Simulation,
                 append_file: bool = False):

        
        super().__init__(filename=filename, update_interval=update_interval, final_update_step=final_update_step,
                         global_parameter_name=global_parameter_name, simulation=simulation, append_file=append_file)
        if not final_update_step >= switch_step >= update_interval:
            raise ValueError("The switch  step must be greater than or equal to the update frequency,"
                             "and less than or equal to the final update step.")
        self._switch_step = switch_step
        if not start_value.unit.is_compatible(end_value.unit):
            raise ValueError(f"The start and end values have incompatible units.")
        self._start_value = start_value.value_in_unit_system(unit.md_unit_system)
        # Check if the start value of the global parameter matches the value in the OpenMM simulation.
        # If the file is being appended to, this check is not necessary since the simulation was resumed in which case
        # the start value is not necessarily the same as the value in the OpenMM simulation.
        if (not append_file
                and abs(self._start_value - simulation.context.getParameters()[self._global_parameter_name]) > 1.0e-12):
            warnings.warn("The start value of the global parameter does not match the value in the OpenMM simulation.")
        self._end_value = end_value.value_in_unit_system(unit.md_unit_system)
    

    def report(self, simulation: openmm.app.Simulation, state) -> None:
        """
         Linearly, unimodally update the value of a global parameter in an OpenMM simulation.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        """
        step = simulation.currentStep
        old_value = simulation.context.getParameter(self._global_parameter_name)
        if step < self._switch_step:
            new_value = old_value + (self._end_value - self._start_value) * self._update_interval / self._switch_step
        else: 
            new_value = old_value - (self._end_value - self._start_value) * self._update_interval / (self._final_update_step - self._switch_step)
        self.set_and_print(simulation, new_value)
    

class SinusoidalUpdateReporter(UpdateReporterAbstract):

    '''
    This class sets up a reporter to sinusoidally change the value of a force-related global parameter over the course 
    of an OpenMM simulation. 
    
    
    If the global parameter can take on negative values, the value will oscillate between the start
    value and the negative of the start value. Otherwise, the value will oscillate between the start value and 0.The start 
    value is determined by the current value of the global parameter in the OpenMM simulation. 


    :param update_interval:
        The interval (in time steps) at which to update the value of the global parameter in the OpenMM simulation.
        The value must be greater than zero.
    :type update_interval: int
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param global_parameter_name:
        The name of the global parameter to be updated.
        This must be one of the global parameters passed into any of the OpenMM Force objects.
    :type global_parameter_name: str
    :param start_value:
        The start value of the global parameter.
        OpenMM does not store the units of global parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: unit.Quantity
    :param amplitude: 
        The amplitude of the sinusoidal function. This will scale the values of the global parameter relative to the 
        specified start value.
    :type amplitude: float
    :param oscillation_steps: 
        The number of steps in which to complete a cycle of oscillation. 
        The value must be greater than or equal to the update_interval.
    :type oscillation_steps: int
    :param phase_shift:
        The horitzontal shift of the sinusoidal function. If phase_shift = 0, the global parameter will be equal to 0 at the 
        start of the simulation. If phase_shift = pi/2, the global parameter will be equal to the amplitude * start_value + vertical_shift
        at the start of the simulation.
    :type phase_shift: float
    :param vertical_shift
        The vertical shift by which to offset the sinusoidal function.
    :type vertical_shift: float



    :raises ValueError:
        If the filename does not end with the .csv extension.
        If the update_interval is not greater than zero.
        If the final_update_step is not greater than or equal to the update_interval.
        If  oscillation_steps is not greater than or equal to the update_interval.
        If the global_parameter_name is not in the simulation context.
    '''

    def __init__(self, filename: str, update_interval: int, final_update_step, global_parameter_name: str,
                 start_value: unit.Quantity, amplitude: float, oscillation_steps: int, phase_shift: float,
                 vertical_shift: float, simulation: openmm.app.Simulation, append_file: bool = False):

        
        super().__init__(filename=filename, update_interval=update_interval, final_update_step=final_update_step,
                         global_parameter_name=global_parameter_name, simulation=simulation, append_file=append_file)
        self._start_value = start_value.value_in_unit_system(unit.md_unit_system)
        # Check if the start value of the global parameter matches the value in the OpenMM simulation.
        # If the file is being appended to, this check is not necessary since the simulation was resumed in which case
        # the start value is not necessarily the same as the value in the OpenMM simulation.
        self._amplitude = amplitude
        if not oscillation_steps  >= update_interval:
            raise ValueError("The oscillation  steps must be greater than or equal to the update frequency.")
        self._oscillation_steps = oscillation_steps
        self._phase_shift = phase_shift
        self._vertical_shift = vertical_shift
        if (not append_file
                and abs(self._start_value - simulation.context.getParameters()[self._global_parameter_name]) > 1.0e-12):
            warnings.warn("The start value of the global parameter does not match the value in the OpenMM simulation.")

    

    def report(self, simulation: openmm.app.Simulation, state) -> None:
        """
        Sinusoidally update the value of a global parameter in an OpenMM simulation.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        """

        step = simulation.currentStep
        period = math.pi / self._oscillation_steps
        current_value = self._amplitude * self._start_value * math.sin(period*step + self._phase_shift)**2 + self._vertical_shift
        self.set_and_print(simulation, current_value)



    