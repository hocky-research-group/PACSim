from abc import abstractmethod, ABC
import math
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
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
    :type update_interval: int
    :param final_update_step:
        The final step at which the value of the global parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param start_value:
        The start value of the global parameter.
        OpenMM does not store the units of global parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: unit.Quantity
    :param global_parameter_name:
        The name of the global parameter to be updated.
        This must be one of the global parameters passed into any of the OpenMM Force objects.
    :type global_parameter_name: str
    :param print_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is printed
        to the output .csv file.
        The value must be greater than zero.
    :type print_interval: int
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool
   

    :raises ValueError:
        If the filename does not end with the .csv extension.
        If the update_interval is not greater than zero.
        If the final_update_step is not greater than or equal to the update_interval.
        If the global_parameter_name is not in the simulation context.        
        If the print_interval is not greater than zero.
    """

    def __init__(self, filename: str, update_interval: int, final_update_step, global_parameter_name: str,
                 start_value: unit.Quantity, print_interval: int, simulation: openmm.app.Simulation,
                 append_file: bool = False):
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
            print(f"timestep,{self._global_parameter_name}", file=self._file, flush=True)
        try: 
            self._start_value = start_value.value_in_unit_system(unit.md_unit_system)
        except AttributeError:
            self._start_value = start_value
        # Check if the start value of the global parameter matches the value in the OpenMM simulation.
        # If the file is being appended to, this check is not necessary since the simulation was resumed in which case
        # the start value is not necessarily the same as the value in the OpenMM simulation.
        if not print_interval > 0:
            raise ValueError("The print frequency must be greater than zero.")
        self._print_interval = print_interval
        '''if (not append_file
                and abs(self._start_value - simulation.context.getParameters()[self._global_parameter_name]) > 1.0e-12):
            warnings.warn("The start value of the global parameter does not match the value in the OpenMM simulation.")
            simulation.context.setParameter(self._global_parameter_name, self._start_value)'''
        if not append_file:
            print(f"0,{self._start_value}", file=self._file)

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
        Update the value of a global parameter during the simulation.

        This function is called by OpenMM when the reporter should generate a report.

        The implementation of this method in the inheriting class should compute the new value of the global value.
        Then, one should call the set_and_print method to update the value of the global parameter in the OpenMM
        simulation context and print the value in the output .csv file.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        raise NotImplementedError

    def set_and_print(self, simulation: openmm.app.Simulation, new_value: float) -> None:
        """
        Update the value of the global parameter in the OpenMM simulation context and print the new parameter value in
        the ouput .csv file.

        :param simulation:
            The OpenMM simulation.
        :type simulation: openmm.app.Simulation
        :param new_value:
            The new value of the global parameter.
        :type new_value: float
        """
        step = simulation.currentStep
        simulation.context.setParameter(self._global_parameter_name, new_value)
        if step % self._print_interval == 0:
            print(f"{step},{new_value}", file=self._file)

    def __del__(self) -> None:
        """Destructor of the UpdateReporter class."""
        try:
            self._file.close()
        except AttributeError:
            # If another error occurred, the '_file' attribute might not exist.
            pass


class RampUpdateReporter(UpdateReporterAbstract):
    """
    This class sets up a reporter to linearly change the value of a force-related global parameter in a ramp over the
    course of an OpenMM simulation.

    Both the start and end values of the global parameter are specified on initialization.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param update_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is updated.
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
    :param print_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is printed
        to the output .csv file.
        The value must be greater than zero.
    :type print_interval: int
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension (via the abstract base class).
        If the update_interval is not greater than zero (via the abstract base class).
        If the print_interval is not greater than zero (via the abstract base class).
        If the final_update_step is not greater than or equal to the update_interval (via the abstract base class).
        If the global_parameter_name is not in the simulation context (via the abstract base class).
        If the start and end values have incompatible units.
    """

    def __init__(self, filename: str, update_interval: int, final_update_step: int, global_parameter_name: str,
                 start_value: unit.Quantity, end_value: unit.Quantity, print_interval: int,
                 simulation: openmm.app.Simulation, append_file: bool = False):
        """Constructor of the LinearMonotonicUpdateReporter class."""
        super().__init__(filename=filename, update_interval=update_interval, final_update_step=final_update_step,
                         global_parameter_name=global_parameter_name, start_value=start_value,
                         print_interval=print_interval, simulation=simulation, append_file=append_file)
        try:
            if not start_value.unit.is_compatible(end_value.unit):
                raise ValueError(f"The start and end values have incompatible units.")
            self._end_value = end_value.value_in_unit_system(unit.md_unit_system)

        except AttributeError:
            self._end_value = end_value

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Linearly change the value of a global parameter during the simulation.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        old_value = simulation.context.getParameter(self._global_parameter_name)
        new_value = old_value + (self._end_value - self._start_value) * self._update_interval / self._final_update_step
        self.set_and_print(simulation, new_value)


class TriangleUpdateReporter(UpdateReporterAbstract):
    """
    This class sets up a reporter to change the value of a force-related global parameter following a triangular wave
    over the course of an OpenMM simulation.

    Both the start and end values of the global parameter during a single increasing or decreasing ramp of the
    triangular wave are specified on initialization. If the end value is greater than the start value, the global
    parameter value increases until the switch step, then decreases back to the start value. Otherwise, the global
    parameter value decreases until the switch step, then increases back to the start value. This is repeated until the
    final update step is reached.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param update_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is updated.
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
    :param switch_step:
        The number of steps after which this reporter switches from increasing to decreasing (or decreasing to
        increasing) the value of the global parameter.
        The value must be a multiple of the update_interval, and less than or equal to the final_update_step.
    :type switch_step: int
    :param print_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is printed
        to the output .csv file.
        The value must be greater than zero.
    :type print_interval: int
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension (via the abstract base class).
        If the update_interval is not greater than zero (via the abstract base class).
        If the print_interval is not greater than zero (via the abstract base class).
        If the final_update_step is not greater than or equal to the update_interval (via the abstract base class).
        If the global_parameter_name is not in the simulation context (via the abstract base class).
        If the start and end values have incompatible units.
        If the switch step is not a multiple of the update frequency.
        If the switch step is not less than or equal to the final update step.
    """

    def __init__(self, filename: str, update_interval: int, final_update_step: int, global_parameter_name: str,
                 start_value: unit.Quantity, end_value: unit.Quantity, switch_step: int, print_interval: int,
                 simulation: openmm.app.Simulation, append_file: bool = False):
        super().__init__(filename=filename, update_interval=update_interval, final_update_step=final_update_step,
                         global_parameter_name=global_parameter_name, start_value=start_value,
                         print_interval=print_interval, simulation=simulation, append_file=append_file)
        try: 
            if not start_value.unit.is_compatible(end_value.unit):
                raise ValueError(f"The start and end values have incompatible units.")
            self._end_value = end_value.value_in_unit_system(unit.md_unit_system)
        except AttributeError:
            self._end_value = end_value
        if not final_update_step >= switch_step >= update_interval:
            raise ValueError("The switch step must be greater than or equal to the update frequency,"
                             "and less than or equal to the final update step.")
        if not switch_step % update_interval == 0:
            raise ValueError("The switch step must be a multiple of the update frequency.")
        self._switch_step = switch_step

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Change the value of a global parameter during the simulation according to a triangular wave.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        current_step = simulation.currentStep
        old_value = simulation.context.getParameter(self._global_parameter_name)
        assert current_step - self._update_interval >= 0
        last_update_remainder = (current_step - self._update_interval) % (2.0 * self._switch_step)
        if last_update_remainder < self._switch_step:
            new_value = old_value + (self._end_value - self._start_value) * self._update_interval / self._switch_step
        else:
            new_value = old_value - (self._end_value - self._start_value) * self._update_interval / self._switch_step
        self.set_and_print(simulation, new_value)


class SquaredSinusoidalUpdateReporter(UpdateReporterAbstract):
    """
    This class sets up a reporter to change the value of a force-related global parameter following a squared sinusoidal
    wave over the course of an OpenMM simulation.

    Both the start and end values of the global parameter during a single increasing or decreasing part of the squared
    sinusoidal wave are specified on initialization. If the end value is greater than the start value, the global
    parameter value increases until the switch step, then decreases back to the start value. Otherwise, the global
     parameter value decreases until the switch step, then increases back to the start value. This is repeated until the
     final update step is reached.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param update_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is updated.
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
    :param switch_step:
        The number of steps after which this reporter switches from increasing to decreasing (or decreasing to
        increasing) the value of the global parameter.
        The value must be a multiple of the update_interval, and less than or equal to the final_update_step.
    :type switch_step: int
    :param print_interval:
        The interval (in time steps) at which the value of the global parameter in the OpenMM simulation is printed
        to the output .csv file.
        The value must be greater than zero.
    :type print_interval: int
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool

    :raises ValueError:
        If the filename does not end with the .csv extension (via the abstract base class).
        If the update_interval is not greater than zero (via the abstract base class).
        If the print_interval is not greater than zero (via the abstract base class).
        If the final_update_step is not greater than or equal to the update_interval (via the abstract base class).
        If the global_parameter_name is not in the simulation context (via the abstract base class).
        If the start and end values have incompatible units.
        If the switch step is not a multiple of the update frequency.
        If the switch step is not less than or equal to the final update step.
    """

    def __init__(self, filename: str, update_interval: int, final_update_step: int, global_parameter_name: str,
                 start_value: unit.Quantity, end_value: unit.Quantity, switch_step: int, print_interval: int,
                 simulation: openmm.app.Simulation, append_file: bool = False):
        super().__init__(filename=filename, update_interval=update_interval, final_update_step=final_update_step,
                         global_parameter_name=global_parameter_name, start_value=start_value,
                         print_interval=print_interval, simulation=simulation, append_file=append_file)
        try:
            if not start_value.unit.is_compatible(end_value.unit):
                raise ValueError(f"The start value and amplitude have incompatible units.")
            end_value_float = end_value.value_in_unit_system(unit.md_unit_system)
        except AttributeError:
            end_value_float = end_value
        self._amplitude = end_value_float - self._start_value
        if not final_update_step >= switch_step >= update_interval:
            raise ValueError("The switch step must be greater than or equal to the update frequency,"
                             "and less than or equal to the final update step.")
        if not switch_step % update_interval == 0:
            raise ValueError("The switch step must be a multiple of the update frequency.")
        self._period = math.pi / (2.0 * switch_step)

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Change the value of a global parameter during the simulation according to a squared sinusoidal wave.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        step = simulation.currentStep
        current_value = self._amplitude * (math.sin(self._period * step) ** 2) + self._start_value
        self.set_and_print(simulation, current_value)