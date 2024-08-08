import openmm.app
from openmm import unit


class UpdateReporter(object):
    """
    Reporter for an OpenMM simulation of colloids that linearly changes the value of a force-related global parameter
    over the course of the simulation.

    The start value is determined by the current value of the global parameter in the OpenMM simulation. The end value
    is specified on initialization.

    This class creates a .csv file that stores the current simulation step and current value of the global parameter
    being updated.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param update_interval:
        The interval (in time steps) at which to update the value of the variant parameter in the OpenMM simulation.
        The value must be greater than zero.
    :type update_interval: int
    :param final_update_step:
        The final step at which the value of the variant parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param global_parameter_name:
        The name of the global parameter to be updated.
        This must be one of the parameters passed into the simulation context.
        The string must be formatted such that it begins with the '-' character followed by the name of the parameter 
        exactly as passed into the openmm Force object. 
    :type global_parameter_name: str
    :param end_value:
        The end value of the variant parameter.
        The value and unit must agree with those of the parameter being updated. For instance, if the variant parameter
        is debye_length, the value must be greater than 0 and the unit must be compatible with nanometers.
    :type end_value: unit.Quantity
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the global parameter to be updated.
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
        If the global_parameter_name does not start with the '-' character.
        If the global_parameter_name is not in the simulation context.
        If the unit of the end_value is not compatible with the unit of the global parameter.
    """
    
    def __init__(self, filename: str, update_interval: int, final_update_step, global_parameter_name: str,
                 end_value: unit.Quantity, simulation: openmm.app.Simulation, append_file: bool = False):
        """Constructor of the UpdateReporter class."""
        if not filename.endswith(".csv"):
            raise ValueError("The file must have the .csv extension.")
        if not update_interval > 0:
            raise ValueError("The update frequency must be greater than zero.")
        if not final_update_step >= update_interval:
            raise ValueError("The final update step must be greater than or equal to the update frequency.")
        if not global_parameter_name.startswith("-"):
            raise ValueError("The variant parameter string is improperly formatted. Must begin with the '-' character.")
        self._update_interval = update_interval
        self._final_update_step = final_update_step
        self._global_parameter_name = global_parameter_name.split("-")[1]
        if self._global_parameter_name not in simulation.context.getParameters():
            raise ValueError("The variant parameter is not in the simulation context.")
        self._start_value = simulation.context.getParameters()[self._global_parameter_name]
        if not end_value.unit.is_compatible(self._start_value.unit):
            raise ValueError("The unit of the end value must be compatible with the unit of the global parameter.")
        self._end_value = end_value
        self._file = open(filename, "a" if append_file else "x")
        print(f"Timestep,{self._global_parameter_name}", file=self._file)

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
        steps = self._update_interval - simulation.currentStep % self._update_interval
        return steps, False, False, False, False, False

    # noinspection PyUnusedLocal
    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Generate a report by changing the value of the global parameter and by storing the updated value in the .csv
        file.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        step = simulation.currentStep
        old_value = simulation.context.getParameter(self._global_parameter_name)
        new_value = old_value + (self._end_value - self._start_value) * self._update_interval / self._final_update_step
        simulation.context.setParameter(self._global_parameter_name, new_value)
        print(f"{step},{new_value}", file=self._file)

    def __del__(self) -> None:
        """Destructor of the OutputReporter class."""
        try:
            self._file.close()
        except AttributeError:
            # If another error occurred, the '_file' attribute might not exist.
            pass
