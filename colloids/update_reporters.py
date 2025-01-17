from abc import abstractmethod, ABC
import math
import warnings
import openmm.app
from openmm import unit


class UpdateReporterAbstract(ABC):
    """
    Abstract class for reporters for an OpenMM simulation of colloids that change the value of a global or per-particle parameter over 
    the course of the simulation.

    The inheriting class must implement the report method. The report method can be used to specify the way the 
    parameter is updated.

    This class creates a .csv file that stores the current simulation step and current value of the parameter
    being updated.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param update_interval:
        The interval (in time steps) at which the value of the parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
    :type update_interval: int
    :param final_update_step:
        The final step at which the value of the parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param start_value:
        The start value of the parameter.
        OpenMM does not store the units of parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: unit.Quantity
    :param parameter_name:
        The name of the parameter to be updated.
        This must be one of the global or per-partice parameters passed into any of the OpenMM Force objects.
    :type parameter_name: str
    :param print_interval:
        The interval (in time steps) at which the value of the parameter in the OpenMM simulation is printed
        to the output .csv file.
        The value must be greater than zero.
    :type print_interval: int
    :param system:
        The OpenMM system containing the forces for the simulation to which the reporter will be added.
    :type system: openmm.System
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The context of this OpenMM simulation must contain the parameter to be updated.
    :type simulation: openmm.app.Simulation
    :param particle_type_index:
        The index of the particle type for which to update the parameter value, if using per-particle parameters.
        Defaults to None (for global parameters).
    :type particle_type_index: int
    :param append_file:
        If True, open an existing csv file to append to. If False, create a new file possibly overwriting an already
        existing file.
        Defaults to False.
    :type append_file: bool
   

    :raises ValueError:
        If the filename does not end with the .csv extension.
        If the update_interval is not greater than zero.
        If the final_update_step is not greater than or equal to the update_interval.
        If the parameter_name is not in the simulation context.        
        If the print_interval is not greater than zero.
    """

    def __init__(self, filename: str, update_interval: int, final_update_step, parameter_name: str,
                 start_value: unit.Quantity, print_interval: int, particle_type_index: list, 
                 system:openmm.System, simulation: openmm.app.Simulation,append_file: bool = False):
        """Constructor of the UpdateReporterAbstract class."""
        if not filename.endswith(".csv"):
            raise ValueError("The file must have the .csv extension.")
        if not update_interval > 0:
            raise ValueError("The update frequency must be greater than zero.")
        if not final_update_step >= update_interval:
            raise ValueError("The final update step must be greater than or equal to the update frequency.")
        self._update_interval = update_interval
        self._final_update_step = final_update_step
        self._parameter_name = parameter_name
        #check if parameter is global
        if self._parameter_name not in simulation.context.getParameters():
            #if not, check if parameter is per-particle (and if so, particle index must be specified)
            if not particle_type_index:
                raise ValueError("The particle indices for which per-partice parameter is being updated must be specified.")
            else:
                self._particle_type_index = particle_type_index
                for force in system.getForces():
                    if self._parameter_name in force.getParticleParameters(self._particle_type_index):
                        print(force)
                        #pass
                    #else:
                    #raise ValueError(f"The parameter {self._parameter_name} is not in the simulation context.")
                self._parameter_type = "per-particle"
                
        else:
            self._parameter_type = "global"
            if particle_type_index is not None:
                raise ValueError("Particle type index must not be specified if the parameter to be updated is a global parameter.")
        self._file = open(filename, "a" if append_file else "w")
        if not append_file:
            print(f"timestep,{self._parameter_name}", file=self._file, flush=True)
        self._start_value = start_value.value_in_unit_system(unit.md_unit_system)
        # Check if the start value of the  parameter matches the value in the OpenMM simulation.
        # If the file is being appended to, this check is not necessary since the simulation was resumed in which case
        # the start value is not necessarily the same as the value in the OpenMM simulation.
        if not print_interval > 0:
            raise ValueError("The print frequency must be greater than zero.")
        self._print_interval = print_interval
        if self._parameter_type == "global":
            if (not append_file
                and abs(self._start_value - simulation.context.getParameters()[self._parameter_name]) > 1.0e-12):
                warnings.warn("The start value of the parameter does not match the value in the OpenMM simulation.")
                simulation.context.setParameter(self._parameter_name, self._start_value)
        else:
            for force in system.getForces():
                if self._parameter_name in force.getParticleParameters(self._particle_type_index):
                    force.setParticleParameter(self._particle_type_index, self._start_value)
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
        Update the value of a global or per-particle parameter during the simulation.

        This function is called by OpenMM when the reporter should generate a report.

        The implementation of this method in the inheriting class should compute the new value of the parameter.
        Then, one should call the set_and_print method to update the value of the parameter in the OpenMM
        simulation context and print the value in the output .csv file.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        raise NotImplementedError

    def set_and_print(self, system: openmm.System, simulation: openmm.app.Simulation, new_value: float, parameter_type: str) -> None:
        """
        Update the value of the global or per-particle parameter in the OpenMM simulation context and print the new parameter value in
        the ouput .csv file.

        :param system: 
            The OpenMM system.
        :type system: openmm.System
        :param simulation:
            The OpenMM simulation.
        :type simulation: openmm.app.Simulation
        :param new_value:
            The new value of the parameter.
        :type new_value: float
        :param parameter_type:
            Indicates whether the parameter to be updated is a Global or PerParticle parameter. This changes the way
            the values are passed into the simulation context.
        :type parameter_type: str
        """
        step = simulation.currentStep
        if parameter_type == "global":
            simulation.context.setParameter(self._parameter_name, new_value)
        elif parameter_type == "per-particle":
            for force in system.getForces():
                if self._parameter_name in force.getParticleParameters(self._particle_type_index):
                    force.setParticleParameter(self._particle_type_index, new_value)
                    force.updateParametersInContext(self, simulation.context)
        else:
            raise ValueError(f"Parameter type {parameter_type} is not supported.")
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
    This class sets up a reporter to linearly change the value of a force-related global or per-particle parameter in a ramp over the
    course of an OpenMM simulation.

    Both the start and end values of the parameter are specified on initialization.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param update_interval:
        The interval (in time steps) at which the value of the parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
    :type update_interval: int
    :param final_update_step:
        The final step at which the value of the parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param parameter_name:
        The name of the parameter to be updated.
        This must be one of the parameters passed into any of the OpenMM Force objects.
    :type parameter_name: str
    :param start_value:
        The start value of the parameter.
        OpenMM does not store the units of parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: unit.Quantity
    :param end_value:
        The end value of the parameter.
        OpenMM does not store the units of parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type end_value: unit.Quantity
    :param print_interval:
        The interval (in time steps) at which the value of the parameter in the OpenMM simulation is printed
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
        If the parameter_name is not in the simulation context (via the abstract base class).
        If the start and end values have incompatible units.
    """

    def __init__(self, filename: str, update_interval: int, final_update_step: int, parameter_name: str,
                 start_value: unit.Quantity, end_value: unit.Quantity, print_interval: int, particle_type_index: int,
                 system: openmm.System, simulation: openmm.app.Simulation, append_file: bool = False):
        """Constructor of the LinearMonotonicUpdateReporter class."""
        super().__init__(filename=filename, update_interval=update_interval, final_update_step=final_update_step,
                         parameter_name=parameter_name, start_value=start_value,
                         print_interval=print_interval,  particle_type_index = particle_type_index,
                         system=system, simulation=simulation, append_file=append_file)
        if not start_value.unit.is_compatible(end_value.unit):
            raise ValueError(f"The start and end values have incompatible units.")
        self._end_value = end_value.value_in_unit_system(unit.md_unit_system)
        self._system = system
        if particle_type_index is not None:
            self._parameter_type = "per-particle"
        else: 
            self._parameter_type = "global"

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Linearly change the value of a global or per-particle parameter during the simulation.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        old_value = simulation.context.getParameter(self._parameter_name)
        new_value = old_value + (self._end_value - self._start_value) * self._update_interval / self._final_update_step
        self.set_and_print(self._system, simulation, new_value, self._parameter_type)


class TriangleUpdateReporter(UpdateReporterAbstract):
    """
    This class sets up a reporter to change the value of a force-related global or per-particle parameter following a triangular wave
    over the course of an OpenMM simulation.

    Both the start and end values of the parameter during a single increasing or decreasing ramp of the
    triangular wave are specified on initialization. If the end value is greater than the start value, the 
    parameter value increases until the switch step, then decreases back to the start value. Otherwise, the 
    parameter value decreases until the switch step, then increases back to the start value. This is repeated until the
    final update step is reached.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param update_interval:
        The interval (in time steps) at which the value of the parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
    :type update_interval: int
    :param final_update_step:
        The final step at which the value of the parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param parameter_name:
        The name of the parameter to be updated.
        This must be one of the global or per-particle parameters passed into any of the OpenMM Force objects.
    :type parameter_name: str
    :param start_value:
        The start value of the parameter.
        OpenMM does not store the units of parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: unit.Quantity
    :param end_value:
        The end value of the parameter.
        OpenMM does not store the units of parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type end_value: unit.Quantity
    :param switch_step:
        The number of steps after which this reporter switches from increasing to decreasing (or decreasing to
        increasing) the value of the parameter.
        The value must be a multiple of the update_interval, and less than or equal to the final_update_step.
    :type switch_step: int
    :param print_interval:
        The interval (in time steps) at which the value of the parameter in the OpenMM simulation is printed
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
        If the parameter_name is not in the simulation context (via the abstract base class).
        If the start and end values have incompatible units.
        If the switch step is not a multiple of the update frequency.
        If the switch step is not less than or equal to the final update step.
    """

    def __init__(self, filename: str, update_interval: int, final_update_step: int, parameter_name: str,
                 start_value: unit.Quantity, end_value: unit.Quantity, switch_step: int, print_interval: int,
                  particle_type_index: int, system: openmm.System, simulation: openmm.app.Simulation, append_file: bool = False):
        super().__init__(filename=filename, update_interval=update_interval, final_update_step=final_update_step,
                         parameter_name=parameter_name, start_value=start_value,
                         print_interval=print_interval, particle_type_index = particle_type_index, 
                         system=system, simulation=simulation, append_file=append_file)
        if not start_value.unit.is_compatible(end_value.unit):
            raise ValueError(f"The start and end values have incompatible units.")
        self._end_value = end_value.value_in_unit_system(unit.md_unit_system)
        if not final_update_step >= switch_step >= update_interval:
            raise ValueError("The switch step must be greater than or equal to the update frequency,"
                             "and less than or equal to the final update step.")
        if not switch_step % update_interval == 0:
            raise ValueError("The switch step must be a multiple of the update frequency.")
        self._switch_step = switch_step
        self._system = system
        if particle_type_index is not None:
            self._parameter_type = "per-particle"
        else: 
            self._parameter_type = "global"

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Change the value of a global or per-particle parameter during the simulation according to a triangular wave.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        current_step = simulation.currentStep
        old_value = simulation.context.getParameter(self._parameter_name)
        assert current_step - self._update_interval >= 0
        last_update_remainder = (current_step - self._update_interval) % (2.0 * self._switch_step)
        if last_update_remainder < self._switch_step:
            new_value = old_value + (self._end_value - self._start_value) * self._update_interval / self._switch_step
        else:
            new_value = old_value - (self._end_value - self._start_value) * self._update_interval / self._switch_step
        self.set_and_print(self._system, simulation, new_value, self._parameter_type)


class SquaredSinusoidalUpdateReporter(UpdateReporterAbstract):
    """
    This class sets up a reporter to change the value of a force-related global or per-particle parameter following a squared sinusoidal
    wave over the course of an OpenMM simulation.

    Both the start and end values of the parameter during a single increasing or decreasing part of the squared
    sinusoidal wave are specified on initialization. If the end value is greater than the start value, the
    parameter value increases until the switch step, then decreases back to the start value. Otherwise, the 
     parameter value decreases until the switch step, then increases back to the start value. This is repeated until the
     final update step is reached.

    :param filename:
        The name of the file to write to.
        The filename must end with the .csv extension.
    :type filename: str
    :param update_interval:
        The interval (in time steps) at which the value of the parameter in the OpenMM simulation is updated.
        The value must be greater than zero.
    :type update_interval: int
    :param final_update_step:
        The final step at which the value of the parameter will be updated.
        The value must be greater than or equal to the update_interval.
    :type final_update_step: int
    :param parameter_name:
        The name of the  parameter to be updated.
        This must be one of the global or per-particle parameters passed into any of the OpenMM Force objects.
    :type parameter_name: str
    :param start_value:
        The start value of the parameter.
        OpenMM does not store the units of parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type start_value: unit.Quantity
    :param end_value:
        The end value of the parameter.
        OpenMM does not store the units of parameters, so the user must make sure to pass in a quantity with a
        sensible unit here. This quantity will only be converted to the unit system of OpenMM.
    :type end_value: unit.Quantity
    :param switch_step:
        The number of steps after which this reporter switches from increasing to decreasing (or decreasing to
        increasing) the value of the parameter.
        The value must be a multiple of the update_interval, and less than or equal to the final_update_step.
    :type switch_step: int
    :param print_interval:
        The interval (in time steps) at which the value of the parameter in the OpenMM simulation is printed
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
        If the parameter_name is not in the simulation context (via the abstract base class).
        If the start and end values have incompatible units.
        If the switch step is not a multiple of the update frequency.
        If the switch step is not less than or equal to the final update step.
    """

    def __init__(self, filename: str, update_interval: int, final_update_step: int, parameter_name: str,
                 start_value: unit.Quantity, end_value: unit.Quantity, switch_step: int, print_interval: int,
                 particle_type_index: int, system: openmm.System, simulation: openmm.app.Simulation, append_file: bool = False):
        super().__init__(filename=filename, update_interval=update_interval, final_update_step=final_update_step,
                         parameter_name=parameter_name, start_value=start_value,
                         print_interval=print_interval, particle_type_index = particle_type_index,
                         system=system, simulation=simulation, append_file=append_file)
        if not start_value.unit.is_compatible(end_value.unit):
            raise ValueError(f"The start value and amplitude have incompatible units.")
        end_value_float = end_value.value_in_unit_system(unit.md_unit_system)
        self._amplitude = end_value_float - self._start_value
        if not final_update_step >= switch_step >= update_interval:
            raise ValueError("The switch step must be greater than or equal to the update frequency,"
                             "and less than or equal to the final update step.")
        if not switch_step % update_interval == 0:
            raise ValueError("The switch step must be a multiple of the update frequency.")
        self._period = math.pi / (2.0 * switch_step)
        self._system = system
        if particle_type_index is not None:
            self._parameter_type = "per-particle"
        else: 
            self._parameter_type = "global"


    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Change the value of a global or per-particle parameter during the simulation according to a squared sinusoidal wave.

        This function is called by OpenMM when the reporter should generate a report.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        print(self._start_value)
        step = simulation.currentStep
        current_value = self._amplitude * (math.sin(self._period * step) ** 2) + self._start_value
        self.set_and_print(self._system, simulation, current_value, self._parameter_type)
