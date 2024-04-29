from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Optional
from openmm import unit
import yaml
from colloids.helper_functions import read_xyz_file


class Quantity(yaml.YAMLObject):
    """
    Wrapper class for an OpenMM quantity that allows for a simple serialization in a yaml file.

    This class defines the application-specific yaml tag !Quantity by following the description in
    https://pyyaml.org/wiki/PyYAMLDocumentation.

    Although the !!python/object tag would in principle allow to serialize an OpenMM quantity in a yaml file, this tag
    would require specifying an enormous amount of (mostly private) attributes. This class circumvents this problem by
    defining a custom yaml tag that only stores the value and a string representation of the unit of an OpenMM quantity.
    The string representation is obtained by calling the get_name method of the unit of the OpenMM quantity.

    :param quantity:
        The OpenMM quantity to be wrapped.
    :type quantity: unit.Quantity

    :ivar value:
        The value of the wrapped quantity.
    :vartype value: float
    :ivar unit:
        The string representation of the unit of the wrapped quantity obtained from the get_name method.
    :vartype unit: str

    :cvar yaml_tag:
        The yaml tag for this class.
    :vartype yaml_tag: str
    """

    yaml_tag = u'!Quantity'

    def __init__(self, quantity: unit.Quantity) -> None:
        self.value = quantity.value_in_unit(quantity.unit)
        self.unit = quantity.unit.get_name()

    def to_openmm_quantity(self) -> unit.Quantity:
        """
        Convert the wrapped quantity to an openmm quantity.

        :return:
            The openmm quantity.
        :rtype: unit.Quantity
        """
        return unit.Quantity(self.value, self._openmm_unit_from_string(self.unit))

    @staticmethod
    def _openmm_unit_from_string(unit_string: str) -> unit.Unit:
        """Convert a string representation of a composite openmm unit (like meter/second) to an openmm unit."""

        # Remove all whitespaces from the string representation of the unit.
        string_wo_whitespaces = "".join(unit_string.split())
        # Composite units that only contain a denominator start with a slash in openmm
        if string_wo_whitespaces.startswith("/"):
            # If more than one unit is in the denominator of the composite, the units are enclosed in parentheses.
            # It appears the composite unit always ends after the closing bracket.
            if string_wo_whitespaces[1] == "(":
                bracket_index = string_wo_whitespaces.index(")")
                assert bracket_index == len(string_wo_whitespaces) - 1
                return Quantity._openmm_unit_from_string(string_wo_whitespaces[2:bracket_index]) ** (-1)
            return Quantity._openmm_unit_from_string(string_wo_whitespaces[1:]) ** (-1)
        # If the composite unit does not start with a slash, it starts with one unit and ends with a multiplication (*),
        # division (/), or power (**).
        stop_index = 0
        while stop_index < len(string_wo_whitespaces) and string_wo_whitespaces[stop_index] not in ["*", "/"]:
            stop_index += 1
        # The first unit in the composite unit may contain a SI prefix that must be found explicitly because units like
        # millivolt are not directly recognized by openmm.
        for si_prefix in unit.si_prefixes:
            if string_wo_whitespaces.startswith(si_prefix.prefix):
                assert stop_index > len(si_prefix.prefix)
                openmm_unit = si_prefix * unit.__dict__[string_wo_whitespaces[len(si_prefix.prefix):stop_index]]
                break
        else:
            openmm_unit = unit.__dict__[string_wo_whitespaces[:stop_index]]
        # If the composite unit only contains one unit, the conversion is finished.
        if stop_index == len(string_wo_whitespaces):
            return openmm_unit
        # Check if the unit that was just found is followed by a power.
        # This power appears to be always positive.
        if string_wo_whitespaces[stop_index] == "*" and string_wo_whitespaces[stop_index + 1] == "*":
            power_index = 1
            assert len(string_wo_whitespaces) > stop_index + 2
            power = int(string_wo_whitespaces[stop_index + 2:stop_index + 2 + power_index])
            # Simply try out how many digits the power has by trying to convert the substring to an integer.
            while stop_index + 2 + power_index < len(string_wo_whitespaces):
                try:
                    power_index += 1
                    power = int(string_wo_whitespaces[stop_index + 2:stop_index + 2 + power_index])
                except ValueError:
                    power_index -= 1
                    break
            openmm_unit = openmm_unit ** power
            stop_index += 2 + power_index
            # Check if the conversion is finished.
            if stop_index == len(string_wo_whitespaces):
                return openmm_unit
        # Check if the unit that was just found (possibly with a power) is followed by a multiplication or division.
        if string_wo_whitespaces[stop_index] == "*":
            return openmm_unit * Quantity._openmm_unit_from_string(string_wo_whitespaces[stop_index + 1:])
        if string_wo_whitespaces[stop_index] == "/":
            # If more than one unit is in the denominator of the composite, the units are enclosed in parentheses.
            # It appears the composite unit always ends after the closing bracket.
            if string_wo_whitespaces[stop_index + 1] == "(":
                bracket_index = string_wo_whitespaces[stop_index + 1:].index(")")
                assert stop_index + 1 + bracket_index == len(string_wo_whitespaces) - 1
                return (openmm_unit
                        / (Quantity._openmm_unit_from_string(
                            string_wo_whitespaces[stop_index + 2:stop_index + 1 + bracket_index])))
            return openmm_unit / Quantity._openmm_unit_from_string(string_wo_whitespaces[stop_index + 1:])
        raise RuntimeError("This should not happen.")


@dataclass(order=True, frozen=True)
class RunParameters(object):
    """
    Data class for the parameters of an OpenMM simulation of colloidal particles periodic boundary conditions.

    This dataclass can be written to and read from a yaml file. The yaml file contains the parameters as key-value
    pairs. Any OpenMM quantities are converted to Quantity objects that can be represented in a readable way in the
    yaml file.

    The potentials are given in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). Any references to equations or symbols in the code refer to this
    paper.

    :param initial_configuration:
        The path to the initial configuration of the system in an xyz file.
        The filename must end with ".xyz".
        Defaults to "colloids/tests/first_frame.xyz".
    :type initial_configuration: str
    :param masses:
        The masses of the different types of colloidal particles that appear in the initial configuration file.
        The keys of the dictionary are the types of the colloidal particles and the values are the masses.
        The unit of the masses must be compatible with atomic mass units and the values must be greater than zero.
        Defaults to {"P": 1.0 * unit.amu, "N": (95.0 / 105.0) ** 3 * unit.amu}.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii of the different types of colloidal particles that appear in the initial configuration file.
        The keys of the dictionary are the types of the colloidal particles and the values are the radii.
        The unit of the radii must be compatible with nanometers and the values must be greater than zero.
        Defaults to {"P": 105.0 * (unit.nano * unit.meter), "N": 95.0 * (unit.nano * unit.meter)}.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials of the different types of colloidal particles that appear in the initial configuration
        file.
        The keys of the dictionary are the types of the colloidal particles and the values are the surface potentials.
        The unit of the surface potentials must be compatible with millivolts.
        Defaults to {"P": 44.0 * (unit.milli * unit.volt), "N": -54.0 * (unit.milli * unit.volt)}.
    :type surface_potentials: dict[str, unit.Quantity]
    :param side_length:
        The side length of the cubic simulation box.
        The unit of the side_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 12328.05 * (unit.nano * unit.meter).
    :type side_length: unit.Quantity
    :param platform_name:
        The name of the platform to use for the simulation.
        Defaults to "Reference". Other possible choices are "CPU", "CUDA", or "OpenCL".
    :type platform_name: str
    :param temperature:
        The temperature of the system.
        The unit of the temperature must be compatible with kelvin and the value must be greater than zero.
        Defaults to 298.0 * unit.kelvin.
    :type temperature: unit.Quantity
    :param collision_rate:
        The collision rate of the Langevin integrator.
        The unit of the collision_rate must be compatible with 1/picoseconds and the value must be greater than zero.
        Defaults to 0.01 / (unit.pico * unit.second).
    :type collision_rate: unit.Quantity
    :param timestep:
        The timestep of the simulation.
        The unit of the timestep must be compatible with picoseconds and the value must be greater than zero.
        Defaults to 0.05 * (unit.pico * unit.second).
    :type timestep: unit.Quantity
    :param brush_density:
        The polymer surface density in the Alexander-de Gennes polymer brush model [i.e., sigma in eq. (1)].
        The unit of the brush_density must be compatible with 1/nanometer^2 and the value must be greater than zero.
        Defaults to 0.09 / ((unit.nano * unit.meter) ** 2).
    :type brush_density: unit.Quantity
    :param brush_length:
        The thickness of the brush in the Alexander-de Gennes polymer brush model [i.e., L in eq. (1)].
        The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 10.6 * (unit.nano * unit.meter).
    :type brush_length: unit.Quantity
    :param debye_length:
        The Debye screening length within DLVO theory [i.e., lambda_D].
        The unit of the debye_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 5.726968 * (unit.nano * unit.meter).
    :type debye_length: unit.Quantity
    :param dielectric_constant:
        The dielectric constant of the solvent [i.e., epsilon].
        The value of the dielectric constant must be greater than zero.
        Defaults to 80.0.
    :type dielectric_constant: float
    :param use_log:
        If True, the electrostatic force uses the more accurate equation involving a logarithm [i.e., eq. (12.5.2) in
        Hunter, Foundations of Colloid Science (Oxford University Press, 2001), 2nd edition] instead of the simpler
        equation that only involves an exponential [i.e., eq. (12.5.5) in Hunter, Foundations of Colloid Science
        (Oxford University Press, 2001), 2nd edition].
        Defaults to False.
    :type use_log: bool
    :param use_tabulated:
        If True, the steric and electrostatic forces are computed based on tabulated functions.
        If False, the steric and electrostatic forces are computed based on algebraic expressions.
        Defaults to False.
    :type use_tabulated: bool
    :param integrator_seed:
        The seed for the random number generator of the integrator.
        If None, a random seed is used.
        Defaults to None.
    :type integrator_seed: Optional[int]
    :param velocity_seed:
        The seed for the random number generator that is used to sample the initial velocities.
        If None, a random seed is used.
        Defaults to None.
    :type velocity_seed: Optional[int]
    :param run_steps:
        The number of time steps to run the simulation.
        The number of time steps must be greater than zero.
        Defaults to 100.
    :type run_steps: int
    :param state_data_interval:
        The interval at which state data is written to a csv file.
        The interval must be greater than zero.
        Defaults to 100.
    :type state_data_interval: int
    :param state_data_filename:
        The name of the csv file to which the state data is written.
        The filename must end with ".csv".
        Defaults to "state_data.csv".
    :type state_data_filename: str
    :param trajectory_interval:
        The interval at which the trajectory is written to a gsd file.
        The interval must be greater than zero.
        Defaults to 100.
    :type trajectory_interval: int
    :param trajectory_filename:
        The name of the gsd file to which the trajectory is written.
        The filename must end with ".gsd".
        Defaults to "trajectory.gsd".
    :type trajectory_filename: str
    :param checkpoint_interval:
        The interval at which the checkpoint is written to a chk file.
        The interval must be greater than zero.
        Defaults to 100.
    :type checkpoint_interval: int
    :param checkpoint_filename:
        The name of the chk file to which the checkpoint is written.
        The filename must end with ".chk".
        Defaults to "checkpoint.chk".
    :type checkpoint_filename: str
    :param minimize_energy_initially:
        If True, the energy of the system is minimized before the simulation starts.
        Defaults to False.
    :type minimize_energy_initially: bool
    :param final_configuration_gsd_filename:
        The name of the gsd file to which the final configuration is written.
        If None, the final configuration is not written to a gsd file.
        The filename must end with ".gsd".
        Defaults to "final_frame.gsd".
    :type final_configuration_gsd_filename: Optional[str]
    :param final_configuration_xyz_filename:
        The name of the xyz file to which the final configuration is written.
        If None, the final configuration is not written to an xyz file.
        The filename must end with ".xyz".
        Defaults to "final_frame.xyz".
    :type final_configuration_xyz_filename: Optional[str]

    :raises TupeError:
        If any of the quantities has an incompatible unit.
    :raises ValueError:
        If any of the parameters has an invalid value.
    """

    initial_configuration: str = "colloids/tests/first_frame.xyz"
    masses: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"P": 1.0 * unit.amu, "N": (95.0 / 105.0) ** 3 * unit.amu})
    radii: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"P": 105.0 * (unit.nano * unit.meter), "N": 95.0 * (unit.nano * unit.meter)})
    surface_potentials: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"P": 44.0 * (unit.milli * unit.volt), "N": -54.0 * (unit.milli * unit.volt)})
    side_length: unit.Quantity = 12328.05 * (unit.nano * unit.meter)
    platform_name: str = "Reference"
    temperature: unit.Quantity = 298.0 * unit.kelvin
    collision_rate: unit.Quantity = 0.01 / (unit.pico * unit.second)
    timestep: unit.Quantity = 0.05 * (unit.pico * unit.second)
    brush_density: unit.Quantity = 0.09 / ((unit.nano * unit.meter) ** 2)
    brush_length: unit.Quantity = 10.6 * (unit.nano * unit.meter)
    debye_length: unit.Quantity = 5.726968 * (unit.nano * unit.meter)
    dielectric_constant: float = 80.0
    use_log: bool = False
    use_tabulated: bool = False
    integrator_seed: Optional[int] = None
    velocity_seed: Optional[int] = None
    run_steps: int = 100
    state_data_interval: int = 100
    state_data_filename: str = "state_data.csv"
    trajectory_interval: int = 100
    trajectory_filename: str = "trajectory.gsd"
    checkpoint_interval: int = 100
    checkpoint_filename: str = "checkpoint.chk"
    minimize_energy_initially: bool = False
    final_configuration_gsd_filename: Optional[str] = "final_frame.gsd"
    final_configuration_xyz_filename: Optional[str] = "final_frame.xyz"

    def __post_init__(self) -> None:
        """Check if the parameters are valid after initialization."""
        if not self.initial_configuration.endswith(".xyz"):
            raise ValueError("The filename of the initial configuration must end with '.xyz'")
        types_from_file, _ = read_xyz_file(self.initial_configuration)
        types = list(dict.fromkeys(types_from_file))
        for t in types:
            if t not in self.masses:
                raise ValueError(f"Type {t} of the initial configuration is not in masses dictionary.")
            if not self.masses[t].unit.is_compatible(unit.amu):
                raise TypeError(f"Mass of type {t} must have a unit compatible with atomic mass units.")
            if self.masses[t] <= 0.0 * unit.amu:
                raise ValueError(f"Mass of type {t} must be greater than zero.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the initial configuration is not in radii dictionary.")
            if not self.radii[t].unit.is_compatible(unit.nano * unit.meter):
                raise TypeError(f"Radius of type {t} must have a unit compatible with nanometers.")
            if self.radii[t] <= 0.0 * (unit.nano * unit.meter):
                raise ValueError(f"Radius of type {t} must be greater than zero.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the initial configuration is not in surface potentials dictionary.")
            if not self.surface_potentials[t].unit.is_compatible(unit.milli * unit.volt):
                raise TypeError(f"Surface potential of type {t} must have a unit compatible with millivolts.")
        for t in self.masses:
            if t not in types:
                raise ValueError(f"Type {t} of the masses dictionary is not in the initial configuration.")
        for t in self.radii:
            if t not in types:
                raise ValueError(f"Type {t} of the radii dictionary is not in the initial configuration.")
        for t in self.surface_potentials:
            if t not in types:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in the initial configuration.")
        if not self.side_length.unit.is_compatible(unit.nano * unit.meter):
            raise TypeError("The side length must have a unit compatible with nanometers.")
        if self.side_length <= 0.0 * (unit.nano * unit.meter):
            raise ValueError("The side length must be greater than zero.")
        if self.platform_name not in ["Reference", "CPU", "CUDA", "OpenCL"]:
            raise ValueError("The platform name must be 'Reference', 'CPU', 'CUDA', or 'OpenCL'.")
        if not self.temperature.unit.is_compatible(unit.kelvin):
            raise TypeError("The temperature must have a unit compatible with kelvin.")
        if self.temperature <= 0.0 * unit.kelvin:
            raise ValueError("The temperature must be greater than zero.")
        if not self.collision_rate.unit.is_compatible((unit.pico * unit.second) ** (-1)):
            raise TypeError("The collision rate must have a unit compatible with 1/picoseconds.")
        if self.collision_rate <= 0.0 * ((unit.pico * unit.second) ** (-1)):
            raise ValueError("The collision rate must be greater than zero.")
        if not self.timestep.unit.is_compatible(unit.pico * unit.second):
            raise TypeError("The timestep must have a unit compatible with picoseconds.")
        if self.timestep <= 0.0 * (unit.pico * unit.second):
            raise ValueError("The timestep must be greater than zero.")
        if not self.brush_density.unit.is_compatible((unit.nano * unit.meter) ** (-2)):
            raise TypeError("The brush density must have a unit compatible with 1/nanometer^2.")
        if self.brush_density <= 0.0 * ((unit.nano * unit.meter) ** (-2)):
            raise ValueError("The brush density must be greater than zero.")
        if not self.brush_length.unit.is_compatible(unit.nano * unit.meter):
            raise TypeError("The brush length must have a unit compatible with nanometers.")
        if self.brush_length <= 0.0 * (unit.nano * unit.meter):
            raise ValueError("The brush length must be greater than zero.")
        if not self.debye_length.unit.is_compatible(unit.nano * unit.meter):
            raise TypeError("The Debye length must have a unit compatible with nanometers.")
        if self.debye_length <= 0.0 * (unit.nano * unit.meter):
            raise ValueError("The Debye length must be greater than zero.")
        if self.dielectric_constant <= 0.0:
            raise ValueError("The dielectric constant must be greater than zero.")
        if self.run_steps <= 0:
            raise ValueError("The number of time steps must be greater than zero.")
        if self.state_data_interval <= 0:
            raise ValueError("The state data interval must be greater than zero.")
        if not self.state_data_filename.endswith(".csv"):
            raise ValueError("The filename of the state data must end with '.csv'.")
        if self.trajectory_interval <= 0:
            raise ValueError("The trajectory interval must be greater than zero.")
        if not self.trajectory_filename.endswith(".gsd"):
            raise ValueError("The filename of the trajectory must end with '.gsd'.")
        if self.checkpoint_interval <= 0:
            raise ValueError("The checkpoint interval must be greater than zero.")
        if not self.checkpoint_filename.endswith(".chk"):
            raise ValueError("The filename of the checkpoint must end with '.chk'.")
        if (self.final_configuration_gsd_filename is not None
                and not self.final_configuration_gsd_filename.endswith(".gsd")):
            raise ValueError("The filename of the final configuration must end with '.gsd'.")
        if (self.final_configuration_xyz_filename is not None
                and not self.final_configuration_xyz_filename.endswith(".xyz")):
            raise ValueError("The filename of the final configuration must end with '.xyz'.")

    @classmethod
    def from_yaml(cls, filename: str) -> "RunParameters":
        """
        Read the parameters of this dataclass from a yaml file.

        Quantities are converted to OpenMM quantities.

        :param filename:
            The name of the yaml file.
        :type filename: str

        :return:
            The parameters.
        :rtype: RunParameters
        """
        with open(filename, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in params.items():
            params[key] = cls._convert_to_openmm_quantity(value)
        return cls(**params)

    def to_yaml(self, filename: str) -> None:
        """
        Write a yaml file with the parameters of this dataclass.

        OpenMM quantities are converted to Quantity objects.

        :param filename:
            The name of the yaml file.
        :type filename: str
        """
        with open(filename, "w") as f:
            yaml.dump(self._as_dictionary(), f, default_flow_style=False)

    def _as_dictionary(self):
        """Represent this dataclass as a dictionary while converting all OpenMM quantities to Quantity objects."""
        result_dict = {}
        for f in fields(self):
            assert f.name not in result_dict
            result_dict[f.name] = self._convert_to_quantity(getattr(self, f.name))
        return result_dict

    @staticmethod
    def _convert_to_quantity(obj):
        """Recursively convert OpenMM quantities to Quantity objects."""
        if isinstance(obj, (list, tuple)):
            return type(obj)(RunParameters._convert_to_quantity(item) for item in obj)
        elif isinstance(obj, dict):
            return dict((RunParameters._convert_to_quantity(key), RunParameters._convert_to_quantity(value))
                        for key, value in obj.items())
        elif isinstance(obj, unit.Quantity):
            return Quantity(obj)
        else:
            return deepcopy(obj)

    @staticmethod
    def _convert_to_openmm_quantity(obj):
        """Recursively convert Quantity objects to OpenMM quantities."""
        if isinstance(obj, (list, tuple)):
            return type(obj)(RunParameters._convert_to_openmm_quantity(item) for item in obj)
        elif isinstance(obj, dict):
            return dict((RunParameters._convert_to_openmm_quantity(key),
                         RunParameters._convert_to_openmm_quantity(value))
                        for key, value in obj.items())
        elif isinstance(obj, Quantity):
            return obj.to_openmm_quantity()
        else:
            return deepcopy(obj)


if __name__ == '__main__':
    RunParameters(initial_configuration="tests/first_frame.xyz").to_yaml("example.yaml")
    parameters = RunParameters.from_yaml("example.yaml")
    print(parameters)
