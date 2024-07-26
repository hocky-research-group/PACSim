from dataclasses import dataclass, field
from typing import Optional
from openmm import unit
from colloids.abstracts import Parameters
from colloids.helper_functions import read_xyz_file
import warnings


@dataclass(order=True, frozen=True)
class RunParameters(Parameters):
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
    :param wall_directions:
        A list of three booleans indicating whether the walls in the x, y, and z directions are active for
        closed-wall simulations with shifted Lennard-Jones potential walls.
        If any of the wall directions is active, epsilon and alpha must be specified.
        Defaults to [False, False, False].
    :type wall_directions: list[bool]
    :param epsilon:
        The unshifted Lennard-Jones potential well-depth for closed-wall simulations with shifted Lennard-Jones
        potential walls.
        If any wall direction is True, epsilon must be not None, its unit must be compatible with kilojoules per mole
        and the value must be greater than zero.
        Defaults to None.
    :type epsilon: Optional[unit.Quantity]
    :param alpha:
        Factor determining the strength of the attractive part of the Lennard-Jones potential for closed-wall 
        simulations with shifted Lennard-Jones potential walls.
        If any wall direction is True, alpha must be not None and 0 <= alpha <= 1.
        Note that the force of this potential is only continuous if alpha = 1.
    :type alpha: Optional[float]
    :param use_depletion:
        A boolean indicating whether to turn on the depletion attraction for the simulation.
        If depletion attraction is on, phi and depletant radius must be specified.
        Defaults to False.
    :type use_depletion: bool
    :param depletion_phi:
        The number density of polymers in the solution.
        If depletion attraction is on, the value of depletion_phi must not be None and 0<= depletion_phi <=1.
        Defaults to None.
    :type depletion_phi: Optional[float]
    :param depletant_radius: 
        The radius of the polymers in solution for a system with depletion attraction.
        If depletion attraction is on, depletant_radius must not be None, its unit must be compatible with nanometers,
        and the value must be greater than zero.
    :type depletant_radius: Optional[unit.Quantity]

    :raises TypeError:
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
    platform_name: str = "Reference"
    temperature: unit.Quantity = field(default_factory=lambda: 298.0 * unit.kelvin)
    collision_rate: unit.Quantity = field(default_factory=lambda: 0.001574074286750681 / (unit.pico * unit.second))
    timestep: unit.Quantity = field(default_factory=lambda: 0.03176470159055431 * (unit.pico * unit.second))
    brush_density: unit.Quantity = field(default_factory=lambda: 0.09 / ((unit.nano * unit.meter) ** 2))
    brush_length: unit.Quantity = field(default_factory=lambda: 10.6 * (unit.nano * unit.meter))
    debye_length: unit.Quantity = field(default_factory=lambda: 5.726968 * (unit.nano * unit.meter))
    use_depletion: bool = False
    dielectric_constant: float = 80.0
    cutoff_factor: float = 21.0
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
    epsilon: Optional[unit.Quantity] = None
    alpha: Optional[float] = None
    wall_directions: list[bool] = field(default_factory=lambda: [False, False, False])
    depletion_phi: Optional[float] = None
    depletant_radius: Optional[unit.Quantity] = None

    def __post_init__(self) -> None:
        """Check if the parameters are valid after initialization."""
        if not self.initial_configuration.endswith(".xyz"):
            raise ValueError("The filename of the initial configuration must end with '.xyz'")
        for t in self.masses:
            if not self.masses[t].unit.is_compatible(unit.amu):
                raise TypeError(f"Mass of type {t} must have a unit compatible with atomic mass units.")
            if self.masses[t] <= 0.0 * unit.amu:
                raise ValueError(f"Mass of type {t} must be greater than zero.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the masses dictionary is not in radii dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the masses dictionary is not in surface potentials dictionary.")
        for t in self.radii:
            if not self.radii[t].unit.is_compatible(unit.nano * unit.meter):
                raise TypeError(f"Radius of type {t} must have a unit compatible with nanometers.")
            if self.radii[t] <= 0.0 * (unit.nano * unit.meter):
                raise ValueError(f"Radius of type {t} must be greater than zero.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the radii dictionary is not in masses dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the initial configuration is not in surface potentials dictionary.")
        for t in self.surface_potentials:
            if not self.surface_potentials[t].unit.is_compatible(unit.milli * unit.volt):
                raise TypeError(f"Surface potential of type {t} must have a unit compatible with millivolts.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in masses dictionary.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in radii dictionary.")
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
        if isinstance(self.wall_directions, str):
            raise ValueError("Wall directions was parsed as a string although it should be a list of bools. "
                             "Make sure that the yaml file is correctly formatted and that there is space after each "
                             "dash in the list of wall directions.")
        if len(self.wall_directions) != 3:
            raise ValueError("Wall directions must be specified for three dimensions.")
        if any(self.wall_directions):
            if self.epsilon is None:
                raise ValueError("Epsilon must be specified if walls are active.")
            if not self.epsilon.unit.is_compatible(unit.kilojoule_per_mole):
                raise TypeError("Epsilon must have a unit compatible with kilojoules per mole.")
            if self.epsilon <= 0.0 * unit.kilojoule_per_mole:
                raise ValueError("epsilon must be greater than zero.")
            if self.alpha is None:
                raise ValueError("Alpha must be specified if walls are active.")
            if not 0.0 <= self.alpha <= 1.0:
                raise ValueError("Alpha must be between zero and one.")
        else:
            if self.epsilon is not None:
                raise ValueError("Epsilon must not be specified if walls are not active.")
            if self.alpha is not None:
                raise ValueError("Alpha must not be specified if walls are not active.")
        if self.use_depletion:
            if self.depletion_phi is None:
                raise ValueError("Phi must be specified if depletion is on.")
            if not 0.0 <= self.depletion_phi <= 1.0:
                raise ValueError("Phi must be between zero and one.")
            if self.depletant_radius is None:
                raise ValueError("Depletant radius must be specified if depletion is on.")
            if not self.depletant_radius.unit.is_compatible(unit.nano * unit.meter):
                raise TypeError("Depletant radius must have a unit compatible with nanometers.")
            if self.depletant_radius <= 0.0 * (unit.nano * unit.meter):
                raise ValueError("Depletant radius must be greater than zero.")
            for t in self.radii:
                if self.depletant_radius/self.radii[t] > 0.1547:
                    warnings.warn("Size ratio of depletant to colloid particles is too large."
                                    " Analytical computation of depletion potential may be invalid.")
        else:
            if self.depletion_phi is not None:
                raise ValueError("Phi must not be specified if depletion potential is not on.")
            if self.depletant_radius is not None:
                raise ValueError("Depletant radius must not be specified if depletion potential is not on.")

    def check_types_of_initial_configuration(self):
        """
        Check if the types of the initial configuration are consistent with the masses, radii, and surface-potentials
        dictionaries.

        :raises ValueError:
            If the types of the initial configuration are not consistent with the masses, radii, and surface-potentials
            dictionaries.
        """
        types_from_file, _, _ = read_xyz_file(self.initial_configuration)
        types = list(dict.fromkeys(types_from_file))
        for t in types:
            if t not in self.masses:
                raise ValueError(f"Type {t} of the initial configuration is not in masses dictionary.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the initial configuration is not in radii dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the initial configuration is not in surface potentials dictionary.")
        for t in self.masses:
            if t not in types:
                raise ValueError(f"Type {t} of the masses dictionary is not in the initial configuration.")
        for t in self.radii:
            if t not in types:
                raise ValueError(f"Type {t} of the radii dictionary is not in the initial configuration.")
        for t in self.surface_potentials:
            if t not in types:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in the initial configuration.")


if __name__ == '__main__':
    RunParameters(initial_configuration="tests/first_frame.xyz").to_yaml("example.yaml")
    parameters = RunParameters.from_yaml("example.yaml")
    print(parameters)
