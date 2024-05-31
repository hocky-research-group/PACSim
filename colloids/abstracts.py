from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, Iterator
from openmm import CustomNonbondedForce, unit
import yaml
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters


class OpenMMPotentialAbstract(ABC):
    """
    Abstract class for a potential implemented with the CustomNonbondedForces class of openmm.

    The inheriting classes must implement the add_particle and yield_potentials methods so that they can be conveniently
    added to an openmm system.

    The add_particle method should be called for every particle in the system before the method yield_potentials is
    used in order to add the potential to the openmm system.
    """

    def __init__(self) -> None:
        """Constructor of the OpenMMPotentialAbstract class."""
        self._add_particle_called = False
        self._yield_potentials_called = False

    @abstractmethod
    def add_particle(self, *args: Any, **kwargs: Any) -> None:
        """
        Add a particle with the given parameters to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        Note that the overriding method in the inheriting class should call this method first because it checks that the
        method yield_potentials was not called before.

        :param args:
            Parameters of the particle as positional arguments.
        :type args: Any
        :param kwargs:
            Parameters of the particle as keyword arguments.
        :type kwargs: Any

        :raises RuntimeError:
            If the method yield_potentials was called before this method.
        """
        if self._yield_potentials_called:
            raise RuntimeError("method add_particle must be called for every particle in the system before the method "
                               "yield_potentials is used")
        self._add_particle_called = True

    # noinspection PyTypeChecker
    @abstractmethod
    def yield_potentials(self) -> Iterator[CustomNonbondedForce]:
        """
        Generate all potentials in the systems that are necessary to properly include the potential in an openmm system.

        This method has to be called after the method add_particle was called for every particle in the system. Note
        that the overriding method in the inheriting class should call this method first because it checks that the
        method add_particle was called before.

        The generated potentials can be added to the openmm system using the system.addForce method.

        :return:
            A generator that yields all potentials handled by this class.
        :rtype: Iterator[CustomNonbondedForce]

        :raises RuntimeError:
            If the method add_particle was not called before this method.
        """
        if not self._add_particle_called:
            raise RuntimeError("method add_particle must be called for every particle in the system before the method "
                               "yield_potentials is used")
        self._yield_potentials_called = True


class ColloidPotentialsAbstract(OpenMMPotentialAbstract):
    """
    Abstract class for the steric and electrostatic pair potentials between colloids in a solution with periodic
    boundary conditions using the CustomNonbondedForces class of openmm.

    The potentials are given in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). They should be implemented in the inheriting classes in one or
    several CustomNonbondedForce instances.

    The inheriting classes must implement the add_particle and yield_potentials methods.

    The steric potential from the Alexander-de Gennes polymer brush model between two colloids depends on their radii
    r_1 and r_2. Similarly, the electrostatic potential from DLVO theory between two colloids depends on their radii r_1
    and r_2 and their surface potentials psi_1 and psi_2. Before the finalized potentials are generated via the
    yield_potentials method in order to add them to the openmm system (using the system.addForce method), the
    add_particle method has to be called for each colloid in the system to define its radius and surface potential.

    :param colloid_potentials_parameters:
        The parameters of the steric and electrostatic pair potentials between colloidal particles.
    :type colloid_potentials_parameters: ColloidPotentialsParameters
    :param periodic_boundary_conditions:
        Whether this force should use periodic cutoffs for the steric and electrostatic potentials.
    :type periodic_boundary_conditions: bool
    """

    _nanometer = unit.nano * unit.meter
    _millivolt = unit.milli * unit.volt

    def __init__(self, colloid_potentials_parameters: ColloidPotentialsParameters,
                 periodic_boundary_conditions: bool) -> None:
        """Constructor of the ColloidPotentialsAbstract class."""
        super().__init__()
        self._parameters = colloid_potentials_parameters
        self._periodic_boundary_conditions = periodic_boundary_conditions

    @abstractmethod
    def add_particle(self, radius: unit.Quantity, surface_potential: unit.Quantity) -> None:
        """
        Add a colloid with a given radius and surface potential to the system.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        Note that the overriding method in the inheriting class should call this method first because it checks the
        input arguments, and that the method yield_potentials was not called before (via the OpenMMPotentialAbstract
        base class).

        :param radius:
            The radius of the colloid.
            The unit of the radius must be compatible with nanometers and the value must be greater than zero.
        :type radius: unit.Quantity
        :param surface_potential:
            The surface potential of the colloid.
            The unit of the surface_potential must be compatible with millivolts.
        :type surface_potential: unit.Quantity

        :raises TypeError:
            If the radius or surface_potential is not a Quantity with a proper unit.
        :raises ValueError:
            If the radius is not greater than zero.
        :raises RuntimeError:
            If the method yield_potentials was called before this method (via the OpenMMPotentialAbstract base class).
        """
        super().add_particle()
        if not radius.unit.is_compatible(self._nanometer):
            raise TypeError("argument radius must have a unit that is compatible with nanometers")
        if not radius.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("argument radius must have a value greater than zero")
        if not surface_potential.unit.is_compatible(unit.milli * unit.volt):
            raise TypeError("argument surface_potential must have a unit that is compatible with volts")


@dataclass(order=True, frozen=True)
class Parameters(object):
    """
    Base dataclass providing methods to store the data in a yaml file and to read the data from a yaml file.

    All standard data types (int, float, str) or standard containers (list, tuple, dict) of them can be stored in an
    inheriting data class. In addition, this dataclass provides methods to store OpenMM quantities in a yaml file by
    wrapping them in a custom class.
    """

    class _Quantity(yaml.YAMLObject):
        """
        Wrapper class for an OpenMM quantity that allows for a simple serialization in a yaml file.

        This class defines the application-specific yaml tag !Quantity by following the description in
        https://pyyaml.org/wiki/PyYAMLDocumentation.

        Although the !!python/object tag would in principle allow to serialize an OpenMM quantity in a yaml file, this
        tag would require specifying an enormous amount of (mostly private) attributes. This class circumvents this
        problem by defining a custom yaml tag that only stores the value and a string representation of the unit of an
        OpenMM quantity. The string representation is obtained by calling the get_name method of the unit of the OpenMM
        quantity.

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

        @classmethod
        def _openmm_unit_from_string(cls, unit_string: str) -> unit.Unit:
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
                    return cls._openmm_unit_from_string(string_wo_whitespaces[2:bracket_index]) ** (-1)
                return cls._openmm_unit_from_string(string_wo_whitespaces[1:]) ** (-1)
            # If the composite unit does not start with a slash, it starts with one unit and ends with a
            # multiplication (*), division (/), or power (**).
            stop_index = 0
            while stop_index < len(string_wo_whitespaces) and string_wo_whitespaces[stop_index] not in ["*", "/"]:
                stop_index += 1
            # The first unit in the composite unit may contain a SI prefix that must be found explicitly because units
            # like millivolt are not directly recognized by openmm.
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
                return openmm_unit * cls._openmm_unit_from_string(string_wo_whitespaces[stop_index + 1:])
            if string_wo_whitespaces[stop_index] == "/":
                # If more than one unit is in the denominator of the composite, the units are enclosed in parentheses.
                # It appears the composite unit always ends after the closing bracket.
                if string_wo_whitespaces[stop_index + 1] == "(":
                    bracket_index = string_wo_whitespaces[stop_index + 1:].index(")")
                    assert stop_index + 1 + bracket_index == len(string_wo_whitespaces) - 1
                    return (openmm_unit
                            / (cls._openmm_unit_from_string(
                                string_wo_whitespaces[stop_index + 2:stop_index + 1 + bracket_index])))
                return openmm_unit / cls._openmm_unit_from_string(string_wo_whitespaces[stop_index + 1:])
            raise RuntimeError("This should not happen.")

    @classmethod
    def from_yaml(cls, filename: str) -> "Parameters":
        """
        Read the parameters of this dataclass from a yaml file.

        Instances of the _Quantity class are converted to OpenMM quantities.

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
        # noinspection PyArgumentList
        return cls(**params)

    def to_yaml(self, filename: str) -> None:
        """
        Write a yaml file with the parameters of this dataclass.

        OpenMM quantities are converted to _Quantity objects.

        :param filename:
            The name of the yaml file.
        :type filename: str
        """
        with open(filename, "w") as f:
            yaml.dump(self._as_dictionary(), f, default_flow_style=False)

    def _as_dictionary(self):
        """Represent this dataclass as a dictionary while converting all OpenMM quantities to _Quantity objects."""
        result_dict = {}
        for f in fields(self):
            assert f.name not in result_dict
            result_dict[f.name] = self._convert_to_quantity(getattr(self, f.name))
        return result_dict

    @staticmethod
    def _convert_to_quantity(obj):
        """Recursively convert OpenMM quantities to _Quantity objects."""
        if isinstance(obj, (list, tuple)):
            return type(obj)(Parameters._convert_to_quantity(item) for item in obj)
        elif isinstance(obj, dict):
            return dict((Parameters._convert_to_quantity(key), Parameters._convert_to_quantity(value))
                        for key, value in obj.items())
        elif isinstance(obj, unit.Quantity):
            return Parameters._Quantity(obj)
        else:
            return deepcopy(obj)

    @staticmethod
    def _convert_to_openmm_quantity(obj):
        """Recursively convert _Quantity objects to OpenMM quantities."""
        if isinstance(obj, (list, tuple)):
            return type(obj)(Parameters._convert_to_openmm_quantity(item) for item in obj)
        elif isinstance(obj, dict):
            return dict((Parameters._convert_to_openmm_quantity(key),
                         Parameters._convert_to_openmm_quantity(value))
                        for key, value in obj.items())
        elif isinstance(obj, Parameters._Quantity):
            return obj.to_openmm_quantity()
        else:
            return deepcopy(obj)
