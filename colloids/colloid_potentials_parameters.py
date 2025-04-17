from openmm import unit
from colloids.units import length_unit, temperature_unit


class ColloidPotentialsParameters(object):
    """
    This class stores the parameters of the steric and electrostatic pair potentials between colloidal particles in a
    solution.

    The potentials are given in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). Any references to equations or symbols in the code refer to this
    paper.

    :param brush_density:
        The polymer surface density in the Alexander-de Gennes polymer brush model [i.e., sigma in eq. (1)].
        The unit of the brush_density must be compatible with 1/nanometer^2 and the value must be greater than zero.
        Defaults to 0.09/nanometer^2.
    :type brush_density: unit.Quantity
    :param brush_length:
        The thickness of the brush in the Alexander-de Gennes polymer brush model [i.e., L in eq. (1)].
        The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 10.0 nanometers.
    :type brush_length: unit.Quantity
    :param debye_length:
        The Debye screening length within DLVO theory [i.e., lambda_D].
        The unit of the debye_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 5.0 nanometers.
    :type debye_length: unit.Quantity
    :param temperature:
        The temperature of the system [i.e., T].
        The unit of the temperature must be compatible with kelvin and the value must be greater than zero.
        Defaults to 298.0 kelvin.
    :type temperature: unit.Quantity
    :param dielectric_constant:
        The dielectric constant of the solvent [i.e., epsilon].
        The value of the dielectric constant must be greater than zero.
        Defaults to 80.0 (i.e., water).
    :type dielectric_constant: float
    :param psi1_scale:
        Scale factor to adjust the surface charge of type 1 particles.
        Defaults to 1.0 (no scaling).
    :type psi1_scale: float
    :param psi2_scale:
        Scale factor to adjust the surface charge of type 2 particles.
        Defaults to 1.0 (no scaling).
    :type psi2_scale: float

    :raises TypeError:
        If the brush_density, brush_length, debye_length, or temperature is not a Quantity with a proper unit.
    :raises ValueError:
        If the brush_density, brush_length, debye_length, temperature, or dielectric_constant is not greater than zero.
    """

    VACUUM_PERMITTIVITY = 8.8541878128e-12 * unit.joule / (unit.volt ** 2 * unit.meter)
    _brush_density_unit = length_unit ** (-2)

    def __init__(self, brush_density: unit.Quantity = 0.09 * _brush_density_unit,
                 brush_length: unit.Quantity = 10.0 * length_unit,
                 debye_length: unit.Quantity = 5.0 * length_unit,
                 temperature: unit.Quantity = 298.0 * temperature_unit,
                 dielectric_constant: float = 80.0):
        """Constructor of the ColloidPotentialsParameters class."""
        if not brush_density.unit.is_compatible(length_unit ** -2):
            raise TypeError("argument brush_density must have a unit that is compatible with 1/nanometer^2")
        if not brush_density.value_in_unit(self._brush_density_unit) > 0.0:
            raise ValueError("argument brush_density must have a value greater than zero")
        if not brush_length.unit.is_compatible(length_unit):
            raise TypeError("argument brush_length must have a unit that is compatible with nanometers")
        if not brush_length.value_in_unit(length_unit) > 0.0:
            raise ValueError("argument brush_length must have a value greater than zero")
        if not debye_length.unit.is_compatible(length_unit):
            raise TypeError("argument debye_length must have a unit that is compatible with nanometers")
        if not debye_length.value_in_unit(length_unit) > 0.0:
            raise ValueError("argument debye_length must have a value greater than zero")
        if not temperature.unit.is_compatible(temperature_unit):
            raise TypeError("argument temperature must have a unit that is compatible with kelvin")
        if not temperature.value_in_unit(temperature_unit) > 0.0:
            raise ValueError("argument temperature must have a value greater than zero")
        if not dielectric_constant > 0.0:
            raise ValueError("argument dielectric_constant must have a value greater than zero")
        self._brush_density = brush_density.in_units_of(self._brush_density_unit)
        self._brush_length = brush_length.in_units_of(length_unit)
        self._debye_length = debye_length.in_units_of(length_unit)
        self._temperature = temperature.in_units_of(temperature_unit)
        self._dielectric_constant = dielectric_constant
        self._psi1_scale = psi1_scale
        self._psi2_scale = psi2_scale

    @property
    def brush_density(self) -> unit.Quantity:
        return self._brush_density

    @property
    def brush_length(self) -> unit.Quantity:
        return self._brush_length

    @property
    def debye_length(self) -> unit.Quantity:
        return self._debye_length

    @property
    def temperature(self) -> unit.Quantity:
        return self._temperature

    @property
    def dielectric_constant(self) -> float:
        return self._dielectric_constant

    @property
    def psi1_scale(self) -> float:
         return self._psi1_scale
 
    @property
     def psi2_scale(self) -> float:
         return self._psi2_scale


if __name__ == '__main__':
    parameters_one = ColloidPotentialsParameters()
    parameters_two = ColloidPotentialsParameters(brush_density=0.1 / (length_unit ** 2))
    print(parameters_one.brush_density)
    print(parameters_two.brush_density)
