from openmm import unit
import pytest
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters


class TestColloidPotentialsParametersExceptions(object):
    def test_exceptions_brush_density(self):
        with pytest.raises(TypeError):
            ColloidPotentialsParameters(brush_density=0.09 / (unit.nano * unit.meter))
        with pytest.raises(ValueError):
            ColloidPotentialsParameters(brush_density=-0.09 / ((unit.nano * unit.meter) ** 2))

    def test_exceptions_brush_length(self):
        with pytest.raises(TypeError):
            ColloidPotentialsParameters(brush_length=10.0 / (unit.nano * unit.meter))
        with pytest.raises(ValueError):
            ColloidPotentialsParameters(brush_length=-10.0 * (unit.nano * unit.meter))

    def test_exceptions_debye_length(self):
        with pytest.raises(TypeError):
            ColloidPotentialsParameters(debye_length=5.0 / (unit.nano * unit.meter))
        with pytest.raises(ValueError):
            ColloidPotentialsParameters(debye_length=-5.0 * (unit.nano * unit.meter))

    def test_exceptions_temperature(self):
        with pytest.raises(TypeError):
            ColloidPotentialsParameters(temperature=298.0 / unit.kelvin)
        with pytest.raises(ValueError):
            ColloidPotentialsParameters(temperature=-298.0 * unit.kelvin)

    def test_exceptions_dielectric_constant(self):
        with pytest.raises(ValueError):
            ColloidPotentialsParameters(dielectric_constant=-80.0)


if __name__ == '__main__':
    pytest.main([__file__])
