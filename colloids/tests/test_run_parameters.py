import os
from openmm import unit
import pytest
from colloids.run_parameters import Quantity, RunParameters


class TestQuantity(object):
    @pytest.mark.parametrize("openmm_quantity",
                             [1.0 * unit.meter,
                              2.0 * (unit.nano * unit.meter),
                              -1.0 * ((unit.nano * unit.meter) ** 2),
                              1.0 / unit.second,
                              -1.0 * ((unit.pico * unit.second) ** -1),
                              -2.0 / ((unit.pico * unit.second) ** 2),
                              (3.0 * (unit.milli * unit.volt) * (unit.micro * unit.ampere)
                               / (unit.angstrom ** 2 * (unit.nano * unit.second))),
                              (-3.0 * (unit.mega * unit.angstrom) / (unit.milli * unit.second)),
                              (3.0 * (unit.mega * unit.angstrom) / ((unit.milli * unit.second) * unit.volt)),
                              (12.0 * (unit.mega * unit.angstrom) / (unit.milli * unit.second) * unit.volt)])
    def test_quantity(self, openmm_quantity):
        print(openmm_quantity.unit.get_symbol())
        new_openmm_quantity = Quantity(openmm_quantity).to_openmm_quantity()
        print(new_openmm_quantity)
        assert new_openmm_quantity == openmm_quantity


class TestRunParameters(object):
    @pytest.fixture
    def parameters(self):
        return RunParameters(initial_configuration="first_frame.xyz")

    @pytest.fixture
    def yaml_file(self, parameters):
        parameters.to_yaml("test.yaml")
        yield "test.yaml"
        os.remove("test.yaml")

    @pytest.fixture
    def yaml_parameters(self, yaml_file):
        return RunParameters.from_yaml(yaml_file)

    def test_run_parameters(self, parameters, yaml_parameters):
        assert yaml_parameters == parameters


if __name__ == '__main__':
    pytest.main([__file__])
