import openmm
from openmm import unit
import pytest
from colloids import integrators
from colloids.colloids_run import initialize_barostat
from colloids.run_parameters import RunParameters


class TestBarostatFactories(object):
    def test_isotropic_barostat_type_and_values(self):
        barostat = integrators.MonteCarloBarostat(298.0 * unit.kelvin, 2.0 * unit.bar, frequency=17)
        assert isinstance(barostat, openmm.MonteCarloBarostat)
        assert barostat.getFrequency() == 17
        assert abs(barostat.getDefaultPressure().value_in_unit(unit.bar) - 2.0) < 1e-9
        assert abs(barostat.getDefaultTemperature().value_in_unit(unit.kelvin) - 298.0) < 1e-9

    def test_anisotropic_barostat_type_and_scale_flags(self):
        barostat = integrators.MonteCarloAnisotropicBarostat(
            298.0 * unit.kelvin, 1.0 * unit.bar, 1.0 * unit.bar, 3.0 * unit.bar,
            scale_x=True, scale_y=False, scale_z=True, frequency=20)
        assert isinstance(barostat, openmm.MonteCarloAnisotropicBarostat)
        assert barostat.getScaleX() is True
        assert barostat.getScaleY() is False
        assert barostat.getScaleZ() is True
        pressure = barostat.getDefaultPressure()
        assert abs(pressure[2].value_in_unit(unit.bar) - 3.0) < 1e-9


class TestNPTRunParameters(object):
    def test_isotropic_npt_valid(self):
        parameters = RunParameters(npt_pressure=1.0 * unit.bar, npt_frequency=25)
        assert parameters.npt_pressure == 1.0 * unit.bar
        assert parameters.npt_frequency == 25

    def test_anisotropic_npt_valid(self):
        parameters = RunParameters(npt_pressure=[1.0 * unit.bar, 1.0 * unit.bar, 2.0 * unit.bar],
                                   npt_frequency=10, npt_scale=[True, True, False])
        assert parameters.npt_scale == [True, True, False]

    def test_npt_yaml_roundtrip(self, tmp_path):
        parameters = RunParameters(npt_pressure=1.5 * unit.bar, npt_frequency=13)
        path = str(tmp_path / "npt.yaml")
        parameters.to_yaml(path)
        restored = RunParameters.from_yaml(path)
        assert abs(restored.npt_pressure.value_in_unit(unit.bar) - 1.5) < 1e-9
        assert restored.npt_frequency == 13

    def test_frequency_required_with_pressure(self):
        with pytest.raises(ValueError):
            RunParameters(npt_pressure=1.0 * unit.bar)

    def test_frequency_forbidden_without_pressure(self):
        with pytest.raises(ValueError):
            RunParameters(npt_frequency=10)

    def test_scale_forbidden_for_isotropic(self):
        with pytest.raises(ValueError):
            RunParameters(npt_pressure=1.0 * unit.bar, npt_frequency=10, npt_scale=[True, True, True])

    def test_bad_pressure_unit(self):
        with pytest.raises(TypeError):
            RunParameters(npt_pressure=1.0 * unit.kelvin, npt_frequency=10)

    def test_wrong_length_pressure_list(self):
        with pytest.raises(ValueError):
            RunParameters(npt_pressure=[1.0 * unit.bar, 1.0 * unit.bar], npt_frequency=10)

    def test_barostat_not_selectable_as_integrator(self):
        with pytest.raises(ValueError):
            RunParameters(integrator="MonteCarloBarostat", integrator_parameters={})


class TestInitializeBarostat(object):
    def test_none_when_no_pressure(self):
        assert initialize_barostat(RunParameters()) is None

    def test_isotropic_dispatch(self):
        barostat = initialize_barostat(RunParameters(npt_pressure=1.0 * unit.bar, npt_frequency=25))
        assert isinstance(barostat, openmm.MonteCarloBarostat)

    def test_anisotropic_dispatch(self):
        barostat = initialize_barostat(RunParameters(
            npt_pressure=[1.0 * unit.bar, 1.0 * unit.bar, 2.0 * unit.bar], npt_frequency=25,
            npt_scale=[True, False, True]))
        assert isinstance(barostat, openmm.MonteCarloAnisotropicBarostat)
        assert barostat.getScaleY() is False


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
