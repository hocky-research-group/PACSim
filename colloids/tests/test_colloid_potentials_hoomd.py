from openmm import unit
import pytest
from colloids import ColloidPotentialsParameters
try:
    import hoomd
    import hoomd.md
    from colloids.colloid_potentials_tabulated_hoomd import ColloidPotentialsTabulatedHoomd
except ModuleNotFoundError:
    # Skip all tests from this module.
    pytestmark = pytest.mark.skipif(True, reason="hoomd is not installed")


class TestColloidPotentialsTabulatedHoomd(object):
    @pytest.fixture
    def parameters(self):
        params = {
            "radius_positive": 105.0 * (unit.nano * unit.meter),
            "radius_negative": 95.0 * (unit.nano * unit.meter),
            "surface_potential_positive": 44.0 * (unit.milli * unit.volt),
            "surface_potential_negative": -54.0 * (unit.milli * unit.volt),
            "colloid_potentials_parameters": ColloidPotentialsParameters(
                brush_density=0.09 / ((unit.nano * unit.meter) ** 2),
                brush_length=10.6 * (unit.nano * unit.meter),
                debye_length=5.726968 * (unit.nano * unit.meter),
                temperature=298.0 * unit.kelvin,
                dielectric_constant=80.0),
            "collision_rate": 0.01 / (unit.pico * unit.second),
            "timestep": 0.05 * (unit.pico * unit.second),
            "mass_positive": 1.0 * unit.amu,
            "side_length": 12328.05 * (unit.nano * unit.meter),
        }
        params["mass_negative"] = (
                (params["radius_negative"] / params["radius_positive"]) ** 3 * params["mass_positive"])
        return params

    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @pytest.mark.parametrize("filename,shift,expected",
                             [["first_frame.gsd", True, -0.4659521251272684],
                              ["first_frame.gsd", False, -0.4911881156335491],
                              ["final_frame.gsd", True, -57526.04734257985],
                              ["final_frame.gsd", False, -57526.117461693]])
    def test_colloid_potentials_tabulated_hoomd(self, parameters, filename, shift, expected):
        # Reference results are from a legacy hoomd code in a jupyter notebook that is not available in the repository.
        hoomd.context.initialize("--mode=cpu --notice-level=0")
        snapshot = hoomd.data.gsd_snapshot(filename)
        hoomd.init.read_snapshot(snapshot)
        nl = hoomd.md.nlist.cell()
        ColloidPotentialsTabulatedHoomd(
            radius_one=parameters["radius_positive"].value_in_unit(unit.nano * unit.meter),
            radius_two=parameters["radius_negative"].value_in_unit(unit.nano * unit.meter),
            surface_potential_one=parameters["surface_potential_positive"].value_in_unit(unit.milli * unit.volt),
            surface_potential_two=parameters["surface_potential_negative"].value_in_unit(unit.milli * unit.volt),
            type_one="P", type_two="N",
            colloid_potentials_parameters=parameters["colloid_potentials_parameters"],
            neighbor_list=nl, shift=shift)
        k_temperature = (parameters["colloid_potentials_parameters"].temperature * unit.BOLTZMANN_CONSTANT_kB
                         * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        hoomd.md.integrate.mode_standard(dt=parameters["timestep"].value_in_unit(unit.pico * unit.second))
        langevin = hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=k_temperature, seed=1)
        langevin.set_gamma("P", parameters["mass_positive"].value_in_unit(unit.amu)
                           * parameters["collision_rate"].value_in_unit((unit.pico * unit.second) ** -1))
        langevin.set_gamma("N", parameters["mass_negative"].value_in_unit(unit.amu)
                           * parameters["collision_rate"].value_in_unit((unit.pico * unit.second) ** -1))
        # noinspection PyTypeChecker
        log = hoomd.analyze.log(filename=None, quantities=["potential_energy"], period=1, overwrite=True, phase=-1)
        hoomd.run(0)
        assert log.query("potential_energy") == pytest.approx(expected, rel=1e-12, abs=1e-12)
