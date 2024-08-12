import os
import pytest
from openmm import unit
from colloids.colloids_run import colloids_run


@pytest.mark.filterwarnings("ignore:The number of time steps is zero.")
@pytest.mark.filterwarnings("ignore:Size ratio of depletant")
class TestEnergies(object):
    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @pytest.fixture(autouse=True)
    def tear_down(self):
        yield
        assert os.path.isfile("final_frame.gsd")
        assert os.path.isfile("final_frame.xyz")
        assert os.path.isfile("state_data.csv")
        assert os.path.isfile("trajectory.gsd")
        os.remove("final_frame.gsd")
        os.remove("final_frame.xyz")
        os.remove("state_data.csv")
        os.remove("trajectory.gsd")

    @pytest.mark.parametrize("yaml_file,expected_energy",
                             [("run_cp.yaml", -10122.62271419458),
                              ("run_cp_depletion.yaml", -44424.402360702596),
                              ("run_cp_walls_substrate.yaml", -12286.15816959927),
                              ("run_cp_walls_substrate_depletion.yaml", -54312.08243054912),
                              ("run_cp_walls_substrate_gravity.yaml", -12286.225633072137),
                              ("run_cp_walls_snowman.yaml", -231.80203391587074),
                              ("run_cp_walls.yaml", -0.13903947128075853)])
    def test_energies(self, yaml_file, expected_energy):
        simulation = colloids_run([yaml_file])
        potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        assert potential_energy.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(expected_energy,
                                                                                        rel=1.0e-12, abs=1.0e-12)


if __name__ == '__main__':
    pytest.main([__file__])
