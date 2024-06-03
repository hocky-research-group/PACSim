import os
import subprocess
import ase.io
import gsd.hoomd
import pytest
from colloids.helper_functions import write_xyz_file_from_gsd_frame


class TestCubicLatticeWithSatellitesGenerator(object):
    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @pytest.fixture
    def run_parameters_file(self):
        return "run_test.yaml"

    @pytest.fixture
    def initial_configuration_filename(self):
        return "first_frame.xyz"

    @pytest.fixture(autouse=True)
    def create_xyz_reference_configurations(self):
        xyz_filenames = []
        for filename in ("reference_configuration.gsd", "reference_configuration_with_padding.gsd"):
            xyz_reference_configuration_filename = filename.replace(".gsd", ".xyz")
            xyz_filenames.append(xyz_reference_configuration_filename)
            # Reference configuration in gsd format created by a legacy script with hoomd.
            with gsd.hoomd.open(filename, "r") as f:
                assert len(f) == 1
                write_xyz_file_from_gsd_frame(xyz_reference_configuration_filename, f[0])
        yield
        for filename in xyz_filenames:
            os.remove(filename)

    @pytest.mark.parametrize("configuration_parameters_file,reference_configuration_filename",
                             [("configuration_test.yaml", "reference_configuration.xyz"),
                              ("configuration_test_with_padding.yaml", "reference_configuration_with_padding.xyz")])
    def test_cubic_lattice_with_satellites_generator(self, run_parameters_file, configuration_parameters_file,
                                                     initial_configuration_filename, reference_configuration_filename):
        subprocess.run(f"colloids-create {run_parameters_file} {configuration_parameters_file}",
                       shell=True, check=True)
        assert os.path.isfile(initial_configuration_filename)
        atoms = ase.io.read(initial_configuration_filename, format="extxyz")
        reference_atoms = ase.io.read(reference_configuration_filename, format="extxyz")
        assert atoms.get_chemical_symbols() == reference_atoms.get_chemical_symbols()
        assert atoms.get_cell() == pytest.approx(reference_atoms.get_cell(), rel=1e-12, abs=1e-12)
        assert atoms.get_positions() == pytest.approx(reference_atoms.get_positions(), rel=1e-5, abs=1e-5)
        os.remove(initial_configuration_filename)


if __name__ == '__main__':
    pytest.main([__file__])
