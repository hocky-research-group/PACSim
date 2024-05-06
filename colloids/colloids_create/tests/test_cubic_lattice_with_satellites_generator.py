import os
import subprocess
import ase.io
import gsd.hoomd
import numpy.testing as npt
import pytest
from colloids.helper_functions import write_xyz_file_from_gsd_frame


class TestCubicLatticeWithSatellitesGenerator(object):
    @pytest.fixture
    def run_parameters_file(self):
        return "run_test.yaml"

    @pytest.fixture
    def configuration_parameters_file(self):
        return "configuration_test.yaml"

    @pytest.fixture
    def initial_configuration_filename(self):
        return "first_frame.xyz"

    @pytest.fixture
    def reference_configuration_filename(self):
        xyz_reference_configuration_filename = "reference_configuration.xyz"
        # Reference configuration in gsd format created by a legacy script with hoomd.
        with gsd.hoomd.open("reference_configuration.gsd", "r") as f:
            assert len(f) == 1
            write_xyz_file_from_gsd_frame(xyz_reference_configuration_filename, f[0])
        yield xyz_reference_configuration_filename
        os.remove(xyz_reference_configuration_filename)

    def test_cubic_lattice_with_satellites_generator(self, run_parameters_file, configuration_parameters_file,
                                                     initial_configuration_filename, reference_configuration_filename):
        subprocess.run(f"colloids-create {run_parameters_file} {configuration_parameters_file}",
                       shell=True, check=True)
        assert os.path.isfile(initial_configuration_filename)
        atoms = ase.io.read(initial_configuration_filename, format="extxyz")
        reference_atoms = ase.io.read(reference_configuration_filename, format="extxyz")
        reference_atoms.translate(-reference_atoms[0].position)
        assert atoms.get_chemical_symbols() == reference_atoms.get_chemical_symbols()
        assert atoms.get_cell() == pytest.approx(reference_atoms.get_cell(), rel=1e-12, abs=1e-12)
        assert atoms.get_positions() == pytest.approx(reference_atoms.get_positions(), rel=1e-5, abs=1e-5)
        os.remove(initial_configuration_filename)


if __name__ == '__main__':
    pytest.main([__file__])
