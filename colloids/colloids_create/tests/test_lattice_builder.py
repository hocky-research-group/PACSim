import os
import subprocess
import gsd.hoomd
import numpy as np
from openmm import unit
import pytest
from colloids.colloids_create.configuration_generators.lattice_builder import LatticeBuilder
from colloids.helper_functions import get_cell_from_box
from colloids.units import electric_potential_unit, length_unit


def _min_image_nearest_neighbor(positions, cell):
    """Smallest minimum-image pair distance for a set of positions in a (row-vector) cell."""
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    minimum = np.inf
    for shift in shifts:
        offset = shift @ cell
        deltas = positions[:, None, :] - (positions[None, :, :] + offset)
        distances = np.sqrt((deltas ** 2).sum(-1))
        if np.all(shift == 0):
            np.fill_diagonal(distances, np.inf)
        minimum = min(minimum, float(distances.min()))
    return minimum


class TestLatticeBuilder(object):
    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @pytest.fixture
    def initial_configuration_filename(self):
        return "first_frame_lattice_builder.gsd"

    def test_cubic_lattice_with_satellites_generator(self, initial_configuration_filename):
        # Comparison to reference configurations in gsd format created by a tested version.
        configuration_parameters_file = "configuration_lattice_builder_test.yaml"
        reference_configuration_filename = "reference_configuration_lattice_builder.gsd"

        subprocess.run(f"pacsim-create {configuration_parameters_file} {initial_configuration_filename}",
                       shell=True, check=True)
        assert os.path.isfile(initial_configuration_filename)
        with (gsd.hoomd.open(initial_configuration_filename, "r") as f_new,
              gsd.hoomd.open(reference_configuration_filename, "r") as f_ref):
            assert len(f_new) == 1
            assert len(f_ref) == 1
            frame_new = f_new[0]
            frame_ref = f_ref[0]
            assert frame_new.particles.N == frame_ref.particles.N
            assert len(frame_new.particles.types) == len(frame_ref.particles.types)
            assert all(frame_new.particles.types[nt] == frame_ref.particles.types[ot]
                       for nt, ot in zip(frame_new.particles.typeid, frame_ref.particles.typeid))
            assert np.allclose(frame_new.configuration.box, frame_ref.configuration.box)
            assert np.allclose(frame_new.particles.position, frame_ref.particles.position)

            assert frame_new.constraints.N == frame_ref.constraints.N == 0

            assert np.allclose(frame_new.particles.mass, frame_ref.particles.mass)
            assert np.allclose(frame_new.particles.charge, frame_ref.particles.charge)
            assert np.allclose(frame_new.particles.diameter, frame_ref.particles.diameter)
        os.remove(initial_configuration_filename)

    @staticmethod
    def _build_lattice_builder(periodic, repeats=2):
        nm = length_unit
        return LatticeBuilder(
            masses={"Th": 1.0 * unit.dalton, "P": 0.614125 * unit.dalton},
            radii={"Th": 120.0 * nm, "P": 102.0 * nm},
            surface_potentials={"Th": -35.0 * electric_potential_unit, "P": 35.0 * electric_potential_unit},
            lattice_specification="Th3P4.cif", lattice_repeats=repeats,
            run_parameters_file="run_lattice_builder_test.yaml", radii_padding=5.0 * nm,
            lattice_padding=0.0 * nm, periodic=periodic)

    def test_periodic_bulk_crystal(self):
        periodic_frame = self._build_lattice_builder(periodic=True).generate_configuration()
        cluster_frame = self._build_lattice_builder(periodic=False).generate_configuration()

        # Periodic and cluster builds contain the same particles (same supercell).
        assert periodic_frame.particles.N == cluster_frame.particles.N

        box = np.asarray(periodic_frame.configuration.box)
        cell = get_cell_from_box(box)  # rows are the box vectors
        positions = np.asarray(periodic_frame.particles.position)

        # Th3P4 is cubic, so the periodic box is cubic and un-tilted.
        assert np.allclose(box[3:], 0.0)
        assert np.isclose(box[0], box[1]) and np.isclose(box[1], box[2])

        # Every particle lies inside the box (fractional coordinates in [0, 1)).
        fractional = positions @ np.linalg.inv(cell)
        assert fractional.min() >= -1.0e-4
        assert fractional.max() < 1.0 + 1.0e-4

        # No cores overlap across periodic images (nearest-neighbor distance exceeds the largest
        # possible sum of two core radii).
        nearest = _min_image_nearest_neighbor(positions, cell)
        assert nearest > 2.0 * 120.0  # 2 * max core radius (nm)

        # The periodic box is much smaller than the padded cluster box (no surrounding vacuum).
        assert box[0] < np.asarray(cluster_frame.configuration.box)[0]

    def test_lattice_to_hoomd_box_preserves_geometry(self):
        # A triclinic lattice (rows = a, b, c) should map to a lower-triangular HOOMD box with the
        # same edge lengths and angles, and fractional coordinates must reconstruct consistently.
        lattice = np.array([[3.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.5, 0.7, 5.0]])
        box, hoomd_matrix = LatticeBuilder._lattice_to_hoomd_box(lattice)
        # Lower triangular (a along x, b in xy-plane).
        assert np.isclose(hoomd_matrix[0, 1], 0.0) and np.isclose(hoomd_matrix[0, 2], 0.0)
        assert np.isclose(hoomd_matrix[1, 2], 0.0)
        # Edge lengths and pairwise angles are preserved by the rotation.
        for original, mapped in zip(lattice, hoomd_matrix):
            assert np.isclose(np.linalg.norm(original), np.linalg.norm(mapped))
        original_gram = lattice @ lattice.T
        mapped_gram = hoomd_matrix @ hoomd_matrix.T
        assert np.allclose(original_gram, mapped_gram)
        # The GSD box tilt factors are consistent with get_cell_from_box.
        assert np.allclose(get_cell_from_box(box), hoomd_matrix)


if __name__ == '__main__':
    pytest.main([__file__])
