"""Tests for the symmetry-aware anisotropic lattice scaling of LatticeBuilder.

A uniform scale factor preserves the CIF's axial ratios, which come from the atomic crystal and are
generally wrong for a colloidal crystal of the same structure type (the ideal ratios depend on the
particle radii). These tests cover the opt-in anisotropic scan that relaxes each symmetry-distinct
lattice vector on its own.
"""
import os
import textwrap

import numpy as np
import pytest
from openmm import unit

from colloids.units import electric_potential_unit
from pymatgen.core import Structure

from colloids.colloids_create.configuration_generators.lattice_builder import LatticeBuilder

# Atomic AlB2: hexagonal P6/mmm, c/a = 3.26254100 / 2.99140213 = 1.0906. With 200 nm / 120 nm
# colloids that ratio strands the attractive Al-B pairs apart while the like-charge B-B pairs jam.
ALB2_CIF = textwrap.dedent("""\
    data_AlB2
    _symmetry_space_group_name_H-M   'P 1'
    _cell_length_a   2.99140213
    _cell_length_b   2.99140213
    _cell_length_c   3.26254100
    _cell_angle_alpha   90.00000000
    _cell_angle_beta   90.00000000
    _cell_angle_gamma   120.00000000
    _symmetry_Int_Tables_number   1
    loop_
     _symmetry_equiv_pos_site_id
     _symmetry_equiv_pos_as_xyz
      1  'x, y, z'
    loop_
     _atom_site_type_symbol
     _atom_site_label
     _atom_site_symmetry_multiplicity
     _atom_site_fract_x
     _atom_site_fract_y
     _atom_site_fract_z
     _atom_site_occupancy
      Al  Al0  1  0.00000000  0.00000000  0.00000000  1
      B  B1  1  0.33333333  0.66666667  0.50000000  1
      B  B2  1  0.66666667  0.33333333  0.50000000  1
    """)

# Simple cubic CsCl-like cell: a = b = c, so symmetry forces a single scale parameter and the
# anisotropic search must reduce exactly to the uniform one.
CUBIC_CIF = textwrap.dedent("""\
    data_CsCl
    _symmetry_space_group_name_H-M   'P 1'
    _cell_length_a   4.00000000
    _cell_length_b   4.00000000
    _cell_length_c   4.00000000
    _cell_angle_alpha   90.00000000
    _cell_angle_beta   90.00000000
    _cell_angle_gamma   90.00000000
    _symmetry_Int_Tables_number   1
    loop_
     _symmetry_equiv_pos_site_id
     _symmetry_equiv_pos_as_xyz
      1  'x, y, z'
    loop_
     _atom_site_type_symbol
     _atom_site_label
     _atom_site_symmetry_multiplicity
     _atom_site_fract_x
     _atom_site_fract_y
     _atom_site_fract_z
     _atom_site_occupancy
      Cs  Cs0  1  0.00000000  0.00000000  0.00000000  1
      Cl  Cl1  1  0.50000000  0.50000000  0.50000000  1
    """)

# Orthorhombic cell: all three axes are symmetry distinct, so all three must be free.
ORTHO_CIF = CUBIC_CIF.replace("_cell_length_b   4.00000000", "_cell_length_b   6.00000000").replace(
    "_cell_length_c   4.00000000", "_cell_length_c   9.00000000")

RUN_YAML = textwrap.dedent("""\
    brush_length: !Quantity {unit: nanometer, value: 10.0}
    brush_density: !Quantity {unit: /(nanometer**2), value: 0.09}
    cutoff_factor: 21.0
    debye_length: !Quantity {unit: nanometer, value: 8.0}
    dielectric_constant: 80.0
    initial_configuration: first_frame.gsd
    integrator: LangevinMiddleIntegrator
    integrator_parameters:
      frictionCoeff: !Quantity {unit: /picosecond, value: 1.0}
      randomNumberSeed: 7
      stepSize: !Quantity {unit: picosecond, value: 0.2}
      temperature: !Copy {key: potential_temperature}
    minimize_energy_initially: false
    platform_name: CPU
    potential_temperature: !Quantity {unit: kelvin, value: 300.0}
    run_steps: 10
    state_data_filename: state_data.csv
    state_data_interval: 10
    steric_radius_average: harmonic
    trajectory_filename: trajectory.gsd
    trajectory_interval: 10
    use_depletion: false
    use_gravity: false
    velocity_seed: 11
    wall_directions: [false, false, false]
    """)


def _write(tmp_path, cif_text, cif_name):
    (tmp_path / cif_name).write_text(cif_text)
    (tmp_path / "run.yaml").write_text(RUN_YAML)
    return str(tmp_path / cif_name), str(tmp_path / "run.yaml")


def _builder(tmp_path, cif_text, cif_name, *, cation, anion, repeats, **kwargs):
    cif_path, run_path = _write(tmp_path, cif_text, cif_name)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return LatticeBuilder(
            masses={cation: 1.0 * unit.dalton, anion: 1.0 * unit.dalton},
            radii={cation: 200.0 * unit.nanometer, anion: 120.0 * unit.nanometer},
            surface_potentials={cation: 40.0 * electric_potential_unit, anion: -60.0 * electric_potential_unit},
            lattice_specification=os.path.basename(cif_path), lattice_repeats=repeats,
            run_parameters_file=os.path.basename(run_path),
            radii_padding=5.0 * unit.nanometer, lattice_padding=0.0 * unit.nanometer,
            periodic=True, **kwargs)
    finally:
        os.chdir(cwd)


class TestAxisGroups(object):
    """The grouping must follow the crystal system, not the lengths that happen to be in the file."""

    @pytest.mark.parametrize("cif_text,expected", [
        (CUBIC_CIF, [[0, 1, 2]]),        # cubic: one free parameter
        (ALB2_CIF, [[0, 1], [2]]),       # hexagonal: a = b, c free
        (ORTHO_CIF, [[0], [1], [2]]),    # orthorhombic: all free
    ])
    def test_grouping_follows_crystal_system(self, tmp_path, cif_text, expected):
        (tmp_path / "s.cif").write_text(cif_text)
        structure = Structure.from_file(str(tmp_path / "s.cif"))
        assert LatticeBuilder._axis_groups(structure) == expected

    def test_symmetry_is_detected_from_coordinates_not_the_header(self, tmp_path):
        # Every CIF above declares P1. The grouping must still reflect the real symmetry, because
        # symmetry-expanded CIFs are routinely written out in P1.
        (tmp_path / "s.cif").write_text(ALB2_CIF)
        structure = Structure.from_file(str(tmp_path / "s.cif"))
        assert "P 1" in ALB2_CIF
        assert LatticeBuilder._axis_groups(structure) == [[0, 1], [2]]


class TestAnisotropicScaling(object):
    def test_rejected_without_optimize_energy(self, tmp_path):
        with pytest.raises(ValueError, match="anisotropic_energy"):
            _builder(tmp_path, CUBIC_CIF, "cscl.cif", cation="Cs", anion="Cl", repeats=2,
                     optimize_energy=False, anisotropic_energy=True)

    def test_cubic_is_unchanged_by_the_flag(self, tmp_path):
        # Symmetry gives a cubic cell a single parameter, so the anisotropic search must reproduce
        # the uniform one exactly. This is the regression guard for existing cubic studies.
        isotropic = _builder(tmp_path, CUBIC_CIF, "cscl.cif", cation="Cs", anion="Cl", repeats=2,
                             optimize_energy=True, energy_scale_samples=12).generate_configuration()
        anisotropic = _builder(tmp_path, CUBIC_CIF, "cscl.cif", cation="Cs", anion="Cl", repeats=2,
                               optimize_energy=True, energy_scale_samples=12,
                               anisotropic_energy=True).generate_configuration()
        assert np.allclose(isotropic.configuration.box, anisotropic.configuration.box)
        assert np.allclose(isotropic.particles.position, anisotropic.particles.position)

    def test_hexagonal_axial_ratio_is_relaxed_away_from_the_atomic_value(self, tmp_path):
        # The whole point: c/a must be free to leave the value baked into the CIF.
        frame = _builder(tmp_path, ALB2_CIF, "alb2.cif", cation="Al", anion="B", repeats=4,
                         optimize_energy=True, energy_scale_samples=20,
                         anisotropic_energy=True).generate_configuration()
        box = np.asarray(frame.configuration.box, dtype=float)
        axial_ratio = (box[2] / 4.0) / (box[0] / 4.0)
        assert not np.isclose(axial_ratio, 1.0906, atol=0.02), "c/a stayed at the atomic value"
        assert 0.85 < axial_ratio < 1.05

    def test_a_and_b_stay_equal_for_a_hexagonal_cell(self, tmp_path):
        # Scaling must not lower the symmetry of the cell.
        frame = _builder(tmp_path, ALB2_CIF, "alb2.cif", cation="Al", anion="B", repeats=4,
                         optimize_energy=True, energy_scale_samples=20,
                         anisotropic_energy=True).generate_configuration()
        box = np.asarray(frame.configuration.box, dtype=float)
        # For the 120-degree cell the HOOMD box carries the b vector as (xy * ly, ly); its length,
        # not the ly component alone, is what must match |a|.
        length_a = box[0]
        length_b = np.hypot(box[3] * box[1], box[1])
        assert np.isclose(length_a, length_b, rtol=1e-6)

    def test_anisotropic_reaches_a_lower_energy_than_uniform(self, tmp_path):
        # Relaxing c/a must not make the lattice worse; for AlB2 at this size ratio it is much better.
        builder_kwargs = dict(cation="Al", anion="B", repeats=4, optimize_energy=True,
                              energy_scale_samples=20)
        uniform = _builder(tmp_path, ALB2_CIF, "alb2.cif", **builder_kwargs)
        relaxed = _builder(tmp_path, ALB2_CIF, "alb2.cif", anisotropic_energy=True, **builder_kwargs)
        structure_full = uniform._structure.make_supercell(4, in_place=False)
        types = [uniform._type_map[z] for z in structure_full.atomic_numbers]
        energy_of = LatticeBuilder._vacuum_energy_function(
            types=types, radii=uniform._radii, surface_potentials=uniform._surface_potentials,
            masses=uniform._masses, run_parameters=uniform._run_parameters)

        def energy_of_frame(frame, builder):
            positions = np.asarray(frame.particles.position, dtype=float)
            return energy_of(positions)

        uniform_energy = energy_of_frame(uniform.generate_configuration(), uniform)
        relaxed_energy = energy_of_frame(relaxed.generate_configuration(), relaxed)
        assert relaxed_energy < uniform_energy


if __name__ == '__main__':
    pytest.main([__file__])
