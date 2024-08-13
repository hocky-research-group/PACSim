from enum import auto, Enum
from typing import Union
from ase import Atom, build
import numpy as np
from openmm import unit
from colloids.colloids_create import ConfigurationGenerator
from colloids.helper_functions import generate_fibonacci_sphere_grid_points


class CubicLattice(Enum):
    # TODO: Add docstrings.
    SC = auto()
    FCC = auto()
    BCC = auto()

    def to_ase_string(self):
        return self.name.lower()

    @staticmethod
    def from_string(string: str):
        return CubicLattice[string.upper()]


class CubicLatticeWithSatellitesGenerator(ConfigurationGenerator):

    _nanometer = unit.nano * unit.meter

    def __init__(self, filename: str, lattice: CubicLattice, lattice_constant: unit.Quantity,
                 lattice_repeats: Union[int, list[int]], orbit_distance: unit.Quantity,
                 padding_distance: unit.Quantity, satellites_per_center: int, type_lattice: str,
                 type_satellite: str) -> None:
        super().__init__(filename)
        if not lattice_constant.unit.is_compatible(self._nanometer):
            raise TypeError("The lattice constant must have a unit that is compatible with nanometers.")
        if not lattice_constant.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("The lattice constant must have a value greater than zero.")
        if isinstance(lattice_repeats, int):
            if not lattice_repeats > 0:
                raise ValueError("The number of lattice repeats must be greater than zero.")
        else:
            if not (isinstance(lattice_repeats, list)
                    and all(isinstance(repeat, int) for repeat in lattice_repeats)
                    and len(lattice_repeats) == 3):
                raise TypeError("The lattice repeats must be an integer or a list of three integers.")
            if not all(repeat > 0 for repeat in lattice_repeats):
                raise ValueError("All lattice repeats must be positive.")
        if not orbit_distance.unit.is_compatible(self._nanometer):
            raise TypeError("The orbit distance must have a unit that is compatible with nanometers.")
        if not orbit_distance.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("The orbit distance must have a value greater than zero.")
        if not padding_distance.unit.is_compatible(self._nanometer):
            raise TypeError("The padding distance must have a unit that is compatible with nanometers.")
        if not padding_distance.value_in_unit(self._nanometer) >= 0.0:
            raise ValueError("The padding distance must have a value greater than or equal to zero.")
        if not satellites_per_center >= 0:
            raise ValueError("The number of satellites per center must be greater than or equal to zero.")
        if not orbit_distance < lattice_constant:
            raise ValueError("The orbit distance must be smaller than the lattice constant.")
        self._lattice = lattice
        self._lattice_constant = lattice_constant
        self._lattice_repeats = lattice_repeats
        self._orbit_distance = orbit_distance
        self._padding_distance = padding_distance
        self._satellites_per_center = satellites_per_center
        self._type_lattice = type_lattice
        self._type_satellite = type_satellite

    def write_positions(self) -> None:
        # Use X as the atom name to avoid a clash with an existing chemical symbol.
        atoms = build.bulk(name="X", crystalstructure=self._lattice.to_ase_string(),
                           a=self._lattice_constant.value_in_unit(self._nanometer),
                           cubic=True)
        # Center the center atoms around the origin.
        atoms.center(about=(0.0, 0.0, 0.0))
        new_atoms = []
        for atom in atoms:
            # Tag for centers.
            atom.tag = 0
            for satellite_position in generate_fibonacci_sphere_grid_points(
                    self._satellites_per_center, self._orbit_distance.value_in_unit(self._nanometer),
                    False):
                new_atoms.append(Atom(symbol="X", position=atom.position + satellite_position, tag=1))
        for new_atom in new_atoms:
            atoms.append(new_atom)
        atoms = atoms.repeat(self._lattice_repeats)
        # Shift all atoms so that the center atoms are centered around the origin again.
        lattice_repeats = (self._lattice_repeats if isinstance(self._lattice_repeats, list)
                           else [self._lattice_repeats, self._lattice_repeats, self._lattice_repeats])
        assert len(lattice_repeats) == len(atoms.cell)
        translation_vector = sum(-(lr - 1) * cv / (2.0 * lr) for cv, lr in zip(atoms.cell, lattice_repeats))
        atoms.translate(translation_vector)
        # Use the extended xyz file format.
        # See https://www.ovito.org/docs/current/reference/file_formats/input/xyz.html#extended-xyz-format
        scaled_cell = atoms.cell.copy()
        for i, cell_vector in enumerate(scaled_cell):
            norm = np.linalg.norm(cell_vector)
            scaling_factor = (norm + 2.0 * self._padding_distance.value_in_unit(self._nanometer)) / norm
            scaled_cell[i] *= scaling_factor
        origin_vector = -0.5 * scaled_cell.sum(axis=0)
        with open(self._filename, "w") as file:
            print(len(atoms), file=file)
            print(f"Lattice=\"{' '.join(map(str, scaled_cell.flatten()))}\" Properties=species:S:1:pos:R:3 "
                  f"Origin=\"{' '.join(map(str, origin_vector))}\"",
                  file=file)
            tag_to_type_dictionary = {0: self._type_lattice, 1: self._type_satellite}
            for atom in atoms:
                print(f"{tag_to_type_dictionary[atom.tag]} "
                      f"{atom.position[0]} {atom.position[1]} {atom.position[2]}", file=file)


if __name__ == '__main__':
    CubicLattice.from_string("sc")
    CubicLatticeWithSatellitesGenerator("test_sc.xyz", CubicLattice.SC, 4.05 * (unit.nano * unit.meter),
                                        3, 1.3 * (unit.nano * unit.meter), 3.0 * (unit.nano * unit.meter),
                                        1, "P", "N").write_positions()

    CubicLatticeWithSatellitesGenerator("test_fcc.xyz", CubicLattice.FCC, 4.05 * (unit.nano * unit.meter),
                                        3, 1.3 * (unit.nano * unit.meter), 3.0 * (unit.nano * unit.meter),
                                        1, "P", "N").write_positions()

    CubicLatticeWithSatellitesGenerator("test_bcc.xyz", CubicLattice.BCC, 4.05 * (unit.nano * unit.meter),
                                        3, 1.3 * (unit.nano * unit.meter), 3.0 * (unit.nano * unit.meter),
                                        1, "P", "N").write_positions()
