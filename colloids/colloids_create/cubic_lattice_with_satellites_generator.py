from enum import auto, Enum
from math import acos, cos, pi, sin, sqrt
from typing import Iterator
from ase import Atom, build
import numpy as np
import numpy.typing as npt
from openmm import unit
from colloids.colloids_create import ConfigurationGenerator


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

    def __init__(self, filename: str, lattice: CubicLattice, lattice_constant: unit.Quantity, lattice_repeats: int,
                 orbit_distance: unit.Quantity, satellites_per_center: int, type_lattice: str,
                 type_satellite: str) -> None:
        super().__init__(filename)
        if not lattice_constant.unit.is_compatible(self._nanometer):
            raise TypeError("The lattice constant must have a unit that is compatible with nanometers.")
        if not lattice_constant.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("The lattice constant must have a value greater than zero.")
        if not lattice_repeats > 0:
            raise ValueError("The number of lattice repeats must be greater than zero.")
        if not orbit_distance.unit.is_compatible(self._nanometer):
            raise TypeError("The orbit distance must have a unit that is compatible with nanometers.")
        if not orbit_distance.value_in_unit(self._nanometer) > 0.0:
            raise ValueError("The orbit distance must have a value greater than zero.")
        if not satellites_per_center >= 0:
            raise ValueError("The number of satellites per center must be greater than or equal to zero.")
        if not orbit_distance < lattice_constant:
            raise ValueError("The orbit distance must be smaller than the lattice constant.")
        self._lattice = lattice
        self._lattice_constant = lattice_constant
        self._lattice_repeats = lattice_repeats
        self._orbit_distance = orbit_distance
        self._satellites_per_center = satellites_per_center
        self._type_lattice = type_lattice
        self._type_satellite = type_satellite

    @staticmethod
    def _generate_fibonacci_sphere_grid_points(number_points: int, radius: float) -> Iterator[npt.NDArray[np.floating]]:
        # See https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
        # Output, real xgB(3,ng): the grid points.
        golden_ratio = (1.0 + sqrt(5.0)) / 2.0
        epsilon = 0.36
        for i in range(number_points):
            theta = 2.0 * pi * i / golden_ratio
            phi = acos(1.0 - 2.0 * (i + epsilon) / (number_points - 1.0 + 2.0 * epsilon))
            yield np.array([cos(theta) * sin(phi) * radius, sin(theta) * sin(phi) * radius, cos(phi) * radius])

    def write_positions(self) -> None:
        # Use X as the atom name to avoid a clash with an existing chemical symbol.
        atoms = build.bulk(name="X", crystalstructure=self._lattice.to_ase_string(),
                           a=self._lattice_constant.value_in_unit(self._nanometer),
                           cubic=True)
        new_atoms = []
        for atom in atoms:
            for satellite_position in self._generate_fibonacci_sphere_grid_points(
                    self._satellites_per_center, self._orbit_distance.value_in_unit(self._nanometer)):
                new_atoms.append(Atom(symbol="Y", position=atom.position + satellite_position))
        for new_atom in new_atoms:
            atoms.append(new_atom)
        atoms = atoms.repeat(self._lattice_repeats)
        # Use the extended xyz file format.
        # See https://www.ovito.org/docs/current/reference/file_formats/input/xyz.html#extended-xyz-format
        with open(self._filename, "w") as file:
            print(len(atoms), file=file)
            print(f"Lattice=\"{' '.join(map(str, atoms.cell.flatten()))}\" Properties=species:S:1:pos:R:3", file=file)
            for atom in atoms:
                print(f"{self._type_lattice if atom.symbol=='X' else self._type_satellite} "
                      f"{atom.position[0]} {atom.position[1]} {atom.position[2]}", file=file)


if __name__ == '__main__':
    CubicLattice.from_string("sc")
    CubicLatticeWithSatellitesGenerator("test_sc.xyz", CubicLattice.SC, 4.05 * (unit.nano * unit.meter),
                                        3, 1.3 * (unit.nano * unit.meter), 1,
                                        "P", "N").write_positions()

    CubicLatticeWithSatellitesGenerator("test_fcc.xyz", CubicLattice.FCC, 4.05 * (unit.nano * unit.meter),
                                        3, 1.3 * (unit.nano * unit.meter), 1,
                                        "P", "N").write_positions()

    CubicLatticeWithSatellitesGenerator("test_bcc.xyz", CubicLattice.BCC, 4.05 * (unit.nano * unit.meter),
                                        3, 1.3 * (unit.nano * unit.meter), 1,
                                        "P", "N").write_positions()
