from enum import auto, Enum
from typing import Union
from ase import Atom, build
from gsd.hoomd import Frame
import numpy as np
from openmm import unit
from colloids.colloids_create import ConfigurationGenerator
from colloids.colloids_create.helper_functions import generate_fibonacci_sphere_grid_points


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
    _lattice_tag = 0
    _satellite_tag = 1

    def __init__(self, lattice: CubicLattice, lattice_constant: unit.Quantity, lattice_repeats: Union[int, list[int]],
                 orbit_distance: unit.Quantity, padding_distance: unit.Quantity, satellites_per_center: int,
                 type_lattice: str, type_satellite: str) -> None:
        super().__init__()
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

    def generate_configuration(self) -> Frame:
        # Use X as the atom name to avoid a clash with an existing chemical symbol.
        atoms = build.bulk(name="X", crystalstructure=self._lattice.to_ase_string(),
                           a=self._lattice_constant.value_in_unit(self._nanometer),
                           cubic=True)
        # Center the center atoms around the origin.
        atoms.center(about=(0.0, 0.0, 0.0))
        new_atoms = []
        for atom in atoms:
            atom.tag = self._lattice_tag
            for satellite_position in generate_fibonacci_sphere_grid_points(
                    self._satellites_per_center, self._orbit_distance.value_in_unit(self._nanometer),
                    False):
                new_atoms.append(Atom(symbol="X", position=atom.position + satellite_position,
                                      tag=self._satellite_tag))
        for new_atom in new_atoms:
            atoms.append(new_atom)
        atoms = atoms.repeat(self._lattice_repeats)
        # Shift all atoms so that the center atoms are centered around the origin again.
        lattice_repeats = (self._lattice_repeats if isinstance(self._lattice_repeats, list)
                           else [self._lattice_repeats, self._lattice_repeats, self._lattice_repeats])
        assert len(lattice_repeats) == len(atoms.cell)
        translation_vector = sum(-(lr - 1) * cv / (2.0 * lr) for cv, lr in zip(atoms.cell, lattice_repeats))
        atoms.translate(translation_vector)
        # Scale the cell vectors.
        for i, cell_vector in enumerate(atoms.cell[:]):
            norm = np.linalg.norm(cell_vector)
            scaling_factor = (norm + 2.0 * self._padding_distance.value_in_unit(self._nanometer)) / norm
            atoms.cell[i] *= scaling_factor

        frame = Frame()
        frame.particles.N = len(atoms)
        frame.particles.position = atoms.positions.astype(np.float32)
        assert all(atom.tag == self._lattice_tag or atom.tag == self._satellite_tag for atom in atoms)
        frame.particles.types = (self._type_lattice, self._type_satellite)
        frame.particles.typeid = np.array([0 if atom.tag == self._lattice_tag else 1 for atom in atoms],
                                          dtype=np.uint32)
        # See http://docs.openmm.org/7.6.0/userguide/theory/05_other_features.html
        # See https://hoomd-blue.readthedocs.io/en/v2.9.3/box.html
        frame.configuration.box = np.array([atoms.cell[0][0], atoms.cell[1][1], atoms.cell[2][2],
                                            atoms.cell[1][0] / atoms.cell[1][1], atoms.cell[2][0] / atoms.cell[2][2],
                                            atoms.cell[2][1] / atoms.cell[2][2]], dtype=np.float32)
        return frame
