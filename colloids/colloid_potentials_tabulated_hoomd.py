import math
import hoomd
import hoomd.md
import numpy as np
import numpy.typing as npt
from openmm import unit
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters


class ColloidPotentialsTabulatedHoomd(object):

    _nanometer = unit.nano * unit.meter
    _millivolt = unit.milli * unit.volt

    def __init__(self, radius_one: float, radius_two: float, surface_potential_one: float, surface_potential_two: float,
                 type_one: str, type_two: str, colloid_potentials_parameters: ColloidPotentialsParameters,
                 neighbor_list: hoomd.md.nlist.nlist, shift: bool = True):
        self._parameters = colloid_potentials_parameters
        self._radius_one = radius_one
        self._radius_two = radius_two
        self._surface_potential_one = surface_potential_one
        self._surface_potential_two = surface_potential_two
        self._type_one = type_one
        self._type_two = type_two
        self._maximum_surface_separation = 20.0 * self._parameters.debye_length.value_in_unit(self._nanometer)
        self._number_samples = 5000
        self._neighbor_list = neighbor_list
        self._shift = shift
        self._tabulated_potential = self._set_up_tabulated_potential()

    def _steric_potential(
            self, prefactor: float, h_values: npt.NDArray[np.floating]) -> (npt.NDArray[np.floating],
                                                                            npt.NDArray[np.floating]):
        """
        Return the steric potential from the Alexander-de Gennes polymer brush model for the given surface-to-surface
        separations.
        """
        double_brush_length = 2.0 * self._parameters.brush_length.value_in_unit(self._nanometer)
        h_over_double_brush_length = h_values / double_brush_length
        double_brush_length_over_h = double_brush_length / h_values
        return (prefactor * np.where(h_values <= double_brush_length,
                                     28.0 * (np.power(double_brush_length_over_h, 0.25) - 1.0)
                                     + 20.0 / 11.0 * (1.0 - np.power(h_over_double_brush_length, 2.75))
                                     + 12.0 * (h_over_double_brush_length - 1.0),
                                     0.0),
                -prefactor / double_brush_length * np.where(h_values <= double_brush_length,
                                                            -7.0 * np.power(double_brush_length_over_h, 1.25)
                                                            - 5.0 * np.power(h_over_double_brush_length, 1.75)
                                                            + 12.0, 0.0))

    def _electrostatic_potential(
            self, prefactor: float, h_values: npt.NDArray[np.floating], h_cut: float) -> (npt.NDArray[np.floating],
                                                                                          npt.NDArray[np.floating]):
        """Return the electrostatic potential from DLVO theory for the given surface-to-surface separations."""
        debye_length = self._parameters.debye_length.value_in_unit(self._nanometer)
        if self._shift:
            a = prefactor / debye_length * np.exp(-h_cut / debye_length)
            b = -prefactor * np.exp(-h_cut / debye_length) - h_cut * a
            return (prefactor * np.exp(-h_values / debye_length) + a * h_values + b,
                    prefactor / debye_length * np.exp(-h_values / debye_length) - a)
        else:
            return (prefactor * np.exp(-h_values / debye_length),
                    prefactor / debye_length * np.exp(-h_values / debye_length))

    def _potential(self, r, _, rmax, steric_prefactor, electrostatic_prefactor, radius_one, radius_two):
        h = r - (radius_one + radius_two)
        h_cut = rmax - (radius_one + radius_two)
        steric_potential, steric_force = self._steric_potential(steric_prefactor, h)
        electrostatic_potential, electrostatic_force = self._electrostatic_potential(electrostatic_prefactor, h, h_cut)
        return steric_potential + electrostatic_potential, steric_force + electrostatic_force

    def _set_up_tabulated_potential(self):
        steric_prefactor_11 = (
                unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature * 16.0 * math.pi *
                self._radius_one * self._nanometer *
                (self._parameters.brush_length ** 2) * (self._parameters.brush_density ** (3 / 2)) / 35.0
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        steric_prefactor_22 = (
                unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature * 16.0 * math.pi *
                self._radius_two * self._nanometer *
                (self._parameters.brush_length ** 2) * (self._parameters.brush_density ** (3 / 2)) / 35.0
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        steric_prefactor_12 = (
                unit.BOLTZMANN_CONSTANT_kB * self._parameters.temperature * 16.0 * math.pi *
                (self._radius_one + self._radius_two) / 2.0 * self._nanometer *
                (self._parameters.brush_length ** 2) * (self._parameters.brush_density ** (3 / 2)) / 35.0
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)

        electrostatic_prefactor_11 = (
                2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
                * self._radius_one * self._surface_potential_one * self._surface_potential_one
                * self._nanometer * self._millivolt ** 2
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        electrostatic_prefactor_22 = (
                2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
                * self._radius_two * self._surface_potential_two * self._surface_potential_two
                * self._nanometer * self._millivolt ** 2
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        electrostatic_prefactor_12 = (
                2.0 * math.pi * self._parameters.VACUUM_PERMITTIVITY * self._parameters.dielectric_constant
                * 2.0 / (1.0 / self._radius_one + 1.0 / self._radius_two)
                * self._surface_potential_one * self._surface_potential_two
                * self._nanometer * self._millivolt ** 2
                * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)

        potential = hoomd.md.pair.table(width=self._number_samples, nlist=self._neighbor_list)
        potential.pair_coeff.set(self._type_one, self._type_one, func=self._potential,
                                 rmin=1.00005 * 2.0 * self._radius_one,
                                 rmax=2.0 * self._radius_one + self._maximum_surface_separation,
                                 coeff=dict(steric_prefactor=steric_prefactor_11,
                                            electrostatic_prefactor=electrostatic_prefactor_11,
                                            radius_one=self._radius_one, radius_two=self._radius_one))
        potential.pair_coeff.set(self._type_two, self._type_two, func=self._potential,
                                 rmin=1.00005 * 2.0 * self._radius_two,
                                 rmax=2.0 * self._radius_two + self._maximum_surface_separation,
                                 coeff=dict(steric_prefactor=steric_prefactor_22,
                                            electrostatic_prefactor=electrostatic_prefactor_22,
                                            radius_one=self._radius_two, radius_two=self._radius_two))
        potential.pair_coeff.set(self._type_one, self._type_two, func=self._potential,
                                 rmin=1.00005 * (self._radius_one + self._radius_two),
                                 rmax=self._radius_one + self._radius_two + self._maximum_surface_separation,
                                 coeff=dict(steric_prefactor=steric_prefactor_12,
                                            electrostatic_prefactor=electrostatic_prefactor_12,
                                            radius_one=self._radius_one, radius_two=self._radius_two))
        return potential

    def yield_potentials(self):
        yield self._tabulated_potential
