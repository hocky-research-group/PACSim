"""
Regression tests for the Frenkel-Ladd free-energy analysis (``free_energy.py``).

The A0 test reproduces the analytic ideal-Einstein-crystal free energies of Table 1 of
Vega et al., J. Phys.: Condens. Matter 20, 153101 (2008) (fcc hard spheres, Lambda_dB = sigma) to
better than 1e-3 N kB T, which pins the exact form of eq. (48).

Run with:  pytest test_free_energy.py
"""
import math
import numpy as np
import pytest
from openmm import unit
from free_energy import (ideal_einstein_free_energy_per_particle, delta_a1_per_particle,
                         FrenkelLaddSchedule, FreeEnergyResult)
from colloids.units import energy_unit, length_unit, temperature_unit

_KB = (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).value_in_unit(energy_unit / temperature_unit)
# Temperature such that kB T = 1 kJ/mol, matching the reduced units of Table 1 (kT = 1).
_UNIT_KT_TEMPERATURE = (1.0 / _KB) * temperature_unit
_spring_constant_unit = energy_unit / (length_unit ** 2)

# (N, rho*, Lambda_E [kT/sigma^2]) -> A0/(N kB T) from Table 1 (Einstein-crystal column).
_TABLE_1 = [
    (108, 1.04086, 632.026, 7.8180),
    (256, 1.04086, 632.026, 7.8929),
    (1372, 1.04086, 1000.00, 8.6304),
    (2048, 1.04086, 1000.00, 8.6347),
]


@pytest.mark.parametrize("n, rho, lambda_e, expected", _TABLE_1)
def test_ideal_einstein_free_energy_matches_table_1(n, rho, lambda_e, expected):
    volume = (n / rho) * length_unit ** 3
    a0 = ideal_einstein_free_energy_per_particle(
        n_particles=n, volume=volume, temperature=_UNIT_KT_TEMPERATURE,
        spring_constant=lambda_e * _spring_constant_unit,
        debroglie_wavelength=1.0 * length_unit)
    assert abs(a0 - expected) < 1.0e-3


def test_debroglie_wavelength_shifts_by_expected_constant():
    # A0/(N kT) contains +3 ln(Lambda_dB) (from the 1/Lambda_dB^{3N} factor), so doubling Lambda_dB
    # raises A0 by +3 ln 2.
    kwargs = dict(n_particles=256, volume=200.0 * length_unit ** 3,
                  temperature=_UNIT_KT_TEMPERATURE, spring_constant=1000.0 * _spring_constant_unit)
    a0_one = ideal_einstein_free_energy_per_particle(debroglie_wavelength=1.0 * length_unit, **kwargs)
    a0_two = ideal_einstein_free_energy_per_particle(debroglie_wavelength=2.0 * length_unit, **kwargs)
    assert np.isclose(a0_two - a0_one, 3.0 * math.log(2.0))


def test_delta_a1_reduces_to_lattice_energy_for_rigid_lattice():
    # If every sampled U_sol equals U_lattice (infinitely stiff springs), dA1/(NkT) = U_lattice/(NkT).
    n = 100
    u_lattice = -6.0 * n * energy_unit  # kB T = 1 kJ/mol here, so this is -6 N kT.
    samples = np.full(500, -6.0 * n)
    delta_a1 = delta_a1_per_particle(samples * energy_unit, u_lattice, n, _UNIT_KT_TEMPERATURE)
    assert np.isclose(delta_a1, -6.0)


def test_frenkel_ladd_schedule_recovers_constant_integrand():
    # If <U_Einstein(bare)> is constant g0 over all s in (0,1], dA2/(NkT) = -g0/(N kT).
    n = 256
    schedule = FrenkelLaddSchedule(n_points=15, temperature=_UNIT_KT_TEMPERATURE,
                                   spring_constant=1000.0 * _spring_constant_unit)
    couplings = schedule.couplings()
    assert np.all(couplings > 0.0) and np.all(couplings <= 1.0 + 1e-9)
    g0 = 400.0  # kJ/mol (= 400 kT)
    delta_a2 = schedule.integrate_delta_a2_per_particle([g0 * energy_unit] * len(couplings), n)
    assert np.isclose(delta_a2, -g0 / n, rtol=1e-6)


def test_free_energy_result_assembly_and_finite_size_term():
    result = FreeEnergyResult(a0=13.61, delta_a1=-3.14, delta_a2=-7.36, n_particles=256)
    assert np.isclose(result.a_sol, 13.61 - 3.14 - 7.36)
    assert np.isclose(result.a_sol_frenkel_ladd, result.a_sol + 2.0 * math.log(256) / 256)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
