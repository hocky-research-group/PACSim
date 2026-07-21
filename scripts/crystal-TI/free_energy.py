"""
Frenkel-Ladd / Einstein-crystal free-energy analysis for PACSim crystals.

This module computes the Helmholtz free energy of a solid from thermodynamic-integration data
produced by PACSim runs that include the ``HarmonicRestraint`` (Einstein) force. It follows the
Einstein-crystal (fixed-center-of-mass) bookkeeping of

    C. Vega, E. Sanz, J. L. F. Abascal and E. G. Noya,
    "Determination of phase diagrams via computer simulation", J. Phys.: Condens. Matter 20,
    153101 (2008)  [arXiv:0901.1823],

with the original method from D. Frenkel and A. J. C. Ladd, J. Chem. Phys. 81, 3188 (1984).

Equation numbers below refer to the Vega et al. (2008) review. The convention for the Einstein
spring energy matches PACSim's ``HarmonicRestraint`` force:

    U_Einstein(bare) = Lambda_E * sum_i |r_i - r_i^0|^2        (no factor of 1/2)

so Lambda_E has units of energy / length^2.

The total free energy is (eq. 47/48)

    A_sol = A0 + dA1 + dA2

with, for an atomic solid and thermal de Broglie wavelength Lambda_dB,

    A0/(N kT) = -(1/N) [ (3(N-1)/2) ln(pi/(beta Lambda_E)) + (1/2) ln N + ln(V/Lambda_dB^3) ]   (eq. 48)

COM bookkeeping (Vega eqs. 47-48): A0 combines the FIXED-center-of-mass ideal Einstein crystal with
the term that RELEASES the constraint (the ln V term is the released-COM translational volume -- a
truly fixed-COM crystal has no extensive V dependence). dA1 and dA2 are *sampled* with fixed COM, but
the assembled A_sol = A0 + dA1 + dA2 is the free energy of the UNCONSTRAINED solid.

    dA1/(N kT) = U_lattice/(N kT) - (1/N) ln < exp[-beta (U_sol - U_lattice)] >_Einstein-ideal   (eq. 35)

    dA2/(N kT) = -(1/(N kT)) integral_0^1 < U_Einstein(bare) >_s ds                              (eq. 37/38)

where s in [0, 1] is the strength of the springs (s = ``lambda_ein`` of the HarmonicRestraint
force): s = 1 is the interacting Einstein crystal (full Lambda_E springs, with interactions on) and
s = 0 is the real solid (springs off, interactions on). The (1/2) ln N coefficient (which folds the
fixed-center-of-mass correction into the (3/2) ln N of a naive reading of eq. 48) is validated to
four decimal places against Table 1 of the review; see ``tests``.

The Frenkel-Ladd finite-size correction (eq. 59) is

    A_sol^FL/(N kT) = A_sol/(N kT) + (2/N) ln N .
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence
import numpy as np
import numpy.typing as npt
from openmm import unit
from scipy.special import logsumexp, roots_legendre

# Reuse PACSim's canonical unit definitions so this module composes with the rest of the package.
try:
    from colloids.units import energy_unit, length_unit, temperature_unit
except Exception:  # pragma: no cover - allow standalone use without the installed package
    energy_unit = unit.kilojoule_per_mole
    length_unit = unit.nano * unit.meter
    temperature_unit = unit.kelvin

_spring_constant_unit = energy_unit / (length_unit ** 2)

# Boltzmann constant in the PACSim molar unit system (kJ/mol/K).
_KB = (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).value_in_unit(energy_unit / temperature_unit)


def _kt(temperature: unit.Quantity) -> float:
    """Return k_B T in the PACSim molar energy unit (kJ/mol)."""
    return _KB * temperature.value_in_unit(temperature_unit)


def ideal_einstein_free_energy_per_particle(
        n_particles: int, volume: unit.Quantity, temperature: unit.Quantity,
        spring_constant: unit.Quantity,
        debroglie_wavelength: unit.Quantity = 1.0 * length_unit) -> float:
    """
    A0/(N kB T): fixed-COM ideal Einstein crystal PLUS the COM-release term (Vega eq. 48).

    This is a purely analytic quantity. The absolute value depends on the (arbitrary) choice of the
    thermal de Broglie wavelength ``debroglie_wavelength``; this choice shifts every solid and
    fluid free energy by the same constant and therefore does not affect phase coexistence. The
    conventional choice (used for the Lennard-Jones benchmark) is ``debroglie_wavelength = sigma``,
    i.e. one length unit.

    :param n_particles: Number of restrained particles N.
    :param volume: Volume V of the simulation box (unit compatible with length^3).
    :param temperature: Temperature T (unit compatible with kelvin).
    :param spring_constant: Einstein spring constant Lambda_E (unit compatible with energy/length^2).
    :param debroglie_wavelength: Thermal de Broglie wavelength Lambda_dB (unit compatible with
        length). Defaults to one length unit.
    :return: A0 / (N kB T), dimensionless.
    """
    n = int(n_particles)
    beta = 1.0 / _kt(temperature)
    lambda_e = spring_constant.value_in_unit(_spring_constant_unit)
    v = volume.value_in_unit(length_unit ** 3)
    lambda_db = debroglie_wavelength.value_in_unit(length_unit)
    # ln of the (dimensionless) bracket of eq. (48): (1/Lambda_dB^{3N}) (pi/(beta Lambda_E))^{3(N-1)/2} N^{1/2} V.
    ln_bracket = (-3.0 * n * math.log(lambda_db)
                  + 1.5 * (n - 1) * math.log(math.pi / (beta * lambda_e))
                  + 0.5 * math.log(n)
                  + math.log(v))
    return -ln_bracket / n


def delta_a1_per_particle(u_sol_samples: npt.ArrayLike, u_lattice: unit.Quantity,
                          n_particles: int, temperature: unit.Quantity) -> float:
    """
    dA1/(N kB T): ideal -> interacting Einstein crystal (eq. 35), sampled with fixed center of mass.

    The samples ``u_sol_samples`` are values of the full interaction energy U_sol evaluated on
    configurations drawn from the *ideal* Einstein crystal (springs at full Lambda_E, interactions
    switched off during sampling). ``u_lattice`` is the interaction energy of the perfect lattice
    (used only to keep the exponential well-conditioned; the result is independent of its value).

    :param u_sol_samples: Interaction energies U_sol sampled in the ideal Einstein crystal
        (unit compatible with energy, or plain floats already in kJ/mol).
    :param u_lattice: Lattice interaction energy U_lattice (unit compatible with energy).
    :param n_particles: Number of particles N.
    :param temperature: Temperature T.
    :return: dA1 / (N kB T), dimensionless.
    """
    kt = _kt(temperature)
    beta = 1.0 / kt
    samples = np.asarray([u.value_in_unit(energy_unit) if unit.is_quantity(u) else float(u)
                          for u in np.atleast_1d(u_sol_samples)], dtype=float)
    u_lat = u_lattice.value_in_unit(energy_unit)
    m = samples.size
    # -kT ln < exp[-beta (U_sol - U_lattice)] >  using a numerically stable log-sum-exp.
    log_mean = logsumexp(-beta * (samples - u_lat)) - math.log(m)
    delta_a1 = u_lat - kt * log_mean
    return delta_a1 / (n_particles * kt)


@dataclass
class FrenkelLaddSchedule:
    """
    Gauss-Legendre thermodynamic-integration schedule for the dA2 spring integral (eqs. 37-39).

    The spring-strength coupling s = ``lambda_ein`` in [0, 1] is integrated after the change of
    variable s -> w = ln(s * kappa + c) recommended by Frenkel & Ladd, where
    kappa = beta * Lambda_E * l^2 is the dimensionless maximum spring strength (l = length unit)
    and c = exp(3.5). This makes the (otherwise multi-decade) integrand smooth and lets a modest
    number of Gauss-Legendre nodes integrate it accurately.

    :param n_points: Number of Gauss-Legendre nodes (10-20 is typical).
    :param temperature: Temperature T.
    :param spring_constant: Einstein spring constant Lambda_E.
    :param c: Offset constant in the logarithmic change of variable. Defaults to exp(3.5).
    """

    n_points: int
    temperature: unit.Quantity
    spring_constant: unit.Quantity
    c: float = math.exp(3.5)

    def __post_init__(self) -> None:
        if self.n_points < 2:
            raise ValueError("n_points must be at least 2.")
        beta = 1.0 / _kt(self.temperature)
        # l = length unit -> l^2 = 1 length_unit^2, so kappa is beta*Lambda_E in 1/length_unit^2.
        self._kappa = beta * self.spring_constant.value_in_unit(_spring_constant_unit)
        nodes, weights = roots_legendre(self.n_points)  # on [-1, 1]
        w_lo, w_hi = math.log(self.c), math.log(self._kappa + self.c)
        # Map Gauss-Legendre nodes/weights from [-1, 1] to [w_lo, w_hi].
        self._w = 0.5 * (w_hi - w_lo) * nodes + 0.5 * (w_hi + w_lo)
        self._gl_weights = 0.5 * (w_hi - w_lo) * weights
        # s(w) = (exp(w) - c) / kappa ; ds/dw = exp(w)/kappa.
        self._s = (np.exp(self._w) - self.c) / self._kappa
        self._ds_dw = np.exp(self._w) / self._kappa

    def couplings(self) -> npt.NDArray[np.float64]:
        """The ``lambda_ein`` values (spring strengths s in (0, 1]) to simulate, one per window."""
        return self._s.copy()

    def integrate_delta_a2_per_particle(self, mean_bare_spring_energy: Sequence[unit.Quantity],
                                        n_particles: int) -> float:
        """
        dA2/(N kB T) = -(1/(N kT)) integral_0^1 <U_Einstein(bare)>_s ds  (eqs. 37-39).

        :param mean_bare_spring_energy: For each coupling returned by :meth:`couplings`, the mean
            *bare* Einstein energy <Lambda_E * sum_i |r_i - r_i^0|^2>_s (i.e. the logged coupled
            restraint energy divided by s). Units compatible with energy (or plain kJ/mol floats).
        :param n_particles: Number of particles N.
        :return: dA2 / (N kB T), dimensionless.
        """
        g = np.asarray([u.value_in_unit(energy_unit) if unit.is_quantity(u) else float(u)
                        for u in mean_bare_spring_energy], dtype=float)
        if g.shape != self._s.shape:
            raise ValueError(f"expected {self._s.size} mean spring energies, got {g.size}.")
        kt = _kt(self.temperature)
        # integral_0^1 g(s) ds = integral g(s(w)) (ds/dw) dw  ~  sum_k gl_weight_k g_k (ds/dw)_k
        integral = float(np.sum(self._gl_weights * g * self._ds_dw))
        return -integral / (n_particles * kt)


@dataclass
class FreeEnergyResult:
    """Container for the assembled Frenkel-Ladd free energy (all terms in units of N kB T)."""
    a0: float
    delta_a1: float
    delta_a2: float
    n_particles: int

    @property
    def a_sol(self) -> float:
        """A_sol / (N kB T) of the unconstrained solid (A0 already contains the COM release)."""
        return self.a0 + self.delta_a1 + self.delta_a2

    @property
    def a_sol_frenkel_ladd(self) -> float:
        """Historical FL proxy: A_sol/(N kB T) + (2/N) ln N (not an exact finite-N correction)."""
        return self.a_sol + 2.0 * math.log(self.n_particles) / self.n_particles

    def __str__(self) -> str:
        return (f"Frenkel-Ladd free energy (units of N kB T):\n"
                f"  A0     = {self.a0:.4f}\n"
                f"  dA1    = {self.delta_a1:.4f}\n"
                f"  dA2    = {self.delta_a2:.4f}\n"
                f"  A_sol  = {self.a_sol:.4f}   (unconstrained; COM release included in A0)\n"
                f"  A_sol + (2/N)lnN = {self.a_sol_frenkel_ladd:.4f}  (historical FL proxy)")
