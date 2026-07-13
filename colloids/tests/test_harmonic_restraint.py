import numpy as np
import openmm
from openmm import unit
import pytest
from colloids import HarmonicRestraint
from colloids.units import energy_unit, length_unit

_spring_constant_unit = energy_unit / (length_unit ** 2)


def _energy(restraint, positions, box=10.0):
    """Potential energy (kJ/mol) of a HarmonicRestraint for the given positions (nm)."""
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(box, 0, 0) * unit.nanometer,
                                        openmm.Vec3(0, box, 0) * unit.nanometer,
                                        openmm.Vec3(0, 0, box) * unit.nanometer)
    for _ in positions:
        system.addParticle(1.0 * unit.amu)
    force = None
    for f in restraint.yield_potentials():
        f.setForceGroup(0)
        system.addForce(f)
        force = f
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(np.asarray(positions) * unit.nanometer)
    return context, force


class TestHarmonicRestraint(object):
    def test_energy_matches_hand_calculation(self):
        k = 100.0 * _spring_constant_unit
        restraint = HarmonicRestraint(spring_constant=k, coupling=1.0)
        refs = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        for i, r in enumerate(refs):
            restraint.add_particle(i, r * unit.nanometer)
        # Displace particle 0 by (0.1, -0.2, 0.3); leave particle 1 at its reference.
        positions = np.array([[0.1, -0.2, 0.3], [1.0, 2.0, 3.0]])
        context, _ = _energy(restraint, positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(energy_unit)
        expected = 100.0 * (0.1 ** 2 + 0.2 ** 2 + 0.3 ** 2)
        assert np.isclose(energy, expected)

    def test_zero_energy_at_reference(self):
        restraint = HarmonicRestraint(spring_constant=50.0 * _spring_constant_unit)
        refs = np.array([[0.5, 0.5, 0.5], [2.0, 1.0, 0.0]])
        for i, r in enumerate(refs):
            restraint.add_particle(i, r * unit.nanometer)
        context, _ = _energy(restraint, refs)
        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(energy_unit)
        assert np.isclose(energy, 0.0)

    def test_coupling_scales_energy_linearly(self):
        restraint = HarmonicRestraint(spring_constant=100.0 * _spring_constant_unit, coupling=1.0)
        restraint.add_particle(0, np.array([0.0, 0.0, 0.0]) * unit.nanometer)
        context, _ = _energy(restraint, np.array([[0.3, 0.0, 0.0]]))
        full = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(energy_unit)
        for coupling in (0.0, 0.25, 0.5, 1.0):
            context.setParameter("lambda_ein", coupling)
            scaled = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(energy_unit)
            assert np.isclose(scaled, coupling * full)

    def test_invalid_spring_constant_unit(self):
        with pytest.raises(TypeError):
            HarmonicRestraint(spring_constant=100.0 * energy_unit)

    def test_non_positive_spring_constant(self):
        with pytest.raises(ValueError):
            HarmonicRestraint(spring_constant=0.0 * _spring_constant_unit)

    def test_negative_coupling(self):
        with pytest.raises(ValueError):
            HarmonicRestraint(spring_constant=1.0 * _spring_constant_unit, coupling=-0.1)

    def test_reference_position_wrong_length(self):
        restraint = HarmonicRestraint(spring_constant=1.0 * _spring_constant_unit)
        with pytest.raises(ValueError):
            restraint.add_particle(0, np.array([0.0, 0.0]) * unit.nanometer)

    def test_reference_position_wrong_unit(self):
        restraint = HarmonicRestraint(spring_constant=1.0 * _spring_constant_unit)
        with pytest.raises(TypeError):
            restraint.add_particle(0, np.array([0.0, 0.0, 0.0]) * unit.nanosecond)

    def test_add_particle_after_yield_raises(self):
        restraint = HarmonicRestraint(spring_constant=1.0 * _spring_constant_unit)
        restraint.add_particle(0, np.array([0.0, 0.0, 0.0]) * unit.nanometer)
        list(restraint.yield_potentials())
        with pytest.raises(RuntimeError):
            restraint.add_particle(1, np.array([1.0, 0.0, 0.0]) * unit.nanometer)


if __name__ == '__main__':
    pytest.main([__file__])
