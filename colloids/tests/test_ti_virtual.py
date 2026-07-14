import dataclasses
import numpy as np
import gsd.hoomd
import openmm
from openmm import unit
import pytest
from colloids.run_parameters import RunParameters
from colloids.ti_parameters import TIParameters
from colloids.colloids_run import set_up_simulation
from colloids.units import energy_unit, length_unit, temperature_unit, time_unit

_spring_constant_unit = energy_unit / (length_unit ** 2)


def _simple_frame(n_side=3, spacing=40.0, mass=1.0):
    """A small cubic lattice of identical colloids in a periodic box (no walls)."""
    positions = np.array([[i * spacing, j * spacing, k * spacing]
                          for i in range(n_side) for j in range(n_side) for k in range(n_side)],
                         dtype=np.float32)
    positions -= positions.mean(axis=0)
    n = len(positions)
    frame = gsd.hoomd.Frame()
    frame.particles.N = n
    frame.particles.types = ["A"]
    frame.particles.typeid = np.zeros(n, dtype=np.uint32)
    frame.particles.position = positions
    frame.particles.diameter = np.full(n, 20.0, dtype=np.float32)
    frame.particles.charge = np.full(n, -30.0, dtype=np.float32)
    frame.particles.mass = np.full(n, mass, dtype=np.float32)
    frame.configuration.box = [120.0, 120.0, 120.0, 0.0, 0.0, 0.0]
    return frame


def _run_parameters():
    return RunParameters(
        initial_configuration="init.gsd", platform_name="Reference", cutoff_factor=6.0,
        debye_length=5.727 * length_unit, brush_length=10.6 * length_unit,
        integrator="LangevinMiddleIntegrator",
        integrator_parameters={"temperature": 298.0 * temperature_unit,
                               "frictionCoeff": 1.0 / time_unit, "stepSize": 0.02 * time_unit,
                               "randomNumberSeed": 3},
        wall_directions=[False, False, False])


def _ti_parameters(use_virtual_particles):
    return TIParameters(spring_constant=50.0 * _spring_constant_unit, coupling=0.5,
                        use_virtual_particles=use_virtual_particles)


def _restraint_energy(simulation, positions):
    if simulation.ti_virtual_reference_positions is not None:
        positions = np.vstack([positions, simulation.ti_virtual_reference_positions])
    simulation.context.setPositions(positions * unit.nanometer)
    group = next(f.getForceGroup() for f in simulation.system.getForces()
                 if f.getName() == "harmonic_restraint_energy")
    return simulation.context.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(energy_unit)


class TestVirtualParticleRestraint(object):
    def test_virtual_matches_external_and_expected(self):
        frame = _simple_frame()
        parameters = _run_parameters()
        rng = np.random.default_rng(0)
        displacement = rng.normal(0.0, 0.3, frame.particles.position.shape)
        displaced = frame.particles.position + displacement
        virtual = _restraint_energy(set_up_simulation(parameters, frame, _ti_parameters(True), frame), displaced)
        external = _restraint_energy(set_up_simulation(parameters, frame, _ti_parameters(False), frame), displaced)
        expected = 0.5 * 50.0 * float((displacement ** 2).sum())  # coupling * Lambda_E * sum d^2
        assert np.isclose(virtual, expected, rtol=1e-5)
        assert np.isclose(external, expected, rtol=1e-5)

    def test_virtual_adds_particles_only_to_system(self):
        frame = _simple_frame()
        simulation = set_up_simulation(_run_parameters(), frame, _ti_parameters(True), frame)
        # System has 2N particles (real + virtual); topology has only the N real particles.
        assert simulation.system.getNumParticles() == 2 * frame.particles.N
        assert simulation.topology.getNumAtoms() == frame.particles.N
        assert simulation.ti_virtual_reference_positions.shape == (frame.particles.N, 3)

    def test_external_mode_keeps_particle_count(self):
        frame = _simple_frame()
        simulation = set_up_simulation(_run_parameters(), frame, _ti_parameters(False), frame)
        assert simulation.system.getNumParticles() == frame.particles.N
        assert simulation.ti_virtual_reference_positions is None

    def test_virtual_particle_is_frozen(self):
        frame = _simple_frame()
        simulation = set_up_simulation(_run_parameters(), frame, _ti_parameters(True), frame)
        # Every virtual particle (indices N..2N-1) has zero mass.
        for i in range(frame.particles.N, 2 * frame.particles.N):
            assert simulation.system.getParticleMass(i).value_in_unit(unit.amu) == 0.0

    def test_substrate_conflict_raises(self):
        frame = _simple_frame()
        masses = np.array(frame.particles.mass)
        masses[0] = 0.0  # an immobile "substrate" particle
        frame.particles.mass = masses
        with pytest.raises(ValueError):
            set_up_simulation(_run_parameters(), frame, _ti_parameters(True), frame)

    def test_reference_size_mismatch_raises(self):
        frame = _simple_frame()
        reference = _simple_frame(n_side=2)  # different particle count
        with pytest.raises(ValueError):
            set_up_simulation(_run_parameters(), frame, _ti_parameters(True), reference)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
