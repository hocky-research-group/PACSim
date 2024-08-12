from openmm import unit
import pytest
from colloids import ColloidPotentialsAlgebraic, ColloidPotentialsParameters
from colloids.colloids_tune.colloids_tune import tune_surface_potential


class TestTuneSurfacePotential(object):
    @pytest.fixture
    def brush_density(self):
        return 0.09 / ((unit.nano * unit.meter) ** 2)

    @pytest.fixture
    def brush_length(self):
        return 10.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def debye_length(self):
        return 5.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def dielectric_constant(self):
        return 80.0

    @pytest.fixture
    def temperature(self):
        return 298.0 * unit.kelvin

    @pytest.fixture
    def colloid_potentials_parameters(self, brush_density, brush_length, debye_length, temperature,
                                      dielectric_constant):
        return ColloidPotentialsParameters(brush_density=brush_density, brush_length=brush_length,
                                           debye_length=debye_length, temperature=temperature,
                                           dielectric_constant=dielectric_constant)

    @pytest.fixture
    def colloid_potentials_algebraic(self, colloid_potentials_parameters):
        return ColloidPotentialsAlgebraic(colloid_potentials_parameters=colloid_potentials_parameters, use_log=False,
                                          cutoff_factor=21.0, periodic_boundary_conditions=False)

    @pytest.fixture
    def colloid_potentials_algebraic_log(self, colloid_potentials_parameters):
        return ColloidPotentialsAlgebraic(colloid_potentials_parameters=colloid_potentials_parameters, use_log=True,
                                          cutoff_factor=21.0, periodic_boundary_conditions=False)

    @pytest.fixture
    def other_radius(self):
        return 85.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def tuned_radius(self):
        return 30.0 * (unit.nano * unit.meter)

    @pytest.fixture
    def other_surface_potential(self):
        return 50.0 * (unit.milli * unit.volt)

    @pytest.fixture
    def tuned_potential_depth(self):
        return -3.0 * 298.0 * unit.kelvin * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

    @pytest.fixture(params=["algebraic", "algebraic_log"])
    def tune_algebraic_colloid_potentials(self, colloid_potentials_algebraic, colloid_potentials_algebraic_log,
                                          request):
        if request.param == "algebraic":
            return colloid_potentials_algebraic
        else:
            return colloid_potentials_algebraic_log

    def test_tune_surface_potential_exceptions_algebraic(self, tune_algebraic_colloid_potentials,
                                                         other_radius, tuned_radius,
                                                         other_surface_potential, tuned_potential_depth):
        # Test exception on radius with wrong unit.
        with pytest.raises(TypeError):
            tune_surface_potential(
                tune_algebraic_colloid_potentials, other_radius * (unit.nano * unit.meter), other_surface_potential,
                tuned_radius, tuned_potential_depth, None)
        with pytest.raises(TypeError):
            tune_surface_potential(
                tune_algebraic_colloid_potentials, other_radius, other_surface_potential,
                tuned_radius * (unit.nano * unit.meter), tuned_potential_depth, None)
        # Test exception on negative radius.
        with pytest.raises(ValueError):
            tune_surface_potential(
                tune_algebraic_colloid_potentials, -other_radius, other_surface_potential, tuned_radius,
                tuned_potential_depth, None)
        with pytest.raises(ValueError):
            tune_surface_potential(
                tune_algebraic_colloid_potentials, other_radius, other_surface_potential, -tuned_radius,
                tuned_potential_depth, None)
        # Test exception on surface potential with wrong unit.
        with pytest.raises(TypeError):
            tune_surface_potential(
                tune_algebraic_colloid_potentials, other_radius, other_surface_potential * (unit.milli * unit.volt),
                tuned_radius, tuned_potential_depth, None)
        # Test exception on potential depth with wrong unit.
        with pytest.raises(TypeError):
            tune_surface_potential(
                tune_algebraic_colloid_potentials, other_radius, other_surface_potential, tuned_radius,
                tuned_potential_depth / unit.kelvin, None)
        # Test exception on positive potential depth.
        with pytest.raises(ValueError):
            tune_surface_potential(
                tune_algebraic_colloid_potentials, other_radius, other_surface_potential, tuned_radius,
                -tuned_potential_depth, None)

    def test_tune_surface_potential_algebraic(self, colloid_potentials_algebraic, other_radius,
                                              tuned_radius, other_surface_potential,
                                              tuned_potential_depth):
        result = tune_surface_potential(
            colloid_potentials_algebraic, other_radius, other_surface_potential, tuned_radius, tuned_potential_depth,
            None)
        # Tested with Mathematica, see test_colloid_potentials.nb.
        assert result.value_in_unit(unit.milli * unit.volt) == pytest.approx(-57.55640041253047)

    def test_tune_surface_potential_algebraic_inverted(self, colloid_potentials_algebraic, other_radius,
                                                       tuned_radius, other_surface_potential,
                                                       tuned_potential_depth):
        result = tune_surface_potential(
            colloid_potentials_algebraic, other_radius, -57.55640041253047 * (unit.milli * unit.volt),
            tuned_radius, tuned_potential_depth, None)
        # Tested with Mathematica, see test_colloid_potentials.nb.
        assert result.value_in_unit(unit.milli * unit.volt) == pytest.approx(
            other_surface_potential.value_in_unit(unit.milli * unit.volt))

    def test_tune_surface_potential_algebraic_negative(self, colloid_potentials_algebraic, other_radius,
                                                       tuned_radius, other_surface_potential,
                                                       tuned_potential_depth):
        result = tune_surface_potential(
            colloid_potentials_algebraic, other_radius, -other_surface_potential,
            tuned_radius, tuned_potential_depth, None)
        # Tested with Mathematica, see test_colloid_potentials.nb.
        assert result.value_in_unit(unit.milli * unit.volt) == pytest.approx(57.55640041253047)

    def test_tune_surface_potential_algebraic_negative_inverted(self, colloid_potentials_algebraic, other_radius,
                                                                tuned_radius, other_surface_potential,
                                                                tuned_potential_depth):
        result = tune_surface_potential(
            colloid_potentials_algebraic, other_radius, 57.55640041253047 * (unit.milli * unit.volt),
            tuned_radius, tuned_potential_depth, None)
        # Tested with Mathematica, see test_colloid_potentials.nb.
        assert result.value_in_unit(unit.milli * unit.volt) == pytest.approx(
            -other_surface_potential.value_in_unit(unit.milli * unit.volt))

    def test_tune_surface_potential_algebraic_log(self, colloid_potentials_algebraic_log, other_radius,
                                                  tuned_radius, other_surface_potential,
                                                  tuned_potential_depth):
        result = tune_surface_potential(
            colloid_potentials_algebraic_log, other_radius, other_surface_potential,
            tuned_radius, tuned_potential_depth, None)
        # Tested with Mathematica, see test_colloid_potentials.nb.
        assert result.value_in_unit(unit.milli * unit.volt) == pytest.approx(-58.23701709900136)

    def test_tune_surface_potential_algebraic_log_inverted(self, colloid_potentials_algebraic_log, other_radius,
                                                           tuned_radius, other_surface_potential,
                                                           tuned_potential_depth):
        result = tune_surface_potential(
            colloid_potentials_algebraic_log, other_radius, -58.23701709900136 * (unit.milli * unit.volt),
            tuned_radius, tuned_potential_depth, None)
        # Tested with Mathematica, see test_colloid_potentials.nb.
        assert result.value_in_unit(unit.milli * unit.volt) == pytest.approx(
            other_surface_potential.value_in_unit(unit.milli * unit.volt))

    def test_tune_surface_potential_algebraic_log_negative(self, colloid_potentials_algebraic_log, other_radius,
                                                           tuned_radius, other_surface_potential,
                                                           tuned_potential_depth):
        result = tune_surface_potential(
            colloid_potentials_algebraic_log, other_radius, -other_surface_potential,
            tuned_radius, tuned_potential_depth, None)
        # Tested with Mathematica, see test_colloid_potentials.nb.
        assert result.value_in_unit(unit.milli * unit.volt) == pytest.approx(58.23701709900136)

    def test_tune_surface_potential_algebraic_log_negative_inverted(self, colloid_potentials_algebraic_log,
                                                                    other_radius, tuned_radius, other_surface_potential,
                                                                    tuned_potential_depth):
        result = tune_surface_potential(
            colloid_potentials_algebraic_log, other_radius, 58.23701709900136 * (unit.milli * unit.volt),
            tuned_radius, tuned_potential_depth, None)
        # Tested with Mathematica, see test_colloid_potentials.nb.
        assert result.value_in_unit(unit.milli * unit.volt) == pytest.approx(
            -other_surface_potential.value_in_unit(unit.milli * unit.volt))


if __name__ == '__main__':
    pytest.main([__file__])
