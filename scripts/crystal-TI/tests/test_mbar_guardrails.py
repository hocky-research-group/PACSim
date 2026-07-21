"""Regression tests for MBAR guardrail selection and unsampled-endpoint diagnostics."""

import numpy as np
import pytest

from run_ti import _recommended_mbar_value, _require_equal_mobile_masses

# The MBAR cross-check is an optional part of the workflow; ``pymbar`` is not a PACSim dependency.
mbar_analysis = pytest.importorskip("mbar_analysis", reason="requires pymbar")


def test_recommended_value_is_nan_while_refinement_is_required():
    result = {"recommendation": "refine", "dA2_full": -8.0, "dA2_hybrid": -7.9}
    assert np.isnan(_recommended_mbar_value(result))


def test_recommended_value_uses_only_approved_estimator():
    assert _recommended_mbar_value(
        {"recommendation": "full", "dA2_full": -8.0, "dA2_hybrid": -7.9}) == -8.0
    assert _recommended_mbar_value(
        {"recommendation": "hybrid", "dA2_full": -8.0, "dA2_hybrid": -7.9}) == -7.9


def test_equal_mass_guard_accepts_current_configs_and_rejects_unequal_masses():
    assert np.array_equal(_require_equal_mobile_masses([1.0, 1.0]), [1.0, 1.0])
    with np.testing.assert_raises_regex(ValueError, "assumes equal"):
        _require_equal_mobile_masses([1.0, 2.0])


def test_unsampled_endpoints_use_effective_sample_fraction_not_zero_overlap_column():
    rng = np.random.default_rng(7)
    temperature = 300.0
    kt = 0.0083144626 * temperature
    n_particles = 4
    dof = 3 * (n_particles - 1)
    couplings = np.array([0.1, 0.2, 0.4, 0.8])
    bare = [rng.gamma(dof / 2.0, kt / coupling, 3000) for coupling in couplings]

    result = mbar_analysis.analyze(bare, couplings, n_particles, temperature)

    # PyMBAR's forward overlap into the unsampled s=1 column is zero by construction, while the
    # Kish endpoint efficiency is finite and therefore usable as the actual reweighting diagnostic.
    assert result["raw_adjacent_overlap"][-1] == 0.0
    assert result["endpoint_efficiency"].shape == (2,)
    assert np.all(result["endpoint_efficiency"] > 0.0)
    assert result["adjacent"][-1] == result["endpoint_efficiency"][-1]
    assert result["min_endpoint_efficiency"] == result["min_endpoint_overlap"]


def test_diagnostic_failure_refuses_mbar_estimate(monkeypatch):
    rng = np.random.default_rng(11)
    temperature = 300.0
    kt = 0.0083144626 * temperature
    couplings = np.array([0.1, 0.3, 0.7])
    bare = [rng.gamma(4.5, kt / coupling, 500) for coupling in couplings]

    def fail_overlap(_self):
        raise RuntimeError("synthetic diagnostic failure")

    monkeypatch.setattr(mbar_analysis.MBAR, "compute_overlap", fail_overlap)
    result = mbar_analysis.analyze(bare, couplings, 4, temperature)

    assert result["recommendation"] == "refine"
    assert np.isnan(result["dA2"])
    assert not result["diagnostics_ok"]
    assert "synthetic diagnostic failure" in result["diagnostic_error"]
