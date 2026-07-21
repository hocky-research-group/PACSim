#!/usr/bin/env python
"""
Descriptive, self-documenting names for PACS simulation outputs.

The convention (see README.md in this folder) is

    <tag>_<Structure>_debye<lambda_D>_rP<r+>_rN<r->_charges_p<psi+>_m<|psi-|>

e.g.  run_CsCl_debye7.5_rP102_rN120_charges_p35_m35  (the '+' colloid is the small one here).

Use ``descriptive_prefix(...)`` when you already have the numbers, or ``prefix_from_config(...)``
to read them straight from a run YAML + a generated crystal GSD. The result goes in the run YAML's
``output_prefix`` key, which PACSim uses as a filename prefix: every output filename still at its
default becomes ``<output_prefix><suffix>``, so

    output_prefix: run_CsCl_debye7.5_rP102_rN120_charges_p35_m35

writes ``run_CsCl_..._m35.trajectory.gsd``, ``....state.csv``, ``....chk`` and ``....final.gsd``
into the working directory. ``output_prefix`` may itself contain a directory, so to keep each run
in its own folder repeat the name (``nested=True`` here, ``--nested`` on the command line):

    output_prefix: run_CsCl_..._m35/run_CsCl_..._m35

which puts the same four files inside ``run_CsCl_..._m35/``. The directory is created at run time.
"""

import os
import sys
import numpy as np
import gsd.hoomd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from colloids.run_parameters import RunParameters  # noqa: E402
from colloids.units import length_unit  # noqa: E402


def _fmt(x):
    """Compact number formatting: 7.5 -> '7.5', 100.0 -> '100', 46.67 -> '46.7'."""
    x = float(x)
    if x == int(x):
        return str(int(x))
    return f"{x:.1f}".rstrip("0").rstrip(".")


def descriptive_prefix(structure, debye_nm, r_plus_nm, r_minus_nm, psi_plus_mV, psi_minus_mV,
                       tag="run", nested=False):
    """Build the descriptive output prefix from explicit parameters.

    With ``nested=True`` the name is repeated as ``<name>/<name>``, so that ``output_prefix`` puts
    the run's four output files inside a directory of that name instead of the working directory.
    """
    name = (f"{tag}_{structure}"
            f"_debye{_fmt(debye_nm)}"
            f"_rP{_fmt(r_plus_nm)}_rN{_fmt(r_minus_nm)}"
            f"_charges_p{_fmt(abs(psi_plus_mV))}_m{_fmt(abs(psi_minus_mV))}")
    return f"{name}/{name}" if nested else name


def species_from_frame(configuration_gsd, frame_index=-1):
    """Return (r_plus, psi_plus, r_minus, psi_minus) in nm/mV, identifying species by charge sign."""
    with gsd.hoomd.open(configuration_gsd, "r") as f:
        frame = f[frame_index]
    typeid = np.asarray(frame.particles.typeid)
    diameter = np.asarray(frame.particles.diameter)
    charge = np.asarray(frame.particles.charge)
    plus = charge > 0
    minus = charge < 0
    if not (np.any(plus) and np.any(minus)):
        raise ValueError("Expected both positively and negatively charged species in the frame.")
    r_plus = float(diameter[plus][0] / 2.0);  psi_plus = float(charge[plus][0])
    r_minus = float(diameter[minus][0] / 2.0); psi_minus = float(charge[minus][0])
    return r_plus, psi_plus, r_minus, psi_minus


def prefix_from_config(structure, run_yaml, configuration_gsd, tag="run", nested=False):
    """Build the descriptive name by reading the Debye length from the run YAML and the radii /
    surface charges from a generated crystal GSD."""
    parameters = RunParameters.from_yaml(run_yaml)
    debye_nm = parameters.debye_length.value_in_unit(length_unit)
    r_plus, psi_plus, r_minus, psi_minus = species_from_frame(configuration_gsd)
    return descriptive_prefix(structure, debye_nm, r_plus, r_minus, psi_plus, psi_minus, tag=tag,
                              nested=nested)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Print the descriptive output_prefix for a run.")
    p.add_argument("structure", help="structure label, e.g. CsCl or Th3P4")
    p.add_argument("run_yaml")
    p.add_argument("configuration_gsd")
    p.add_argument("--tag", default="run")
    p.add_argument("--nested", action="store_true",
                   help="print '<name>/<name>' so that output_prefix groups the run's outputs "
                        "inside a directory of that name")
    a = p.parse_args()
    print(prefix_from_config(a.structure, a.run_yaml, a.configuration_gsd, tag=a.tag,
                             nested=a.nested))
