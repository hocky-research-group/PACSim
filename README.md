# PACSim

PACSim is a Python package for building, running, and resuming simulations of patchy and ionic colloidal systems. The package is centered on OpenMM-based molecular simulation workflows and uses GSD files for configurations and trajectories.

The repository includes:

- A simulation runner for colloidal dynamics in OpenMM.
- A configuration generator that builds initial GSD structures from LAMMPS-style cluster definitions.
- Tests and benchmark scripts covering the implemented forces and workflows.

## What PACSim can do

PACSim currently exposes the following capabilities in code:

- Simulate colloidal systems with steric and electrostatic pair interactions based on the parameterization described in Hueckel, Hocky, Palacci, and Sacanna, *Nature* 580, 487-490 (2020).
- Optionally include shifted Lennard-Jones confining walls in the x, y, and z directions.
- Optionally include an implicit charged substrate wall.
- Optionally include gravity and analytical depletion interactions.
- Optionally include PLUMED-driven biasing or collective-variable terms through `openmm-plumed`.
- Start from GSD configurations containing particle positions, velocities, types, diameters, surface potentials, masses, box information, and constraints.
- Resume simulations from OpenMM checkpoint files.
- Write trajectories, state data, checkpoints, and final configurations during a run.
- Ramp or otherwise update force parameters during a simulation via custom update reporters.
- Generate initial configurations from one or more cluster templates stored as LAMMPS data files.
- Apply configurable initial and final modifiers during configuration generation.

## Main command-line tools

Installing the package creates `pacsim-*` command-line tools:

- `pacsim-run`
- `pacsim-create`

Legacy `colloids-*` command names are still provided as compatibility aliases.

### `pacsim-run`

`pacsim-run` runs an OpenMM simulation from a YAML parameter file:

```bash
pacsim-run run.yaml
```

The run configuration controls:

- Input GSD file and frame selection.
- OpenMM platform and integrator settings.
- Pair-potential parameters such as brush density, brush length, Debye length, and dielectric constant.
- Optional walls, depletion, gravity, implicit substrate, and PLUMED forces.
- Output filenames and reporting intervals.
- Optional equilibration, minimization, and velocity initialization.
- Optional time-dependent parameter updates via update reporters.

An example configuration can be written with:

```bash
pacsim-run --example
```

See [`colloids/tests/run_test.yaml`](colloids/tests/run_test.yaml) for a working example.

### Resuming a run

Simulation restarts are handled by `pacsim-run` itself using the `-c/--checkpoint_file` option:

```bash
pacsim-run run.yaml -c checkpoint.chk
```

When a checkpoint is provided, PACSim reloads the OpenMM state and continues writing trajectory, state-data, and update-reporter outputs in append mode.

### `pacsim-create`

`pacsim-create` builds an initial GSD configuration for a simulation:

```bash
pacsim-create configuration.yaml initial_configuration.gsd
```

The configuration-generation workflow supports:

- Reading one or more cluster templates from LAMMPS data files.
- Combining cluster types with specified relative weights.
- Repeating clusters on a lattice.
- Optional random cluster rotation.
- Per-type masses, radii, and surface potentials.
- Optional initial and final modifiers that alter the generated structure.

An example configuration can be written with:

```bash
pacsim-create --example
```

See [`colloids/colloids_create/configuration.yaml`](colloids/colloids_create/configuration.yaml) and [`colloids/colloids_create/cluster.lmp`](colloids/colloids_create/cluster.lmp) for examples.

## Installation

Install the package in your active environment from the repository root:

```bash
pip install -e .
```

The Python package requires Python 3.10 or newer. The `pyproject.toml` dependencies cover the core OpenMM and analysis workflow.

Additional optional components may require manual installation:

- `hoomd` for the older HOOMD-related scripts and tests in this repository.
- `plumed` and `openmm-plumed` for PLUMED-enabled simulations.

If you use PLUMED, some modules needed by this codebase are not enabled by default. The repository previously documented enabling:

```bash
./configure --enable-modules=crystallization+multicolvar+adjmat
```

For `openmm-plumed`, building against your installed PLUMED version is the safest route.

## Typical workflow

1. Generate an initial configuration:

```bash
pacsim-create configuration.yaml first_frame.gsd
```

2. Run the simulation:

```bash
pacsim-run run.yaml
```

3. Resume later if needed:

```bash
pacsim-run run.yaml -c checkpoint.chk
```

## Outputs

Depending on the run configuration, PACSim writes:

- A GSD trajectory.
- A CSV file with time, kinetic energy, potential energy, temperature, and speed.
- An OpenMM checkpoint file.
- CSV files produced by update reporters when parameter ramps are enabled.
- An optional final GSD configuration.

## Testing

Run the test suite from the repository root with:

```bash
pytest colloids
```

Some tests are skipped automatically when optional dependencies such as HOOMD are not installed.

## Repository layout

- [`pyproject.toml`](pyproject.toml): package metadata and entry points.
- [`colloids/`](colloids): core simulation code.
- [`colloids/colloids_create/`](colloids/colloids_create): initial-configuration generation tools.
- [`colloids/tests/`](colloids/tests): regression and validation tests.
