# Crystal free energies by Frenkel-Ladd / Einstein-crystal thermodynamic integration

These scripts compute the absolute Helmholtz free energy `A_sol` of a bulk PACS colloidal crystal,
so that competing candidate structures (e.g. CsCl vs Th3P4) can be ranked thermodynamically at the
same conditions. They drive the TI machinery that lives in the package itself
(`colloids.ti_parameters`, the `HarmonicRestraint` Einstein force, the NPT barostat and the periodic
bulk-crystal option of `LatticeBuilder`).

The method follows

- D. Frenkel and A. J. C. Ladd, *J. Chem. Phys.* **81**, 3188 (1984);
- C. Vega, E. Sanz, J. L. F. Abascal and E. G. Noya, *J. Phys.: Condens. Matter* **20**, 153101
  (2008) — equation numbers in the source refer to this review.

It is validated against the Lennard-Jones Frenkel-Ladd benchmark (`A_sol = 3.104` vs the literature
`3.11 N kB T`), and `tests/test_free_energy.py` reproduces the analytic ideal-Einstein free energies
of Table 1 of Vega et al. to better than `1e-3 N kB T`.

These scripts are standalone command-line tools, not part of the importable `colloids` package.
Run them with `python scripts/crystal-TI/<script>.py`, or add the folder to `PYTHONPATH`.

## Contents

| Script | Purpose |
| --- | --- |
| `run_ti.py` | End-to-end driver: NPT relaxation, spring auto-tuning, TI windows, `A_sol`. |
| `free_energy.py` | Library: the ideal-Einstein `A0`, `dA1`, and the Gauss-Legendre schedule. |
| `mbar_analysis.py` | Optional MBAR recombination of the same window data, with overlap guardrails. |
| `ti_diagnostics.py` | Four-panel diagnostics figure for a finished TI run. |
| `sampling_scan.py` | Sub-sampling convergence test: run windows longer, or add more windows? |
| `finite_size_scaling.py` | Extrapolate `A_sol/NkT` to `N -> infinity` from runs at several sizes. |
| `crystal_stability.py` | Box-invariant g(r) / Lindemann verdict on whether a crystal stayed ordered. |
| `pacs_naming.py` | Build the descriptive `output_prefix` directory name for a run. |

## Workflow

### 1. Build a periodic bulk crystal

Use `pacsim-create` with a `LatticeBuilder` configuration that sets `periodic: true`, so the box is
the scaled supercell lattice (a true bulk crystal under PBC) rather than a padded vacuum cluster.

```bash
pacsim-create configuration_bulk.yaml first_frame.gsd
```

Make sure the electrostatic cutoff `2*r_max + cutoff_factor*debye_length` is smaller than half the
box; increase `lattice_repeats` if it is not.

### 2. Run the thermodynamic integration

```bash
python scripts/crystal-TI/run_ti.py run.yaml --output-dir ti \
    --windows 12 --equil-steps 20000 --prod-steps 60000 --sample-interval 200
```

`run_ti.py` reports `A_sol = A0 + dA1 + dA2` in units of `N kB T`:

- **A0** — the analytic ideal Einstein crystal at fixed centre of mass, plus the COM-release term
  (Vega eq. 48), so the assembled `A_sol` is the *unconstrained* solid free energy;
- **dA2** — Gauss-Legendre integration over the Einstein spring coupling. Each window is a separate
  `pacsim-run run.yaml -t ti.yaml` invocation; the mean restraint energy is read back from the
  window's GSD trajectory log;
- **dA1** — ideal to interacting Einstein crystal, by directly Gaussian-sampling the ideal Einstein
  crystal and scoring the PACS energy on those configurations with a PACS-only context.

By default the driver first relaxes the box under the NPT barostat at `--npt-pressure` (the built
lattice sits at the *vacuum* energy minimum, not the finite-temperature equilibrium density), then
energy-minimizes at the mean box and runs the (fixed-volume) TI against that structure. Pass
`--no-npt` to integrate the as-built lattice instead.

The spring constant `Lambda_E` is auto-tuned to a target Einstein displacement, by default 5% of the
crystal's *measured* unrestrained rms displacement (`--no-autotune-spring` falls back to a fraction
of the interaction decay length `min(Debye, brush)`). The number of windows is auto-scaled with the
resulting spring stiffness unless you pass `--no-auto-windows`.

Useful options:

- `--window-platform CUDA --max-parallel 12` — run the windows on a GPU, several at once. Small
  colloid systems barely saturate a GPU, so many windows share one card efficiently.
- `--spring-constant` — set `Lambda_E` (kJ/mol/nm²) explicitly instead of auto-tuning.
- `--delta-a1-samples` — ideal-Einstein samples for `dA1` (default 4000).
- `--mbar`, `--refine-mbar-tol 0.1` — cross-check with MBAR, adaptively inserting windows at the
  lowest-overlap gaps until MBAR and TI agree.

**Check the printed `dA1 log correction`: it must be small (~0.02).** If it is not, lower
`--einstein-displacement-fraction` to stiffen `Lambda_E`. Confirm convergence by checking that
`A_sol` does not move when `Lambda_E` is varied.

### 3. Check the diagnostics

`run_ti.py` writes `diagnostics.png` automatically (disable with `--no-plots`); re-make it at any
time without rerunning the TI:

```bash
python scripts/crystal-TI/ti_diagnostics.py ti
```

The four panels are the `dA2` integrand vs coupling (should be smooth and monotonic), the per-window
spring-energy distributions (overlap), the running mean per window (should flatten well before the
end), and the `dA1` reweighting histogram with its effective sample fraction.

### 4. Decide how much more sampling you need

```bash
python scripts/crystal-TI/sampling_scan.py ti
```

Recomputes `dA2` from the first 1/8, 1/4, 1/2, 3/4 and all of each window's frames. If the TI error
and the MBAR–TI gap both shrink like `1/sqrt(t)`, the residual is statistical — run each window
**longer**. If the gap plateaus while the TI estimate has converged, it is MBAR overlap bias — add
**more windows** (`--refine-mbar-tol`), not more time.

### 5. Extrapolate to the thermodynamic limit

Structures with different numbers of atoms per cell are compared at different `N`, so their
finite-size corrections differ. Run the TI at several sizes under identical conditions and fit:

```bash
python scripts/crystal-TI/finite_size_scaling.py ti_N128 ti_N432 ti_N1024 --label CsCl
```

This reports `a_inf ± uncertainty`, the finite-size bias at each `N`, and writes `fss_<label>.png`.

### 6. Confirm the crystal actually stayed a crystal

```bash
python scripts/crystal-TI/crystal_stability.py trajectory.gsd
```

Reports the start-vs-end g(r) correlation and the Lindemann parameter. Both are made box-invariant
by mapping every frame into the reference box via fractional coordinates, so an NPT breathing box is
not mistaken for melting. Q6 is printed for reference but is not used in the verdict (it is weak for
some structures, e.g. Th3P4).

## Temperature convention

In the normal case **`potential_temperature` should equal the thermostat (integrator) temperature** —
they are the physical temperature of the system. Tie them together in the run YAML with the `!Copy`
tag:

```yaml
integrator_parameters:
  temperature: !Copy {key: potential_temperature}
```

Both the NPT barostat and `run_ti.py` take the thermal energy `kT` from the thermostat temperature,
so with the two tied together everything is consistent. Decoupling them is a special case (e.g. to
reproduce a specific attraction-dominated regime), not the recommended default.

## Platform notes

- **TI runs on all platforms.** The Einstein restraint uses `periodicdistance`, so it is continuous
  across periodic images and stable on CUDA and OpenCL. `run_ti.py` still defaults the windows to
  `CPU` for reproducibility; pass `--window-platform CUDA` for a GPU. The restraint-free `dA1`
  energy evaluations default to a GPU (`--energy-platform`).
- Ordinary (non-TI) PACS runs, including NPT equilibration and stability runs, are fine on OpenCL.
- For larger systems use CUDA (double/mixed precision) on a remote GPU.

## Recommended production settings

- 12–20 Gauss-Legendre windows (the default auto-scales upward for stiff springs).
- Per window: equilibration ≥ 20k steps, production ≥ 60k steps — longer for large `N` or stiff
  springs.
- `dA1`: ≥ 4000 ideal-Einstein samples.
- Check `Lambda_E` invariance, and the finite-size behavior (the `(2/N) ln N` Frenkel-Ladd proxy
  term is reported as `A_sol_FL_NkT`).

## Output layout

`run_ti.py --output-dir` keeps the results at the top and groups the per-window MD underneath:

```
ti/
├── free_energy.json          # A_sol/NkT, the A0/dA1/dA2 decomposition, per-window data
├── diagnostics.png           # integrand, window overlap, convergence, dA1 reweighting
├── delta_a1_samples.npz      # ideal-Einstein samples (to re-plot diagnostics)
└── windows/
    ├── window_00/            # run.yaml, ti.yaml, trajectory.gsd, state.csv, checkpoint.chk, log
    ├── window_01/
    └── ...
```

For the surrounding simulations, a self-describing `output_prefix` makes every output filename
identify the run it came from:

```
<tag>_<Structure>_debye<lambda_D>_rP<r+>_rN<r->_charges_p<psi+>_m<|psi-|>
```

where `rP`/`rN` are the radii (nm) of the positively/negatively charged colloid and `p`/`m` the
magnitudes (mV) of their surface potentials. Build the name from a run YAML and a generated crystal:

```bash
python scripts/crystal-TI/pacs_naming.py CsCl run.yaml initial.gsd
# -> run_CsCl_debye12_rP102_rN120_charges_p35_m35
```

PACSim treats `output_prefix` as a **filename prefix**: every output filename still at its default
becomes `<output_prefix><suffix>`, so the name above yields
`run_CsCl_..._m35.trajectory.gsd`, `....state.csv`, `....chk` and `....final.gsd` in the working
directory. Because the prefix may itself contain a directory, repeating the name groups a run's
outputs in a folder of its own — `--nested` prints that form:

```bash
python scripts/crystal-TI/pacs_naming.py CsCl run.yaml initial.gsd --nested
# -> run_CsCl_debye12_rP102_rN120_charges_p35_m35/run_CsCl_debye12_rP102_rN120_charges_p35_m35
```

```
run_CsCl_debye12_rP102_rN120_charges_p35_m35/
├── run_CsCl_debye12_rP102_rN120_charges_p35_m35.trajectory.gsd
├── run_CsCl_debye12_rP102_rN120_charges_p35_m35.state.csv
├── run_CsCl_debye12_rP102_rN120_charges_p35_m35.chk
└── run_CsCl_debye12_rP102_rN120_charges_p35_m35.final.gsd
```

The directory is created at run time. An explicitly set output filename always overrides the prefix,
including `final_configuration_gsd_filename: null` to skip writing the final frame. `run_ti.py` sets
`output_prefix` to `null` in every YAML it generates and lays its own windows out under
`--output-dir`, so the TI driver is unaffected by this setting either way.

## Dependencies

Everything except the MBAR cross-check uses only PACSim's own dependencies. `mbar_analysis.py`,
`sampling_scan.py` and `run_ti.py --mbar` additionally need [`pymbar`](https://github.com/choderalab/pymbar):

```bash
pip install pymbar
```

## Tests

```bash
pytest scripts/crystal-TI/tests
```

The MBAR guardrail tests are skipped automatically when `pymbar` is not installed.
