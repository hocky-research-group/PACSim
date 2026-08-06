# Non-cubic CIFs: relaxing the axial ratio to the colloid size ratio

`LatticeBuilder` scales a CIF **uniformly**, so the cell keeps the axial ratios (`b/a`, `c/a`)
written in the file. Those ratios describe the *atomic* crystal. The ideal ratios for a *colloidal*
crystal of the same structure type depend on the ratio of the particle radii, so for any non-cubic
structure they are generally wrong.

This example is AlB2 built from a **200 nm cation (Al)** and a **120 nm anion (B x2)** at
+40 / -60 mV, Debye 8 nm. Build both variants and compare:

```bash
cd Uniform      && pacsim-create configuration.yaml first_frame.gsd
cd ../Anisotropic && pacsim-create configuration.yaml first_frame.gsd
```

The two differ by one line in `configuration.yaml`:

```yaml
  optimize_energy: true
  anisotropic_energy: true     # Anisotropic/ only
```

## What you get

| | `c/a` | as-built `U/NkT` | Al–B gap | B–B gap |
|---|---|---|---|---|
| `Uniform/` | 1.0906 (the atomic value) | **+0.000** | +207.1 nm | +135.0 nm |
| `Anisotropic/` | **0.9743** | **−90.438** | −10.3 nm | −10.4 nm |

Gaps are centre-to-centre distance minus the sum of the two effective radii
(radius + brush 10 nm + `radii_padding` 5 nm); negative means the brush layers are in contact, which
is what a bound colloidal crystal looks like. Cores never overlap: the Al–B centres sit 339.7 nm
apart against 320 nm of hard core.

The uniform build also prints

```
UserWarning: Energy minimum at boundary of scan range (scale=234.4989).
```

## Why the uniform build fails, in two stages

**The geometry is wrong.** At `c/a = 1.0906` the honeycomb of small anions jams first and sets the
scale, leaving the *attractive* Al–B pairs 21 nm apart while only the *like-charge* B–B pairs touch.
The lattice is then net repulsive, +8.5 NkT. (Build with `optimize_energy: false` to see this
just-touching state directly.)

**The energy scan then makes it worse.** Because that lattice has no cohesive minimum, energy keeps
falling as the cell expands, so the scan runs to the top of its range — hence the boundary warning —
and returns a cell so dilute that nothing interacts at all. That is the `U = +0.000` and the +207 nm
Al–B gap above: the "crystal" is 192 non-interacting particles. Any stability test on it melts
immediately, which is easy to misread as a physical result about AlB2.

With the axial ratio free, both the small–small in-plane contacts and the large–small interlayer
contacts close together, the attraction switches on, and the lattice is strongly bound.

## What to do

- **Cubic CIF** (CsCl, Th3P4, Cu3Au, …): nothing. Uniform scaling is exact and
  `anisotropic_energy` is a no-op that yields a bit-identical configuration.
- **Non-cubic CIF**: set `anisotropic_energy: true` whenever the colloid radius ratio differs from
  the atomic one, i.e. essentially always.
- **Check the build regardless.** Two cheap tells:
  - as-built energy near zero, or a boundary warning from the scale scan;
  - the touching pair being a *like-charge* pair. In a charge-ordered crystal the closest contact
    should be an oppositely charged pair. Compare each pair *type*'s minimum centre-to-centre
    distance with the sum of its effective radii.

Axes are grouped by crystal system, so rescaling never lowers the cell's symmetry: cubic keeps one
free parameter, hexagonal/tetragonal/trigonal two (`a = b`, `c`), orthorhombic/monoclinic/triclinic
three. The space group is determined from the coordinates, not read from the CIF header, so a file
written in `P1` — as symmetry-expanded CIFs usually are — is still grouped correctly.

Note this is about the *starting* lattice. `scripts/crystal-TI/run_ti.py` relaxes the box with an
anisotropic NPT barostat, which can refine the axial ratio further, but only if the starting
structure is bound well enough to survive; it cannot rescue a non-interacting one.
