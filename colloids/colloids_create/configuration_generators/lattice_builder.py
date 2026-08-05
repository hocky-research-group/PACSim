from typing import Optional, Sequence, Union
import warnings
from gsd.hoomd import Frame
import numpy as np
import openmm
from openmm import unit
from pymatgen.core import Element
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial import distance_matrix
from colloids.colloid_potentials_algebraic import ColloidPotentialsAlgebraic
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters
from colloids.run_parameters import RunParameters
from colloids.units import length_unit, mass_unit, energy_unit
from .abstracts import ConfigurationGenerator


class LatticeBuilder(ConfigurationGenerator):
    """
    Generator for an initial configuration in a gsd.hoomd.Frame instance for a colloid simulation based on a
    crystal lattice structure defined in a CIF file.

    The lattice structure is loaded from a CIF file and expanded into a supercell. The supercell is then uniformly
    scaled so that no particles overlap, accounting for colloid radii, brush length, and an extra radii padding gap.
    The optimal scale factor is computed directly as the maximum ratio of the sum of effective radii to the distance
    for all particle pairs, using a small (3, 3, 3) test supercell for efficiency. This scale factor is then applied to
    the full supercell defined by the lattice repeats.

    Optionally, the scale factor can be further optimized by minimizing the steric + electrostatic energy evaluated in
    vacuum (non-periodic boundary conditions) using OpenMM. A grid of uniformly spaced scale factors around the
    geometric scale factor is evaluated, and the one with the minimum energy is selected.

    The scaled supercell is centered at the origin and embedded in a cubic orthorhombic simulation box. The box side
    length is chosen so that the outermost particle (including its effective radius) plus a lattice padding gap fits
    within the box in every direction.

    :param masses:
        The masses dictionary with the particle types as keys and the masses as values.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii dictionary with the particle types as keys and the radii as values.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials dictionary with the particle types as keys and the surface potentials as values.
    :type surface_potentials: dict[str, unit.Quantity]
    :param lattice_specification:
        The .cif file that specifies the desired lattice structure.
    :type lattice_specification: str
    :param lattice_repeats:
        Number of repetitions of the unit cell in each direction to create the supercell. This can be specified as a
        single integer (if the same number of repetitions is desired in all directions) or as a sequence of three
        integers (if different numbers of repetitions are desired in different directions).
    :type lattice_repeats: Union[int, Sequence[int]]
    :param run_parameters_file:
        The path to the YAML file containing the run parameters. The brush length used for computing the effective
        radii is read from this file.
    :type run_parameters_file: str
    :param radii_padding:
        Extra gap added to the effective radii when checking for overlaps.
        The unit of the radii padding should be compatible with nanometers and the value must be greater than or equal
        to zero.
    :type radii_padding: unit.Quantity
    :param lattice_padding:
        Extra gap added to the box dimensions.
        The unit of the lattice padding should be compatible with nanometers and the value must be greater than or
        equal to zero.
    :type lattice_padding: unit.Quantity
    :param optimize_energy:
        If True, the geometric scale factor is replaced by the scale factor that minimizes the steric +
        electrostatic energy evaluated in vacuum (non-periodic boundary conditions). The energy is computed for a
        grid of uniformly spaced scale factors around the geometric scale factor and the one with the minimum energy
        is selected.
        Defaults to False.
    :type optimize_energy: bool
    :param energy_scale_range:
        The range of scale factors to evaluate, specified as (min_factor, max_factor) relative to the geometric
        scale factor. For example, (0.5, 1.5) means the scan range is from 50% to 150% of the geometric scale
        factor.
        Only allowed when optimize_energy is True. Defaults to (0.5, 1.5) when not specified.
    :type energy_scale_range: Optional[Sequence[float]]
    :param energy_scale_samples:
        The number of uniformly spaced scale factors to evaluate in the scan range.
        Only allowed when optimize_energy is True. Defaults to 50 when not specified.
    :type energy_scale_samples: Optional[int]
    :param anisotropic_energy:
        If True, optimize the three lattice vector lengths INDEPENDENTLY instead of applying one
        uniform scale factor. Only allowed when optimize_energy is True. Defaults to False.

        A uniform scale preserves the CIF's axial ratios (b/a, c/a). Those ratios come from the
        atomic crystal the CIF describes, and they are in general wrong for a colloidal crystal
        built from the same structure type, because the ideal ratios depend on the ratio of the
        particle radii. Carrying the atomic ratios over can jam one sublattice while holding
        another apart -- for AlB2 with 200 nm / 120 nm colloids the atomic c/a = 1.0906 leaves the
        attractive large-small pairs ~21 nm apart while only the like-charge small-small pairs
        touch, so the lattice comes out net repulsive purely as an artefact of the fixed ratio.
        Optimizing the axes independently recovers the physical structure (for that example
        c/a ~ 0.95 and a cohesive lattice).

        Lattice vectors of equal length are scaled together, so the crystal family is preserved:
        a cubic cell keeps one free parameter (identical to the uniform scan), a hexagonal or
        tetragonal cell gets two (a=b, c), and an orthorhombic cell gets three. The search is a
        coordinate descent over those parameters, which costs a few hundred single-point energies
        rather than the scale_samples**3 of a full grid.
    :type anisotropic_energy: bool
    :param periodic:
        If True, generate a bulk periodic crystal: the simulation box is the (uniformly scaled)
        supercell lattice itself, so the crystal tiles space under periodic boundary conditions.
        The lattice_padding is ignored in this case (there is no surrounding vacuum). If False (the
        default), the scaled supercell is centered in a large cubic box padded by lattice_padding,
        i.e. an isolated crystal cluster in vacuum.
        Defaults to False.
    :type periodic: bool

    :raises ValueError:
        If the lattice specification file is not a .cif file.
        If the CIF file does not contain exactly one structure.
        If the lattice repeats is not a positive integer or a sequence of three positive integers.
        If the run parameters file is not a .yaml file.
        If the radii padding is not compatible with nanometers or is negative.
        If the lattice padding is not compatible with nanometers or is negative.
        If energy_scale_range or energy_scale_samples is set when optimize_energy is False.
        If the energy scale range does not consist of exactly two floats satisfying 0 < min < max.
        If the energy scale samples is less than 2.
    :raises RuntimeError:
        If optimize_energy is True and no valid scale factor is found during the energy scan.
    """

    def __init__(self, masses: dict[str, unit.Quantity], radii: dict[str, unit.Quantity],
                 surface_potentials: dict[str, unit.Quantity], lattice_specification: str,
                 lattice_repeats: Union[int, Sequence[int]], run_parameters_file: str,
                 radii_padding: unit.Quantity, lattice_padding: unit.Quantity,
                 optimize_energy: bool = False, energy_scale_range: Optional[Sequence[float]] = None,
                 energy_scale_samples: Optional[int] = None, periodic: bool = False,
                 anisotropic_energy: bool = False) -> None:
        """Constructor of the LatticeBuilder class."""
        super().__init__(masses=masses, radii=radii, surface_potentials=surface_potentials)
        self._periodic = periodic
        if not lattice_specification.endswith('.cif'):
            raise ValueError("The lattice specification must be a .cif file.")
        parser = CifParser(lattice_specification, site_tolerance=0.0, frac_tolerance=0.0)
        structures = parser.parse_structures(check_occu=False, primitive=False)
        if len(structures) != 1:
            raise ValueError("The CIF file must contain exactly one structure.")
        self._structure = structures[0]
        self._radii = radii
        self._surface_potentials = surface_potentials
        self._masses = masses
        if not run_parameters_file.endswith('.yaml'):
            raise ValueError("The run parameters file must be a .yaml file.")
        run_parameters = RunParameters.from_yaml(run_parameters_file)
        self._run_parameters = run_parameters
        self._brush_length = run_parameters.brush_length
        self._lattice_repeats = lattice_repeats
        self._radii_padding = radii_padding
        self._lattice_padding = lattice_padding
        self._optimize_energy = optimize_energy
        self._anisotropic_energy = anisotropic_energy
        if not optimize_energy:
            if energy_scale_range is not None:
                raise ValueError("energy_scale_range must not be set when optimize_energy is False.")
            if energy_scale_samples is not None:
                raise ValueError("energy_scale_samples must not be set when optimize_energy is False.")
            if anisotropic_energy:
                raise ValueError("anisotropic_energy must not be set when optimize_energy is False.")
        self._energy_scale_range = tuple(energy_scale_range) if energy_scale_range is not None else (0.5, 1.5)
        self._energy_scale_samples = energy_scale_samples if energy_scale_samples is not None else 50
        # Label atoms as their element symbols based on atomic number, e.g. 'Fe', 'O', etc.
        self._type_map = {atomic_number: str(Element.from_Z(atomic_number))
                          for atomic_number in np.unique(self._structure.atomic_numbers)}
        if isinstance(self._lattice_repeats, int):
            if not self._lattice_repeats > 0:
                raise ValueError("The lattice repeats must be greater than zero.")
        else:
            if not isinstance(self._lattice_repeats, Sequence) or len(self._lattice_repeats) != 3:
                raise ValueError("The lattice repeats must be either a single integer or a sequence of three integers.")
            if not all(r > 0 for r in self._lattice_repeats):
                raise ValueError("All values in the lattice repeats must be greater than zero.")
        if not self._radii_padding.unit.is_compatible(length_unit):
            raise TypeError("The radii padding must have a unit that is compatible with nanometers.")
        if not self._radii_padding.value_in_unit(length_unit) >= 0.0:
            raise ValueError("The radii padding must have a value greater than or equal to zero.")
        if not self._lattice_padding.unit.is_compatible(length_unit):
            raise TypeError("The lattice padding must have a unit that is compatible with nanometers.")
        if not self._lattice_padding.value_in_unit(length_unit) >= 0.0:
            raise ValueError("The lattice padding must have a value greater than or equal to zero.")
        if len(self._energy_scale_range) != 2:
            raise ValueError("The energy scale range must be a sequence of exactly two floats.")
        if not 0.0 < self._energy_scale_range[0] < self._energy_scale_range[1]:
            raise ValueError("The energy scale range must satisfy 0 < min < max.")
        if not self._energy_scale_samples >= 2:
            raise ValueError("The energy scale samples must be at least 2.")

    def types(self) -> set[str]:
        """
        Return the set of particle types that will be generated by this configuration generator.

        :return:
            The set of particle types that will be generated by this configuration generator.
        :rtype: set[str]
        """
        return set(self._type_map.values())

    @staticmethod
    def _optimize_scale_factor(base_cart_coords: np.ndarray, types: list[str], radii: dict[str, unit.Quantity],
                               surface_potentials: dict[str, unit.Quantity], masses: dict[str, unit.Quantity],
                               run_parameters: RunParameters, geometric_scale_factor: float,
                               scale_range: tuple[float, float], scale_samples: int) -> float:
        """
        Find the scale factor that minimizes the steric + electrostatic energy.

        A grid of uniformly spaced scale factors is evaluated. For each scale factor, the positions
        are computed as base_cart_coords * scale (centered at the origin), and the total energy is
        computed in vacuum (non-periodic). The scale factor with the minimum energy is returned.

        :param base_cart_coords:
            The unscaled Cartesian coordinates from pymatgen, shape (N, 3).
        :type base_cart_coords: np.ndarray
        :param types:
            The type label for each particle.
        :type types: list[str]
        :param radii:
            The radii dictionary with the particle types as keys and the radii as values.
        :type radii: dict[str, unit.Quantity]
        :param surface_potentials:
            The surface potentials dictionary with the particle types as keys and the surface potentials as values.
        :type surface_potentials: dict[str, unit.Quantity]
        :param masses:
            The masses dictionary with the particle types as keys and the masses as values.
        :type masses: dict[str, unit.Quantity]
        :param run_parameters:
            The run parameters.
        :type run_parameters: RunParameters
        :param geometric_scale_factor:
            The geometric (overlap-avoidance) scale factor.
        :type geometric_scale_factor: float
        :param scale_range:
            (min_factor, max_factor) relative to the geometric scale factor.
        :type scale_range: tuple[float, float]
        :param scale_samples:
            The number of uniformly spaced scale factors to evaluate.
        :type scale_samples: int

        :return:
            The optimal scale factor that minimizes the total energy.
        :rtype: float

        :raises RuntimeError:
            If no valid scale factor is found (all energies are infinite or NaN).
        """
        scale_min = geometric_scale_factor * scale_range[0]
        scale_max = geometric_scale_factor * scale_range[1]
        candidates = np.linspace(scale_min, scale_max, scale_samples)

        energy_of = LatticeBuilder._vacuum_energy_function(
            types=types, radii=radii, surface_potentials=surface_potentials, masses=masses,
            run_parameters=run_parameters)

        energy_table = []
        for scale in candidates:
            energy_table.append((float(scale), energy_of(base_cart_coords * scale)))

        # Select the scale with minimum finite energy.
        finite_entries = [(s, e) for s, e in energy_table if np.isfinite(e)]
        if not finite_entries:
            raise RuntimeError("No valid scale factor found. All candidates had infinite or NaN energy.")
        best_scale, best_energy = min(finite_entries, key=lambda x: x[1])

        finite_scales = [s for s, _ in finite_entries]
        minimum_at_boundary = (best_scale == finite_scales[0] or best_scale == finite_scales[-1])
        if minimum_at_boundary:
            warnings.warn(f"Energy minimum at boundary of scan range (scale={best_scale:.4f}). "
                          f"Consider widening energy_scale_range.")

        return best_scale

    @staticmethod
    def _vacuum_energy_function(types: list[str], radii: dict[str, unit.Quantity],
                                surface_potentials: dict[str, unit.Quantity],
                                masses: dict[str, unit.Quantity], run_parameters: RunParameters):
        """
        Build the steric + electrostatic system once and return a callable giving its energy.

        The returned function takes Cartesian coordinates (shape (N, 3), nanometers), centers them
        at the origin, and returns the potential energy as a float in the standard energy unit. The
        energy is evaluated in vacuum (non-periodic boundary conditions), so it measures the
        cohesion of the lattice itself and not of its periodic images.

        :param types:
            The type label for each particle.
        :type types: list[str]
        :param radii:
            The radii dictionary with the particle types as keys and the radii as values.
        :type radii: dict[str, unit.Quantity]
        :param surface_potentials:
            The surface potentials dictionary with the particle types as keys and the surface
            potentials as values.
        :type surface_potentials: dict[str, unit.Quantity]
        :param masses:
            The masses dictionary with the particle types as keys and the masses as values.
        :type masses: dict[str, unit.Quantity]
        :param run_parameters:
            The run parameters.
        :type run_parameters: RunParameters

        :return:
            A function mapping Cartesian coordinates to the potential energy as a float.
        :rtype: Callable[[np.ndarray], float]
        """
        system = openmm.System()

        potentials_parameters = ColloidPotentialsParameters(
            brush_density=run_parameters.brush_density, brush_length=run_parameters.brush_length,
            debye_length=run_parameters.debye_length, temperature=run_parameters.potential_temperature,
            dielectric_constant=run_parameters.dielectric_constant)

        colloid_potentials = ColloidPotentialsAlgebraic(
            colloid_potentials_parameters=potentials_parameters, use_log=run_parameters.use_log,
            cutoff_factor=run_parameters.cutoff_factor, periodic_boundary_conditions=False,
            steric_radius_average=run_parameters.steric_radius_average,
            electrostatic_radius_average=run_parameters.electrostatic_radius_average)

        for t in types:
            system.addParticle(masses[t].value_in_unit(mass_unit))
            colloid_potentials.add_particle(radius=radii[t], surface_potential=surface_potentials[t])

        for force in colloid_potentials.yield_potentials():
            force.setForceGroup(system.getNumForces())
            system.addForce(force)

        assert not system.usesPeriodicBoundaryConditions()
        platform = openmm.Platform.getPlatformByName("Reference")
        dummy_integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(system, dummy_integrator, platform)

        def energy_of(positions: np.ndarray) -> float:
            centered = positions - positions.mean(axis=0)
            context.setPositions(centered * length_unit)
            state = context.getState(getEnergy=True)
            return float(state.getPotentialEnergy().value_in_unit(energy_unit))

        return energy_of

    @staticmethod
    def _axis_groups(structure: Structure, tolerance: float = 1e-6) -> list[list[int]]:
        """
        Group lattice vector indices that the structure's symmetry requires to scale together.

        The crystal system fixes which axis lengths are constrained to stay equal, so scaling the
        axes within each group by a common factor rescales the cell without lowering its symmetry:

            cubic                    -> [[0, 1, 2]]  (a = b = c; equivalent to a uniform scale)
            trigonal (rhombohedral)  -> [[0, 1, 2]]  (a = b = c)
            hexagonal, tetragonal,
            trigonal (hexagonal ax.) -> [[0, 1], [2]]  (a = b, c free)
            orthorhombic, monoclinic,
            triclinic                -> [[0], [1], [2]]  (all free)

        The space group is determined from the coordinates rather than read from the CIF header, so
        a file written in P1 -- as symmetry-expanded CIFs commonly are -- still yields its true
        symmetry. Groups are indexed by lattice vector, which is why they apply unchanged to a
        supercell: repeating along an axis scales that vector but does not change its direction or
        which other axes it is equivalent to.

        If symmetry cannot be determined the grouping falls back to equal lattice vector lengths,
        which preserves symmetry whenever the cell is already in a conventional setting.

        :param structure:
            The structure whose symmetry determines the grouping. Pass the unit cell, not a
            supercell: anisotropic lattice repeats can make equivalent axes unequal in length and
            obscure the symmetry.
        :type structure: Structure
        :param tolerance:
            Relative tolerance within which two lattice vector lengths count as equal, used only by
            the fallback.
        :type tolerance: float

        :return:
            A list of groups, each a list of lattice vector indices.
        :rtype: list[list[int]]
        """
        try:
            crystal_system = SpacegroupAnalyzer(structure).get_crystal_system()
        except Exception:
            crystal_system = None

        if crystal_system in ("cubic",):
            return [[0, 1, 2]]
        if crystal_system in ("hexagonal", "tetragonal", "trigonal"):
            # A trigonal cell may be given on rhombohedral axes (a = b = c) or on the more common
            # hexagonal axes (a = b, c free); the lengths tell the two settings apart.
            lengths = np.linalg.norm(structure.lattice.matrix, axis=1)
            if crystal_system == "trigonal" and np.allclose(lengths, lengths[0], rtol=tolerance):
                return [[0, 1, 2]]
            return [[0, 1], [2]]
        if crystal_system in ("orthorhombic", "monoclinic", "triclinic"):
            return [[0], [1], [2]]

        lengths = np.linalg.norm(structure.lattice.matrix, axis=1)
        groups: list[list[int]] = []
        for index, length in enumerate(lengths):
            for group in groups:
                if abs(length - lengths[group[0]]) <= tolerance * max(length, lengths[group[0]]):
                    group.append(index)
                    break
            else:
                groups.append([index])
        return groups

    @staticmethod
    def _optimize_anisotropic_scale_factors(frac_coords: np.ndarray, lattice_matrix: np.ndarray,
                                            types: list[str], radii: dict[str, unit.Quantity],
                                            surface_potentials: dict[str, unit.Quantity],
                                            masses: dict[str, unit.Quantity],
                                            run_parameters: RunParameters,
                                            geometric_scale_factor: float,
                                            scale_range: tuple[float, float],
                                            scale_samples: int,
                                            symmetry_structure: Structure) -> np.ndarray:
        """
        Find per-lattice-vector scale factors that minimize the steric + electrostatic energy.

        Starting from the isotropic geometric scale factor, each group of equivalent axes (see
        _axis_groups) is scanned in turn while the others are held fixed, and the best value is
        kept. Sweeps repeat until no group moves, or until a sweep cap is reached. This coordinate
        descent costs O(sweeps * groups * scale_samples) energy evaluations instead of the
        scale_samples ** 3 of a dense grid, which matters because each evaluation is a full
        pairwise energy on the Reference platform.

        Coordinate descent finds a local minimum. That is the intended behavior here: the search
        starts from the geometric non-overlap scale, so it descends into the basin belonging to the
        structure as specified, rather than wandering to a different packing.

        :param frac_coords:
            The fractional coordinates of the particles, shape (N, 3).
        :type frac_coords: np.ndarray
        :param lattice_matrix:
            The unscaled lattice matrix whose rows are the lattice vectors.
        :type lattice_matrix: np.ndarray
        :param types:
            The type label for each particle.
        :type types: list[str]
        :param radii:
            The radii dictionary with the particle types as keys and the radii as values.
        :type radii: dict[str, unit.Quantity]
        :param surface_potentials:
            The surface potentials dictionary with the particle types as keys and the surface
            potentials as values.
        :type surface_potentials: dict[str, unit.Quantity]
        :param masses:
            The masses dictionary with the particle types as keys and the masses as values.
        :type masses: dict[str, unit.Quantity]
        :param run_parameters:
            The run parameters.
        :type run_parameters: RunParameters
        :param geometric_scale_factor:
            The geometric (overlap-avoidance) scale factor, used as the isotropic starting point.
        :type geometric_scale_factor: float
        :param scale_range:
            (min_factor, max_factor) relative to the geometric scale factor.
        :type scale_range: tuple[float, float]
        :param scale_samples:
            The number of uniformly spaced scale factors to evaluate per axis group per sweep.
        :type scale_samples: int
        :param symmetry_structure:
            The unit cell whose symmetry decides which axes are scaled together (see _axis_groups).
        :type symmetry_structure: Structure

        :return:
            The optimal scale factor for each of the three lattice vectors, shape (3,).
        :rtype: np.ndarray

        :raises RuntimeError:
            If no valid scale factors are found (all candidates are infinite or NaN).
        """
        max_sweeps = 10
        energy_of = LatticeBuilder._vacuum_energy_function(
            types=types, radii=radii, surface_potentials=surface_potentials, masses=masses,
            run_parameters=run_parameters)

        def energy_at(factors: np.ndarray) -> float:
            return energy_of(frac_coords @ (factors[:, np.newaxis] * lattice_matrix))

        scale_factors = np.full(3, float(geometric_scale_factor))
        best_energy = energy_at(scale_factors)
        if not np.isfinite(best_energy):
            # The isotropic starting point overlaps; fall back to the widest finite candidate so the
            # descent has somewhere to start from.
            for trial in np.linspace(geometric_scale_factor, geometric_scale_factor * scale_range[1],
                                     scale_samples):
                candidate_energy = energy_at(np.full(3, float(trial)))
                if np.isfinite(candidate_energy):
                    scale_factors = np.full(3, float(trial))
                    best_energy = candidate_energy
                    break
            else:
                raise RuntimeError(
                    "No valid scale factors found. All candidates had infinite or NaN energy.")

        groups = LatticeBuilder._axis_groups(symmetry_structure)
        candidates = np.linspace(geometric_scale_factor * scale_range[0],
                                 geometric_scale_factor * scale_range[1], scale_samples)
        hit_boundary = False
        for _ in range(max_sweeps):
            improved = False
            for group in groups:
                trial_factors = scale_factors.copy()
                group_best_value, group_best_energy, group_finite = scale_factors[group[0]], best_energy, []
                for candidate in candidates:
                    trial_factors[group] = candidate
                    candidate_energy = energy_at(trial_factors)
                    if not np.isfinite(candidate_energy):
                        continue
                    group_finite.append(candidate)
                    if candidate_energy < group_best_energy:
                        group_best_value, group_best_energy = float(candidate), candidate_energy
                trial_factors[group] = group_best_value
                if group_best_energy < best_energy:
                    scale_factors, best_energy, improved = trial_factors.copy(), group_best_energy, True
                else:
                    trial_factors[group] = scale_factors[group[0]]
                if group_finite and group_best_value in (group_finite[0], group_finite[-1]):
                    hit_boundary = True
            if not improved:
                break

        if hit_boundary:
            warnings.warn(f"Energy minimum at boundary of scan range (scale factors="
                          f"{np.array2string(scale_factors, precision=4)}). "
                          f"Consider widening energy_scale_range.")
        return scale_factors

    def generate_configuration(self) -> Frame:
        """
        Generate the initial positions of the colloids in a gsd.hoomd.Frame instance.

        :return:
            The initial configuration of the colloids.
        :rtype: gsd.hoomd.Frame
        """
        # Find optimal scale factor using a small test supercell to speed up the search.
        small_supercell = self._structure.make_supercell((3, 3, 3), in_place=False, to_unit_cell=True)
        dists = distance_matrix(small_supercell.cart_coords, small_supercell.cart_coords)
        effective_radii = [self._radii[self._type_map[atomic_number]].value_in_unit(length_unit)
                           + self._brush_length.value_in_unit(length_unit)
                           + self._radii_padding.value_in_unit(length_unit)
                           for atomic_number in small_supercell.atomic_numbers]
        required_scale_factor = 0.0
        for i in range(len(effective_radii)):
            for j in range(i + 1, len(effective_radii)):
                required_scale_factor = max(required_scale_factor,
                                            (effective_radii[i] + effective_radii[j]) / dists[i, j])

        # Apply scale factor to the full supercell.
        structure_full = self._structure.make_supercell(self._lattice_repeats, in_place=False)
        types = [self._type_map[atomic_number] for atomic_number in structure_full.atomic_numbers]

        # Optionally optimize the scale factor by energy minimization. scale_factors holds one
        # factor per lattice vector; the uniform case simply repeats the same value three times, so
        # everything downstream can treat the two cases identically.
        scale_factors = np.full(3, float(required_scale_factor))
        if self._optimize_energy and self._anisotropic_energy:
            scale_factors = self._optimize_anisotropic_scale_factors(
                frac_coords=structure_full.frac_coords, lattice_matrix=structure_full.lattice.matrix,
                types=types, radii=self._radii, surface_potentials=self._surface_potentials,
                masses=self._masses, run_parameters=self._run_parameters,
                geometric_scale_factor=required_scale_factor, scale_range=self._energy_scale_range,
                scale_samples=self._energy_scale_samples, symmetry_structure=self._structure)
        elif self._optimize_energy:
            # noinspection PyTypeChecker
            required_scale_factor = self._optimize_scale_factor(
                base_cart_coords=structure_full.cart_coords, types=types, radii=self._radii,
                surface_potentials=self._surface_potentials, masses=self._masses,
                run_parameters=self._run_parameters, geometric_scale_factor=required_scale_factor,
                scale_range=self._energy_scale_range, scale_samples=self._energy_scale_samples)
            scale_factors = np.full(3, float(required_scale_factor))

        # Scale each lattice vector by its own factor. For the uniform case this reproduces the
        # previous behavior exactly, since frac_coords @ (s * matrix) == s * cart_coords.
        scaled_lattice = scale_factors[:, np.newaxis] * structure_full.lattice.matrix

        # --- Build the Frame ---
        frame = Frame()
        frame.particles.types = sorted(self.types())
        frame.particles.typeid = np.array(
            [frame.particles.types.index(t) for t in types], dtype=np.uint32)

        if self._periodic:
            # Bulk periodic crystal: the simulation box is the uniformly scaled supercell lattice,
            # and the particles are placed at their fractional coordinates within that box. This
            # tiles space under periodic boundary conditions (no surrounding vacuum, lattice_padding
            # is ignored).
            box, hoomd_matrix = self._lattice_to_hoomd_box(scaled_lattice)
            # Wrap fractional coordinates into [0, 1), then shift to [-0.5, 0.5) so the crystal is
            # centered on the origin. The HOOMD/GSD box is centered at the origin (spanning
            # [-L/2, L/2)); placing particles at the raw fractional coordinates would offset the whole
            # crystal by half a box length in each direction.
            fractional_coordinates = (structure_full.frac_coords % 1.0) - 0.5
            positions = fractional_coordinates @ hoomd_matrix
            frame.particles.N = len(positions)
            frame.particles.position = np.array(positions, dtype=np.float32)
            frame.configuration.box = np.array(box, dtype=np.float32)
            return frame

        positions = structure_full.frac_coords @ scaled_lattice

        # Center at origin.
        positions -= positions.mean(axis=0)
        effective_radii = [self._radii[t].value_in_unit(length_unit)
                           + self._brush_length.value_in_unit(length_unit)
                           + self._radii_padding.value_in_unit(length_unit)
                           for t in types]

        # Embed in cubic box with padding.
        box_length = 2.0 * (float(np.max(np.abs(positions))) + float(np.max(effective_radii))
                            + self._lattice_padding.value_in_unit(length_unit))

        frame.particles.N = len(positions)
        frame.particles.position = np.array(positions, dtype=np.float32)
        frame.configuration.box = np.array([box_length, box_length, box_length, 0.0, 0.0, 0.0], dtype=np.float32)

        return frame

    @staticmethod
    def _lattice_to_hoomd_box(lattice_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a lattice matrix (rows = lattice vectors a, b, c) to the HOOMD/GSD box convention.

        The GSD box is [Lx, Ly, Lz, xy, xz, yz], where (xy, xz, yz) are the dimensionless tilt
        factors and the box vectors in the rotated frame are the rows of the returned lower-triangular
        matrix (see helper_functions.get_cell_from_box):
        a = (Lx, 0, 0), b = (Ly*xy, Ly, 0), c = (Lz*xz, Lz*yz, Lz).

        Because the box is expressed in a rotated frame, particle positions must be reconstructed from
        their (rotation-invariant) fractional coordinates using the returned matrix.

        :param lattice_matrix:
            The lattice matrix with the lattice vectors a, b, c as its rows.
        :type lattice_matrix: np.ndarray

        :return:
            A tuple (box, hoomd_matrix) where box is the length-6 GSD box array and hoomd_matrix is
            the 3x3 lower-triangular matrix whose rows are the box vectors in the rotated frame.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        a, b, c = lattice_matrix[0], lattice_matrix[1], lattice_matrix[2]
        lx = np.linalg.norm(a)
        a_hat = a / lx
        xy_component = np.dot(b, a_hat)
        ly = np.sqrt(np.dot(b, b) - xy_component ** 2)
        xz_component = np.dot(c, a_hat)
        yz_component = (np.dot(b, c) - xy_component * xz_component) / ly
        lz = np.sqrt(np.dot(c, c) - xz_component ** 2 - yz_component ** 2)
        hoomd_matrix = np.array([[lx, 0.0, 0.0],
                                 [xy_component, ly, 0.0],
                                 [xz_component, yz_component, lz]])
        box = np.array([lx, ly, lz, xy_component / ly, xz_component / lz, yz_component / lz])
        return box, hoomd_matrix
