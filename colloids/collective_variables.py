from typing import Optional, Sequence
from abc import abstractmethod, ABC
from dataclasses import dataclass

import openmm
import openmm.app
from openmm import unit

from colloids.helper_functions import get_cell_from_box
from colloids.units import length_unit

from openmm import CustomGBForce, CustomCVForce

try:
    from openmmtorch import TorchForce
    import torch
    from .rsh import rsh_cart_6
except ModuleNotFoundError:
    pass

_EPSILON = 1e-12

    
class SwitchingFunctions:

    @staticmethod
    def get_exponential_switching_function_str(r="r", d0="d0", r0="r0", dmax=None):
        s_r =  f"1 / (1 + exp(({r} - {d0}) / {r0}))"

        if dmax is not None:
            s_r = f"step({dmax} - {r}) * ({s_r})"

        return s_r
    
    @staticmethod
    def get_rational_switching_function_str(nn=12, mm=24, r="r", d0="d0", r0="r0"):
        
        x = f"({r} - {d0}) / {r0}"
        
        s_r = f"(1 - ({x})^{nn}) / (1 - ({x})^{mm} + eps)"

        return s_r

    @staticmethod
    def get_more_than_str(x="x", threshold="threshold", nn=12):

        mt = f"({x})^{nn}/(({threshold})^{nn} + ({x})^{nn} + eps)"

        return mt
    
    @staticmethod
    def get_gaussian_switch_torch(r, r0: float, d0: float):
        x = torch.square(r-d0)/(2*r0**2)
        return torch.exp(-x)
    

class OpenMMCollectiveVariableAbstract(ABC):
    """
    Abstract class for the implementation of custom collective variable forces in OpenMM.
    
    :param system:
        The OpenMM system that this force will be added to.
        The system is used to check the periodic boundary conditions.
    :type simulation: openmm.System
    :param topology:
        The topology of the OpenMM system. This is used to check the particle types. 
    :type topology: openmm.app.Topology
    """

    def __init__(self, topology=openmm.app.Topology, system=openmm.System):
        self.topology = topology
        self.system = system
        self._add_force_called = False
    
    @abstractmethod
    def compute_cv(self) -> openmm.Force:
        if self._add_force_called:
            raise RuntimeError("method compute_cv must be called before a CV-associated " 
                               "force is used")
        self._compute_cv_called = True

    @abstractmethod
    def get_force(self) -> openmm.CustomCVForce:
        #raise NotImplementedError
        self._add_force_called = True
    
class HighCoordCompositionCV(OpenMMCollectiveVariableAbstract):
    
    """
    Custom collective variable defined in OpenMM to determine particles of high coordination and compute the number ratio of a 
    specified target particle type among these coordinated particles. A switching function is applied to make the CV smooth and 
    differentiable.

    :param topology:
        The topology of the OpenMM system. 
    :type topology: openmm.app.Topology
    :param system:
        The OpenMM system that this force will be added to.
    :type simulation: openmm.System
    :param target_particle_type: 
        The particle type name for which the number ratio is being computed. The number of particles of this type
        that satisfy the high coordination requirement will become the numerator of this custom CV expression.
    :type target_particle_type: str
    :param coordination_r0: 
        The decay length of the exponential switching function.
    :type coordination_r0: unit.Quantity
    :param coordination_d0:
        The offset distance (midpoint) of the exponential switching function.
    :type coordination_d0: unit.Quantity
    :param coordination_dmax:
        The max cutoff distance of the exponential switching function. Beyond this distance,
        the value of the switch is zero.
    :type coordination_dmax: unit.Quantity
    :param highcoord_threshold: 
        The minimum coordination number particles must have to be counted. 
    :type highcoord_threshold: float
    :param ignore_types:
        A list of particle types to ignore when counting high-coordination particles.
        If None, all particle types are considered.
        Defaults to None.
    :type ignore_types: Optional[Sequence[str]] 

    :raises TypeError:
        If coordination_r0, coordination_d0, or coordination_dmax are not Quantities with 
        proper length units.
    :raises ValueError:
        If coordination_r0, coordination_d0, or coordination_dmax are not greater than 0.
        If the highcoord_threshold is not greater than 0.
        If the target_particle_type is not a valid particle type in the OpenMM system.     
    """

    def __init__(self, topology: openmm.app.Topology, system: openmm.System, target_particle_type: str, coordination_d0: unit.Quantity, 
                coordination_r0: unit.Quantity, coordination_dmax: unit.Quantity, highcoord_threshold: float, 
                ignore_types: Optional[Sequence[str]] = None):
        self._uses_pbc = system.usesPeriodicBoundaryConditions()
        self._particle_types = [atom.name for atom in topology.atoms()]
        if target_particle_type not in self._particle_types:
            raise ValueError("The target particle for which high coordination",
            "count is being determined is not in the OpenMM system")
        
        if highcoord_threshold <=0:
            raise ValueError("The high coordination threshold must be a positive value.")
        if not all(param.unit.is_compatible(length_unit) for param in (coordination_r0, coordination_d0, coordination_dmax)):
            raise TypeError("Switching function parameters must all have a unit that is compatible with nanometers")
        if coordination_r0.value_in_unit(length_unit) <=0 or coordination_d0.value_in_unit(length_unit) <=0 or coordination_dmax.value_in_unit(length_unit) <=0: 
            raise ValueError("Switching function parameters must all be positive values")

        self._nn_high_coord = 12
        self._highcoord_threshold = highcoord_threshold
        
        self._is_target = []
        self._target_particle_type = target_particle_type
    
        self._ignore_types = ignore_types
        self._include_particles = []
        
        for particle_type in self._particle_types:
            if self._ignore_types is None:
                include_particle = 1
            else:
                include_particle = int(particle_type not in self._ignore_types)

            self._include_particles.append(include_particle)
            self._is_target.append(int(particle_type == self._target_particle_type))

        
        self._coordination_d0 = coordination_d0
        self._coordination_r0 =  coordination_r0
        self._coordination_dmax = coordination_dmax


    def compute_cv(self, name: str, weight_parameter: str):
        force = CustomGBForce()
        force.setName(name)

        force.addGlobalParameter(
            "coordination_d0",
            self._coordination_d0.value_in_unit(length_unit),
        )
        force.addGlobalParameter(
            "coordination_r0",
            self._coordination_r0.value_in_unit(length_unit),
        )
        force.addGlobalParameter(
            "coordination_dmax",
            self._coordination_dmax.value_in_unit(length_unit),
        )

        force.addGlobalParameter("highcoord_threshold", self._highcoord_threshold)
        force.addGlobalParameter("nn", self._nn_high_coord)
        force.addGlobalParameter("eps", _EPSILON)

        force.addPerParticleParameter("include_particles")
        force.addPerParticleParameter("is_target")

        s_r_exp= SwitchingFunctions.get_exponential_switching_function_str("r", d0 = "coordination_d0",
                                                                 r0 = "coordination_r0", dmax = "coordination_dmax")
        
        high_coord = SwitchingFunctions.get_more_than_str(x="coord", threshold="highcoord_threshold", nn="nn")

        force.addComputedValue(
            "coord",
            f"include_particles2 * ({s_r_exp})",
            CustomGBForce.ParticlePairNoExclusions,
        )

        force.addEnergyTerm(
            f"{weight_parameter} * ({high_coord})",
            CustomGBForce.SingleParticle,
            )


        for include_particle, is_target in zip(self._include_particles,
                                                self._is_target):
            force.addParticle([include_particle, is_target])

        if self._uses_pbc:
            force.setNonbondedMethod(CustomGBForce.CutoffPeriodic)
        else:
            force.setNonbondedMethod(CustomGBForce.CutoffNonPeriodic)

        force.setCutoffDistance(
            self._coordination_dmax.value_in_unit(length_unit)
            )

        return force

    def get_force(self):

        target_highcoord = self.compute_cv(
            name="target_highcoord",
            weight_parameter="include_particles * is_target",
            )

        all_highcoord = self.compute_cv(
                name="all_highcoord",
            weight_parameter="include_particles",
            )
        
        x_force = CustomCVForce("x_i;"
                                 "x_i = target_highcoord / (all_highcoord + eps)"
                                 )
        
        x_force.setName("highcoord_composition_cv")

        x_force.addGlobalParameter("eps", _EPSILON)
        x_force.addCollectiveVariable("target_highcoord", target_highcoord)
        x_force.addCollectiveVariable("all_highcoord", all_highcoord)
        
        return x_force
    

class HighCoordDensityCV(OpenMMCollectiveVariableAbstract):
    """
    Smooth density CV: counts particles in locally dense/high-coordination environments.

    N_p = sum_i s(n_i - n_threshold)

    where n_i is a smooth coordination number around particle i.
    """

    def __init__(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        coordination_d0: unit.Quantity,
        coordination_r0: unit.Quantity,
        coordination_dmax: unit.Quantity,
        highcoord_threshold: float,
        ignore_types: Optional[Sequence[str]] = None,
    ):
        super().__init__(topology=topology, system=system)

        self._uses_pbc = system.usesPeriodicBoundaryConditions()
        self._particle_types = [atom.name for atom in topology.atoms()]

        if highcoord_threshold <= 0:
            raise ValueError("The high coordination threshold must be positive.")

        if not all(
            param.unit.is_compatible(length_unit)
            for param in (coordination_r0, coordination_d0, coordination_dmax)
        ):
            raise TypeError("Switching function parameters must have length units.")

        if (
            coordination_r0.value_in_unit(length_unit) <= 0
            or coordination_d0.value_in_unit(length_unit) <= 0
            or coordination_dmax.value_in_unit(length_unit) <= 0
        ):
            raise ValueError("Switching function parameters must be positive.")

        self._coordination_d0 = coordination_d0
        self._coordination_r0 = coordination_r0
        self._coordination_dmax = coordination_dmax
        self._highcoord_threshold = highcoord_threshold
        self._nn_high_coord = 12

        self._ignore_types = ignore_types
        self._include_particles = []

        for particle_type in self._particle_types:
            if self._ignore_types is None:
                include_particle = 1
            else:
                include_particle = int(particle_type not in self._ignore_types)

            self._include_particles.append(include_particle)

    def compute_cv(self, name: str = "highcoord_density") -> openmm.Force:
        force = CustomGBForce()
        force.setName(name)

        force.addGlobalParameter(
            "coordination_d0",
            self._coordination_d0.value_in_unit(length_unit),
        )
        force.addGlobalParameter(
            "coordination_r0",
            self._coordination_r0.value_in_unit(length_unit),
        )
        force.addGlobalParameter(
            "coordination_dmax",
            self._coordination_dmax.value_in_unit(length_unit),
        )
        force.addGlobalParameter("highcoord_threshold", self._highcoord_threshold)
        force.addGlobalParameter("nn", self._nn_high_coord)
        force.addGlobalParameter("eps", _EPSILON)

        force.addPerParticleParameter("include_particles")

        s_r_exp = SwitchingFunctions.get_exponential_switching_function_str(
            r="r",
            d0="coordination_d0",
            r0="coordination_r0",
            dmax="coordination_dmax",
        )

        high_coord = SwitchingFunctions.get_more_than_str(
            x="coord",
            threshold="highcoord_threshold",
            nn="nn",
        )

        # coord_i = smooth count of included neighbors around particle i.
        force.addComputedValue(
            "coord",
            f"include_particles2 * ({s_r_exp})",
            CustomGBForce.ParticlePairNoExclusions,
        )

        # N_p = sum_i include_i * smooth_indicator(coord_i > threshold)
        force.addEnergyTerm(
            f"include_particles * ({high_coord})",
            CustomGBForce.SingleParticle,
        )

        for include_particle in self._include_particles:
            force.addParticle([include_particle])

        if self._uses_pbc:
            force.setNonbondedMethod(CustomGBForce.CutoffPeriodic)
        else:
            force.setNonbondedMethod(CustomGBForce.CutoffNonPeriodic)

        force.setCutoffDistance(
            self._coordination_dmax.value_in_unit(length_unit)
        )

        return force

    def get_force(self) -> openmm.CustomCVForce:
        highcoord_density = self.compute_cv(name="highcoord_density_raw")

        cv_force = CustomCVForce("highcoord_density")
        cv_force.setName("highcoord_density_cv")
        cv_force.addCollectiveVariable("highcoord_density", highcoord_density)

        return cv_force
    
    
class XPositionCV(OpenMMCollectiveVariableAbstract):
    """Simple scalar CV for tests: x position of one particle."""

    def __init__(self, topology=None, system=None, particle_index: int = 0):
        super().__init__(topology=topology, system=system)

    def compute_cv(self) -> openmm.Force:
        force = openmm.CustomExternalForce("x")
        force.setName("x_position_cv")
        force.addParticle(self._particle_index, [])
        return force

    def get_force(self) -> openmm.Force:
        return self.compute_cv()
    
class ExactSortedPIVDistanceModule(torch.nn.Module):
    """
    Exact sorted-PIV squared distance to one reference structure.

    Returns:
        D(X, X_ref) = sum_k (PIV_k(X) - PIV_k(X_ref))^2
    """

    def __init__(
        self,
        pair_i,
        pair_j,
        block_id,
        reference_piv,
        r0_by_block,
        box_lengths,
        nn=6,
        mm=12,
        use_pbc=True,
    ):
        super().__init__()

        self.register_buffer("pair_i", torch.as_tensor(pair_i, dtype=torch.long))
        self.register_buffer("pair_j", torch.as_tensor(pair_j, dtype=torch.long))
        self.register_buffer("block_id", torch.as_tensor(block_id, dtype=torch.long))
        self.register_buffer("reference_piv", torch.as_tensor(reference_piv, dtype=torch.float32))
        self.register_buffer("r0_by_block", torch.as_tensor(r0_by_block, dtype=torch.float32))
        self.register_buffer("box_lengths", torch.as_tensor(box_lengths, dtype=torch.float32))

        self.nn = nn
        self.mm = mm
        self.use_pbc = use_pbc
        self.n_blocks = len(r0_by_block)

    def rational_switch(self, r, r0):
        x = r / r0
        numerator = 1.0 - x**self.nn
        denominator = 1.0 - x**self.mm

        # Limit at r == r0 is NN/MM.
        limit_value = float(self.nn) / float(self.mm)
        near_singular = torch.abs(denominator) < 1.0e-6
        safe_denominator = torch.where(
            near_singular,
            torch.ones_like(denominator),
            denominator,
        )
        value = numerator / safe_denominator
        value = torch.where(near_singular, torch.full_like(value, limit_value), value)

        return torch.clamp(value, min=0.0, max=1.0)

    def minimum_image(self, displacements):
        # Assumes orthorhombic/cubic boxes, which matches the current colloid setup.
        return displacements - self.box_lengths * torch.round(displacements / self.box_lengths)

    def forward(self, positions):
        positions = positions.float()

        displacements = positions[self.pair_j] - positions[self.pair_i]

        if self.use_pbc:
            displacements = self.minimum_image(displacements)

        distances = torch.linalg.norm(displacements, dim=1)

        r0 = self.r0_by_block[self.block_id]
        switched = self.rational_switch(distances, r0)

        sorted_blocks = []
        for block in range(self.n_blocks):
            block_values = switched[self.block_id == block]

            # Match the Python classifier sorting convention.
            sorted_values = torch.sort(block_values).values
            sorted_blocks.append(sorted_values)

        piv = torch.cat(sorted_blocks)

        delta = piv - self.reference_piv
        return torch.sum(delta * delta)


class ExactSortedPIVDifferenceModule(torch.nn.Module):
    """
    Exact sorted-PIV squared-distance difference between two references.

    Returns:
        D(X, X_positive_ref) - D(X, X_negative_ref)
    """

    def __init__(
        self,
        pair_i,
        pair_j,
        block_id,
        positive_reference_piv,
        negative_reference_piv,
        r0_by_block,
        box_lengths,
        nn=6,
        mm=12,
        use_pbc=True,
    ):
        super().__init__()

        self.register_buffer("pair_i", torch.as_tensor(pair_i, dtype=torch.long))
        self.register_buffer("pair_j", torch.as_tensor(pair_j, dtype=torch.long))
        self.register_buffer("block_id", torch.as_tensor(block_id, dtype=torch.long))
        self.register_buffer(
            "positive_reference_piv",
            torch.as_tensor(positive_reference_piv, dtype=torch.float32),
        )
        self.register_buffer(
            "negative_reference_piv",
            torch.as_tensor(negative_reference_piv, dtype=torch.float32),
        )
        self.register_buffer("r0_by_block", torch.as_tensor(r0_by_block, dtype=torch.float32))
        self.register_buffer("box_lengths", torch.as_tensor(box_lengths, dtype=torch.float32))

        self.nn = nn
        self.mm = mm
        self.use_pbc = use_pbc
        self.n_blocks = len(r0_by_block)

    def rational_switch(self, r, r0):
        x = r / r0
        numerator = 1.0 - x**self.nn
        denominator = 1.0 - x**self.mm

        # Limit at r == r0 is NN/MM.
        limit_value = float(self.nn) / float(self.mm)
        near_singular = torch.abs(denominator) < 1.0e-6
        safe_denominator = torch.where(
            near_singular,
            torch.ones_like(denominator),
            denominator,
        )
        value = numerator / safe_denominator
        value = torch.where(near_singular, torch.full_like(value, limit_value), value)

        return torch.clamp(value, min=0.0, max=1.0)

    def minimum_image(self, displacements):
        # Assumes orthorhombic/cubic boxes, which matches the current colloid setup.
        return displacements - self.box_lengths * torch.round(displacements / self.box_lengths)

    def forward(self, positions):
        positions = positions.float()

        displacements = positions[self.pair_j] - positions[self.pair_i]

        if self.use_pbc:
            displacements = self.minimum_image(displacements)

        distances = torch.linalg.norm(displacements, dim=1)

        r0 = self.r0_by_block[self.block_id]
        switched = self.rational_switch(distances, r0)

        sorted_blocks = []
        for block in range(self.n_blocks):
            block_values = switched[self.block_id == block]

            # Match the Python classifier sorting convention.
            sorted_values = torch.sort(block_values).values
            sorted_blocks.append(sorted_values)

        piv = torch.cat(sorted_blocks)

        positive_delta = piv - self.positive_reference_piv
        negative_delta = piv - self.negative_reference_piv
        positive_distance = torch.sum(positive_delta * positive_delta)
        negative_distance = torch.sum(negative_delta * negative_delta)
        return positive_distance - negative_distance
	    
def build_piv_pair_index(particle_type_names, particle_type_a, particle_type_b):
    pair_i = []
    pair_j = []
    block_id = []

    n_particles = len(particle_type_names)

    for i in range(n_particles - 1):
        for j in range(i + 1, n_particles):
            ti = particle_type_names[i]
            tj = particle_type_names[j]

            if ti == particle_type_a and tj == particle_type_a:
                block = 0
            elif ti == particle_type_b and tj == particle_type_b:
                block = 1
            elif {ti, tj} == {particle_type_a, particle_type_b}:
                block = 2
            else:
                continue

            pair_i.append(i)
            pair_j.append(j)
            block_id.append(block)

    return pair_i, pair_j, block_id

def load_reference_gsd(reference_file):
    import gsd.hoomd
    import numpy as np

    with gsd.hoomd.open(reference_file, "r") as traj:
        frame = traj[-1]
        positions = np.asarray(frame.particles.position, dtype=float)
        box_lengths = np.asarray(frame.configuration.box[:3], dtype=float)

    return positions, box_lengths


def rational_switch_numpy(distances, r0, nn, mm):
    import numpy as np

    x = distances / r0
    numerator = 1.0 - x**nn
    denominator = 1.0 - x**mm

    values = numerator / np.where(np.abs(denominator) < 1.0e-12, np.nan, denominator)
    values = np.where(np.abs(denominator) < 1.0e-12, nn / mm, values)

    return np.clip(values, 0.0, 1.0)


def compute_reference_piv_numpy(
    positions,
    box_lengths,
    pair_i,
    pair_j,
    block_id,
    r0_by_block,
    nn,
    mm,
    sort_blocks=True,
):
    import numpy as np

    pair_i = np.asarray(pair_i, dtype=int)
    pair_j = np.asarray(pair_j, dtype=int)
    block_id = np.asarray(block_id, dtype=int)

    displacements = positions[pair_j] - positions[pair_i]
    displacements -= box_lengths * np.rint(displacements / box_lengths)

    distances = np.linalg.norm(displacements, axis=1)

    piv_blocks = []
    for block in range(3):
        block_distances = distances[block_id == block]
        block_values = rational_switch_numpy(
            block_distances,
            r0=r0_by_block[block],
            nn=nn,
            mm=mm,
        )

        if sort_blocks:
            block_values = np.sort(block_values)

        piv_blocks.append(block_values)

    return np.concatenate(piv_blocks)


def get_orthorhombic_box_lengths(system):
    vectors = system.getDefaultPeriodicBoxVectors()
    box_lengths = []

    for axis, vector in enumerate(vectors):
        component = vector[axis]
        if hasattr(component, "value_in_unit"):
            component = component.value_in_unit(length_unit)
        box_lengths.append(float(component))

    return box_lengths
    
class ExactSortedPIVDistanceCV(OpenMMCollectiveVariableAbstract):
    def __init__(
        self,
        topology,
        system,
        reference_file,
        particle_type_a,
        particle_type_b,
        switch_r0,
        switch_nn=6,
        switch_mm=12,
        sort_blocks=True,
    ):
        super().__init__(topology=topology, system=system)

        self._uses_pbc = system.usesPeriodicBoundaryConditions()

        particle_type_names = [atom.name for atom in topology.atoms()]

        pair_i, pair_j, block_id = build_piv_pair_index(
            particle_type_names=particle_type_names,
            particle_type_a=particle_type_a,
            particle_type_b=particle_type_b,
        )

        reference_positions, reference_box = load_reference_gsd(reference_file)
        box_lengths = get_orthorhombic_box_lengths(system)

        reference_piv = compute_reference_piv_numpy(
            positions=reference_positions,
            box_lengths=reference_box,
            pair_i=pair_i,
            pair_j=pair_j,
            block_id=block_id,
            r0_by_block=switch_r0,
            nn=switch_nn,
            mm=switch_mm,
            sort_blocks=sort_blocks,
        )

        self._module = ExactSortedPIVDistanceModule(
            pair_i=pair_i,
            pair_j=pair_j,
            block_id=block_id,
            reference_piv=reference_piv,
            r0_by_block=switch_r0,
            box_lengths=box_lengths,
            nn=switch_nn,
            mm=switch_mm,
            use_pbc=self._uses_pbc,
        )

    def compute_cv(self):
        scripted = torch.jit.script(self._module)

        torch_force = TorchForce(scripted)
        torch_force.setUsesPeriodicBoundaryConditions(self._uses_pbc)

        cv_force = openmm.CustomCVForce("piv_D")
        cv_force.setName("exact_sorted_piv_distance_cv")
        cv_force.addCollectiveVariable("piv_D", torch_force)

        return cv_force

    def get_force(self):
        return self.compute_cv()


class ExactSortedPIVDifferenceCV(OpenMMCollectiveVariableAbstract):
    """
    Exact sorted-PIV distance-difference CV.

    The returned scalar is:
        D(X, positive_reference_file) - D(X, negative_reference_file)

    For example, positive_reference_file=Th3P4-like.gsd and
    negative_reference_file=blob.gsd gives d_th3p4_blob.
    """

    def __init__(
        self,
        topology,
        system,
        positive_reference_file,
        negative_reference_file,
        particle_type_a,
        particle_type_b,
        switch_r0,
        switch_nn=6,
        switch_mm=12,
        sort_blocks=True,
    ):
        super().__init__(topology=topology, system=system)

        self._uses_pbc = system.usesPeriodicBoundaryConditions()

        particle_type_names = [atom.name for atom in topology.atoms()]

        pair_i, pair_j, block_id = build_piv_pair_index(
            particle_type_names=particle_type_names,
            particle_type_a=particle_type_a,
            particle_type_b=particle_type_b,
        )

        positive_positions, positive_box = load_reference_gsd(positive_reference_file)
        negative_positions, negative_box = load_reference_gsd(negative_reference_file)
        box_lengths = get_orthorhombic_box_lengths(system)

        positive_reference_piv = compute_reference_piv_numpy(
            positions=positive_positions,
            box_lengths=positive_box,
            pair_i=pair_i,
            pair_j=pair_j,
            block_id=block_id,
            r0_by_block=switch_r0,
            nn=switch_nn,
            mm=switch_mm,
            sort_blocks=sort_blocks,
        )
        negative_reference_piv = compute_reference_piv_numpy(
            positions=negative_positions,
            box_lengths=negative_box,
            pair_i=pair_i,
            pair_j=pair_j,
            block_id=block_id,
            r0_by_block=switch_r0,
            nn=switch_nn,
            mm=switch_mm,
            sort_blocks=sort_blocks,
        )

        self._module = ExactSortedPIVDifferenceModule(
            pair_i=pair_i,
            pair_j=pair_j,
            block_id=block_id,
            positive_reference_piv=positive_reference_piv,
            negative_reference_piv=negative_reference_piv,
            r0_by_block=switch_r0,
            box_lengths=box_lengths,
            nn=switch_nn,
            mm=switch_mm,
            use_pbc=self._uses_pbc,
        )

    def compute_cv(self):
        scripted = torch.jit.script(self._module)

        torch_force = TorchForce(scripted)
        torch_force.setUsesPeriodicBoundaryConditions(self._uses_pbc)

        cv_force = openmm.CustomCVForce("piv_difference")
        cv_force.setName("exact_sorted_piv_difference_cv")
        cv_force.addCollectiveVariable("piv_difference", torch_force)

        return cv_force

    def get_force(self):
        return self.compute_cv()

class SteinhardtOrderModule(torch.nn.Module):
    def __init__(self, num_nbs, order, r0, d0):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self._num_nbs = num_nbs
        self._order = order
        self._r0 = r0
        self._d0 = d0

    def calc_stein_single(self, i: int, pos, device: torch.device):
        disps = pos - pos[i]
        dists = torch.square(disps)
        dists = torch.sum(dists, dim=(1))
        dists = torch.sqrt(dists)
        
        inds_nbs = torch.argsort(dists)[1:self._num_nbs+1]
        disps = pos[inds_nbs] - pos[i]
        dists = torch.sqrt(torch.sum(torch.square(disps), dim=(1)))
        dists = torch.clamp(dists, min=1e-8)
        disps_nbs = disps.T/dists
        disps_nbs = disps_nbs.T
        
        Y_nm_real = rsh_cart_6(disps_nbs, device)

        sigma = SwitchingFunctions.get_gaussian_switch_torch(dists, self._r0, self._d0)
        sigma_sum = torch.sum(sigma)
        sigma_sum = torch.clamp(torch.sum(sigma), min=1e-12)
        q_nm_real = torch.sum(sigma[:,None]*Y_nm_real[:,self._order*(self._order+1)-self._order:self._order*(self._order+1)+self._order+1], dim=(0))/sigma_sum
    
        return q_nm_real

    def calc_stein(self, pos, device: torch.device):
        natoms = len(pos)

        q_nm_i = self.calc_stein_single(0, pos, device)
        q_n = torch.sqrt(torch.sum(q_nm_i*q_nm_i))
        for i in range(1, natoms):
            q_nm_i = self.calc_stein_single(i, pos, device)
            q_n_i = torch.sqrt(torch.sum(q_nm_i*q_nm_i))
            q_n = q_n + q_n_i
        return q_n/natoms

    
    def forward(self, positions):
        positions = positions.float()
        q_n = self.calc_stein(positions, self.device)
        return q_n
    

class SteinhardtOrderCV(OpenMMCollectiveVariableAbstract):
    def __init__(self, num_nbs, order, r0, d0, system, topology=None):
        super().__init__(topology=topology, system=system)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self._num_nbs = num_nbs
        self._order = order
        self._r0 = r0
        self._d0 = d0

        self._uses_pbc = system.usesPeriodicBoundaryConditions()
    
    def compute_cv(self) -> openmm.Force:
        cvmodule = torch.jit.script(SteinhardtOrderModule(self._num_nbs, self._order, self._r0, self._d0))
        cv = TorchForce(cvmodule)
        cv.setUsesPeriodicBoundaryConditions(self._uses_pbc)
        cv_force = openmm.CustomCVForce('cv')
        cv_force.addCollectiveVariable('cv', cv)

        return cv_force

    def get_force(self) -> openmm.Force:
        return self.compute_cv()
         
