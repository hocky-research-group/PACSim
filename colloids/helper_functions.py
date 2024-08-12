from math import acos, cos, pi, sin, sqrt
from typing import Iterator
import ase.io
import gsd.hoomd
import numpy as np
import numpy.typing as npt
import openmm
from openmm import app
from openmm import unit
from scipy.spatial.transform import Rotation


def read_xyz_file(filename: str) -> (list[str], npt.NDArray[float], npt.NDArray[float]):
    if not filename.endswith(".xyz"):
        raise ValueError("The file must have the .xyz extension.")
    atoms = ase.io.read(filename, format="extxyz")
    cell = atoms.get_cell()[:]
    assert cell.shape == (3, 3)
    return atoms.get_chemical_symbols(), atoms.get_positions(), cell


# noinspection PyUnresolvedReferences
def write_gsd_file(filename: str, openmm_simulation: app.Simulation, radius_dict: dict[str, unit.Quantity],
                   surface_potentials_dict: dict[str, unit.Quantity], cell: npt.NDArray[unit.Quantity]) -> None:
    nanometer = unit.nano * unit.meter
    millivolt = unit.milli * unit.volt

    positions = (
        openmm_simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True))
    topology = openmm_simulation.topology
    assert topology.getNumChains() == 1
    assert topology.getNumResidues() == 1
    assert topology.getNumAtoms() == openmm_simulation.system.getNumParticles() == len(positions)
    assert len(cell) == 3
    assert cell[0][1].value_in_unit(nanometer) == 0.0
    assert cell[0][2].value_in_unit(nanometer) == 0.0
    assert cell[1][2].value_in_unit(nanometer) == 0.0

    frame = gsd.hoomd.Frame()
    frame.particles.N = topology.getNumAtoms()
    frame.particles.position = positions.value_in_unit(nanometer)
    # Use a dictionary instead of a set to preserve the order of the types.
    # See https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
    # Works since Python 3.7.
    types_set = list(dict.fromkeys(atom.name for atom in topology.atoms()))
    types = list(types_set)
    assert all(t in radius_dict for t in types)
    frame.particles.types = types
    typeid = [types.index(atom.name) for atom in topology.atoms()]
    frame.particles.typeid = typeid
    frame.particles.type_shapes = [
        {"type": "Sphere", "diameter": 2.0 * radius_dict[t].value_in_unit(nanometer)} for t in types]
    frame.particles.diameter = [
        2.0 * radius_dict[types[typeid[atom_index]]].value_in_unit(nanometer)
        for atom_index in range(topology.getNumAtoms())]
    frame.particles.charge = [
        surface_potentials_dict[types[typeid[atom_index]]].value_in_unit(millivolt)
        for atom_index in range(topology.getNumAtoms())]
    frame.particles.mass = [openmm_simulation.system.getParticleMass(atom_index).value_in_unit(unit.amu)
                            for atom_index in range(topology.getNumAtoms())]
    # See http://docs.openmm.org/7.6.0/userguide/theory/05_other_features.html
    # See https://hoomd-blue.readthedocs.io/en/v2.9.3/box.html
    frame.configuration.box = [
        cell[0][0].value_in_unit(nanometer),
        cell[1][1].value_in_unit(nanometer),
        cell[2][2].value_in_unit(nanometer),
        cell[1][0] / cell[1][1],
        cell[2][0] / cell[2][2],
        cell[2][1] / cell[2][2]
    ]
    with gsd.hoomd.open(name=filename, mode="w") as f:
        f.append(frame)


# noinspection PyUnresolvedReferences
def write_xyz_file(filename: str, openmm_simulation: app.Simulation, cell: npt.NDArray[unit.Quantity]) -> None:
    nanometer = unit.nano * unit.meter
    positions = (
        openmm_simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True))
    positions = positions.value_in_unit(nanometer)
    topology = openmm_simulation.topology
    assert topology.getNumChains() == 1
    assert topology.getNumResidues() == 1
    assert topology.getNumAtoms() == openmm_simulation.system.getNumParticles() == len(positions)
    assert len(list(topology.atoms())) == len(positions)
    assert len(cell) == 3
    assert cell[0][1].value_in_unit(nanometer) == 0.0
    assert cell[0][2].value_in_unit(nanometer) == 0.0
    assert cell[1][2].value_in_unit(nanometer) == 0.0
    box = [cell[0][0].value_in_unit(nanometer),
           cell[1][1].value_in_unit(nanometer),
           cell[2][2].value_in_unit(nanometer),
           cell[1][0] / cell[1][1],
           cell[2][0] / cell[2][2],
           cell[2][1] / cell[2][2]]
    with open(filename, "w") as file:
        print(openmm_simulation.system.getNumParticles(), file=file)
        print(f"Lattice=\"{box[0]} 0.0 0.0 {box[3] * box[1]} {box[1]} 0.0 {box[4] * box[2]} {box[5] * box[2]} {box[2]}"
              f"\" Properties=species:S:1:pos:R:3 Origin=\"{-box[0] / 2.0} {-box[1] / 2.0} {-box[2] / 2.0}\"",
              file=file)
        for atom, position in zip(topology.atoms(), positions):
            assert len(position) == 3
            print(f"{atom.name} {position[0]} {position[1]} {position[2]}", file=file)


# noinspection PyUnresolvedReferences
def write_xyz_file_from_gsd_frame(filename: str, gsd_frame: gsd.hoomd.Frame) -> None:
    with open(filename, "w") as file:
        print(gsd_frame.particles.N, file=file)
        # Use the extended xyz file format.
        # See https://www.ovito.org/docs/current/reference/file_formats/input/xyz.html#extended-xyz-format
        # See https://gsd.readthedocs.io/en/stable/schema-hoomd.html#chunk-configuration-box
        # See https://hoomd-blue.readthedocs.io/en/v2.9.4/box.html
        box = gsd_frame.configuration.box
        assert len(box) == 6
        print(f"Lattice=\"{box[0]} 0.0 0.0 {box[3] * box[1]} {box[1]} 0.0 {box[4] * box[2]} {box[5] * box[2]} {box[2]}"
              f"\" Properties=species:S:1:pos:R:3", file=file)
        for index in range(gsd_frame.particles.N):
            position = gsd_frame.particles.position[index, :]
            t = gsd_frame.particles.types[gsd_frame.particles.typeid[index]]
            print(f"{t} {position[0]} {position[1]} {position[2]}", file=file)


def generate_fibonacci_sphere_grid_points(number_points: int, radius: float,
                                          random_rotation: bool) -> Iterator[npt.NDArray[np.floating]]:
    # See https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    # Output, real xgB(3,ng): the grid points.
    golden_ratio = (1.0 + sqrt(5.0)) / 2.0
    epsilon = 0.36
    random_rotation = Rotation.random() if random_rotation else Rotation.identity()
    for i in range(number_points):
        theta = 2.0 * pi * i / golden_ratio
        phi = acos(1.0 - 2.0 * (i + epsilon) / (number_points - 1.0 + 2.0 * epsilon))
        yield random_rotation.apply([cos(theta) * sin(phi) * radius, sin(theta) * sin(phi) * radius, cos(phi) * radius])


def main() -> None:
    radius_positive = 105.0 * (unit.nano * unit.meter)
    radius_negative = 95.0 * (unit.nano * unit.meter)
    mass_positive = 1.0 * unit.amu
    mass_negative = (radius_negative / radius_positive) ** 3 * mass_positive
    temperature = 298.0 * unit.kelvin
    # noinspection PyUnresolvedReferences
    collision_rate = 0.01 / (unit.pico * unit.second)
    # noinspection PyUnresolvedReferences
    timestep = 0.05 * (unit.pico * unit.second)

    types, positions, cell = read_xyz_file("tests/first_frame.xyz")
    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res1", chain)
    for t, position in zip(types, positions):
        topology.addAtom(t, None, residue)
    topology.setPeriodicBoxVectors(cell)

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(*cell[0]), openmm.Vec3(*cell[1]), openmm.Vec3(*cell[2]))
    platform = openmm.Platform.getPlatformByName("Reference")
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    for t, position in zip(types, positions):
        if t == "P":
            system.addParticle(mass_positive)
        else:
            assert t == "N"
            system.addParticle(mass_negative)
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    write_gsd_file("tests/first_frame.gsd", simulation,
                   {"P": radius_positive, "N": radius_negative},
                   {"P": 44.0 * (unit.milli * unit.volt), "N": -54.0 * (unit.milli * unit.volt)},
                   cell * (unit.nano * unit.meter))


if __name__ == '__main__':
    main()
