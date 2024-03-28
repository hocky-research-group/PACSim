from typing import Union
import gsd.hoomd
import hoomd
import numpy as np
import numpy.typing as npt
import openmm
from openmm import app
from openmm import unit


def read_xyz_file(filename: str, units: bool = True) -> (npt.NDArray[str],
                                                         Union[npt.NDArray[float], npt.NDArray[unit.Quantity]]):
    with open(filename, "r") as f:
        line = next(f)
        ls = line.split()
        number_atoms = int(ls[0])
        _ = next(f)
        types = np.zeros(number_atoms, dtype=str)
        positions = np.zeros((number_atoms, 3), dtype=unit.Quantity if units else float)
        for index_atom in range(number_atoms):
            line = next(f)
            ls = line.split()
            types[index_atom] = ls[0]
            positions[index_atom, 0] = float(ls[1]) * (unit.nanometer if units else 1.0)
            positions[index_atom, 1] = float(ls[2]) * (unit.nanometer if units else 1.0)
            positions[index_atom, 2] = float(ls[3]) * (unit.nanometer if units else 1.0)
    return types, positions


def write_gsd_file(filename: str, openmm_simulation: app.Simulation, radius_dict: dict[str, float]) -> None:
    positions = (
        openmm_simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True))
    topology = openmm_simulation.topology
    assert topology.getNumChains() == 1
    assert topology.getNumResidues() == 1
    assert topology.getNumAtoms() == openmm_simulation.system.getNumParticles() == len(positions)
    periodic_box_vectors = openmm_simulation.system.getDefaultPeriodicBoxVectors()
    assert len(periodic_box_vectors) == 3
    side_length = periodic_box_vectors[0][0].value_in_unit(unit.nanometer)
    assert periodic_box_vectors[0][1].value_in_unit(unit.nanometer) == 0.0
    assert periodic_box_vectors[0][2].value_in_unit(unit.nanometer) == 0.0
    assert periodic_box_vectors[1][0].value_in_unit(unit.nanometer) == 0.0
    assert periodic_box_vectors[1][1].value_in_unit(unit.nanometer) == side_length
    assert periodic_box_vectors[1][2].value_in_unit(unit.nanometer) == 0.0
    assert periodic_box_vectors[2][0].value_in_unit(unit.nanometer) == 0.0
    assert periodic_box_vectors[2][1].value_in_unit(unit.nanometer) == 0.0
    assert periodic_box_vectors[2][2].value_in_unit(unit.nanometer) == side_length

    frame = gsd.hoomd.Frame()
    frame.particles.N = topology.getNumAtoms()
    # Shift positions so that they are in the range [-side_length / 2, side_length / 2] (openmm uses [0, side_length]).
    frame.particles.position = positions.value_in_unit(unit.nanometer) - np.array([side_length / 2.0 for _ in range(3)])
    types_set = set(atom.name for atom in topology.atoms())
    assert len(types_set) == 2
    types = list(types_set)
    assert all(t in radius_dict for t in types)
    frame.particles.types = types
    frame.particles.typeid = [types.index(atom.name) for atom in topology.atoms()]
    frame.particles.type_shapes = [{"type": "Sphere", "diameter": 2.0 * radius_dict[t]} for t in types]
    frame.particles.mass = [openmm_simulation.system.getParticleMass(atom_index).value_in_unit(unit.amu)
                            for atom_index in range(topology.getNumAtoms())]
    frame.configuration.box = [side_length, side_length, side_length, 0, 0, 0]
    with gsd.hoomd.open(name=filename, mode="x") as f:
        f.append(frame)


# noinspection PyUnresolvedReferences
def write_xyz_file(filename: str, gsd_snapshot: hoomd.data.SnapshotParticleData) -> None:
    with open(filename, "x") as file:
        print(gsd_snapshot.particles.N, file=file)
        print("Atom positions:", file=file)
        for index in range(gsd_snapshot.particles.N):
            position = gsd_snapshot.particles.position[index, :]
            t = gsd_snapshot.particles.types[index]
            if t == "positive":
                print(f"P {position[0]} {position[1]} {position[2]}", file=file)
            else:
                assert t == "negative"
                print(f"N {position[0]} {position[1]} {position[2]}", file=file)


def main() -> None:
    radius_positive = 105.0 * unit.nanometer
    radius_negative = 95.0 * unit.nanometer
    mass_positive = 1.0 * unit.amu
    mass_negative = (radius_negative / radius_positive) ** 3 * mass_positive
    side_length = 12328.05 * unit.nanometer
    temperature = 298.0 * unit.kelvin
    # noinspection PyUnresolvedReferences
    collision_rate = 0.01 / unit.picosecond
    # noinspection PyUnresolvedReferences
    timestep = 0.05 * unit.picosecond

    types, positions = read_xyz_file("tests/first_frame.xyz")
    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res1", chain)
    app.element.Element(0, "positive", "positive", mass_positive)
    app.element.Element(1, "negative", "negative", mass_negative)
    for t, position in zip(types, positions):
        if t == "P":
            topology.addAtom("positive", app.element.Element.getBySymbol("positive"), residue)
        else:
            assert t == "N"
            topology.addAtom("negative", app.element.Element.getBySymbol("negative"), residue)
    topology.setPeriodicBoxVectors(np.array([[side_length.value_in_unit(unit.nanometer), 0.0, 0.0],
                                             [0.0, side_length.value_in_unit(unit.nanometer), 0.0],
                                             [0.0, 0.0, side_length.value_in_unit(unit.nanometer)]]))

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(side_length.value_in_unit(unit.nanometer), 0.0, 0.0),
                                        openmm.Vec3(0.0, side_length.value_in_unit(unit.nanometer), 0.0),
                                        openmm.Vec3(0.0, 0.0, side_length.value_in_unit(unit.nanometer)))
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

    write_gsd_file("test.gsd", simulation,
                   {"positive": radius_positive.value_in_unit(unit.nanometer),
                    "negative": radius_negative.value_in_unit(unit.nanometer)})


if __name__ == '__main__':
    main()
