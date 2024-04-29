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
    if not filename.endswith(".xyz"):
        raise ValueError("The file must have the .xyz extension.")
    with open(filename, "r") as f:
        line = next(f)
        ls = line.split()
        number_atoms = int(ls[0])
        _ = next(f)
        types = np.zeros(number_atoms, dtype=str)
        positions = np.zeros((number_atoms, 3), dtype=unit.Quantity if units else float)
        # Do not create a unit within the loop because this is slow.
        u = (unit.nano * unit.meter) if units else 1.0
        for index_atom in range(number_atoms):
            line = next(f)
            ls = line.split()
            types[index_atom] = ls[0]
            positions[index_atom, 0] = float(ls[1]) * u
            positions[index_atom, 1] = float(ls[2]) * u
            positions[index_atom, 2] = float(ls[3]) * u
    return types, positions


def write_gsd_file(filename: str, openmm_simulation: app.Simulation, radius_dict: dict[str, unit.Quantity],
                   surface_potentials_dict: dict[str, unit.Quantity]) -> None:
    nanometer = unit.nano * unit.meter
    millivolt = unit.milli * unit.volt

    positions = (
        openmm_simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True))
    topology = openmm_simulation.topology
    assert topology.getNumChains() == 1
    assert topology.getNumResidues() == 1
    assert topology.getNumAtoms() == openmm_simulation.system.getNumParticles() == len(positions)
    periodic_box_vectors = openmm_simulation.system.getDefaultPeriodicBoxVectors()
    assert len(periodic_box_vectors) == 3
    assert periodic_box_vectors[0][1].value_in_unit(nanometer) == 0.0
    assert periodic_box_vectors[0][2].value_in_unit(nanometer) == 0.0
    assert periodic_box_vectors[1][2].value_in_unit(nanometer) == 0.0

    frame = gsd.hoomd.Frame()
    frame.particles.N = topology.getNumAtoms()
    frame.particles.position = positions.value_in_unit(nanometer)
    # Use a dictionary instead of a set to preserve the order of the types.
    # See https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
    # Works since Python 3.7.
    types_set = list(dict.fromkeys(atom.name for atom in topology.atoms()))
    assert len(types_set) == 2
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
        periodic_box_vectors[0][0].value_in_unit(nanometer),
        periodic_box_vectors[1][1].value_in_unit(nanometer),
        periodic_box_vectors[2][2].value_in_unit(nanometer),
        periodic_box_vectors[1][0] / periodic_box_vectors[1][1],
        periodic_box_vectors[2][0] / periodic_box_vectors[2][2],
        periodic_box_vectors[2][1] / periodic_box_vectors[2][2]
    ]
    with gsd.hoomd.open(name=filename, mode="w") as f:
        f.append(frame)


def write_xyz_file(filename: str, openmm_simulation: app.Simulation) -> None:
    positions = (
        openmm_simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True))
    positions = positions.value_in_unit(unit.nano * unit.meter)
    topology = openmm_simulation.topology
    assert topology.getNumChains() == 1
    assert topology.getNumResidues() == 1
    assert topology.getNumAtoms() == openmm_simulation.system.getNumParticles() == len(positions)
    assert len(list(topology.atoms())) == len(positions)
    with open(filename, "w") as file:
        print(openmm_simulation.system.getNumParticles(), file=file)
        print("Atom positions:", file=file)
        for atom, position in zip(topology.atoms(), positions):
            assert len(position) == 3
            print(f"{atom.name} {position[0]} {position[1]} {position[2]}", file=file)


# noinspection PyUnresolvedReferences
def write_xyz_file_from_gsd_snapshot(filename: str, gsd_snapshot: hoomd.data.SnapshotParticleData) -> None:
    with open(filename, "w") as file:
        print(gsd_snapshot.particles.N, file=file)
        print("Atom positions:", file=file)
        for index in range(gsd_snapshot.particles.N):
            position = gsd_snapshot.particles.position[index, :]
            t = gsd_snapshot.particles.types[gsd_snapshot.particles.typeid[index]]
            print(f"{t} {position[0]} {position[1]} {position[2]}", file=file)


def main() -> None:
    radius_positive = 105.0 * (unit.nano * unit.meter)
    radius_negative = 95.0 * (unit.nano * unit.meter)
    mass_positive = 1.0 * unit.amu
    mass_negative = (radius_negative / radius_positive) ** 3 * mass_positive
    side_length = 12328.05 * (unit.nano * unit.meter)
    temperature = 298.0 * unit.kelvin
    # noinspection PyUnresolvedReferences
    collision_rate = 0.01 / (unit.pico * unit.second)
    # noinspection PyUnresolvedReferences
    timestep = 0.05 * (unit.pico * unit.second)

    types, positions = read_xyz_file("tests/first_frame.xyz")
    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res1", chain)
    for t, position in zip(types, positions):
        topology.addAtom(t, None, residue)
    topology.setPeriodicBoxVectors(np.array([[side_length.value_in_unit(unit.nano * unit.meter), 0.0, 0.0],
                                             [0.0, side_length.value_in_unit(unit.nano * unit.meter), 0.0],
                                             [0.0, 0.0, side_length.value_in_unit(unit.nano * unit.meter)]]))

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(side_length.value_in_unit(unit.nano * unit.meter), 0.0, 0.0),
                                        openmm.Vec3(0.0, side_length.value_in_unit(unit.nano * unit.meter), 0.0),
                                        openmm.Vec3(0.0, 0.0, side_length.value_in_unit(unit.nano * unit.meter)))
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
                   {"P": radius_positive, "N": radius_negative})


if __name__ == '__main__':
    main()
