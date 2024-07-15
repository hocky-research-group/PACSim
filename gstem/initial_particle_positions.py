"""
Code taken from https://openmmtools.readthedocs.io/en/stable/_modules/openmmtools/testsystems.html#LennardJonesFluid
"""
import itertools
import numpy as np
from openmm import unit


def halton_sequence(p, n):
    """
    Halton deterministic sequence on [0,1].

    Parameters
    ----------
    p : int
       Prime number for sequence.
    n : int
       Sequence length to generate.

    Returns
    -------
    u : numpy.array of double
       Sequence on [0,1].

    Notes
    -----
    Code source: https://blue.math.buffalo.edu/sauer2py/
    More info: https://en.wikipedia.org/wiki/Halton_sequence
    """
    eps = np.finfo(np.double).eps
    # largest number of digits (adding one for halton_sequence(2,64) corner case)
    b = np.zeros(int(np.ceil(np.log(n) / np.log(p))) + 1)
    u = np.empty(n)
    for j in range(n):
        i = 0
        b[0] += 1  # add one to current integer
        while b[i] > p - 1 + eps:  # this loop does carrying in base p
            b[i] = 0
            i = i + 1
            b[i] += 1
        u[j] = 0
        for k in range(len(b)):  # add up reversed digits
            u[j] += b[k] * p ** -(k + 1)
    return u


def subrandom_particle_positions(nparticles, box_vectors):
    """Generate a deterministic list of subrandom particle positions.

    Parameters
    ----------
    nparticles : int
        The number of particles.
    box_vectors : openmm.unit.Quantity of (3,3) with units compatible with nanometer
        Periodic box vectors in which particles should lie.

    Returns
    -------
    positions : openmm.unit.Quantity of (natoms,3) with units compatible with nanometer
        The particle positions.
    """
    # Create positions array.
    positions = unit.Quantity(np.zeros([nparticles, 3], np.float32), unit.nano * unit.meter)

    primes = [2, 3, 5]  # prime bases for Halton sequence
    for dim in range(3):
        x = halton_sequence(primes[dim], nparticles)
        length = box_vectors[dim][dim]
        positions[:, dim] = unit.Quantity(x * length / length.unit, length.unit)

    return positions


def build_lattice_cell():
    """Build a single (4 atom) unit cell of a FCC lattice, assuming a cell length
    of 1.0.

    Returns
    -------
    xyz : np.ndarray, shape=(4, 3), dtype=float
        Coordinates of each particle in cell
    """
    xyz = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]]
    xyz = np.array(xyz)

    return xyz


def build_lattice(n_particles):
    n = ((n_particles / 4.) ** (1 / 3.))

    if np.abs(n - np.round(n)) > 1E-10:
        raise ValueError("Must input 4 m^3 particles for some integer m!")
    else:
        n = int(np.round(n))

    xyz = []
    cell = build_lattice_cell()
    x, y, z = np.eye(3)
    for atom, (i, j, k) in enumerate(itertools.product(np.arange(n), repeat=3)):
        xi = cell + i * x + j * y + k * z
        xyz.append(xi)

    xyz = np.concatenate(xyz)

    return xyz, n


def lattice_particle_positions(nparticles, box_vectors):
    assert len(box_vectors) == 3
    assert len(box_vectors[0]) == 3
    assert len(box_vectors[1]) == 3
    assert len(box_vectors[2]) == 3
    assert box_vectors[0][0] == box_vectors[1][1] == box_vectors[2][2]
    assert box_vectors[0][1] == box_vectors[0][2] == 0.0 * unit.nanometer
    assert box_vectors[1][0] == box_vectors[1][2] == 0.0 * unit.nanometer
    assert box_vectors[2][0] == box_vectors[2][1] == 0.0 * unit.nanometer
    box_nm = box_vectors[0][0] / unit.nanometer
    xyz, box = build_lattice(nparticles)
    xyz *= (box_nm / box)
    return xyz
