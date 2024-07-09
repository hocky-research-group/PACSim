"""
Code taken from https://openmmtools.readthedocs.io/en/stable/_modules/openmmtools/testsystems.html#LennardJonesFluid
"""
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
        b[0] += 1                       # add one to current integer
        while b[i] > p - 1 + eps:           # this loop does carrying in base p
            b[i] = 0
            i = i + 1
            b[i] += 1
        u[j] = 0
        for k in range(len(b)):         # add up reversed digits
            u[j] += b[k] * p**-(k + 1)
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
