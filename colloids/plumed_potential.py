import os
from typing import Iterator
try:
    # openmmplumed is only optional. Make sure that code still works without it.
    import openmmplumed
except ImportError:
    # Set variable so that we can throw an error later.
    openmmplumed = None
from colloids.abstracts import OpenMMPotentialAbstract


class PlumedPotential(OpenMMPotentialAbstract):
    """
    This class sets up a PLUMED potential using the OpenMM PLUMED plugin.

    This class requires the OpenMM PLUMED plugin to be installed (see https://github.com/openmm/openmm-plumed).
    This plugin requires PLUMED to be installed (see https://www.plumed.org/doc-v2.10/user-doc/html/index.html).

    The filename to the PLUMED control script defining the potential is provided on initialization of this class.

    :param plumed_filename:
        The filename to the PLUMED control script defining the potential.
    :type plumed_filename: str

    :raises ImportError:
        If the OpenMM PLUMED plugin is not installed.
    :raises FileNotFoundError:
        If the provided PLUMED filename does not exist.
    """

    def __init__(self, plumed_filename: str) -> None:
        """Constructor of the PlumedPotential class."""
        super().__init__()
        if openmmplumed is None:
            raise ImportError("The OpenMM PLUMED plugin is required to use PlumedPotential (see "
                              "https://github.com/openmm/openmm-plumed).")
        if not os.path.exists(plumed_filename):
            raise FileNotFoundError(f"The provided PLUMED filename '{plumed_filename}' does not exist.")
        self._plumed_potential = plumed_filename
        with open(plumed_filename, 'r') as file:
            plumed_input = file.read()
        self._plumed_potential = openmmplumed.PlumedForce(plumed_input)

    def add_particle(self) -> None:
        """
        Add a particle to the PLUMED potential.

        This method has to be called for every particle in the system before the method yield_potentials is used.

        For the PLUMED potential, we do not need to do anything.
        """
        super().add_particle()


    def yield_potentials(self) -> "Iterator[openmmplumed.PlumedForce]":
        """
        Generate the PLUMED force for an OpenMM system.

        This method has to be called after the method add_particle was called for every particle in the system.

        :return:
            An iterator yielding a single PLUMED force.
        :rtype: Iterator[openmmplumed.PlumedForce]
        """
        yield self._plumed_potential
