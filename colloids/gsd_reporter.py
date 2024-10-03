from typing import Optional
import gsd.hoomd
import numpy as np
import numpy.typing as npt
import openmm.app
from openmm import unit


class GSDReporter(object):
    """
    Reporter for an OpenMM simulation of colloids that writes the trajectory to a GSD file starting at the initial
    configuration.

    For general information on GSD files, see https://gsd.readthedocs.io/en/stable/python-module-gsd.hoomd.html.

    The gsd file will store the current time step, positions, velocities, and box vectors of the simulation at every
    reported time step of the simulation.

    The cell vectors of the OpenMM simulation were possibly artificially enlarged in the presence of walls to prevent
    particles interacting through walls when periodic boundary conditions are used. Thus, one can optionally specify
    a cell on initialization that will be stored in the gsd file at every reported time step of the simulation instead
    of the cell vectors within OpenMM.

    The log of the gsd file will store the time, potential energy, and kinetic energy of the simulation at every
    reported time step of the simulation. The log can be accessed with the gsd.hoomd.read_log function.

    Only in the first frame at time zero, this reporter will store the number, types, radii, surface potentials, and
    masses of the colloidal particles in the particle data in the gsd file. These quantities should not change during a
    simulation.

    :param filename:
        The name of the file to write to.
        The filename must end with the .gsd extension.
    :type filename: str
    :param report_interval:
        The interval (in time steps) at which to write frames of the trajectory in the OpenMM simulation.
        The value must be greater than zero.
    :type report_interval: int
    :param radii:
        The radii of the different types of colloidal particles that appear in the OpenMM simulation.
        The keys of the dictionary are the types of the colloidal particles and the values are the radii.
        The unit of the radii must be compatible with nanometers and the values must be greater than zero.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials of the different types of colloidal particles that appear in the OpenMM simulation.
        The keys of the dictionary are the types of the colloidal particles and the values are the surface potentials.
        The unit of the surface potentials must be compatible with millivolts.
    :type surface_potentials: dict[str, unit.Quantity]
    :param simulation:
        The OpenMM simulation that this reporter will be added to.
        The topology of the simulation is used to check that a radius and surface potential is defined for every type of
        atom in the simulation.
        The state of the OpenMM simulation is used to store the initial configuration of the simulation in the gsd file.
    :type simulation: openmm.app.Simulation
    :param append_file:
        If True, open an existing GSD file to append to. If False, try to create a new file, and throw an error if the
        file already exists.
        Defaults to False.
    :type append_file: bool
    :param cell:
        The cell vectors that should be stored in the gsd file at every reported time step.
        If None, the cell vectors of the OpenMM simulation are used.
        Defaults to None.
    :type cell: Optional[npt.NDArray[unit.Quantity]]

    :raises ValueError:
        If the filename does not end with the .gsd extension.
    :raises ValueError:
        If the report interval is not greater than zero.
    :raises ValueError:
        If not all types of the topology of the OpenMM simulation have a radius.
    :raises ValueError:
        If not all types with a radius appear in the topology of the OpenMM simulation.
    :raises TypeError:
        If a radius does not have a unit compatible with nanometers.
    :raises ValueError:
        If a radius is not greater than zero.
    :raises ValueError:
        If not all types of the topology of the OpenMM simulation have a surface potential.
    :raises ValueError:
        If not all types with a surface potential appear in the topology of the OpenMM simulation.
    :raises TypeError:
        If a surface potential does not have a unit compatible with millivolts.
    """

    _nanometer = unit.nano * unit.meter
    _nanometer_per_picosecond = (unit.nano * unit.meter) / (unit.pico * unit.second)
    _millivolt = unit.milli * unit.volt

    def __init__(self, filename: str, report_interval: int, radii: dict[str, unit.Quantity],
                 surface_potentials: dict[str, unit.Quantity], simulation: openmm.app.Simulation,
                 append_file: bool = False, cell: Optional[npt.NDArray[unit.Quantity]] = None) -> None:
        """Constructor of the GSDReporter class."""
        if not filename.endswith(".gsd"):
            raise ValueError("The file must have the .gsd extension.")
        if not report_interval > 0:
            raise ValueError("The report interval must be greater than zero.")
        assert simulation.topology.getNumChains() == 1
        assert simulation.topology.getNumResidues() == 1
        assert simulation.topology.getNumAtoms() == simulation.system.getNumParticles()
        types = list(dict.fromkeys(atom.name for atom in simulation.topology.atoms()))
        if not all(t in radii for t in types if t!='wall'):
            raise ValueError("All types of the simulation must have a radius.")
        if not all(rt in types for rt in radii if rt!='wall'):
            raise ValueError("All types with a radius must appear in the simulation.")
        if not all(t in surface_potentials for t in types if t!='wall'):
            raise ValueError("All types of the simulation must have a surface potential.")
        if not all(st in types for st in surface_potentials if st!='wall'):
            raise ValueError("All types with a surface potential must appear in the simulation.")
        for r in radii.values():
            if not r.unit.is_compatible(self._nanometer):
                raise TypeError("Radius must have a unit compatible with nanometers.")
            if r <= 0.0 * self._nanometer:
                raise ValueError("Radius must be greater than zero.")
        for s in surface_potentials.values():
            if not s.unit.is_compatible(self._millivolt):
                raise TypeError("Surface potential must have a unit compatible with millivolts.")

        self._report_interval = report_interval
        self._radii = radii
        self._surface_potentials = surface_potentials
        self._append_file = append_file
        self._file = gsd.hoomd.open(name=filename, mode="r+" if self._append_file else "w")
        self._frame = self._set_up_frame(simulation)
        self._cell = cell
        if not self._append_file:
            # Include initial configuration in frame.
            self.report(simulation, simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True,
                                                                enforcePeriodicBox=False))

    def _set_up_frame(self, simulation: openmm.app.Simulation) -> gsd.hoomd.Frame:
        if not self._append_file:
            # Assume that the following properties are constant throughout the simulation.
            frame = gsd.hoomd.Frame()
            frame.particles.N = simulation.topology.getNumAtoms()
            # Use a dictionary instead of a set to preserve the order of the types.
            # See https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
            # Works since Python 3.7.
            types = list(dict.fromkeys(atom.name for atom in simulation.topology.atoms()))
            frame.particles.types = types
            frame.particles.type_shapes = [
                {"type": "Sphere", "diameter": 2.0 * self._radii[t].value_in_unit(self._nanometer)}
                for t in types]
            assert (list(atom.index for atom in simulation.topology.atoms())
                    == list(range(simulation.topology.getNumAtoms())))
            typeid = [types.index(atom.name) for atom in simulation.topology.atoms()]
            frame.particles.typeid = typeid
            frame.particles.diameter = [
                2.0 * self._radii[types[typeid[atom_index]]].value_in_unit(self._nanometer)
                for atom_index in range(simulation.topology.getNumAtoms())]
            frame.particles.charge = [
                self._surface_potentials[types[typeid[atom_index]]].value_in_unit(self._millivolt)
                for atom_index in range(simulation.topology.getNumAtoms())]
            frame.particles.mass = [simulation.system.getParticleMass(atom_index).value_in_unit(unit.amu)
                                    for atom_index in range(simulation.topology.getNumAtoms())]
            frame.configuration.dimensions = 3
        else:
            # Copy constant properties from initial frame.
            assert len(self._file) > 0
            frame = self._file[0]
        return frame

    # noinspection PyPep8Naming
    def describeNextReport(self, simulation: openmm.app.Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        """Get information about the next report this reporter will generate.

        This method is called by OpenMM once this reporter is added to the list of reporters of a simulation.

        :param simulation:
            The simulation to generate a report for.
        :type simulation: openmm.app.Simulation

        :returns:
            (Number of steps until next report,
            Whether the next report requires positions (True),
            Whether the next report requires velocities (True),
            Whether the next report requires forces (False),
            Whether the next report requires energies (True),
            Whether positions should be wrapped to lie in a single periodic box (False))
        :rtype: tuple[int, bool, bool, bool, bool, bool]
        """
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return steps, True, True, False, True, False

    # noinspection PyUnresolvedReferences
    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        """
        Generate a report by storing information about the trajectory in the GSD file.

        :param simulation:
            The OpenMM simulation to generate a report for.
        :type simulation: openmm.app.Simulation
        :param state:
            The current state of the OpenMM simulation.
        :type state: openmm.State
        """
        assert state.getStepCount() == simulation.currentStep
        self._frame.configuration.step = state.getStepCount()
        positions = state.getPositions(asNumpy=True)
        assert len(positions) == self._frame.particles.N
        self._frame.particles.position = positions.value_in_unit(self._nanometer)
        velocities = state.getVelocities(asNumpy=True)
        assert len(velocities) == self._frame.particles.N
        self._frame.particles.velocity = velocities.value_in_unit(self._nanometer_per_picosecond)
        periodic_box_vectors = self._cell if self._cell is not None else state.getPeriodicBoxVectors()
        assert len(periodic_box_vectors) == 3
        assert periodic_box_vectors[0][1].value_in_unit(self._nanometer) == 0.0
        assert periodic_box_vectors[0][2].value_in_unit(self._nanometer) == 0.0
        assert periodic_box_vectors[1][2].value_in_unit(self._nanometer) == 0.0
        # See http://docs.openmm.org/7.6.0/userguide/theory/05_other_features.html
        # See https://hoomd-blue.readthedocs.io/en/v2.9.3/box.html
        self._frame.configuration.box = [
            periodic_box_vectors[0][0].value_in_unit(self._nanometer),
            periodic_box_vectors[1][1].value_in_unit(self._nanometer),
            periodic_box_vectors[2][2].value_in_unit(self._nanometer),
            periodic_box_vectors[1][0] / periodic_box_vectors[1][1],
            periodic_box_vectors[2][0] / periodic_box_vectors[2][2],
            periodic_box_vectors[2][1] / periodic_box_vectors[2][2]
        ]
        # To prevent warnings about implicit data copies, the scalar values should be stored explicitly in a 1D array.
        self._frame.log = {
            "time": np.array([state.getTime().value_in_unit(unit.picosecond)]),
            "potential_energy": np.array([state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)]),
            "kinetic_energy": np.array([state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)])
        }
        self._file.append(self._frame)
        self._file.flush()

    def __del__(self) -> None:
        """Destructor of the GSDReporter class."""
        try:
            self._file.close()
        except AttributeError:
            # If another error occured, the '_file' attribute might not exist.
            pass
