from abc import ABC, abstractmethod


class ConfigurationGenerator(ABC):
    """
    Abstract base class for a generator of an initial configuration in a xyz file of an OpenMM simulation of colloids.

    :param filename:
        The name of the file to write the positions to.
    :type filename: str

    :raises ValueError:
        If the filename does not have the .xyz extension.
    """

    def __init__(self, filename: str):
        if not filename.endswith(".xyz"):
            raise ValueError("The filename must have the .xyz extension.")
        self._filename = filename

    @abstractmethod
    def write_positions(self) -> None:
        """
        Generate the initial positions of the colloids and write them to an xyz file using the extended xyz format.
        """
        raise NotImplementedError
