from abc import ABC, abstractmethod
import pathlib


class Plotter(ABC):
    def __init__(self, working_directory: str):
        self._working_directory = pathlib.Path(working_directory)
        if not self._working_directory.exists() and self._working_directory.is_dir():
            raise ValueError("The working directory does not exist.")

    @abstractmethod
    def plot(self) -> None:
        raise NotImplementedError
