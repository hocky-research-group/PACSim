from abc import ABC, abstractmethod
from dataclasses import dataclass
import pathlib
from typing import Sequence
import matplotlib.pyplot as plt
from colloids.run_parameters import RunParameters

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{siunitx}"
})


@dataclass(order=True, frozen=True)
class LabeledRunParametersWithPath(object):
    path: pathlib.Path
    label: str
    run_parameters: RunParameters


class Plotter(ABC):
    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath]):
        self._working_directory = pathlib.Path(working_directory)
        if not self._working_directory.exists() and self._working_directory.is_dir():
            raise ValueError("The working directory does not exist.")
        self._run_parameters = run_parameters

    @abstractmethod
    def plot(self) -> None:
        raise NotImplementedError
