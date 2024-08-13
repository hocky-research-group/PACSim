from dataclasses import dataclass
from typing import Union
from colloids.abstracts import Parameters


@dataclass(order=True, frozen=True)
class ConfigurationParameters(Parameters):
    # TODO: Add docstrings.
    lattice_type: str = "sc"
    lattice_spacing_factor: float = 8.0
    lattice_repeats: Union[int, list[int]] = 8
    orbit_factor: float = 1.3
    satellites_per_center: int = 1
    padding_factor: float = 0.0

    def __post_init__(self):
        if self.lattice_spacing_factor <= 0.0:
            raise ValueError("The lattice spacing factor must be positive.")
        if isinstance(self.lattice_repeats, int):
            if self.lattice_repeats <= 0:
                raise ValueError("The number of lattice repeats must be positive.")
        else:
            if not (isinstance(self.lattice_repeats, list)
                    and all(isinstance(repeat, int) for repeat in self.lattice_repeats)
                    and len(self.lattice_repeats) == 3):
                raise TypeError("The lattice repeats must be an integer or a list of three integers.")
            if not all(repeat > 0 for repeat in self.lattice_repeats):
                raise ValueError("All lattice repeats must be positive.")
        if self.orbit_factor <= 0.0:
            raise ValueError("The orbit factor must be positive.")
        if self.satellites_per_center < 0:
            raise ValueError("The number of satellites per center must be zero or positive.")
        if self.lattice_type not in ["sc", "bcc", "fcc"]:
            raise ValueError("The lattice type must be sc, bcc, or fcc.")
        if self.padding_factor < 0.0:
            raise ValueError("The padding factor must be non-negative.")
