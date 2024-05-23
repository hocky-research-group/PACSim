from dataclasses import dataclass
from colloids.abstracts import Parameters


@dataclass(order=True, frozen=True)
class ConfigurationParameters(Parameters):
    # TODO: Add docstrings.
    lattice_type: str = "sc"
    lattice_spacing_factor: float = 8.0
    lattice_repeats: int = 8
    orbit_factor: float = 1.3
    satellites_per_center: int = 1
    padding_factor: float = 0.0

    def __post_init__(self):
        if self.lattice_spacing_factor <= 0.0:
            raise ValueError("The lattice spacing factor must be positive.")
        if self.lattice_repeats <= 0:
            raise ValueError("The number of lattice repeats must be positive.")
        if self.orbit_factor <= 0.0:
            raise ValueError("The orbit factor must be positive.")
        if self.satellites_per_center < 0:
            raise ValueError("The number of satellites per center must be zero or positive.")
        if self.lattice_type not in ["sc", "bcc", "fcc"]:
            raise ValueError("The lattice type must be sc, bcc, or fcc.")
        if self.padding_factor < 0.0:
            raise ValueError("The padding factor must be non-negative.")
