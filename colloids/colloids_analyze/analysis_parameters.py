from dataclasses import dataclass
from typing import Optional
from colloids.abstracts import Parameters


@dataclass(order=True, frozen=True)
class AnalysisParameters(Parameters):
    # TODO: Add docstrings.
    working_directory: str = "./output"
    plot_state_data: bool = True
    state_data_labels: Optional[list[str]] = None
