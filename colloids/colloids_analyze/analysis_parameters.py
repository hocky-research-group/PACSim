from dataclasses import dataclass
from typing import Any, Optional
from colloids.abstracts import Parameters


@dataclass(order=True, frozen=True)
class AnalysisParameters(Parameters):
    # TODO: Add docstrings.
    working_directory: str = "./output"
    labels: Optional[list[str]] = None
    plot_state_data: bool = True
    plot_rdf: bool = True
    rdf_parameters: Optional[dict[str, Any]] = None
    plot_coordination_numbers: bool = True
    coordination_numbers_parameters: Optional[dict[str, Any]] = None
