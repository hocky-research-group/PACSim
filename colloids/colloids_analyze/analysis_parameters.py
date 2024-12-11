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
    plot_sdf: bool = True
    sdf_parameters: Optional[dict[str, Any]] = None
    plot_coordination_numbers: bool = True
    coordination_numbers_parameters: Optional[dict[str, Any]] = None
    plot_snowman_orientation_distribution: bool = True
    snowman_orientation_distribution_parameters: Optional[dict[str, Any]] = None
    run_cluster_analysis: bool = True
    cluster_analysis_parameters: Optional[dict[str, Any]] = None
    run_cubic_cluster_rotator: bool = False
    cubic_cluster_rotator_parameters: Optional[dict[str, Any]] = None
    animate_snowman_orientation_distribution: bool = False
    snowman_orientation_distribution_animator_parameters: Optional[dict[str, Any]] = None
    plot_snowman_orientation_rmsd: bool = False
    snowman_orientation_rmsd_parameters: Optional[dict[str, Any]] = None
    plot_snowman_orientation_correlation: bool = False
    snowman_orientation_correlation_parameters: Optional[dict[str, Any]] = None
