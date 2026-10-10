"""Ocean data analysis tools for the research agent."""

from .dataset_tools import DATASETS, extract_depth, inspect_dataset
from .statistics_tools import calculate_statistics
from .comparison_tools import compare_datasets
from .plot_tools import plot_bias_map, plot_map

__all__ = [
    "DATASETS",
    "inspect_dataset",
    "extract_depth",
    "calculate_statistics",
    "compare_datasets",
    "plot_map",
    "plot_bias_map",
]
