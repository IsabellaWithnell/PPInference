"""Mini‑Bayes PPI: Bayesian multi‑task PPI inference from scRNA‑seq counts."""
from importlib import metadata as _md

try:
    __version__ = _md.version("mini_bayes_ppi")  # Use actual package name
except _md.PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from .core import MBModel, export_networks  # noqa: F401
from .io import load_string_prior  # noqa: F401

# Import visualization functions if matplotlib is available
try:
    from .viz import (  # noqa: F401
        plot_gene_coverage,
        plot_network,
        plot_training_history,
        plot_edge_confidence,
        plot_cell_type_comparison,
    )
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False

__all__ = ["MBModel", "export_networks", "load_string_prior"]

# Add viz functions to __all__ if available
if _VIZ_AVAILABLE:
    __all__.extend([
        "plot_gene_coverage",
        "plot_network", 
        "plot_training_history",
        "plot_edge_confidence",
        "plot_cell_type_comparison",
    ])
