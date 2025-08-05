"""Mini‑Bayes PPI: Bayesian multi‑task PPI inference from scRNA‑seq counts."""
from importlib import metadata as _md

try:
    __version__ = _md.version("mini_bayes_ppi")  # Use actual package name
except _md.PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from .core import MBModel, export_networks  # noqa: F401
from .io import load_string_prior  # noqa: F401

__all__ = ["MBModel", "export_networks", "load_string_prior"]
