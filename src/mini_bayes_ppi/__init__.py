"""
Context-Specific Protein Interaction Activity Inference from scRNA-seq

This package infers the activity of known protein-protein interactions
in different cell types and conditions based on gene co-expression patterns
from single-cell RNA-seq data.

"""

from importlib import metadata as _md

try:
    __version__ = _md.version("mini_bayes_ppi")
except _md.PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# Import main classes and functions
from .core import PPIActivityModel, export_activity_scores
from .io import (
    load_string_interactions,
    load_pathway_interactions,
    export_to_cytoscape,
    validate_gene_names,
)

__all__ = [
    "PPIActivityModel",
    "export_activity_scores",
    "load_string_interactions", 
    "load_pathway_interactions",
    "export_to_cytoscape",
    "validate_gene_names",
]
