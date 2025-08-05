"""
Enhanced I/O for loading known protein interactions with confidence scores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


def load_string_interactions(
    source: str | pd.DataFrame | list,
    *,
    adata_var_names: Iterable[str],
    score_cutoff: int = 400,
    score_type: str = "combined_score",
    include_confidence: bool = True,
    species: int = 9606,  # Human by default
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float] | None]:
    """
    Load protein interactions from STRING database or other sources.
    
    Parameters
    ----------
    source : str, pd.DataFrame, or list
        - str: Path to STRING TSV file or 'download' to fetch from STRING
        - pd.DataFrame: Pre-loaded interaction data
        - list: List of interactions as ["GENE1 GENE2", ...] or [(gene1, gene2), ...]
    adata_var_names : Iterable[str]
        Gene names in the AnnData object
    score_cutoff : int
        Minimum confidence score (0-1000 for STRING)
    score_type : str
        Which STRING score to use ('combined_score', 'experimental', etc.)
    include_confidence : bool
        Whether to return confidence scores
    species : int
        NCBI taxonomy ID for species (only used if downloading)
        
    Returns
    -------
    edges : list[tuple[int, int]]
        Gene index pairs for interactions
    confidences : dict[tuple[int, int], float] or None
        Confidence scores for each edge (normalized to 0-1)
    """
    name_to_idx = {g: i for i, g in enumerate(adata_var_names)}
    
    if isinstance(source, list):
        # Handle explicit list of interactions
        edges = []
        confidences = {}
        
        for item in source:
            if isinstance(item, str):
                parts = item.strip().split()
                if len(parts) == 2:
                    g1, g2 = parts
                    conf = 0.5  # Default confidence
                elif len(parts) == 3:
                    g1, g2, conf = parts
                    conf = float(conf)
                else:
                    raise ValueError(f"Invalid format: {item}")
            elif isinstance(item, (tuple, list)):
                if len(item) == 2:
                    g1, g2 = item
                    conf = 0.5
                elif len(item) == 3:
                    g1, g2, conf = item
                    conf = float(conf)
                else:
                    raise ValueError(f"Invalid format: {item}")
            else:
                raise ValueError(f"Unknown format: {item}")
            
            if g1 in name_to_idx and g2 in name_to_idx:
                i, j = name_to_idx[g1], name_to_idx[g2]
                edge = (min(i, j), max(i, j))
                edges.append(edge)
                if include_confidence:
                    confidences[edge] = conf
        
        logger.info(f"Loaded {len(edges)} interactions from list")
        return edges, confidences if include_confidence else None
    
    # Load DataFrame
    if isinstance(source, str):
        if source.lower() == "download":
            df = download_string_data(species, version="11.5")
        else:
            path = Path(source).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            # Detect file format
            if path.suffix.lower() in ['.tsv', '.txt']:
                df = pd.read_csv(path, sep='\t')
            elif path.suffix.lower() == '.csv':
                df = pd.read_csv(path)
            else:
                # Try to infer
                df = pd.read_csv(path, sep=None, engine='python')
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise ValueError(f"Unknown source type: {type(source)}")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Check for required columns
    if 'protein1' not in df.columns or 'protein2' not in df.columns:
        # Try alternative names
        if 'gene1' in df.columns and 'gene2' in df.columns:
            df = df.rename(columns={'gene1': 'protein1', 'gene2': 'protein2'})
        elif 'source' in df.columns and 'target' in df.columns:
            df = df.rename(columns={'source': 'protein1', 'target': 'protein2'})
        else:
            raise ValueError("Could not find protein/gene columns in data")
    
    # Filter by score if available
    if score_type in df.columns:
        df = df[df[score_type] >= score_cutoff]
        logger.info(f"Filtered to {len(df)} interactions with {score_type} >= {score_cutoff}")
    elif 'score' in df.columns:
        df = df[df['score'] >= score_cutoff / 1000]  # Assume 0-1 scale
        logger.info(f"Filtered to {len(df)} interactions with score >= {score_cutoff/1000}")
    
    # Extract edges and confidence scores
    edges = []
    confidences = {}
    
    for _, row in df.iterrows():
        g1, g2 = row['protein1'], row['protein2']
        
        if g1 in name_to_idx and g2 in name_to_idx:
            i, j = name_to_idx[g1], name_to_idx[g2]
            edge = (min(i, j), max(i, j))
            edges.append(edge)
            
            if include_confidence:
                # Normalize confidence to 0-1
                if score_type in df.columns:
                    conf = row[score_type] / 1000  # STRING uses 0-1000
                elif 'score' in df.columns:
                    conf = row['score']
                else:
                    conf = 0.5  # Default
                confidences[edge] = np.clip(conf, 0, 1)
    
    # Remove duplicates
    edges = list(set(edges))
    
    logger.info(
        f"Loaded {len(edges)} unique interactions "
        f"({len(df)} total, {len(df) - len(edges)} filtered/duplicates)"
    )
    
    return edges, confidences if include_confidence else None


def download_string_data(
    species: int = 9606,
    version: str = "11.5",
    score_type: str = "combined_score",
    min_score: int = 400,
) -> pd.DataFrame:
    """
    Download protein interaction data from STRING database.
    
    Parameters
    ----------
    species : int
        NCBI taxonomy ID (9606 for human, 10090 for mouse)
    version : str
        STRING version
    score_type : str
        Score type to include
    min_score : int
        Minimum score threshold
        
    Returns
    -------
    pd.DataFrame
        Interaction data
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests package needed for downloading. Install with: pip install requests")
    
    # STRING API endpoint
    base_url = f"https://stringdb-static.org/download/protein.links.v{version}/"
    file_name = f"{species}.protein.links.v{version}.txt.gz"
    url = base_url + file_name
    
    logger.info(f"Downloading STRING data for species {species} from {url}")
    
    # Download and parse
    try:
        df = pd.read_csv(url, sep=' ', compression='gzip')
        
        # Extract gene names from protein IDs
        df['protein1'] = df['protein1'].str.split('.').str[1]
        df['protein2'] = df['protein2'].str.split('.').str[1]
        
        # Filter by score
        df = df[df[score_type] >= min_score]
        
        logger.info(f"Downloaded {len(df)} interactions from STRING")
        return df
        
    except Exception as e:
        logger.error(f"Failed to download STRING data: {e}")
        logger.info("Please download manually from https://string-db.org/")
        raise


def load_pathway_interactions(
    pathway_file: str,
    adata_var_names: Iterable[str],
    pathway_name: str | None = None,
) -> tuple[list[tuple[int, int]], dict[str, list[str]]]:
    """
    Load interactions from pathway databases (KEGG, Reactome, etc.).
    
    Parameters
    ----------
    pathway_file : str
        Path to pathway file (GMT, TXT, or CSV format)
    adata_var_names : Iterable[str]
        Gene names in the AnnData object
    pathway_name : str or None
        Specific pathway to load (if None, loads all)
        
    Returns
    -------
    edges : list[tuple[int, int]]
        All pairwise interactions within pathways
    pathways : dict[str, list[str]]
        Pathway membership for validation
    """
    path = Path(pathway_file).expanduser()
    name_to_idx = {g: i for i, g in enumerate(adata_var_names)}
    
    pathways = {}
    
    # Read pathway file
    if path.suffix.lower() == '.gmt':
        # GMT format (MSigDB, etc.)
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    pw_name = parts[0]
                    pw_url = parts[1]  # Usually URL or description
                    pw_genes = parts[2:]
                    
                    if pathway_name is None or pathway_name in pw_name:
                        pathways[pw_name] = pw_genes
    else:
        # Assume simple format: pathway_name gene1 gene2 ...
        df = pd.read_csv(path, sep='\t' if path.suffix == '.tsv' else ',')
        for _, row in df.iterrows():
            pw_name = row.iloc[0]
            pw_genes = row.iloc[1:].dropna().tolist()
            
            if pathway_name is None or pathway_name in pw_name:
                pathways[pw_name] = pw_genes
    
    # Generate all pairwise interactions within pathways
    edges = set()
    for pw_name, genes in pathways.items():
        # Filter to genes in data
        valid_genes = [g for g in genes if g in name_to_idx]
        
        # Create all pairs
        for i, g1 in enumerate(valid_genes):
            for g2 in valid_genes[i+1:]:
                idx1, idx2 = name_to_idx[g1], name_to_idx[g2]
                edges.add((min(idx1, idx2), max(idx1, idx2)))
    
    logger.info(f"Loaded {len(edges)} interactions from {len(pathways)} pathways")
    
    return list(edges), pathways


def export_to_cytoscape(
    activity_df: pd.DataFrame,
    output_file: str,
    node_attributes: dict | None = None,
) -> None:
    """
    Export network for Cytoscape visualization.
    
    Parameters
    ----------
    activity_df : pd.DataFrame
        Activity scores from the model
    output_file : str
        Output file path (.xgmml or .csv)
    node_attributes : dict or None
        Additional node attributes to include
    """
    path = Path(output_file).expanduser()
    
    if path.suffix.lower() == '.csv':
        # Simple edge list format
        export_df = activity_df[['protein1', 'protein2', 'activity_score']].copy()
        export_df.columns = ['source', 'target', 'weight']
        export_df.to_csv(path, index=False)
        logger.info(f"Exported {len(export_df)} edges to {path}")
        
    elif path.suffix.lower() in ['.xgmml', '.xml']:
        # XGMML format for Cytoscape
        import xml.etree.ElementTree as ET
        
        # Create graph element
        graph = ET.Element('graph', {
            'label': 'PPI_Activity_Network',
            'xmlns': 'http://www.cs.rpi.edu/XGMML'
        })
        
        # Add nodes
        nodes_added = set()
        for _, row in activity_df.iterrows():
            for protein in [row['protein1'], row['protein2']]:
                if protein not in nodes_added:
                    node = ET.SubElement(graph, 'node', {
                        'id': protein,
                        'label': protein
                    })
                    
                    # Add attributes if provided
                    if node_attributes and protein in node_attributes:
                        for attr_name, attr_value in node_attributes[protein].items():
                            att = ET.SubElement(node, 'att', {
                                'name': attr_name,
                                'value': str(attr_value),
                                'type': 'string'
                            })
                    
                    nodes_added.add(protein)
        
        # Add edges
        for idx, row in activity_df.iterrows():
            edge = ET.SubElement(graph, 'edge', {
                'id': f"edge_{idx}",
                'source': row['protein1'],
                'target': row['protein2'],
                'label': f"{row['protein1']}-{row['protein2']}"
            })
            
            # Add edge attributes
            for col in ['activity_score', 'prior_confidence']:
                if col in row:
                    att = ET.SubElement(edge, 'att', {
                        'name': col,
                        'value': str(row[col]),
                        'type': 'real'
                    })
        
        # Write to file
        tree = ET.ElementTree(graph)
        tree.write(path, encoding='utf-8', xml_declaration=True)
        logger.info(f"Exported network to {path} (XGMML format)")
    
    else:
        raise ValueError(f"Unknown output format: {path.suffix}")


def validate_gene_names(
    gene_list: list[str],
    reference_genes: Iterable[str],
    similarity_threshold: float = 0.8,
) -> dict[str, str]:
    """
    Validate and map gene names, finding close matches.
    
    Parameters
    ----------
    gene_list : list[str]
        Genes to validate
    reference_genes : Iterable[str]
        Valid gene names
    similarity_threshold : float
        Minimum similarity for fuzzy matching
        
    Returns
    -------
    dict[str, str]
        Mapping from input genes to valid genes
    """
    from difflib import SequenceMatcher
    
    reference_set = set(reference_genes)
    mapping = {}
    
    for gene in gene_list:
        if gene in reference_set:
            mapping[gene] = gene
        else:
            # Try case-insensitive match
            gene_upper = gene.upper()
            for ref in reference_set:
                if ref.upper() == gene_upper:
                    mapping[gene] = ref
                    logger.debug(f"Mapped {gene} -> {ref} (case)")
                    break
            else:
                # Try fuzzy matching
                best_match = None
                best_score = 0
                
                for ref in reference_set:
                    score = SequenceMatcher(None, gene.upper(), ref.upper()).ratio()
                    if score > best_score and score >= similarity_threshold:
                        best_score = score
                        best_match = ref
                
                if best_match:
                    mapping[gene] = best_match
                    logger.warning(f"Fuzzy matched {gene} -> {best_match} (score: {best_score:.2f})")
                else:
                    logger.warning(f"No match found for gene: {gene}")
    
    return mapping
