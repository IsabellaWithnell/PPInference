# mini_bayes_ppi

Bayesian **P**rotein–**P**rotein **I**nteraction inference from single‑cell RNA‑seq counts.

[![PyPI](https://img.shields.io/pypi/v/mini_bayes_ppi.svg)](https://pypi.org/project/mini_bayes_ppi/)
[![CI](https://github.com/IsabellaWithnell/mini_bayes_ppi/actions/workflows/ci.yml/badge.svg)](https://github.com/IsabellaWithnell/mini_bayes_ppi/actions)
[![Documentation Status](https://readthedocs.org/projects/mini-bayes-ppi/badge/?version=latest)](https://mini-bayes-ppi.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

`mini_bayes_ppi` is a Python package for inferring protein-protein interaction networks from single-cell RNA-seq data using Bayesian multi-task learning. The package models gene co-expression patterns across different cell types to identify potential protein interactions.

### Key Features

- 🧬 **Multi-task Learning**: Jointly models interactions across multiple cell types
- 🎯 **Bayesian Inference**: Provides uncertainty quantification for predicted interactions  
- 🔧 **Flexible Priors**: Supports spike-and-slab or horseshoe priors for edge selection
- 📊 **Cell-type Specific**: Can export both global and cell-type specific networks
- ⚡ **GPU Acceleration**: Supports CUDA for faster training on large datasets

### ⚠️ Important Note

This tool infers potential protein interactions based on gene co-expression patterns. While co-expression can indicate functional relationships, it does not directly measure physical protein-protein interactions. Results should be validated experimentally or cross-referenced with protein interaction databases.

## Installation

```bash
# Basic installation
pip install mini_bayes_ppi

# With GPU support (ensure CUDA toolkit matches PyTorch version)
pip install mini_bayes_ppi --extra-index-url https://download.pytorch.org/whl/cu118

# Development installation
git clone https://github.com/IsabellaWithnell/mini_bayes_ppi
cd mini_bayes_ppi
pip install -e ".[test]"
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1
- AnnData ≥ 0.10
- Pyro-PPL ≥ 1.9

## Quick Start

```python
import anndata as ad
from mini_bayes_ppi import MBModel, load_string_prior

# Load your single-cell data
adata = ad.read_h5ad("pbmc3k.h5ad")

# Define prior edges (known or hypothesized interactions)
prior = load_string_prior([
    "CD3D CD3E",      # T cell receptor components
    "LYZ CTSB",       # Lysosomal proteins
    "GNLY PRF1",      # Cytotoxic granule proteins
    "HLA-A HLA-B",    # MHC class I molecules
], adata_var_names=adata.var_names)

# Initialize and train model
mb = MBModel(
    adata, 
    prior_edges=prior,
    cell_key="cell_type",  # Column in adata.obs with cell types
    device="cuda",         # Use GPU if available
)

# Fit the model
history = mb.fit(epochs=2000, lr=1e-2)

# Export inferred networks
ppi = mb.export_networks(
    threshold=0.9,              # High confidence edges only
    include_confidence=True,    # Include uncertainty estimates
)

print(ppi.head())
#     gene_i  gene_j  probability  weight  prob_lower  prob_upper
# 0     CD3D    CD3E        0.95    0.82        0.91        0.98
# 1      LYZ    CTSB        0.88    0.65        0.83        0.92
# ...
```

## Detailed Usage

### Loading Prior Information

Prior edges can be specified in multiple ways:

```python
# 1. List of gene pairs
prior = load_string_prior(
    ["GENE1 GENE2", "GENE3 GENE4"],
    adata_var_names=adata.var_names
)

# 2. List of tuples
prior = load_string_prior(
    [("GENE1", "GENE2"), ("GENE3", "GENE4")],
    adata_var_names=adata.var_names
)

# 3. From STRING database TSV file
prior = load_string_prior(
    "string_interactions.tsv",
    adata_var_names=adata.var_names,
    score_cutoff=700  # STRING confidence score threshold
)
```

### Model Configuration

```python
# Advanced model configuration
model = MBModel(
    adata,
    prior_edges=prior,
    cell_key="cell_type",
    batch_size=512,          # Smaller batches for limited memory
    lr=5e-3,                 # Learning rate
    device="cuda",           # or "cpu"
    prior_type="spike_slab", # or "horseshoe"
    edge_prior_prob=0.1,     # Prior probability of interaction
)

# Training with early stopping
history = model.fit(
    epochs=5000,
    patience=50,      # Stop if no improvement for 50 epochs
    min_delta=1e-4,   # Minimum improvement threshold
    verbose=True,     # Print progress
)
```

### Exporting Results

```python
# Global network (aggregated across cell types)
global_ppi = model.export_networks(
    threshold=0.9,
    include_confidence=True,
)

# Cell-type specific networks
ct_networks = model.export_networks(
    threshold=0.9,
    return_cell_type_specific=True,
)

# Access individual cell type networks
for cell_type, network_df in ct_networks.items():
    print(f"\n{cell_type} network: {len(network_df)} edges")
    print(network_df.head())

# Save results
global_ppi.to_csv("inferred_ppi_network.csv", index=False)
```

## Understanding the Model

### Statistical Framework

The model uses a hierarchical Bayesian approach:

1. **Edge Existence**: Each potential edge has a probability of existing
   - Spike-and-slab prior: Binary indicator with Beta prior
   - Horseshoe prior: Continuous shrinkage with heavy-tailed prior

2. **Interaction Strength**: How strongly genes influence each other
   - Global strength per edge
   - Cell-type specific modulation

3. **Expression Model**: Negative binomial likelihood
   - Accounts for overdispersion in count data
   - Library size normalization

### Interpretation

- **Probability** (spike-slab only): Posterior probability that an interaction exists
- **Weight**: Strength of the interaction effect
- **Confidence intervals**: Uncertainty in estimates

### Limitations

1. **Correlation ≠ Causation**: The model identifies co-expression, not direct interactions
2. **Technical noise**: scRNA-seq has high technical variability
3. **Sparsity**: Zero-inflation in single-cell data can affect inference
4. **Prior dependency**: Results influenced by quality of prior edges

## Advanced Features

### Custom Prior Configuration

```python
# Use stronger prior for well-known interactions
model = MBModel(
    adata,
    prior_edges=prior,
    edge_prior_prob=0.3,  # Higher prior probability
)
```

### Horseshoe Prior (Alternative Shrinkage)

```python
# Horseshoe prior for automatic relevance determination
model = MBModel(
    adata,
    prior_edges=prior,
    prior_type="horseshoe",
)

# Export uses magnitude-based thresholding
ppi = model.export_networks(
    threshold=0.95,  # Top 5% of edges by magnitude
)
```

### Prediction and Validation

```python
# Generate expression predictions for validation
predicted_expr = model.predict_expression(
    cell_types=None,  # Random cell types
    n_samples=100,
)
```

## Best Practices

### 1. Data Preparation

- **Quality Control**: Filter low-quality cells and genes
- **Normalization**: Use raw counts (model handles normalization)
- **Gene Selection**: Focus on protein-coding genes
- **Cell Type Annotation**: Ensure accurate cell type labels

### 2. Prior Selection

- Use high-confidence interactions from databases (STRING, BioGRID)
- Include known pathways relevant to your biological system
- Balance between too few (underconstrained) and too many (computational burden)

### 3. Model Training

- Start with default hyperparameters
- Monitor training loss for convergence
- Use early stopping to prevent overfitting
- Try both prior types if unsure

### 4. Result Validation

- Cross-reference with protein interaction databases
- Validate high-confidence predictions experimentally
- Consider biological plausibility
- Check cell-type specificity makes biological sense

### Pathway-Focused Analysis

```python
# Load pathway-specific interactions
pathways = {
    "TCR_signaling": ["CD3D CD3E", "CD3E CD3G", "LCK ZAP70"],
    "Cytotoxicity": ["GNLY PRF1", "GZMB PRF1", "GZMA PRF1"],
    "Antigen_presentation": ["HLA-A B2M", "HLA-B B2M", "HLA-C B2M"],
}

# Combine all pathway edges
all_edges = []
for pathway, edges in pathways.items():
    all_edges.extend(edges)

prior = load_string_prior(all_edges, adata_var_names=adata.var_names)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `batch_size`
   - Use sparse matrices in AnnData
   - Consider GPU with more memory

2. **Poor Convergence**
   - Increase training epochs
   - Adjust learning rate (try 1e-3 to 1e-2)
   - Check data quality

3. **No Edges Found**
   - Lower threshold
   - Ensure prior genes exist in data
   - Check for sufficient expression variation

4. **Numerical Instability**
   - Update PyTorch/Pyro versions
   - Check for extreme expression values
   - Consider log-transformation if needed

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/IsabellaWithnell/mini_bayes_ppi
cd mini_bayes_ppi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[test]"

# Run tests
pytest tests/ -v

# Run linting
ruff check src/
```

## Citation

If you use `mini_bayes_ppi` in your research, please cite:

```bibtex
@software{withnell2025minibayes,
  author = {Withnell, Isabella},
  title = {mini_bayes_ppi: Bayesian Protein-Protein Interaction Inference from Single-Cell RNA-seq},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/IsabellaWithnell/mini_bayes_ppi}
}
```

## License

MIT © Isabella Withnell, 2025

## Acknowledgments

This package builds on:
- [Pyro](https://pyro.ai/) for probabilistic programming
- [AnnData](https://anndata.readthedocs.io/) for single-cell data handling
- [PyTorch](https://pytorch.org/) for deep learning infrastructure

## Support

- 📖 [Documentation](https://mini-bayes-ppi.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/IsabellaWithnell/mini_bayes_ppi/issues)
- 💬 [Discussions](https://github.com/IsabellaWithnell/mini_bayes_ppi/discussions)
