# mini_bayes_ppi

**Context-Specific Protein Interaction Activity Inference from Single-Cell RNA-seq**

[![PyPI](https://img.shields.io/pypi/v/mini_bayes_ppi.svg)](https://pypi.org/project/mini_bayes_ppi/)
[![CI](https://github.com/IsabellaWithnell/mini_bayes_ppi/actions/workflows/ci.yml/badge.svg)](https://github.com/IsabellaWithnell/mini_bayes_ppi/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚠️ Important: What This Tool Does and Doesn't Do

### ✅ **DOES**: 
- Infer which **known** protein interactions are likely **active** in specific cell types
- Identify cell-type-specific pathway activity based on co-expression
- Provide uncertainty estimates for interaction activity
- Help prioritize interactions for experimental validation

### ❌ **DOES NOT**:
- Discover new protein-protein interactions
- Prove physical interaction between proteins

---

## Overview

`mini_bayes_ppi` uses Bayesian inference to determine which known protein-protein interactions are likely active in different cell types based on gene co-expression patterns in single-cell RNA-seq data. 

### Scientific Rationale

While we cannot infer new protein interactions from RNA data alone (due to post-translational modifications, localization, etc.), we CAN infer which known interactions are likely functional in specific contexts:

1. **Active interactions often show co-expression**: If two proteins interact functionally, their genes are often co-regulated
2. **Cell-type specificity**: Different cell types use different molecular machinery
3. **Pathway activity**: Co-expression of pathway members suggests pathway activity

### Key Features

- 🎯 **Bayesian Framework**: Provides uncertainty quantification for activity scores
- 🧬 **Cell-Type Specific**: Identifies which interactions are active in which cell types
- 📊 **STRING Integration**: Uses high-confidence interactions from STRING database
- 🔬 **Validation Tools**: Compare results against known protein complexes
- 📈 **FDR Control**: Statistical correction for multiple testing

## Installation

```bash
# Basic installation
pip install mini_bayes_ppi

# With visualization support
pip install mini_bayes_ppi[viz]

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
- pandas, numpy, scipy

## Quick Start

```python
import anndata as ad
from mini_bayes_ppi import PPIActivityModel, load_string_interactions

# Load your single-cell data
adata = ad.read_h5ad("pbmc3k.h5ad")

# Load known interactions from STRING (or your own list)
interactions, confidence_scores = load_string_interactions(
    "string_interactions.tsv",  # Download from STRING database
    adata_var_names=adata.var_names,
    score_cutoff=700,  # High confidence
    include_confidence=True
)

# Or use a simple list of known interactions
interactions, confidence_scores = load_string_interactions(
    [
        "CD3D CD3E",       # T cell receptor
        "CD4 HLA-DRB1",    # MHC-II interaction
        "ITGAL ICAM1",     # Cell adhesion
        "CD8A CD8B",       # CD8 complex
    ],
    adata_var_names=adata.var_names
)

# Initialize model
model = PPIActivityModel(
    adata,
    interactions,
    cell_key="cell_type",
    interaction_confidence=confidence_scores,
    device="cuda",  # Use GPU if available
)

# Fit the model
history = model.fit(
    epochs=1000,
    patience=100,
    verbose=True
)

# Export activity scores
activities = model.export_activity_scores(
    threshold=0.7,
    include_confidence=True,
    fdr_control=True,  # Apply multiple testing correction
)

print("\nTop Active Interactions:")
print(activities.head(10))

# Get cell-type specific activities
ct_activities = model.export_activity_scores(
    threshold=0.7,
    return_cell_type_specific=True
)

for cell_type, df in ct_activities.items():
    print(f"\n{cell_type}: {len(df)} active interactions")
    print(df.head(3))
```

## Detailed Usage

### 1. Loading Known Interactions

#### From STRING Database

```python
# Download STRING data for your organism
# Go to https://string-db.org/cgi/download
# Download: 9606.protein.links.v11.5.txt.gz (human)

interactions, confidences = load_string_interactions(
    "9606.protein.links.v11.5.txt",
    adata_var_names=adata.var_names,
    score_cutoff=700,  # 0-1000 scale
    score_type="combined_score"
)
```

#### From Pathway Databases

```python
# Load interactions from pathways (KEGG, Reactome, etc.)
interactions, pathways = load_pathway_interactions(
    "c2.cp.kegg.v7.5.1.symbols.gmt",  # MSigDB GMT file
    adata_var_names=adata.var_names,
    pathway_name="T_CELL_RECEPTOR"  # Optional: specific pathway
)
```

#### Custom Interactions

```python
# Define your own high-confidence interactions
my_interactions = [
    ("GENE1", "GENE2", 0.9),  # With confidence
    ("GENE3", "GENE4", 0.85),
    ("GENE5", "GENE6"),  # Default confidence = 0.5
]

interactions, confidences = load_string_interactions(
    my_interactions,
    adata_var_names=adata.var_names
)
```

### 2. Model Configuration

```python
model = PPIActivityModel(
    adata,
    interactions,
    cell_key="cell_type",
    batch_size=512,
    lr=0.01,
    device="cuda",
    interaction_confidence=confidences,  # Prior knowledge
    min_expression_percentile=10.0,  # Filter lowly expressed genes
)
```

### 3. Interpreting Results

```python
# Get activity scores
activities = model.export_activity_scores(threshold=0.7)

# Interpret the output
for _, row in activities.head(10).iterrows():
    print(f"{row['protein1']} - {row['protein2']}:")
    print(f"  Activity: {row['activity_score']:.2f}")
    print(f"  Prior confidence: {row['prior_confidence']:.2f}")
    print(f"  Interpretation: {row['interpretation']}")
    if 'confidence_width' in row:
        print(f"  Uncertainty: ±{row['confidence_width']:.3f}")
```

### 4. Visualization

```python
# Heatmap of top variable interactions across cell types
model.plot_activity_heatmap(top_n=50)

# Export for Cytoscape
export_to_cytoscape(
    activities,
    "network.csv",
    node_attributes={"CD3D": {"type": "receptor"}}  # Optional
)
```

### 5. Validation

```python
# Validate against known complexes
known_complexes = {
    "TCR_complex": ["CD3D", "CD3E", "CD3G", "CD247"],
    "MHC_I": ["HLA-A", "HLA-B", "HLA-C", "B2M"],
    "Proteasome": ["PSMA1", "PSMA2", "PSMB1", "PSMB2"],
}

validation = model.validate_against_known_complexes(known_complexes)
print(validation)
```

## Best Practices

### 1. Data Preparation
- Use **raw counts** (model handles normalization)
- Ensure **accurate cell type annotations**
- Include **sufficient cells per type** (>50 recommended)
- Focus on **protein-coding genes**

### 2. Interaction Selection
- Use **high-confidence** interactions (STRING score > 700)
- Include **pathway-relevant** interactions
- Consider **tissue-specific** interaction databases
- Balance coverage and confidence

### 3. Interpretation
- High activity (>0.8) suggests **functional interaction**
- Compare to **global activity** for cell-type specificity
- Look for **pathway coherence** (multiple interactions in same pathway)
- Always consider **biological plausibility**

### 4. Validation
- Compare to **known biology** of cell types
- Check **pathway enrichment** of active interactions
- Validate key findings **experimentally**
- Use **orthogonal data** (proteomics, IF, IP) when available

## Limitations and Caveats

1. **Cannot discover new interactions** - only assesses known PPIs
2. **RNA correlation ≠ protein interaction** - many factors affect protein function
3. **Technical noise** - scRNA-seq has dropout and batch effects
4. **Temporal dynamics** - snapshot data may miss transient interactions
5. **Cellular localization** - not captured by expression alone

## Example Analysis: PBMC Immune Interactions

```python
# Complete example with immune cells
import scanpy as sc
import mini_bayes_ppi as mbp

# Load and preprocess data
adata = sc.datasets.pbmc3k()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Load immune-relevant interactions
interactions, conf = mbp.load_string_interactions(
    [
        # T cell signaling
        "CD3D CD3E", "CD3E CD3G", "CD3G CD247",
        "CD4 HLA-DRA", "CD4 HLA-DRB1",
        "CD8A CD8B",
        
        # Cytotoxicity
        "GNLY PRF1", "GZMB PRF1",
        
        # Cell adhesion
        "ITGAL ICAM1", "ITGB2 ICAM1",
        
        # Cytokine signaling
        "IL2 IL2RA", "IL2 IL2RB",
    ],
    adata_var_names=adata.var_names
)

# Run analysis
model = mbp.PPIActivityModel(
    adata,
    interactions,
    cell_key="louvain",  # or your annotation
)

model.fit(epochs=1000)

# Examine T cell specific interactions
activities = model.export_activity_scores(
    return_cell_type_specific=True
)

t_cell_activities = activities.get("CD4 T cells", activities.get("2"))  # Cluster 2
print("T cell-specific active interactions:")
print(t_cell_activities[t_cell_activities['activity_score'] > 0.8])
```

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{withnell2025minibayes,
  author = {Withnell, Isabella},
  title = {mini_bayes_ppi: Context-Specific Protein Interaction Activity Inference},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/IsabellaWithnell/mini_bayes_ppi}
}
```

Also cite STRING database if using their data:
```bibtex
@article{szklarczyk2019string,
  title={STRING v11: protein--protein association networks with increased coverage},
  author={Szklarczyk, Damian and others},
  journal={Nucleic acids research},
  volume={47},
  number={D1},
  pages={D607--D613},
  year={2019}
}
```

## Support

- 📖 [Documentation](https://mini-bayes-ppi.readthedocs.io)
- 🐛 [Issues](https://github.com/IsabellaWithnell/mini_bayes_ppi/issues)
- 💬 [Discussions](https://github.com/IsabellaWithnell/mini_bayes_ppi/discussions)
