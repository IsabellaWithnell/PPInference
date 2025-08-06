#!/usr/bin/env python
"""Example demonstrating coverage-based confidence and visualization features."""

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from mini_bayes_ppi import MBModel, load_string_prior
from mini_bayes_ppi import (
    plot_gene_coverage, 
    plot_network,
    plot_training_history,
    plot_edge_confidence,
    plot_cell_type_comparison,
)


def create_example_data():
    """Create example scRNA-seq data with varying coverage."""
    np.random.seed(42)
    n_cells = 500
    n_genes = 30
    
    # Create data with different coverage levels
    X = np.zeros((n_cells, n_genes))
    
    # High coverage genes (0-9): high expression, low noise
    X[:, :10] = np.random.negative_binomial(20, 0.3, size=(n_cells, 10))
    
    # Medium coverage genes (10-19): moderate expression
    X[:, 10:20] = np.random.negative_binomial(10, 0.5, size=(n_cells, 10))
    
    # Low coverage genes (20-29): low expression, high dropout
    X[:, 20:] = np.random.negative_binomial(3, 0.8, size=(n_cells, 10))
    
    # Add some correlations
    X[:, 1] = X[:, 0] * 1.5 + np.random.poisson(2, n_cells)  # High coverage correlation
    X[:, 11] = X[:, 10] * 1.2 + np.random.poisson(1, n_cells)  # Medium coverage
    X[:, 21] = X[:, 20] * 1.1 + np.random.poisson(0.5, n_cells)  # Low coverage
    
    # Create AnnData object
    adata = ad.AnnData(X.astype(np.float32))
    adata.obs["cell_type"] = (
        ["T_cell"] * 200 + 
        ["B_cell"] * 150 + 
        ["Monocyte"] * 150
    )
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    
    return adata


def main():
    """Run example analysis with coverage weighting and visualization."""
    
    print("Creating example data...")
    adata = create_example_data()
    
    # Define prior edges across different coverage levels
    prior_edges = [
        "Gene_0 Gene_1",   # High coverage pair
        "Gene_0 Gene_10",  # High-medium coverage
        "Gene_10 Gene_11", # Medium coverage pair
        "Gene_20 Gene_21", # Low coverage pair
        "Gene_5 Gene_15",  # Mixed coverage
        "Gene_25 Gene_26", # Low coverage
        # Add some false edges
        "Gene_2 Gene_3",
        "Gene_12 Gene_13",
        "Gene_22 Gene_23",
    ]
    
    prior = load_string_prior(prior_edges, adata_var_names=adata.var_names)
    
    print("\n=== Running Standard Model (No Coverage Weighting) ===")
    model_standard = MBModel(
        adata,
        prior,
        cell_key="cell_type",
        batch_size=100,
        lr=0.01,
        device="cpu",
        coverage_weight_type="none",  # No coverage weighting
    )
    
    history_standard = model_standard.fit(epochs=200, verbose=True, log_interval=50)
    networks_standard = model_standard.export_networks(
        threshold=0.7,
        include_confidence=True,
        include_reliability=False,  # No reliability scores for standard model
    )
    
    print(f"\nStandard model found {len(networks_standard)} edges")
    print("\nTop edges (standard):")
    print(networks_standard.head(10))
    
    print("\n=== Running Enhanced Model (With Coverage Weighting) ===")
    model_enhanced = MBModel(
        adata,
        prior,
        cell_key="cell_type",
        batch_size=100,
        lr=0.01,
        device="cpu",
        coverage_weight_type="adaptive",  # Enable coverage weighting
        min_coverage=2.0,  # Minimum mean expression threshold
    )
    
    history_enhanced = model_enhanced.fit(epochs=200, verbose=True, log_interval=50)
    networks_enhanced = model_enhanced.export_networks(
        threshold=0.7,
        include_confidence=True,
        include_reliability=True,  # Include reliability scores
    )
    
    print(f"\nEnhanced model found {len(networks_enhanced)} edges")
    print("\nTop edges (enhanced with reliability):")
    print(networks_enhanced.head(10))
    
    # Get gene statistics
    gene_stats = model_enhanced.get_gene_stats()
    print("\nGene Statistics:")
    print(gene_stats.head(10))
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    
    # 1. Gene coverage analysis
    fig1 = plot_gene_coverage(model_enhanced)
    fig1.savefig("gene_coverage_analysis.png", dpi=150, bbox_inches='tight')
    print("Saved: gene_coverage_analysis.png")
    
    # 2. Training history comparison
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(history_standard["loss"], label="Standard Model", alpha=0.7)
    ax1.plot(history_enhanced["loss"], label="Coverage-Weighted Model", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ELBO Loss")
    ax1.set_title("Training Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot just enhanced model history
    ax2.plot(history_enhanced["loss"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ELBO Loss")
    ax2.set_title("Coverage-Weighted Model Training")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Training History", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig2.savefig("training_history.png", dpi=150, bbox_inches='tight')
    print("Saved: training_history.png")
    
    # 3. Network visualization
    fig3 = plot_network(
        networks_enhanced,
        top_n=20,
        node_size_by="reliability" if "gene_i_reliability" in networks_enhanced.columns else "degree",
        edge_width_by="weight",
        layout="spring",
        title="Enhanced PPI Network (Coverage-Weighted)"
    )
    fig3.savefig("network_graph.png", dpi=150, bbox_inches='tight')
    print("Saved: network_graph.png")
    
    # 4. Edge confidence intervals
    if "prob_lower" in networks_enhanced.columns or "weight_lower" in networks_enhanced.columns:
        fig4 = plot_edge_confidence(networks_enhanced, top_n=15)
        fig4.savefig("edge_confidence.png", dpi=150, bbox_inches='tight')
        print("Saved: edge_confidence.png")
    
    # 5. Cell type comparison
    ct_networks = model_enhanced.export_networks(
        threshold=0.7,
        return_cell_type_specific=True,
        include_reliability=True,
    )
    
    fig5 = plot_cell_type_comparison(
        ct_networks,
        gene_pair=("Gene_0", "Gene_1"),  # Highlight high-coverage pair
        top_edges=15,
    )
    fig5.savefig("cell_type_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved: cell_type_comparison.png")
    
    # 6. Compare reliability impact
    if "edge_reliability" in networks_enhanced.columns:
        fig6, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot: weight vs reliability
        axes[0].scatter(
            networks_enhanced["edge_reliability"],
            networks_enhanced["weight"],
            alpha=0.6
        )
        axes[0].set_xlabel("Edge Reliability Score")
        axes[0].set_ylabel("Edge Weight")
        axes[0].set_title("Edge Weight vs Reliability")
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of reliability scores
        axes[1].hist(networks_enhanced["edge_reliability"], bins=20, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel("Edge Reliability Score")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Edge Reliability")
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle("Impact of Coverage-Based Reliability", fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig6.savefig("reliability_analysis.png", dpi=150, bbox_inches='tight')
        print("Saved: reliability_analysis.png")
    
    print("\n=== Analysis Complete ===")
    print(f"Standard model edges: {len(networks_standard)}")
    print(f"Enhanced model edges: {len(networks_enhanced)}")
    
    # Show which edges were affected by coverage
    if "edge_reliability" in networks_enhanced.columns:
        low_reliability = networks_enhanced[networks_enhanced["edge_reliability"] < 0.5]
        if not low_reliability.empty:
            print(f"\nLow reliability edges (< 0.5): {len(low_reliability)}")
            print(low_reliability[["gene_i", "gene_j", "edge_reliability", "weight"]].head())
    
    plt.show()


if __name__ == "__main__":
    main()
