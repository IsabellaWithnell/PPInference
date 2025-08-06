"""Visualization utilities for mini_bayes_ppi."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd
    from .core import MBModel

__all__ = [
    "plot_gene_coverage",
    "plot_network",
    "plot_training_history",
    "plot_edge_confidence",
    "plot_cell_type_comparison",
]


def plot_gene_coverage(model: MBModel, figsize: tuple[int, int] = (12, 10)) -> plt.Figure:
    """Plot gene coverage statistics.
    
    Parameters
    ----------
    model : MBModel
        Fitted model with gene statistics
    figsize : tuple
        Figure size (width, height)
    
    Returns
    -------
    plt.Figure
        Matplotlib figure with coverage plots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Mean expression distribution
    axes[0, 0].hist(model.gene_mean.cpu(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(model.min_coverage, color='red', linestyle='--', 
                       label=f'Min coverage: {model.min_coverage:.1f}')
    axes[0, 0].set_xlabel('Mean Expression')
    axes[0, 0].set_ylabel('Number of Genes')
    axes[0, 0].set_title('Gene Expression Coverage')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dropout rate distribution
    axes[0, 1].hist(model.gene_dropout.cpu(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Dropout Rate')
    axes[0, 1].set_ylabel('Number of Genes')
    axes[0, 1].set_title('Gene Dropout Rates')
    axes[0, 1].grid(True, alpha=0.3)
    
    # CV distribution
    cv_values = model.gene_cv.cpu().numpy()
    cv_values = cv_values[cv_values < np.percentile(cv_values, 95)]  # Remove outliers
    axes[1, 0].hist(cv_values, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Coefficient of Variation')
    axes[1, 0].set_ylabel('Number of Genes')
    axes[1, 0].set_title('Expression Variability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reliability scores
    axes[1, 1].hist(model.gene_reliability.cpu(), bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Reliability Score')
    axes[1, 1].set_ylabel('Number of Genes')
    axes[1, 1].set_title('Composite Gene Reliability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Gene Coverage and Reliability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_network(
    network_df: pd.DataFrame,
    top_n: int = 50,
    node_size_by: str = "degree",
    edge_width_by: str = "weight",
    layout: str = "spring",
    figsize: tuple[int, int] = (12, 10),
    title: str = "Protein-Protein Interaction Network",
) -> plt.Figure:
    """Plot PPI network graph.
    
    Parameters
    ----------
    network_df : pd.DataFrame
        Network dataframe from export_networks
    top_n : int
        Show top N edges by weight
    node_size_by : str
        'degree', 'reliability', or 'uniform'
    edge_width_by : str
        'weight', 'probability', or 'uniform'
    layout : str
        'spring', 'circular', or 'kamada_kawai'
    figsize : tuple
        Figure size
    title : str
        Plot title
    
    Returns
    -------
    plt.Figure
        Network visualization
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required for network visualization. Install with: pip install networkx")
    
    # Select top edges
    if "weight" in network_df.columns:
        df_sorted = network_df.nlargest(min(top_n, len(network_df)), "weight")
    else:
        df_sorted = network_df.head(top_n)
    
    # Create graph
    G = nx.Graph()
    
    # Add edges
    for _, row in df_sorted.iterrows():
        weight = row.get("weight", 1.0)
        G.add_edge(row["gene_i"], row["gene_j"], weight=abs(weight))
    
    # Calculate layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Node sizes
    if node_size_by == "degree":
        node_sizes = [300 * G.degree(n) for n in G.nodes()]
    elif node_size_by == "reliability" and "gene_i_reliability" in network_df.columns:
        reliability_dict = {}
        for _, row in network_df.iterrows():
            reliability_dict[row["gene_i"]] = row.get("gene_i_reliability", 0.5)
            reliability_dict[row["gene_j"]] = row.get("gene_j_reliability", 0.5)
        node_sizes = [1000 * reliability_dict.get(n, 0.5) for n in G.nodes()]
    else:
        node_sizes = 500
    
    # Edge widths
    if edge_width_by == "weight":
        edge_widths = [3 * G[u][v]["weight"] for u, v in G.edges()]
    elif edge_width_by == "probability" and "probability" in network_df.columns:
        prob_dict = {(row["gene_i"], row["gene_j"]): row["probability"] 
                    for _, row in df_sorted.iterrows()}
        edge_widths = [3 * prob_dict.get((u, v), prob_dict.get((v, u), 0.5)) 
                      for u, v in G.edges()]
    else:
        edge_widths = 1.5
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos, 
        node_color='lightblue',
        node_size=node_sizes,
        alpha=0.7,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.5,
        edge_color='gray',
        ax=ax
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    if "probability" in network_df.columns:
        avg_prob = df_sorted["probability"].mean()
        ax.text(0.02, 0.98, f'Avg probability: {avg_prob:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_training_history(
    history: dict,
    figsize: tuple[int, int] = (10, 6),
    show_smoothed: bool = True,
    window_size: int = 10,
) -> plt.Figure:
    """Plot training loss history.
    
    Parameters
    ----------
    history : dict
        Training history from model.fit()
    figsize : tuple
        Figure size
    show_smoothed : bool
        Whether to show smoothed loss curve
    window_size : int
        Window size for smoothing
    
    Returns
    -------
    plt.Figure
        Training history plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    losses = history["loss"]
    epochs = range(1, len(losses) + 1)
    
    # Plot raw loss
    ax.plot(epochs, losses, 'b-', alpha=0.3, label='Raw loss')
    
    # Plot smoothed loss
    if show_smoothed and len(losses) > window_size:
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_epochs = range(window_size//2 + 1, len(losses) - window_size//2 + 1)
        ax.plot(smoothed_epochs, smoothed, 'b-', linewidth=2, label=f'Smoothed (window={window_size})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('ELBO Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add final loss annotation
    final_loss = losses[-1]
    ax.annotate(f'Final: {final_loss:.2f}',
                xy=(len(losses), final_loss),
                xytext=(len(losses)*0.9, final_loss*1.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_edge_confidence(
    network_df: pd.DataFrame,
    top_n: int = 30,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot edge confidence intervals.
    
    Parameters
    ----------
    network_df : pd.DataFrame
        Network dataframe with confidence intervals
    top_n : int
        Number of top edges to show
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        Confidence interval plot
    """
    # Check for required columns
    has_prob_ci = "prob_lower" in network_df.columns and "prob_upper" in network_df.columns
    has_weight_ci = "weight_lower" in network_df.columns and "weight_upper" in network_df.columns
    
    if not (has_prob_ci or has_weight_ci):
        raise ValueError("No confidence intervals found. Run export_networks with include_confidence=True")
    
    # Sort by weight or probability
    if "weight" in network_df.columns:
        df_sorted = network_df.nlargest(min(top_n, len(network_df)), "weight")
    elif "probability" in network_df.columns:
        df_sorted = network_df.nlargest(min(top_n, len(network_df)), "probability")
    else:
        df_sorted = network_df.head(top_n)
    
    # Create edge labels
    edge_labels = [f"{row['gene_i']}-{row['gene_j']}" for _, row in df_sorted.iterrows()]
    
    n_subplots = sum([has_prob_ci, has_weight_ci])
    fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
    if n_subplots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot probability confidence intervals
    if has_prob_ci:
        ax = axes[plot_idx]
        plot_idx += 1
        
        probs = df_sorted["probability"].values
        lower = df_sorted["prob_lower"].values
        upper = df_sorted["prob_upper"].values
        
        positions = range(len(edge_labels))
        ax.errorbar(probs, positions, 
                   xerr=[probs - lower, upper - probs],
                   fmt='o', capsize=5, capthick=2,
                   markersize=6, elinewidth=1.5)
        
        ax.set_yticks(positions)
        ax.set_yticklabels(edge_labels, fontsize=8)
        ax.set_xlabel('Edge Probability', fontsize=12)
        ax.set_title('Edge Probability with 90% CI', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)
        
        # Add threshold line
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    # Plot weight confidence intervals
    if has_weight_ci:
        ax = axes[plot_idx]
        
        weights = df_sorted["weight"].values
        lower = df_sorted["weight_lower"].values
        upper = df_sorted["weight_upper"].values
        
        positions = range(len(edge_labels))
        ax.errorbar(weights, positions,
                   xerr=[weights - lower, upper - weights],
                   fmt='o', capsize=5, capthick=2,
                   markersize=6, elinewidth=1.5, color='green')
        
        ax.set_yticks(positions)
        ax.set_yticklabels(edge_labels, fontsize=8)
        ax.set_xlabel('Edge Weight', fontsize=12)
        ax.set_title('Edge Weight with 90% CI', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add zero line
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero')
    
    plt.suptitle('Edge Confidence Intervals', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_cell_type_comparison(
    ct_networks: dict[str, pd.DataFrame],
    gene_pair: tuple[str, str] | None = None,
    top_edges: int = 20,
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure:
    """Compare networks across cell types.
    
    Parameters
    ----------
    ct_networks : dict
        Cell-type specific networks from export_networks
    gene_pair : tuple or None
        Specific gene pair to highlight
    top_edges : int
        Number of top edges to compare
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        Comparison plot
    """
    import pandas as pd
    
    # Collect all edges across cell types
    all_edges = set()
    for df in ct_networks.values():
        for _, row in df.iterrows():
            all_edges.add((row["gene_i"], row["gene_j"]))
    
    # If gene_pair specified, ensure it's included
    if gene_pair:
        all_edges.add(gene_pair)
        all_edges.add((gene_pair[1], gene_pair[0]))  # Both directions
    
    # Create matrix of weights
    edge_list = sorted(list(all_edges))[:top_edges]
    cell_types = sorted(ct_networks.keys())
    
    weight_matrix = np.zeros((len(edge_list), len(cell_types)))
    
    for j, ct in enumerate(cell_types):
        df = ct_networks[ct]
        edge_dict = {}
        for _, row in df.iterrows():
            edge_dict[(row["gene_i"], row["gene_j"])] = row.get("weight", 0)
            edge_dict[(row["gene_j"], row["gene_i"])] = row.get("weight", 0)
        
        for i, edge in enumerate(edge_list):
            weight_matrix[i, j] = edge_dict.get(edge, 0)
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                   gridspec_kw={'width_ratios': [3, 1]})
    
    # Heatmap
    sns.heatmap(weight_matrix, 
                xticklabels=cell_types,
                yticklabels=[f"{e[0]}-{e[1]}" for e in edge_list],
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Edge Weight'},
                ax=ax1)
    
    ax1.set_title('Edge Weights Across Cell Types', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Cell Type', fontsize=10)
    ax1.set_ylabel('Gene Pair', fontsize=10)
    
    # Highlight specific gene pair
    if gene_pair and gene_pair in edge_list:
        idx = edge_list.index(gene_pair)
        ax1.add_patch(plt.Rectangle((0, idx), len(cell_types), 1, 
                                   fill=False, edgecolor='green', lw=2))
    
    # Summary statistics
    mean_weights = weight_matrix.mean(axis=0)
    std_weights = weight_matrix.std(axis=0)
    
    x_pos = np.arange(len(cell_types))
    ax2.bar(x_pos, mean_weights, yerr=std_weights, capsize=5,
           color='skyblue', edgecolor='black', linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cell_types, rotation=45, ha='right')
    ax2.set_ylabel('Mean Edge Weight', fontsize=10)
    ax2.set_title('Average Network Strength', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Cell Type Network Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
