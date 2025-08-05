"""
Context-Specific Protein Interaction Activity Inference from scRNA-seq

This module infers which known protein-protein interactions are likely active
in specific cell types based on gene expression patterns. It does NOT discover
new protein interactions, but rather identifies context-specific activity of
known interactions.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as spsp
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    import pandas as pd
    from anndata import AnnData

__all__ = ["PPIActivityModel", "export_activity_scores"]

logger = logging.getLogger(__name__)


def _validate_edges(n_genes: int, edges: Sequence[tuple[int, int]]) -> torch.Tensor:
    """Validate and create edge tensor from gene pairs."""
    if not edges:
        raise ValueError("No edges provided. Need known PPIs from STRING or similar database.")
    
    edge_set = set()
    for i, j in edges:
        if not (0 <= i < n_genes and 0 <= j < n_genes):
            raise ValueError(f"Edge ({i}, {j}) contains invalid gene indices")
        if i == j:
            logger.warning(f"Self-loop detected and skipped: ({i}, {j})")
            continue
        # Store edges in canonical form (smaller index first)
        edge_set.add((min(i, j), max(i, j)))
    
    if not edge_set:
        raise ValueError("No valid edges after filtering")
    
    # Convert to tensor
    edge_list = list(edge_set)
    return torch.tensor(edge_list, dtype=torch.long).t()


class PPIActivityModel:
    """
    Infer context-specific activity of known protein-protein interactions.
    
    This model takes known PPIs (e.g., from STRING) and infers which interactions
    are likely active in different cell types based on gene co-expression patterns.
    It does NOT discover new interactions.
    """
    
    def __init__(
        self,
        adata: AnnData,
        known_interactions: Iterable[tuple[int, int]],
        *,
        cell_key: str = "cell_type",
        batch_size: int = 512,
        lr: float = 1e-2,
        device: str | None = None,
        interaction_confidence: dict[tuple[int, int], float] | None = None,
        min_expression_percentile: float = 10.0,
    ) -> None:
        """
        Initialize PPI activity inference model.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell expression data
        known_interactions : Iterable[tuple[int, int]]
            Known protein interactions as gene index pairs (e.g., from STRING)
        cell_key : str
            Column in adata.obs with cell type labels
        batch_size : int
            Batch size for training
        lr : float
            Learning rate
        device : str or None
            Device ('cuda', 'cpu', or None for auto)
        interaction_confidence : dict or None
            Prior confidence scores for each interaction (0-1)
        min_expression_percentile : float
            Minimum expression percentile for genes to be considered
        """
        # Device setup
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {self.device}")
        
        # Validate inputs
        if cell_key not in adata.obs.columns:
            raise ValueError(f"Cell type key '{cell_key}' not found in adata.obs")
        
        # Store parameters
        self.adata = adata
        self.batch_size = batch_size
        self.lr = lr
        self.min_expr_pct = min_expression_percentile
        
        # Gene information
        self.genes = list(adata.var_names)
        self.n_genes = len(self.genes)
        self.n_cells = adata.n_obs
        
        # Cell type information
        self.ctypes = sorted(adata.obs[cell_key].astype(str).unique())
        self.n_types = len(self.ctypes)
        ct_lookup = {c: i for i, c in enumerate(self.ctypes)}
        
        self.ct_idx = torch.tensor(
            [ct_lookup[c] for c in adata.obs[cell_key].astype(str)],
            dtype=torch.long,
            device=self.device
        )
        
        # Process known interactions
        edge_list = list(known_interactions)
        self.edge_idx = _validate_edges(self.n_genes, edge_list).to(self.device)
        self.n_edges = self.edge_idx.shape[1]
        
        # Store confidence scores if provided
        if interaction_confidence:
            self.edge_confidence = torch.zeros(self.n_edges, device=self.device)
            for idx, (i, j) in enumerate(edge_list):
                key = (min(i, j), max(i, j))
                self.edge_confidence[idx] = interaction_confidence.get(key, 0.5)
        else:
            self.edge_confidence = torch.ones(self.n_edges, device=self.device) * 0.5
        
        logger.info(
            f"Model initialized: {self.n_genes} genes, {self.n_types} cell types, "
            f"{self.n_edges} known interactions, {self.n_cells} cells"
        )
        
        # Process expression data
        self._prepare_expression_data()
        
        # Build model
        self._build_model()
    
    def _prepare_expression_data(self):
        """Prepare and normalize expression data."""
        # Convert to dense if sparse
        X_arr = self.adata.X.toarray() if spsp.issparse(self.adata.X) else self.adata.X.copy()
        
        # Log-transform and normalize
        X_arr = np.log1p(X_arr)  # log(x + 1) transformation
        
        # Filter low-expression genes for interaction inference
        gene_expr_pct = np.percentile(X_arr, self.min_expr_pct, axis=0)
        self.expressed_mask = torch.tensor(gene_expr_pct > 0, device=self.device)
        
        # Library size normalization
        library_size = np.sum(X_arr, axis=1, keepdims=True)
        X_norm = X_arr / (library_size + 1e-10) * np.median(library_size)
        
        # Convert to tensor
        self.X = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
        
        # Create data loader
        dataset = TensorDataset(self.X, self.ct_idx)
        self.loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
            num_workers=0,
        )
    
    def _model(self, x: torch.Tensor, ct: torch.Tensor) -> None:
        """
        Bayesian model for PPI activity inference.
        
        This model infers which known interactions are active based on
        expression correlation patterns.
        """
        batch_size = x.shape[0]
        
        # Global interaction activity (baseline across all cell types)
        with pyro.plate("edges_global", self.n_edges):
            # Use confidence scores as prior mean
            global_activity = pyro.sample(
                "global_activity",
                dist.Beta(
                    self.edge_confidence * 10 + 1,  # α parameter
                    (1 - self.edge_confidence) * 10 + 1  # β parameter
                )
            )
        
        # Cell type-specific modulation of activity
        with pyro.plate("cell_types", self.n_types, dim=-2):
            with pyro.plate("edges_ct", self.n_edges, dim=-1):
                # How much each cell type modulates each interaction
                ct_modulation = pyro.sample(
                    "ct_modulation",
                    dist.Normal(0.0, 0.3)
                )
        
        # Compute cell-type specific activity scores
        # Shape: (n_edges, n_types) -> select for current batch -> (batch_size, n_edges)
        ct_activity_logits = torch.logit(global_activity).unsqueeze(-1) + ct_modulation
        ct_activity = torch.sigmoid(ct_activity_logits)
        
        # Select activities for current batch's cell types
        batch_activity = ct_activity[:, ct].T  # (batch_size, n_edges)
        
        # Model co-expression given interaction activity
        i, j = self.edge_idx
        
        # Calculate pairwise expression products (co-expression strength)
        expr_i = x[:, i]  # (batch_size, n_edges)
        expr_j = x[:, j]  # (batch_size, n_edges)
        
        # Check if both genes are expressed
        both_expressed = self.expressed_mask[i] & self.expressed_mask[j]
        
        # Co-expression strength (using correlation-like measure)
        # Standardize within batch
        expr_i_std = (expr_i - expr_i.mean(dim=0)) / (expr_i.std(dim=0) + 1e-6)
        expr_j_std = (expr_j - expr_j.mean(dim=0)) / (expr_j.std(dim=0) + 1e-6)
        
        coexpr_strength = expr_i_std * expr_j_std  # Element-wise product
        
        # Expected co-expression given activity
        expected_coexpr = batch_activity * both_expressed.float()
        
        # Observation model: co-expression strength depends on interaction activity
        with pyro.plate("cells", batch_size, dim=-2):
            with pyro.plate("edges_obs", self.n_edges, dim=-1):
                pyro.sample(
                    "coexpr_obs",
                    dist.Normal(expected_coexpr, 0.5),
                    obs=torch.sigmoid(coexpr_strength)  # Squash to [0, 1]
                )
    
    def _build_model(self):
        """Build the model and inference components."""
        self.guide = AutoNormal(self._model)
        self.svi = SVI(
            self._model,
            self.guide,
            pyro.optim.Adam({"lr": self.lr}),
            Trace_ELBO()
        )
    
    def fit(
        self,
        epochs: int = 1000,
        verbose: bool = True,
        log_interval: int = 100,
        patience: int = 100,
        min_delta: float = 1e-4,
    ) -> dict:
        """
        Fit the model using variational inference.
        
        Parameters
        ----------
        epochs : int
            Maximum training epochs
        verbose : bool
            Print progress
        log_interval : int
            Epochs between progress updates
        patience : int
            Early stopping patience
        min_delta : float
            Minimum improvement for early stopping
            
        Returns
        -------
        dict
            Training history
        """
        pyro.clear_param_store()
        
        history = {"loss": []}
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for x_batch, ct_batch in self.loader:
                loss = self.svi.step(x_batch, ct_batch)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            history["loss"].append(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if verbose and epoch % log_interval == 0:
                logger.info(f"[Epoch {epoch:04d}] Loss: {avg_loss:.4f}")
        
        return history
    
    def export_activity_scores(
        self,
        threshold: float = 0.7,
        return_cell_type_specific: bool = False,
        include_confidence: bool = True,
        fdr_control: bool = True,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Export inferred interaction activity scores.
        
        Parameters
        ----------
        threshold : float
            Activity score threshold (0-1)
        return_cell_type_specific : bool
            Return cell-type specific activities
        include_confidence : bool
            Include confidence intervals
        fdr_control : bool
            Apply FDR correction to scores
            
        Returns
        -------
        pd.DataFrame or dict[str, pd.DataFrame]
            Interaction activity scores
        """
        import pandas as pd
        from scipy import stats
        
        # Get posterior estimates
        quantiles = [0.025, 0.5, 0.975] if include_confidence else [0.5]
        posterior = self.guide.quantiles(quantiles)
        
        global_activity = posterior["global_activity"]
        ct_modulation = posterior["ct_modulation"]
        
        if not return_cell_type_specific:
            # Global activity scores
            activity_median = global_activity[len(quantiles)//2].cpu().numpy()
            
            # Apply FDR if requested
            if fdr_control:
                # Convert to p-values (distance from 0.5)
                p_values = 2 * np.minimum(activity_median, 1 - activity_median)
                # Benjamini-Hochberg correction
                from scipy.stats import false_discovery_control
                adjusted_p = false_discovery_control(p_values)
                keep_mask = adjusted_p < (1 - threshold)
            else:
                keep_mask = activity_median >= threshold
            
            # Get edge indices
            i_indices = self.edge_idx[0, keep_mask].cpu().numpy()
            j_indices = self.edge_idx[1, keep_mask].cpu().numpy()
            
            result = {
                "protein1": [self.genes[i] for i in i_indices],
                "protein2": [self.genes[j] for j in j_indices],
                "activity_score": activity_median[keep_mask],
                "prior_confidence": self.edge_confidence[keep_mask].cpu().numpy(),
            }
            
            if include_confidence:
                result["activity_lower"] = global_activity[0][keep_mask].cpu().numpy()
                result["activity_upper"] = global_activity[-1][keep_mask].cpu().numpy()
                result["confidence_width"] = result["activity_upper"] - result["activity_lower"]
            
            df = pd.DataFrame(result)
            df = df.sort_values("activity_score", ascending=False)
            
            # Add interpretation column
            df["interpretation"] = df["activity_score"].apply(
                lambda x: "High" if x > 0.8 else "Moderate" if x > 0.6 else "Low"
            )
            
            return df
            
        else:
            # Cell type-specific activities
            results = {}
            
            for ct_idx, ct_name in enumerate(self.ctypes):
                # Compute cell-type specific activity
                ct_logits = (
                    torch.logit(global_activity[len(quantiles)//2]) +
                    ct_modulation[len(quantiles)//2][:, ct_idx]
                )
                ct_activity = torch.sigmoid(ct_logits).cpu().numpy()
                
                # Filter by threshold
                keep_mask = ct_activity >= threshold
                
                i_indices = self.edge_idx[0, keep_mask].cpu().numpy()
                j_indices = self.edge_idx[1, keep_mask].cpu().numpy()
                
                ct_result = {
                    "protein1": [self.genes[i] for i in i_indices],
                    "protein2": [self.genes[j] for j in j_indices],
                    "activity_score": ct_activity[keep_mask],
                    "global_activity": global_activity[len(quantiles)//2][keep_mask].cpu().numpy(),
                }
                
                df = pd.DataFrame(ct_result)
                df["cell_type_specificity"] = df["activity_score"] - df["global_activity"]
                df = df.sort_values("activity_score", ascending=False)
                
                results[ct_name] = df
            
            return results
    
    def plot_activity_heatmap(self, top_n: int = 50) -> None:
        """
        Plot heatmap of interaction activities across cell types.
        
        Parameters
        ----------
        top_n : int
            Number of top variable interactions to show
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get posterior estimates
        posterior = self.guide.quantiles([0.5])
        global_activity = posterior["global_activity"][0]
        ct_modulation = posterior["ct_modulation"][0]
        
        # Compute cell-type specific activities
        activities = []
        for ct_idx in range(self.n_types):
            ct_logits = torch.logit(global_activity) + ct_modulation[:, ct_idx]
            activities.append(torch.sigmoid(ct_logits).cpu().numpy())
        
        activities = np.array(activities).T  # (n_edges, n_types)
        
        # Find most variable interactions
        activity_var = np.var(activities, axis=1)
        top_indices = np.argsort(activity_var)[-top_n:]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Prepare labels
        edge_labels = []
        for idx in top_indices:
            i, j = self.edge_idx[:, idx].cpu().numpy()
            edge_labels.append(f"{self.genes[i]}-{self.genes[j]}")
        
        # Plot
        sns.heatmap(
            activities[top_indices],
            xticklabels=self.ctypes,
            yticklabels=edge_labels,
            cmap="RdBu_r",
            center=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Interaction Activity"},
            ax=ax
        )
        
        ax.set_title("Cell Type-Specific Protein Interaction Activity")
        ax.set_xlabel("Cell Type")
        ax.set_ylabel("Protein Interaction")
        
        plt.tight_layout()
        plt.show()
    
    def validate_against_known_complexes(
        self,
        known_complexes: dict[str, list[str]]
    ) -> pd.DataFrame:
        """
        Validate results against known protein complexes.
        
        Parameters
        ----------
        known_complexes : dict
            Dictionary mapping complex names to lists of gene names
            
        Returns
        -------
        pd.DataFrame
            Validation metrics for each complex
        """
        import pandas as pd
        
        # Get activity scores
        activities = self.export_activity_scores(threshold=0.0)  # Get all
        
        # Create edge set for easy lookup
        active_edges = set()
        for _, row in activities.iterrows():
            if row["activity_score"] > 0.7:
                active_edges.add((row["protein1"], row["protein2"]))
                active_edges.add((row["protein2"], row["protein1"]))
        
        results = []
        for complex_name, genes in known_complexes.items():
            # Count possible edges in complex
            n_genes = len(genes)
            n_possible = n_genes * (n_genes - 1) // 2
            
            # Count how many are active
            n_active = 0
            for i, g1 in enumerate(genes):
                for g2 in genes[i+1:]:
                    if (g1, g2) in active_edges or (g2, g1) in active_edges:
                        n_active += 1
            
            results.append({
                "complex": complex_name,
                "n_proteins": n_genes,
                "n_possible_interactions": n_possible,
                "n_active_interactions": n_active,
                "coverage": n_active / n_possible if n_possible > 0 else 0
            })
        
        return pd.DataFrame(results)


def export_activity_scores(
    model: PPIActivityModel,
    threshold: float = 0.7,
    **kwargs
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Export activity scores from a fitted model.
    
    Parameters
    ----------
    model : PPIActivityModel
        Fitted model
    threshold : float
        Activity threshold
    **kwargs
        Additional arguments
        
    Returns
    -------
    pd.DataFrame or dict
        Activity scores
    """
    return model.export_activity_scores(threshold, **kwargs)
