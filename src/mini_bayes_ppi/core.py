from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from torch.utils.data import DataLoader, TensorDataset
import scipy.sparse as spsp

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    
    import pandas as pd
    from anndata import AnnData

__all__ = ["MBModel", "export_networks"]

logger = logging.getLogger(__name__)


def _edge_index(n_genes: int, edges: Sequence[tuple[int, int]]) -> torch.Tensor:
    """Return a symmetric edge index tensor for the given gene pairs."""
    if not edges:
        raise ValueError("prior_edges is empty; nothing to learn.")
    
    # Validate edge indices
    for i, j in edges:
        if not (0 <= i < n_genes and 0 <= j < n_genes):
            raise ValueError(f"Edge ({i}, {j}) contains invalid gene indices")
        if i == j:
            raise ValueError(f"Self-loop detected: ({i}, {j})")
    
    idx = torch.tensor(edges, dtype=torch.long).t()
    rev = idx[[1, 0], :]
    idx = torch.cat([idx, rev], dim=1).unique(dim=1)
    return idx


class MBModel:
    def __init__(
        self,
        adata: AnnData,
        prior_edges: Iterable[tuple[int, int]],
        *,
        cell_key: str = "cell_type",
        batch_size: int = 1024,
        temp: float = 1,
        lr: float = 5e-3,
        device: str | None = None,
        prior_type: str = "spike_slab",  # New: choice of prior
        edge_prior_prob: float = 0.1,    # New: customizable prior
    ) -> None:
        """Initialize the Bayesian PPI inference model.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with gene expression counts
        prior_edges : Iterable[tuple[int, int]]
            Prior edges as gene index pairs
        cell_key : str
            Key in adata.obs containing cell type labels
        batch_size : int
            Batch size for training
        lr : float
            Learning rate for optimization
        device : str or None
            Device to use ('cuda', 'cpu', or None for auto)
        prior_type : str
            Type of prior: 'spike_slab' or 'horseshoe'
        edge_prior_prob : float
            Prior probability of edge existence (for spike-slab)
        """
        # Device setup
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {self.device}")
        
        # Validate inputs
        if cell_key not in adata.obs.columns:
            raise ValueError(f"Cell type key '{cell_key}' not found in adata.obs")
        
        if prior_type not in ["spike_slab", "horseshoe"]:
            raise ValueError(f"Unknown prior_type: {prior_type}")
        
        # Store parameters
        self.adata = adata
        self.batch_size = batch_size
        self.lr = lr
        self.prior_type = prior_type
        self.edge_prior_prob = edge_prior_prob
        self.z_temperature   = temp

        # Gene information
        self.genes: list[str] = list(adata.var_names)
        self.n_genes = len(self.genes)
        self.n_cells = adata.n_obs

        # Cell type information
        self.ctypes: list[str] = sorted(adata.obs[cell_key].astype(str).unique())
        self.n_types = len(self.ctypes)
        ct_lookup: dict[str, int] = {c: i for i, c in enumerate(self.ctypes)}
        
        self.ct_idx = torch.tensor(
            [ct_lookup[c] for c in adata.obs[cell_key].astype(str)],
            dtype=torch.long,
            device=self.device
        )

        # Edge information
        edge_list = list(prior_edges)
        self.edge_idx = _edge_index(self.n_genes, edge_list).to(self.device)
        self.n_edges = self.edge_idx.shape[1]
        
        logger.info(f"Model initialized: {self.n_genes} genes, {self.n_types} cell types, "
                   f"{self.n_edges} edges, {self.n_cells} cells")

        # Expression data
        if spsp.issparse(adata.X):
            X_arr = adata.X.toarray()
        else:
            X_arr = adata.X
        x = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
        
        self.X = x.to(self.device)

        # Library size normalization
        libsize = torch.sum(self.X, dim=1, keepdim=True) + 1.0
        self.log_lib = libsize.log()

        # Create data loader
        dataset = TensorDataset(self.X, self.ct_idx, self.log_lib)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )

        self._build_model()

    def _model(self, xs: torch.Tensor, ct: torch.Tensor, log_lib: torch.Tensor) -> None:
        """Bayesian generative model for PPI inference.
        
        Parameters
        ----------
        xs : torch.Tensor
            Gene expression counts (batch_size, n_genes)
        ct : torch.Tensor
            Cell type indices (batch_size,)
        log_lib : torch.Tensor
            Log library size (batch_size, 1)
        """
        batch_size = xs.shape[0]
        
        # Gene-specific baseline expression
        with pyro.plate("genes_bias", self.n_genes):
            bias = pyro.sample("bias", dist.Normal(0.0, 1.0))
        
        # Negative binomial dispersion (inverse dispersion)
        with pyro.plate("genes_r", self.n_genes):
            r = pyro.sample("r", dist.Gamma(2.0, 0.1))
        
        # Edge weights with chosen prior
        if self.prior_type == "spike_slab":
            temp = torch.tensor(self.z_temperature, device=self.device)
            # Spike-and-slab prior for edge selection
            with pyro.plate("edges_z", self.n_edges):
                z = pyro.sample("z", dist.RelaxedBernoulli(temp, probs=self.edge_prior_prob))
            
            with pyro.plate("edges_tau", self.n_edges):
                tau = pyro.sample("tau", dist.Normal(0.0, 0.5))
            
            # Cell type-specific modulation
            with pyro.plate("cell_types", self.n_types), pyro.plate("edges_phi", self.n_edges):
                phi = pyro.sample("phi", dist.Normal(0.0, 0.2))
            phi
            
            # Compute edge weights
            phi_batch = phi[:, ct].T  # Shape: (batch_size, n_edges)
            W_edges = z * (tau + phi_batch)  # Spike-and-slab multiplication
            
        else:  # horseshoe
            # Horseshoe prior for edge selection
            tau_global = pyro.sample("tau_global", dist.HalfCauchy(1.0))
            
            with pyro.plate("edges_local", self.n_edges):
                tau_local = pyro.sample("tau_local", dist.HalfCauchy(1.0))
            
            with pyro.plate("edges_w", self.n_edges):
                w = pyro.sample("w", dist.Normal(0.0, 1.0))
            
            # Cell type-specific modulation
            with pyro.plate("cell_types", self.n_types), pyro.plate("edges_phi", self.n_edges):
                phi = pyro.sample("phi", dist.Normal(0.0, 0.2))
            phi = phi.T
            
            # Compute edge weights
            phi_batch = phi[:, ct].T
            base_weight = tau_global * tau_local * w
            W_edges = base_weight + phi_batch
        
        # Apply edge interactions efficiently
        i, j = self.edge_idx
        
        # Numerical stability: clamp expression values
        xs_clamped = torch.clamp(xs, min=0, max=1e4)
        
        # Compute interaction effects
        interaction_effects_i = xs_clamped[:, j] * W_edges  # Gene j -> Gene i
        interaction_effects_j = xs_clamped[:, i] * W_edges  # Gene i -> Gene j
        
        # Accumulate effects
        Wx = torch.zeros_like(xs)
        Wx.scatter_add_(1, i.unsqueeze(0).expand(batch_size, -1), interaction_effects_i)
        Wx.scatter_add_(1, j.unsqueeze(0).expand(batch_size, -1), interaction_effects_j)
        
        # Numerical stability: clamp interaction effects
        Wx = torch.clamp(Wx, min=-10, max=10)
        
        # Compute log rates with stability
        logits = log_lib + bias + Wx
        logits = torch.clamp(logits, min=-20, max=20)  # Prevent overflow
        
        # Prepare r for broadcasting
        r_expanded = r.unsqueeze(0).expand(batch_size, -1)
        
        # Likelihood
        counts = x_batch.round().to(torch.int64)

        # batch‐plate over cells
        with pyro.plate("cells", counts.size(0)):
            pyro.sample("obs",
                    dist.NegativeBinomial(total_count=self.r_edges, logits=self.logits),
                    obs=counts)


    def _build_model(self) -> None:
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
        epochs: int = 400, 
        lr: float | None = None, 
        *, 
        verbose: bool = True,
        log_interval: int = 50,
        patience: int = 50,
        min_delta: float = 1e-4,
    ) -> dict:
        """Fit the model using stochastic variational inference.
        
        Parameters
        ----------
        epochs : int
            Maximum number of training epochs
        lr : float or None
            Learning rate (updates the optimizer if provided)
        verbose : bool
            Whether to print training progress
        log_interval : int
            Epochs between progress updates
        patience : int
            Early stopping patience (epochs without improvement)
        min_delta : float
            Minimum change in loss for early stopping
            
        Returns
        -------
        dict
            Training history with 'loss' key
        """
        if lr is not None:
            self.lr = lr
            self._build_model()
        
        pyro.clear_param_store()
        
        history = {"loss": []}
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            self.z_temperature = max(0.01, self.z_temperature * 0.95)
            for xb, ct, lib in self.loader:
                loss = self.svi.step(xb, ct, lib)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            history["loss"].append(avg_loss)
            
            # Early stopping check
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
                logger.info(f"[Epoch {epoch:04d}] ELBO: {avg_loss:.2f}")
        
        return history

    def export_networks(
        self, 
        threshold: float = 0.9,
        return_cell_type_specific: bool = False,
        include_confidence: bool = True,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Export inferred PPI networks.
        
        Parameters
        ----------
        threshold : float
            Probability threshold for edge inclusion (spike-slab only)
        return_cell_type_specific : bool
            If True, return cell-type specific networks
        include_confidence : bool
            If True, include confidence intervals
        quantiles : list[float] or None
            Quantiles to compute for confidence intervals
            
        Returns
        -------
        pd.DataFrame or dict[str, pd.DataFrame]
            Network edges with probabilities and weights
        """
        import pandas as pd
        
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
        
        # Get posterior quantiles
        q_dict = self.guide.quantiles(quantiles)
        
        if self.prior_type == "spike_slab":
            # Extract edge probabilities
            z_samples = q_dict["z"]  # Shape: (n_quantiles, n_edges)
            edge_probs = z_samples[len(quantiles)//2]  # Median
            
            # Filter by threshold
            keep_mask = edge_probs >= threshold
            
        else:  # horseshoe
            # For horseshoe, use weight magnitude
            tau_global = q_dict["tau_global"][len(quantiles)//2]
            tau_local = q_dict["tau_local"][len(quantiles)//2]
            w = q_dict["w"][len(quantiles)//2]
            
            edge_weights = tau_global * tau_local * w
            weight_magnitude = torch.abs(edge_weights)
            
            # Use threshold as percentile for horseshoe
            threshold_value = torch.quantile(weight_magnitude, threshold)
            keep_mask = weight_magnitude >= threshold_value
        
        if not return_cell_type_specific:
            # Global network
            i_indices = self.edge_idx[0, keep_mask].cpu().numpy()
            j_indices = self.edge_idx[1, keep_mask].cpu().numpy()
            
            result_data = {
                "gene_i": [self.genes[i] for i in i_indices],
                "gene_j": [self.genes[j] for j in j_indices],
            }
            
            if self.prior_type == "spike_slab":
                result_data["probability"] = edge_probs[keep_mask].cpu().numpy()
                
                if include_confidence:
                    result_data["prob_lower"] = z_samples[0][keep_mask].cpu().numpy()
                    result_data["prob_upper"] = z_samples[-1][keep_mask].cpu().numpy()
            
            # Add weights
            tau = q_dict["tau"][len(quantiles)//2] if "tau" in q_dict else edge_weights
            result_data["weight"] = tau[keep_mask].cpu().numpy()
            
            if include_confidence and "tau" in q_dict:
                result_data["weight_lower"] = q_dict["tau"][0][keep_mask].cpu().numpy()
                result_data["weight_upper"] = q_dict["tau"][-1][keep_mask].cpu().numpy()
            
            return pd.DataFrame(result_data)
        
        else:
            # Cell-type specific networks
            networks = {}
            phi = q_dict["phi"][len(quantiles)//2]  # Shape: (n_types, n_edges)
            
            for ct_idx, ct_name in enumerate(self.ctypes):
                # Cell-type specific weights
                if self.prior_type == "spike_slab":
                    ct_weights = edge_probs * (q_dict["tau"][len(quantiles)//2] + phi[ct_idx])
                else:
                    ct_weights = edge_weights + phi[ct_idx]
                
                # Apply same filtering
                ct_keep = keep_mask  # Could add cell-type specific filtering
                
                i_indices = self.edge_idx[0, ct_keep].cpu().numpy()
                j_indices = self.edge_idx[1, ct_keep].cpu().numpy()
                
                ct_data = {
                    "gene_i": [self.genes[i] for i in i_indices],
                    "gene_j": [self.genes[j] for j in j_indices],
                    "weight": ct_weights[ct_keep].cpu().numpy(),
                }
                
                if self.prior_type == "spike_slab":
                    ct_data["probability"] = edge_probs[ct_keep].cpu().numpy()
                
                networks[ct_name] = pd.DataFrame(ct_data)
            
            return networks

    def predict_expression(
        self, 
        cell_types: torch.Tensor | None = None,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """Generate expression predictions from the fitted model.
        
        Parameters
        ----------
        cell_types : torch.Tensor or None
            Cell type indices for prediction
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        torch.Tensor
            Predicted expression matrix
        """
        if cell_types is None:
            # Sample random cell types
            cell_types = torch.randint(0, self.n_types, (n_samples,), device=self.device)
        
        # Use the guide to sample from posterior
        with torch.no_grad():
            # Sample library sizes from training data
            lib_sizes = torch.exp(self.log_lib[torch.randint(0, len(self.log_lib), (n_samples, 1))])
            
            # Generate predictions
            trace = pyro.poutine.trace(self.guide).get_trace(
                torch.zeros(n_samples, self.n_genes, device=self.device),
                cell_types,
                lib_sizes.log()
            )
            
            # Run model with sampled parameters
            conditioned_model = pyro.poutine.replay(self._model, trace)
            
            with pyro.plate("prediction", n_samples):
                trace = pyro.poutine.trace(conditioned_model).get_trace(
                    torch.zeros(n_samples, self.n_genes, device=self.device),
                    cell_types,
                    lib_sizes.log()
                )
            
            return trace.nodes["obs"]["value"]


def export_networks(model: MBModel, threshold: float = 0.9, **kwargs):
    """Export networks from a fitted model.
    
    Parameters
    ----------
    model : MBModel
        Fitted model instance
    threshold : float
        Probability threshold for edge inclusion
    **kwargs
        Additional arguments passed to model.export_networks()
        
    Returns
    -------
    pd.DataFrame or dict
        Exported networks
    """
    return model.export_networks(threshold, **kwargs)
