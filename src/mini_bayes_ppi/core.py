from __future__ import annotations

import logging
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
        prior_type: str = "spike_slab",
        edge_prior_prob: float = 0.5,
        coverage_weight_type: str = "adaptive",  # New parameter
        min_coverage: float = 1.0,  # New parameter
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
        coverage_weight_type : str
            'adaptive': Weight by gene coverage
            'binary': Threshold-based inclusion
            'none': No coverage weighting
        min_coverage : float
            Minimum mean expression for gene inclusion
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
        self.z_temperature = temp
        self.coverage_weight_type = coverage_weight_type
        self.min_coverage = min_coverage

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
        X_arr = adata.X.toarray() if spsp.issparse(adata.X) else adata.X
        x = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
        self.X = x.to(self.device)

        # Library size normalization
        libsize = torch.sum(self.X, dim=1, keepdim=True) + 1.0
        self.log_lib = libsize.log()
        
        # Calculate gene-level statistics for coverage weighting
        self._calculate_gene_stats()

        # Create data loader
        dataset = TensorDataset(self.X, self.ct_idx, self.log_lib)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
            num_workers=0,
        )

        self._build_model()
    
    def _calculate_gene_stats(self):
        """Calculate coverage and reliability metrics per gene."""
        X_cpu = self.X.cpu().numpy()
        
        # Mean expression (coverage proxy)
        self.gene_mean = torch.tensor(
            np.mean(X_cpu, axis=0), 
            device=self.device, 
            dtype=torch.float32
        )
        
        # Fraction of zero expression (dropout rate)
        self.gene_dropout = torch.tensor(
            np.mean(X_cpu == 0, axis=0),
            device=self.device,
            dtype=torch.float32
        )
        
        # Coefficient of variation (noise level)
        gene_std = np.std(X_cpu, axis=0)
        gene_cv = np.divide(
            gene_std, 
            self.gene_mean.cpu().numpy() + 1e-6,
            out=np.ones_like(gene_std),
            where=self.gene_mean.cpu().numpy() > 0
        )
        self.gene_cv = torch.tensor(gene_cv, device=self.device, dtype=torch.float32)
        
        # Composite reliability score (higher = more reliable)
        self.gene_reliability = self._compute_reliability_score()
        
        # Log statistics
        logger.info(f"Gene coverage stats:")
        logger.info(f"  Mean expression: {self.gene_mean.mean():.2f} (±{self.gene_mean.std():.2f})")
        logger.info(f"  Mean dropout: {self.gene_dropout.mean():.2%}")
        logger.info(f"  Genes below min_coverage: {(self.gene_mean < self.min_coverage).sum()}")
    
    def _compute_reliability_score(self):
        """Compute composite reliability score for each gene."""
        # Normalize each metric to [0, 1]
        coverage_score = torch.log1p(self.gene_mean) / (torch.log1p(self.gene_mean.max()) + 1e-6)
        dropout_penalty = 1 - self.gene_dropout
        cv_penalty = 1 / (1 + self.gene_cv)
        
        # Weighted combination
        reliability = (
            0.5 * coverage_score +
            0.3 * dropout_penalty + 
            0.2 * cv_penalty
        )
        
        return reliability

    def _model(self, xs: torch.Tensor, ct: torch.Tensor, log_lib: torch.Tensor) -> None:
        """Bayesian generative model for PPI inference with coverage weighting."""
        batch_size = xs.shape[0]
        
        # Gene-specific baseline expression
        with pyro.plate("genes_bias", self.n_genes):
            bias = pyro.sample("bias", dist.Normal(0.0, 1.0))
        
        # Coverage-adjusted dispersion
        with pyro.plate("genes_r", self.n_genes):
            if self.coverage_weight_type == "adaptive":
                # Scale dispersion by reliability
                r_scale = 0.5 + 1.5 * self.gene_reliability
                r = pyro.sample("r", dist.Gamma(2.0 * r_scale, 0.1))
            else:
                r = pyro.sample("r", dist.Gamma(2.0, 0.1))
        
        # Edge interactions
        with pyro.plate("cell_types", self.n_types), pyro.plate("edges_phi", self.n_edges):
            phi = pyro.sample("phi", dist.Normal(0.0, 0.2))

        phi_batch = phi[:, ct].T

        if self.prior_type == "spike_slab":
            temp = torch.tensor(self.z_temperature, device=self.device)
            
            # Coverage-weighted edge probabilities
            if self.coverage_weight_type == "adaptive":
                i, j = self.edge_idx
                edge_coverage_weight = torch.sqrt(
                    self.gene_reliability[i] * self.gene_reliability[j]
                )
                edge_prior = self.edge_prior_prob * (0.5 + 0.5 * edge_coverage_weight)
            else:
                edge_prior = self.edge_prior_prob
            
            with pyro.plate("edges_z", self.n_edges):
                z = pyro.sample("z", dist.RelaxedBernoulli(temp, probs=edge_prior))
            
            with pyro.plate("edges_tau", self.n_edges):
                if self.coverage_weight_type == "adaptive":
                    tau_scale = 0.5 * (0.5 + 0.5 * edge_coverage_weight)
                else:
                    tau_scale = 0.5
                tau = pyro.sample("tau", dist.Normal(0.0, tau_scale))
            
            W_edges = z * (tau + phi_batch)
        else:  # horseshoe
            tau_global = pyro.sample("tau_global", dist.HalfCauchy(1.0))
            
            with pyro.plate("edges_local", self.n_edges):
                if self.coverage_weight_type == "adaptive":
                    i, j = self.edge_idx
                    edge_coverage_weight = torch.sqrt(
                        self.gene_reliability[i] * self.gene_reliability[j]
                    )
                    tau_local = pyro.sample(
                        "tau_local", 
                        dist.HalfCauchy(1.0 / (0.5 + 0.5 * edge_coverage_weight))
                    )
                else:
                    tau_local = pyro.sample("tau_local", dist.HalfCauchy(1.0))
            
            with pyro.plate("edges_w", self.n_edges):
                w = pyro.sample("w", dist.Normal(0.0, 1.0))
            
            base_weight = tau_global * tau_local * w
            W_edges = base_weight + phi_batch
        
        # Apply edge interactions
        i, j = self.edge_idx
        xs_clamped = torch.clamp(xs, min=0, max=1e4)
        
        if self.coverage_weight_type == "binary":
            # Mask out low-coverage genes
            coverage_mask = (self.gene_mean > self.min_coverage).float()
            xs_masked = xs_clamped * coverage_mask.unsqueeze(0)
        else:
            xs_masked = xs_clamped
        
        # Compute interaction effects
        interaction_effects_i = xs_masked[:, j] * W_edges
        interaction_effects_j = xs_masked[:, i] * W_edges
        
        # Accumulate effects
        Wx = torch.zeros_like(xs)
        Wx.scatter_add_(1, i.unsqueeze(0).expand(batch_size, -1), interaction_effects_i)
        Wx.scatter_add_(1, j.unsqueeze(0).expand(batch_size, -1), interaction_effects_j)
        
        # Numerical stability
        Wx = torch.clamp(Wx, min=-10, max=10)
        logits = log_lib + bias + Wx
        logits = torch.clamp(logits, min=-20, max=20)
        
        # Likelihood with coverage-based confidence
        counts = xs.round().to(torch.int64)
        
        if self.coverage_weight_type == "adaptive":
            # Scale dispersion by reliability
            with pyro.plate("cells", counts.size(0)):
                pyro.sample(
                    "obs",
                    dist.NegativeBinomial(
                        total_count=r * self.gene_reliability,
                        logits=logits
                    ).to_event(1),
                    obs=counts,
                )
        else:
            with pyro.plate("cells", counts.size(0)):
                pyro.sample(
                    "obs",
                    dist.NegativeBinomial(total_count=r, logits=logits).to_event(1),
                    obs=counts,
                )

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
        """Fit the model using stochastic variational inference."""
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
        threshold: float = 0.5,
        return_cell_type_specific: bool = False,
        include_confidence: bool = True,
        include_reliability: bool = True,  # New parameter
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Export inferred PPI networks with reliability scores."""
        import pandas as pd
        
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
        
        # Get posterior quantiles
        q_dict = self.guide.quantiles(quantiles)
        
        if self.prior_type == "spike_slab":
            # Extract edge probabilities
            z_samples = q_dict["z"]
            edge_probs = z_samples[len(quantiles)//2]
            
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
            
            # Add reliability scores
            if include_reliability and self.coverage_weight_type != "none":
                gene_i_rel = [float(self.gene_reliability[i]) for i in i_indices]
                gene_j_rel = [float(self.gene_reliability[j]) for j in j_indices]
                result_data["gene_i_reliability"] = gene_i_rel
                result_data["gene_j_reliability"] = gene_j_rel
                result_data["edge_reliability"] = np.sqrt(
                    np.array(gene_i_rel) * np.array(gene_j_rel)
                )
            
            return pd.DataFrame(result_data)
        
        else:
            # Cell-type specific networks
            networks = {}
            phi = q_dict["phi"][len(quantiles)//2]
            
            for ct_idx, ct_name in enumerate(self.ctypes):
                phi_ct = phi[:, ct_idx]
                if self.prior_type == "spike_slab":
                    tau_med = q_dict["tau"][len(quantiles)//2]
                    ct_weights = edge_probs * (tau_med + phi_ct)
                else:
                    ct_weights = edge_weights + phi_ct
                
                ct_keep = keep_mask
                
                i_indices = self.edge_idx[0, ct_keep].cpu().numpy()
                j_indices = self.edge_idx[1, ct_keep].cpu().numpy()
                
                ct_data = {
                    "gene_i": [self.genes[i] for i in i_indices],
                    "gene_j": [self.genes[j] for j in j_indices],
                    "weight": ct_weights[ct_keep].cpu().numpy(),
                }
                
                if self.prior_type == "spike_slab":
                    ct_data["probability"] = edge_probs[ct_keep].cpu().numpy()
                
                # Add reliability scores
                if include_reliability and self.coverage_weight_type != "none":
                    gene_i_rel = [float(self.gene_reliability[i]) for i in i_indices]
                    gene_j_rel = [float(self.gene_reliability[j]) for j in j_indices]
                    ct_data["gene_i_reliability"] = gene_i_rel
                    ct_data["gene_j_reliability"] = gene_j_rel
                    ct_data["edge_reliability"] = np.sqrt(
                        np.array(gene_i_rel) * np.array(gene_j_rel)
                    )
                
                networks[ct_name] = pd.DataFrame(ct_data)
            
            return networks

    def predict_expression(
        self, 
        cell_types: torch.Tensor | None = None,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """Generate expression predictions from the fitted model."""
        if cell_types is None:
            cell_types = torch.randint(0, self.n_types, (n_samples,), device=self.device)
        
        with torch.no_grad():
            lib_sizes = torch.exp(self.log_lib[torch.randint(0, len(self.log_lib), (n_samples, 1))])
            
            trace = pyro.poutine.trace(self.guide).get_trace(
                torch.zeros(n_samples, self.n_genes, device=self.device),
                cell_types,
                lib_sizes.log()
            )
            
            conditioned_model = pyro.poutine.replay(self._model, trace)
            
            with pyro.plate("prediction", n_samples):
                trace = pyro.poutine.trace(conditioned_model).get_trace(
                    torch.zeros(n_samples, self.n_genes, device=self.device),
                    cell_types,
                    lib_sizes.log()
                )
            
            return trace.nodes["obs"]["value"]
    
    def get_gene_stats(self) -> pd.DataFrame:
        """Return gene-level statistics as a DataFrame."""
        import pandas as pd
        
        return pd.DataFrame({
            "gene": self.genes,
            "mean_expression": self.gene_mean.cpu().numpy(),
            "dropout_rate": self.gene_dropout.cpu().numpy(),
            "cv": self.gene_cv.cpu().numpy(),
            "reliability": self.gene_reliability.cpu().numpy(),
        })


def export_networks(model: MBModel, threshold: float = 0.5, **kwargs):
    """Export networks from a fitted model."""
    return model.export_networks(threshold, **kwargs)or dict
        Exported networks
    """
    return model.export_networks(threshold, **kwargs)
