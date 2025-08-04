from __future__ import annotations

from collections.abc import Iterable, Sequence

import pyro
import pyro.distributions as dist
import torch
from anndata import AnnData
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["MBModel", "export_networks"]


def _edge_index(n_genes: int, edges: Sequence[tuple[int, int]]) -> torch.Tensor:
    """Return a symmetric edge index tensor for the given gene pairs."""
    if not edges:
        raise ValueError("`prior_edges` is empty; nothing to learn.")
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
        lr: float = 5e-3,
        device: str | None = None,
    ) -> None:
        """Initialize the model and prepare tensors for training."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.adata = adata
        self.batch_size = batch_size
        self.lr = lr

        self.genes: list[str] = list(adata.var_names)
        self.n_genes = len(self.genes)

        self.ctypes: list[str] = list(adata.obs[cell_key].astype(str).unique())
        self.n_types = len(self.ctypes)
        ct_lookup: dict[str, int] = {c: i for i, c in enumerate(self.ctypes)}
        
        # Fix 1: Proper indentation and device placement
        self.ct_idx = torch.tensor(
            [ct_lookup[c] for c in adata.obs[cell_key].astype(str)],
            dtype=torch.long,
            device=self.device
        )

        self.edge_idx = _edge_index(self.n_genes, list(prior_edges)).to(self.device)
        self.n_edges = self.edge_idx.shape[1]

        x = torch.tensor(adata.X.A if hasattr(adata.X, "A") else adata.X, dtype=torch.float32)
        self.X = x.to(self.device)  # Also move X to device

        libsize = torch.sum(self.X, dim=1, keepdim=True) + 1.0
        self.log_lib = libsize.log()

        self.loader = DataLoader(
            TensorDataset(self.X, self.ct_idx, self.log_lib),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        self._build_model()

    # Fix 2: Move _model method inside the class with proper indentation
    def _model(self, xs: torch.Tensor, ct: torch.Tensor, log_lib: torch.Tensor) -> None:
        """Complete Bayesian generative model for PPI inference.
        
        Args:
            xs: Gene expression counts (batch_size, n_genes)
            ct: Cell type indices (batch_size,)
            log_lib: Library size normalization (batch_size, 1)
        """
        
        # Global edge interaction probabilities (do interactions exist?)
        with pyro.plate("edges", self.n_edges):
            theta = pyro.sample("theta", dist.Beta(1.0, 9.0))  # ~10% prior interaction prob
        
        # Global edge interaction strengths (how strong are interactions?)
        with pyro.plate("edges_tau", self.n_edges):
            tau = pyro.sample("tau", dist.Normal(0.0, 0.5))  # Moderate interaction strengths
        
        # Cell type-specific modulation of interactions
        with pyro.plate("cell_types", self.n_types):
            with pyro.plate("edges_phi", self.n_edges):
                phi = pyro.sample("phi", dist.Normal(0.0, 0.2))  # Slightly larger cell-type effects
        phi = phi.T

        # Gene-specific baseline expression
        with pyro.plate("genes_bias", self.n_genes):
            bias = pyro.sample("bias", dist.Normal(0.0, 1.0))
        
        # Negative binomial dispersion
        with pyro.plate("genes_r", self.n_genes):
            r = pyro.sample("r", dist.Gamma(2.0, 0.1))
        r = r.unsqueeze(0).expand(xs.shape[0], -1)

        # Cell type-specific interaction weights
        phi_batch = phi[ct]  # Shape: (batch_size, n_edges)
        W_edges = theta * (tau + phi_batch)  # Shape: (batch_size, n_edges)
        
        # Apply edge interactions (vectorized for speed)
        i, j = self.edge_idx
        
        # Compute interaction effects for all edges at once
        interaction_effects_i = xs[:, j] * W_edges  # Gene j -> Gene i
        interaction_effects_j = xs[:, i] * W_edges  # Gene i -> Gene j
        
        # Accumulate effects using scatter_add for efficiency
        Wx = torch.zeros_like(xs)
        Wx.scatter_add_(1, i.unsqueeze(0).expand(xs.shape[0], -1), interaction_effects_i)
        Wx.scatter_add_(1, j.unsqueeze(0).expand(xs.shape[0], -1), interaction_effects_j)
        
        # Optional: Add numerical stability
        Wx = torch.clamp(Wx, min=-10, max=10)  # Prevent extreme values
        
        # Compute log rates
        logits = log_lib + bias + Wx
        
        # Likelihood
        pyro.sample(
            "obs",
            dist.NegativeBinomial(total_count=r, logits=logits).to_event(1),
            obs=xs,
        )

    def _build_model(self) -> None:
        self.guide = AutoNormal(self._model)
        self.svi = SVI(self._model, self.guide, pyro.optim.Adam({"lr": self.lr}), Trace_ELBO())

    def train(self, max_epochs: int = 400, *, verbose: bool = True) -> None:
        """Run stochastic variational inference for a number of epochs."""
        pyro.clear_param_store()
        for epoch in range(max_epochs):
            total_loss = 0.0
            for xb, ct, lib in self.loader:
                xb, ct, lib = xb.to(self.device), ct.to(self.device), lib.to(self.device)
                total_loss += self.svi.step(xb, ct, lib)
            if verbose and epoch % 50 == 0:
                print(f"[{epoch:03d}] ELBO ≈ {total_loss:.2f}")

    def fit(self, epochs: int = 400, lr: float | None = None, *, verbose: bool = True) -> None:
        """Fit wrapper that matches the public README/API."""
        if lr is not None:
            self.lr = lr
            self._build_model()
        self.train(max_epochs=epochs, verbose=verbose)

    def export_networks(self, threshold: float = 0.9):
        """Return a DataFrame of inferred PPIs above the given probability."""
        cut = threshold
        import pandas as pd

        qdict = self.guide.quantiles([0.5])
        probs = qdict["theta"].squeeze()
        weights = qdict["tau"].squeeze()

        keep = probs >= cut
        i, j = self.edge_idx[:, keep]
        df = pd.DataFrame(
            {
                "gene_i": [self.genes[k] for k in i.tolist()],
                "gene_j": [self.genes[k] for k in j.tolist()],
                "prob": probs[keep].detach().cpu().numpy(),
                "weight": weights[keep].detach().cpu().numpy(),
            }
        )
        return df


def export_networks(model: MBModel, threshold: float = 0.9):
    """Helper to call :meth:`MBModel.export_networks` on a model instance."""
    return model.export_networks(threshold)
