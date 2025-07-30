
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import pyro
import pyro.distributions as dist
import torch
from anndata import AnnData
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["MBModel", "export_networks"]


def _edge_index(n_genes: int, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
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
        prior_edges: Iterable[Tuple[int, int]],
        *,
        cell_key: str = "cell_type",
        batch_size: int = 1024,
        lr: float = 5e-3,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.adata = adata
        self.batch_size = batch_size
        self.lr = lr

        self.genes: List[str] = list(adata.var_names)
        self.n_genes = len(self.genes)

        self.ctypes: List[str] = list(adata.obs[cell_key].astype(str).unique())
        self.n_types = len(self.ctypes)
        ct_lookup: Dict[str, int] = {c: i for i, c in enumerate(self.ctypes)}
        self.ct_idx = torch.tensor(
            [ct_lookup[c] for c in adata.obs[cell_key].astype(str)],
            dtype=torch.long,
            
        )

        self.edge_idx = _edge_index(self.n_genes, list(prior_edges)).to(self.device)
        self.n_edges = self.edge_idx.shape[1]

        x = torch.tensor(adata.X.A if hasattr(adata.X, "A") else adata.X, dtype=torch.float32)
        self.X = x

        libsize = torch.sum(self.X, dim=1, keepdim=True) + 1.0
        self.log_lib = libsize.log()

        self.loader = DataLoader(
            TensorDataset(self.X, self.ct_idx, self.log_lib),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        self._build_model()

    # -------------------- model --------------------
    def _model(self, xs: torch.Tensor, ct: torch.Tensor, log_lib: torch.Tensor) -> None:
        temperature = pyro.sample("temp", dist.LogNormal(-1.0, 0.3))
        pi = pyro.sample("pi", dist.Beta(1.0, 9.0))
        sigma_tau = pyro.sample("sigma_tau", dist.HalfCauchy(1.0))
        sigma_phi = pyro.sample("sigma_phi", dist.HalfCauchy(0.5))

        with pyro.plate("edges", self.n_edges):
            theta = pyro.sample(
                "theta",
                dist.RelaxedBernoulliStraightThrough(
                    temperature, probs=pi.expand([self.n_edges])
                ),
            )
            tau = pyro.sample("tau", dist.Normal(0.0, sigma_tau))

        with pyro.plate("cell_type", self.n_types):
            phi = pyro.sample(
                "phi",
                dist.Normal(0.0, sigma_phi).expand([self.n_edges]).to_event(1),
            )

        bias = pyro.sample(
            "bias", dist.Normal(0.0, 1.0).expand([self.n_genes]).to_event(1)
        )
        with pyro.plate("genes", self.n_genes):
            r = pyro.sample("r", dist.Gamma(2.0, 0.1))
        r = r.unsqueeze(0).expand(xs.shape[0], -1)

        phi_batch = phi[ct]
        W_edges = theta * tau + phi_batch

        i, j = self.edge_idx
        Wx = torch.zeros_like(xs)
        Wx[:, i] += xs[:, j] * W_edges
        Wx[:, j] += xs[:, i] * W_edges

        logits = log_lib + bias + Wx

        pyro.sample(
            "obs",
            dist.NegativeBinomial(total_count=r, logits=logits).to_event(1),
            obs=xs,
        )

    def _build_model(self) -> None:
        self.guide = AutoNormal(self._model)
        self.svi = SVI(self._model, self.guide, pyro.optim.Adam({"lr": self.lr}), Trace_ELBO())

    def train(self, max_epochs: int = 400, *, verbose: bool = True) -> None:
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
        cut = threshold
        import pandas as pd

        qdict = self.guide.quantiles([0.5])
        probs = qdict["theta"].squeeze()
        weights = qdict["tau"].squeeze()

        keep = probs >= cut
        i, j = self.edge_idx[:, keep]
        df = pd.DataFrame(
            dict(
                gene_i=[self.genes[k] for k in i.tolist()],
                gene_j=[self.genes[k] for k in j.tolist()],
                prob=probs[keep].detach().cpu().numpy(),
                weight=weights[keep].detach().cpu().numpy(),
            )
        )
        return df


def export_networks(model: MBModel, threshold: float = 0.9):
    return model.export_networks(threshold)