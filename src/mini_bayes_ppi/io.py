from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def load_string_prior(
    tsv_or_taxid: str | int | list[str | tuple[str, str]],
    *,
    adata_var_names: Iterable[str],
    score_cutoff: int = 700,
) -> list[tuple[int, int]]:
    """Load STRING prior edges from a TSV file or explicit list."""
    
    if isinstance(tsv_or_taxid, list):
        # Treat as an explicit list of edges like ["GENE1 GENE2", ...] or [(gene1, gene2), ...]
        edges: list[tuple[int, int]] = []
        name_to_ix = {g: i for i, g in enumerate(adata_var_names)}
        for pair in tsv_or_taxid:
            if isinstance(pair, str):
                parts = pair.strip().split()
                if len(parts) != 2:
                    raise ValueError(f"Expected 'GENEA GENEB' format, got {pair!r}")
                a, b = parts
            elif isinstance(pair, (tuple, list)) and len(pair) == 2:
                a, b = pair
            else:
                raise ValueError(f'Edge specification {pair!r} is neither string nor 2‑tuple.')
            if a in name_to_ix and b in name_to_ix:
                edges.append((name_to_ix[a], name_to_ix[b]))
        return edges
    if isinstance(tsv_or_taxid, int):
        raise NotImplementedError("Automatic STRING download not implemented yet.")

    path = Path(tsv_or_taxid).expanduser()
    df = pd.read_csv(path, sep="\t")
    df = df[df["combined_score"] >= score_cutoff]

    name_to_ix = {g: i for i, g in enumerate(adata_var_names)}
    edges = [
        (name_to_ix[a], name_to_ix[b])
        for a, b in zip(df["protein1"], df["protein2"])




import numpy as np
import anndata as ad
import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch
import pyro


from mini_bayes_ppi import MBModel
from mini_bayes_ppi import MBModel, load_string_prior
from mini_bayes_ppi.core import _edge_index




def test_elbo_decreases():
def test_elbo_decreases():
    rng = np.random.default_rng(0)
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(60, 30)).astype(np.float32)
    X = rng.poisson(1.0, size=(60, 30)).astype(np.float32)
    adata = ad.AnnData(X)
    adata = ad.AnnData(X)
    adata.obs["cell_type"] = ["A"] * 30 + ["B"] * 30
    adata.obs["cell_type"] = ["A"] * 30 + ["B"] * 30
    adata.var_names = [f"g{i}" for i in range(30)]
    adata.var_names = [f"g{i}" for i in range(30)]


    prior = [(0, 1), (2, 3)]
    prior = [(0, 1), (2, 3)]
    model = MBModel(adata, prior_edges=prior, batch_size=16, device="cpu")
    model = MBModel(adata, prior_edges=prior, batch_size=16, device="cpu")


    loss0 = model.svi.evaluate_loss(model.X[:32], model.ct_idx[:32], model.log_lib[:32])
    loss0 = model.svi.evaluate_loss(model.X[:32], model.ct_idx[:32], model.log_lib[:32])
    model.train(max_epochs=5, verbose=False)
    model.train(max_epochs=5, verbose=False)
    loss1 = model.svi.evaluate_loss(model.X[:32], model.ct_idx[:32], model.log_lib[:32])
    loss1 = model.svi.evaluate_loss(model.X[:32], model.ct_idx[:32], model.log_lib[:32])


    assert loss1 < loss0, "ELBO did not improve"
    assert loss1 < loss0, "ELBO did not improve"
    assert not torch.isnan(torch.tensor(loss1)), "ELBO NaN detected"
    assert not torch.isnan(torch.tensor(loss1)), "ELBO NaN detected"


def test_load_string_prior_list():
    names = ["A", "B", "C", "D"]
    edges = load_string_prior(["A B", ("C", "D")], adata_var_names=names)
    assert edges == [(0, 1), (2, 3)]


def test_load_string_prior_tsv(tmp_path):
    df = pd.DataFrame(
        {
            "protein1": ["A", "C"],
            "protein2": ["B", "D"],
            "combined_score": [800, 400],
        }
    )
    path = tmp_path / "prior.tsv"
    df.to_csv(path, sep="\t", index=False)
    edges = load_string_prior(str(path), adata_var_names=["A", "B", "C", "D"], score_cutoff=500)
    assert edges == [(0, 1)]


def test_edge_index_symmetry():
    idx = _edge_index(3, [(0, 1)])
    expected = torch.tensor([[0, 1], [1, 0]])
    assert torch.equal(idx, expected)
