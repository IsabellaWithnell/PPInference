
import anndata as ad
import numpy as np
import pandas as pd
import torch

from mini_bayes_ppi import MBModel, load_string_prior
from mini_bayes_ppi.core import _edge_index


def test_elbo_decreases():
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(60, 30)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.obs["cell_type"] = ["A"] * 30 + ["B"] * 30
    adata.var_names = [f"g{i}" for i in range(30)]

    prior = [(0, 1), (2, 3)]
    model = MBModel(adata, prior_edges=prior, batch_size=16, device="cpu")

    loss0 = model.svi.evaluate_loss(model.X[:32], model.ct_idx[:32], model.log_lib[:32])
    model.train(max_epochs=5, verbose=False)
    loss1 = model.svi.evaluate_loss(model.X[:32], model.ct_idx[:32], model.log_lib[:32])

    assert loss1 < loss0, "ELBO did not improve"
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
