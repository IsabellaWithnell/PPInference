
import numpy as np
import anndata as ad
import torch
import pyro

from mini_bayes_ppi import MBModel


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
