# mini_bayes_ppi

Bayesian **P**rotein–**P**rotein **I**nteraction inference from single‑cell RNA‑seq counts.

[![PyPI](https://img.shields.io/pypi/v/mini_bayes_ppi.svg)](https://pypi.org/project/mini_bayes_ppi/)
[![CI](https://github.com/IsabellaWithnell/mini_bayes_ppi/actions/workflows/ci.yml/badge.svg)](https://github.com/IsabellaWithnell/mini_bayes_ppi/actions)

---

## Installation
```bash
pip install mini_bayes_ppi

# or, with GPU support
# Ensure your CUDA toolkit matches the PyTorch wheel version, then:
pip install mini_bayes_ppi --extra-index-url https://download.pytorch.org/whl/cu118
```

## Quickstart
```python
import anndata as ad
from mini_bayes_ppi import MBModel, load_string_prior

adata = ad.read_h5ad("pbmc3k.h5ad")
prior = load_string_prior(["CD3D CD3E", "LYZ CTSB", "GNLY PRF1"])

mb = MBModel(adata, prior_edges=prior)
mb.fit(epochs=2000, lr=1e-2)

ppi = mb.export_networks(threshold=0.5)
print(ppi.head())
```

## Citation
If you use **mini_bayes_ppi** in your research, please cite:
> *Citation coming soon.*

---

## License
MIT © Isabella Withnell, 2025