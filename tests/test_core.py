import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from mini_bayes_ppi import MBModel, export_networks, load_string_prior
from mini_bayes_ppi.core import _edge_index


class TestEdgeIndex:
    """Test edge index creation and validation."""
    
    def test_edge_index_symmetry(self):
        """Test that edge indices are properly symmetrized."""
        idx = _edge_index(3, [(0, 1)])
        expected = torch.tensor([[0, 1], [1, 0]])
        assert torch.equal(idx, expected)
    
    def test_edge_index_uniqueness(self):
        """Test that duplicate edges are removed."""
        idx = _edge_index(3, [(0, 1), (1, 0), (0, 1)])
        assert idx.shape[1] == 2  # Only unique edges
    
    def test_edge_index_validation(self):
        """Test edge validation."""
        with pytest.raises(ValueError, match="invalid gene indices"):
            _edge_index(3, [(0, 5)])
        
        with pytest.raises(ValueError, match="Self-loop detected"):
            _edge_index(3, [(1, 1)])
        
        with pytest.raises(ValueError, match="empty"):
            _edge_index(3, [])


class TestDataLoading:
    """Test data loading and preparation."""
    
    @pytest.fixture
    def simple_adata(self):
        """Create a simple test dataset."""
        rng = np.random.default_rng(42)
        X = rng.poisson(5.0, size=(100, 20)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = ["TypeA"] * 50 + ["TypeB"] * 50
        adata.var_names = [f"Gene_{i}" for i in range(20)]
        return adata
    
    def test_model_initialization(self, simple_adata):
        """Test basic model initialization."""
        prior = [(0, 1), (2, 3), (4, 5)]
        model = MBModel(simple_adata, prior, device="cpu")
        
        assert model.n_genes == 20
        assert model.n_types == 2
        assert model.n_edges == 6  # 3 edges * 2 (bidirectional)
        assert model.n_cells == 100
    
    def test_invalid_cell_key(self, simple_adata):
        """Test error on invalid cell type key."""
        with pytest.raises(ValueError, match="not found in adata.obs"):
            MBModel(simple_adata, [(0, 1)], cell_key="invalid_key")
    
    def test_sparse_matrix_handling(self):
        """Test handling of sparse matrices."""
        from scipy.sparse import csr_matrix
        
        X_dense = np.random.poisson(2.0, size=(50, 10)).astype(np.float32)
        X_sparse = csr_matrix(X_dense)
        
        adata = ad.AnnData(X_sparse)
        adata.obs["cell_type"] = ["A"] * 25 + ["B"] * 25
        adata.var_names = [f"G{i}" for i in range(10)]
        
        model = MBModel(adata, [(0, 1)], device="cpu")
        assert isinstance(model.X, torch.Tensor)
        assert torch.allclose(model.X.cpu(), torch.tensor(X_dense))


class TestPriorLoading:
    """Test prior edge loading functionality."""
    
    def test_load_string_prior_list(self):
        """Test loading edges from list."""
        names = ["A", "B", "C", "D"]
        
        # Test string format
        edges = load_string_prior(["A B", "C D"], adata_var_names=names)
        assert edges == [(0, 1), (2, 3)]
        
        # Test tuple format
        edges = load_string_prior([("A", "B"), ("C", "D")], adata_var_names=names)
        assert edges == [(0, 1), (2, 3)]
        
        # Test mixed format
        edges = load_string_prior(["A B", ("C", "D")], adata_var_names=names)
        assert edges == [(0, 1), (2, 3)]
    
    def test_load_string_prior_filtering(self):
        """Test that non-existent genes are filtered."""
        names = ["A", "B", "C"]
        edges = load_string_prior(["A B", "C D", "E F"], adata_var_names=names)
        assert edges == [(0, 1)]  # Only A-B exists
    
    def test_load_string_prior_tsv(self, tmp_path):
        """Test loading from TSV file."""
        df = pd.DataFrame({
            "protein1": ["A", "B", "C", "D"],
            "protein2": ["B", "C", "D", "E"],
            "combined_score": [900, 800, 400, 950]
        })
        
        path = tmp_path / "string.tsv"
        df.to_csv(path, sep="\t", index=False)
        
        edges = load_string_prior(
            str(path), 
            adata_var_names=["A", "B", "C", "D", "E"],
            score_cutoff=700
        )
        assert len(edges) == 3  # A-B, B-C, D-E
    
    def test_invalid_edge_format(self):
        """Test error handling for invalid edge formats."""
        with pytest.raises(ValueError, match="Expected 'GENEA GENEB' format"):
            load_string_prior(["A B C"], adata_var_names=["A", "B", "C"])
        
        with pytest.raises(ValueError, match="neither string nor 2-tuple"):
            load_string_prior([123], adata_var_names=["A", "B"])


class TestModelTraining:
    """Test model training and inference."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and minimally train a model."""
        rng = np.random.default_rng(42)
        # Create data with known structure
        n_cells, n_genes = 200, 10
        X = np.zeros((n_cells, n_genes))
        
        # Add some structure: genes 0-1 and 2-3 are correlated
        base_expr = rng.poisson(5.0, size=(n_cells, 1))
        X[:, 0] = rng.poisson(base_expr[:, 0] * 1.5)
        X[:, 1] = rng.poisson(base_expr[:, 0] * 1.2)
        
        base_expr2 = rng.poisson(3.0, size=(n_cells, 1))
        X[:, 2] = rng.poisson(base_expr2[:, 0] * 1.3)
        X[:, 3] = rng.poisson(base_expr2[:, 0] * 1.1)
        
        # Add noise to other genes
        X[:, 4:] = rng.poisson(2.0, size=(n_cells, n_genes - 4))
        
        adata = ad.AnnData(X.astype(np.float32))
        adata.obs["cell_type"] = ["TypeA"] * 100 + ["TypeB"] * 100
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        
        # Prior includes true edges and false edges
        prior = [(0, 1), (2, 3), (4, 5), (6, 7)]
        
        model = MBModel(adata, prior, device="cpu", batch_size=50)
        history = model.fit(epochs=50, verbose=False)
        
        return model, history
    
    def test_training_convergence(self, trained_model):
        """Test that training reduces loss."""
        model, history = trained_model
        losses = history["loss"]
        
        assert len(losses) > 0
        assert losses[-1] < losses[0]  # Loss decreased
        assert all(not np.isnan(loss) for loss in losses)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        rng = np.random.default_rng(42)
        X = rng.poisson(3.0, size=(50, 10)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = ["A"] * 50
        adata.var_names = [f"G{i}" for i in range(10)]
        
        model = MBModel(adata, [(0, 1)], device="cpu")
        history = model.fit(
            epochs=1000, 
            patience=5, 
            min_delta=1e-6,
            verbose=False
        )
        
        # Should stop early
        assert len(history["loss"]) < 1000
    
    def test_different_priors(self):
        """Test both spike-slab and horseshoe priors."""
        rng = np.random.default_rng(42)
        X = rng.poisson(3.0, size=(100, 10)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = ["A"] * 50 + ["B"] * 50
        adata.var_names = [f"G{i}" for i in range(10)]
        
        for prior_type in ["spike_slab", "horseshoe"]:
            model = MBModel(
                adata, 
                [(0, 1), (2, 3)], 
                device="cpu",
                prior_type=prior_type
            )
            history = model.fit(epochs=10, verbose=False)
            assert len(history["loss"]) > 0


class TestNetworkExport:
    """Test network export functionality."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create a simple fitted model."""
        rng = np.random.default_rng(42)
        X = rng.poisson(3.0, size=(100, 6)).astype(np.float32)
        
        # Add correlation structure
        X[:, 1] = X[:, 0] * 1.5 + rng.poisson(1.0, 100)
        X[:, 3] = X[:, 2] * 1.2 + rng.poisson(1.0, 100)
        
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = ["TypeA"] * 50 + ["TypeB"] * 50
        adata.var_names = [f"Gene_{i}" for i in range(6)]
        
        model = MBModel(
            adata, 
            [(0, 1), (2, 3), (4, 5)],
            device="cpu",
            batch_size=25
        )
        model.fit(epochs=100, verbose=False)
        return model
    
    def test_export_global_network(self, fitted_model):
        """Test exporting global network."""
        df = fitted_model.export_networks(threshold=0.5)
        
        assert isinstance(df, pd.DataFrame)
        assert "gene_i" in df.columns
        assert "gene_j" in df.columns
        assert "weight" in df.columns
        
        if fitted_model.prior_type == "spike_slab":
            assert "probability" in df.columns
            assert all(0 <= p <= 1 for p in df["probability"])
    
    def test_export_with_confidence(self, fitted_model):
        """Test exporting with confidence intervals."""
        df = fitted_model.export_networks(
            threshold=0.5,
            include_confidence=True
        )
        
        if fitted_model.prior_type == "spike_slab":
            assert "prob_lower" in df.columns
            assert "prob_upper" in df.columns
        
        assert "weight_lower" in df.columns
        assert "weight_upper" in df.columns
        
        # Check ordering
        if "prob_lower" in df.columns:
            assert all(df["prob_lower"] <= df["probability"])
            assert all(df["probability"] <= df["prob_upper"])
    
    def test_export_cell_type_specific(self, fitted_model):
        """Test exporting cell-type specific networks."""
        networks = fitted_model.export_networks(
            threshold=0.5,
            return_cell_type_specific=True
        )
        
        assert isinstance(networks, dict)
        assert set(networks.keys()) == {"TypeA", "TypeB"}
        
        for _ct_name, df in networks.items():
            assert isinstance(df, pd.DataFrame)
            assert "gene_i" in df.columns
            assert "gene_j" in df.columns
            assert "weight" in df.columns
    
    def test_export_function(self, fitted_model):
        """Test the standalone export function."""
        df1 = fitted_model.export_networks(threshold=0.5)
        df2 = export_networks(fitted_model, threshold=0.5)
        
        pd.testing.assert_frame_equal(df1, df2)


class TestGPUSupport:
    """Test GPU-related functionality."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_training(self):
        """Test training on GPU."""
        rng = np.random.default_rng(42)
        X = rng.poisson(3.0, size=(100, 10)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = ["A"] * 50 + ["B"] * 50
        adata.var_names = [f"G{i}" for i in range(10)]
        
        model = MBModel(adata, [(0, 1)], device="cuda")
        assert model.device.type == "cuda"
        assert model.X.device.type == "cuda"
        
        history = model.fit(epochs=10, verbose=False)
        assert len(history["loss"]) > 0
    
    def test_auto_device_selection(self):
        """Test automatic device selection."""
        rng = np.random.default_rng(42)
        X = rng.poisson(3.0, size=(50, 10)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = ["A"] * 50
        adata.var_names = [f"G{i}" for i in range(10)]
        
        model = MBModel(adata, [(0, 1)])  # No device specified
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert model.device.type == expected_device


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_cell_type(self):
        """Test with only one cell type."""
        rng = np.random.default_rng(42)
        X = rng.poisson(3.0, size=(50, 10)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = ["TypeA"] * 50
        adata.var_names = [f"G{i}" for i in range(10)]
        
        model = MBModel(adata, [(0, 1), (2, 3)])
        assert model.n_types == 1
        
        history = model.fit(epochs=10, verbose=False)
        assert len(history["loss"]) > 0
    
    def test_many_cell_types(self):
        """Test with many cell types."""
        rng = np.random.default_rng(42)
        n_types = 10
        X = rng.poisson(3.0, size=(200, 15)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = np.repeat([f"Type{i}" for i in range(n_types)], 20)
        adata.var_names = [f"G{i}" for i in range(15)]
        
        model = MBModel(adata, [(0, 1), (2, 3)], batch_size=20)
        assert model.n_types == n_types
        
        history = model.fit(epochs=10, verbose=False)
        assert len(history["loss"]) > 0
    
    def test_zero_expression(self):
        """Test handling of all-zero genes."""
        X = np.zeros((100, 10), dtype=np.float32)
        X[:, [0, 2, 4, 6, 8]] = np.random.poisson(3.0, size=(100, 5))
        
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = ["A"] * 50 + ["B"] * 50
        adata.var_names = [f"G{i}" for i in range(10)]
        
        # Include edges with zero-expression genes
        model = MBModel(adata, [(0, 1), (1, 3)])
        history = model.fit(epochs=10, verbose=False)
        
        # Should still train without NaN
        assert all(not np.isnan(loss) for loss in history["loss"])
    
    def test_large_values(self):
        """Test numerical stability with large expression values."""
        rng = np.random.default_rng(42)
        X = rng.poisson(1000.0, size=(50, 10)).astype(np.float32)
        
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = ["A"] * 50
        adata.var_names = [f"G{i}" for i in range(10)]
        
        model = MBModel(adata, [(0, 1), (2, 3)])
        history = model.fit(epochs=10, verbose=False)
        
        # Should handle large values without overflow
        assert all(not np.isnan(loss) for loss in history["loss"])
        assert all(not np.isinf(loss) for loss in history["loss"])


class TestSimulationStudy:
    """Test model performance on simulated data with known ground truth."""
    
    def simulate_data_with_interactions(
        self, 
        n_cells=500, 
        n_genes=20, 
        n_true_edges=5,
        interaction_strength=0.5,
        seed=42
    ):
        """Simulate data with known interactions."""
        rng = np.random.default_rng(seed)
        
        # Base expression levels
        base_expr = rng.gamma(2.0, 2.0, size=(n_cells, n_genes))
        
        # True edges
        true_edges = [(i, i+1) for i in range(0, n_true_edges*2, 2)]
        
        # Add interactions
        for i, j in true_edges:
            # Gene j expression influences gene i
            base_expr[:, i] += interaction_strength * base_expr[:, j]
            base_expr[:, j] += interaction_strength * base_expr[:, i]
        
        # Sample counts
        X = rng.poisson(base_expr).astype(np.float32)
        
        # Create AnnData
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = (
            ["TypeA"] * (n_cells // 2) + 
            ["TypeB"] * (n_cells - n_cells // 2)
        )
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        
        return adata, true_edges
    
    def test_recovery_of_true_interactions(self):
        """Test if model recovers true interactions."""
        adata, true_edges = self.simulate_data_with_interactions(
            n_cells=1000,
            n_genes=20,
            n_true_edges=3,
            interaction_strength=0.8
        )
        
        # Add some false edges as well
        all_edges = true_edges + [(10, 11), (12, 13), (14, 15)]
        
        model = MBModel(
            adata, 
            all_edges, 
            device="cpu",
            batch_size=100,
            lr=0.01
        )
        
        model.fit(epochs=200, verbose=False)
        
        # Export networks
        df = model.export_networks(threshold=0.5)
        
        # Check if true edges have higher weights/probabilities
        true_edge_set = set()
        for i, j in true_edges:
            true_edge_set.add((f"Gene_{i}", f"Gene_{j}"))
            true_edge_set.add((f"Gene_{j}", f"Gene_{i}"))
        
        recovered_edges = set()
        for _, row in df.iterrows():
            recovered_edges.add((row["gene_i"], row["gene_j"]))
        
        # Calculate recovery metrics
        true_positives = len(true_edge_set.intersection(recovered_edges))
        false_positives = len(recovered_edges - true_edge_set)
        
        precision = true_positives / (true_positives + false_positives) if recovered_edges else 0
        recall = true_positives / len(true_edge_set) if true_edge_set else 0
        
        # We expect reasonable recovery
        assert recall > 0.5, f"Poor recall: {recall:.2f}"
        assert precision > 0.5, f"Poor precision: {precision:.2f}"


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete pipeline from data loading to network export."""
        # Create synthetic data
        rng = np.random.default_rng(42)
        n_cells, n_genes = 300, 15
        X = rng.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
        
        adata = ad.AnnData(X)
        adata.obs["cell_type"] = (
            ["Neuron"] * 100 + 
            ["Astrocyte"] * 100 + 
            ["Microglia"] * 100
        )
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        
        # Create prior file
        prior_data = []
        for i in range(0, 10, 2):
            prior_data.append({
                "protein1": f"Gene_{i}",
                "protein2": f"Gene_{i+1}",
                "combined_score": rng.integers(400, 999)
            })
        
        prior_df = pd.DataFrame(prior_data)
        prior_path = tmp_path / "prior.tsv"
        prior_df.to_csv(prior_path, sep="\t", index=False)
        
        # Load prior
        prior_edges = load_string_prior(
            str(prior_path),
            adata_var_names=adata.var_names,
            score_cutoff=500
        )
        
        # Train model
        model = MBModel(
            adata,
            prior_edges,
            batch_size=50,
            lr=0.01,
            device="cpu"
        )
        
        model.fit(
            epochs=100,
            verbose=False,
            patience=20
        )
        
        # Export results
        global_net = model.export_networks(
            threshold=0.7,
            include_confidence=True
        )
        
        ct_nets = model.export_networks(
            threshold=0.7,
            return_cell_type_specific=True
        )
        
        # Validate outputs
        assert isinstance(global_net, pd.DataFrame)
        assert len(global_net) > 0
        assert all(col in global_net.columns for col in ["gene_i", "gene_j", "weight"])
        
        assert isinstance(ct_nets, dict)
        assert set(ct_nets.keys()) == {"Neuron", "Astrocyte", "Microglia"}
        
        # Save results
        output_path = tmp_path / "results.csv"
        global_net.to_csv(output_path, index=False)
        assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
