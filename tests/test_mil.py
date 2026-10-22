import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

if HAS_TORCH:
    from pilot.mil import AttentionMIL, mil_loss  # noqa: E402


@unittest.skipUnless(HAS_TORCH, "torch is required")
class TestAttentionMIL(unittest.TestCase):
    def test_shapes_and_attention_normalisation(self):
        torch.manual_seed(0)
        model = AttentionMIL(feature_dim=16, n_classes=5, hidden=32, descriptor_dim=5)
        feats = torch.randn(2, 7, 16)
        logits, attn, topo = model(feats)
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(tuple(attn.shape), (2, 7))
        self.assertEqual(tuple(topo.shape), (2, 7, 5))
        self.assertTrue(torch.allclose(attn.sum(1), torch.ones(2), atol=1e-5))

    def test_topology_fusion(self):
        model = AttentionMIL(
            feature_dim=16,
            n_classes=5,
            hidden=32,
            fusion_dim=5,
            fusion_hidden=8,
        )
        feats = torch.randn(2, 7, 16)
        desc = torch.randn(2, 7, 5)
        logits, attn, topo = model(feats, desc)
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertIsNone(topo)

    def test_predicted_fusion_requires_descriptor_head(self):
        with self.assertRaises(ValueError):
            AttentionMIL(feature_dim=16, n_classes=5, use_predicted_fusion=True)

    def test_loss_with_topology(self):
        torch.manual_seed(0)
        model = AttentionMIL(feature_dim=16, n_classes=5, hidden=32, descriptor_dim=5)
        feats = torch.randn(2, 7, 16)
        desc = torch.randn(2, 7, 5)
        logits, attn, topo = model(feats)
        loss, parts = mil_loss(logits, torch.tensor([0, 3]), topo, desc, lam_topo=0.5)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("ce", parts)
        self.assertIn("topo", parts)


if __name__ == "__main__":
    unittest.main()
