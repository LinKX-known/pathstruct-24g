"""Minimal auxiliary heads (structure regression, counterfactual yes/no)."""

from __future__ import annotations

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    nn = object
    HAS_TORCH = False

MASK_DIM = 6


def mask_features(center, radius, kind):
    onehot = [1.0 if kind == k else 0.0 for k in ("collapse", "expand", "translate")]
    return [float(center[0]), float(center[1]), float(radius)] + onehot


if HAS_TORCH:

    class StructureHead(nn.Module):
        def __init__(self, d_model: int, n_desc: int, hidden: int = 1024, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_desc),
            )

        def forward(self, hidden, attention_mask=None):
            if hidden.dim() == 3:
                if attention_mask is not None:
                    w = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * w).sum(1) / w.sum(1).clamp(min=1.0)
                else:
                    pooled = hidden.mean(1)
            else:
                pooled = hidden
            return self.net(pooled)

    class CounterfactualHead(nn.Module):
        def __init__(self, d_model: int, hidden: int = 512, dropout: float = 0.1):
            super().__init__()
            self.mask = nn.Sequential(nn.Linear(MASK_DIM, hidden), nn.GELU())
            self.fuse = nn.Sequential(nn.Linear(d_model + hidden, hidden), nn.GELU(), nn.Dropout(dropout))
            self.cls = nn.Linear(hidden, 1)

        def forward(self, pooled, mask_feats):
            h = self.fuse(torch.cat([pooled, self.mask(mask_feats)], dim=-1))
            return self.cls(h).squeeze(-1)

    def build_heads(d_model: int, n_desc: int, hidden: int = 1024):
        return nn.ModuleDict(
            {
                "structure": StructureHead(d_model, n_desc, hidden),
                "cf": CounterfactualHead(d_model, max(128, hidden // 2)),
            }
        )

else:  # pragma: no cover

    def build_heads(*args, **kwargs):
        raise ImportError("torch is required for the pilot heads")
