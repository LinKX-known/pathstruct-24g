"""Minimal auxiliary heads (structure regression, counterfactual yes/no).

Both heads consume the *image-token grid*, not a global mean. This matters:

* the structure target contains a 3x3 per-tile composition block. A global
  mean vector has thrown away which tile a cell came from, so a head fed the
  mean cannot learn that block, no matter how long it trains;
* the counterfactual target depends on the cells *inside the edit region*.
  A global mean lets the head learn dataset priors but not the local
  arrangement that actually decides the answer.

``grid_from_hidden`` turns the last hidden state into an ``(gh, gw, d)`` grid
using ``image_grid_thw`` and the vision tower's merge size.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    F = None
    nn = object
    HAS_TORCH = False

MASK_DIM = 8  # centre x, centre y, radius, translation x, y, kind one-hot (3)
GRID_OUT = 3


def mask_features(center, radius, kind, vector=(0.0, 0.0)):
    onehot = [1.0 if kind == k else 0.0 for k in ("collapse", "expand", "translate")]
    return [float(center[0]), float(center[1]), float(radius), float(vector[0]), float(vector[1])] + onehot


if HAS_TORCH:

    def grid_from_hidden(hidden, image_mask, grid_thw, merge_size: int = 2):
        """Reshape image-token hidden states into ``(gh, gw, d)``.

        Returns ``None`` when the shapes do not line up (wrong token id, a
        batch larger than one, an unexpected merge size). Callers fall back to
        a repeated global mean so training still runs, but the failure is
        visible in the logs.
        """
        if hidden is None or hidden.dim() != 3 or image_mask is None or grid_thw is None:
            return None
        if hidden.shape[0] != 1:
            return None
        tokens = hidden[0][image_mask[0].bool()]
        try:
            t, h, w = [int(v) for v in grid_thw[0].tolist()]
        except Exception:  # noqa: BLE001
            return None
        merge = max(int(merge_size), 1)
        gh, gw = h // merge, w // merge
        if gh <= 0 or gw <= 0 or tokens.shape[0] != gh * gw * max(t, 1):
            return None
        if t > 1:
            tokens = tokens.reshape(t, gh, gw, -1).mean(0)
        return tokens.reshape(gh, gw, -1)

    class StructureHead(nn.Module):
        """Predict the structural descriptor from a spatial 3x3 pooling."""

        def __init__(
            self,
            d_model: int,
            n_desc: int,
            hidden: int = 1024,
            dropout: float = 0.1,
            proj_dim: int = 256,
        ):
            super().__init__()
            self.proj = nn.Conv2d(d_model, proj_dim, kernel_size=1)
            self.net = nn.Sequential(
                nn.LayerNorm(proj_dim * GRID_OUT * GRID_OUT),
                nn.Linear(proj_dim * GRID_OUT * GRID_OUT, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_desc),
            )

        def forward(self, grid):
            # grid: (gh, gw, d) for one image. The heads stay fp32: the model
            # runs bf16, but a small regression/classification head in fp32 is
            # more stable and costs only a few MB.
            x = grid.float().permute(2, 0, 1).unsqueeze(0)  # (1, d, gh, gw)
            x = self.proj(x)
            x = F.adaptive_avg_pool2d(x, (GRID_OUT, GRID_OUT)).flatten(1)
            return self.net(x)

    class CounterfactualHead(nn.Module):
        """Judge an edit from local visual features plus the edit geometry."""

        def __init__(self, d_model: int, hidden: int = 512, dropout: float = 0.1):
            super().__init__()
            self.mask = nn.Sequential(nn.Linear(MASK_DIM, hidden), nn.GELU())
            self.fuse = nn.Sequential(
                nn.Linear(d_model + hidden, hidden), nn.GELU(), nn.Dropout(dropout)
            )
            self.cls = nn.Linear(hidden, 1)

        def forward(self, grid, mask_feats):
            if grid is None:
                raise ValueError("CounterfactualHead needs an image-token grid")
            grid = grid.float()
            mask_feats = mask_feats.float()
            gh, gw, d = grid.shape
            cx = mask_feats[0, 0]
            cy = mask_feats[0, 1]
            radius = mask_feats[0, 2].clamp(min=1e-4)
            ys = (torch.arange(gh, device=grid.device, dtype=grid.dtype) + 0.5) / gh
            xs = (torch.arange(gw, device=grid.device, dtype=grid.dtype) + 0.5) / gw
            dist2 = ((xs[None, :] - cx) / radius) ** 2 + ((ys[:, None] - cy) / radius) ** 2
            selected = dist2 <= 1.0
            if bool(selected.any()):
                region = grid[selected].mean(0)
            else:  # tiny region: use the nearest token
                flat = grid.reshape(-1, d)
                region = flat[int(torch.argmin(dist2))]
            h = self.fuse(torch.cat([region.unsqueeze(0), self.mask(mask_feats)], dim=-1))
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

    def grid_from_hidden(*args, **kwargs):
        raise ImportError("torch is required for the pilot heads")
