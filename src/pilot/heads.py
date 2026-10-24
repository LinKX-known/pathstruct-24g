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
        batch larger than one, an unexpected merge size). Callers should use
        :func:`grid_or_global_mean` when they need a fallback, so the fallback
        is counted instead of being silent.
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

    def grid_or_global_mean(hidden, image_mask, grid_thw, merge_size: int = 2):
        """Grid if the token layout is known, otherwise a repeated global mean.

        Returns ``(grid, status)`` with status ``"grid"``, ``"fallback"`` or
        ``"failed"``.  The fallback keeps training/evaluation running when
        ``image_grid_thw`` is missing, but it destroys the localisation the
        heads depend on, so every caller must count and report it.
        """
        grid = grid_from_hidden(hidden, image_mask, grid_thw, merge_size)
        if grid is not None:
            return grid, "grid"
        if hidden is None or hidden.dim() != 3 or image_mask is None:
            return None, "failed"
        if hidden.shape[0] != 1:
            return None, "failed"
        vis = image_mask[0].bool()
        if not bool(vis.any()):
            return None, "failed"
        w = vis.to(hidden.dtype).unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
        pooled = (hidden * w).sum(1) / w.sum(1).clamp(min=1.0)
        return pooled[0].view(1, 1, -1).expand(3, 3, -1), "fallback"

    class StructureHead(nn.Module):
        """Predict the structural descriptor from a spatial 3x3 pooling."""

        def __init__(
            self,
            d_model: int,
            n_desc: int,
            hidden: int = 1024,
            dropout: float = 0.1,
            proj_dim: int = 256,
            presence_dims: int = 0,
        ):
            super().__init__()
            self.n_desc = int(n_desc)
            self.presence_dims = int(presence_dims)
            self.proj = nn.Conv2d(d_model, proj_dim, kernel_size=1)
            self.net = nn.Sequential(
                nn.LayerNorm(proj_dim * GRID_OUT * GRID_OUT),
                nn.Linear(proj_dim * GRID_OUT * GRID_OUT, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                # First n_desc outputs are log1p-space regressions; the extra
                # presence_dims are presence logits for the zero-inflated dims.
                nn.Linear(hidden, n_desc + presence_dims),
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
            if mask_feats.ndim != 2 or mask_feats.shape[1] != MASK_DIM:
                raise ValueError(
                    f"mask_feats must be (1, {MASK_DIM}); got {tuple(mask_feats.shape)}"
                )
            if not bool(torch.isfinite(mask_feats).all()):
                raise ValueError("mask_feats contains NaN/Inf")
            grid = grid.float()
            mask_feats = mask_feats.float()
            gh, gw, d = grid.shape
            cx, cy = float(mask_feats[0, 0]), float(mask_feats[0, 1])
            radius_raw = float(mask_feats[0, 2])
            # The head pools the token grid in normalised [0, 1] coordinates.
            # A pixel-space centre (e.g. 512, 512) or a pixel-space radius
            # (e.g. 60) would silently select one token or the whole grid, so
            # fail loudly instead of pretending the geometry is meaningful.
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                raise ValueError(
                    "mask centre must be normalised to [0, 1]; "
                    f"got ({cx:.3f}, {cy:.3f}). Did you pass pixel coordinates?"
                )
            if not (0.0 < radius_raw <= 2.0):
                raise ValueError(
                    "mask radius must be normalised to (0, 2]; "
                    f"got {radius_raw:.3f}. Did you pass a pixel radius?"
                )
            if float(mask_feats[0, 3:5].abs().max()) > 2.0:
                raise ValueError("mask translation vector must be normalised")
            radius = torch.as_tensor(radius_raw, device=grid.device, dtype=grid.dtype).clamp(min=1e-4)
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

    def build_heads(
        d_model: int,
        n_desc: int,
        hidden: int = 1024,
        presence_dims: int = 0,
    ):
        return nn.ModuleDict(
            {
                "structure": StructureHead(
                    d_model, n_desc, hidden, presence_dims=presence_dims
                ),
                "cf": CounterfactualHead(d_model, max(128, hidden // 2)),
            }
        )

else:  # pragma: no cover

    def build_heads(*args, **kwargs):
        raise ImportError("torch is required for the pilot heads")

    def grid_from_hidden(*args, **kwargs):
        raise ImportError("torch is required for the pilot heads")
