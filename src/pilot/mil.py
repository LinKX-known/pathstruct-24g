"""Attention MIL for slide-level labels, with an optional topology branch.

DHMC only has a *slide-level* predominant growth pattern.  A patch classifier
trained on the slide label sees noisy labels; an attention-MIL head aggregates
patch embeddings into one slide prediction and is the protocol used by
CLAM/PMIL-style pipelines.  This module is deliberately small: it is a head,
not a training framework.

Two topology modes
------------------
``descriptor_dim > 0``
    The head also regresses the patch-level topology descriptor from each patch
    embedding (an auxiliary loss).  This is the cheapest way to keep the
    backbone representation topology-aware while the bag label is slide-level.

``fusion_dim > 0``
    The patch topology descriptor (measured from the cell table, not predicted)
    is encoded and attention-pooled alongside the visual features, then
    concatenated before the slide classifier.  This mirrors PMIL's feature-level
    fusion of RGB and persistent-homology features.  If ``fusion_dim`` equals
    ``descriptor_dim`` and ``use_predicted_fusion`` is true, the *predicted*
    descriptor can be used instead (a topology bottleneck).

All tensors are batches of bags: ``(B, N, D)`` for patch features,
``(B, N, T)`` for descriptors, and ``(B,)`` for slide labels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        hidden: int = 256,
        dropout: float = 0.1,
        descriptor_dim: int = 0,
        fusion_dim: int = 0,
        fusion_hidden: int = 64,
        use_predicted_fusion: bool = False,
    ):
        super().__init__()
        if descriptor_dim < 0 or fusion_dim < 0:
            raise ValueError("descriptor_dim/fusion_dim must be >= 0")
        if use_predicted_fusion and descriptor_dim <= 0:
            raise ValueError("predicted fusion needs descriptor_dim > 0")
        self.feature_dim = feature_dim
        self.descriptor_dim = descriptor_dim
        self.fusion_dim = fusion_dim
        self.use_predicted_fusion = use_predicted_fusion

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.topo_reg = (
            nn.Linear(feature_dim, descriptor_dim) if descriptor_dim > 0 else None
        )
        self.topo_encoder = (
            nn.Sequential(nn.Linear(fusion_dim, fusion_hidden), nn.GELU())
            if fusion_dim > 0
            else None
        )
        fused = feature_dim + (fusion_hidden if fusion_dim > 0 else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, n_classes),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        descriptors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(slide_logits, attention_weights, predicted_descriptors)``.

        ``patch_features``: ``(B, N, D)``.  ``descriptors``: ``(B, N, T)``,
        required when ``fusion_dim > 0``; ignored otherwise.
        """
        if patch_features.dim() != 3:
            raise ValueError("patch_features must be (B, N, D)")
        if patch_features.shape[0] == 0 or patch_features.shape[1] == 0:
            raise ValueError("empty bag")
        attn_logits = self.attention(patch_features).squeeze(-1)  # (B, N)
        attn = F.softmax(attn_logits, dim=1)
        bag = (attn.unsqueeze(-1) * patch_features).sum(1)  # (B, D)

        topo_pred = self.topo_reg(patch_features) if self.topo_reg is not None else None

        if self.topo_encoder is not None:
            if descriptors is None:
                raise ValueError("descriptors are required when fusion_dim > 0")
            if descriptors.shape[:2] != patch_features.shape[:2]:
                raise ValueError(
                    f"descriptor bag {tuple(descriptors.shape)} does not match "
                    f"patch bag {tuple(patch_features.shape)}"
                )
            source = topo_pred if self.use_predicted_fusion else descriptors
            if source.shape[-1] != self.fusion_dim:
                raise ValueError(
                    f"fusion expects {self.fusion_dim} descriptor dims, got {source.shape[-1]}"
                )
            topo = self.topo_encoder(source.float())
            topo_bag = (attn.unsqueeze(-1) * topo).sum(1)
            bag = torch.cat([bag, topo_bag], dim=-1)

        logits = self.classifier(bag)
        return logits, attn, topo_pred


def mil_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topo_pred: Optional[torch.Tensor] = None,
    topo_target: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
    lam_topo: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """Slide CE plus an optional patch-level topology regression.

    ``topo_target`` is expected in log1p space with shape ``(B, N, T)``; use
    ``pilot.structure_eval.log1p_target`` on the raw descriptors.  The topology
    loss is only computed where the target is finite.
    """
    ce = F.cross_entropy(logits, labels, weight=class_weights)
    loss = ce
    parts = {"ce": float(ce.detach())}
    if topo_pred is not None and topo_target is not None:
        if topo_pred.shape != topo_target.shape:
            raise ValueError(
                f"topology prediction {tuple(topo_pred.shape)} != target {tuple(topo_target.shape)}"
            )
        mask = torch.isfinite(topo_target)
        if mask.any():
            tl = F.smooth_l1_loss(topo_pred[mask], topo_target[mask])
            loss = loss + lam_topo * tl
            parts["topo"] = float(tl.detach())
    return loss, parts
