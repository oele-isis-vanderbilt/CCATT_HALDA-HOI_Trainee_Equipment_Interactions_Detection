"""
RoleDetectionModel: a stock ultralytics YOLOv8 DetectionModel plus one small addition — a
role-embedding head that reads the P3 feature map (already computed by the shared
backbone/neck) and produces a per-box embedding, trained with a supervised-contrastive loss
over the existing role labels (Nurse/Physician/RT/Additional Staff). Detection behavior is
otherwise unchanged.

Verified against the installed ultralytics==8.3.13 source (ultralytics/nn/tasks.py):
- DetectionModel._predict_once calls `x = m(x)` for each layer m in self.model; for the final
  Detect module, m.f (e.g. [15, 18, 21]) selects 3 earlier layer outputs, so Detect receives
  `x = [P3, P4, P5]` as a list. A forward-pre-hook on self.model[-1] observes exactly this
  list without touching _predict_once.
- The Detect module's own cv2[0][0].conv.in_channels gives P3's channel count directly
  (confirmed: 256 for yolov8l) — no dummy forward pass needed to size the embedding head.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.ops import roi_align
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel


class RoleEmbedHead(nn.Module):
    """Conv -> RoIAlign(GT boxes) -> global-avg-pool -> Linear -> L2-normalize."""

    def __init__(self, in_channels: int, embed_dim: int = 128, hidden_channels: int = 256, roi_output_size: int = 7):
        super().__init__()
        self.conv = Conv(in_channels, hidden_channels, k=3)
        self.roi_output_size = roi_output_size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, embed_dim)

    def forward(self, feature_map: torch.Tensor, rois: torch.Tensor, spatial_scale: float) -> torch.Tensor:
        """
        feature_map: (B, C, H, W) — P3 feature map for the whole batch.
        rois: (N, 5) — [batch_idx, x1, y1, x2, y2] in absolute input-image pixel coords.
        spatial_scale: feature_map_size / input_image_size (nominally 1/8 for P3).

        Returns (N, embed_dim) L2-normalized embeddings, one per row of `rois`.
        """
        if rois.shape[0] == 0:
            return torch.zeros((0, self.fc.out_features), device=feature_map.device, dtype=feature_map.dtype)
        x = self.conv(feature_map)

        # sampling_ratio is pinned to a small fixed value rather than left at torchvision's
        # adaptive default (-1, which scales the internal sampling grid with box size). The
        # adaptive default caused a real CUDA OOM here (tried to allocate 13+ GiB for a single
        # RoIAlign call) because torchvision's roi_align falls back to a pure-Python reference
        # implementation in this environment, which fully materializes a [K, C, PH, PW, IY, IX]
        # tensor — for person-sized boxes (hundreds of px tall) the adaptive IY/IX blow up.
        # sampling_ratio=2 is the standard fixed value used in production RoIAlign heads
        # (e.g. Detectron2) for exactly this reason, and bounds memory regardless of box size.
        if x.device.type == "mps":
            # torchvision.ops.roi_align has no native MPS kernel in torch==2.2.0; its Python
            # fallback internally mixes MPS and CPU tensors and raises a device-mismatch
            # RuntimeError. Route just this op through CPU (tensors here are tiny — one
            # feature map + a handful of boxes) and move the result back.
            pooled = roi_align(
                x.cpu(), rois.cpu(), output_size=self.roi_output_size, spatial_scale=spatial_scale,
                sampling_ratio=2, aligned=True,
            ).to(x.device)
        else:
            pooled = roi_align(
                x, rois, output_size=self.roi_output_size, spatial_scale=spatial_scale,
                sampling_ratio=2, aligned=True,
            )
        pooled = self.pool(pooled).flatten(1)
        embed = self.fc(pooled)
        return nn.functional.normalize(embed, dim=1)


class RoleDetectionModel(DetectionModel):
    """DetectionModel + a role-embedding head trained jointly via JointDetRoleLoss."""

    def __init__(self, cfg="yolov8l.yaml", ch=3, nc=None, verbose=True, embed_dim: int = 128):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        detect = self.model[-1]
        p3_channels = detect.cv2[0][0].conv.in_channels
        self.role_embed_head = RoleEmbedHead(in_channels=p3_channels, embed_dim=embed_dim)

        self._p3_feat: Optional[torch.Tensor] = None
        self._fpn_feats: Optional[list] = None  # [P3, P4, P5], used by gradcam_role.py
        detect.register_forward_pre_hook(self._capture_p3_hook)

    def _capture_p3_hook(self, module: nn.Module, args: tuple) -> None:
        feats = args[0]  # [P3, P4, P5], see module docstring
        self._p3_feat = feats[0]
        # Detect.forward later does `x[i] = torch.cat((cv2(x[i]), cv3(x[i])), 1)` — an
        # in-place mutation of THIS SAME list object. Store a shallow copy so _fpn_feats
        # keeps pointing at the original backbone/neck tensors, not the post-mutation
        # (box+cls concatenated) ones, once the full forward pass has completed.
        self._fpn_feats = list(feats)

    def init_criterion(self):
        from joint_loss import JointDetRoleLoss

        return JointDetRoleLoss(self)
