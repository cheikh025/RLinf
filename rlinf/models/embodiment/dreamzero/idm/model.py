# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inverse Dynamics Model (IDM) for the DreamZero PRM consistency term.

Maps a dreamed (or real) multi-view video clip to the action chunk that
produced it: ``(B, n_views, n_frames, 3, H, W)`` -> ``(B, action_horizon,
action_dim)``. Input format mirrors the WAM's decoded output (1 real
conditioning frame + dreamed frames at the multi-anchor offsets), so at
inference the decoded clip is consumed as-is after splitting the side-by-side
camera canvas.

Architecture (design record: ``dreamzero_prm_milestone3_idm_README.md``):

- **Pairwise early fusion** (default): adjacent frame pairs are stacked into
  6-channel images through a shared ImageNet-pretrained ResNet-18 whose conv1
  is inflated 3 -> 6 channels. Inverse dynamics is motion estimation, and
  pair fusion computes motion with convolutions at full input resolution --
  where a few-pixel shift is trivially detectable -- instead of asking
  attention to recover sub-cell correspondence from stride-32 features. The
  spatial map is kept (no global pooling) so fine, local motion such as the
  gripper fingers survives into the tokens.
- ``pair_fusion=False`` switches to single-frame late fusion (same backbone,
  3-channel inputs, one token set per frame) as a one-flag ablation; nothing
  else changes.
- Tokens (pairs x views x spatial positions) get summed learned embeddings
  for interval index, camera id, and spatial position, then a small
  transformer encoder mixes cross-view / cross-interval context.
- ``action_horizon`` learned action queries cross-attend over the tokens in a
  transformer decoder; the frame<->action alignment is *learned*, never
  hand-coded (mirror-don't-assume, carried into the architecture).
- Heads: shared MLP -> arm deltas (standardized scale), shared linear ->
  per-step gripper logit.

Loss (``compute_loss``): SmoothL1 (beta=0.1) on per-dim standardized arm
targets + weighted BCE on the binarized gripper (the sim consumes only
``sign(action)``), combined as ``L_arm + lambda_grip * L_grip``.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class IDMConfig:
    """Configuration for :class:`IDM`.

    Defaults mirror the DreamZero/LIBERO decoded clip: 9 frames (1 real +
    8 dreamed) of a 160x320 canvas split into 2 views of 160x160, scoring a
    16-step chunk of 7-dim actions (6 arm + 1 gripper).
    """

    n_views: int = 2
    n_frames: int = 9
    action_horizon: int = 16
    action_dim: int = 7
    image_size: tuple = (160, 160)

    pair_fusion: bool = True
    pretrained_backbone: bool = True

    d_model: int = 256
    n_heads: int = 8
    ffn_dim: int = 1024
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1
    head_hidden: int = 256


def _build_resnet18_trunk(in_channels: int, pretrained: bool) -> nn.Module:
    """ResNet-18 up to layer4 (no avgpool/fc), conv1 inflated to in_channels.

    Inflation duplicates the pretrained RGB kernels across the extra channels
    and rescales so the expected response magnitude is preserved.
    """
    import torchvision.models as tvm

    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = tvm.resnet18(weights=weights)

    if in_channels != 3:
        old = net.conv1
        new = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        if pretrained:
            repeats = in_channels // 3
            with torch.no_grad():
                w = old.weight.repeat(1, repeats, 1, 1) / repeats
                new.weight.copy_(w)
        net.conv1 = new

    return nn.Sequential(
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool,
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4,
    )


class IDM(nn.Module):
    """Video clip -> action chunk inverse dynamics model (~16M params)."""

    BACKBONE_DIM = 512
    BACKBONE_STRIDE = 32

    def __init__(self, cfg: Optional[IDMConfig] = None):
        super().__init__()
        self.cfg = cfg or IDMConfig()
        c = self.cfg

        in_channels = 6 if c.pair_fusion else 3
        self.backbone = _build_resnet18_trunk(in_channels, c.pretrained_backbone)
        self.token_proj = nn.Linear(self.BACKBONE_DIM, c.d_model)

        feat_h = c.image_size[0] // self.BACKBONE_STRIDE
        feat_w = c.image_size[1] // self.BACKBONE_STRIDE
        self._feat_hw = (feat_h, feat_w)
        n_steps = c.n_frames - 1 if c.pair_fusion else c.n_frames

        self.time_embed = nn.Embedding(n_steps, c.d_model)
        self.cam_embed = nn.Embedding(c.n_views, c.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, feat_h * feat_w, c.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.ffn_dim,
            dropout=c.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=c.encoder_layers, norm=nn.LayerNorm(c.d_model)
        )

        self.action_queries = nn.Parameter(torch.zeros(1, c.action_horizon, c.d_model))
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.ffn_dim,
            dropout=c.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=c.decoder_layers, norm=nn.LayerNorm(c.d_model)
        )

        num_arm = c.action_dim - 1
        self.arm_head = nn.Sequential(
            nn.LayerNorm(c.d_model),
            nn.Linear(c.d_model, c.head_hidden),
            nn.GELU(),
            nn.Linear(c.head_hidden, num_arm),
        )
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(c.d_model),
            nn.Linear(c.d_model, 1),
        )

        self.register_buffer(
            "img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0
        )
        self.register_buffer(
            "img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0
        )
        # Per-dim standardization of arm targets (rotation dims are much
        # smaller than translation; without this they under-train). Identity
        # until set_action_stats() is called from the dataset statistics; the
        # buffers persist in the checkpoint, so inference needs no side files.
        self.register_buffer("arm_mean", torch.zeros(num_arm))
        self.register_buffer("arm_std", torch.ones(num_arm))

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set arm-dim standardization stats (computed over the demo set)."""
        self.arm_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.arm_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def _tokenize(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, V, T, 3, H, W)`` RGB [0, 255] -> ``(B, n_tokens, d_model)``."""
        c = self.cfg
        b, v, t, ch, h, w = video.shape
        if (v, t, ch) != (c.n_views, c.n_frames, 3):
            raise ValueError(
                f"expected (B, {c.n_views}, {c.n_frames}, 3, H, W), "
                f"got {tuple(video.shape)}"
            )
        if (h, w) != tuple(c.image_size):
            raise ValueError(f"expected image size {c.image_size}, got {(h, w)}")

        x = video.reshape(b * v * t, ch, h, w).float()
        x = (x - self.img_mean) / self.img_std
        x = x.view(b, v, t, ch, h, w)

        if c.pair_fusion:
            # Adjacent frames stacked channel-wise: motion is computed by the
            # conv stem at full resolution, one token set per interval.
            x = torch.cat([x[:, :, :-1], x[:, :, 1:]], dim=3)  # (B,V,T-1,6,H,W)
        n_steps = x.shape[2]

        feats = self.backbone(x.reshape(b * v * n_steps, x.shape[3], h, w))
        fh, fw = feats.shape[-2:]
        if (fh, fw) != self._feat_hw:
            raise RuntimeError(
                f"backbone produced {fh}x{fw} map, expected {self._feat_hw}"
            )
        feats = feats.flatten(2).transpose(1, 2)  # (B*V*S, fh*fw, 512)
        tokens = self.token_proj(feats).view(b, v, n_steps, fh * fw, -1)

        cam_idx = torch.arange(v, device=video.device)
        time_idx = torch.arange(n_steps, device=video.device)
        tokens = (
            tokens
            + self.cam_embed(cam_idx).view(1, v, 1, 1, -1)
            + self.time_embed(time_idx).view(1, 1, n_steps, 1, -1)
            + self.pos_embed.view(1, 1, 1, fh * fw, -1)
        )
        return tokens.reshape(b, v * n_steps * fh * fw, -1)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict an action chunk from a multi-view clip.

        Args:
            video: ``(B, n_views, n_frames, 3, H, W)`` RGB, uint8 or float in
                [0, 255]. Frame 0 is the real conditioning frame.

        Returns:
            ``arm``: ``(B, action_horizon, action_dim - 1)`` arm deltas in
            *standardized* scale (see ``arm_mean`` / ``arm_std``).
            ``gripper_logit``: ``(B, action_horizon)`` binary gripper logits.
        """
        tokens = self.encoder(self._tokenize(video))
        queries = self.action_queries.expand(video.shape[0], -1, -1)
        h_out = self.decoder(queries, tokens)  # (B, horizon, d_model)
        return {
            "arm": self.arm_head(h_out),
            "gripper_logit": self.gripper_head(h_out).squeeze(-1),
        }

    @torch.no_grad()
    def predict(self, video: torch.Tensor) -> torch.Tensor:
        """Inference: env-space action chunk ``(B, action_horizon, action_dim)``.

        Arm dims are de-standardized; the gripper is the policy-convention
        binary command (logit > 0 -> +1.0, else -1.0).
        """
        out = self.forward(video)
        arm = out["arm"] * self.arm_std + self.arm_mean
        grip = torch.where(out["gripper_logit"] > 0, 1.0, -1.0).unsqueeze(-1)
        return torch.cat([arm, grip], dim=-1)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    target_actions: torch.Tensor,
    model: IDM,
    lambda_grip: float = 0.05,
    arm_beta: float = 0.1,
    gripper_pos_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """IDM training loss.

    Arm: SmoothL1 with small ``beta`` on standardized targets -- with typical
    normalized-action magnitudes ~0.1, the default beta=1.0 would be pure-MSE
    regime (maximal regression-to-the-mean); beta=0.1 keeps typical errors
    quadratic while capping gradients on demo outliers. Gripper: weighted BCE
    on the binarized command (``> 0``), since the sim consumes only its sign.

    Args:
        outputs: dict from :meth:`IDM.forward`.
        target_actions: ``(B, action_horizon, action_dim)`` env-space actions.
        model: the IDM (for the standardization buffers).
        lambda_grip: gripper loss weight; raw BCE is ~2 orders larger than
            SmoothL1 on standardized residuals, so this must stay small.
        arm_beta: SmoothL1 transition point in standardized units.
        gripper_pos_weight: optional scalar tensor for BCE class imbalance.

    Returns:
        ``(total_loss, metrics)`` with per-term values and binary gripper
        accuracy for logging.
    """
    target = torch.as_tensor(target_actions, dtype=torch.float32)
    arm_target = (target[..., :-1] - model.arm_mean) / model.arm_std
    arm_loss = F.smooth_l1_loss(outputs["arm"], arm_target, beta=arm_beta)

    grip_target = (target[..., -1] > 0).float()
    grip_loss = F.binary_cross_entropy_with_logits(
        outputs["gripper_logit"], grip_target, pos_weight=gripper_pos_weight
    )

    total = arm_loss + lambda_grip * grip_loss
    grip_acc = ((outputs["gripper_logit"] > 0).float() == grip_target).float().mean()
    return total, {
        "loss": total.item(),
        "arm_loss": arm_loss.item(),
        "grip_loss": grip_loss.item(),
        "grip_acc": grip_acc.item(),
    }


def split_canvas(canvas: torch.Tensor, n_views: int = 2) -> torch.Tensor:
    """Split the side-by-side camera canvas into per-view clips.

    The WAM decodes to a ``(B, T, H, n_views * H, 3)`` canvas (LIBERO: left
    half exterior, right half wrist). Returns ``(B, n_views, T, 3, H, W)``
    ready for :meth:`IDM.forward`. The same split is applied to training
    clips so train and inference layouts match exactly.
    """
    if canvas.ndim != 5 or canvas.shape[-1] != 3:
        raise ValueError(f"expected (B, T, H, W, 3), got {tuple(canvas.shape)}")
    width = canvas.shape[3]
    if width % n_views != 0:
        raise ValueError(f"canvas width {width} not divisible by {n_views} views")
    views = torch.stack(canvas.chunk(n_views, dim=3), dim=1)  # (B,V,T,H,W/V,3)
    return views.permute(0, 1, 2, 5, 3, 4).contiguous()


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
