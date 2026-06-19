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

"""Latent-space Inverse Dynamics Model for the DreamZero PRM consistency term.

Maps the WAM's *video latent* directly to the action chunk that produced the
motion it depicts: ``(B, C, T, H, W)`` -> ``(B, action_horizon, action_dim)``.
Unlike :class:`rlinf.models.embodiment.dreamzero.idm.model.IDM` (which decodes
the latent to RGB and reads pixels), this model consumes the WAN-VAE latent the
DiT produces (``video_pred``) with **no decode** -- so training and inference
share one latent space and the VAE decoder leg of the domain gap disappears.

Design (see ``dreamzero_latent_idm_design_README.md``):

- The latent is a small spatio-temporal feature volume; each ``(t, h, w)`` cell
  is one token whose feature vector is the ``C`` latent channels. A single
  ``Linear(C -> d_model)`` is the embedding (the analogue of ``token_proj`` in
  the pixel IDM, minus the ResNet -- the VAE already did the spatial feature
  extraction).
- Tokens get summed learned embeddings for time slot and spatial position
  (a transformer is order-blind; position is what lets it read motion =
  change of the same spatial cell across time). No camera embedding: the dual
  camera canvas is jointly encoded, so it is never split -- column position
  carries left/right, and the seam coupling is kept intact.
- A plain transformer encoder mixes the tokens; ``action_horizon`` learned
  action queries cross-attend over them in a transformer decoder. The
  ``slot -> action`` alignment is *learned*, never hand-coded -- the 3 latent
  slots map non-uniformly (1+4+4 causal) onto the 16 steps.
- Heads: shared MLP -> arm deltas (standardized scale), shared linear ->
  per-step gripper logit.

Loss and parameter counting are shared with the pixel IDM
(:func:`...idm.model.compute_loss`, :func:`...idm.model.count_parameters`):
SmoothL1 (beta=0.1) on per-dim standardized arm targets + weighted BCE on the
binarized gripper, combined as ``L_arm + lambda_grip * L_grip``.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from rlinf.models.embodiment.dreamzero.idm.model import (  # noqa: F401
    compute_loss,
    count_parameters,
)


@dataclass
class LatentIDMConfig:
    """Configuration for :class:`LatentActionIDM`.

    Defaults mirror the DreamZero/LIBERO Wan2.2 video latent: the 160x320
    dual-camera canvas encodes (z_dim=48, 16x spatial, 4x temporal) to a
    ``[48, 3, 10, 20]`` latent, scoring a 16-step chunk of 7-dim actions
    (6 arm + 1 gripper).
    """

    latent_channels: int = 48
    latent_t: int = 3
    latent_h: int = 10
    latent_w: int = 20
    action_horizon: int = 16
    action_dim: int = 7

    d_model: int = 256
    n_heads: int = 8
    ffn_dim: int = 1024
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1
    head_hidden: int = 256


class LatentActionIDM(nn.Module):
    """Video latent -> action chunk inverse dynamics model (~4M params)."""

    def __init__(self, cfg: Optional[LatentIDMConfig] = None):
        super().__init__()
        self.cfg = cfg or LatentIDMConfig()
        c = self.cfg

        # Embedding: each (t, h, w) cell's C latent channels -> d_model token.
        self.token_proj = nn.Linear(c.latent_channels, c.d_model)

        n_pos = c.latent_h * c.latent_w
        self.time_embed = nn.Embedding(c.latent_t, c.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_pos, c.d_model))
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

    def _tokenize(self, latent: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W)`` latent -> ``(B, T*H*W, d_model)`` tokens."""
        c = self.cfg
        b, ch, t, h, w = latent.shape
        if (ch, t, h, w) != (c.latent_channels, c.latent_t, c.latent_h, c.latent_w):
            raise ValueError(
                f"expected (B, {c.latent_channels}, {c.latent_t}, {c.latent_h}, "
                f"{c.latent_w}), got {tuple(latent.shape)}"
            )

        # Move channels last so each (t, h, w) cell is a C-vector, then embed.
        x = latent.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, C)
        x = x.view(b, t, h * w, ch).float()  # (B, T, HW, C)
        tokens = self.token_proj(x)  # (B, T, HW, d_model)

        time_idx = torch.arange(t, device=latent.device)
        tokens = (
            tokens
            + self.time_embed(time_idx).view(1, t, 1, -1)  # varies over TIME
            + self.pos_embed.view(1, 1, h * w, -1)  # varies over POSITION
        )
        return tokens.reshape(b, t * h * w, -1)  # (B, T*HW, d_model)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict an action chunk from a video latent.

        Args:
            latent: ``(B, C, T, H, W)`` WAN-VAE video latent (``video_pred`` at
                inference, or ``VAE.encode(real canvas)`` at training).

        Returns:
            ``arm``: ``(B, action_horizon, action_dim - 1)`` arm deltas in
            *standardized* scale (see ``arm_mean`` / ``arm_std``).
            ``gripper_logit``: ``(B, action_horizon)`` binary gripper logits.
        """
        tokens = self.encoder(self._tokenize(latent))
        queries = self.action_queries.expand(latent.shape[0], -1, -1)
        h_out = self.decoder(queries, tokens)  # (B, horizon, d_model)
        return {
            "arm": self.arm_head(h_out),
            "gripper_logit": self.gripper_head(h_out).squeeze(-1),
        }

    @torch.no_grad()
    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        """Inference: env-space action chunk ``(B, action_horizon, action_dim)``.

        Arm dims are de-standardized; the gripper is the policy-convention
        binary command (logit > 0 -> +1.0, else -1.0).
        """
        out = self.forward(latent)
        arm = out["arm"] * self.arm_std + self.arm_mean
        grip = torch.where(out["gripper_logit"] > 0, 1.0, -1.0).unsqueeze(-1)
        return torch.cat([arm, grip], dim=-1)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> "LatentActionIDM":
        """Load a trained latent IDM from a checkpoint directory or ``.pt`` file.

        ``path`` may be a directory written by the trainer (picks ``best.pt`` >
        ``final.pt`` > ``latest.pt``) or a direct ``.pt`` file. The checkpoint
        carries ``idm_cfg`` and the model ``state_dict`` including the
        standardization buffers (``arm_mean`` / ``arm_std``), so no side files
        or :meth:`set_action_stats` call is needed at load time. Returned frozen
        and in eval mode; defaults to fp32 since :meth:`forward` floats its
        input (run under autocast for bf16).
        """
        import os

        ckpt_path = path
        if os.path.isdir(path):
            for name in ("best.pt", "final.pt", "latest.pt"):
                candidate = os.path.join(path, name)
                if os.path.isfile(candidate):
                    ckpt_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"No best.pt/final.pt/latest.pt in IDM checkpoint dir {path!r}"
                )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "idm_cfg" not in ckpt or "model" not in ckpt:
            raise KeyError(
                f"Latent IDM checkpoint {ckpt_path!r} missing 'idm_cfg'/'model'; "
                "expected a checkpoint saved by the latent IDM trainer."
            )
        cfg = LatentIDMConfig(**ckpt["idm_cfg"])
        model = cls(cfg)
        model.load_state_dict(ckpt["model"])
        return model.eval().to(device=device, dtype=dtype).requires_grad_(False)
