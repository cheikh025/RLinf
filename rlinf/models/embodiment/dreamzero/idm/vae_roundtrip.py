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

"""On-the-fly WAN-VAE roundtrip augmentation for IDM training (domain gap).

The IDM trains on real sim frames but at inference only ever consumes
WAN-VAE-decoded dreams. ``VaeRoundtrip`` passes a training canvas through
``decode(encode(.))`` of the exact production VAE
(:class:`WanVideoVAE38`, z_dim=48), so the IDM sees the same VAE-artifact
distribution it will score at inference. The VAE encode returns the posterior
mean and never samples (verified by ``dreamzero_verify_vae_determinism.py``),
so the roundtrip is a *fixed* transform applied per step -- no disk cache.

``RoundtripPrefetcher`` keeps the GPU busy: the roundtrip is convolution-heavy
work, while the IDM (~15M params) on a CPU-bound video dataloader leaves the
GPU idle. Running the *next* batch's host->device copy and roundtrip on a side
CUDA stream overlaps that work with the current batch's IDM step, so the
roundtrip's cost is largely hidden rather than serialized in front of every
step.

Design record: ``dreamzero_prm_milestone3_idm_README.md`` (domain-gap section).
"""

import random
from typing import Optional

import torch


class VaeRoundtrip:
    """``decode(encode(.))`` of the production WAN2.2 VAE on the full canvas.

    The dream is decoded as one 160x320 canvas at inference, so the roundtrip
    re-joins the two 160x160 views, transforms the whole canvas, then re-splits
    -- matching the inference decode's boundary condition at the center seam.
    The VAE runs frozen, in ``dtype`` (bf16 mirrors the action head), with
    ``tiled=False`` (at 160x320 the 10x20 latent is a single tile anyway).
    """

    def __init__(
        self,
        vae_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        # Lazy import so the module (and RoundtripPrefetcher) is importable
        # without groot/VAE deps; only instantiating this class needs them.
        from rlinf.models.embodiment.dreamzero.patch.wan_video_vae import (
            WanVideoVAE38,
        )

        self.device = torch.device(device)
        self.dtype = dtype
        vae = WanVideoVAE38(z_dim=48)
        vae.model.load_state_dict(torch.load(vae_path, map_location="cpu"))
        self.vae = (
            vae.to(device=self.device, dtype=dtype).eval().requires_grad_(False)
        )

    @torch.no_grad()
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Roundtrip a batch of split-canvas clips.

        Args:
            video: ``[B, 2, T, 3, H, Wv]`` uint8, the dataset's two-view layout
                (view 0 = left/exterior half, view 1 = right/wrist half).

        Returns:
            Same shape and uint8 dtype, each frame replaced by its VAE
            ``decode(encode(.))`` reconstruction.
        """
        b, v, t, c, h, wv = video.shape
        if v != 2 or c != 3:
            raise ValueError(f"expected [B, 2, T, 3, H, W], got {tuple(video.shape)}")

        # Two halves -> full canvas [B, 3, T, H, 2*Wv] in [-1, 1] (VAE BCTHW).
        canvas = torch.cat([video[:, 0], video[:, 1]], dim=-1)  # [B, T, 3, H, 2Wv]
        canvas = canvas.permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, H, 2Wv]
        x = canvas.to(self.dtype).div_(127.5).sub_(1.0)

        z = self.vae.encode(x, tiled=False)
        y = self.vae.decode(z, tiled=False)  # [-1, 1]

        # Back to uint8 [0, 255] (matches the policy's decode normalization and
        # the dataset canvas dtype), then re-split into the two views.
        y = y.add_(1.0).mul_(127.5).clamp_(0, 255).to(torch.uint8)  # [B,3,T,H,2Wv]
        y = y.permute(0, 2, 1, 3, 4)  # [B, T, 3, H, 2Wv]
        return torch.stack([y[..., :wv], y[..., wv:]], dim=1)  # [B, 2, T, 3, H, Wv]


class RoundtripPrefetcher:
    """Double-buffered loader that hides the VAE roundtrip behind the IDM step.

    Iterating yields ``(video, actions)`` already on ``device`` (and
    roundtripped, if a ``roundtrip`` is given). The next batch's host->device
    copy and roundtrip are issued on a side CUDA stream during the current
    ``__next__``, so they overlap the caller's forward/backward on the default
    stream. On CPU-only devices it degrades to a plain synchronous transform.

    Args:
        loader: a DataLoader yielding ``{"video", "actions", ...}`` (see
            :func:`rlinf.models.embodiment.dreamzero.idm.data.collate_idm`);
            use ``pin_memory=True`` so the host->device copy can overlap.
        roundtrip: optional :class:`VaeRoundtrip`; ``None`` = copy only.
        device: target CUDA device.
        prob: fraction of batches to roundtrip (1.0 = every batch). Skipped
            batches are copied to device unchanged. A modeling knob (keep some
            clean frames for real-frame accuracy), not a memory one.
    """

    def __init__(
        self,
        loader,
        roundtrip: Optional[VaeRoundtrip],
        device: str = "cuda",
        prob: float = 1.0,
    ):
        self.loader = loader
        self.rt = roundtrip
        self.device = torch.device(device)
        self.prob = float(prob)
        self.use_stream = self.device.type == "cuda"
        self.stream = torch.cuda.Stream(self.device) if self.use_stream else None
        self._it = None
        self._next = None

    def __iter__(self) -> "RoundtripPrefetcher":
        self._it = iter(self.loader)
        self._preload()
        return self

    def _move_and_roundtrip(self, batch: dict):
        video = batch["video"].to(self.device, non_blocking=True)
        actions = batch["actions"].to(self.device, non_blocking=True)
        if self.rt is not None and random.random() < self.prob:
            video = self.rt(video)
        return video, actions

    def _preload(self) -> None:
        try:
            batch = next(self._it)
        except StopIteration:
            self._next = None
            return
        if self.use_stream:
            # Wait so the side stream doesn't race a buffer the default stream
            # may still be consuming, then issue copy + roundtrip on the side.
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self._next = self._move_and_roundtrip(batch)
        else:
            self._next = self._move_and_roundtrip(batch)

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self.use_stream:
            # Block the default stream until the side stream's roundtrip is done,
            # then mark the tensors as in-use so the allocator won't recycle them.
            torch.cuda.current_stream().wait_stream(self.stream)
            video, actions = self._next
            video.record_stream(torch.cuda.current_stream())
            actions.record_stream(torch.cuda.current_stream())
        else:
            video, actions = self._next
        self._preload()
        return video, actions
