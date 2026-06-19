
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

"""Training data for the latent-space IDM: encode real clips, never decode.

The latent IDM (:class:`...idm.latent_model.LatentActionIDM`) consumes the
WAN-VAE *latent* the DiT produces (``video_pred``), so its training input must
be the **same latent space**: ``z = VAE.encode(real canvas)``.

This module reuses the pixel IDM's sampling/canvas pipeline unchanged
(:class:`...idm.data.IDMChunkDataset`) and adds only the encode step:

- :class:`IDMLatentDataset` -- the same ``(9-frame canvas, 16-action chunk)``
  sampler, with pixel augmentation and the VAE *roundtrip* disabled (the latent
  path never decodes, so there is no decoder artifact to mimic).
- :class:`LatentEncoder` -- the encode-only sibling of
  :class:`...idm.vae_roundtrip.VaeRoundtrip`: it re-joins the two views into the
  160x320 canvas and runs ``VAE.encode`` of the exact production
  :class:`WanVideoVAE38` (z_dim=48), giving ``[B, 48, 3, 10, 20]``. Using the
  same ``encode`` call as the production VAE keeps the latent-scaling
  convention identical to the ``video_pred`` the IDM scores at inference
  (train/test scale parity -- the whole point of staying in latent).
- :class:`LatentPrefetcher` -- the encode-only sibling of
  :class:`...idm.vae_roundtrip.RoundtripPrefetcher`: it hides the (deterministic)
  GPU encode behind the IDM step on a side CUDA stream and yields
  ``(latent, actions)`` already on device.

The VAE encode returns the posterior mean and never samples (verified by
``dreamzero_verify_vae_determinism.py``), so the on-the-fly encode is a *fixed*
transform -- identical to a precomputed cache, while composing with the free
per-frame anchor jitter of the parent dataset.

Design record: ``dreamzero_latent_idm_design_README.md``.
"""

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rlinf.models.embodiment.dreamzero.idm.data import (  # noqa: F401
    IDMChunkDataset,
    collate_idm,
)


class IDMLatentDataset(IDMChunkDataset):
    """``IDMChunkDataset`` configured for the latent IDM.

    Identical sampling and canvas construction as the parent -- yields the
    two-view RGB canvas ``video`` ``[2, T, 3, H, W/2]`` and the raw env-space
    ``actions`` ``[16, 7]`` -- but the WAN-VAE encode to ``[48, 3, 10, 20]``
    happens on the GPU in :class:`LatentPrefetcher`, not here (encode is
    convolution-heavy GPU work, unfit for CPU dataloader workers).

    The latent IDM trains on real encoded clips only: never on dreams, and
    never on a pixel roundtrip -- so ``train_aug`` and ``frame_postprocess``
    are forced off (the latent path has no decoder artifact to imitate).
    ``compute_action_stats`` is inherited unchanged (arm mean/std + gripper
    ``pos_weight``); the per-channel latent normalization was intentionally
    dropped, so no latent statistics are needed.
    """

    def __init__(self, *args, skip_invalid: bool = False, **kwargs):
        kwargs.pop("train_aug", None)
        kwargs.pop("frame_postprocess", None)
        super().__init__(*args, train_aug=False, frame_postprocess=None, **kwargs)
        # Precompute pass uses this: return None on an invalid anchor instead of
        # resampling, so the latent cache covers each valid anchor exactly once.
        self._skip_invalid = bool(skip_invalid)

    def __getitem__(self, idx):
        if not getattr(self, "_skip_invalid", False):
            return super().__getitem__(idx)
        from rlinf.data.datasets.dreamzero.sampling_strategy import (
            EmptyTemporalSampleError,
        )

        global_idx = int(self._allowed_indices[idx])
        try:
            raw = self._build_modality_dict(self._load_raw_sample(global_idx))
            return self._to_idm_sample(raw)
        except EmptyTemporalSampleError:
            return None


class LatentEncoder:
    """``VAE.encode`` of the production WAN2.2 VAE on the full canvas.

    The dream is a single 160x320 canvas at inference, so training clips are
    re-joined into one canvas and encoded whole -- matching the inference
    latent's boundary condition at the center seam (the dual-camera canvas is
    jointly encoded; it is never split in latent space). The VAE runs frozen,
    in ``dtype`` (bf16 mirrors the action head), ``tiled=False`` (the 10x20
    latent is a single tile at this resolution).
    """

    def __init__(
        self,
        vae_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        # Lazy import so this module is importable without groot/VAE deps;
        # only instantiating this class needs them.
        from rlinf.models.embodiment.dreamzero.patch.wan_video_vae import (
            WanVideoVAE38,
        )

        self.device = torch.device(device)
        self.dtype = dtype
        vae = WanVideoVAE38(z_dim=48)
        vae.model.load_state_dict(torch.load(vae_path, map_location="cpu"))
        self.vae = vae.to(device=self.device, dtype=dtype).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a batch of split-canvas clips to WAN-VAE latents.

        Args:
            video: ``[B, 2, T, 3, H, Wv]`` uint8, the dataset's two-view layout
                (view 0 = left/exterior half, view 1 = right/wrist half).

        Returns:
            ``[B, z_dim, T_lat, H_lat, W_lat]`` latent in ``dtype`` -- for the
            LIBERO 5B canvas, ``[B, 48, 3, 10, 20]``. (The IDM floats it.)
        """
        b, v, t, c, h, wv = video.shape
        if v != 2 or c != 3:
            raise ValueError(f"expected [B, 2, T, 3, H, W], got {tuple(video.shape)}")

        # Two halves -> full canvas [B, 3, T, H, 2*Wv] in [-1, 1] (VAE BCTHW),
        # exactly as the policy normalizes the decode/encode (see VaeRoundtrip).
        canvas = torch.cat([video[:, 0], video[:, 1]], dim=-1)  # [B, T, 3, H, 2Wv]
        canvas = canvas.permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, H, 2Wv]
        x = canvas.to(self.dtype).div_(127.5).sub_(1.0)

        return self.vae.encode(x, tiled=False)  # [B, z_dim, T_lat, H_lat, W_lat]


class LatentPrefetcher:
    """Double-buffered loader that hides the VAE encode behind the IDM step.

    Iterating yields ``(latent, actions)`` already on ``device``. The next
    batch's host->device copy and encode are issued on a side CUDA stream
    during the current ``__next__``, so they overlap the caller's
    forward/backward on the default stream. On CPU-only devices it degrades to
    a plain synchronous transform. Encode-only sibling of
    :class:`...idm.vae_roundtrip.RoundtripPrefetcher`; the encode is mandatory
    and deterministic, so there is no per-batch probability knob.

    Args:
        loader: a DataLoader yielding ``{"video", "actions", ...}`` (see
            :func:`...idm.data.collate_idm`); use ``pin_memory=True`` so the
            host->device copy can overlap.
        encoder: a :class:`LatentEncoder` (or any
            ``[B,2,T,3,H,Wv] -> [B,z,T',H',W']`` callable).
        device: target CUDA device.
    """

    def __init__(self, loader, encoder, device: str = "cuda"):
        self.loader = loader
        self.encoder = encoder
        self.device = torch.device(device)
        self.use_stream = self.device.type == "cuda"
        self.stream = torch.cuda.Stream(self.device) if self.use_stream else None
        self._it = None
        self._next = None

    def __iter__(self) -> "LatentPrefetcher":
        self._it = iter(self.loader)
        self._preload()
        return self

    def _move_and_encode(self, batch: dict):
        video = batch["video"].to(self.device, non_blocking=True)
        actions = batch["actions"].to(self.device, non_blocking=True)
        latent = self.encoder(video)
        return latent, actions

    def _preload(self) -> None:
        try:
            batch = next(self._it)
        except StopIteration:
            self._next = None
            return
        if self.use_stream:
            # Wait so the side stream doesn't race a buffer the default stream
            # may still be consuming, then issue copy + encode on the side.
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self._next = self._move_and_encode(batch)
        else:
            self._next = self._move_and_encode(batch)

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self.use_stream:
            # Block the default stream until the side stream's encode is done,
            # then mark the tensors as in-use so the allocator won't recycle them.
            torch.cuda.current_stream().wait_stream(self.stream)
            latent, actions = self._next
            latent.record_stream(torch.cuda.current_stream())
            actions.record_stream(torch.cuda.current_stream())
        else:
            latent, actions = self._next
        self._preload()
        return latent, actions


def _collate_skip_none(batch: list):
    """``collate_idm`` that drops ``None`` samples (invalid anchors)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return collate_idm(batch)


def precompute_latent_cache(
    dataset: IDMLatentDataset,
    encoder: LatentEncoder,
    cache_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = "cuda",
    action_dim: int = 7,
) -> dict:
    """Encode every valid anchor of ``dataset`` once and write an fp32 cache.

    Streams ``VAE.encode`` outputs to ``<cache_dir>/latents.bin`` (raw fp32,
    ``[count, *latent_shape]``), the matching env-space chunks to
    ``actions.npy``, and a ``meta.json`` carrying ``count``, shapes, dtype, and
    the action statistics (``arm_mean`` / ``arm_std`` / ``gripper_pos_weight``)
    so the trainer can ``set_action_stats`` without re-reading the dataset.

    ``dataset`` must be an :class:`IDMLatentDataset` with ``skip_invalid=True``
    (each invalid anchor yields ``None`` and is skipped, so every valid anchor
    is cached exactly once -- the free per-frame jitter is fully preserved).
    The encode is deterministic, so the cache is bit-identical to on-the-fly
    encoding; it just trades ~27 GB of disk (fp32, full LIBERO) for skipping the
    VAE on every epoch.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **k):
            return x

    os.makedirs(cache_dir, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_collate_skip_none,
        pin_memory=True,
    )

    latents_path = os.path.join(cache_dir, "latents.bin")
    actions_chunks: list[np.ndarray] = []
    count = 0
    latent_shape: Optional[tuple] = None
    with open(latents_path, "wb") as f:
        for batch in tqdm(loader, desc="encoding latents"):
            if batch is None:
                continue
            video = batch["video"].to(device, non_blocking=True)
            z = encoder(video).float().cpu().numpy().astype(np.float32, copy=False)
            if latent_shape is None:
                latent_shape = tuple(z.shape[1:])
            f.write(np.ascontiguousarray(z).tobytes())
            actions_chunks.append(batch["actions"].numpy().astype(np.float32))
            count += int(z.shape[0])

    if count == 0:
        raise RuntimeError("precompute_latent_cache wrote 0 samples (empty split?)")

    actions = np.concatenate(actions_chunks, axis=0)
    np.save(os.path.join(cache_dir, "actions.npy"), actions)

    stats = dataset.compute_action_stats(action_dim=action_dim)
    meta = {
        "count": int(count),
        "latent_shape": list(latent_shape),
        "action_shape": list(actions.shape[1:]),
        "dtype": "float32",
        **stats,
    }
    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


class CachedLatentDataset(Dataset):
    """Reads the fp32 latent cache written by :func:`precompute_latent_cache`.

    Serves ``{"latent": [C,T,H,W] fp32, "actions": [horizon, action_dim] fp32}``
    by memory-mapping ``latents.bin`` (no decode, no VAE at train time). The
    action statistics from ``meta.json`` are exposed as :attr:`action_stats`
    for the trainer's ``set_action_stats`` / BCE ``pos_weight``.
    """

    def __init__(self, cache_dir: str):
        with open(os.path.join(cache_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta
        self.count = int(meta["count"])
        self.latent_shape = tuple(meta["latent_shape"])
        self.latents = np.memmap(
            os.path.join(cache_dir, "latents.bin"),
            dtype=np.float32,
            mode="r",
            shape=(self.count, *self.latent_shape),
        )
        self.actions = np.load(os.path.join(cache_dir, "actions.npy"), mmap_mode="r")
        self.action_stats = {
            k: meta[k]
            for k in ("arm_mean", "arm_std", "gripper_pos_weight")
            if k in meta
        }

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # np.array (not asarray) forces a writable copy out of the read-only
        # memmap so torch.from_numpy does not warn / alias the mapping.
        return {
            "latent": torch.from_numpy(np.array(self.latents[idx], dtype=np.float32)),
            "actions": torch.from_numpy(np.array(self.actions[idx], dtype=np.float32)),
        }


def collate_latent(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack cached latent samples into a training batch."""
    return {
        "latent": torch.stack([b["latent"] for b in batch], dim=0),
        "actions": torch.stack([b["actions"] for b in batch], dim=0),
    }
