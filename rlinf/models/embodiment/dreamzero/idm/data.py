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

"""IDM training dataset: mirror the DreamZero SFT pipeline, never re-derive it.

Every alignment- or appearance-relevant choice is inherited from the exact
code the WAM was trained with, so there is no hand-derived frame<->action
mapping anywhere:

- **Temporal sampling**: :class:`DreamZeroLeRobotDataset` in ``multi_anchor``
  mode with ``max_chunk_size=1`` -- per sample, video frames at chunk offsets
  ``(0, 3, ..., 21)`` plus the boundary frame at 24 (9 frames total, matching
  the 9 decoded dream frames) and the 16 contiguous chunk actions. The parent
  dataset is frame-indexed, so every demo frame is a candidate anchor: the
  ~16x anchor jitter falls out for free.
- **Visuals**: groot's own ``VideoCrop(0.95)`` / ``VideoResize`` transforms
  (the SFT chain, eval mode = center crop, no jitter -- matching how the
  WAM's conditioning obs is processed at inference), then the libero_sim
  exterior|wrist width-concat, then the final squash to the
  ``target_video_height x target_video_width`` model canvas. Each half nets
  out to 160x160, exactly the decoded dream layout.
- **Actions**: raw env-space ``[16, 7]`` chunks straight from the LeRobot
  columns (the SFT q99 normalization lives in ``StateActionTransform``,
  which is deliberately *not* applied -- the IDM standardizes with its own
  ``set_action_stats`` buffers).

``frame_postprocess`` is the domain-adaptation hook: it receives the final
``[T, H, W, 3]`` uint8 canvas (the thing the WAN VAE sees at inference) and
is where the offline VAE roundtrip plugs in.

The one step replicated rather than imported is the final canvas squash
(``F.interpolate`` bilinear); groot performs it inside its WAN processing.
Verifying pixel parity of that step against a decoded dream is part of the
M3a checklist (``dreamzero_prm_milestone3_idm_README.md``).
"""

from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.data.datasets.dreamzero.lerobot_dataset import DreamZeroLeRobotDataset
from rlinf.data.datasets.dreamzero.sampling_strategy import EmptyTemporalSampleError

_VIDEO_KEYS = ["video.image", "video.wrist_image"]
_STATE_KEYS = ["state.state"]
_ACTION_KEYS = ["action.actions"]
_LANGUAGE_KEYS = ["annotation.task"]


def _build_sft_video_pipeline(
    height: int, width: int, crop_scale: float, train_aug: bool
):
    """Groot's libero_sim video transforms, verbatim (lazy groot import).

    Eval mode (default) = deterministic center crop + resize, matching how
    the WAM's conditioning obs is processed at rollout. ``train_aug=True``
    switches to train mode (random crop) and adds the SFT color jitter as
    optional robustness augmentation.
    """
    from groot.vla.data.transform.base import ComposedModalityTransform
    from groot.vla.data.transform.video import (
        VideoColorJitter,
        VideoCrop,
        VideoResize,
        VideoToNumpy,
        VideoToTensor,
    )

    vk = list(_VIDEO_KEYS)
    transforms: list[Any] = [
        VideoToTensor(apply_to=vk, backend="torchvision"),
        VideoCrop(apply_to=vk, backend="torchvision", scale=crop_scale),
        VideoResize(
            apply_to=vk,
            backend="torchvision",
            height=height,
            width=width,
            interpolation="linear",
        ),
    ]
    if train_aug:
        # Same parameters as the SFT chain in data_transforms/libero_sim.py.
        transforms.append(
            VideoColorJitter(
                apply_to=vk,
                backend="torchvision",
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            )
        )
    transforms.append(VideoToNumpy(apply_to=vk, backend="torchvision"))

    pipeline = ComposedModalityTransform(transforms=transforms)
    if train_aug:
        pipeline.train()
    else:
        pipeline.eval()
    return pipeline


class IDMChunkDataset(DreamZeroLeRobotDataset):
    """(9-frame clip, 16-action chunk) samples for IDM training.

    Subclasses the SFT dataset for index resolution, multi-anchor temporal
    sampling, and raw modality loading; replaces the heavy SFT transform
    (tokenizer, q99 normalization, 32-dim padding) with the IDM-specific
    video path described in the module docstring.

    Sample dict:
        ``video``: uint8 ``[n_views, T, 3, view_h, view_w]`` (2, 9, 3, 160, 160)
        ``actions``: float32 ``[action_horizon, 7]`` raw env-space chunk
        ``episode_index`` / ``frame_index``: ints (anchor provenance)

    Args:
        data_path: LeRobot dataset path/repo id (e.g.
            ``physical-intelligence/libero``), same as SFT
            ``data.train_data_paths``.
        split: ``"train"`` or ``"val"``; episode-level split (no window
            leakage), deterministic in ``split_seed``.
        action_horizon / video_in_chunk_offsets / macro_stride: leave at
            defaults to mirror the LIBERO SFT config exactly.
        target_video_height / target_video_width: the WAM canvas size
            (LIBERO 5B: 160 x 320).
        train_aug: enable SFT-style random crop + color jitter (off by
            default: the dream canvas mirrors eval-processed conditioning).
        frame_postprocess: optional ``[T, H, W, 3] uint8 -> same`` applied to
            the final canvas before the view split (VAE roundtrip hook).
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        val_fraction: float = 0.05,
        split_seed: int = 0,
        action_horizon: int = 16,
        video_in_chunk_offsets: Optional[tuple] = None,
        macro_stride: Optional[int] = None,
        target_video_height: int = 160,
        target_video_width: int = 320,
        crop_scale: float = 0.95,
        train_aug: bool = False,
        frame_postprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        video_backend: str = "pyav",
        resample_attempts: int = 8,
    ):
        super().__init__(
            data_path=data_path,
            video_keys=list(_VIDEO_KEYS),
            state_keys=list(_STATE_KEYS),
            action_keys=list(_ACTION_KEYS),
            language_keys=list(_LANGUAGE_KEYS),
            data_transform=None,  # parent transform unused; __getitem__ overridden
            lazy_load=True,
            num_frames=9,
            state_horizon=1,
            action_horizon=action_horizon,
            max_chunk_size=1,
            video_backend=video_backend,
            sampling_mode="multi_anchor",
            multi_anchor_resample_attempts=1,  # retries handled split-aware below
            macro_stride=macro_stride,
            video_in_chunk_offsets=video_in_chunk_offsets,
        )
        self.canvas_height = int(target_video_height)
        self.canvas_width = int(target_video_width)
        self.frame_postprocess = frame_postprocess
        self.resample_attempts = max(1, int(resample_attempts))
        self._video_pipeline = _build_sft_video_pipeline(
            self.canvas_height, self.canvas_width, float(crop_scale), bool(train_aug)
        )
        self._allowed_indices = self._split_indices(
            split, float(val_fraction), int(split_seed)
        )

    # ------------------------------------------------------------------
    # Episode-level split
    # ------------------------------------------------------------------

    def _episode_frame_ranges(self) -> list[tuple[int, int]]:
        """Per-episode ``[start, end)`` global frame index ranges."""
        if self._use_lazy_video_tree:
            starts = self._episode_starts
            return [
                (int(starts[i]), int(starts[i + 1])) for i in range(len(self._episodes))
            ]
        if getattr(self, "_use_v2_image_parquet", False):
            cumulative = self._cumulative
            return [
                (0 if i == 0 else int(cumulative[i - 1]), int(cumulative[i]))
                for i in range(len(cumulative))
            ]
        raise RuntimeError("IDMChunkDataset requires lazy video or v2 parquet layout")

    def _split_indices(self, split: str, val_fraction: float, seed: int) -> np.ndarray:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        ranges = self._episode_frame_ranges()
        order = np.random.default_rng(seed).permutation(len(ranges))
        n_val = max(1, int(len(ranges) * val_fraction))
        chosen = order[-n_val:] if split == "val" else order[:-n_val]
        indices = [np.arange(*ranges[int(ep)], dtype=np.int64) for ep in chosen]
        return np.concatenate(indices) if indices else np.array([], dtype=np.int64)

    def __len__(self) -> int:
        return int(self._allowed_indices.size)

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_error: Optional[Exception] = None
        rng = None
        for attempt in range(self.resample_attempts):
            global_idx = int(self._allowed_indices[idx])
            try:
                raw = self._build_modality_dict(self._load_raw_sample(global_idx))
                return self._to_idm_sample(raw)
            except EmptyTemporalSampleError as exc:
                # Anchor too close to the episode end (needs +24 frames):
                # resample within this split only, never across the boundary.
                last_error = exc
                if rng is None:
                    rng = np.random.default_rng((idx, attempt))
                idx = int(rng.integers(0, len(self)))
        raise EmptyTemporalSampleError(
            f"No valid IDM anchor after {self.resample_attempts} attempts: {last_error}"
        )

    def _to_idm_sample(self, raw: dict[str, Any]) -> dict[str, Any]:
        views = self._video_pipeline(
            {key: raw[key] for key in _VIDEO_KEYS}
        )  # each: [T, h, w, 3] uint8 after crop+resize

        # Exterior | wrist width-concat, then squash to the model canvas --
        # the same double-resample the SFT/WAN path performs, so training
        # frames and decoded dreams share one geometry.
        exterior = np.asarray(views[_VIDEO_KEYS[0]])
        wrist = np.asarray(views[_VIDEO_KEYS[1]])
        concat = np.concatenate([exterior, wrist], axis=2)  # [T, h, 2w, 3]
        canvas = self._resize_canvas(concat)  # [T, H, W, 3] uint8

        if self.frame_postprocess is not None:
            canvas = self.frame_postprocess(canvas)

        half = self.canvas_width // 2
        video = np.stack([canvas[:, :, :half], canvas[:, :, half:]], axis=0)
        video = np.transpose(video, (0, 1, 4, 2, 3))  # [V, T, 3, H, W/2]

        actions = np.asarray(raw[_ACTION_KEYS[0]], dtype=np.float32)
        if actions.shape[0] != self.action_horizon:
            raise ValueError(
                f"expected {self.action_horizon} chunk actions, got {actions.shape}"
            )
        return {
            "video": torch.from_numpy(np.ascontiguousarray(video)),
            "actions": torch.from_numpy(actions),
            "episode_index": int(raw.get("episode_index", -1)),
            "frame_index": int(raw.get("frame_index", -1)),
        }

    def _resize_canvas(self, frames_thwc: np.ndarray) -> np.ndarray:
        """Bilinear-resize the concat canvas to (canvas_height, canvas_width)."""
        t, h, w, _ = frames_thwc.shape
        if (h, w) == (self.canvas_height, self.canvas_width):
            return frames_thwc
        x = torch.from_numpy(np.ascontiguousarray(frames_thwc)).float()
        x = x.permute(0, 3, 1, 2)  # [T, 3, h, w]
        x = F.interpolate(
            x,
            size=(self.canvas_height, self.canvas_width),
            mode="bilinear",
            align_corners=False,
        )
        return x.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

    # ------------------------------------------------------------------
    # Action statistics (feed IDM.set_action_stats / BCE pos_weight)
    # ------------------------------------------------------------------

    def compute_action_stats(self, action_dim: int = 7) -> dict[str, Any]:
        """Arm mean/std and gripper class balance over this split's episodes.

        Reads full action columns episode-by-episode (exact, not sampled).
        Returns ``arm_mean``/``arm_std`` (``action_dim - 1`` floats),
        ``gripper_pos_weight`` (BCEWithLogits ``pos_weight`` = neg/pos for
        the ``> 0`` binarization), and ``num_steps``.
        """
        ranges = self._episode_frame_ranges()
        allowed = set(self._allowed_indices.tolist())
        source = self._action_components[_ACTION_KEYS[0]][0]

        total = np.zeros(action_dim - 1, dtype=np.float64)
        total_sq = np.zeros(action_dim - 1, dtype=np.float64)
        n_steps = 0
        n_pos = 0
        for ep_pos, (start, end) in enumerate(ranges):
            if start not in allowed:
                continue
            rows = np.arange(end - start, dtype=np.int64)
            if self._use_lazy_video_tree:
                table = self._get_episode_table(int(self._episodes[ep_pos]))
            else:
                table = self._read_v2_episode(ep_pos)
            acts = self._read_list_column(table, source, rows)[:, :action_dim]
            arm = acts[:, :-1].astype(np.float64)
            total += arm.sum(axis=0)
            total_sq += (arm**2).sum(axis=0)
            n_steps += arm.shape[0]
            n_pos += int((acts[:, -1] > 0).sum())

        if n_steps == 0:
            raise RuntimeError("No episodes in split; cannot compute action stats")
        mean = total / n_steps
        var = np.maximum(total_sq / n_steps - mean**2, 1e-12)
        n_neg = n_steps - n_pos
        return {
            "arm_mean": mean.astype(np.float32).tolist(),
            "arm_std": np.sqrt(var).astype(np.float32).tolist(),
            "gripper_pos_weight": float(n_neg / max(1, n_pos)),
            "num_steps": int(n_steps),
        }


def collate_idm(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack IDM samples into a training batch."""
    return {
        "video": torch.stack([b["video"] for b in batch], dim=0),
        "actions": torch.stack([b["actions"] for b in batch], dim=0),
        "episode_index": torch.tensor(
            [b["episode_index"] for b in batch], dtype=torch.long
        ),
        "frame_index": torch.tensor(
            [b["frame_index"] for b in batch], dtype=torch.long
        ),
    }
